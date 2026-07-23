//! Stable in-process sampling facade for MyBuddy.
//!
//! The public types in this crate are owned by MyBuddy, not by the Grok CLI.
//! It never starts a child process, runs no interactive login, and makes
//! requests only to the custom model endpoint configured by the desktop app.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use tokio::sync::mpsc;
use xai_grok_sampler::{
    AuthScheme, RequestId, RetryPolicy, SamplerActor, SamplerConfig, SamplerHandle,
    SamplingChannel, SamplingEvent,
};
use xai_grok_sampling_types::{
    ApiBackend, AssistantItem, ConversationItem, ConversationRequest, ConversationToolChoice,
    ReasoningEffort, ToolCall as SamplingToolCall, ToolSpec,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// API wire protocol used by a custom provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    ChatCompletions,
    Responses,
    Messages,
}

/// Reasoning effort exposed at the MyBuddy product boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

/// Custom model endpoint. Its debug implementation always redacts the key.
#[derive(Clone)]
pub struct ModelConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub protocol: Protocol,
    pub context_window: u64,
    pub reasoning_effort: Option<Effort>,
    pub max_retries: u32,
    pub idle_timeout_secs: u64,
}

impl Debug for ModelConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("model", &self.model)
            .field("protocol", &self.protocol)
            .field("context_window", &self.context_window)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("max_retries", &self.max_retries)
            .field("idle_timeout_secs", &self.idle_timeout_secs)
            .finish()
    }
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !(self.base_url.starts_with("https://") || self.base_url.starts_with("http://")) {
            return Err(ConfigError::new(
                "base_url must start with http:// or https://",
            ));
        }
        if self.model.trim().is_empty() {
            return Err(ConfigError::new("model must not be empty"));
        }
        if self.context_window == 0 {
            return Err(ConfigError::new("context_window must be greater than zero"));
        }
        if self.idle_timeout_secs == 0 {
            return Err(ConfigError::new(
                "idle_timeout_secs must be greater than zero",
            ));
        }
        if self
            .api_key
            .as_ref()
            .is_some_and(|key| key.bytes().any(|byte| byte < 0x20 || byte == 0x7f))
        {
            return Err(ConfigError::new(
                "api_key contains characters that are not valid in an HTTP header",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for ConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum InputItem {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: String,
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        call_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnRequest {
    pub items: Vec<InputItem>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub reasoning_tokens: u32,
    pub cached_prompt_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnOutput {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: Option<String>,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEvent {
    StreamStarted,
    FirstToken,
    TextDelta {
        text: String,
    },
    ReasoningDelta {
        text: String,
    },
    ToolCallDelta {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    Retrying {
        attempt: u32,
        max_retries: u32,
        reason: String,
    },
    ModelMetadata {
        context_window: Option<u64>,
        max_completion_tokens: Option<u64>,
    },
    BackendToolStarted {
        call_id: String,
        name: String,
    },
    BackendToolCompleted {
        call_id: String,
        name: String,
        result: Option<Value>,
    },
    Completed {
        output: TurnOutput,
    },
    Failed {
        kind: String,
        message: String,
        retryable: bool,
    },
}

/// In-process sampler entry point. Construction performs no network I/O.
#[derive(Clone)]
pub struct EmbeddedSampler {
    config: ModelConfig,
}

impl EmbeddedSampler {
    pub fn new(config: ModelConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Starts one streaming turn on the caller's Tokio runtime.
    pub fn start_turn(&self, request: TurnRequest) -> TurnHandle {
        let request_id = RequestId::random();
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let sampler = SamplerActor::spawn(
            sampler_config(&self.config),
            RetryPolicy {
                max_retries: self.config.max_retries,
                ..RetryPolicy::default()
            },
            event_tx,
        );
        sampler.submit(request_id.clone(), conversation_request(request));
        TurnHandle {
            request_id,
            sampler,
            events: event_rx,
        }
    }
}

impl Debug for EmbeddedSampler {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddedSampler")
            .field("config", &self.config)
            .finish()
    }
}

pub struct TurnHandle {
    request_id: RequestId,
    sampler: SamplerHandle,
    events: mpsc::UnboundedReceiver<SamplingEvent>,
}

impl TurnHandle {
    pub fn request_id(&self) -> &str {
        self.request_id.as_str()
    }

    pub fn cancel(&self) {
        self.sampler.cancel(self.request_id.clone());
    }

    pub async fn next_event(&mut self) -> Option<RuntimeEvent> {
        while let Some(event) = self.events.recv().await {
            if event_request_id(&event) == &self.request_id {
                return Some(map_event(event));
            }
        }
        None
    }
}

fn sampler_config(config: &ModelConfig) -> SamplerConfig {
    let extra_headers = if config.protocol == Protocol::Messages {
        [("anthropic-version".into(), ANTHROPIC_VERSION.into())]
            .into_iter()
            .collect()
    } else {
        Default::default()
    };
    SamplerConfig {
        api_key: config.api_key.clone(),
        base_url: config.base_url.trim_end_matches('/').to_string(),
        model: config.model.clone(),
        api_backend: match config.protocol {
            Protocol::ChatCompletions => ApiBackend::ChatCompletions,
            Protocol::Responses => ApiBackend::Responses,
            Protocol::Messages => ApiBackend::Messages,
        },
        auth_scheme: match config.protocol {
            Protocol::Messages => AuthScheme::XApiKey,
            Protocol::ChatCompletions | Protocol::Responses => AuthScheme::Bearer,
        },
        extra_headers,
        context_window: config.context_window,
        reasoning_effort: config.reasoning_effort.map(map_effort),
        max_retries: Some(config.max_retries),
        idle_timeout_secs: Some(config.idle_timeout_secs),
        stream_tool_calls: true,
        origin_client: Some(xai_grok_sampler::OriginClientInfo {
            product: "mybuddy-desktop".into(),
            version: Some(env!("CARGO_PKG_VERSION").into()),
        }),
        client_identifier: Some("mybuddy-desktop".into()),
        ..SamplerConfig::default()
    }
}

fn map_effort(effort: Effort) -> ReasoningEffort {
    match effort {
        Effort::None => ReasoningEffort::None,
        Effort::Minimal => ReasoningEffort::Minimal,
        Effort::Low => ReasoningEffort::Low,
        Effort::Medium => ReasoningEffort::Medium,
        Effort::High => ReasoningEffort::High,
        Effort::Xhigh => ReasoningEffort::Xhigh,
        Effort::Max => ReasoningEffort::Max,
    }
}

fn conversation_request(request: TurnRequest) -> ConversationRequest {
    ConversationRequest {
        items: request
            .items
            .into_iter()
            .map(|item| match item {
                InputItem::System { content } => ConversationItem::system(content),
                InputItem::User { content } => ConversationItem::user(content),
                InputItem::Assistant {
                    content,
                    model,
                    tool_calls,
                } => ConversationItem::Assistant(AssistantItem {
                    content: Arc::<str>::from(content),
                    model_id: model,
                    tool_calls: tool_calls
                        .into_iter()
                        .map(|call| SamplingToolCall {
                            id: Arc::<str>::from(call.id),
                            name: call.name,
                            arguments: Arc::<str>::from(call.arguments),
                        })
                        .collect(),
                    model_fingerprint: None,
                    reasoning_effort: None,
                }),
                InputItem::ToolResult { call_id, content } => {
                    ConversationItem::tool_result(call_id, content)
                }
            })
            .collect(),
        tools: request
            .tools
            .into_iter()
            .map(|tool| ToolSpec {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
            })
            .collect(),
        tool_choice: Some(match request.tool_choice {
            ToolChoice::Auto => ConversationToolChoice::Auto,
            ToolChoice::None => ConversationToolChoice::None,
            ToolChoice::Required => ConversationToolChoice::Required,
        }),
        temperature: request.temperature,
        max_output_tokens: request.max_output_tokens,
        ..ConversationRequest::default()
    }
}

fn event_request_id(event: &SamplingEvent) -> &RequestId {
    match event {
        SamplingEvent::StreamStarted { request_id, .. }
        | SamplingEvent::FirstToken { request_id }
        | SamplingEvent::ChannelToken { request_id, .. }
        | SamplingEvent::ToolCallDelta { request_id, .. }
        | SamplingEvent::Completed { request_id, .. }
        | SamplingEvent::Retrying { request_id, .. }
        | SamplingEvent::Failed { request_id, .. }
        | SamplingEvent::ModelMetadata { request_id, .. }
        | SamplingEvent::BackendToolCallStarted { request_id, .. }
        | SamplingEvent::BackendToolCallCompleted { request_id, .. } => request_id,
    }
}

fn map_event(event: SamplingEvent) -> RuntimeEvent {
    match event {
        SamplingEvent::StreamStarted { .. } => RuntimeEvent::StreamStarted,
        SamplingEvent::FirstToken { .. } => RuntimeEvent::FirstToken,
        SamplingEvent::ChannelToken { channel, text, .. } => match channel {
            SamplingChannel::Text => RuntimeEvent::TextDelta { text },
            SamplingChannel::Reasoning => RuntimeEvent::ReasoningDelta { text },
        },
        SamplingEvent::ToolCallDelta {
            tool_index,
            id,
            name,
            arguments_delta,
            ..
        } => RuntimeEvent::ToolCallDelta {
            index: tool_index,
            id,
            name,
            arguments_delta,
        },
        SamplingEvent::Retrying {
            attempt,
            max_retries,
            reason,
            ..
        } => RuntimeEvent::Retrying {
            attempt,
            max_retries,
            reason,
        },
        SamplingEvent::ModelMetadata { metadata, .. } => RuntimeEvent::ModelMetadata {
            context_window: metadata.context_window,
            max_completion_tokens: metadata.max_completion_tokens.map(u64::from),
        },
        SamplingEvent::BackendToolCallStarted { call_id, name, .. } => {
            RuntimeEvent::BackendToolStarted { call_id, name }
        }
        SamplingEvent::BackendToolCallCompleted {
            call_id,
            name,
            result,
            ..
        } => RuntimeEvent::BackendToolCompleted {
            call_id,
            name,
            result,
        },
        SamplingEvent::Completed { response, .. } => {
            let usage = response.usage.as_ref().map(|usage| TokenUsage {
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                reasoning_tokens: usage.reasoning_tokens,
                cached_prompt_tokens: usage.cached_prompt_tokens,
            });
            let tool_calls = response
                .tool_calls()
                .iter()
                .map(|call| ToolCall {
                    id: call.id.to_string(),
                    name: call.name.clone(),
                    arguments: call.arguments.to_string(),
                })
                .collect();
            RuntimeEvent::Completed {
                output: TurnOutput {
                    text: response.assistant_text(),
                    tool_calls,
                    stop_reason: response
                        .stop_reason
                        .map(|reason| reason.as_str().to_string()),
                    usage,
                },
            }
        }
        SamplingEvent::Failed { error, .. } => RuntimeEvent::Failed {
            kind: error.kind.as_str().to_string(),
            message: error.message,
            retryable: error.is_retryable,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_config(protocol: Protocol) -> ModelConfig {
        ModelConfig {
            base_url: "https://provider.example/v1".into(),
            api_key: Some("super-secret".into()),
            model: "model-a".into(),
            protocol,
            context_window: 128_000,
            reasoning_effort: Some(Effort::High),
            max_retries: 2,
            idle_timeout_secs: 300,
        }
    }

    #[test]
    fn model_debug_redacts_api_key() {
        let rendered = format!("{:?}", model_config(Protocol::Responses));
        assert!(rendered.contains("[REDACTED]"));
        assert!(!rendered.contains("super-secret"));
    }

    #[test]
    fn messages_uses_x_api_key_auth() {
        let config = sampler_config(&model_config(Protocol::Messages));
        assert_eq!(config.api_backend, ApiBackend::Messages);
        assert_eq!(config.auth_scheme, AuthScheme::XApiKey);
        assert_eq!(
            config
                .extra_headers
                .get("anthropic-version")
                .map(String::as_str),
            Some(ANTHROPIC_VERSION)
        );
    }

    #[test]
    fn openai_protocols_do_not_receive_anthropic_headers() {
        for protocol in [Protocol::ChatCompletions, Protocol::Responses] {
            assert!(
                !sampler_config(&model_config(protocol))
                    .extra_headers
                    .contains_key("anthropic-version")
            );
        }
    }

    #[test]
    fn rejects_header_control_characters_in_key() {
        let mut config = model_config(Protocol::ChatCompletions);
        config.api_key = Some("secret\nheader".into());
        assert!(config.validate().is_err());
    }

    #[test]
    fn converts_product_request_without_shell_types() {
        let request = conversation_request(TurnRequest {
            items: vec![
                InputItem::System {
                    content: "You are MyBuddy".into(),
                },
                InputItem::User {
                    content: "Hello".into(),
                },
            ],
            tools: vec![ToolDefinition {
                name: "read_file".into(),
                description: Some("Read a file".into()),
                parameters: serde_json::json!({"type": "object"}),
            }],
            tool_choice: ToolChoice::Auto,
            temperature: None,
            max_output_tokens: Some(1024),
        });
        assert_eq!(request.items.len(), 2);
        assert_eq!(request.tools[0].name, "read_file");
        assert!(matches!(
            request.tool_choice,
            Some(ConversationToolChoice::Auto)
        ));
    }
}
