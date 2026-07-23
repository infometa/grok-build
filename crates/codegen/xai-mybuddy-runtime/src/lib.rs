//! Stable in-process sampling facade for MyBuddy.
//!
//! The public types in this crate are owned by MyBuddy, not by the Grok CLI.
//! It never starts a child process, runs no interactive login, and makes
//! requests only to the custom model endpoint configured by the desktop app.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use tokio::sync::mpsc;
use xai_grok_compaction::{
    build_summary_prompt, format_compact_summary_content, is_degenerate_summary,
};
use xai_grok_sampler::{
    AuthScheme, RequestId, RetryPolicy, SamplerActor, SamplerConfig, SamplerHandle,
    SamplingChannel, SamplingEvent,
};
use xai_grok_sampling_types::{
    ApiBackend, AssistantItem, ConversationItem, ConversationRequest, ConversationToolChoice,
    ReasoningEffort, ToolCall as SamplingToolCall, ToolSpec,
};
use xai_grok_subagent_resolution::{
    SubagentPersona as UpstreamPersona, SubagentRole as UpstreamRole, intersect_capability_modes,
    resolve_effective_overrides,
};
use xai_grok_tools::implementations::grok_build::task::types::{
    ModelOverrideProvenance, SubagentRuntimeOverrides,
};
use xai_tool_types::{SubagentCapabilityMode, SubagentIsolationMode};

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

/// A validated compaction failure returned by the Grok Build-backed facade.
///
/// The desktop host owns when to compact and persistence. This error keeps the
/// Grok Build summary quality gate at the product boundary without exposing
/// upstream internal error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionError {
    message: String,
}

impl CompactionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for CompactionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompactionError {}

/// Build the canonical Grok Build full-history compaction request.
///
/// The existing history is sent verbatim and Grok Build's structured summary
/// prompt is appended as the final user message, matching the upstream
/// full-replace compaction flow. Tools are explicitly disabled so a provider
/// can never execute actions while summarising prior context.
pub fn compaction_request(history: Vec<InputItem>) -> Result<TurnRequest, CompactionError> {
    if history.is_empty() {
        return Err(CompactionError::new("cannot compact an empty history"));
    }

    let mut items = history;
    items.push(InputItem::User {
        content: build_summary_prompt(None),
    });
    Ok(TurnRequest {
        items,
        tools: Vec::new(),
        tool_choice: ToolChoice::None,
        temperature: None,
        max_output_tokens: None,
    })
}

/// Rebuild a compacted MyBuddy history using Grok Build's continuation carrier.
///
/// All prior assistant/tool output is replaced. Original system instructions
/// and the most recent user request remain verbatim; the cleaned summary then
/// carries the earlier context forward. A short or malformed model response is
/// rejected instead of silently destroying the session history.
pub fn compacted_history(
    history: &[InputItem],
    raw_summary: &str,
) -> Result<Vec<InputItem>, CompactionError> {
    if is_degenerate_summary(raw_summary) {
        return Err(CompactionError::new(
            "compaction response was too short to preserve session context",
        ));
    }

    let mut compacted = history
        .iter()
        .filter_map(|item| match item {
            InputItem::System { content } => Some(InputItem::System {
                content: content.clone(),
            }),
            InputItem::User { .. } | InputItem::Assistant { .. } | InputItem::ToolResult { .. } => {
                None
            }
        })
        .collect::<Vec<_>>();

    if let Some(InputItem::User { content }) = history
        .iter()
        .rev()
        .find(|item| matches!(item, InputItem::User { .. }))
    {
        compacted.push(InputItem::User {
            content: content.clone(),
        });
    }
    compacted.push(InputItem::User {
        content: format_compact_summary_content(raw_summary),
    });
    Ok(compacted)
}

/// Tool-access mode requested for a MyBuddy subagent.
///
/// The enum is deliberately product-owned. It is converted to Grok Build's
/// capability mode only inside this facade, so the desktop app never imports
/// upstream tool configuration types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentCapability {
    ReadOnly,
    ReadWrite,
    Execute,
    All,
}

/// Workspace isolation mode for a MyBuddy subagent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubagentIsolation {
    /// Reuse the parent workspace. This is the default and does not create a
    /// Git worktree.
    Shared,
    /// Ask the eventual host runner to allocate an isolated Git worktree.
    /// Resolution is platform-agnostic; availability is enforced by the host.
    Worktree,
}

/// Named role preset used to resolve a child-agent execution plan.
///
/// `instructions` intentionally stays inline. MyBuddy does not allow an
/// untrusted model response to choose arbitrary prompt files on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentRolePreset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_ceiling: Option<SubagentCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolation: Option<SubagentIsolation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// Named persona preset applied after a role and before the parent defaults.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentPersonaPreset {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolation: Option<SubagentIsolation>,
    pub instructions: String,
}

/// Product-owned catalogue of roles and personas available to an agent run.
///
/// A catalogue-wide ceiling can only reduce tool access; it can never grant
/// an ability a role or spawn request did not otherwise resolve to.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentCatalog {
    #[serde(default)]
    pub roles: Vec<SubagentRolePreset>,
    #[serde(default)]
    pub personas: Vec<SubagentPersonaPreset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_ceiling: Option<SubagentCapability>,
}

/// Defaults inherited from the parent agent and its selected model provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentParentDefaults {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
}

/// A model-facing spawn specification. It does not execute a child; the host
/// owns scheduling, cancellation, persistence and permission prompts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSpec {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_mode: Option<SubagentCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub isolation: Option<SubagentIsolation>,
}

/// Resolved, safe-to-schedule child-agent configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedSubagent {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capability_mode: Option<SubagentCapability>,
    pub isolation: SubagentIsolation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persona: Option<String>,
    /// Prompt layers in deterministic role-then-persona order.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub instructions: String,
}

/// Validation failure for a product-owned subagent plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubagentResolutionError {
    message: String,
}

impl SubagentResolutionError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for SubagentResolutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SubagentResolutionError {}

/// Resolve a MyBuddy subagent plan with Grok Build's role/persona precedence.
///
/// Precedence for model and reasoning effort is: explicit spawn field > role
/// > persona > parent. Capability requests are intersected with both the role
/// and catalogue ceilings, preventing an individual spawn from escalating
/// beyond policy. This performs no file I/O and no network activity.
pub fn resolve_subagent(
    spec: &SubagentSpec,
    catalog: &SubagentCatalog,
    parent: &SubagentParentDefaults,
) -> Result<ResolvedSubagent, SubagentResolutionError> {
    let parent_model = normalized_required(&parent.model, "parent model")?;
    let roles = index_roles(catalog)?;
    let personas = index_personas(catalog)?;

    let role_name = normalized_optional(spec.role.as_deref(), "role")?;
    let persona_name = normalized_optional(spec.persona.as_deref(), "persona")?;
    let selected_role = match role_name.as_deref() {
        Some(name) => Some(roles.get(name).ok_or_else(|| {
            SubagentResolutionError::new(format!("unknown subagent role: {name}"))
        })?),
        None => None,
    };
    if let Some(name) = persona_name.as_deref()
        && !personas.contains_key(name)
    {
        return Err(SubagentResolutionError::new(format!(
            "unknown subagent persona: {name}"
        )));
    }

    let overrides = SubagentRuntimeOverrides {
        model: normalized_optional(spec.model.as_deref(), "subagent model")?,
        model_override_provenance: ModelOverrideProvenance::Harness,
        reasoning_effort: spec.reasoning_effort.map(effort_name),
        persona: persona_name.clone(),
        capability_mode: spec.capability_mode.map(to_upstream_capability),
        isolation: spec.isolation.map(to_upstream_isolation),
        harness_agent_type: None,
        completion_output_cap: None,
        spawn_depth: None,
        output_token_budget: None,
        output_schema: None,
        loop_task_id: None,
    };

    let upstream_personas = personas
        .iter()
        .map(|(name, preset)| {
            (
                name.clone(),
                UpstreamPersona {
                    instructions: Some(preset.instructions.clone()),
                    model: preset.model.clone(),
                    reasoning_effort: preset.reasoning_effort.map(effort_name),
                    default_isolation: preset.isolation.map(isolation_name).map(str::to_owned),
                    ..Default::default()
                },
            )
        })
        .collect::<HashMap<_, _>>();
    let upstream_role = selected_role.map(|role| UpstreamRole {
        description: role.description.clone(),
        default_capability_mode: role
            .capability_ceiling
            .map(capability_name)
            .map(str::to_owned),
        model: role.model.clone(),
        reasoning_effort: role.reasoning_effort.map(effort_name),
        default_isolation: role.isolation.map(isolation_name).map(str::to_owned),
        ..Default::default()
    });
    let effective = resolve_effective_overrides(
        &overrides,
        upstream_role.as_ref(),
        &upstream_personas,
        None,
        role_name.clone(),
    );
    if let Some(error) = effective.persona_error {
        return Err(SubagentResolutionError::new(error));
    }

    let capability_mode = intersect_capability_modes(
        effective.capability_mode,
        catalog.capability_ceiling.map(to_upstream_capability),
    )
    .map(from_upstream_capability);
    let instructions = [
        selected_role.and_then(|role| role.instructions.as_deref()),
        effective.persona_instructions.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(str::trim)
    .filter(|text| !text.is_empty())
    .collect::<Vec<_>>()
    .join("\n\n");

    Ok(ResolvedSubagent {
        model: effective.model.unwrap_or(parent_model),
        reasoning_effort: effective
            .reasoning_effort
            .as_deref()
            .and_then(parse_effort)
            .or(parent.reasoning_effort),
        capability_mode,
        isolation: from_upstream_isolation(effective.isolation),
        role: effective.role_name,
        persona: effective.persona,
        instructions,
    })
}

fn index_roles(
    catalog: &SubagentCatalog,
) -> Result<HashMap<String, &SubagentRolePreset>, SubagentResolutionError> {
    let mut indexed = HashMap::with_capacity(catalog.roles.len());
    for role in &catalog.roles {
        let name = normalized_required(&role.name, "subagent role name")?;
        if indexed.insert(name.clone(), role).is_some() {
            return Err(SubagentResolutionError::new(format!(
                "duplicate subagent role: {name}"
            )));
        }
    }
    Ok(indexed)
}

fn index_personas(
    catalog: &SubagentCatalog,
) -> Result<HashMap<String, &SubagentPersonaPreset>, SubagentResolutionError> {
    let mut indexed = HashMap::with_capacity(catalog.personas.len());
    for persona in &catalog.personas {
        let name = normalized_required(&persona.name, "subagent persona name")?;
        if normalized_required(&persona.instructions, "subagent persona instructions").is_err() {
            return Err(SubagentResolutionError::new(format!(
                "subagent persona {name} has empty instructions"
            )));
        }
        if indexed.insert(name.clone(), persona).is_some() {
            return Err(SubagentResolutionError::new(format!(
                "duplicate subagent persona: {name}"
            )));
        }
    }
    Ok(indexed)
}

fn normalized_required(value: &str, label: &str) -> Result<String, SubagentResolutionError> {
    normalized_optional(Some(value), label)?
        .ok_or_else(|| SubagentResolutionError::new(format!("{label} must not be empty")))
}

fn normalized_optional(
    value: Option<&str>,
    label: &str,
) -> Result<Option<String>, SubagentResolutionError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let normalized = value.trim();
    if normalized.is_empty() {
        return Err(SubagentResolutionError::new(format!(
            "{label} must not be empty"
        )));
    }
    Ok(Some(normalized.to_string()))
}

fn effort_name(effort: Effort) -> String {
    match effort {
        Effort::None => "none".into(),
        Effort::Minimal => "minimal".into(),
        Effort::Low => "low".into(),
        Effort::Medium => "medium".into(),
        Effort::High => "high".into(),
        Effort::Xhigh => "xhigh".into(),
        Effort::Max => "max".into(),
    }
}

fn parse_effort(value: &str) -> Option<Effort> {
    match value {
        "none" => Some(Effort::None),
        "minimal" => Some(Effort::Minimal),
        "low" => Some(Effort::Low),
        "medium" => Some(Effort::Medium),
        "high" => Some(Effort::High),
        "xhigh" => Some(Effort::Xhigh),
        "max" => Some(Effort::Max),
        _ => None,
    }
}

const fn to_upstream_capability(capability: SubagentCapability) -> SubagentCapabilityMode {
    match capability {
        SubagentCapability::ReadOnly => SubagentCapabilityMode::ReadOnly,
        SubagentCapability::ReadWrite => SubagentCapabilityMode::ReadWrite,
        SubagentCapability::Execute => SubagentCapabilityMode::Execute,
        SubagentCapability::All => SubagentCapabilityMode::All,
    }
}

const fn from_upstream_capability(capability: SubagentCapabilityMode) -> SubagentCapability {
    match capability {
        SubagentCapabilityMode::ReadOnly => SubagentCapability::ReadOnly,
        SubagentCapabilityMode::ReadWrite => SubagentCapability::ReadWrite,
        SubagentCapabilityMode::Execute => SubagentCapability::Execute,
        SubagentCapabilityMode::All => SubagentCapability::All,
    }
}

const fn to_upstream_isolation(isolation: SubagentIsolation) -> SubagentIsolationMode {
    match isolation {
        SubagentIsolation::Shared => SubagentIsolationMode::None,
        SubagentIsolation::Worktree => SubagentIsolationMode::Worktree,
    }
}

const fn from_upstream_isolation(isolation: SubagentIsolationMode) -> SubagentIsolation {
    match isolation {
        SubagentIsolationMode::None => SubagentIsolation::Shared,
        SubagentIsolationMode::Worktree => SubagentIsolation::Worktree,
    }
}

const fn capability_name(capability: SubagentCapability) -> &'static str {
    match capability {
        SubagentCapability::ReadOnly => "read-only",
        SubagentCapability::ReadWrite => "read-write",
        SubagentCapability::Execute => "execute",
        SubagentCapability::All => "all",
    }
}

const fn isolation_name(isolation: SubagentIsolation) -> &'static str {
    match isolation {
        SubagentIsolation::Shared => "none",
        SubagentIsolation::Worktree => "worktree",
    }
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

    #[test]
    fn compaction_request_uses_the_grok_build_summary_prompt_without_tools() {
        let request = compaction_request(vec![
            InputItem::System {
                content: "You are MyBuddy".into(),
            },
            InputItem::User {
                content: "Implement the runtime".into(),
            },
        ])
        .expect("history can be compacted");

        assert!(request.tools.is_empty());
        assert_eq!(request.tool_choice, ToolChoice::None);
        assert!(matches!(
            request.items.last(),
            Some(InputItem::User { content }) if content.contains("1. Primary Request and Intent")
        ));
    }

    #[test]
    fn compacted_history_keeps_system_and_last_request_with_clean_summary() {
        let history = vec![
            InputItem::System {
                content: "You are MyBuddy".into(),
            },
            InputItem::User {
                content: "First request".into(),
            },
            InputItem::Assistant {
                content: "First response".into(),
                model: None,
                tool_calls: Vec::new(),
            },
            InputItem::User {
                content: "Continue with the second request".into(),
            },
            InputItem::ToolResult {
                call_id: "call-1".into(),
                content: "unbounded tool output".into(),
            },
        ];
        let raw_summary = format!(
            "<analysis>private scratchpad</analysis><summary>\n1. Primary Request and Intent: complete MyBuddy.\n{}\n</summary>",
            "Details that must survive compaction. ".repeat(20)
        );

        let compacted = compacted_history(&history, &raw_summary).expect("summary is sufficient");
        assert_eq!(compacted.len(), 3);
        assert!(matches!(
            &compacted[0],
            InputItem::System { content } if content == "You are MyBuddy"
        ));
        assert!(matches!(
            &compacted[1],
            InputItem::User { content } if content == "Continue with the second request"
        ));
        assert!(matches!(
            &compacted[2],
            InputItem::User { content }
                if content.contains("This session is being continued")
                    && content.contains("Summary:\n1. Primary Request")
                    && !content.contains("private scratchpad")
        ));
    }

    #[test]
    fn compacted_history_rejects_a_degenerate_summary() {
        let history = vec![InputItem::User {
            content: "Keep this task".into(),
        }];
        assert!(compacted_history(&history, "<summary>too short</summary>").is_err());
    }

    fn parent_defaults() -> SubagentParentDefaults {
        SubagentParentDefaults {
            model: "user-selected-model".into(),
            reasoning_effort: Some(Effort::Medium),
        }
    }

    #[test]
    fn subagent_resolution_uses_grok_precedence_and_policy_ceiling() {
        let catalog = SubagentCatalog {
            roles: vec![SubagentRolePreset {
                name: "research".into(),
                description: "Research a bounded question".into(),
                model: Some("role-model".into()),
                reasoning_effort: Some(Effort::High),
                capability_ceiling: Some(SubagentCapability::All),
                isolation: Some(SubagentIsolation::Worktree),
                instructions: Some("Collect evidence before concluding.".into()),
            }],
            personas: vec![SubagentPersonaPreset {
                name: "concise".into(),
                model: Some("persona-model".into()),
                reasoning_effort: Some(Effort::Low),
                isolation: Some(SubagentIsolation::Shared),
                instructions: "State the answer first.".into(),
            }],
            capability_ceiling: Some(SubagentCapability::ReadOnly),
        };
        let plan = resolve_subagent(
            &SubagentSpec {
                role: Some("research".into()),
                persona: Some("concise".into()),
                model: Some("explicit-model".into()),
                reasoning_effort: Some(Effort::Max),
                capability_mode: Some(SubagentCapability::All),
                isolation: None,
            },
            &catalog,
            &parent_defaults(),
        )
        .expect("valid plan");

        assert_eq!(plan.model, "explicit-model");
        assert_eq!(plan.reasoning_effort, Some(Effort::Max));
        assert_eq!(plan.capability_mode, Some(SubagentCapability::ReadOnly));
        assert_eq!(plan.isolation, SubagentIsolation::Worktree);
        assert_eq!(plan.role.as_deref(), Some("research"));
        assert_eq!(plan.persona.as_deref(), Some("concise"));
        assert_eq!(
            plan.instructions,
            "Collect evidence before concluding.\n\nState the answer first."
        );
    }

    #[test]
    fn subagent_resolution_inherits_the_selected_provider_model() {
        let plan = resolve_subagent(
            &SubagentSpec::default(),
            &SubagentCatalog::default(),
            &parent_defaults(),
        )
        .expect("empty catalogue is valid");

        assert_eq!(plan.model, "user-selected-model");
        assert_eq!(plan.reasoning_effort, Some(Effort::Medium));
        assert_eq!(plan.capability_mode, None);
        assert_eq!(plan.isolation, SubagentIsolation::Shared);
        assert!(plan.instructions.is_empty());
    }

    #[test]
    fn subagent_resolution_fails_closed_for_unknown_or_duplicate_presets() {
        let unknown = resolve_subagent(
            &SubagentSpec {
                role: Some("missing".into()),
                ..SubagentSpec::default()
            },
            &SubagentCatalog::default(),
            &parent_defaults(),
        )
        .expect_err("unknown roles must not silently fall back");
        assert!(unknown.to_string().contains("unknown subagent role"));

        let duplicate = resolve_subagent(
            &SubagentSpec::default(),
            &SubagentCatalog {
                roles: vec![
                    SubagentRolePreset {
                        name: "review".into(),
                        description: String::new(),
                        model: None,
                        reasoning_effort: None,
                        capability_ceiling: None,
                        isolation: None,
                        instructions: None,
                    },
                    SubagentRolePreset {
                        name: "review".into(),
                        description: String::new(),
                        model: None,
                        reasoning_effort: None,
                        capability_ceiling: None,
                        isolation: None,
                        instructions: None,
                    },
                ],
                ..SubagentCatalog::default()
            },
            &parent_defaults(),
        )
        .expect_err("duplicate names make policy ambiguous");
        assert!(duplicate.to_string().contains("duplicate subagent role"));
    }
}
