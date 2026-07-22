# MyBuddy Grok Build Fork

This branch is the MyBuddy-owned runtime baseline. MyBuddy does not depend on
upstream `main` for development or releases.

## Baseline

- Upstream repository: `xai-org/grok-build`
- Upstream commit: `3af4d5d39897855bdcc74f23e690024a5dc05573`
- Source revision: `0f4d7c91b8b2b408333f6de1e8a76cb8eaa71899`
- Product branch: `mybuddy/runtime-v1`

## Update policy

1. Never merge upstream `main` directly into this branch.
2. Evaluate useful upstream changes individually.
3. Cherry-pick or reimplement accepted changes on this branch.
4. Pin MyBuddy releases to an exact commit from this branch.
5. Run the MyBuddy runtime regression matrix before moving the pinned commit.

## Product constraints

- No xAI/Grok login or official model fallback.
- Custom model providers only.
- No external `grok` CLI process as the product boundary.
- Grok Build types remain behind the MyBuddy `EmbeddedGrokRuntime` facade.
- macOS arm64 and Windows x64 must pass the runtime regression matrix together.
- Skills, MCP, Hooks, system commands, permissions, cancellation and session
  recovery remain required capabilities.

## Embedded product facade

MyBuddy consumes the dedicated `xai-mybuddy-runtime` crate from this branch.
That crate is the stable in-process product seam over Grok Build's sampler and
must not expose shell, ACP, authentication, or internal sampler types to the
MyBuddy repository. Product releases pin this repository by commit, never by a
moving branch name.

The facade disables the `runtime-tool-definitions` default feature on the
sampler data layer. This keeps model sampling independent from the full Grok
Build tool closure (filesystem extractors, cloud SDKs, shell/runtime tooling
and workspace services). Full-workspace consumers retain the existing default;
MyBuddy attaches its own reviewed command, permission and Skills layers above
the sampler instead.
