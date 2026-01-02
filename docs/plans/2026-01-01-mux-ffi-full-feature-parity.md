# mux-ffi Full Feature Parity Design

**Date:** 2026-01-01
**Status:** Approved
**Approach:** Monolithic MuxEngine (Approach A)

## Overview

Expose 100% of mux-rs facilities via UniFFI bindings for iOS/macOS consumers. Currently missing: subagents, hooks, transcripts, multi-backend LLM, custom tools, and permission policies.

## New FFI Types

### Provider Enum (Expanded)

```rust
#[derive(uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,      // OpenAI-compatible (including OpenRouter)
    Gemini,
    Ollama,
}
```

### Agent Configuration

```rust
#[derive(uniffi::Record)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    pub model: Option<String>,        // None = inherit from workspace
    pub allowed_tools: Vec<String>,   // Empty = all tools
    pub denied_tools: Vec<String>,
    pub max_iterations: u32,
}
```

### Hook Types

```rust
#[derive(uniffi::Enum)]
pub enum HookEventType {
    PreToolUse { tool_name: String, input: String },
    PostToolUse { tool_name: String, input: String, result: String },
    AgentStart { agent_id: String, task: String },
    AgentStop { agent_id: String },
    Iteration { agent_id: String, iteration: u32 },
}

#[derive(uniffi::Enum)]
pub enum HookResponse {
    Continue,
    Block { reason: String },
    Transform { new_input: String },  // Only valid for PreToolUse
}
```

### Transcript & Results

```rust
#[derive(uniffi::Record)]
pub struct TranscriptData {
    pub agent_id: String,
    pub messages_json: String,  // JSON serialized
}

#[derive(uniffi::Record)]
pub struct SubagentResult {
    pub agent_id: String,
    pub content: String,
    pub tool_use_count: u32,
    pub iterations: u32,
    pub transcript_json: Option<String>,
}

#[derive(uniffi::Record)]
pub struct ToolExecutionResult {
    pub content: String,
    pub is_error: bool,
}
```

## Callback Interfaces

Swift/Kotlin implements these to receive events:

```rust
#[uniffi::export(callback_interface)]
pub trait HookHandler: Send + Sync {
    fn on_event(&self, event: HookEventType) -> HookResponse;
}

#[uniffi::export(callback_interface)]
pub trait SubagentCallback: Send + Sync {
    fn on_text_delta(&self, agent_id: String, text: String);
    fn on_tool_use(&self, agent_id: String, request: ToolUseRequest);
    fn on_tool_result(&self, agent_id: String, tool_id: String, result: String);
    fn on_complete(&self, result: SubagentResult);
    fn on_error(&self, agent_id: String, error: String);
}

#[uniffi::export(callback_interface)]
pub trait CustomTool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    fn schema_json(&self) -> String;
    fn execute(&self, input_json: String) -> ToolExecutionResult;
}
```

## New MuxEngine Methods

### Agent Management

```rust
pub fn register_agent(&self, config: AgentConfig) -> Result<(), MuxFfiError>;
pub fn list_agents(&self) -> Vec<String>;
pub fn spawn_agent(self: Arc<Self>, agent_name: String, task: String, callback: Box<dyn SubagentCallback>);
pub fn resume_agent(self: Arc<Self>, transcript: TranscriptData, callback: Box<dyn SubagentCallback>);
```

### Hooks

```rust
pub fn set_hook_handler(&self, handler: Box<dyn HookHandler>);
pub fn clear_hook_handler(&self);
```

### Custom Tools

```rust
pub fn register_custom_tool(&self, tool: Box<dyn CustomTool>) -> Result<(), MuxFfiError>;
pub fn unregister_tool(&self, name: String) -> Result<(), MuxFfiError>;
pub fn list_tools(&self, workspace_id: String) -> Vec<ToolInfo>;
```

### Multi-Provider

```rust
pub fn set_provider_config(&self, provider: Provider, api_key: String, base_url: Option<String>);
pub fn set_default_provider(&self, provider: Provider);
```

## Internal Architecture

### New MuxEngine Fields

```rust
pub struct MuxEngine {
    // Existing...
    data_dir: PathBuf,
    workspaces: Arc<RwLock<HashMap<String, Workspace>>>,
    conversations: Arc<RwLock<HashMap<String, Vec<Conversation>>>>,
    api_keys: Arc<RwLock<HashMap<Provider, ProviderConfig>>>,

    // NEW
    agent_registry: AgentRegistry,
    hook_handler: Arc<RwLock<Option<Box<dyn HookHandler>>>>,
    custom_tools: Arc<RwLock<HashMap<String, Box<dyn CustomTool>>>>,
    transcript_store: Arc<MemoryTranscriptStore>,
    builtin_tools: Vec<Arc<dyn Tool>>,
}

struct ProviderConfig {
    api_key: String,
    base_url: Option<String>,
}
```

### Data Flow for spawn_agent

1. Look up `AgentConfig` from registry
2. Build `AgentDefinition` from config
3. Create `LlmClient` based on workspace provider
4. Wrap `HookHandler` in `FfiHookBridge`
5. Create `SubAgent` with filtered tool registry
6. Run think-act loop on background thread
7. Stream events via `SubagentCallback`
8. On completion, optionally save transcript

## FFI Bridge Implementations

### FfiHookBridge

Bridges Swift `HookHandler` to Rust `Hook` trait:

```rust
struct FfiHookBridge {
    handler: Arc<Box<dyn HookHandler>>,
}

#[async_trait]
impl Hook for FfiHookBridge {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        let ffi_event = convert_to_ffi(event);
        let response = self.handler.on_event(ffi_event);
        convert_from_ffi(response)
    }
}
```

### FfiToolBridge

Bridges Swift `CustomTool` to Rust `Tool` trait:

```rust
struct FfiToolBridge {
    tool: Box<dyn CustomTool>,
    cached_schema: serde_json::Value,
}

#[async_trait]
impl Tool for FfiToolBridge {
    async fn execute(&self, params: Value) -> Result<ToolResult, Error> {
        let input = serde_json::to_string(&params)?;
        let result = self.tool.execute(input);
        Ok(ToolResult::new(result.content, result.is_error))
    }
}
```

## Error Handling

```rust
#[derive(uniffi::Error)]
pub enum MuxFfiError {
    Engine { message: String },
    AgentNotFound { name: String },
    ToolNotFound { name: String },
    ProviderNotConfigured { provider: String },
    TranscriptInvalid { reason: String },
    HookFailed { reason: String },
}
```

## Files to Modify

| File | Changes |
|------|---------|
| `mux-ffi/src/types.rs` | Add AgentConfig, HookEventType, HookResponse, TranscriptData, SubagentResult, ToolExecutionResult, expand Provider |
| `mux-ffi/src/callback.rs` | Add HookHandler, SubagentCallback, CustomTool traits |
| `mux-ffi/src/engine.rs` | Add new fields, implement all new methods |
| `mux-ffi/src/bridge.rs` | NEW - FfiHookBridge, FfiToolBridge implementations |
| `mux-ffi/src/lib.rs` | Export new modules |

## Testing Strategy

- Unit tests for each bridge type
- Integration test spawning a subagent with mock callback
- Test custom tool registration and execution
- Test hook interception and transform

## Swift Usage Example

```swift
let engine = MuxEngine(dataDir: path)

// Multi-provider
engine.setProviderConfig(.anthropic, apiKey: "sk-...", baseUrl: nil)
engine.setProviderConfig(.gemini, apiKey: "AI...", baseUrl: nil)

// Register agent type
engine.registerAgent(AgentConfig(
    name: "researcher",
    systemPrompt: "You are a research assistant.",
    model: nil,
    allowedTools: ["search", "read_file"],
    deniedTools: ["bash"],
    maxIterations: 20
))

// Set hook handler
engine.setHookHandler(MyHookHandler())

// Register custom tool
engine.registerCustomTool(MyWeatherTool())

// Spawn agent
engine.spawnAgent(
    agentName: "researcher",
    task: "Find information about X",
    callback: MyAgentCallback()
)
```
