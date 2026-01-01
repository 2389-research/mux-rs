# Hooks, Edit Tool, and Agent Resume Design

## Goal

Add three features inspired by Claude Code architecture:
1. **Hook system** - Extensibility points for tool/agent lifecycle
2. **Edit tool** - Precise string replacement vs full file overwrite
3. **Agent resume** - Continue agents from previous transcripts

## 1. Hook System

### Events

```rust
pub enum HookEvent {
    PreToolUse { tool_name: String, input: serde_json::Value },
    PostToolUse { tool_name: String, input: serde_json::Value, result: ToolResult },
    AgentStart { agent_id: String, task: String },
    AgentStop { agent_id: String, result: SubAgentResult },
    Iteration { agent_id: String, iteration: usize },
}
```

### Hook Trait

```rust
#[async_trait]
pub trait Hook: Send + Sync {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error>;
}

pub enum HookAction {
    Continue,
    Block(String),
    Transform(serde_json::Value),
}
```

### HookRegistry

```rust
pub struct HookRegistry {
    hooks: Vec<Arc<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self;
    pub fn register(&mut self, hook: impl Hook + 'static);
    pub async fn fire(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error>;
}
```

### Integration

SubAgent accepts optional HookRegistry and fires events at:
- Before/after tool execution
- Agent start/stop
- Each think-act iteration

## 2. Edit Tool

### Schema

```json
{
    "type": "object",
    "properties": {
        "file_path": { "type": "string" },
        "old_string": { "type": "string" },
        "new_string": { "type": "string" },
        "replace_all": { "type": "boolean", "default": false }
    },
    "required": ["file_path", "old_string", "new_string"]
}
```

### Behavior

1. Read file content
2. Count occurrences of `old_string`
3. If count == 0: error "String not found"
4. If count > 1 and !replace_all: error "String appears N times, use replace_all or add context"
5. Replace and write file

### Safety

- Prevents accidental overwrites from ambiguous matches
- Forces agent to provide sufficient context
- `replace_all` for intentional bulk replacements (rename variable, etc.)

## 3. Agent Resume

### SubAgent Changes

```rust
impl SubAgent {
    pub fn resume(
        agent_id: String,
        definition: AgentDefinition,
        client: Arc<dyn LlmClient>,
        registry: Registry,
        transcript: Vec<Message>,
    ) -> Self;

    pub fn transcript(&self) -> &[Message];
}
```

### TaskTool Changes

Add optional `resume_agent_id` parameter. If provided:
1. Look up transcript by agent_id (requires transcript storage)
2. Use SubAgent::resume() instead of new()

### Transcript Storage

Add TranscriptStore trait for pluggable storage:

```rust
#[async_trait]
pub trait TranscriptStore: Send + Sync {
    async fn save(&self, agent_id: &str, messages: &[Message]) -> Result<(), anyhow::Error>;
    async fn load(&self, agent_id: &str) -> Result<Option<Vec<Message>>, anyhow::Error>;
}
```

Default in-memory implementation included.

## Module Structure

```
src/
├── hook/
│   ├── mod.rs      # HookEvent, Hook trait, HookAction
│   └── registry.rs # HookRegistry
├── tools/
│   └── edit.rs     # NEW: EditTool
└── agent/
    ├── runner.rs   # Updated with hooks and resume
    ├── task.rs     # Updated with resume parameter
    └── transcript.rs # NEW: TranscriptStore
```

## Implementation Order

1. Hook system (foundation for extensibility)
2. Edit tool (standalone, no dependencies)
3. Agent resume (builds on transcript storage)
