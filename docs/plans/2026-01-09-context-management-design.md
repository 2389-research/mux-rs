# Context Management APIs for mux-ffi

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable mux-ffi to work with small context models (4K tokens) like Apple Foundation Models, with context tracking, warnings, and automatic compaction.

**Architecture:** Per-model context configuration with two compaction modes - LLM summarization (default for large context) and truncate-oldest (opt-in for small context). Context usage tracked and reported via callbacks.

**Tech Stack:** Rust, UniFFI, character-based token estimation with safety margin

---

## Data Types

```rust
/// Context usage statistics for a conversation
#[derive(Debug, Clone, uniffi::Record)]
pub struct ContextUsage {
    pub message_count: u32,
    pub estimated_tokens: u32,
    pub context_limit: Option<u32>,
    pub usage_percent: Option<f32>,
}

/// How to handle context overflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum CompactionMode {
    /// Ask LLM to summarize old messages (default for large context)
    Summarize,
    /// Drop oldest messages, keep recent (for small context models)
    TruncateOldest,
}

/// Per-model context configuration
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelContextConfig {
    pub model: String,
    pub context_limit: u32,
    pub compaction_mode: CompactionMode,
    pub warning_threshold: f32,  // e.g., 0.8 for 80%
}
```

**Defaults:**
- `compaction_mode`: `Summarize`
- `warning_threshold`: `0.8` (80%)
- `context_limit`: None (no limit) unless configured

---

## MuxEngine API Methods

```rust
impl MuxEngine {
    // === Configuration ===

    /// Set context configuration for a specific model
    pub fn set_model_context_config(&self, config: ModelContextConfig);

    /// Get context configuration for a model (returns defaults if not set)
    pub fn get_model_context_config(&self, model: String) -> ModelContextConfig;

    // === Query ===

    /// Get current context usage for a conversation
    pub fn get_context_usage(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError>;

    // === Manual Control ===

    /// Clear conversation context (start fresh, keep conversation ID)
    pub fn clear_context(&self, conversation_id: String) -> Result<(), MuxFfiError>;

    /// Manually trigger compaction (uses configured mode for current model)
    pub fn compact_context(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError>;
}
```

---

## Callback Interface Changes

```rust
/// Updated ChatResult (returned in onComplete)
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatResult {
    pub conversation_id: String,
    pub final_text: String,
    pub tool_use_count: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub context_usage: ContextUsage,  // NEW
}

/// Updated ChatCallback interface
#[uniffi::export(callback_interface)]
pub trait ChatCallback: Send + Sync {
    fn on_text_delta(&self, text: String);
    fn on_tool_use(&self, request: ToolUseRequest);
    fn on_tool_result(&self, tool_id: String, result: String);
    fn on_complete(&self, result: ChatResult);
    fn on_error(&self, error: String);

    // NEW: Called when context usage exceeds warning threshold
    fn on_context_warning(&self, usage: ContextUsage);
}
```

**When `on_context_warning` fires:**
- After each assistant response, before `on_complete`
- Only if `usage_percent >= warning_threshold`
- Swift can show UI warning, offer to clear/compact, or just log

---

## Internal Implementation

### Token Estimation

```rust
const APPROX_BYTES_PER_TOKEN: usize = 4;
const SAFETY_MARGIN: f32 = 0.8;  // Treat 4K as 3.2K effective

fn estimate_tokens(text: &str) -> u32 {
    (text.len() / APPROX_BYTES_PER_TOKEN) as u32
}

fn effective_limit(limit: u32) -> u32 {
    (limit as f32 * SAFETY_MARGIN) as u32
}
```

### Compaction Strategies

**TruncateOldest:**
1. Keep system prompt (if any)
2. Keep last N messages that fit in limit
3. Drop oldest messages
4. Fast, no LLM call

**Summarize:**
1. Ask current LLM to summarize old messages
2. Keep system prompt + recent user messages + summary
3. Replace history with compacted version
4. Fires `on_context_warning` before summarizing

### Auto-compaction Triggers

- Before sending message, if `estimated_tokens > effective_limit`
- Run compaction based on model's `compaction_mode`
- Then send message with compacted context

---

## Summarization Prompt

For `.summarize` mode:

```
Summarize the conversation so far in a concise paragraph.
Focus on:
- Key topics discussed
- Decisions made
- Important context the assistant needs to remember
- Any pending tasks or questions

Keep under 500 tokens. Be factual, not conversational.
```

---

## Swift Integration Example

```swift
// App startup - register Foundation Models
engine.registerLlmProvider(name: "foundation-models", provider: myProvider)
engine.setDefaultProvider(provider: .custom(name: "foundation-models"))

// Configure small context behavior (opt-in to truncation)
engine.setModelContextConfig(ModelContextConfig(
    model: "apple-foundation-3b",
    contextLimit: 4096,
    compactionMode: .truncateOldest,
    warningThreshold: 0.8
))

// Cloud models use summarization by default
engine.setModelContextConfig(ModelContextConfig(
    model: "claude-sonnet-4-20250514",
    contextLimit: 200000,
    compactionMode: .summarize,
    warningThreshold: 0.8
))

// Implement callback
class MyChatCallback: ChatCallback {
    func onContextWarning(usage: ContextUsage) {
        // Show warning in UI
        showContextWarning(percent: usage.usagePercent ?? 0)
    }

    func onComplete(result: ChatResult) {
        // Update context indicator
        updateContextBar(usage: result.contextUsage)
    }
}
```

---

## File Structure

```
mux-ffi/src/
├── context.rs          # NEW: ContextUsage, CompactionMode, token estimation
├── compaction.rs       # NEW: TruncateOldest, Summarize implementations
├── engine.rs           # Updated: model configs, auto-compaction hooks
├── types.rs            # Updated: ChatResult with context_usage
├── callback.rs         # Updated: on_context_warning in ChatCallback
├── lib.rs              # Updated: export new modules
```

---

## Testing Strategy

1. **Unit tests for token estimation** - verify bytes-to-tokens conversion
2. **Unit tests for TruncateOldest** - verify message ordering, system prompt preservation
3. **Integration test for auto-compaction** - fill context, verify compaction triggers
4. **Integration test for warning callback** - verify threshold triggers callback

---

## Migration Notes

- `ChatResult` gains new field - Swift must update callback implementation
- `ChatCallback` gains new method - Swift must implement `on_context_warning`
- Existing conversations work unchanged (no context limit = no compaction)
