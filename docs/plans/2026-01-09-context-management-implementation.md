# Context Management Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add context tracking, warnings, and compaction APIs to mux-ffi for small context models like Apple Foundation Models (4K tokens).

**Architecture:** New `context.rs` module for types and token estimation, update `ChatCallback` with `on_context_warning`, add `ContextUsage` to `ChatResult`, implement TruncateOldest and Summarize compaction modes in engine.

**Tech Stack:** Rust, UniFFI, character-based token estimation (~4 bytes/token)

---

### Task 1: Add Context Types

**Files:**
- Create: `mux-ffi/src/context.rs`
- Modify: `mux-ffi/src/lib.rs`

**Step 1: Create context.rs with types and token estimation**

```rust
// ABOUTME: Context management types and token estimation utilities.
// ABOUTME: Supports small context models like Apple Foundation Models (4K).

/// Approximate bytes per token for estimation (conservative)
pub const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Safety margin to avoid hitting exact limit
pub const SAFETY_MARGIN: f32 = 0.8;

/// Context usage statistics for a conversation
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct ContextUsage {
    pub message_count: u32,
    pub estimated_tokens: u32,
    pub context_limit: Option<u32>,
    pub usage_percent: Option<f32>,
}

impl ContextUsage {
    pub fn new(message_count: u32, estimated_tokens: u32, context_limit: Option<u32>) -> Self {
        let usage_percent = context_limit.map(|limit| {
            if limit > 0 {
                (estimated_tokens as f32 / limit as f32) * 100.0
            } else {
                0.0
            }
        });
        Self {
            message_count,
            estimated_tokens,
            context_limit,
            usage_percent,
        }
    }
}

/// How to handle context overflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, uniffi::Enum)]
pub enum CompactionMode {
    /// Ask LLM to summarize old messages (default for large context)
    #[default]
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
    pub warning_threshold: f32,
}

impl ModelContextConfig {
    pub fn new(model: String, context_limit: u32) -> Self {
        Self {
            model,
            context_limit,
            compaction_mode: CompactionMode::default(),
            warning_threshold: 0.8,
        }
    }

    pub fn with_compaction_mode(mut self, mode: CompactionMode) -> Self {
        self.compaction_mode = mode;
        self
    }

    pub fn with_warning_threshold(mut self, threshold: f32) -> Self {
        self.warning_threshold = threshold;
        self
    }
}

/// Estimate token count from text using byte-based heuristic
pub fn estimate_tokens(text: &str) -> u32 {
    let bytes = text.len();
    ((bytes + APPROX_BYTES_PER_TOKEN - 1) / APPROX_BYTES_PER_TOKEN) as u32
}

/// Calculate effective limit with safety margin
pub fn effective_limit(limit: u32) -> u32 {
    (limit as f32 * SAFETY_MARGIN) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // 4 bytes = 1 token
        assert_eq!(estimate_tokens("test"), 1);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        // 12 bytes = 3 tokens
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_effective_limit() {
        // 4096 * 0.8 = 3276
        assert_eq!(effective_limit(4096), 3276);
    }

    #[test]
    fn test_context_usage_percent() {
        let usage = ContextUsage::new(5, 2000, Some(4096));
        assert!(usage.usage_percent.is_some());
        let percent = usage.usage_percent.unwrap();
        assert!(percent > 48.0 && percent < 49.0);
    }

    #[test]
    fn test_context_usage_no_limit() {
        let usage = ContextUsage::new(5, 2000, None);
        assert!(usage.usage_percent.is_none());
    }
}
```

**Step 2: Add module to lib.rs**

In `mux-ffi/src/lib.rs`, add after `mod callback_client;`:

```rust
mod context;
pub use context::*;
```

**Step 3: Run tests**

Run: `cargo test -p mux-ffi context`
Expected: All context tests pass

**Step 4: Commit**

```bash
git add mux-ffi/src/context.rs mux-ffi/src/lib.rs
git commit -m "feat(mux-ffi): add context management types and token estimation"
```

---

### Task 2: Update ChatResult with ContextUsage

**Files:**
- Modify: `mux-ffi/src/callback.rs:15-23`

**Step 1: Update ChatResult struct**

Replace the `ChatResult` struct in `callback.rs`:

```rust
/// Represents the final result of a chat completion.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatResult {
    pub conversation_id: String,
    pub final_text: String,
    pub tool_use_count: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub context_usage: crate::context::ContextUsage,
}
```

**Step 2: Run tests to verify compilation**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds (tests may fail until engine is updated)

**Step 3: Commit**

```bash
git add mux-ffi/src/callback.rs
git commit -m "feat(mux-ffi): add context_usage to ChatResult"
```

---

### Task 3: Add on_context_warning to ChatCallback

**Files:**
- Modify: `mux-ffi/src/callback.rs:25-43`

**Step 1: Add on_context_warning method**

Update `ChatCallback` trait to add the new method after `on_error`:

```rust
/// Callback interface that Swift implements to receive streaming chat updates.
/// UniFFI generates the necessary bridging code for cross-language calls.
#[uniffi::export(callback_interface)]
pub trait ChatCallback: Send + Sync {
    /// Called when new text content is streamed from the LLM.
    fn on_text_delta(&self, text: String);

    /// Called when the LLM requests to use a tool.
    fn on_tool_use(&self, request: ToolUseRequest);

    /// Called when a tool execution completes with a result.
    fn on_tool_result(&self, tool_id: String, result: String);

    /// Called when the entire chat completion finishes successfully.
    fn on_complete(&self, result: ChatResult);

    /// Called when an error occurs during chat processing.
    fn on_error(&self, error: String);

    /// Called when context usage exceeds the warning threshold.
    /// Swift can use this to show a UI warning or trigger compaction.
    fn on_context_warning(&self, usage: crate::context::ContextUsage);
}
```

**Step 2: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add mux-ffi/src/callback.rs
git commit -m "feat(mux-ffi): add on_context_warning to ChatCallback"
```

---

### Task 4: Add Model Context Config Storage to Engine

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add model_context_configs field to MuxEngine struct**

Find the `MuxEngine` struct definition and add after `callback_providers`:

```rust
    /// Per-model context configuration (context limit, compaction mode, etc.)
    model_context_configs: Arc<RwLock<HashMap<String, ModelContextConfig>>>,
```

**Step 2: Initialize in MuxEngine::new()**

In the `new()` function, add to the struct initialization:

```rust
            model_context_configs: Arc::new(RwLock::new(HashMap::new())),
```

**Step 3: Add import at top of engine.rs**

Add to imports:

```rust
use crate::context::{CompactionMode, ContextUsage, ModelContextConfig, estimate_tokens, effective_limit};
```

**Step 4: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 5: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add model context config storage to engine"
```

---

### Task 5: Implement Model Context Config API Methods

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add set_model_context_config method**

Add to the `#[uniffi::export]` impl block:

```rust
    /// Set context configuration for a specific model.
    /// Call this during app init to configure small context models.
    #[uniffi::method]
    pub fn set_model_context_config(&self, config: ModelContextConfig) {
        self.model_context_configs
            .write()
            .insert(config.model.clone(), config);
    }

    /// Get context configuration for a model.
    /// Returns a default config if not explicitly set.
    #[uniffi::method]
    pub fn get_model_context_config(&self, model: String) -> ModelContextConfig {
        self.model_context_configs
            .read()
            .get(&model)
            .cloned()
            .unwrap_or_else(|| ModelContextConfig {
                model,
                context_limit: 0, // 0 means no limit
                compaction_mode: CompactionMode::Summarize,
                warning_threshold: 0.8,
            })
    }
```

**Step 2: Add test**

Add to the `tests` module in `lib.rs`:

```rust
    #[test]
    fn test_model_context_config() {
        let engine = MuxEngine::new("/tmp/mux-test-context".to_string()).unwrap();

        // Default config
        let default_config = engine.get_model_context_config("unknown-model".to_string());
        assert_eq!(default_config.context_limit, 0);
        assert_eq!(default_config.compaction_mode, CompactionMode::Summarize);

        // Set custom config
        let config = ModelContextConfig {
            model: "foundation-3b".to_string(),
            context_limit: 4096,
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.8,
        };
        engine.set_model_context_config(config.clone());

        let retrieved = engine.get_model_context_config("foundation-3b".to_string());
        assert_eq!(retrieved.context_limit, 4096);
        assert_eq!(retrieved.compaction_mode, CompactionMode::TruncateOldest);
    }
```

**Step 3: Run tests**

Run: `cargo test -p mux-ffi test_model_context_config`
Expected: PASS

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs mux-ffi/src/lib.rs
git commit -m "feat(mux-ffi): add set/get_model_context_config API methods"
```

---

### Task 6: Implement get_context_usage

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add helper to estimate conversation tokens**

Add a private helper method:

```rust
    /// Estimate total tokens in a conversation's message history
    fn estimate_conversation_tokens(&self, conversation_id: &str) -> (u32, u32) {
        let history = self.message_history.read();
        if let Some(messages) = history.get(conversation_id) {
            let message_count = messages.len() as u32;
            let total_tokens: u32 = messages
                .iter()
                .map(|m| {
                    m.content
                        .iter()
                        .map(|block| match block {
                            mux::llm::ContentBlock::Text { text } => estimate_tokens(text),
                            mux::llm::ContentBlock::ToolUse { input, .. } => {
                                estimate_tokens(&input.to_string())
                            }
                            mux::llm::ContentBlock::ToolResult { content, .. } => {
                                estimate_tokens(content)
                            }
                        })
                        .sum::<u32>()
                })
                .sum();
            (message_count, total_tokens)
        } else {
            (0, 0)
        }
    }
```

**Step 2: Add get_context_usage method**

Add to the `#[uniffi::export]` impl block:

```rust
    /// Get current context usage for a conversation.
    #[uniffi::method]
    pub fn get_context_usage(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError> {
        // Check conversation exists
        let conversations = self.conversations.read();
        let exists = conversations
            .values()
            .flatten()
            .any(|c| c.id == conversation_id);
        if !exists {
            return Err(MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            });
        }

        let (message_count, estimated_tokens) = self.estimate_conversation_tokens(&conversation_id);

        // Get workspace to find model
        let workspace_id = self.get_workspace_for_conversation(&conversation_id);
        let context_limit = if let Some(ws_id) = workspace_id {
            let workspaces = self.workspaces.read();
            workspaces.get(&ws_id).and_then(|ws| {
                ws.llm_config.as_ref().and_then(|config| {
                    let model_config = self.model_context_configs.read();
                    model_config.get(&config.model).map(|c| c.context_limit)
                })
            })
        } else {
            None
        };

        Ok(ContextUsage::new(message_count, estimated_tokens, context_limit))
    }
```

**Step 3: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add get_context_usage API method"
```

---

### Task 7: Implement clear_context

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add clear_context method**

Add to the `#[uniffi::export]` impl block:

```rust
    /// Clear conversation context (start fresh, keep conversation ID).
    /// Useful when context is full and user wants to start over.
    #[uniffi::method]
    pub fn clear_context(&self, conversation_id: String) -> Result<(), MuxFfiError> {
        // Check conversation exists
        let conversations = self.conversations.read();
        let exists = conversations
            .values()
            .flatten()
            .any(|c| c.id == conversation_id);
        if !exists {
            return Err(MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            });
        }
        drop(conversations);

        // Clear message history
        {
            let mut history = self.message_history.write();
            if let Some(messages) = history.get_mut(&conversation_id) {
                messages.clear();
            }
        }

        // Persist cleared state
        self.save_messages(&conversation_id);

        Ok(())
    }
```

**Step 2: Add test**

Add to tests in `lib.rs`:

```rust
    #[test]
    fn test_clear_context() {
        let engine = MuxEngine::new("/tmp/mux-test-clear".to_string()).unwrap();
        let ws = engine.create_workspace("Test".to_string(), None).unwrap();
        let conv = engine.create_conversation(ws.id.clone(), "Test Conv".to_string()).unwrap();

        // Clear should work on empty conversation
        engine.clear_context(conv.id.clone()).unwrap();

        // Clear nonexistent should fail
        let result = engine.clear_context("nonexistent".to_string());
        assert!(result.is_err());

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }
```

**Step 3: Run tests**

Run: `cargo test -p mux-ffi test_clear_context`
Expected: PASS

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs mux-ffi/src/lib.rs
git commit -m "feat(mux-ffi): add clear_context API method"
```

---

### Task 8: Implement TruncateOldest Compaction

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add truncate_oldest helper**

Add a private helper method:

```rust
    /// Truncate oldest messages to fit within token limit.
    /// Keeps most recent messages, drops oldest.
    fn truncate_oldest(&self, conversation_id: &str, target_tokens: u32) {
        let mut history = self.message_history.write();
        if let Some(messages) = history.get_mut(conversation_id) {
            // Calculate tokens from end, keep messages that fit
            let mut kept_tokens = 0u32;
            let mut keep_from = messages.len();

            for (i, msg) in messages.iter().enumerate().rev() {
                let msg_tokens: u32 = msg
                    .content
                    .iter()
                    .map(|block| match block {
                        mux::llm::ContentBlock::Text { text } => estimate_tokens(text),
                        mux::llm::ContentBlock::ToolUse { input, .. } => {
                            estimate_tokens(&input.to_string())
                        }
                        mux::llm::ContentBlock::ToolResult { content, .. } => {
                            estimate_tokens(content)
                        }
                    })
                    .sum();

                if kept_tokens + msg_tokens <= target_tokens {
                    kept_tokens += msg_tokens;
                    keep_from = i;
                } else {
                    break;
                }
            }

            // Remove messages before keep_from
            if keep_from > 0 {
                messages.drain(0..keep_from);
            }
        }
    }
```

**Step 2: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add truncate_oldest compaction helper"
```

---

### Task 9: Implement compact_context API

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add compact_context method**

Add to the `#[uniffi::export]` impl block:

```rust
    /// Manually trigger context compaction.
    /// Uses the configured compaction mode for the conversation's model.
    /// For TruncateOldest: drops oldest messages to fit in limit.
    /// For Summarize: (TODO) asks LLM to summarize old messages.
    #[uniffi::method]
    pub fn compact_context(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError> {
        // Check conversation exists and get workspace
        let workspace_id = self.get_workspace_for_conversation(&conversation_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            })?;

        // Get model config
        let (model, config) = {
            let workspaces = self.workspaces.read();
            let ws = workspaces.get(&workspace_id).ok_or_else(|| MuxFfiError::Engine {
                message: "Workspace not found".to_string(),
            })?;

            let model = ws.llm_config
                .as_ref()
                .map(|c| c.model.clone())
                .unwrap_or_default();

            let model_configs = self.model_context_configs.read();
            let config = model_configs.get(&model).cloned();
            (model, config)
        };

        // If no config or no limit, nothing to compact
        let config = match config {
            Some(c) if c.context_limit > 0 => c,
            _ => return self.get_context_usage(conversation_id),
        };

        let target_tokens = effective_limit(config.context_limit);

        match config.compaction_mode {
            CompactionMode::TruncateOldest => {
                self.truncate_oldest(&conversation_id, target_tokens);
                self.save_messages(&conversation_id);
            }
            CompactionMode::Summarize => {
                // TODO: Implement LLM-based summarization
                // For now, fall back to truncate
                self.truncate_oldest(&conversation_id, target_tokens);
                self.save_messages(&conversation_id);
            }
        }

        self.get_context_usage(conversation_id)
    }
```

**Step 2: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 3: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add compact_context API method"
```

---

### Task 10: Update send_message to Report Context Usage

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Find the send_message_impl function**

Locate where `ChatResult` is created in `send_message_impl` (around line 1228 and other places where `on_complete` is called).

**Step 2: Update ChatResult creation to include context_usage**

Every place where `ChatResult` is created needs to include `context_usage`. Find all occurrences and update them. Example:

```rust
let context_usage = self.get_context_usage(conversation_id.clone()).unwrap_or_default();

callback.on_complete(ChatResult {
    conversation_id: conversation_id.clone(),
    final_text: response_text.clone(),
    tool_use_count,
    input_tokens: total_input_tokens,
    output_tokens: total_output_tokens,
    context_usage,
});
```

**Step 3: Add ContextUsage::default() impl**

In `context.rs`, the `ContextUsage` already has `#[derive(Default)]`, so this should work.

**Step 4: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 5: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): include context_usage in ChatResult"
```

---

### Task 11: Add Context Warning Check

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add check_and_warn_context helper**

Add a private helper method:

```rust
    /// Check context usage and fire warning callback if above threshold.
    /// Returns true if warning was fired.
    fn check_and_warn_context(
        &self,
        conversation_id: &str,
        callback: &dyn ChatCallback,
    ) -> bool {
        let usage = match self.get_context_usage(conversation_id.to_string()) {
            Ok(u) => u,
            Err(_) => return false,
        };

        // Get model config for threshold
        let threshold = self.get_workspace_for_conversation(conversation_id)
            .and_then(|ws_id| {
                let workspaces = self.workspaces.read();
                workspaces.get(&ws_id).and_then(|ws| {
                    ws.llm_config.as_ref().and_then(|config| {
                        let model_configs = self.model_context_configs.read();
                        model_configs.get(&config.model).map(|c| c.warning_threshold)
                    })
                })
            })
            .unwrap_or(0.8);

        if let Some(percent) = usage.usage_percent {
            if percent >= threshold * 100.0 {
                callback.on_context_warning(usage);
                return true;
            }
        }

        false
    }
```

**Step 2: Call check_and_warn_context before on_complete**

In `send_message_impl`, before calling `callback.on_complete(...)`, add:

```rust
self.check_and_warn_context(&conversation_id, callback.as_ref());
```

**Step 3: Run tests**

Run: `cargo test -p mux-ffi`
Expected: Compilation succeeds

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add context warning callback"
```

---

### Task 12: Add Integration Test

**Files:**
- Modify: `mux-ffi/src/lib.rs`

**Step 1: Add integration test for context workflow**

Add to tests module:

```rust
    #[test]
    fn test_context_workflow() {
        use crate::context::{CompactionMode, ModelContextConfig};

        let engine = MuxEngine::new("/tmp/mux-test-context-flow".to_string()).unwrap();

        // Create workspace with model config
        let ws = engine.create_workspace("Context Test".to_string(), None).unwrap();

        // Configure small context model
        engine.set_model_context_config(ModelContextConfig {
            model: "test-small-model".to_string(),
            context_limit: 100, // Very small for testing
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.5,
        });

        // Create conversation
        let conv = engine.create_conversation(ws.id.clone(), "Context Test".to_string()).unwrap();

        // Check initial usage
        let usage = engine.get_context_usage(conv.id.clone()).unwrap();
        assert_eq!(usage.message_count, 0);
        assert_eq!(usage.estimated_tokens, 0);

        // Clear context works
        engine.clear_context(conv.id.clone()).unwrap();

        // Compact context works (no-op on empty)
        let usage = engine.compact_context(conv.id.clone()).unwrap();
        assert_eq!(usage.message_count, 0);

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }
```

**Step 2: Run tests**

Run: `cargo test -p mux-ffi test_context_workflow`
Expected: PASS

**Step 3: Commit**

```bash
git add mux-ffi/src/lib.rs
git commit -m "test(mux-ffi): add context management integration test"
```

---

### Task 13: Bump Version and Final Test

**Files:**
- Modify: `mux-ffi/Cargo.toml`

**Step 1: Bump version to 0.5.0**

This is a feature release with new APIs. Update version:

```toml
version = "0.5.0"
```

**Step 2: Run full test suite**

Run: `cargo test -p mux-ffi`
Expected: All tests pass

**Step 3: Run cargo clippy**

Run: `cargo clippy -p mux-ffi`
Expected: No warnings (or acceptable warnings only)

**Step 4: Commit and tag**

```bash
git add mux-ffi/Cargo.toml Cargo.lock
git commit -m "chore(mux-ffi): bump version to 0.5.0 for context management"
git tag v0.5.0-ffi
```

---

## Summary

13 tasks covering:
1. Context types and token estimation
2. ChatResult with ContextUsage
3. ChatCallback with on_context_warning
4. Engine storage for model configs
5. set/get_model_context_config APIs
6. get_context_usage API
7. clear_context API
8. TruncateOldest compaction
9. compact_context API
10. Context usage in ChatResult
11. Context warning callback
12. Integration test
13. Version bump and release

**Note:** LLM-based summarization (CompactionMode::Summarize) is stubbed - it falls back to truncation. This can be implemented in a follow-up task once the basic infrastructure is in place.
