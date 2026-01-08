# Callback-Based LLM Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement callback-based LLM provider for mux-ffi

**Architecture:** UniFFI callback interfaces + Rust adapter implementing LlmClient

**Tech Stack:** UniFFI 0.30, tokio channels, async-trait

---

## Task 1: Add LlmUsage FFI type

**Files:**
- Modify: `mux-ffi/src/types.rs`

**Step 1: Add LlmUsage struct**

```rust
/// Usage statistics from LLM generation.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/types.rs
git commit -m "feat(mux-ffi): add LlmUsage FFI type"
```

---

## Task 2: Add LlmRequest FFI type

**Files:**
- Modify: `mux-ffi/src/types.rs`

**Step 1: Add LlmRequest struct**

```rust
/// Request for LLM generation via callback provider.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LlmRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<FfiToolDefinition>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
}
```

Note: `ChatMessage` and `FfiToolDefinition` should already exist. If `FfiToolDefinition` doesn't exist, we need to add it - it's a UniFFI-safe version of `ToolDefinition`.

**Step 2: Add FfiToolDefinition if needed**

```rust
/// Tool definition for FFI - UniFFI-safe version of ToolDefinition.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,  // JSON string since serde_json::Value isn't FFI-safe
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/types.rs
git commit -m "feat(mux-ffi): add LlmRequest and FfiToolDefinition types"
```

---

## Task 3: Add LlmStreamCallback trait

**Files:**
- Modify: `mux-ffi/src/callback.rs`

**Step 1: Add the callback trait**

```rust
/// Callback that Rust provides to Swift for streaming LLM results.
/// Swift calls these methods as tokens arrive from the LLM.
#[uniffi::export(callback_interface)]
pub trait LlmStreamCallback: Send + Sync {
    /// Called when text content is generated.
    fn on_text(&self, delta: String);

    /// Called when the LLM wants to use a tool.
    /// Arguments is a JSON string of the tool parameters.
    fn on_tool_call(&self, id: String, name: String, arguments: String);

    /// Called when generation completes successfully.
    fn on_complete(&self, usage: crate::types::LlmUsage);

    /// Called when an error occurs during generation.
    fn on_error(&self, error: String);
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/callback.rs
git commit -m "feat(mux-ffi): add LlmStreamCallback trait"
```

---

## Task 4: Add LlmProvider trait

**Files:**
- Modify: `mux-ffi/src/callback.rs`

**Step 1: Add the provider trait**

```rust
/// Callback interface that Swift implements to provide LLM generation.
/// This allows on-device models (like Apple Foundation Models) to integrate
/// with Mux's orchestration system.
#[uniffi::export(callback_interface)]
pub trait LlmProvider: Send + Sync {
    /// Generate a response for the given request.
    /// Call the callback methods as tokens arrive:
    /// - on_text() for text chunks
    /// - on_tool_call() when model wants to use a tool
    /// - on_complete() when done
    /// - on_error() if something goes wrong
    fn generate(&self, request: crate::types::LlmRequest, callback: Box<dyn LlmStreamCallback>);
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/callback.rs
git commit -m "feat(mux-ffi): add LlmProvider callback trait"
```

---

## Task 5: Extend Provider enum with Custom variant

**Files:**
- Modify: `mux-ffi/src/types.rs`

**Step 1: Add Custom variant to Provider**

Find the existing Provider enum and add the Custom variant:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    Ollama,
    Custom { name: String },
}
```

**Step 2: Update any match statements in engine.rs**

Search for `match provider` or `Provider::` and add handling for `Provider::Custom { name }`.

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/types.rs mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add Custom variant to Provider enum"
```

---

## Task 6: Create CallbackLlmClient adapter

**Files:**
- Create: `mux-ffi/src/callback_client.rs`
- Modify: `mux-ffi/src/lib.rs` (add module)

**Step 1: Create the file with struct and constructor**

```rust
// ABOUTME: Adapts Swift's LlmProvider callback to Rust's LlmClient trait.
// ABOUTME: Enables on-device models to integrate with Mux orchestration.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use mux::error::LlmError;
use mux::llm::{ContentBlock, LlmClient, Request, Response, StreamEvent, StopReason, Usage};

use crate::callback::{LlmProvider, LlmStreamCallback};
use crate::types::{ChatMessage, FfiToolDefinition, LlmRequest, LlmUsage};

/// Adapter that wraps a Swift-provided LlmProvider as a Rust LlmClient.
pub struct CallbackLlmClient {
    provider: Arc<Box<dyn LlmProvider>>,
}

impl CallbackLlmClient {
    pub fn new(provider: Box<dyn LlmProvider>) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }
}
```

**Step 2: Add module to lib.rs**

```rust
mod callback_client;
pub use callback_client::CallbackLlmClient;
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/callback_client.rs mux-ffi/src/lib.rs
git commit -m "feat(mux-ffi): add CallbackLlmClient struct"
```

---

## Task 7: Implement conversion helpers

**Files:**
- Modify: `mux-ffi/src/callback_client.rs`

**Step 1: Add Request to LlmRequest conversion**

```rust
impl CallbackLlmClient {
    /// Convert mux Request to FFI LlmRequest
    fn convert_request(req: &Request) -> LlmRequest {
        let messages: Vec<ChatMessage> = req
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: match m.role {
                    mux::llm::Role::User => crate::types::ChatRole::User,
                    mux::llm::Role::Assistant => crate::types::ChatRole::Assistant,
                },
                content: m.content.iter().filter_map(|b| {
                    match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None, // Tool results handled separately
                    }
                }).collect::<Vec<_>>().join(""),
            })
            .collect();

        let tools: Vec<FfiToolDefinition> = req
            .tools
            .iter()
            .map(|t| FfiToolDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema_json: serde_json::to_string(&t.input_schema).unwrap_or_default(),
            })
            .collect();

        LlmRequest {
            messages,
            tools,
            system_prompt: req.system.clone(),
            max_tokens: req.max_tokens,
        }
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/callback_client.rs
git commit -m "feat(mux-ffi): add request conversion for CallbackLlmClient"
```

---

## Task 8: Implement StreamCallbackImpl for collecting results

**Files:**
- Modify: `mux-ffi/src/callback_client.rs`

**Step 1: Add internal event type**

```rust
/// Internal events from the callback.
#[derive(Debug)]
enum CallbackEvent {
    Text(String),
    ToolCall { id: String, name: String, arguments: String },
    Complete(LlmUsage),
    Error(String),
}
```

**Step 2: Add StreamCallbackImpl**

```rust
/// Implementation of LlmStreamCallback that sends events to a channel.
struct StreamCallbackImpl {
    sender: Mutex<Option<mpsc::UnboundedSender<CallbackEvent>>>,
}

impl StreamCallbackImpl {
    fn new(sender: mpsc::UnboundedSender<CallbackEvent>) -> Self {
        Self {
            sender: Mutex::new(Some(sender)),
        }
    }
}

impl LlmStreamCallback for StreamCallbackImpl {
    fn on_text(&self, delta: String) {
        if let Some(sender) = self.sender.lock().as_ref() {
            let _ = sender.send(CallbackEvent::Text(delta));
        }
    }

    fn on_tool_call(&self, id: String, name: String, arguments: String) {
        if let Some(sender) = self.sender.lock().as_ref() {
            let _ = sender.send(CallbackEvent::ToolCall { id, name, arguments });
        }
    }

    fn on_complete(&self, usage: LlmUsage) {
        if let Some(sender) = self.sender.lock().take() {
            let _ = sender.send(CallbackEvent::Complete(usage));
        }
    }

    fn on_error(&self, error: String) {
        if let Some(sender) = self.sender.lock().take() {
            let _ = sender.send(CallbackEvent::Error(error));
        }
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/callback_client.rs
git commit -m "feat(mux-ffi): add StreamCallbackImpl for result collection"
```

---

## Task 9: Implement LlmClient::create_message

**Files:**
- Modify: `mux-ffi/src/callback_client.rs`

**Step 1: Implement create_message**

```rust
#[async_trait]
impl LlmClient for CallbackLlmClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let llm_request = Self::convert_request(req);

        // Create channel for receiving results
        let (tx, mut rx) = mpsc::unbounded_channel();
        let callback = Box::new(StreamCallbackImpl::new(tx));

        // Call Swift provider (this blocks in Swift, streams via callback)
        let provider = self.provider.clone();
        tokio::task::spawn_blocking(move || {
            provider.generate(llm_request, callback);
        })
        .await
        .map_err(|e| LlmError::Api {
            status: 0,
            message: format!("Provider task failed: {}", e),
        })?;

        // Collect results from channel
        let mut text_content = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
        let mut usage = Usage::default();

        while let Some(event) = rx.recv().await {
            match event {
                CallbackEvent::Text(delta) => text_content.push_str(&delta),
                CallbackEvent::ToolCall { id, name, arguments } => {
                    tool_calls.push((id, name, arguments));
                }
                CallbackEvent::Complete(llm_usage) => {
                    usage.input_tokens = llm_usage.input_tokens;
                    usage.output_tokens = llm_usage.output_tokens;
                    break;
                }
                CallbackEvent::Error(error) => {
                    return Err(LlmError::Api {
                        status: 0,
                        message: error,
                    });
                }
            }
        }

        // Build content blocks
        let mut content: Vec<ContentBlock> = Vec::new();
        if !text_content.is_empty() {
            content.push(ContentBlock::Text { text: text_content });
        }
        for (id, name, arguments) in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&arguments).unwrap_or_default();
            content.push(ContentBlock::ToolUse { id, name, input });
        }

        // Determine stop reason
        let stop_reason = if content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. })) {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        Ok(Response {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            stop_reason,
            model: "callback-provider".to_string(),
            usage,
        })
    }

    fn create_message_stream(
        &self,
        _req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        // For now, return an empty stream - streaming can be added later
        Box::pin(futures::stream::empty())
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/callback_client.rs
git commit -m "feat(mux-ffi): implement LlmClient for CallbackLlmClient"
```

---

## Task 10: Add provider storage to MuxEngine

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add field to MuxEngine struct**

```rust
/// Registered callback LLM providers, keyed by name
callback_providers: Arc<RwLock<HashMap<String, Arc<CallbackLlmClient>>>>,
```

**Step 2: Initialize in MuxEngine::new**

```rust
callback_providers: Arc::new(RwLock::new(HashMap::new())),
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add callback provider storage to MuxEngine"
```

---

## Task 11: Implement register_llm_provider

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Add the method**

```rust
/// Register a callback-based LLM provider.
/// The provider can then be used via `set_default_provider(Provider::Custom { name })`.
pub fn register_llm_provider(&self, name: String, provider: Box<dyn LlmProvider>) {
    let client = CallbackLlmClient::new(provider);
    self.callback_providers
        .write()
        .insert(name, Arc::new(client));
}

/// Unregister a callback LLM provider.
pub fn unregister_llm_provider(&self, name: String) -> Result<(), MuxFfiError> {
    if self.callback_providers.write().remove(&name).is_none() {
        return Err(MuxFfiError::Engine {
            message: format!("LLM provider '{}' not found", name),
        });
    }
    Ok(())
}

/// List registered callback LLM providers.
pub fn list_llm_providers(&self) -> Vec<String> {
    self.callback_providers.read().keys().cloned().collect()
}
```

**Step 2: Add import for LlmProvider and CallbackLlmClient**

```rust
use crate::callback::LlmProvider;
use crate::callback_client::CallbackLlmClient;
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): add register/unregister_llm_provider methods"
```

---

## Task 12: Update do_send_message to use callback providers

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Update client creation logic**

Find the section in `do_send_message` where the client is created. Currently it's:

```rust
let client = AnthropicClient::new(&api_key);
```

Replace with logic that checks for custom provider:

```rust
// Get the LLM client based on provider type
let provider = self.default_provider.read().clone();
let client: Arc<dyn LlmClient> = match &provider {
    Provider::Custom { name } => {
        self.callback_providers
            .read()
            .get(name)
            .cloned()
            .map(|c| c as Arc<dyn LlmClient>)
            .ok_or_else(|| format!("Callback provider '{}' not registered", name))?
    }
    Provider::Anthropic => {
        let api_key = self.api_keys.read().get(&Provider::Anthropic)
            .map(|c| c.api_key.clone())
            .ok_or_else(|| "Anthropic API key not set".to_string())?;
        Arc::new(AnthropicClient::new(&api_key))
    }
    Provider::OpenAI | Provider::Ollama => {
        let config = self.api_keys.read().get(&provider)
            .cloned()
            .ok_or_else(|| format!("{:?} not configured", provider))?;
        let mut c = OpenAIClient::new(&config.api_key);
        if let Some(url) = &config.base_url {
            c = c.with_base_url(url);
        }
        Arc::new(c)
    }
    Provider::Gemini => {
        let api_key = self.api_keys.read().get(&Provider::Gemini)
            .map(|c| c.api_key.clone())
            .ok_or_else(|| "Gemini API key not set".to_string())?;
        Arc::new(GeminiClient::new(&api_key))
    }
};
```

**Step 2: Update the client.create_message call**

Change from using the local `client` variable to using the Arc'd client.

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): route messages to callback providers when selected"
```

---

## Task 13: Update client factory for TaskTool

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Update execute_task_tool client factory**

Find the `client_factory` closure in `execute_task_tool` and add support for Custom providers:

```rust
// Clone callback_providers for the closure
let callback_providers = self.callback_providers.clone();

let client_factory = move |_model: &str| -> Arc<dyn LlmClient> {
    match &provider_clone {
        Provider::Custom { name } => {
            callback_providers
                .read()
                .get(name)
                .cloned()
                .map(|c| c as Arc<dyn LlmClient>)
                .unwrap_or_else(|| {
                    // Fallback to Anthropic if callback provider not found
                    Arc::new(AnthropicClient::new(&api_key))
                })
        }
        Provider::Anthropic => Arc::new(AnthropicClient::new(&api_key)),
        Provider::OpenAI | Provider::Ollama => {
            let mut c = OpenAIClient::new(&api_key);
            if let Some(ref url) = base_url {
                c = c.with_base_url(url);
            }
            Arc::new(c)
        }
        Provider::Gemini => Arc::new(GeminiClient::new(&api_key)),
    }
};
```

**Step 2: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 3: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): support callback providers in TaskTool client factory"
```

---

## Task 14: Update spawn_agent and resume_agent

**Files:**
- Modify: `mux-ffi/src/engine.rs`

**Step 1: Update do_spawn_agent client creation**

Add the same provider switching logic to `do_spawn_agent`:

```rust
let client: Arc<dyn LlmClient> = match &provider {
    Provider::Custom { name } => {
        self.callback_providers
            .read()
            .get(name)
            .cloned()
            .map(|c| c as Arc<dyn LlmClient>)
            .ok_or_else(|| format!("Callback provider '{}' not registered", name))?
    }
    // ... other providers
};
```

**Step 2: Update do_resume_agent similarly**

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/engine.rs
git commit -m "feat(mux-ffi): support callback providers in spawn/resume agent"
```

---

## Task 15: Add ChatRole if missing

**Files:**
- Modify: `mux-ffi/src/types.rs`

**Step 1: Check if ChatRole exists, add if not**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ChatRole {
    User,
    Assistant,
}
```

**Step 2: Update ChatMessage if needed**

Ensure ChatMessage uses ChatRole:

```rust
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}
```

**Step 3: Verify it compiles**

Run: `cargo build -p mux-ffi`

**Step 4: Commit**

```bash
git add mux-ffi/src/types.rs
git commit -m "feat(mux-ffi): ensure ChatRole and ChatMessage types exist"
```

---

## Task 16: Write unit test for CallbackLlmClient

**Files:**
- Modify: `mux-ffi/src/callback_client.rs` or `mux-ffi/src/lib.rs`

**Step 1: Add test module**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct EchoProvider;

    impl LlmProvider for EchoProvider {
        fn generate(&self, request: LlmRequest, callback: Box<dyn LlmStreamCallback>) {
            // Echo back the last message
            let response = request.messages.last()
                .map(|m| format!("Echo: {}", m.content))
                .unwrap_or_else(|| "No message".to_string());

            callback.on_text(response);
            callback.on_complete(LlmUsage {
                input_tokens: 10,
                output_tokens: 5,
            });
        }
    }

    #[tokio::test]
    async fn test_callback_client_echo() {
        let client = CallbackLlmClient::new(Box::new(EchoProvider));
        let request = Request::new("test-model")
            .message(mux::llm::Message::user("Hello"));

        let response = client.create_message(&request).await.unwrap();

        assert_eq!(response.text(), "Echo: Hello");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }
}
```

**Step 2: Run the test**

Run: `cargo test -p mux-ffi test_callback_client`

**Step 3: Commit**

```bash
git add mux-ffi/src/callback_client.rs
git commit -m "test(mux-ffi): add unit test for CallbackLlmClient"
```

---

## Task 17: Run full test suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `cargo test --workspace`

**Step 2: Fix any failures**

Address any compilation or test failures.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix(mux-ffi): address test failures from callback provider implementation"
```

---

## Task 18: Reply to BBS thread

**Files:** None

**Step 1: Post implementation complete message**

Use `mcp__bbs__post_message` to reply to the feature request thread with:
- Implementation complete
- How to use it (Swift example)
- Version number

**Step 2: Bump version and tag**

- Update version in `Cargo.toml` files
- Create git tag

---

## Summary

18 tasks implementing:
1. FFI types (LlmUsage, LlmRequest, FfiToolDefinition)
2. Callback traits (LlmStreamCallback, LlmProvider)
3. Provider enum extension
4. CallbackLlmClient adapter
5. Engine integration (storage, registration, routing)
6. Tests
7. Documentation update (BBS)
