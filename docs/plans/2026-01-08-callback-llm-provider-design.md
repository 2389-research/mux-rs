# Callback-Based LLM Provider Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow Swift/Kotlin to provide LLM implementations via UniFFI callbacks, enabling on-device models (Apple Foundation Models) to use Mux's orchestration.

**Architecture:** Swift implements `LlmProvider` trait, Rust wraps it in `CallbackLlmClient` implementing `LlmClient`, engine routes requests to callback provider when selected.

**Tech Stack:** UniFFI callback_interface, tokio channels for async bridging, existing mux LlmClient trait

---

## Problem

Mux only supports cloud LLM providers via HTTP. Apple's Foundation Models framework provides an on-device 3B LLM, but it's Swift-native and can't be called from Rust directly.

Currently Hibi has two separate code paths:
- `MuxProvider` → MuxFFI → Cloud APIs
- `FoundationModelsProvider` → Apple's framework directly

Foundation Models misses out on Mux's orchestration: subagents, tool dispatch, conversation history.

## Solution

Add callback-based provider pattern (like `CustomTool` but for LLM generation).

## Interfaces

### LlmProvider (Swift implements)

```rust
#[uniffi::export(callback_interface)]
pub trait LlmProvider: Send + Sync {
    /// Generate a response. Call callback methods as tokens arrive.
    fn generate(&self, request: LlmRequest, callback: Box<dyn LlmStreamCallback>);
}
```

### LlmStreamCallback (Rust provides to Swift)

```rust
#[uniffi::export(callback_interface)]
pub trait LlmStreamCallback: Send + Sync {
    fn on_text(&self, delta: String);
    fn on_tool_call(&self, id: String, name: String, arguments: String);
    fn on_complete(&self, usage: LlmUsage);
    fn on_error(&self, error: String);
}
```

### Supporting Types

```rust
#[derive(uniffi::Record)]
pub struct LlmRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<FfiToolDefinition>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
}

#[derive(uniffi::Record)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}
```

## Rust Implementation

### CallbackLlmClient

New file `mux-ffi/src/callback_client.rs`:

```rust
pub struct CallbackLlmClient {
    provider: Arc<Box<dyn LlmProvider>>,
}

impl CallbackLlmClient {
    pub fn new(provider: Box<dyn LlmProvider>) -> Self {
        Self { provider: Arc::new(provider) }
    }
}

#[async_trait]
impl LlmClient for CallbackLlmClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        // Convert Request to LlmRequest
        // Create channel for results
        // Create StreamCallbackImpl that sends to channel
        // Call provider.generate()
        // Collect from channel, build Response
    }

    fn create_message_stream(&self, req: &Request) -> Pin<Box<dyn Stream<...>>> {
        // Similar but return stream instead of collecting
    }
}
```

### Provider Enum Extension

```rust
#[derive(Clone, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    Ollama,
    Custom { name: String },
}
```

### Engine Methods

```rust
impl MuxEngine {
    pub fn register_llm_provider(&self, name: String, provider: Box<dyn LlmProvider>);
    pub fn unregister_llm_provider(&self, name: String) -> Result<(), MuxFfiError>;
    pub fn list_llm_providers(&self) -> Vec<String>;
}
```

## Data Flow

```
1. Swift: engine.sendMessage(conversationId, content, callback)
2. Rust: MuxEngine.do_send_message()
3. Rust: Checks provider == Custom("foundation-models")
4. Rust: Gets CallbackLlmClient for that provider
5. Rust: client.create_message(request)
6. Rust: Converts to LlmRequest, creates channel
7. Rust: Calls Swift's provider.generate(request, streamCallback)
8. Swift: Foundation Models generates response
9. Swift: Calls streamCallback.onText() for each chunk
10. Swift: Calls streamCallback.onToolCall() if model wants tools
11. Swift: Calls streamCallback.onComplete(usage)
12. Rust: Collects events from channel into Response
13. Rust: Returns to agentic loop
14. Rust: Executes tools if needed
15. Rust: Loops back to step 5 if tool_use
16. Rust: Returns final result to Swift via ChatCallback
```

## Swift Usage Example

```swift
final class FoundationModelsProvider: LlmProvider {
    func generate(request: LlmRequest, callback: LlmStreamCallback) {
        Task {
            do {
                let session = LanguageModelSession()
                // Convert request.tools to Foundation Models format
                let tools = request.tools.map { convertTool($0) }
                session.tools = tools

                // Convert messages
                let messages = request.messages.map { convertMessage($0) }

                // Stream response
                let stream = session.streamResponse(to: messages)
                for try await response in stream {
                    if let text = response.text {
                        callback.onText(delta: text)
                    }
                    if let toolCall = response.toolCall {
                        callback.onToolCall(
                            id: toolCall.id,
                            name: toolCall.name,
                            arguments: toolCall.arguments
                        )
                    }
                }
                callback.onComplete(usage: LlmUsage(inputTokens: 0, outputTokens: 0))
            } catch {
                callback.onError(error: error.localizedDescription)
            }
        }
    }
}

// Registration
let provider = FoundationModelsProvider()
engine.registerLlmProvider(name: "foundation-models", provider: provider)
engine.setDefaultProvider(provider: .custom(name: "foundation-models"))
```

## Benefits

1. **Unified orchestration** - Mux handles subagents, context, tools for ALL providers
2. **On-device LLM** - Foundation Models runs locally with zero latency
3. **Same tool system** - HibiTools work identically across cloud and on-device
4. **Extensible** - Any Swift/Kotlin LLM SDK can be wrapped (MLX, llama.cpp, etc.)

## Testing Strategy

1. Unit tests for CallbackLlmClient with mock LlmProvider
2. Integration test with simple echo provider
3. Manual testing with Foundation Models on device
