# LLM Backends Expansion Design

## Goal

Add support for Gemini, OpenRouter, and Ollama backends to mux-rs.

## Approach

**Parameterized OpenAI client** for OpenAI-compatible APIs (OpenRouter, Ollama, Azure, vLLM, etc.) plus a standalone GeminiClient for Google's different API.

## Changes

### 1. OpenAIClient Enhancement

Add `base_url` field with builder pattern:

```rust
pub struct OpenAIClient {
    api_key: String,
    base_url: String,  // defaults to "https://api.openai.com/v1"
    http: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(api_key: impl Into<String>) -> Self;
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self;

    // Convenience constructors
    pub fn openrouter(api_key: impl Into<String>) -> Self;
    pub fn openrouter_from_env() -> Result<Self, LlmError>;  // OPENROUTER_API_KEY
    pub fn ollama() -> Self;
    pub fn ollama_at(host: &str) -> Self;
}
```

### 2. GeminiClient

New file `src/llm/gemini.rs` implementing `LlmClient` for Google's Gemini API.

**API mapping:**
| mux concept | Gemini equivalent |
|-------------|-------------------|
| `system` | `systemInstruction` field |
| `messages` | `contents[]` with `role: "user"/"model"` |
| `tools` | `tools[].functionDeclarations[]` |
| `tool_use` response | `functionCall` in response |
| `tool_result` | `functionResponse` content part |

**Endpoints:**
- Non-streaming: `POST /v1beta/models/{model}:generateContent`
- Streaming: `POST /v1beta/models/{model}:streamGenerateContent?alt=sse`

**Environment:** `GEMINI_API_KEY` (fallback: `GOOGLE_API_KEY`)

### 3. Module Structure

```
src/llm/
├── mod.rs          # exports all clients
├── client.rs       # LlmClient trait (unchanged)
├── types.rs        # Request, Response, etc. (unchanged)
├── anthropic.rs    # AnthropicClient (unchanged)
├── openai.rs       # OpenAIClient (enhanced)
└── gemini.rs       # NEW: GeminiClient
```

## Environment Variables

| Client | Env var |
|--------|---------|
| `AnthropicClient::from_env()` | `ANTHROPIC_API_KEY` |
| `OpenAIClient::from_env()` | `OPENAI_API_KEY` |
| `OpenAIClient::openrouter_from_env()` | `OPENROUTER_API_KEY` |
| `GeminiClient::from_env()` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `OpenAIClient::ollama()` | None required |
