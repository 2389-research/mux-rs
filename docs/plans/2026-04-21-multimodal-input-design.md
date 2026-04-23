# Multimodal Input Support — Design

Date: 2026-04-21
Status: Approved, pending implementation plan

## Summary

Add multimodal input (images, documents, audio, video) to `mux`'s LLM abstraction. Callers compose messages with a new `ContentBlock::Media` variant; each provider client serializes that into its native wire format. Providers that don't support a media type raise a typed error at serialize time, and frontends can query capability via a new method on `LlmClient`.

MCP tool-result image passthrough is explicitly out of scope for this design and deferred to a follow-up.

**URL handling posture.** The library never fetches URLs. `MediaSource::Url` is passed to providers that accept URL sources natively (Anthropic for images, OpenAI `image_url` parts, OpenRouter via OpenAI compat). Providers that require inline bytes (Gemini for all kinds) reject `MediaSource::Url` pre-flight with `LlmError::UnsupportedSource`. Callers that want to use a URL with an inline-only provider must fetch the bytes themselves and pass `MediaSource::Base64`. This mirrors the Go side's posture and eliminates the SSRF class.

## Scope

**In scope:**
- Image, document, audio, and video input on user messages
- Source shapes: base64 bytes, URL, local file path
- Providers: Anthropic, OpenAI, Gemini, Ollama, OpenRouter (OpenAI-compatible), CallbackClient (Apple Foundation Models via mux-ffi)
- UniFFI bindings for Swift callers
- Persistence round-trip for the new variant
- Context-management token estimation for media blocks

**Out of scope (deferred):**
- MCP tool-result image passthrough (currently stringified to `"[image]"` in `mux-ffi/src/engine/tool_wrappers.rs:17`) — tracked separately
- Provider file-ID sources (Gemini File API, OpenAI Files) — additive, can be a new `MediaSource` variant later
- Model-level capability detection (e.g., "does this specific Ollama model support vision") — server-side concern
- Output/generation of media from assistant responses — this design is input-only
- URL auto-fetch for inline-only providers — callers fetch; the library never issues outbound HTTP on behalf of a `MediaSource::Url`

## Core Types (src/llm/types.rs)

One new variant on `ContentBlock`, three supporting types:

```rust
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
    Media {
        kind: MediaKind,
        source: MediaSource,
        mime_type: String,
    },
}

pub enum MediaKind { Image, Document, Audio, Video }

pub enum MediaSource {
    Base64(String),
    Url(String),
    Path(PathBuf),
}
```

Constructors:
- `ContentBlock::image_base64(mime, data)`, `::image_url(url)`, `::image_path(p)`
- Equivalents for `document_*`, `audio_*`, `video_*`
- `Message::user_with(Vec<ContentBlock>)` for multi-block user messages

`MediaSource::Path` is resolved by `resolve_request_media` inside each provider's `create_message` / `create_message_stream`, before the fallible `try_into_<provider>_request(&resolved)` helpers are invoked — not in the constructor. Path-read errors surface as `LlmError::Io`; files exceeding `MAX_MEDIA_BYTES` (20MB) surface as `LlmError::MediaTooLarge`. `MediaSource::Url` is never fetched by the library: providers that accept URL sources natively pass them through, and providers that require inline bytes reject pre-flight with `LlmError::UnsupportedSource`.

## Provider Wire Format Mapping

| Provider | Image | Document | Audio | Video |
|---|---|---|---|---|
| Anthropic | `image` block, `source: {type: base64\|url, media_type, data}` | `document` block, same source shape | unsupported → error | unsupported → error |
| OpenAI | `image_url` content part (data URL or https URL) | `input_file` with base64 `file_data`; URL rejected pre-flight | `input_audio` with `{data, format}`; URL rejected pre-flight | unsupported → error |
| Gemini | `inline_data` only; URL rejected pre-flight | `inline_data` only; URL rejected pre-flight | `inline_data` only; URL rejected pre-flight | `inline_data` only; URL rejected pre-flight |
| Ollama | `image_url` (vision model dependent) | unsupported | unsupported | unsupported |
| OpenRouter | same as OpenAI | same | same | unsupported |

`MediaSource::Url` is passed through natively where the provider accepts URL sources; otherwise it's rejected pre-flight with `LlmError::UnsupportedSource` and the caller must fetch themselves. `MediaSource::Path` always resolves to base64 (bounded by `MAX_MEDIA_BYTES`). `MediaSource::Base64` passes straight through.

Ollama's vision support is model-dependent and cannot be detected client-side. `supports_media(Image)` returns `true` and the server rejects with a clear error when the loaded model isn't vision-capable.

## Error Handling & Capability Query

New error variants in `LlmError`:

```rust
LlmError::UnsupportedMedia {
    provider: &'static str,
    kind: MediaKind,
}

LlmError::UnsupportedSource {
    provider: &'static str,
    kind: MediaKind,
    source_kind: MediaSourceKind,  // Base64 | Url | Path
}

LlmError::MediaTooLarge {
    limit: usize,
    actual: u64,
}
```

`UnsupportedMedia` is raised at serialize time inside each provider's fallible `try_into_<provider>_request` helper when a `Media` block's kind isn't supported. `UnsupportedSource` is raised pre-flight (before any network call) when a provider accepts a given media kind but not the supplied source form (e.g. Gemini rejecting `MediaSource::Url`). `MediaTooLarge` surfaces when a local `MediaSource::Path` exceeds `MAX_MEDIA_BYTES`.

Capability query on the `LlmClient` trait:

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    // existing ...
    fn supports_media(&self, kind: MediaKind) -> bool { false }
}
```

Default is `false` (conservative). Per-provider overrides:
- `AnthropicClient`: `Image | Document`
- `OpenAIClient`: `Image | Document | Audio`
- `GeminiClient`: `Image | Document | Audio | Video`
- `OllamaClient`: `Image`
- `CallbackClient`: dynamic — declared by the Swift-side registration

Method rather than associated const so `CallbackClient` can query through its callback at runtime.

## UniFFI / mux-ffi Impact

New FFI types:

```rust
#[derive(uniffi::Enum)] enum FfiMediaKind { Image, Document, Audio, Video }
#[derive(uniffi::Enum)] enum FfiMediaSource {
    Base64 { data: String },
    Url { url: String },
    Path { path: String },  // String, not PathBuf — UniFFI limitation
}
#[derive(uniffi::Record)] struct FfiMedia {
    kind: FfiMediaKind,
    source: FfiMediaSource,
    mime_type: String,
}
```

Changes required in `mux-ffi/src/`:
- `engine/messaging.rs`, `engine/persistence.rs`, `callback_client.rs`: add `Media` arms to existing `ContentBlock` matches (~8 spots)
- `callback_client.rs`: add `supports_media: HashSet<FfiMediaKind>` to Swift-provided registration; plumb through `LlmClient::supports_media`
- `engine/context_mgmt.rs`: add `Media` arm to token-estimation matches — rough heuristic (e.g., image ≈ 1000 tokens baseline; audio/video/document scaled by byte size). Upper-bound accuracy is the goal, not precision.

Persistence: `engine/persistence.rs` serializes `ContentBlock` as JSON. Forward-compat (new code reads old data) is free. Backward-compat (old binaries reading new-format files) requires either an existing schema version bump or a migration check — to be confirmed during implementation.

## Testing Strategy

Follows the TDD rule in `CLAUDE.md`: no mocks, real data/APIs.

**Unit tests:**
- `src/llm/types_test.rs`: constructors, serde round-trip for `Media`, `MediaSource::Path` resolution against a real temp file
- Per-provider serialization tests: `try_into_anthropic_request(&req)` (and equivalents for other providers) with each supported `Media` kind produces the exact expected JSON shape. Uses small committed fixtures (`tests/fixtures/`: tiny PNG, PDF, WAV, MP4)
- Unsupported-media tests: assert `LlmError::UnsupportedMedia` for each (provider, unsupported-kind) pair
- Capability query tests: each client returns its documented set

**Integration tests** (real HTTP, env-var gated, matches the `ANTHROPIC_API_KEY` pattern in `src/llm/anthropic_test.rs`):
- One round-trip per (provider, supported-kind) sending a tiny fixture; assert model response mentions expected content ("describe this image" → keyword check)
- Skipped automatically without the relevant API key; CI without keys stays green

**FFI tests:**
- Round-trip a media block through the `mux-ffi` callback bridge; assert the FFI enum survives intact

**Persistence tests:**
- `Vec<ContentBlock>` with `Media` round-trips through JSON
- Load a pre-existing no-media conversation file; confirm backwards-compat

**Explicitly out of scope:**
- Mocked HTTP clients (project rule)
- Fuzzing
- Performance benchmarks

## Risks & Open Questions

- **Path resolution at serialize time** means a bad file path produces a per-request error, not a per-construction error. Callers that want eager validation can call a `resolve()` helper we expose on `MediaSource` (cheap, non-breaking addition).
- **URL sources with inline-only providers** (Gemini) require callers to fetch themselves. The library deliberately does not fetch URLs — this eliminates an SSRF class and mirrors the Go implementation. Callers needing URL → Gemini should use their platform's HTTP client (e.g. `URLSession` on Apple) and pass the bytes as `MediaSource::Base64`.
- **Token estimates for media** are approximate. Production callers should rely on actual returned usage, not pre-flight estimates, for billing or hard limits.
- **Persistence backward-compat** depends on whether a schema version currently exists in `engine/persistence.rs`. Resolved during implementation.

## Transition

Next step: invoke the writing-plans skill to produce a step-by-step implementation plan derived from this design.
