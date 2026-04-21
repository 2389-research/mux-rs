# Multimodal Input Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multimodal input (images, documents, audio, video) to `mux`'s LLM abstraction across all providers, with a capability query for frontends and typed errors on unsupported kinds.

**Architecture:** Introduce a unified `ContentBlock::Media { kind, source, mime_type }` variant. Each provider serializes media into its native wire format via existing `From<&Request>` paths; unsupported (provider, kind) pairs raise `LlmError::UnsupportedMedia` at serialize time. A new `LlmClient::supports_media(kind)` method lets frontends query capability. FFI bindings mirror the new types for Swift callers.

**Tech Stack:** Rust 2024, `reqwest`, `async-trait`, `serde`, `thiserror`, UniFFI (for mux-ffi).

**Reference:** `docs/plans/2026-04-21-multimodal-input-design.md` for the approved design.

**Commit rhythm:** one commit per task. Every task ends with `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, and the relevant test command passing. Do not skip hooks.

---

## Phase 1 — Core types

### Task 1: Add MediaKind, MediaSource, and Media variant

**Files:**
- Modify: `src/llm/types.rs` (add variant to `ContentBlock`, add types, add constructors)
- Modify: `src/llm/types_test.rs` (add tests)

**Step 1: Write failing tests**

Append to `src/llm/types_test.rs`:

```rust
#[test]
fn test_media_image_base64_constructor() {
    let block = ContentBlock::image_base64("image/png", "aGVsbG8=");
    match block {
        ContentBlock::Media { kind, source, mime_type } => {
            assert_eq!(kind, MediaKind::Image);
            assert!(matches!(source, MediaSource::Base64(ref s) if s == "aGVsbG8="));
            assert_eq!(mime_type, "image/png");
        }
        _ => panic!("expected Media variant"),
    }
}

#[test]
fn test_media_serde_roundtrip() {
    let block = ContentBlock::Media {
        kind: MediaKind::Image,
        source: MediaSource::Url("https://example.com/a.png".to_string()),
        mime_type: "image/png".to_string(),
    };
    let json = serde_json::to_string(&block).unwrap();
    let back: ContentBlock = serde_json::from_str(&json).unwrap();
    match back {
        ContentBlock::Media { kind, mime_type, .. } => {
            assert_eq!(kind, MediaKind::Image);
            assert_eq!(mime_type, "image/png");
        }
        _ => panic!("expected Media variant"),
    }
}

#[test]
fn test_message_user_with_multiple_blocks() {
    let msg = Message::user_with(vec![
        ContentBlock::text("look at this"),
        ContentBlock::image_url("https://example.com/a.png"),
    ]);
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.content.len(), 2);
}

#[test]
fn test_document_audio_video_constructors() {
    let d = ContentBlock::document_base64("application/pdf", "JVBE");
    let a = ContentBlock::audio_base64("audio/wav", "UklGR");
    let v = ContentBlock::video_base64("video/mp4", "AAAAG");
    for (b, expected_kind) in [
        (d, MediaKind::Document),
        (a, MediaKind::Audio),
        (v, MediaKind::Video),
    ] {
        match b {
            ContentBlock::Media { kind, .. } => assert_eq!(kind, expected_kind),
            _ => panic!(),
        }
    }
}
```

**Step 2: Run tests — expect failures**

```
cargo test -p mux --lib llm::types_test
```

Expected: compile errors on `MediaKind`, `MediaSource`, `Message::user_with`, `ContentBlock::image_*`, etc.

**Step 3: Implement**

In `src/llm/types.rs`:

1. At top of file, add `use std::path::PathBuf;`
2. Add after `StopReason`:

```rust
/// Kind of media payload in a `ContentBlock::Media`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Document,
    Audio,
    Video,
}

/// Source of media bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MediaSource {
    Base64(String),
    Url(String),
    Path(PathBuf),
}
```

3. Add a fourth arm to `ContentBlock`:

```rust
Media {
    kind: MediaKind,
    source: MediaSource,
    mime_type: String,
},
```

4. Add constructors to the `impl ContentBlock` block:

```rust
pub fn media_base64(kind: MediaKind, mime: impl Into<String>, data: impl Into<String>) -> Self {
    Self::Media { kind, source: MediaSource::Base64(data.into()), mime_type: mime.into() }
}
pub fn media_url(kind: MediaKind, mime: impl Into<String>, url: impl Into<String>) -> Self {
    Self::Media { kind, source: MediaSource::Url(url.into()), mime_type: mime.into() }
}
pub fn media_path(kind: MediaKind, mime: impl Into<String>, path: impl Into<PathBuf>) -> Self {
    Self::Media { kind, source: MediaSource::Path(path.into()), mime_type: mime.into() }
}

pub fn image_base64(mime: impl Into<String>, data: impl Into<String>) -> Self { Self::media_base64(MediaKind::Image, mime, data) }
pub fn image_url(url: impl Into<String>) -> Self { Self::Media { kind: MediaKind::Image, source: MediaSource::Url(url.into()), mime_type: String::new() } }
pub fn image_path(path: impl Into<PathBuf>) -> Self { Self::Media { kind: MediaKind::Image, source: MediaSource::Path(path.into()), mime_type: String::new() } }

pub fn document_base64(mime: impl Into<String>, data: impl Into<String>) -> Self { Self::media_base64(MediaKind::Document, mime, data) }
pub fn audio_base64(mime: impl Into<String>, data: impl Into<String>) -> Self { Self::media_base64(MediaKind::Audio, mime, data) }
pub fn video_base64(mime: impl Into<String>, data: impl Into<String>) -> Self { Self::media_base64(MediaKind::Video, mime, data) }
```

> Note: `image_url`/`image_path` intentionally leave `mime_type` empty — the provider will infer from URL extension or file sniffing. `image_base64` requires explicit mime because it's opaque. Document this in the doc comment on each constructor.

5. Add to `impl Message`:

```rust
pub fn user_with(content: Vec<ContentBlock>) -> Self {
    Self { role: Role::User, content }
}
```

**Step 4: Run tests — expect pass**

```
cargo test -p mux --lib llm::types_test
```

Every test added in Step 1 passes.

**Step 5: Run the broader suite to catch non-exhaustive match warnings**

```
cargo build -p mux --all-targets 2>&1 | head -100
```

Expected: compile errors in every file that matches on `ContentBlock` without the new arm. These are the files you'll touch in Phase 3. Note them but don't fix yet — they get fixed per-provider in their own tasks. **For this task only**, silence them by adding a temporary `_ => unreachable!("Media handled in later task")` arm in files outside `src/llm/` is NOT permitted; instead add a temporary `ContentBlock::Media { .. } => {}` noop arm to every non-llm file that matches `ContentBlock`, and leave `src/llm/` provider files as-is (they stay broken until Phase 3). We'll fix the noop arms later.

Better plan: after Task 1, leave the compile errors in `src/llm/anthropic.rs`, `src/llm/openai.rs`, etc. as intentional red — they signal Phase 3 work. Commit only `src/llm/types.rs` and `src/llm/types_test.rs` and the crate won't build until Phase 3 completes.

Decision: **don't leave main broken.** Add minimal `ContentBlock::Media { .. } => unreachable!("filled in later task")` arms to every non-test match site in `src/` and `mux-ffi/src/`. Each Phase 3 task replaces its own provider's placeholder with real code. List the exact sites:

- `src/llm/anthropic.rs:187` (in `From<&ContentBlock> for AnthropicContent`)
- `src/llm/openai.rs:262` (in `OpenAIMessage::from`)
- `src/llm/openai.rs:280` (in text collection filter_map)
- `src/llm/gemini.rs:243` (in `convert_message_to_content`)
- `mux-ffi/src/callback_client.rs:38` (in content filter_map)
- `mux-ffi/src/engine/context_mgmt.rs:184`, `:287`, `:378`, `:482`, `:290`, `:189` (token estimation matches — add arm returning 0)
- `mux-ffi/src/engine/mod.rs:374` (if any ContentBlock match exists — grep to confirm)

For this task add the minimal arms only (no real logic), so the crate builds. Each provider task replaces the arm with its real implementation.

**Step 6: Confirm build + tests pass**

```
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --workspace
```

**Step 7: Commit**

```
git add -A
git commit -m "feat(llm): add ContentBlock::Media variant and MediaKind/MediaSource types"
```

---

### Task 2: Add `UnsupportedMedia` and `Io` variants to `LlmError`

**Files:**
- Modify: `src/error.rs`

**Step 1: Write failing test**

In `src/error.rs`, append:

```rust
#[cfg(test)]
mod error_test {
    use super::*;
    use crate::llm::MediaKind;

    #[test]
    fn test_unsupported_media_display() {
        let err = LlmError::UnsupportedMedia { provider: "anthropic", kind: MediaKind::Audio };
        assert_eq!(err.to_string(), "anthropic does not support audio media");
    }

    #[test]
    fn test_io_display() {
        let err = LlmError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
        assert!(err.to_string().contains("missing"));
    }
}
```

**Step 2: Run — expect failure**

```
cargo test -p mux --lib error_test
```

**Step 3: Implement**

Add two variants to the existing `LlmError` enum (line 22-37 of `src/error.rs`):

```rust
#[error("{provider} does not support {kind:?} media")]
UnsupportedMedia { provider: &'static str, kind: crate::llm::MediaKind },

#[error("IO error: {0}")]
Io(#[from] std::io::Error),
```

Adjust the `Debug` format for `kind` — use a lowercased manual format if `#[derive(Debug)]`'s capitalization is ugly. Quick fix: add `impl std::fmt::Display for MediaKind` in `src/llm/types.rs` returning `"image"`/`"document"`/`"audio"`/`"video"` and change error message to `"{provider} does not support {kind} media"`.

**Step 4: Run — expect pass**

```
cargo test -p mux --lib error_test
```

**Step 5: Commit**

```
git add src/error.rs src/llm/types.rs
git commit -m "feat(llm): add UnsupportedMedia and Io variants to LlmError"
```

---

### Task 3: Add `supports_media` to `LlmClient` trait

**Files:**
- Modify: `src/llm/client.rs`

**Step 1: Write failing test**

There's no test file for `client.rs` directly. Add inline tests in `src/llm/client.rs`:

```rust
#[cfg(test)]
mod client_test {
    use super::*;
    use crate::llm::MediaKind;

    struct DefaultClient;
    #[async_trait]
    impl LlmClient for DefaultClient {
        async fn create_message(&self, _: &Request) -> Result<Response, LlmError> { unimplemented!() }
        fn create_message_stream(&self, _: &Request) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> { unimplemented!() }
    }

    #[test]
    fn default_supports_media_is_false() {
        let c = DefaultClient;
        assert!(!c.supports_media(MediaKind::Image));
        assert!(!c.supports_media(MediaKind::Document));
        assert!(!c.supports_media(MediaKind::Audio));
        assert!(!c.supports_media(MediaKind::Video));
    }
}
```

**Step 2: Run — expect failure** (method not found)

```
cargo test -p mux --lib llm::client::client_test
```

**Step 3: Implement**

Add to `LlmClient` trait in `src/llm/client.rs`:

```rust
/// Does this client support the given media kind for input?
/// Default: false. Providers override with their supported set.
fn supports_media(&self, _kind: super::MediaKind) -> bool { false }
```

**Step 4: Run — expect pass**

```
cargo test -p mux --lib llm::client::client_test
```

**Step 5: Commit**

```
git add src/llm/client.rs
git commit -m "feat(llm): add supports_media to LlmClient trait"
```

---

## Phase 2 — Source resolution helper

### Task 4: `resolve_to_base64` helper

**Files:**
- Create: `src/llm/media.rs`
- Modify: `src/llm/mod.rs` (add `mod media; pub use media::*;`)

**Why a helper:** Every provider that needs base64 bytes from a `MediaSource` does the same thing (read file, fetch URL, or passthrough). Centralize.

**Step 1: Write tests**

Create `src/llm/media.rs`:

```rust
// ABOUTME: Helpers for resolving MediaSource variants to base64-encoded bytes
// ABOUTME: and inferring mime types from URL/path extensions.

use super::{MediaSource};
use crate::error::LlmError;
use base64::{engine::general_purpose::STANDARD, Engine as _};

/// Resolve a `MediaSource` to base64-encoded data plus an effective mime type.
///
/// - `Base64`: returned as-is (mime_type passed through).
/// - `Path`: bytes read from disk, encoded, mime inferred from extension if
///   `mime_hint` is empty.
/// - `Url`: fetched via the provided client, encoded, mime inferred from
///   Content-Type header or URL extension.
pub async fn resolve_to_base64(
    source: &MediaSource,
    mime_hint: &str,
    http: &reqwest::Client,
) -> Result<(String, String), LlmError> {
    match source {
        MediaSource::Base64(data) => Ok((data.clone(), mime_hint.to_string())),
        MediaSource::Path(p) => {
            let bytes = std::fs::read(p)?;
            let mime = if !mime_hint.is_empty() { mime_hint.to_string() } else { mime_from_path(p) };
            Ok((STANDARD.encode(bytes), mime))
        }
        MediaSource::Url(url) => {
            let resp = http.get(url).send().await?;
            let mime = if !mime_hint.is_empty() {
                mime_hint.to_string()
            } else {
                resp.headers().get(reqwest::header::CONTENT_TYPE)
                    .and_then(|h| h.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| mime_from_url(url))
            };
            let bytes = resp.bytes().await?;
            Ok((STANDARD.encode(bytes), mime))
        }
    }
}

fn mime_from_path(p: &std::path::Path) -> String {
    mime_from_ext(p.extension().and_then(|e| e.to_str()).unwrap_or(""))
}

fn mime_from_url(url: &str) -> String {
    let ext = url.rsplit('.').next().unwrap_or("").split('?').next().unwrap_or("");
    mime_from_ext(ext)
}

fn mime_from_ext(ext: &str) -> String {
    match ext.to_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        _ => "application/octet-stream",
    }.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_base64_passthrough() {
        let src = MediaSource::Base64("aGVsbG8=".to_string());
        let http = reqwest::Client::new();
        let (data, mime) = resolve_to_base64(&src, "image/png", &http).await.unwrap();
        assert_eq!(data, "aGVsbG8=");
        assert_eq!(mime, "image/png");
    }

    #[tokio::test]
    async fn test_path_reads_and_encodes() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        let src = MediaSource::Path(tmp.path().to_path_buf());
        let http = reqwest::Client::new();
        let (data, _) = resolve_to_base64(&src, "image/png", &http).await.unwrap();
        assert_eq!(data, "aGVsbG8=");
    }

    #[test]
    fn test_mime_inference() {
        assert_eq!(mime_from_ext("png"), "image/png");
        assert_eq!(mime_from_ext("pdf"), "application/pdf");
        assert_eq!(mime_from_ext("unknown"), "application/octet-stream");
    }
}
```

**Step 2: Add deps**

Check `Cargo.toml` — `base64` probably isn't in deps. Add to `[dependencies]`:

```
base64 = "0.22"
```

And `tempfile` to `[dev-dependencies]` if missing:

```
tempfile = "3"
```

Run `cargo build` to confirm.

**Step 3: Wire into mod**

Edit `src/llm/mod.rs`, add:

```rust
mod media;
pub use media::*;
```

**Step 4: Run tests — expect pass**

```
cargo test -p mux --lib llm::media
```

**Step 5: Commit**

```
git add src/llm/media.rs src/llm/mod.rs Cargo.toml Cargo.lock
git commit -m "feat(llm): add resolve_to_base64 helper for MediaSource"
```

---

## Phase 3 — Per-provider serialization

Providers are split into their own tasks because each has its own wire shape. Every task replaces the `ContentBlock::Media { .. } => unreachable!(...)` placeholder from Task 1.

### Task 5: Anthropic — Media serialization + `supports_media`

**Files:**
- Modify: `src/llm/anthropic.rs`
- Modify: `src/llm/anthropic_test.rs`

**Step 1: Write tests**

Append to `src/llm/anthropic_test.rs`:

```rust
#[test]
fn test_anthropic_image_base64_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(
        Message::user_with(vec![
            ContentBlock::text("what is this?"),
            ContentBlock::image_base64("image/png", "aGVsbG8="),
        ]),
    );
    let ar = AnthropicRequest::from(&req);
    let json = serde_json::to_value(&ar).unwrap();
    let content = &json["messages"][0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "base64");
    assert_eq!(content[1]["source"]["media_type"], "image/png");
    assert_eq!(content[1]["source"]["data"], "aGVsbG8=");
}

#[test]
fn test_anthropic_image_url_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(
        Message::user_with(vec![ContentBlock::image_url("https://example.com/a.png")]),
    );
    let ar = AnthropicRequest::from(&req);
    let json = serde_json::to_value(&ar).unwrap();
    let src = &json["messages"][0]["content"][0]["source"];
    assert_eq!(src["type"], "url");
    assert_eq!(src["url"], "https://example.com/a.png");
}

#[test]
fn test_anthropic_document_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(
        Message::user_with(vec![ContentBlock::document_base64("application/pdf", "JVBE")]),
    );
    let ar = AnthropicRequest::from(&req);
    let json = serde_json::to_value(&ar).unwrap();
    assert_eq!(json["messages"][0]["content"][0]["type"], "document");
}

#[test]
fn test_anthropic_audio_errors() {
    let req = Request::new("claude-sonnet-4-20250514").message(
        Message::user_with(vec![ContentBlock::audio_base64("audio/wav", "UklGR")]),
    );
    let result = try_into_anthropic_request(&req);
    assert!(matches!(result, Err(LlmError::UnsupportedMedia { kind: MediaKind::Audio, .. })));
}

#[test]
fn test_anthropic_supports_media() {
    let client = AnthropicClient::new("fake");
    assert!(client.supports_media(MediaKind::Image));
    assert!(client.supports_media(MediaKind::Document));
    assert!(!client.supports_media(MediaKind::Audio));
    assert!(!client.supports_media(MediaKind::Video));
}
```

**Step 2: Run — expect failure**

```
cargo test -p mux --lib llm::anthropic_test
```

**Step 3: Implement**

In `src/llm/anthropic.rs`:

1. **Add new wire types** after `AnthropicContent`:

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}
```

2. **Extend `AnthropicContent`** with:

```rust
Image { source: AnthropicSource },
Document { source: AnthropicSource },
```

3. **Replace the infallible `From<&ContentBlock> for AnthropicContent`** with a fallible `try_from` function. Anthropic doesn't support audio/video, so the conversion can fail. Add:

```rust
fn try_anthropic_content(block: &ContentBlock) -> Result<AnthropicContent, LlmError> {
    match block {
        ContentBlock::Text { text } => Ok(AnthropicContent::Text { text: text.clone() }),
        ContentBlock::ToolUse { id, name, input } => Ok(AnthropicContent::ToolUse {
            id: id.clone(), name: name.clone(), input: input.clone(),
        }),
        ContentBlock::ToolResult { tool_use_id, content, is_error } => Ok(AnthropicContent::ToolResult {
            tool_use_id: tool_use_id.clone(), content: content.clone(), is_error: *is_error,
        }),
        ContentBlock::Media { kind, source, mime_type } => {
            let ant_source = match source {
                MediaSource::Base64(data) => AnthropicSource::Base64 { media_type: mime_type.clone(), data: data.clone() },
                MediaSource::Url(url) => AnthropicSource::Url { url: url.clone() },
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved via resolve_to_base64 before serialization".into()
                    ));
                }
            };
            match kind {
                MediaKind::Image => Ok(AnthropicContent::Image { source: ant_source }),
                MediaKind::Document => Ok(AnthropicContent::Document { source: ant_source }),
                MediaKind::Audio => Err(LlmError::UnsupportedMedia { provider: "anthropic", kind: MediaKind::Audio }),
                MediaKind::Video => Err(LlmError::UnsupportedMedia { provider: "anthropic", kind: MediaKind::Video }),
            }
        }
    }
}
```

> Note: this does not resolve `Path` — that's done earlier, in the client before converting. We'll handle that in `create_message`/`create_message_stream`.

4. **Replace `From<&Request> for AnthropicRequest`** with `try_from`. Add:

```rust
pub fn try_into_anthropic_request(req: &Request) -> Result<AnthropicRequest, LlmError> {
    let messages = req.messages.iter()
        .map(|m| Ok::<_, LlmError>(AnthropicMessage {
            role: match m.role {
                super::Role::User => "user".to_string(),
                super::Role::Assistant => "assistant".to_string(),
            },
            content: m.content.iter().map(try_anthropic_content).collect::<Result<_, _>>()?,
        }))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AnthropicRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens.unwrap_or(4096),
        system: req.system.clone(),
        temperature: req.temperature,
        tools: req.tools.iter().map(AnthropicTool::from).collect(),
        stream: None,
    })
}
```

Delete the old `impl From<&Request> for AnthropicRequest` and `impl From<&ContentBlock>` — replaced.

5. **Update `create_message` and `create_message_stream`** to call `try_into_anthropic_request`, and before that, resolve any `MediaSource::Path` / `MediaSource::Url` in the request. Add a helper `async fn resolve_request_media(req: &Request, http: &reqwest::Client) -> Result<Request, LlmError>` that walks the request, resolves every `MediaSource::Path` to `Base64` via `resolve_to_base64`, and returns a new `Request`. URLs stay as URLs (Anthropic fetches them server-side).

6. **Implement `supports_media`** inside `impl LlmClient for AnthropicClient`:

```rust
fn supports_media(&self, kind: MediaKind) -> bool {
    matches!(kind, MediaKind::Image | MediaKind::Document)
}
```

**Step 4: Run tests — expect pass**

```
cargo test -p mux --lib llm::anthropic
```

**Step 5: Commit**

```
git add src/llm/anthropic.rs src/llm/anthropic_test.rs
git commit -m "feat(llm): Anthropic image/document media serialization"
```

---

### Task 6: OpenAI — Media serialization

**Files:**
- Modify: `src/llm/openai.rs`

**Key wire difference:** OpenAI's `content` field on a message can be either a string *or* an array of content parts. Parts have `{type: "text", text}`, `{type: "image_url", image_url: {url}}`, `{type: "input_file", file_data}` (base64 PDF), or `{type: "input_audio", input_audio: {data, format}}`.

**Step 1: Write tests**

Append to the existing `mod openai_test` in `src/llm/openai.rs`:

```rust
#[test]
fn test_openai_image_base64_becomes_data_url() {
    let req = Request::new("gpt-4o").message(
        Message::user_with(vec![
            ContentBlock::text("what?"),
            ContentBlock::image_base64("image/png", "aGVsbG8="),
        ]),
    );
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    let content = &json["messages"][0]["content"];
    assert!(content.is_array());
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(url.starts_with("data:image/png;base64,"));
}

#[test]
fn test_openai_video_errors() {
    let req = Request::new("gpt-4o").message(
        Message::user_with(vec![ContentBlock::video_base64("video/mp4", "AAAAG")]),
    );
    let result = try_into_openai_request(&req);
    assert!(matches!(result, Err(LlmError::UnsupportedMedia { kind: MediaKind::Video, .. })));
}

#[test]
fn test_openai_supports_media() {
    let c = OpenAIClient::new("fake");
    assert!(c.supports_media(MediaKind::Image));
    assert!(c.supports_media(MediaKind::Document));
    assert!(c.supports_media(MediaKind::Audio));
    assert!(!c.supports_media(MediaKind::Video));
}
```

**Step 2: Run — expect failure**

```
cargo test -p mux --lib llm::openai
```

**Step 3: Implement**

1. **Change `OpenAIMessage.content` type** from `Option<String>` to `Option<OpenAIContent>` where:

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    String(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
    InputFile { file_data: String },  // base64 PDF
    InputAudio { input_audio: OpenAIInputAudio },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIImageUrl { pub url: String }

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIInputAudio { pub data: String, pub format: String }
```

2. **Add fallible converter** `try_into_openai_request(req: &Request) -> Result<OpenAIRequest, LlmError>` that replaces the existing `From<&Request>`. Delete the old `From<&Message> for OpenAIMessage` and `convert_messages` — rebuild them as fallible.

3. **Media handling inside `try_into_openai_request`**:
   - `Image` + Base64 → `ImageUrl { url: "data:{mime};base64,{data}" }`
   - `Image` + Url → `ImageUrl { url }`
   - `Document` (PDF) + Base64 → `InputFile { file_data: base64 }`
   - `Audio` + Base64 → `InputAudio { data, format: mime.split('/').nth(1).unwrap_or("wav") }`
   - `Video` → `Err(UnsupportedMedia)`
   - `Path` → error from serialize (should've been resolved upstream)

4. **Update `create_message`/`create_message_stream`** to call `resolve_request_media` first (share the helper with `anthropic.rs` — move it to `src/llm/media.rs` and export).

5. **Add `supports_media`** returning `Image | Document | Audio`.

6. Update the existing `convert_messages` call sites — tool-result messages still use string content. Mixed-media assistant replies are OK too.

**Step 4: Run — expect pass**

```
cargo test -p mux --lib llm::openai
```

**Step 5: Commit**

```
git add src/llm/openai.rs src/llm/media.rs src/llm/mod.rs
git commit -m "feat(llm): OpenAI image/document/audio media serialization"
```

---

### Task 7: OpenRouter + Ollama — inherit OpenAI serialization, override `supports_media`

**Files:**
- Modify: `src/llm/openrouter.rs`
- Modify: `src/llm/ollama.rs`

**Step 1: Write tests**

Append to the test module in each file:

```rust
#[test]
fn test_openrouter_supports_media() {
    let c = OpenRouterClient::new("fake");
    assert!(c.supports_media(MediaKind::Image));
    assert!(c.supports_media(MediaKind::Document));
    assert!(c.supports_media(MediaKind::Audio));
    assert!(!c.supports_media(MediaKind::Video));
}
```

```rust
#[test]
fn test_ollama_supports_media() {
    let c = OllamaClient::new("llava");
    assert!(c.supports_media(MediaKind::Image));
    assert!(!c.supports_media(MediaKind::Document));
    assert!(!c.supports_media(MediaKind::Audio));
    assert!(!c.supports_media(MediaKind::Video));
}
```

**Step 2: Run — expect failure**

**Step 3: Implement**

In both files, the `LlmClient` impl already forwards the request to OpenAI-compatible logic. Because they use `OpenAIRequest::from`, update to call `try_into_openai_request` now that it's fallible. Override `supports_media`:

```rust
// openrouter.rs — same as OpenAI
fn supports_media(&self, kind: MediaKind) -> bool {
    matches!(kind, MediaKind::Image | MediaKind::Document | MediaKind::Audio)
}

// ollama.rs — images only
fn supports_media(&self, kind: MediaKind) -> bool {
    matches!(kind, MediaKind::Image)
}
```

For Ollama specifically: since it currently uses the OpenAI serialization but only supports images, we need to reject Document/Audio at serialize time before OpenAI's serializer tries to produce e.g. `InputFile`. Add a pre-check in Ollama's `create_message`/`_stream` that scans the request for non-image media and returns `UnsupportedMedia` before calling the shared serializer.

**Step 4: Run — expect pass**

**Step 5: Commit**

```
git add src/llm/openrouter.rs src/llm/ollama.rs
git commit -m "feat(llm): OpenRouter/Ollama capability queries and Ollama non-image rejection"
```

---

### Task 8: Gemini — Media serialization

**Files:**
- Modify: `src/llm/gemini.rs`

Gemini uses `inline_data: {mimeType, data}` for base64 and `file_data: {fileUri}` for URLs/file API.

**Step 1: Write tests**

Add inline `#[cfg(test)]` module if missing, or extend existing:

```rust
#[test]
fn test_gemini_image_base64_serialization() {
    let req = Request::new("gemini-1.5-flash").message(
        Message::user_with(vec![ContentBlock::image_base64("image/png", "aGVsbG8=")]),
    );
    let gr = try_into_gemini_request(&req).unwrap();
    let json = serde_json::to_value(&gr).unwrap();
    let part = &json["contents"][0]["parts"][0];
    assert_eq!(part["inlineData"]["mimeType"], "image/png");
    assert_eq!(part["inlineData"]["data"], "aGVsbG8=");
}

#[test]
fn test_gemini_supports_all_four() {
    let c = GeminiClient::new("fake");
    for k in [MediaKind::Image, MediaKind::Document, MediaKind::Audio, MediaKind::Video] {
        assert!(c.supports_media(k));
    }
}
```

**Step 2: Run — expect failure**

**Step 3: Implement**

1. Extend `GeminiPart`:

```rust
#[serde(skip_serializing_if = "Option::is_none")]
pub inline_data: Option<GeminiInlineData>,
#[serde(skip_serializing_if = "Option::is_none")]
pub file_data: Option<GeminiFileData>,
```

2. Add structs:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiInlineData { pub mime_type: String, pub data: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiFileData { pub mime_type: String, pub file_uri: String }
```

3. Replace `From<&Request> for GeminiRequest` with `try_into_gemini_request(req) -> Result<_, LlmError>`. For `Media` blocks: base64 → `InlineData`, url → `FileData`, path → error (should've been resolved).

4. Add `supports_media` returning all four kinds.

5. Update `create_message`/`_stream` to resolve paths and call fallible serializer.

**Step 4: Run — expect pass**

**Step 5: Commit**

```
git add src/llm/gemini.rs
git commit -m "feat(llm): Gemini media serialization for all four kinds"
```

---

## Phase 4 — Integration tests and fixtures

### Task 9: Add media fixtures

**Files:**
- Create: `tests/fixtures/tiny.png` (a 1×1 PNG)
- Create: `tests/fixtures/tiny.pdf` (a minimal PDF)
- Create: `tests/fixtures/tiny.wav` (200ms silent WAV)
- Create: `tests/fixtures/tiny.mp4` (tiny MP4)

**Step 1: Generate**

```
mkdir -p tests/fixtures
python3 -c "import base64; open('tests/fixtures/tiny.png','wb').write(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAeImBZsAAAAASUVORK5CYII='))"
```

For PDF/WAV/MP4, use small well-known fixture generators or bundle similar minimal files. Confirm each is under 4KB. Commit as binary.

**Step 2: Commit**

```
git add tests/fixtures
git commit -m "test: add tiny multimodal fixtures"
```

---

### Task 10: Provider integration tests (env-var gated)

**Files:**
- Create: `tests/integration_media.rs`

**Pattern:** mirrors `src/llm/anthropic_test.rs::test_client_from_env_missing`. Each test checks for an API key and skips if missing, so CI without keys stays green.

**Step 1: Write tests**

```rust
// ABOUTME: Integration tests for multimodal input across providers.
// ABOUTME: Each test is gated on the relevant API key env var.

use mux::llm::*;
use mux::prelude::*;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[tokio::test]
async fn anthropic_sees_image() {
    let Ok(key) = std::env::var("ANTHROPIC_API_KEY") else { eprintln!("skip: no ANTHROPIC_API_KEY"); return; };
    let client = AnthropicClient::new(key);
    let req = Request::new("claude-haiku-4-5-20251001")
        .message(Message::user_with(vec![
            ContentBlock::text("Describe this image in one word."),
            ContentBlock::image_path(fixture("tiny.png")),
        ]))
        .max_tokens(64);
    let resp = client.create_message(&req).await.expect("api call");
    assert!(!resp.text().is_empty());
}

// Repeat for: openai_sees_image, gemini_sees_image, openai_reads_pdf,
// gemini_reads_pdf, openai_hears_audio, gemini_watches_video.
// Gate each on the corresponding env var.
```

**Step 2: Run (manually if keys are set)**

```
cargo test --test integration_media -- --nocapture
```

Expected: pass when keys set, skip messages printed otherwise.

**Step 3: Commit**

```
git add tests/integration_media.rs
git commit -m "test: provider round-trip integration tests for media"
```

---

## Phase 5 — FFI bindings

### Task 11: Add FFI media types

**Files:**
- Create: `mux-ffi/src/media.rs`
- Modify: `mux-ffi/src/lib.rs` (add `mod media; pub use media::*;`)
- Modify: `mux-ffi/src/types.rs` — extend `ChatMessage` or add parallel type

**Step 1: Create `mux-ffi/src/media.rs`**

```rust
// ABOUTME: FFI-safe mirrors of mux::llm::Media types for UniFFI.
// ABOUTME: Converts to/from core mux types at the FFI boundary.

use mux::llm::{MediaKind, MediaSource};

#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiMediaKind { Image, Document, Audio, Video }

#[derive(Debug, Clone, uniffi::Enum)]
pub enum FfiMediaSource {
    Base64 { data: String },
    Url { url: String },
    Path { path: String },
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiMedia {
    pub kind: FfiMediaKind,
    pub source: FfiMediaSource,
    pub mime_type: String,
}

impl From<FfiMediaKind> for MediaKind {
    fn from(k: FfiMediaKind) -> Self {
        match k {
            FfiMediaKind::Image => MediaKind::Image,
            FfiMediaKind::Document => MediaKind::Document,
            FfiMediaKind::Audio => MediaKind::Audio,
            FfiMediaKind::Video => MediaKind::Video,
        }
    }
}

impl From<MediaKind> for FfiMediaKind { /* inverse */ }

impl From<FfiMediaSource> for MediaSource {
    fn from(s: FfiMediaSource) -> Self {
        match s {
            FfiMediaSource::Base64 { data } => MediaSource::Base64(data),
            FfiMediaSource::Url { url } => MediaSource::Url(url),
            FfiMediaSource::Path { path } => MediaSource::Path(path.into()),
        }
    }
}

impl From<MediaSource> for FfiMediaSource { /* inverse */ }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_kind_roundtrip() {
        for k in [FfiMediaKind::Image, FfiMediaKind::Document, FfiMediaKind::Audio, FfiMediaKind::Video] {
            let core: MediaKind = k.clone().into();
            let back: FfiMediaKind = core.into();
            assert!(matches!((k, back), (FfiMediaKind::Image, FfiMediaKind::Image) | (FfiMediaKind::Document, FfiMediaKind::Document) | (FfiMediaKind::Audio, FfiMediaKind::Audio) | (FfiMediaKind::Video, FfiMediaKind::Video)));
        }
    }
}
```

**Step 2: Run**

```
cargo test -p mux-ffi media
```

**Step 3: Commit**

```
git add mux-ffi/src/media.rs mux-ffi/src/lib.rs
git commit -m "feat(mux-ffi): FfiMedia/FfiMediaKind/FfiMediaSource types"
```

---

### Task 12: Extend `ChatMessage` with optional media attachments

**Files:**
- Modify: `mux-ffi/src/types.rs:338` — add `media: Vec<FfiMedia>` field
- Modify: `mux-ffi/src/callback_client.rs` — populate media in `convert_request`
- Modify: `mux-ffi/src/engine/messaging.rs` — pass media through when assembling requests

**Step 1: Write a test** in `mux-ffi/src/callback_client.rs` under the existing test mod:

```rust
#[tokio::test]
async fn test_callback_client_forwards_media() {
    // EchoProvider stub that records what it saw
    // ... use a new provider that captures the LlmRequest and asserts media.len() == 1
}
```

**Step 2: Run — expect failure**

**Step 3: Implement**

1. Add `pub media: Vec<FfiMedia>` to `ChatMessage`. Update all call sites (there are ~3 outside this file). Swift binding regeneration is automatic.
2. In `CallbackLlmClient::convert_request`, walk `ContentBlock::Media` blocks in each message and push corresponding `FfiMedia` entries.
3. In `mux-ffi/src/engine/messaging.rs` — wherever `ChatMessage` is constructed, plumb any attached media through (usually `Vec::new()` for the happy text-only path).

**Step 4: Run — expect pass**

**Step 5: Commit**

```
git add mux-ffi/src
git commit -m "feat(mux-ffi): thread media attachments through ChatMessage"
```

---

### Task 13: `CallbackLlmClient::supports_media` from Swift registration

**Files:**
- Modify: `mux-ffi/src/callback.rs` — add `supports_media` to `LlmProvider` trait
- Modify: `mux-ffi/src/callback_client.rs` — implement `LlmClient::supports_media` by querying the provider

**Step 1: Write test**

Add to existing test mod in `callback_client.rs`:

```rust
struct ImageOnlyProvider;
impl LlmProvider for ImageOnlyProvider {
    fn generate(&self, _: LlmRequest) -> LlmResponse { unimplemented!() }
    fn supports_media(&self, kind: FfiMediaKind) -> bool {
        matches!(kind, FfiMediaKind::Image)
    }
}

#[test]
fn test_callback_client_delegates_supports_media() {
    let c = CallbackLlmClient::new(Box::new(ImageOnlyProvider));
    assert!(c.supports_media(MediaKind::Image));
    assert!(!c.supports_media(MediaKind::Video));
}
```

**Step 2: Run — expect failure**

**Step 3: Implement**

1. Add to `LlmProvider` trait in `mux-ffi/src/callback.rs:179`:

```rust
fn supports_media(&self, kind: crate::media::FfiMediaKind) -> bool;
```

Note: breaking change to the Swift-side trait. Acceptable because the crate is pre-1.0.

2. In `CallbackLlmClient`:

```rust
fn supports_media(&self, kind: MediaKind) -> bool {
    self.provider.supports_media(kind.into())
}
```

**Step 4: Run — expect pass**

**Step 5: Commit**

```
git add mux-ffi/src
git commit -m "feat(mux-ffi): LlmProvider::supports_media bridged to LlmClient"
```

---

### Task 14: Replace Media placeholder arms with real handling

**Files:**
- Modify: `mux-ffi/src/engine/context_mgmt.rs` (4 match sites at lines ~184, ~287, ~378, ~482)
- Modify: any remaining `ContentBlock::Media { .. } => unreachable!()` sites from Task 1

**Step 1: Write test**

In a new `mux-ffi/src/engine/context_mgmt_test.rs` (or inline):

```rust
#[test]
fn test_media_token_estimate_nonzero() {
    // Construct a StoredMessage with an image Media block of N bytes of base64
    // Assert estimate > 0
}
```

**Step 2: Implement**

For each match site, replace the `ContentBlock::Media { .. } => 0` placeholder with a rough estimate:

```rust
ContentBlock::Media { kind, source, .. } => estimate_media_tokens(*kind, source),
```

Where `estimate_media_tokens` returns:
- Image: 1000 (flat baseline — matches Anthropic's docs on image token cost for small images)
- Document: `base64_len / 4` (rough byte count, ~4 chars per byte, 1 token per 4 bytes → len/16) — implement as `approx_bytes(source) / 4`
- Audio: `approx_bytes(source) / 1000` (audio tokens are much cheaper per byte)
- Video: `approx_bytes(source) / 500`

`approx_bytes(source)` returns the base64 length × 3/4 for `Base64`, 0 for `Url`/`Path` (unknown without fetching — conservative under-estimate is fine since these are upper-bound heuristics anyway).

**Step 3: Run tests**

```
cargo test -p mux-ffi
```

**Step 4: Commit**

```
git add mux-ffi/src/engine
git commit -m "feat(mux-ffi): token estimation for Media blocks"
```

---

## Phase 6 — Persistence and final polish

### Task 15: Persistence backwards-compat test

**Files:**
- Modify: `mux-ffi/src/engine/persistence.rs` — add test, confirm no schema version exists

**Step 1: Write test**

```rust
#[test]
fn test_load_legacy_messages_without_media() {
    // Construct a JSON blob with pre-media ContentBlock variants only
    // Confirm it deserializes to StoredMessage successfully
    // No schema version field exists; serde's untagged additive policy handles it
    let json = r#"[{"role":"user","content":[{"type":"text","text":"hi"}]}]"#;
    let msgs: Vec<StoredMessage> = serde_json::from_str(json).unwrap();
    assert_eq!(msgs.len(), 1);
}

#[test]
fn test_save_and_load_roundtrip_with_media() {
    let msg = StoredMessage {
        role: Role::User,
        content: vec![ContentBlock::image_base64("image/png", "aGVsbG8=")],
    };
    let json = serde_json::to_string(&vec![msg]).unwrap();
    let back: Vec<StoredMessage> = serde_json::from_str(&json).unwrap();
    assert_eq!(back.len(), 1);
}
```

**Step 2: Run**

```
cargo test -p mux-ffi engine::persistence
```

**Step 3: Commit**

```
git add mux-ffi/src/engine/persistence.rs
git commit -m "test(mux-ffi): persistence round-trip for Media blocks"
```

---

### Task 16: Final sweep

**Step 1: Full workspace check**

```
cargo fmt --all
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace
```

Fix any remaining warnings. Every Phase 3 provider should have already cleaned up its own placeholders; this is the catch-all.

**Step 2: Grep for stale placeholders**

```
grep -rn "unreachable!" src mux-ffi/src | grep -i "media\|later task"
```

Must return nothing.

**Step 3: Commit if anything changed**

```
git status
# if clean, skip
git add -A && git commit -m "chore: final polish for multimodal input"
```

---

## Scope Notes

**Explicitly not in this plan (tracked in design doc as deferred):**
- MCP image passthrough (`mux-ffi/src/engine/tool_wrappers.rs:17`)
- Provider file-ID sources (Gemini File API, OpenAI Files)
- Assistant output media
- Actual Swift-side binding updates for `agent-test-tui`, `code-agent`, or downstream iOS apps

**Risks to flag during execution:**
- If `GeminiClient::from_env` or similar already imports from a removed path after `try_into_*` refactors, the crate fails to build — fix forward, don't bypass.
- The `ChatMessage.media` field addition is a breaking change to Swift callers. Mention it in the commit message for Task 12.
- `reqwest::Client` must have a timeout on URL fetches or a huge/slow URL could hang. If the existing clients don't already configure a timeout, file a follow-up note in the commit message rather than fixing scope-creep in this plan.
