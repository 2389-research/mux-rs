// ABOUTME: Core types for LLM communication - messages, content blocks,
// ABOUTME: tool definitions, requests, and responses.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Role of a message sender.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Kind of media payload in a `ContentBlock::Media`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaKind {
    Image,
    Document,
    Audio,
    Video,
}

impl std::fmt::Display for MediaKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaKind::Image => write!(f, "image"),
            MediaKind::Document => write!(f, "document"),
            MediaKind::Audio => write!(f, "audio"),
            MediaKind::Video => write!(f, "video"),
        }
    }
}

/// Source of media bytes. Base64 is inline; Url is remote; Path is local file.
///
/// Externally tagged (e.g. `{"base64": "..."}` rather than `{"kind":"base64","data":"..."}`)
/// because serde does not support internally-tagged newtype variants over primitives.
/// Providers use their own wire types, so this only affects on-disk JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MediaSource {
    Base64(String),
    Url(String),
    Path(PathBuf),
}

/// A block of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    Media {
        kind: MediaKind,
        source: MediaSource,
        mime_type: String,
    },
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool result content block.
    pub fn tool_result(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error tool result content block.
    pub fn tool_error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: error.into(),
            is_error: true,
        }
    }

    /// Generic media constructor with base64-encoded data.
    pub fn media_base64(kind: MediaKind, mime: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Media {
            kind,
            source: MediaSource::Base64(data.into()),
            mime_type: mime.into(),
        }
    }

    /// Generic media constructor from a remote URL. Mime inferred by provider if empty.
    pub fn media_url(kind: MediaKind, mime: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Media {
            kind,
            source: MediaSource::Url(url.into()),
            mime_type: mime.into(),
        }
    }

    /// Generic media constructor from a local file. Mime inferred by provider if empty.
    pub fn media_path(kind: MediaKind, mime: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::Media {
            kind,
            source: MediaSource::Path(path.into()),
            mime_type: mime.into(),
        }
    }

    /// Create an image content block from base64-encoded bytes.
    pub fn image_base64(mime: impl Into<String>, data: impl Into<String>) -> Self {
        Self::media_base64(MediaKind::Image, mime, data)
    }
    /// Create an image content block from a URL. Mime type is inferred by the provider from the URL extension at serialize time.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Media {
            kind: MediaKind::Image,
            source: MediaSource::Url(url.into()),
            mime_type: String::new(),
        }
    }
    /// Create an image content block from a local file path. Mime type is inferred from the file extension when the provider resolves the path.
    pub fn image_path(path: impl Into<PathBuf>) -> Self {
        Self::Media {
            kind: MediaKind::Image,
            source: MediaSource::Path(path.into()),
            mime_type: String::new(),
        }
    }
    /// Create a document (e.g. PDF) content block from base64-encoded bytes.
    pub fn document_base64(mime: impl Into<String>, data: impl Into<String>) -> Self {
        Self::media_base64(MediaKind::Document, mime, data)
    }
    /// Create an audio content block from base64-encoded bytes.
    pub fn audio_base64(mime: impl Into<String>, data: impl Into<String>) -> Self {
        Self::media_base64(MediaKind::Audio, mime, data)
    }
    /// Create a video content block from base64-encoded bytes.
    pub fn video_base64(mime: impl Into<String>, data: impl Into<String>) -> Self {
        Self::media_base64(MediaKind::Video, mime, data)
    }
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create a user message with tool results.
    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: results,
        }
    }

    /// Create a user message with pre-built content blocks (text, media, etc.).
    pub fn user_with(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }
}

/// Definition of a tool for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Token usage statistics from a single API response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens read from cache (Anthropic prompt caching).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_read_tokens: u32,
    /// Tokens written to cache (Anthropic prompt caching).
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub cache_write_tokens: u32,
}

/// Helper for skip_serializing_if on u32.
fn is_zero_u32(val: &u32) -> bool {
    *val == 0
}

/// Helper for skip_serializing_if on u64.
fn is_zero_u64(val: &u64) -> bool {
    *val == 0
}

/// Request to create a message.
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    pub system: Option<String>,
    pub temperature: Option<f64>,
}

impl Request {
    /// Create a new request with the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Add a message to the request.
    pub fn message(mut self, message: Message) -> Self {
        self.messages.push(message);
        self
    }

    /// Add messages to the request.
    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.messages.extend(messages);
        self
    }

    /// Add a tool definition.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add tool definitions.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Set the system prompt.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Set max tokens.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
}

/// Response from creating a message.
#[derive(Debug, Clone)]
pub struct Response {
    pub id: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,
    pub usage: Usage,
}

impl Response {
    /// Check if the response contains tool use blocks.
    pub fn has_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
    }

    /// Extract all tool use blocks from the response.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect()
    }

    /// Extract concatenated text content from the response.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

use std::fmt;
use std::sync::{Arc, Mutex};

/// Thread-safe token usage accumulator for tracking usage across multiple API calls.
///
/// This type wraps an inner struct with `Arc<Mutex<>>` to allow safe sharing
/// across async tasks.
#[derive(Clone)]
pub struct TokenUsage {
    inner: Arc<Mutex<TokenUsageInner>>,
}

/// Internal storage for token usage data.
#[derive(Debug, Default)]
struct TokenUsageInner {
    input_tokens: u64,
    output_tokens: u64,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    request_count: u64,
}

impl TokenUsage {
    /// Create a new token usage accumulator.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TokenUsageInner::default())),
        }
    }

    /// Add usage from an API response.
    ///
    /// Increments the request count and adds all token counts including cache tokens.
    pub fn add(&self, usage: &Usage) {
        let mut inner = self.inner.lock().unwrap();
        inner.input_tokens += usage.input_tokens as u64;
        inner.output_tokens += usage.output_tokens as u64;
        inner.cache_read_tokens += usage.cache_read_tokens as u64;
        inner.cache_write_tokens += usage.cache_write_tokens as u64;
        inner.request_count += 1;
    }

    /// Add usage with explicit cache token counts.
    ///
    /// Use this when cache tokens are reported separately from the Usage struct.
    pub fn add_with_cache(&self, usage: &Usage, cache_read: u32, cache_write: u32) {
        let mut inner = self.inner.lock().unwrap();
        inner.input_tokens += usage.input_tokens as u64;
        inner.output_tokens += usage.output_tokens as u64;
        inner.cache_read_tokens += cache_read as u64;
        inner.cache_write_tokens += cache_write as u64;
        inner.request_count += 1;
    }

    /// Get total tokens used (input + output).
    pub fn total(&self) -> u64 {
        let inner = self.inner.lock().unwrap();
        inner.input_tokens + inner.output_tokens
    }

    /// Get a snapshot of current usage statistics.
    ///
    /// Returns a copy that can be safely used without holding the lock.
    pub fn snapshot(&self) -> TokenUsageSnapshot {
        let inner = self.inner.lock().unwrap();
        TokenUsageSnapshot {
            input_tokens: inner.input_tokens,
            output_tokens: inner.output_tokens,
            cache_read_tokens: inner.cache_read_tokens,
            cache_write_tokens: inner.cache_write_tokens,
            request_count: inner.request_count,
        }
    }

    /// Reset all usage statistics to zero.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.input_tokens = 0;
        inner.output_tokens = 0;
        inner.cache_read_tokens = 0;
        inner.cache_write_tokens = 0;
        inner.request_count = 0;
    }
}

impl Default for TokenUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TokenUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("TokenUsage")
            .field("input_tokens", &inner.input_tokens)
            .field("output_tokens", &inner.output_tokens)
            .field("cache_read_tokens", &inner.cache_read_tokens)
            .field("cache_write_tokens", &inner.cache_write_tokens)
            .field("request_count", &inner.request_count)
            .finish()
    }
}

/// A snapshot of token usage statistics.
///
/// This is a plain data struct that can be safely copied and serialized.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsageSnapshot {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub cache_read_tokens: u64,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub cache_write_tokens: u64,
    pub request_count: u64,
}

impl TokenUsageSnapshot {
    /// Get total tokens (input + output).
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl fmt::Display for TokenUsageSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} input + {} output = {} total ({} requests)",
            self.input_tokens,
            self.output_tokens,
            self.total(),
            self.request_count
        )
    }
}
