// ABOUTME: Anthropic Claude API client implementation.
// ABOUTME: Implements LlmClient trait for Claude models.

use super::client::StreamEvent;
use super::media::resolve_request_media;
use super::{CacheControl, ContentBlock, Message, Request, Response, StopReason, ToolDefinition, Usage};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// System prompt field. Either a plain string (legacy) or an array of typed
/// content blocks with optional cache_control markers (Anthropic prompt
/// caching). Serializes via the `untagged` enum so the JSON shape on the
/// wire is exactly what the Anthropic API expects.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    String(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

/// One typed system-prompt block. `block_type` is always `"text"` for now.
#[derive(Debug, Serialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Anthropic API request format.
#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// Anthropic message format.
#[derive(Debug, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContent>,
}

/// Anthropic image/document source block. Either base64 or URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

/// Anthropic content block.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContent {
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
    Image {
        source: AnthropicSource,
    },
    Document {
        source: AnthropicSource,
    },
}

/// Anthropic tool definition.
#[derive(Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Anthropic API response format.
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    pub content: Vec<AnthropicContent>,
    pub stop_reason: String,
    pub model: String,
    pub usage: AnthropicUsage,
}

/// Anthropic usage stats.
#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens read from cache (prompt caching feature).
    #[serde(default)]
    pub cache_read_input_tokens: Option<u32>,
    /// Tokens written to cache (prompt caching feature).
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
}

/// Anthropic API error response.
#[derive(Debug, Deserialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub error: AnthropicErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

/// Server-sent event from Anthropic streaming API.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicMessageStart,
    },
    ContentBlockStart {
        index: usize,
        content_block: AnthropicContent,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: AnthropicMessageDeltaData,
        usage: AnthropicUsage,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicErrorDetail,
    },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    pub model: String,
    /// Initial usage stats. Anthropic includes input_tokens +
    /// cache_creation_input_tokens + cache_read_input_tokens here, but
    /// only output_tokens: 0 in message_start. The final output_tokens
    /// shows up in message_delta. Optional because not all stream events
    /// from all providers carry it.
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageDeltaData {
    pub stop_reason: Option<String>,
}

/// Client for the Anthropic Claude API.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: ANTHROPIC_DEFAULT_BASE_URL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Override the base URL for Anthropic-compatible APIs.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Create a new Anthropic client from the ANTHROPIC_API_KEY environment variable.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "ANTHROPIC_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::new(api_key))
    }
}

fn try_anthropic_content(block: &ContentBlock) -> Result<AnthropicContent, LlmError> {
    use crate::llm::{MediaKind, MediaSource};
    match block {
        ContentBlock::Text { text } => Ok(AnthropicContent::Text { text: text.clone() }),
        ContentBlock::ToolUse { id, name, input } => Ok(AnthropicContent::ToolUse {
            id: id.clone(),
            name: name.clone(),
            input: input.clone(),
        }),
        ContentBlock::ToolResult {
            tool_use_id,
            content,
            is_error,
        } => Ok(AnthropicContent::ToolResult {
            tool_use_id: tool_use_id.clone(),
            content: content.clone(),
            is_error: *is_error,
        }),
        ContentBlock::Media {
            kind,
            source,
            mime_type,
        } => {
            // Anthropic does not accept URL sources for documents. Images are fine.
            if matches!(kind, MediaKind::Document) && matches!(source, MediaSource::Url(_)) {
                return Err(LlmError::Configuration(
                    "anthropic does not accept URL sources for documents (only base64, text, content, or file)"
                        .into(),
                ));
            }
            if matches!(source, MediaSource::Base64(_)) && mime_type.is_empty() {
                return Err(LlmError::Configuration(format!(
                    "anthropic {} base64 media requires a non-empty mime_type",
                    kind
                )));
            }
            let ant_source = match source {
                MediaSource::Base64(data) => AnthropicSource::Base64 {
                    media_type: mime_type.clone(),
                    data: data.clone(),
                },
                MediaSource::Url(url) => AnthropicSource::Url { url: url.clone() },
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved before serialization".into(),
                    ));
                }
            };
            match kind {
                MediaKind::Image => Ok(AnthropicContent::Image { source: ant_source }),
                MediaKind::Document => Ok(AnthropicContent::Document { source: ant_source }),
                MediaKind::Audio => Err(LlmError::UnsupportedMedia {
                    provider: "anthropic",
                    kind: MediaKind::Audio,
                }),
                MediaKind::Video => Err(LlmError::UnsupportedMedia {
                    provider: "anthropic",
                    kind: MediaKind::Video,
                }),
            }
        }
    }
}

impl From<AnthropicContent> for ContentBlock {
    fn from(content: AnthropicContent) -> Self {
        use crate::llm::{MediaKind, MediaSource};
        match content {
            AnthropicContent::Text { text } => ContentBlock::Text { text },
            AnthropicContent::ToolUse { id, name, input } => {
                ContentBlock::ToolUse { id, name, input }
            }
            AnthropicContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            },
            AnthropicContent::Image { source } => {
                let (mime_type, src) = match source {
                    AnthropicSource::Base64 { media_type, data } => {
                        (media_type, MediaSource::Base64(data))
                    }
                    AnthropicSource::Url { url } => (String::new(), MediaSource::Url(url)),
                };
                ContentBlock::Media {
                    kind: MediaKind::Image,
                    source: src,
                    mime_type,
                }
            }
            AnthropicContent::Document { source } => {
                let (mime_type, src) = match source {
                    AnthropicSource::Base64 { media_type, data } => {
                        (media_type, MediaSource::Base64(data))
                    }
                    AnthropicSource::Url { url } => (String::new(), MediaSource::Url(url)),
                };
                ContentBlock::Media {
                    kind: MediaKind::Document,
                    source: src,
                    mime_type,
                }
            }
        }
    }
}

pub(super) fn try_anthropic_message(msg: &Message) -> Result<AnthropicMessage, LlmError> {
    let content: Vec<AnthropicContent> = msg
        .content
        .iter()
        .map(try_anthropic_content)
        .collect::<Result<_, _>>()?;
    Ok(AnthropicMessage {
        role: match msg.role {
            super::Role::User => "user".to_string(),
            super::Role::Assistant => "assistant".to_string(),
        },
        content,
    })
}

impl From<&ToolDefinition> for AnthropicTool {
    fn from(tool: &ToolDefinition) -> Self {
        AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
            cache_control: tool.cache_control.clone(),
        }
    }
}

/// Resolve a [`Request`]'s system prompt into the Anthropic wire shape.
/// When `system_blocks` is non-empty it takes precedence and serializes as an
/// array of typed blocks (preserving cache_control). Otherwise the plain
/// `system` string is used. Returns None when both are empty.
fn build_anthropic_system(req: &Request) -> Option<AnthropicSystem> {
    if !req.system_blocks.is_empty() {
        let blocks = req
            .system_blocks
            .iter()
            .map(|b| AnthropicSystemBlock {
                block_type: "text".to_string(),
                text: b.text.clone(),
                cache_control: b.cache_control.clone(),
            })
            .collect();
        Some(AnthropicSystem::Blocks(blocks))
    } else {
        req.system.clone().map(AnthropicSystem::String)
    }
}

pub fn try_into_anthropic_request(req: &Request) -> Result<AnthropicRequest, LlmError> {
    let messages: Vec<AnthropicMessage> = req
        .messages
        .iter()
        .map(try_anthropic_message)
        .collect::<Result<_, _>>()?;
    Ok(AnthropicRequest {
        model: req.model.clone(),
        messages,
        max_tokens: req.max_tokens.unwrap_or(4096),
        system: build_anthropic_system(req),
        temperature: req.temperature,
        tools: req.tools.iter().map(AnthropicTool::from).collect(),
        stream: None,
    })
}

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

impl From<AnthropicResponse> for Response {
    fn from(resp: AnthropicResponse) -> Self {
        Response {
            id: resp.id,
            content: resp.content.into_iter().map(ContentBlock::from).collect(),
            stop_reason: parse_stop_reason(&resp.stop_reason),
            model: resp.model,
            usage: Usage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                cache_read_tokens: resp.usage.cache_read_input_tokens.unwrap_or(0),
                cache_write_tokens: resp.usage.cache_creation_input_tokens.unwrap_or(0),
            },
        }
    }
}

fn parse_sse_event(event_str: &str) -> Option<StreamEvent> {
    let mut data = None;

    for line in event_str.lines() {
        if let Some(rest) = line.strip_prefix("data: ") {
            data = Some(rest.to_string());
        }
    }

    let data = data?;
    let anthropic_event: AnthropicStreamEvent = serde_json::from_str(&data).ok()?;

    match anthropic_event {
        AnthropicStreamEvent::MessageStart { message } => {
            let usage = message
                .usage
                .as_ref()
                .map(|u| Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    cache_read_tokens: u.cache_read_input_tokens.unwrap_or(0),
                    cache_write_tokens: u.cache_creation_input_tokens.unwrap_or(0),
                })
                .unwrap_or_default();
            Some(StreamEvent::MessageStart {
                id: message.id,
                model: message.model,
                usage,
            })
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => Some(StreamEvent::ContentBlockStart {
            index,
            block: ContentBlock::from(content_block),
        }),
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
            AnthropicDelta::TextDelta { text } => {
                Some(StreamEvent::ContentBlockDelta { index, text })
            }
            AnthropicDelta::InputJsonDelta { partial_json } => Some(StreamEvent::InputJsonDelta {
                index,
                partial_json,
            }),
        },
        AnthropicStreamEvent::ContentBlockStop { index } => {
            Some(StreamEvent::ContentBlockStop { index })
        }
        AnthropicStreamEvent::MessageDelta { delta, usage } => Some(StreamEvent::MessageDelta {
            stop_reason: delta.stop_reason.map(|s| parse_stop_reason(&s)),
            usage: Usage {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                cache_read_tokens: usage.cache_read_input_tokens.unwrap_or(0),
                cache_write_tokens: usage.cache_creation_input_tokens.unwrap_or(0),
            },
        }),
        AnthropicStreamEvent::MessageStop => Some(StreamEvent::MessageStop),
        AnthropicStreamEvent::Ping => None,
        AnthropicStreamEvent::Error { error } => {
            eprintln!("Stream error: {}", error.message);
            None
        }
    }
}

#[async_trait]
impl super::client::LlmClient for AnthropicClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let resolved = resolve_request_media(req, &self.http).await?;
        let anthropic_req = try_into_anthropic_request(&resolved)?;

        let url = format!("{}/v1/messages", self.base_url);
        let response = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&anthropic_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error: AnthropicError = response.json().await?;
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: error.error.message,
            });
        }

        let anthropic_resp: AnthropicResponse = response.json().await?;
        Ok(Response::from(anthropic_resp))
    }

    fn create_message_stream(
        &self,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let http = self.http.clone();
        let req = req.clone();

        Box::pin(async_stream::try_stream! {
            let resolved = resolve_request_media(&req, &http).await?;
            let mut anthropic_req = try_into_anthropic_request(&resolved)?;
            anthropic_req.stream = Some(true);

            let url = format!("{}/v1/messages", base_url);
            let response = http
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("content-type", "application/json")
                .json(&anthropic_req)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await?;
                let error: AnthropicError = serde_json::from_str(&error_text)?;
                Err(LlmError::Api {
                    status: status.as_u16(),
                    message: error.error.message,
                })?;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete SSE events
                while let Some(pos) = buffer.find("\n\n") {
                    let event_str = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(event) = parse_sse_event(&event_str) {
                        yield event;
                    }
                }
            }
        })
    }

    fn supports_media(&self, kind: super::MediaKind) -> bool {
        use super::MediaKind;
        matches!(kind, MediaKind::Image | MediaKind::Document)
    }
}

// ---------------------------------------------------------------------------
// SSE parser tests — regression guards for cache_creation/read extraction
// from streaming responses. The non-streaming path of `create_message`
// already pulls these fields; before this branch the streaming path
// silently zeroed them out (everything `..Default::default()`), which made
// it look like caching was inactive on streaming agents.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod sse_parser_tests {
    use super::*;

    #[test]
    fn parse_sse_event_extracts_cache_tokens_from_message_start() {
        // Anthropic carries the cache_creation_input_tokens and
        // cache_read_input_tokens fields on message_start (with
        // output_tokens=0). Verify parse_sse_event surfaces them via
        // StreamEvent::MessageStart.usage.
        let raw = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_01ABC","model":"claude-sonnet-4-5","usage":{"input_tokens":1500,"output_tokens":0,"cache_creation_input_tokens":4747,"cache_read_input_tokens":2200}}}"#;

        let parsed = parse_sse_event(raw).expect("should parse message_start");
        match parsed {
            StreamEvent::MessageStart { id, model, usage } => {
                assert_eq!(id, "msg_01ABC");
                assert_eq!(model, "claude-sonnet-4-5");
                assert_eq!(usage.input_tokens, 1500);
                assert_eq!(usage.output_tokens, 0);
                assert_eq!(usage.cache_write_tokens, 4747, "cache_creation_input_tokens must map to cache_write_tokens");
                assert_eq!(usage.cache_read_tokens, 2200, "cache_read_input_tokens must map to cache_read_tokens");
            }
            other => panic!("expected MessageStart, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_event_extracts_cache_tokens_from_message_delta() {
        // The streaming bug was here: previous impl built Usage with
        // `..Default::default()` instead of pulling cache_read_input_tokens
        // and cache_creation_input_tokens. This regression-guards the fix.
        let raw = r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1500,"output_tokens":120,"cache_creation_input_tokens":3000,"cache_read_input_tokens":1100}}"#;

        let parsed = parse_sse_event(raw).expect("should parse message_delta");
        match parsed {
            StreamEvent::MessageDelta { usage, .. } => {
                assert_eq!(usage.output_tokens, 120);
                assert_eq!(usage.cache_write_tokens, 3000, "MessageDelta must surface cache_creation_input_tokens");
                assert_eq!(usage.cache_read_tokens, 1100, "MessageDelta must surface cache_read_input_tokens");
            }
            other => panic!("expected MessageDelta, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_event_message_start_without_usage_defaults_to_zero() {
        // Non-Anthropic providers (or older Anthropic format) may omit usage
        // on message_start. Verify we default to Usage::default() instead of
        // panicking on the missing field.
        let raw = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_x","model":"claude-x"}}"#;

        let parsed = parse_sse_event(raw).expect("should parse message_start without usage");
        match parsed {
            StreamEvent::MessageStart { usage, .. } => {
                assert_eq!(usage.input_tokens, 0);
                assert_eq!(usage.cache_write_tokens, 0);
                assert_eq!(usage.cache_read_tokens, 0);
            }
            other => panic!("expected MessageStart, got {:?}", other),
        }
    }

    #[test]
    fn parse_sse_event_message_delta_without_cache_fields_defaults_to_zero() {
        // Backward-compat: pre-caching responses don't include cache fields.
        // Should parse cleanly with cache_write/cache_read = 0.
        let raw = r#"event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":100,"output_tokens":50}}"#;

        let parsed = parse_sse_event(raw).expect("should parse message_delta sans cache fields");
        match parsed {
            StreamEvent::MessageDelta { usage, .. } => {
                assert_eq!(usage.output_tokens, 50);
                assert_eq!(usage.cache_write_tokens, 0);
                assert_eq!(usage.cache_read_tokens, 0);
            }
            other => panic!("expected MessageDelta, got {:?}", other),
        }
    }
}
