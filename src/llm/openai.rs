// ABOUTME: OpenAI API client implementation.
// ABOUTME: Implements LlmClient trait for GPT models.

use super::client::StreamEvent;
use super::media::resolve_request_media;
use super::{
    ContentBlock, MediaKind, MediaSource, Message, Request, Response, Role, StopReason,
    ToolDefinition, Usage,
};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// OpenAI API request format.
#[derive(Debug, Serialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    /// Used by older models (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Used by newer reasoning models (o1, o3, gpt-5, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

/// OpenAI message format.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// OpenAI message content: either a string (legacy) or an array of content parts (multimodal).
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAIContent {
    String(String),
    Parts(Vec<OpenAIContentPart>),
}

/// Content part for a multimodal message.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAIImageUrl },
    File { file: OpenAIFile },
    InputAudio { input_audio: OpenAIInputAudio },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIImageUrl {
    pub url: String,
}

/// OpenAI file content for the Chat Completions API.
/// `file_data` is a data URL of the form `data:{mime};base64,{data}`.
/// `filename` is an optional hint for the document name.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIFile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    pub file_data: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIInputAudio {
    pub data: String,
    pub format: String,
}

/// OpenAI tool call in a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAIFunctionCall,
}

/// OpenAI function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunctionCall {
    pub name: String,
    pub arguments: String,
}

/// OpenAI tool definition.
#[derive(Debug, Serialize)]
pub struct OpenAITool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIFunction,
}

/// OpenAI function definition.
#[derive(Debug, Serialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// OpenAI API response format.
#[derive(Debug, Deserialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIChoice>,
    pub usage: Option<OpenAIUsage>,
}

/// OpenAI response choice.
#[derive(Debug, Deserialize)]
pub struct OpenAIChoice {
    pub index: usize,
    pub message: OpenAIResponseMessage,
    pub finish_reason: Option<String>,
}

/// OpenAI response message.
#[derive(Debug, Deserialize)]
pub struct OpenAIResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
}

/// OpenAI usage stats.
#[derive(Debug, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI API error response.
#[derive(Debug, Deserialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}

/// OpenAI streaming chunk.
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAIStreamChoice>,
}

/// OpenAI streaming choice.
#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChoice {
    pub index: usize,
    pub delta: OpenAIDelta,
    pub finish_reason: Option<String>,
}

/// OpenAI streaming delta.
#[derive(Debug, Deserialize)]
pub struct OpenAIDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

/// OpenAI streaming tool call delta.
#[derive(Debug, Deserialize)]
pub struct OpenAIToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<OpenAIFunctionDelta>,
}

/// OpenAI streaming function delta.
#[derive(Debug, Deserialize)]
pub struct OpenAIFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Client for OpenAI and OpenAI-compatible APIs (OpenRouter, Ollama, etc.).
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl OpenAIClient {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_DEFAULT_BASE_URL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Create a new OpenAI client from the OPENAI_API_KEY environment variable.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "OPENAI_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::new(api_key))
    }

    /// Override the base URL for OpenAI-compatible APIs.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Create an OpenRouter client with the given API key.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(api_key).with_base_url("https://openrouter.ai/api/v1")
    }

    /// Create an OpenRouter client from the OPENROUTER_API_KEY environment variable.
    pub fn openrouter_from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "OPENROUTER_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::openrouter(api_key))
    }

    /// Create an Ollama client connecting to localhost:11434.
    pub fn ollama() -> Self {
        Self::new("ollama").with_base_url("http://localhost:11434/v1")
    }

    /// Create an Ollama client connecting to a custom host.
    pub fn ollama_at(host: impl Into<String>) -> Self {
        Self::new("ollama").with_base_url(format!("{}/v1", host.into()))
    }
}

impl From<&ToolDefinition> for OpenAITool {
    fn from(tool: &ToolDefinition) -> Self {
        OpenAITool {
            tool_type: "function".to_string(),
            function: OpenAIFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        }
    }
}

fn try_openai_messages(messages: &[Message]) -> Result<Vec<OpenAIMessage>, LlmError> {
    let mut result = Vec::new();
    for msg in messages {
        // Split out tool-result blocks — each becomes its own `role: "tool"` message
        let tool_results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => Some((tool_use_id.clone(), content.clone())),
                _ => None,
            })
            .collect();

        if !tool_results.is_empty() {
            if msg.content.len() != tool_results.len() {
                return Err(LlmError::Configuration(
                    "ToolResult blocks cannot be mixed with text, media, or tool calls in the same message".into(),
                ));
            }
            for (tool_use_id, content) in tool_results {
                result.push(OpenAIMessage {
                    role: "tool".to_string(),
                    content: Some(OpenAIContent::String(content)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                });
            }
            continue;
        }

        // For non-tool-result messages: build content (string or parts) + tool_calls
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .to_string();

        let has_media = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Media { .. }));

        let tool_calls: Vec<OpenAIToolCall> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(OpenAIToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                }),
                _ => None,
            })
            .collect();

        let content = if has_media {
            // Build parts array
            let mut parts: Vec<OpenAIContentPart> = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        parts.push(OpenAIContentPart::Text { text: text.clone() });
                    }
                    ContentBlock::Media {
                        kind,
                        source,
                        mime_type,
                    } => {
                        parts.push(try_media_part(*kind, source, mime_type)?);
                    }
                    // ToolUse is handled by the tool_calls field above; skip here
                    ContentBlock::ToolUse { .. } => {}
                    ContentBlock::ToolResult { .. } => {
                        // Shouldn't be reachable here (tool results are split above)
                        return Err(LlmError::Configuration(
                            "ToolResult mixed with non-tool-result content in same message".into(),
                        ));
                    }
                }
            }
            Some(OpenAIContent::Parts(parts))
        } else {
            // Collect text only
            let text: String = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                None
            } else {
                Some(OpenAIContent::String(text))
            }
        };

        result.push(OpenAIMessage {
            role,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        });
    }
    Ok(result)
}

fn try_media_part(
    kind: MediaKind,
    source: &MediaSource,
    mime_type: &str,
) -> Result<OpenAIContentPart, LlmError> {
    match kind {
        MediaKind::Image => {
            let url = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "image_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    format!("data:{};base64,{}", mime_type, data)
                }
                MediaSource::Url(url) => url.clone(),
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved before serialization".into(),
                    ));
                }
            };
            Ok(OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl { url },
            })
        }
        MediaKind::Document => {
            // Chat Completions expects the file_data as a data URL
            // ("data:{mime};base64,{data}") — not plain base64.
            let data_url = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "document_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    format!("data:{};base64,{}", mime_type, data)
                }
                MediaSource::Url(_) => {
                    // OpenAI's file input does NOT accept URLs; only file_data (base64) or file_id.
                    // file_id belongs to the Files API — out of scope for this task.
                    return Err(LlmError::Configuration(
                        "openai file input requires base64 data; URL sources are not supported"
                            .into(),
                    ));
                }
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved before serialization".into(),
                    ));
                }
            };
            Ok(OpenAIContentPart::File {
                file: OpenAIFile {
                    filename: None,
                    file_data: data_url,
                },
            })
        }
        MediaKind::Audio => {
            let data = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "audio_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    data.clone()
                }
                MediaSource::Url(_) | MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "openai input_audio requires base64 data (URL/Path must be resolved)"
                            .into(),
                    ));
                }
            };
            let format = audio_format_from_mime(mime_type)?;
            Ok(OpenAIContentPart::InputAudio {
                input_audio: OpenAIInputAudio { data, format },
            })
        }
        MediaKind::Video => Err(LlmError::UnsupportedMedia {
            provider: "openai",
            kind: MediaKind::Video,
        }),
    }
}

fn audio_format_from_mime(mime: &str) -> Result<String, LlmError> {
    let format = match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        _ => {
            return Err(LlmError::Configuration(format!(
                "unsupported input_audio mime_type: '{}' (only audio/wav, audio/x-wav, audio/mpeg, audio/mp3 accepted)",
                mime
            )));
        }
    };
    Ok(format.to_string())
}

/// Check if a model requires max_completion_tokens instead of max_tokens.
fn uses_max_completion_tokens(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    model_lower.starts_with("o1")
        || model_lower.starts_with("o3")
        || model_lower.starts_with("gpt-5")
}

pub fn try_into_openai_request(req: &Request) -> Result<OpenAIRequest, LlmError> {
    let mut messages: Vec<OpenAIMessage> = Vec::new();

    if let Some(system) = req.effective_system() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(OpenAIContent::String(system)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.extend(try_openai_messages(&req.messages)?);

    let (max_tokens, max_completion_tokens) = if uses_max_completion_tokens(&req.model) {
        (None, req.max_tokens)
    } else {
        (req.max_tokens, None)
    };

    Ok(OpenAIRequest {
        model: req.model.clone(),
        messages,
        max_tokens,
        max_completion_tokens,
        temperature: req.temperature,
        tools: req.tools.iter().map(OpenAITool::from).collect(),
        stream: None,
    })
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

impl From<OpenAIResponse> for Response {
    fn from(resp: OpenAIResponse) -> Self {
        let choice = resp.choices.into_iter().next().unwrap_or(OpenAIChoice {
            index: 0,
            message: OpenAIResponseMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        });

        let mut content = Vec::new();

        // Add text content if present
        if let Some(text) = choice.message.content {
            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        // Add tool calls if present
        if let Some(tool_calls) = choice.message.tool_calls {
            for call in tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&call.function.arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse {
                    id: call.id,
                    name: call.function.name,
                    input,
                });
            }
        }

        let usage = resp.usage.unwrap_or(OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

        Response {
            id: resp.id,
            content,
            stop_reason: parse_stop_reason(choice.finish_reason.as_deref()),
            model: resp.model,
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                ..Default::default()
            },
        }
    }
}

/// Parse an SSE line into an OpenAI stream chunk.
/// Used internally by OpenAI-compatible clients.
pub fn parse_sse_line(line: &str) -> Option<OpenAIStreamChunk> {
    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str(data).ok()
}

#[async_trait]
impl super::client::LlmClient for OpenAIClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let resolved = resolve_request_media(req, &self.http).await?;
        let openai_req = try_into_openai_request(&resolved)?;
        let url = format!("{}/chat/completions", self.base_url);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error: OpenAIError = response.json().await?;
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: error.error.message,
            });
        }

        let openai_resp: OpenAIResponse = response.json().await?;
        Ok(Response::from(openai_resp))
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
            let mut openai_req = try_into_openai_request(&resolved)?;
            openai_req.stream = Some(true);

            let url = format!("{}/chat/completions", base_url);
            let response = http
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&openai_req)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await?;
                let error: OpenAIError = serde_json::from_str(&error_text)?;
                Err(LlmError::Api {
                    status: status.as_u16(),
                    message: error.error.message,
                })?;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut message_started = false;
            let mut text_block_index: Option<usize> = None;
            let mut next_block_index = 0usize;
            // Track tool calls: (id, name, args, block_index, block_started)
            let mut current_tool_calls: Vec<(String, String, String, usize, bool)> = Vec::new();

            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }

                    if let Some(chunk) = parse_sse_line(&line) {
                        if !message_started {
                            yield StreamEvent::MessageStart {
                                id: chunk.id.clone(),
                                model: chunk.model.clone(),
                                usage: crate::llm::Usage::default(),
                            };
                            message_started = true;
                        }

                        for choice in chunk.choices {
                            // Handle text content
                            if let Some(text) = choice.delta.content {
                                // Emit ContentBlockStart for text on first text delta
                                if text_block_index.is_none() {
                                    let idx = next_block_index;
                                    next_block_index += 1;
                                    yield StreamEvent::ContentBlockStart {
                                        index: idx,
                                        block: ContentBlock::Text { text: String::new() },
                                    };
                                    text_block_index = Some(idx);
                                }
                                yield StreamEvent::ContentBlockDelta {
                                    index: text_block_index.unwrap(),
                                    text,
                                };
                            }

                            // Handle tool calls
                            if let Some(tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let tc_idx = tc.index;

                                    // Ensure we have space for this tool call
                                    while current_tool_calls.len() <= tc_idx {
                                        current_tool_calls.push((String::new(), String::new(), String::new(), 0, false));
                                    }

                                    // Accumulate tool call data
                                    if let Some(id) = tc.id {
                                        current_tool_calls[tc_idx].0 = id;
                                    }
                                    if let Some(func) = tc.function {
                                        if let Some(name) = func.name {
                                            current_tool_calls[tc_idx].1 = name;
                                        }

                                        // Emit ContentBlockStart when we have id and name (before JSON deltas)
                                        let (ref id, ref name, _, ref mut block_idx, ref mut started) = current_tool_calls[tc_idx];
                                        if !*started && !id.is_empty() && !name.is_empty() {
                                            *block_idx = next_block_index;
                                            next_block_index += 1;
                                            yield StreamEvent::ContentBlockStart {
                                                index: *block_idx,
                                                block: ContentBlock::ToolUse {
                                                    id: id.clone(),
                                                    name: name.clone(),
                                                    input: serde_json::Value::Object(serde_json::Map::new()),
                                                },
                                            };
                                            *started = true;
                                        }

                                        if let Some(args) = func.arguments {
                                            current_tool_calls[tc_idx].2.push_str(&args);
                                            // Yield as input JSON delta for tool argument accumulation
                                            yield StreamEvent::InputJsonDelta {
                                                index: current_tool_calls[tc_idx].3,
                                                partial_json: args,
                                            };
                                        }
                                    }
                                }
                            }

                            // Handle finish reason
                            if let Some(reason) = choice.finish_reason {
                                // Close text block if started
                                if let Some(idx) = text_block_index {
                                    yield StreamEvent::ContentBlockStop { index: idx };
                                }

                                // Close tool call blocks
                                for (id, _, _, block_idx, started) in current_tool_calls.iter() {
                                    if *started && !id.is_empty() {
                                        yield StreamEvent::ContentBlockStop { index: *block_idx };
                                    }
                                }

                                yield StreamEvent::MessageDelta {
                                    stop_reason: Some(parse_stop_reason(Some(&reason))),
                                    usage: Usage::default(),
                                };
                                yield StreamEvent::MessageStop;
                            }
                        }
                    }
                }
            }
        })
    }

    fn supports_media(&self, kind: super::MediaKind) -> bool {
        use super::MediaKind;
        matches!(
            kind,
            MediaKind::Image | MediaKind::Document | MediaKind::Audio
        )
    }
}

#[cfg(test)]
mod openai_test {
    use super::*;

    #[test]
    fn test_client_from_env_missing() {
        // SAFETY: This test runs in isolation and only affects this process
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
        }
        let result = OpenAIClient::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_request_serialization() {
        let req = Request::new("gpt-4o")
            .message(Message::user("Hello"))
            .system("Be helpful")
            .max_tokens(100);

        let openai_req = try_into_openai_request(&req).unwrap();
        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.messages.len(), 2); // system + user
        assert_eq!(openai_req.messages[0].role, "system");
        assert_eq!(openai_req.messages[1].role, "user");
    }

    #[test]
    fn test_tool_definition_conversion() {
        let tool = ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the weather".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }),
            cache_control: None,
        };

        let openai_tool = OpenAITool::from(&tool);
        assert_eq!(openai_tool.tool_type, "function");
        assert_eq!(openai_tool.function.name, "get_weather");
    }

    #[test]
    fn test_openai_image_base64_becomes_data_url() {
        let req = Request::new("gpt-4o").message(Message::user_with(vec![
            ContentBlock::text("what?"),
            ContentBlock::image_base64("image/png", "aGVsbG8="),
        ]));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        let content = &json["messages"][0]["content"];
        assert!(
            content.is_array(),
            "content should be parts array when media is present"
        );
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what?");
        assert_eq!(content[1]["type"], "image_url");
        let url = content[1]["image_url"]["url"].as_str().unwrap();
        assert!(
            url.starts_with("data:image/png;base64,"),
            "got url: {}",
            url
        );
        assert!(url.ends_with("aGVsbG8="));
    }

    #[test]
    fn test_openai_image_url_passthrough() {
        let req =
            Request::new("gpt-4o").message(Message::user_with(vec![ContentBlock::image_url(
                "https://example.com/cat.png",
            )]));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        assert_eq!(
            json["messages"][0]["content"][0]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[test]
    fn test_openai_document_as_file() {
        let req = Request::new("gpt-4o").message(Message::user_with(vec![
            ContentBlock::document_base64("application/pdf", "JVBERi0="),
        ]));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        let part = &json["messages"][0]["content"][0];
        assert_eq!(part["type"], "file");
        let file_data = part["file"]["file_data"]
            .as_str()
            .expect("file_data string");
        assert!(file_data.starts_with("data:application/pdf;base64,"));
        assert!(file_data.ends_with("JVBERi0="));
    }

    #[test]
    fn test_openai_audio_with_format_inference() {
        let req = Request::new("gpt-4o-audio-preview").message(Message::user_with(vec![
            ContentBlock::audio_base64("audio/wav", "UklGR"),
        ]));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        let part = &json["messages"][0]["content"][0];
        assert_eq!(part["type"], "input_audio");
        assert_eq!(part["input_audio"]["data"], "UklGR");
        assert_eq!(part["input_audio"]["format"], "wav");
    }

    #[test]
    fn test_openai_video_errors() {
        let req =
            Request::new("gpt-4o").message(Message::user_with(vec![ContentBlock::video_base64(
                "video/mp4",
                "AAAAG",
            )]));
        let result = try_into_openai_request(&req);
        assert!(matches!(
            result,
            Err(crate::error::LlmError::UnsupportedMedia {
                kind: MediaKind::Video,
                ..
            })
        ));
    }

    #[test]
    fn test_openai_text_only_keeps_string_content() {
        // Ensure we don't regress backwards compat for pure-text messages.
        let req = Request::new("gpt-4o").message(Message::user("hello"));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        assert!(
            json["messages"][0]["content"].is_string(),
            "expected string content when no media present; got {:?}",
            json["messages"][0]["content"]
        );
        assert_eq!(json["messages"][0]["content"], "hello");
    }

    #[test]
    fn test_openai_tool_result_message_preserved() {
        // Tool result messages use the legacy {"role":"tool","content":"...","tool_call_id":"..."} shape.
        // Ensure that path still works after the refactor.
        let req =
            Request::new("gpt-4o").message(Message::tool_results(vec![ContentBlock::tool_result(
                "tu_1",
                "Hello, Alice!",
            )]));
        let oa = try_into_openai_request(&req).unwrap();
        let json = serde_json::to_value(&oa).unwrap();
        assert_eq!(json["messages"][0]["role"], "tool");
        assert_eq!(json["messages"][0]["tool_call_id"], "tu_1");
        assert_eq!(json["messages"][0]["content"], "Hello, Alice!");
    }

    #[test]
    fn test_openai_mixed_tool_result_and_text_errors() {
        let req = Request::new("gpt-4o").message(Message {
            role: Role::User,
            content: vec![
                ContentBlock::text("accompanying note"),
                ContentBlock::tool_result("tu_1", "result"),
            ],
        });
        let result = try_into_openai_request(&req);
        assert!(matches!(
            result,
            Err(crate::error::LlmError::Configuration(_))
        ));
    }

    #[test]
    fn test_openai_audio_unknown_mime_errors() {
        let req = Request::new("gpt-4o-audio-preview").message(Message::user_with(vec![
            ContentBlock::audio_base64("audio/ogg", "xxx"),
        ]));
        let result = try_into_openai_request(&req);
        assert!(matches!(
            result,
            Err(crate::error::LlmError::Configuration(_))
        ));
    }

    #[test]
    fn test_openai_supports_media() {
        use super::super::client::LlmClient;
        let c = OpenAIClient::new("fake");
        assert!(c.supports_media(MediaKind::Image));
        assert!(c.supports_media(MediaKind::Document));
        assert!(c.supports_media(MediaKind::Audio));
        assert!(!c.supports_media(MediaKind::Video));
    }
}
