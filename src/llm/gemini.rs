// ABOUTME: Google Gemini API client implementation.
// ABOUTME: Implements LlmClient trait for Gemini models.

use super::client::StreamEvent;
use super::{ContentBlock, Message, Request, Response, Role, StopReason, ToolDefinition, Usage};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

const GEMINI_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Gemini API request format.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiRequest {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<GeminiTool>,
}

/// Gemini content (message).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<GeminiPart>,
}

/// Gemini content part.
/// Gemini uses a flat object with optional fields, not a tagged union.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<GeminiFunctionResponse>,
}

impl GeminiPart {
    /// Create a text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: Some(text.into()),
            function_call: None,
            function_response: None,
        }
    }

    /// Create a function call part.
    pub fn function_call(name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            text: None,
            function_call: Some(GeminiFunctionCall {
                name: name.into(),
                args,
            }),
            function_response: None,
        }
    }

    /// Create a function response part.
    pub fn function_response(name: impl Into<String>, response: serde_json::Value) -> Self {
        Self {
            text: None,
            function_call: None,
            function_response: Some(GeminiFunctionResponse {
                name: name.into(),
                response,
            }),
        }
    }
}

/// Gemini function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    pub args: serde_json::Value,
}

/// Gemini function response (tool result).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
}

/// Gemini generation config.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
}

/// Gemini tool definition.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiTool {
    pub function_declarations: Vec<GeminiFunctionDeclaration>,
}

/// Gemini function declaration.
#[derive(Debug, Serialize)]
pub struct GeminiFunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Gemini API response format.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    pub candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsageMetadata>,
}

/// Gemini response candidate.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiCandidate {
    pub content: GeminiContent,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Gemini usage metadata.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiUsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u32,
    #[serde(default)]
    pub candidates_token_count: u32,
    #[serde(default)]
    pub total_token_count: u32,
}

/// Gemini API error response.
#[derive(Debug, Deserialize)]
pub struct GeminiError {
    pub error: GeminiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct GeminiErrorDetail {
    pub code: i32,
    pub message: String,
    pub status: String,
}

/// Client for the Google Gemini API.
#[derive(Debug, Clone)]
pub struct GeminiClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl GeminiClient {
    /// Create a new Gemini client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: GEMINI_DEFAULT_BASE_URL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Create a new Gemini client from environment variable.
    /// Checks GEMINI_API_KEY first, then falls back to GOOGLE_API_KEY.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| LlmError::Api {
                status: 0,
                message: "GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set".to_string(),
            })?;
        Ok(Self::new(api_key))
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Build the endpoint URL for a given model and method.
    fn endpoint(&self, model: &str, method: &str) -> String {
        format!("{}/models/{}:{}", self.base_url, model, method)
    }
}

impl From<&ToolDefinition> for GeminiFunctionDeclaration {
    fn from(tool: &ToolDefinition) -> Self {
        GeminiFunctionDeclaration {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        }
    }
}

/// Build a lookup from tool_use_id to function name from message history.
fn build_tool_name_lookup(messages: &[Message]) -> std::collections::HashMap<String, String> {
    let mut lookup = std::collections::HashMap::new();
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, .. } = block {
                lookup.insert(id.clone(), name.clone());
            }
        }
    }
    lookup
}

fn convert_message_to_content(
    msg: &Message,
    tool_name_lookup: &std::collections::HashMap<String, String>,
) -> GeminiContent {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "model",
    };

    let parts: Vec<GeminiPart> = msg
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(GeminiPart::text(text)),
            ContentBlock::ToolUse { name, input, .. } => {
                Some(GeminiPart::function_call(name, input.clone()))
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                // Look up the function name from the tool_use_id.
                // Fall back to tool_use_id if not found (shouldn't happen in valid conversation).
                let name = tool_name_lookup
                    .get(tool_use_id)
                    .cloned()
                    .unwrap_or_else(|| tool_use_id.clone());
                Some(GeminiPart::function_response(
                    name,
                    serde_json::json!({ "result": content }),
                ))
            }
        })
        .collect();

    GeminiContent {
        role: Some(role.to_string()),
        parts,
    }
}

impl From<&Request> for GeminiRequest {
    fn from(req: &Request) -> Self {
        // Build lookup for tool_use_id -> function name mapping
        let tool_name_lookup = build_tool_name_lookup(&req.messages);

        let contents: Vec<GeminiContent> = req
            .messages
            .iter()
            .map(|msg| convert_message_to_content(msg, &tool_name_lookup))
            .collect();

        let system_instruction = req.system.as_ref().map(|s| GeminiContent {
            role: None,
            parts: vec![GeminiPart::text(s)],
        });

        let generation_config = if req.max_tokens.is_some() || req.temperature.is_some() {
            Some(GeminiGenerationConfig {
                max_output_tokens: req.max_tokens,
                temperature: req.temperature,
            })
        } else {
            None
        };

        let tools = if req.tools.is_empty() {
            Vec::new()
        } else {
            vec![GeminiTool {
                function_declarations: req.tools.iter().map(GeminiFunctionDeclaration::from).collect(),
            }]
        };

        GeminiRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        }
    }
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("STOP") => StopReason::EndTurn,
        Some("MAX_TOKENS") => StopReason::MaxTokens,
        Some("TOOL_CODE") | Some("FUNCTION_CALL") => StopReason::ToolUse,
        _ => StopReason::EndTurn,
    }
}

fn convert_gemini_response(resp: GeminiResponse, model: String) -> Result<Response, LlmError> {
    let candidate = resp.candidates.into_iter().next().ok_or_else(|| LlmError::Api {
        status: 0,
        message: "Gemini returned empty candidates (possibly blocked by safety filters)".to_string(),
    })?;

    let blocks: Vec<ContentBlock> = candidate
        .content
        .parts
        .into_iter()
        .filter_map(|part| {
            if let Some(text) = part.text {
                Some(ContentBlock::Text { text })
            } else if let Some(fc) = part.function_call {
                Some(ContentBlock::ToolUse {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: fc.name,
                    input: fc.args,
                })
            } else {
                // function_response is input, not output
                None
            }
        })
        .collect();

    let stop_reason = parse_stop_reason(candidate.finish_reason.as_deref());

    let usage = resp.usage_metadata.unwrap_or(GeminiUsageMetadata {
        prompt_token_count: 0,
        candidates_token_count: 0,
        total_token_count: 0,
    });

    Ok(Response {
        id: uuid::Uuid::new_v4().to_string(),
        content: blocks,
        stop_reason,
        model,
        usage: Usage {
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
            ..Default::default()
        },
    })
}

/// Parse an SSE line from Gemini streaming response.
fn parse_gemini_sse(line: &str) -> Option<GeminiResponse> {
    let data = line.strip_prefix("data: ")?;
    serde_json::from_str(data).ok()
}

#[async_trait]
impl super::client::LlmClient for GeminiClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let gemini_req = GeminiRequest::from(req);
        let url = format!("{}?key={}", self.endpoint(&req.model, "generateContent"), self.api_key);

        let response = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&gemini_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error: GeminiError = response.json().await?;
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: error.error.message,
            });
        }

        let gemini_resp: GeminiResponse = response.json().await?;
        convert_gemini_response(gemini_resp, req.model.clone())
    }

    fn create_message_stream(
        &self,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        let gemini_req = GeminiRequest::from(req);
        let url = format!(
            "{}?key={}&alt=sse",
            self.endpoint(&req.model, "streamGenerateContent"),
            self.api_key
        );
        let model = req.model.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let response = http
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&gemini_req)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await?;
                let error: GeminiError = serde_json::from_str(&error_text)?;
                Err(LlmError::Api {
                    status: status.as_u16(),
                    message: error.error.message,
                })?;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut message_started = false;
            let mut current_text_index: Option<usize> = None;
            let mut block_index = 0usize;

            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Some(gemini_resp) = parse_gemini_sse(&line) {
                        if !message_started {
                            yield StreamEvent::MessageStart {
                                id: uuid::Uuid::new_v4().to_string(),
                                model: model.clone(),
                            };
                            message_started = true;
                        }

                        for candidate in gemini_resp.candidates {
                            for part in candidate.content.parts {
                                // Handle text content
                                if let Some(text) = part.text {
                                    // Start text block if needed
                                    if current_text_index.is_none() {
                                        yield StreamEvent::ContentBlockStart {
                                            index: block_index,
                                            block: ContentBlock::Text { text: String::new() },
                                        };
                                        current_text_index = Some(block_index);
                                        block_index += 1;
                                    }
                                    yield StreamEvent::ContentBlockDelta {
                                        index: current_text_index.unwrap(),
                                        text,
                                    };
                                }

                                // Handle function calls
                                if let Some(fc) = part.function_call {
                                    // Close text block if open
                                    if let Some(idx) = current_text_index.take() {
                                        yield StreamEvent::ContentBlockStop { index: idx };
                                    }

                                    let tool_index = block_index;
                                    block_index += 1;

                                    yield StreamEvent::ContentBlockStart {
                                        index: tool_index,
                                        block: ContentBlock::ToolUse {
                                            id: uuid::Uuid::new_v4().to_string(),
                                            name: fc.name,
                                            input: serde_json::Value::Object(serde_json::Map::new()),
                                        },
                                    };

                                    // Emit args as JSON delta
                                    let args_json = serde_json::to_string(&fc.args).unwrap_or_default();
                                    yield StreamEvent::InputJsonDelta {
                                        index: tool_index,
                                        partial_json: args_json,
                                    };

                                    yield StreamEvent::ContentBlockStop { index: tool_index };
                                }

                                // function_response is input, not output - ignore
                            }

                            // Check for finish
                            if let Some(reason) = candidate.finish_reason {
                                // Close text block if open
                                if let Some(idx) = current_text_index.take() {
                                    yield StreamEvent::ContentBlockStop { index: idx };
                                }

                                let usage = gemini_resp.usage_metadata.as_ref().map(|u| Usage {
                                    input_tokens: u.prompt_token_count,
                                    output_tokens: u.candidates_token_count,
                                    ..Default::default()
                                }).unwrap_or_default();

                                yield StreamEvent::MessageDelta {
                                    stop_reason: Some(parse_stop_reason(Some(&reason))),
                                    usage,
                                };
                                yield StreamEvent::MessageStop;
                            }
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_from_env_missing() {
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        let result = GeminiClient::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_request_serialization() {
        let req = Request::new("gemini-2.0-flash")
            .message(Message::user("Hello"))
            .system("Be helpful")
            .max_tokens(100);

        let gemini_req = GeminiRequest::from(&req);
        assert_eq!(gemini_req.contents.len(), 1);
        assert!(gemini_req.system_instruction.is_some());
        assert!(gemini_req.generation_config.is_some());
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
        };

        let gemini_func = GeminiFunctionDeclaration::from(&tool);
        assert_eq!(gemini_func.name, "get_weather");
        assert_eq!(gemini_func.description, "Get the weather");
    }
}
