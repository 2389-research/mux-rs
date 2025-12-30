// ABOUTME: OpenAI API client implementation.
// ABOUTME: Implements LlmClient trait for GPT models.

use super::client::StreamEvent;
use super::{ContentBlock, Message, Request, Response, Role, StopReason, ToolDefinition, Usage};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI API request format.
#[derive(Debug, Serialize)]
pub struct OpenAIRequest {
    pub model: String,
    pub messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
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
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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

/// Client for the OpenAI API.
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    api_key: String,
    http: reqwest::Client,
}

impl OpenAIClient {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
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

impl From<&Message> for OpenAIMessage {
    fn from(msg: &Message) -> Self {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        // Check if this is a tool result message
        let has_tool_results = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if has_tool_results {
            // For tool results, we need to create multiple messages (handled separately)
            // This branch shouldn't be hit directly
            OpenAIMessage {
                role: "tool".to_string(),
                content: None,
                tool_calls: None,
                tool_call_id: None,
            }
        } else {
            // Check for tool use blocks (assistant with tool calls)
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

            // Extract text content
            let text: String = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");

            OpenAIMessage {
                role: role.to_string(),
                content: if text.is_empty() { None } else { Some(text) },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                tool_call_id: None,
            }
        }
    }
}

fn convert_messages(messages: &[Message]) -> Vec<OpenAIMessage> {
    let mut result = Vec::new();

    for msg in messages {
        // Check if this message contains tool results
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
            // Create separate messages for each tool result
            for (tool_use_id, content) in tool_results {
                result.push(OpenAIMessage {
                    role: "tool".to_string(),
                    content: Some(content),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                });
            }
        } else {
            result.push(OpenAIMessage::from(msg));
        }
    }

    result
}

impl From<&Request> for OpenAIRequest {
    fn from(req: &Request) -> Self {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = req.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(system.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert conversation messages
        messages.extend(convert_messages(&req.messages));

        OpenAIRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            tools: req.tools.iter().map(OpenAITool::from).collect(),
            stream: None,
        }
    }
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
            },
        }
    }
}

fn parse_sse_line(line: &str) -> Option<OpenAIStreamChunk> {
    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str(data).ok()
}

#[async_trait]
impl super::client::LlmClient for OpenAIClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let openai_req = OpenAIRequest::from(req);

        let response = self
            .http
            .post(OPENAI_API_URL)
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
        let mut openai_req = OpenAIRequest::from(req);
        openai_req.stream = Some(true);

        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let response = http
                .post(OPENAI_API_URL)
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
            let mut text_block_started = false;
            // Track tool calls: (id, name, args, block_started)
            let mut current_tool_calls: Vec<(String, String, String, bool)> = Vec::new();

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
                            };
                            message_started = true;
                        }

                        for choice in chunk.choices {
                            // Handle text content
                            if let Some(text) = choice.delta.content {
                                // Emit ContentBlockStart for text on first text delta
                                if !text_block_started {
                                    yield StreamEvent::ContentBlockStart {
                                        index: 0,
                                        block: ContentBlock::Text { text: String::new() },
                                    };
                                    text_block_started = true;
                                }
                                yield StreamEvent::ContentBlockDelta {
                                    index: 0,
                                    text,
                                };
                            }

                            // Handle tool calls
                            if let Some(tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let idx = tc.index;

                                    // Ensure we have space for this tool call
                                    while current_tool_calls.len() <= idx {
                                        current_tool_calls.push((String::new(), String::new(), String::new(), false));
                                    }

                                    // Accumulate tool call data
                                    if let Some(id) = tc.id {
                                        current_tool_calls[idx].0 = id;
                                    }
                                    if let Some(func) = tc.function {
                                        if let Some(name) = func.name {
                                            current_tool_calls[idx].1 = name;
                                        }

                                        // Emit ContentBlockStart when we have id and name (before JSON deltas)
                                        let (ref id, ref name, _, ref mut started) = current_tool_calls[idx];
                                        if !*started && !id.is_empty() && !name.is_empty() {
                                            yield StreamEvent::ContentBlockStart {
                                                index: idx + 1, // offset by 1 for tool calls (text is 0)
                                                block: ContentBlock::ToolUse {
                                                    id: id.clone(),
                                                    name: name.clone(),
                                                    input: serde_json::Value::Object(serde_json::Map::new()),
                                                },
                                            };
                                            *started = true;
                                        }

                                        if let Some(args) = func.arguments {
                                            current_tool_calls[idx].2.push_str(&args);
                                            // Yield as input JSON delta for tool argument accumulation
                                            yield StreamEvent::InputJsonDelta {
                                                index: idx + 1, // offset by 1 for tool calls
                                                partial_json: args,
                                            };
                                        }
                                    }
                                }
                            }

                            // Handle finish reason
                            if let Some(reason) = choice.finish_reason {
                                // Close text block if started
                                if text_block_started {
                                    yield StreamEvent::ContentBlockStop { index: 0 };
                                }

                                // Close tool call blocks
                                for (idx, (id, _, _, started)) in current_tool_calls.iter().enumerate() {
                                    if *started && !id.is_empty() {
                                        yield StreamEvent::ContentBlockStop { index: idx + 1 };
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

        let openai_req = OpenAIRequest::from(&req);
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
        };

        let openai_tool = OpenAITool::from(&tool);
        assert_eq!(openai_tool.tool_type, "function");
        assert_eq!(openai_tool.function.name, "get_weather");
    }
}
