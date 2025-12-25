// ABOUTME: Anthropic Claude API client implementation.
// ABOUTME: Implements LlmClient trait for Claude models.

use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use super::{ContentBlock, Message, Request, Response, StopReason, ToolDefinition, Usage};
use super::client::StreamEvent;
use crate::error::LlmError;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic API request format.
#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
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

/// Anthropic content block.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContent {
    Text { text: String },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String, #[serde(default)] is_error: bool },
}

/// Anthropic tool definition.
#[derive(Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
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

/// Client for the Anthropic Claude API.
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    api_key: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            http: reqwest::Client::new(),
        }
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

impl From<&ContentBlock> for AnthropicContent {
    fn from(block: &ContentBlock) -> Self {
        match block {
            ContentBlock::Text { text } => AnthropicContent::Text { text: text.clone() },
            ContentBlock::ToolUse { id, name, input } => AnthropicContent::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                AnthropicContent::ToolResult {
                    tool_use_id: tool_use_id.clone(),
                    content: content.clone(),
                    is_error: *is_error,
                }
            }
        }
    }
}

impl From<AnthropicContent> for ContentBlock {
    fn from(content: AnthropicContent) -> Self {
        match content {
            AnthropicContent::Text { text } => ContentBlock::Text { text },
            AnthropicContent::ToolUse { id, name, input } => ContentBlock::ToolUse { id, name, input },
            AnthropicContent::ToolResult { tool_use_id, content, is_error } => {
                ContentBlock::ToolResult { tool_use_id, content, is_error }
            }
        }
    }
}

impl From<&Message> for AnthropicMessage {
    fn from(msg: &Message) -> Self {
        AnthropicMessage {
            role: match msg.role {
                super::Role::User => "user".to_string(),
                super::Role::Assistant => "assistant".to_string(),
            },
            content: msg.content.iter().map(AnthropicContent::from).collect(),
        }
    }
}

impl From<&ToolDefinition> for AnthropicTool {
    fn from(tool: &ToolDefinition) -> Self {
        AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        }
    }
}

impl From<&Request> for AnthropicRequest {
    fn from(req: &Request) -> Self {
        AnthropicRequest {
            model: req.model.clone(),
            messages: req.messages.iter().map(AnthropicMessage::from).collect(),
            max_tokens: req.max_tokens.unwrap_or(4096),
            system: req.system.clone(),
            temperature: req.temperature,
            tools: req.tools.iter().map(AnthropicTool::from).collect(),
            stream: None,
        }
    }
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
            },
        }
    }
}

#[async_trait]
impl super::client::LlmClient for AnthropicClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let anthropic_req = AnthropicRequest::from(req);

        let response = self
            .http
            .post(ANTHROPIC_API_URL)
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
        _req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        // TODO: Implement streaming in next task
        Box::pin(futures::stream::empty())
    }
}
