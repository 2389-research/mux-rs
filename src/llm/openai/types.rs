// ABOUTME: Wire types for the OpenAI Chat Completions API (request and response structs/enums).
// ABOUTME: All types are pub and re-exported from the openai module.

use serde::{Deserialize, Serialize};

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
