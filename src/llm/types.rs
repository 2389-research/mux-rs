// ABOUTME: Core types for LLM communication - messages, content blocks,
// ABOUTME: tool definitions, requests, and responses.

use serde::{Deserialize, Serialize};

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
}

/// Definition of a tool for the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
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
        self.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }))
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
