// ABOUTME: Defines the ToolResult type - a unified structure for tool
// ABOUTME: execution outcomes with content, error state, and metadata.

use std::collections::HashMap;

use serde::Serialize;

/// Result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The output content.
    pub content: String,

    /// Whether this result represents an error.
    pub is_error: bool,

    /// Optional metadata about the execution.
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolResult {
    /// Create a successful text result.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: HashMap::new(),
        }
    }

    /// Create an error result.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            is_error: true,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the result.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.metadata.insert(key.into(), v);
        }
        self
    }
}

impl Default for ToolResult {
    fn default() -> Self {
        Self::text("")
    }
}
