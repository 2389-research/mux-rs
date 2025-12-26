// ABOUTME: McpProxyTool wraps an MCP server tool for use in the registry.
// ABOUTME: Forwards tool calls to the MCP server.

use std::sync::Arc;

use async_trait::async_trait;

use super::{McpClient, McpContentBlock, McpToolInfo};
use crate::tool::{Tool, ToolResult};

/// A tool that proxies calls to an MCP server.
pub struct McpProxyTool {
    client: Arc<McpClient>,
    info: McpToolInfo,
    prefixed_name: String,
}

impl McpProxyTool {
    /// Create a new proxy tool.
    pub fn new(client: Arc<McpClient>, info: McpToolInfo, prefix: Option<&str>) -> Self {
        let prefixed_name = match prefix {
            Some(p) => format!("{}_{}", p, info.name),
            None => info.name.clone(),
        };
        Self {
            client,
            info,
            prefixed_name,
        }
    }
}

#[async_trait]
impl Tool for McpProxyTool {
    fn name(&self) -> &str {
        &self.prefixed_name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn schema(&self) -> serde_json::Value {
        self.info.input_schema.clone()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let result = self.client.call_tool(&self.info.name, params).await?;

        // Convert MCP result to ToolResult by extracting text from content blocks
        // Note: Image content is represented as a placeholder since ToolResult is text-only
        let content = result
            .content
            .iter()
            .map(|c| match c {
                McpContentBlock::Text { text } => text.clone(),
                McpContentBlock::Image { mime_type, .. } => {
                    format!("[Image: {}]", mime_type)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Ok(ToolResult::error(content))
        } else {
            Ok(ToolResult::text(content))
        }
    }
}
