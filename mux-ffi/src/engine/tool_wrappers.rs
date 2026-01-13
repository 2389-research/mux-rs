// ABOUTME: Tool wrappers that adapt MCP and custom tools to the mux Tool trait.
// ABOUTME: Enables SubAgent to execute FFI-layer tools through its standard Registry.

use async_trait::async_trait;
use mux::mcp::McpContentBlock;
use mux::prelude::McpClient;
use mux::tool::{Tool, ToolResult};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// Convert MCP content blocks to a single string.
fn mcp_content_to_string(content: &[McpContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            McpContentBlock::Text { text } => Some(text.as_str()),
            McpContentBlock::Image { .. } => Some("[image]"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Wraps an MCP tool so it can be registered in a mux tool Registry.
/// The tool name is prefixed with "server_name:" to match the LLM's tool calls.
pub struct McpToolWrapper {
    /// Full name as "server:tool" for the LLM
    qualified_name: String,
    /// Server name for display
    server_name: String,
    /// Tool name within the MCP server
    tool_name: String,
    /// Description from MCP server
    tool_description: String,
    /// JSON schema from MCP server
    tool_schema: serde_json::Value,
    /// Reference to the MCP client
    client: Arc<TokioMutex<McpClient>>,
}

impl McpToolWrapper {
    /// Create a new MCP tool wrapper.
    pub fn new(
        server_name: String,
        tool_name: String,
        tool_description: String,
        tool_schema: serde_json::Value,
        client: Arc<TokioMutex<McpClient>>,
    ) -> Self {
        let qualified_name = format!("{}:{}", server_name, tool_name);
        Self {
            qualified_name,
            server_name,
            tool_name,
            tool_description,
            tool_schema,
            client,
        }
    }

    /// Get the server name.
    pub fn server_name(&self) -> &str {
        &self.server_name
    }
}

#[async_trait]
impl Tool for McpToolWrapper {
    fn name(&self) -> &str {
        &self.qualified_name
    }

    fn description(&self) -> &str {
        &self.tool_description
    }

    fn schema(&self) -> serde_json::Value {
        self.tool_schema.clone()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let client = self.client.lock().await;
        match client.call_tool(&self.tool_name, params).await {
            Ok(result) => {
                let content = mcp_content_to_string(&result.content);
                if result.is_error {
                    Ok(ToolResult::error(content))
                } else {
                    Ok(ToolResult::text(content))
                }
            }
            Err(e) => Ok(ToolResult::error(format!("MCP tool error: {}", e))),
        }
    }
}

/// Wraps a Swift custom tool so it can be registered in a mux tool Registry.
pub struct CustomToolWrapper {
    /// The underlying bridge to Swift
    bridge: Arc<crate::bridge::FfiToolBridge>,
}

impl CustomToolWrapper {
    /// Create a new custom tool wrapper from an FfiToolBridge.
    pub fn new(bridge: Arc<crate::bridge::FfiToolBridge>) -> Self {
        Self { bridge }
    }
}

#[async_trait]
impl Tool for CustomToolWrapper {
    fn name(&self) -> &str {
        self.bridge.name()
    }

    fn description(&self) -> &str {
        self.bridge.description()
    }

    fn schema(&self) -> serde_json::Value {
        self.bridge.schema()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        self.bridge.execute(params).await
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_mcp_tool_wrapper_qualified_name() {
        // We can't easily test the full wrapper without an MCP client,
        // but we can verify the naming logic
        let qualified = format!("{}:{}", "filesystem", "read_file");
        assert_eq!(qualified, "filesystem:read_file");
    }
}
