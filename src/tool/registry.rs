// ABOUTME: Implements the Registry - a thread-safe container for discovering
// ABOUTME: and managing available tools at runtime.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::Tool;
use crate::error::McpError;
use crate::llm::ToolDefinition;
use crate::mcp::{McpClient, McpProxyTool};

/// A thread-safe registry of tools.
#[derive(Default)]
pub struct Registry {
    tools: Arc<RwLock<HashMap<String, Arc<dyn Tool>>>>,
}

impl Registry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool.
    pub async fn register<T: Tool + 'static>(&self, tool: T) {
        self.register_arc(Arc::new(tool)).await;
    }

    /// Register a tool from an Arc.
    pub async fn register_arc(&self, tool: Arc<dyn Tool>) {
        let mut tools = self.tools.write().await;
        tools.insert(tool.name().to_string(), tool);
    }

    /// Unregister a tool by name.
    pub async fn unregister(&self, name: &str) {
        let mut tools = self.tools.write().await;
        tools.remove(name);
    }

    /// Get a tool by name.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.get(name).cloned()
    }

    /// List all tool names, sorted alphabetically.
    pub async fn list(&self) -> Vec<String> {
        let tools = self.tools.read().await;
        let mut names: Vec<_> = tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get all registered tools.
    pub async fn all(&self) -> Vec<Arc<dyn Tool>> {
        let tools = self.tools.read().await;
        tools.values().cloned().collect()
    }

    /// Get the number of registered tools.
    pub async fn count(&self) -> usize {
        let tools = self.tools.read().await;
        tools.len()
    }

    /// Convert all tools to LLM tool definitions.
    pub async fn to_definitions(&self) -> Vec<ToolDefinition> {
        let tools = self.tools.read().await;
        tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.schema(),
            })
            .collect()
    }

    /// Merge tools from an MCP client into the registry.
    pub async fn merge_mcp(
        &self,
        client: Arc<McpClient>,
        prefix: Option<&str>,
    ) -> Result<usize, McpError> {
        let tools = client.list_tools().await?;
        let count = tools.len();

        for info in tools {
            let proxy = McpProxyTool::new(client.clone(), info, prefix);
            self.register(proxy).await;
        }

        Ok(count)
    }
}

impl Clone for Registry {
    fn clone(&self) -> Self {
        Self {
            tools: Arc::clone(&self.tools),
        }
    }
}
