// ABOUTME: FilteredRegistry - a decorator that restricts tool access.
// ABOUTME: Implements allowlist/denylist filtering on top of a base Registry.

use std::sync::Arc;

use crate::llm::ToolDefinition;
use crate::tool::{Registry, Tool};

/// A filtered view of a Registry that restricts tool access.
///
/// Uses the decorator pattern to wrap a Registry and filter tool access
/// based on allowlist/denylist rules. Denylist takes precedence.
pub struct FilteredRegistry {
    source: Registry,
    allowed_tools: Option<Vec<String>>,
    denied_tools: Vec<String>,
}

impl FilteredRegistry {
    /// Create a new filtered registry from a source registry.
    pub fn new(source: Registry) -> Self {
        Self {
            source,
            allowed_tools: None,
            denied_tools: Vec::new(),
        }
    }

    /// Set the allowlist of tools. If None, all tools are allowed.
    pub fn allowed(mut self, tools: Option<Vec<String>>) -> Self {
        self.allowed_tools = tools;
        self
    }

    /// Set the denylist of tools. Takes precedence over allowlist.
    pub fn denied(mut self, tools: Vec<String>) -> Self {
        self.denied_tools = tools;
        self
    }

    /// Check if a tool name passes the filter.
    pub fn is_allowed(&self, name: &str) -> bool {
        // Denylist always wins
        if self.denied_tools.iter().any(|d| d == name) {
            return false;
        }

        // If no allowlist, everything (not denied) is allowed
        match &self.allowed_tools {
            None => true,
            Some(allowed) => allowed.iter().any(|a| a == name),
        }
    }

    /// Get a tool by name if it passes the filter.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        if !self.is_allowed(name) {
            return None;
        }
        self.source.get(name).await
    }

    /// List all tool names that pass the filter.
    pub async fn list(&self) -> Vec<String> {
        self.source
            .list()
            .await
            .into_iter()
            .filter(|name| self.is_allowed(name))
            .collect()
    }

    /// Get all tools that pass the filter.
    pub async fn all(&self) -> Vec<Arc<dyn Tool>> {
        let tools = self.source.all().await;
        tools
            .into_iter()
            .filter(|t| self.is_allowed(t.name()))
            .collect()
    }

    /// Convert filtered tools to LLM tool definitions.
    pub async fn to_definitions(&self) -> Vec<ToolDefinition> {
        self.all()
            .await
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.schema(),
            })
            .collect()
    }

    /// Get the number of tools that pass the filter.
    pub async fn count(&self) -> usize {
        self.list().await.len()
    }
}

impl Clone for FilteredRegistry {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            allowed_tools: self.allowed_tools.clone(),
            denied_tools: self.denied_tools.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolResult;
    use async_trait::async_trait;

    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }
        fn description(&self) -> &str {
            "A mock tool"
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }
        async fn execute(&self, _params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
            Ok(ToolResult::text("ok"))
        }
    }

    #[tokio::test]
    async fn test_filtered_no_restrictions() {
        let registry = Registry::new();
        registry.register(MockTool { name: "read".into() }).await;
        registry.register(MockTool { name: "write".into() }).await;

        let filtered = FilteredRegistry::new(registry);

        assert_eq!(filtered.count().await, 2);
        assert!(filtered.get("read").await.is_some());
        assert!(filtered.get("write").await.is_some());
    }

    #[tokio::test]
    async fn test_filtered_allowlist() {
        let registry = Registry::new();
        registry.register(MockTool { name: "read".into() }).await;
        registry.register(MockTool { name: "write".into() }).await;
        registry.register(MockTool { name: "delete".into() }).await;

        let filtered = FilteredRegistry::new(registry)
            .allowed(Some(vec!["read".into(), "write".into()]));

        assert_eq!(filtered.count().await, 2);
        assert!(filtered.get("read").await.is_some());
        assert!(filtered.get("write").await.is_some());
        assert!(filtered.get("delete").await.is_none());
    }

    #[tokio::test]
    async fn test_filtered_denylist() {
        let registry = Registry::new();
        registry.register(MockTool { name: "read".into() }).await;
        registry.register(MockTool { name: "write".into() }).await;
        registry.register(MockTool { name: "delete".into() }).await;

        let filtered = FilteredRegistry::new(registry)
            .denied(vec!["delete".into()]);

        assert_eq!(filtered.count().await, 2);
        assert!(filtered.get("read").await.is_some());
        assert!(filtered.get("write").await.is_some());
        assert!(filtered.get("delete").await.is_none());
    }

    #[tokio::test]
    async fn test_filtered_denylist_overrides_allowlist() {
        let registry = Registry::new();
        registry.register(MockTool { name: "read".into() }).await;
        registry.register(MockTool { name: "write".into() }).await;

        let filtered = FilteredRegistry::new(registry)
            .allowed(Some(vec!["read".into(), "write".into()]))
            .denied(vec!["write".into()]);

        assert_eq!(filtered.count().await, 1);
        assert!(filtered.get("read").await.is_some());
        assert!(filtered.get("write").await.is_none());
    }

    #[tokio::test]
    async fn test_filtered_to_definitions() {
        let registry = Registry::new();
        registry.register(MockTool { name: "read".into() }).await;
        registry.register(MockTool { name: "write".into() }).await;

        let filtered = FilteredRegistry::new(registry)
            .allowed(Some(vec!["read".into()]));

        let defs = filtered.to_definitions().await;
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "read");
    }
}
