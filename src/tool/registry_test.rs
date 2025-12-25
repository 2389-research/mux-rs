// ABOUTME: Tests for tool Registry - registration, lookup, thread safety.
// ABOUTME: Uses a mock tool for testing.

use super::*;

/// A simple test tool.
struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echoes input back"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let message = params["message"].as_str().unwrap_or("");
        Ok(ToolResult::text(message))
    }
}

#[tokio::test]
async fn test_register_and_get() {
    let registry = Registry::new();
    registry.register(EchoTool).await;

    let tool = registry.get("echo").await;
    assert!(tool.is_some());
    assert_eq!(tool.unwrap().name(), "echo");
}

#[tokio::test]
async fn test_get_nonexistent() {
    let registry = Registry::new();
    let tool = registry.get("nonexistent").await;
    assert!(tool.is_none());
}

#[tokio::test]
async fn test_unregister() {
    let registry = Registry::new();
    registry.register(EchoTool).await;
    assert_eq!(registry.count().await, 1);

    registry.unregister("echo").await;
    assert_eq!(registry.count().await, 0);
    assert!(registry.get("echo").await.is_none());
}

#[tokio::test]
async fn test_list() {
    let registry = Registry::new();
    registry.register(EchoTool).await;

    let names = registry.list().await;
    assert_eq!(names, vec!["echo"]);
}

#[tokio::test]
async fn test_to_definitions() {
    let registry = Registry::new();
    registry.register(EchoTool).await;

    let defs = registry.to_definitions().await;
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "echo");
    assert_eq!(defs[0].description, "Echoes input back");
}

#[tokio::test]
async fn test_clone_shares_state() {
    let registry = Registry::new();
    let clone = registry.clone();

    registry.register(EchoTool).await;
    assert_eq!(clone.count().await, 1);
}
