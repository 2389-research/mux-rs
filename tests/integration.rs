// ABOUTME: Integration tests verifying modules work together.
// ABOUTME: Tests the full workflow without external dependencies.

use mux::prelude::*;

/// A test tool for integration testing.
struct GreetTool;

#[async_trait::async_trait]
impl Tool for GreetTool {
    fn name(&self) -> &str {
        "greet"
    }

    fn description(&self) -> &str {
        "Greet a person by name"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name to greet"
                }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let name = params["name"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing name parameter"))?;
        Ok(ToolResult::text(format!("Hello, {}!", name)))
    }
}

#[tokio::test]
async fn test_registry_with_policy() {
    // Create registry and register tool
    let registry = Registry::new();
    registry.register(GreetTool).await;

    // Create policy
    let policy = Policy::builder()
        .allow("greet")
        .deny_pattern("dangerous_*")
        .default(Decision::Deny)
        .build();

    // Get tool and check policy
    let tool = registry.get("greet").await.expect("Tool should exist");
    let params = serde_json::json!({"name": "World"});

    let decision = policy.evaluate(tool.name(), &params);
    assert_eq!(decision, Decision::Allow);

    // Execute tool
    let result = tool
        .execute(params)
        .await
        .expect("Execution should succeed");
    assert_eq!(result.content, "Hello, World!");
    assert!(!result.is_error);
}

#[tokio::test]
async fn test_tool_definitions_for_llm() {
    let registry = Registry::new();
    registry.register(GreetTool).await;

    let definitions = registry.to_definitions().await;
    assert_eq!(definitions.len(), 1);

    let def = &definitions[0];
    assert_eq!(def.name, "greet");
    assert_eq!(def.description, "Greet a person by name");
    assert!(def.input_schema["properties"]["name"].is_object());
}

#[tokio::test]
async fn test_message_construction() {
    let user_msg = Message::user("Hello");
    let assistant_msg = Message::assistant("Hi there!");

    assert_eq!(user_msg.role, Role::User);
    assert_eq!(assistant_msg.role, Role::Assistant);
}

#[tokio::test]
async fn test_request_building() {
    let registry = Registry::new();
    registry.register(GreetTool).await;

    let request = Request::new("claude-sonnet-4-20250514")
        .message(Message::user("Greet Alice"))
        .tools(registry.to_definitions().await)
        .system("You are helpful")
        .max_tokens(1024);

    assert_eq!(request.model, "claude-sonnet-4-20250514");
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.tools.len(), 1);
    assert_eq!(request.system, Some("You are helpful".to_string()));
}

#[tokio::test]
async fn test_conditional_policy() {
    let policy = Policy::builder()
        .conditional("greet", |params| {
            let name = params["name"].as_str().unwrap_or("");
            if name.is_empty() {
                Decision::Deny
            } else {
                Decision::Allow
            }
        })
        .default(Decision::Deny)
        .build();

    assert_eq!(
        policy.evaluate("greet", &serde_json::json!({"name": "Alice"})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("greet", &serde_json::json!({"name": ""})),
        Decision::Deny
    );
}
