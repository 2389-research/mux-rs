// ABOUTME: Bridge types that adapt Swift callbacks to Rust traits.
// ABOUTME: Enables Swift to implement hooks and custom tools.

use crate::callback::{CustomTool, HookHandler};
use crate::types::{HookEventType, HookResponse};
use async_trait::async_trait;
use mux::hook::{Hook, HookAction, HookEvent};
use mux::tool::{Tool, ToolResult};
use std::sync::Arc;

/// Bridges Swift HookHandler to Rust Hook trait
pub struct FfiHookBridge {
    handler: Arc<Box<dyn HookHandler>>,
}

impl FfiHookBridge {
    pub fn new(handler: Box<dyn HookHandler>) -> Self {
        Self {
            handler: Arc::new(handler),
        }
    }
}

#[async_trait]
impl Hook for FfiHookBridge {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        // Serialize input with proper error handling
        let ffi_event = match event {
            HookEvent::PreToolUse { tool_name, input } => {
                let input_json = serde_json::to_string(input)
                    .map_err(|e| anyhow::anyhow!("Failed to serialize tool input: {}", e))?;
                HookEventType::PreToolUse {
                    tool_name: tool_name.clone(),
                    input: input_json,
                }
            }
            HookEvent::PostToolUse {
                tool_name,
                tool_use_id: _,
                input,
                result,
            } => {
                let input_json = serde_json::to_string(input)
                    .map_err(|e| anyhow::anyhow!("Failed to serialize tool input: {}", e))?;
                HookEventType::PostToolUse {
                    tool_name: tool_name.clone(),
                    input: input_json,
                    result: result.content.clone(),
                }
            }
            HookEvent::AgentStart { agent_id, task } => HookEventType::AgentStart {
                agent_id: agent_id.clone(),
                task: task.clone(),
            },
            HookEvent::AgentStop {
                agent_id,
                result: _,
            } => HookEventType::AgentStop {
                agent_id: agent_id.clone(),
            },
            HookEvent::Iteration {
                agent_id,
                iteration,
            } => HookEventType::Iteration {
                agent_id: agent_id.clone(),
                iteration: *iteration as u32,
            },
            // Session, subagent, and response events - pass through without FFI callback
            HookEvent::SessionStart { .. }
            | HookEvent::SessionEnd { .. }
            | HookEvent::Stop { .. }
            | HookEvent::SubagentStart { .. }
            | HookEvent::SubagentStop { .. }
            | HookEvent::ResponseReceived { .. }
            | HookEvent::StreamDelta { .. }
            | HookEvent::StreamUsage { .. } => {
                return Ok(HookAction::Continue);
            }
        };

        // Clone handler Arc to move into blocking task
        let handler = self.handler.clone();

        // Run the synchronous FFI callback on a blocking thread to avoid blocking
        // the async runtime. This prevents deadlocks when Swift callbacks do slow work.
        let response = tokio::task::spawn_blocking(move || handler.on_event(ffi_event))
            .await
            .map_err(|e| anyhow::anyhow!("Hook callback task panicked: {}", e))?;

        match response {
            HookResponse::Continue => Ok(HookAction::Continue),
            HookResponse::Block { reason } => Ok(HookAction::Block(reason)),
            HookResponse::Transform { new_input } => {
                let value = serde_json::from_str(&new_input)
                    .map_err(|e| anyhow::anyhow!("Invalid JSON in transform: {}", e))?;
                Ok(HookAction::Transform(value))
            }
        }
    }

    fn accepts(&self, _event: &HookEvent) -> bool {
        true
    }
}

/// Bridges Swift CustomTool to Rust Tool trait
pub struct FfiToolBridge {
    tool: Box<dyn CustomTool>,
    name: String,
    description: String,
    schema: serde_json::Value,
}

impl std::fmt::Debug for FfiToolBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiToolBridge")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("schema", &self.schema)
            .finish()
    }
}

impl FfiToolBridge {
    pub fn new(tool: Box<dyn CustomTool>) -> Result<Self, anyhow::Error> {
        let name = tool.name();
        let description = tool.description();
        let schema_json = tool.schema_json();
        let schema: serde_json::Value = serde_json::from_str(&schema_json)
            .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?;

        Ok(Self {
            tool,
            name,
            description,
            schema,
        })
    }
}

#[async_trait]
impl Tool for FfiToolBridge {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let input = serde_json::to_string(&params)?;

        // We can't move the tool into spawn_blocking since Box<dyn CustomTool> isn't Send.
        // Instead, we execute synchronously but note that custom tool implementations
        // should be fast. If Swift needs to do slow work, it should dispatch to its own
        // background queue and return immediately.
        let result = self.tool.execute(input);

        if result.is_error {
            Ok(ToolResult::error(result.content))
        } else {
            Ok(ToolResult::text(result.content))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolExecutionResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock hook handler for testing
    struct MockHookHandler {
        call_count: AtomicUsize,
        response: HookResponse,
    }

    impl MockHookHandler {
        fn new(response: HookResponse) -> Self {
            Self {
                call_count: AtomicUsize::new(0),
                response,
            }
        }
    }

    impl HookHandler for MockHookHandler {
        fn on_event(&self, _event: HookEventType) -> HookResponse {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            self.response.clone()
        }
    }

    /// Mock custom tool for testing
    struct MockCustomTool {
        tool_name: String,
        tool_description: String,
        tool_schema: String,
        result: ToolExecutionResult,
    }

    impl MockCustomTool {
        fn new(name: &str, result: ToolExecutionResult) -> Self {
            Self {
                tool_name: name.to_string(),
                tool_description: format!("Mock tool: {}", name),
                tool_schema: r#"{"type": "object", "properties": {}}"#.to_string(),
                result,
            }
        }
    }

    impl CustomTool for MockCustomTool {
        fn name(&self) -> String {
            self.tool_name.clone()
        }

        fn description(&self) -> String {
            self.tool_description.clone()
        }

        fn schema_json(&self) -> String {
            self.tool_schema.clone()
        }

        fn execute(&self, _input_json: String) -> ToolExecutionResult {
            self.result.clone()
        }
    }

    #[tokio::test]
    async fn test_ffi_hook_bridge_continue() {
        let handler = Box::new(MockHookHandler::new(HookResponse::Continue));
        let bridge = FfiHookBridge::new(handler);

        let event = HookEvent::PreToolUse {
            tool_name: "test_tool".to_string(),
            input: serde_json::json!({"key": "value"}),
        };

        let result = bridge.on_event(&event).await.unwrap();
        assert!(matches!(result, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_ffi_hook_bridge_block() {
        let handler = Box::new(MockHookHandler::new(HookResponse::Block {
            reason: "Not allowed".to_string(),
        }));
        let bridge = FfiHookBridge::new(handler);

        let event = HookEvent::PreToolUse {
            tool_name: "dangerous_tool".to_string(),
            input: serde_json::json!({}),
        };

        let result = bridge.on_event(&event).await.unwrap();
        match result {
            HookAction::Block(reason) => assert_eq!(reason, "Not allowed"),
            _ => panic!("Expected Block action"),
        }
    }

    #[tokio::test]
    async fn test_ffi_hook_bridge_transform() {
        let handler = Box::new(MockHookHandler::new(HookResponse::Transform {
            new_input: r#"{"modified": true}"#.to_string(),
        }));
        let bridge = FfiHookBridge::new(handler);

        let event = HookEvent::PreToolUse {
            tool_name: "test_tool".to_string(),
            input: serde_json::json!({"original": true}),
        };

        let result = bridge.on_event(&event).await.unwrap();
        match result {
            HookAction::Transform(value) => {
                assert_eq!(value["modified"], true);
            }
            _ => panic!("Expected Transform action"),
        }
    }

    #[tokio::test]
    async fn test_ffi_hook_bridge_accepts_all() {
        let handler = Box::new(MockHookHandler::new(HookResponse::Continue));
        let bridge = FfiHookBridge::new(handler);

        let events = vec![
            HookEvent::PreToolUse {
                tool_name: "test".to_string(),
                input: serde_json::json!({}),
            },
            HookEvent::AgentStart {
                agent_id: "agent-1".to_string(),
                task: "test task".to_string(),
            },
        ];

        for event in events {
            assert!(bridge.accepts(&event));
        }
    }

    #[test]
    fn test_ffi_tool_bridge_creation() {
        let tool = Box::new(MockCustomTool::new(
            "test_tool",
            ToolExecutionResult::success("result".to_string()),
        ));
        let bridge = FfiToolBridge::new(tool).unwrap();

        assert_eq!(bridge.name(), "test_tool");
        assert_eq!(bridge.description(), "Mock tool: test_tool");
    }

    #[test]
    fn test_ffi_tool_bridge_invalid_schema() {
        struct BadSchemaTool;

        impl CustomTool for BadSchemaTool {
            fn name(&self) -> String {
                "bad_tool".to_string()
            }
            fn description(&self) -> String {
                "A tool with invalid schema".to_string()
            }
            fn schema_json(&self) -> String {
                "not valid json{".to_string()
            }
            fn execute(&self, _input_json: String) -> ToolExecutionResult {
                ToolExecutionResult::success("".to_string())
            }
        }

        let result = FfiToolBridge::new(Box::new(BadSchemaTool));
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid JSON schema")
        );
    }

    #[tokio::test]
    async fn test_ffi_tool_bridge_execute_success() {
        let tool = Box::new(MockCustomTool::new(
            "calculator",
            ToolExecutionResult::success("42".to_string()),
        ));
        let bridge = FfiToolBridge::new(tool).unwrap();

        let result = bridge
            .execute(serde_json::json!({"a": 1, "b": 2}))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content, "42");
    }

    #[tokio::test]
    async fn test_ffi_tool_bridge_execute_error() {
        let tool = Box::new(MockCustomTool::new(
            "failing_tool",
            ToolExecutionResult::error("Something went wrong".to_string()),
        ));
        let bridge = FfiToolBridge::new(tool).unwrap();

        let result = bridge.execute(serde_json::json!({})).await.unwrap();

        assert!(result.is_error);
        assert_eq!(result.content, "Something went wrong");
    }
}
