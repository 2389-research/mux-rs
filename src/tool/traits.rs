// ABOUTME: Defines the Tool trait - the core abstraction for agent capabilities.
// ABOUTME: Tools have a name, description, schema, and async execute method.

use async_trait::async_trait;

use super::ToolResult;

/// A tool that can be executed by an agent.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the unique name of this tool.
    fn name(&self) -> &str;

    /// Returns a human-readable description for the LLM.
    fn description(&self) -> &str;

    /// Returns the JSON Schema for the tool's input parameters.
    fn schema(&self) -> serde_json::Value;

    /// Check if this invocation requires user approval.
    fn requires_approval(&self, _params: &serde_json::Value) -> bool {
        false
    }

    /// Execute the tool with the given parameters.
    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error>;
}

/// Trait for typed tool execution.
#[async_trait]
pub trait ToolExecute: Send + Sync {
    /// Execute the tool with typed parameters (struct fields).
    async fn execute(&self) -> Result<ToolResult, anyhow::Error>;
}
