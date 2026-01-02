// ABOUTME: Callback interfaces for async operations from Rust to Swift.
// ABOUTME: Swift implements these traits to receive streaming updates.

use crate::types::{HookEventType, HookResponse, SubagentResult, ToolExecutionResult};

/// Represents a tool use request that will be sent to Swift for display/logging.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolUseRequest {
    pub id: String,
    pub tool_name: String,
    pub server_name: String,
    pub arguments: String,
}

/// Represents the final result of a chat completion.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatResult {
    pub conversation_id: String,
    pub final_text: String,
    pub tool_use_count: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Callback interface that Swift implements to receive streaming chat updates.
/// UniFFI generates the necessary bridging code for cross-language calls.
#[uniffi::export(callback_interface)]
pub trait ChatCallback: Send + Sync {
    /// Called when new text content is streamed from the LLM.
    fn on_text_delta(&self, text: String);

    /// Called when the LLM requests to use a tool.
    fn on_tool_use(&self, request: ToolUseRequest);

    /// Called when a tool execution completes with a result.
    fn on_tool_result(&self, tool_id: String, result: String);

    /// Called when the entire chat completion finishes successfully.
    fn on_complete(&self, result: ChatResult);

    /// Called when an error occurs during chat processing.
    fn on_error(&self, error: String);
}

/// Hook handler interface - Swift implements to intercept lifecycle events.
/// Hooks allow interception of tool execution and agent lifecycle for
/// validation, logging, or transformation of inputs/outputs.
#[uniffi::export(callback_interface)]
pub trait HookHandler: Send + Sync {
    /// Called when a hook event occurs. Returns a response indicating
    /// whether to continue, block, or transform the operation.
    fn on_event(&self, event: HookEventType) -> HookResponse;
}

/// Callback for receiving subagent streaming updates.
/// Swift implements this to receive real-time updates from subagent execution.
#[uniffi::export(callback_interface)]
pub trait SubagentCallback: Send + Sync {
    /// Called when new text content is streamed from the subagent.
    fn on_text_delta(&self, agent_id: String, text: String);

    /// Called when the subagent requests to use a tool.
    fn on_tool_use(&self, agent_id: String, request: ToolUseRequest);

    /// Called when a tool execution completes with a result.
    fn on_tool_result(&self, agent_id: String, tool_id: String, result: String);

    /// Called when the subagent completes successfully.
    fn on_complete(&self, result: SubagentResult);

    /// Called when an error occurs during subagent processing.
    fn on_error(&self, agent_id: String, error: String);
}

/// Custom tool interface - Swift implements to provide custom tools.
/// Allows Swift code to register tools that can be called by the LLM.
#[uniffi::export(callback_interface)]
pub trait CustomTool: Send + Sync {
    /// Returns the unique name of the tool.
    fn name(&self) -> String;

    /// Returns a human-readable description of what the tool does.
    fn description(&self) -> String;

    /// Returns the JSON schema defining the tool's input parameters.
    fn schema_json(&self) -> String;

    /// Executes the tool with the given JSON input and returns the result.
    fn execute(&self, input_json: String) -> ToolExecutionResult;
}
