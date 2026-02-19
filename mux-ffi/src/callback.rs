// ABOUTME: Callback interfaces for async operations from Rust to Swift.
// ABOUTME: Swift implements these traits to receive streaming updates.

use crate::types::{
    HookEventType, HookResponse, LlmRequest, LlmResponse, SubagentResult, ToolExecutionResult,
};

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
    pub context_usage: crate::context::ContextUsage,
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

    /// Called when context usage exceeds the warning threshold.
    /// Swift can use this to show a UI warning or trigger compaction.
    fn on_context_warning(&self, usage: crate::context::ContextUsage);
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

/// Event handler for TaskTool-spawned subagents during chat.
/// Swift implements this to receive streaming updates when the main agent
/// spawns subagents via the TaskTool. Events fire during the parent's
/// tool execution, before the tool result is returned.
#[uniffi::export(callback_interface)]
pub trait SubagentEventHandler: Send + Sync {
    /// Called when a subagent is spawned by the TaskTool.
    fn on_agent_started(
        &self,
        subagent_id: String,
        agent_type: String,
        task: String,
        description: String,
    );

    /// Called when the subagent uses a tool.
    fn on_tool_use(&self, subagent_id: String, tool_name: String, arguments_json: String);

    /// Called when a tool execution completes.
    fn on_tool_result(
        &self,
        subagent_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    );

    /// Called when the subagent completes an iteration of its think-act loop.
    fn on_iteration(&self, subagent_id: String, iteration: u32);

    /// Called when the subagent completes successfully.
    fn on_agent_completed(
        &self,
        subagent_id: String,
        content: String,
        tool_use_count: u32,
        iterations: u32,
        transcript_saved: bool,
    );

    /// Called when the subagent encounters an error.
    fn on_agent_error(&self, subagent_id: String, error: String);

    /// Called for each text token during streaming.
    /// Only fires when the agent's definition has `streaming = true`.
    fn on_stream_delta(&self, subagent_id: String, text: String);

    /// Called when token usage is reported during streaming.
    /// Only fires when the agent's definition has `streaming = true`.
    fn on_stream_usage(
        &self,
        subagent_id: String,
        input_tokens: u32,
        output_tokens: u32,
    );
}

// ============================================================================
// Callback LLM Provider
// ============================================================================

/// Callback interface that Swift implements to provide LLM generation.
/// This allows on-device models (like Apple Foundation Models) to integrate
/// with Mux's orchestration system.
///
/// # Example (Swift)
/// ```swift
/// final class FoundationModelsProvider: LlmProvider {
///     func generate(request: LlmRequest) -> LlmResponse {
///         let session = LanguageModelSession()
///         let response = try session.respond(to: request.messages)
///         return LlmResponse(
///             text: response.text,
///             toolCalls: response.toolCalls.map { ... },
///             usage: LlmUsage(inputTokens: 0, outputTokens: 0),
///             error: nil
///         )
///     }
/// }
/// ```
#[uniffi::export(callback_interface)]
pub trait LlmProvider: Send + Sync {
    /// Generate a response for the given request.
    /// This is a blocking call - implement with async internally if needed.
    /// Return LlmResponse with error field set if generation fails.
    fn generate(&self, request: LlmRequest) -> LlmResponse;
}
