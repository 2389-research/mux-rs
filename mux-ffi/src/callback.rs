// ABOUTME: Callback interfaces for async operations from Rust to Swift.
// ABOUTME: Swift implements these traits to receive streaming updates.

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
