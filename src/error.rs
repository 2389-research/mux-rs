// ABOUTME: Defines all error types for the mux library using thiserror.
// ABOUTME: Each submodule has its own error enum, unified under MuxError.

/// Top-level error type for the mux library.
#[derive(Debug, thiserror::Error)]
pub enum MuxError {
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    #[error("Permission error: {0}")]
    Permission(#[from] PermissionError),

    #[error("MCP error: {0}")]
    Mcp(#[from] McpError),
}

/// Errors from LLM client operations.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Stream closed unexpectedly")]
    StreamClosed,

    #[error("Deserialization error: {0}")]
    Deserialize(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Errors from tool operations.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Execution failed: {0}")]
    Execution(#[source] anyhow::Error),
}

/// Errors from permission checks.
#[derive(Debug, thiserror::Error)]
pub enum PermissionError {
    #[error("Tool '{0}' denied by policy")]
    Denied(String),

    #[error("Approval rejected for tool '{0}'")]
    Rejected(String),

    #[error("Approval handler error: {0}")]
    Handler(#[source] anyhow::Error),
}

/// Errors from MCP operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("RPC error ({code}): {message}")]
    Rpc { code: i32, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
