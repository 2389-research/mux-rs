// ABOUTME: Prelude module - convenient imports for common use cases.
// ABOUTME: Use `use mux::prelude::*;` to get started quickly.

pub use crate::error::{LlmError, McpError, MuxError, PermissionError, ToolError};
pub use crate::llm::{
    AnthropicClient, ContentBlock, LlmClient, Message, Request, Response, Role, StopReason,
    StreamEvent, ToolDefinition, Usage,
};
pub use crate::mcp::{
    McpClient, McpProxyTool, McpServerConfig, McpToolInfo, McpToolResult, McpTransport,
};
pub use crate::permission::{
    AlwaysApprove, AlwaysReject, ApprovalContext, ApprovalHandler, Decision, Policy, PolicyBuilder,
};
pub use crate::tool::{Registry, Tool, ToolExecute, ToolResult};
