// ABOUTME: Prelude module - convenient imports for common use cases.
// ABOUTME: Use `use mux::prelude::*;` to get started quickly.

pub use crate::agent::{
    AgentDefinition, AgentRegistry, FilteredRegistry, SubAgent, SubAgentResult, TaskTool,
};
pub use crate::error::{LlmError, McpError, MuxError, PermissionError, ToolError};
pub use crate::llm::{
    AnthropicClient, ContentBlock, LlmClient, Message, OpenAIClient, Request, Response, Role,
    StopReason, StreamEvent, ToolDefinition, Usage,
};
pub use crate::mcp::{
    McpClient, McpContentBlock, McpLogLevel, McpPromptGetResult, McpPromptInfo,
    McpPromptsListResult, McpProxyTool, McpResourceContent, McpResourceInfo,
    McpResourcesListResult, McpRoot, McpSamplingParams, McpSamplingResult, McpServerCapabilities,
    McpServerConfig, McpToolInfo, McpToolResult, McpTransport, SseTransport, StdioTransport,
    Transport,
};
pub use crate::permission::{
    AlwaysApprove, AlwaysReject, ApprovalContext, ApprovalHandler, Decision, Policy, PolicyBuilder,
};
pub use crate::tool::{Registry, Tool, ToolExecute, ToolResult};
pub use crate::tools::{
    BashTool, ListFilesTool, ReadFileTool, SearchResult, SearchTool, WebFetchTool, WebSearchTool,
    WriteFileTool,
};
