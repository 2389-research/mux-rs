// ABOUTME: MCP (Model Context Protocol) integration for MuxEngine: shared handle type and module wiring.
// ABOUTME: Public FFI operations live in api.rs; internal async connection/tool helpers in internal.rs.

use mux::mcp::{
    McpPromptInfo as MuxMcpPromptInfo, McpResourceInfo as MuxMcpResourceInfo,
    McpResourceTemplate as MuxMcpResourceTemplate,
};
use mux::prelude::{McpClient, McpToolInfo};
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

#[cfg(test)]
use crate::types::{ApprovalDecision, McpTransportType};

/// Holds a connected MCP client and its cached capabilities.
pub(super) struct McpClientHandle {
    pub client: Arc<TokioMutex<McpClient>>,
    pub tools: Vec<McpToolInfo>,
    pub resources: Vec<MuxMcpResourceInfo>,
    pub resource_templates: Vec<MuxMcpResourceTemplate>,
    pub prompts: Vec<MuxMcpPromptInfo>,
    pub server_name: String,
}

mod api;
mod internal;

#[cfg(test)]
#[path = "mcp_test.rs"]
mod tests;
