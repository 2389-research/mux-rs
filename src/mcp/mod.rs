// ABOUTME: MCP module - Model Context Protocol client implementation.
// ABOUTME: Connects to MCP servers via stdio or SSE and proxies their tools.

mod client;
mod proxy;
mod transport;
mod types;

pub use client::McpClient;
pub use proxy::McpProxyTool;
pub use transport::{HttpTransport, SseTransport, StdioTransport, Transport};
pub use types::*;

#[cfg(test)]
mod types_test;
