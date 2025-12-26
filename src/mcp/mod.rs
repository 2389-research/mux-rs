// ABOUTME: MCP module - Model Context Protocol client implementation.
// ABOUTME: Connects to MCP servers and proxies their tools.

mod client;
mod proxy;
mod types;

pub use client::McpClient;
pub use proxy::McpProxyTool;
pub use types::*;

#[cfg(test)]
mod types_test;
