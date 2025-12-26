// ABOUTME: MCP module - Model Context Protocol client implementation.
// ABOUTME: Connects to MCP servers and proxies their tools.

mod client;
mod types;

pub use client::McpClient;
pub use types::*;

#[cfg(test)]
mod types_test;
