// ABOUTME: Transport abstraction for MCP communication.
// ABOUTME: Re-exports Stdio, SSE, and HTTP transport implementations.

mod http;
mod sse;
mod stdio;

pub use http::HttpTransport;
pub use sse::SseTransport;
pub use stdio::StdioTransport;

use async_trait::async_trait;

use super::{McpNotification, McpRequest, McpResponse};
use crate::error::McpError;

/// Trait for MCP transport implementations.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a request and receive a response.
    async fn send(&self, request: McpRequest) -> Result<McpResponse, McpError>;

    /// Send a notification (no response expected).
    async fn notify(&self, notification: McpNotification) -> Result<(), McpError>;

    /// Shutdown the transport.
    async fn shutdown(&self) -> Result<(), McpError>;
}
