// ABOUTME: HTTP transport for MCP communication.
// ABOUTME: Simple request/response over HTTP with optional session management.

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::Transport;
use crate::error::McpError;
use crate::mcp::{McpNotification, McpRequest, McpResponse};

/// HTTP transport - simple request/response over HTTP.
///
/// MCP Streamable HTTP transport uses:
/// - POST with JSON-RPC request body
/// - JSON-RPC response in response body
/// - Optional streaming via chunked transfer encoding
pub struct HttpTransport {
    endpoint_url: String,
    http_client: reqwest::Client,
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    /// Connect to an HTTP MCP server.
    pub async fn connect(url: &str) -> Result<Self, McpError> {
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent(format!("mux-rs/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| McpError::Connection(format!("Failed to create HTTP client: {}", e)))?;

        // Validate URL format
        let _parsed = reqwest::Url::parse(url)
            .map_err(|e| McpError::Connection(format!("Invalid URL: {}", e)))?;

        Ok(Self {
            endpoint_url: url.to_string(),
            http_client,
            session_id: Mutex::new(None),
        })
    }

    /// Get the endpoint URL.
    #[allow(dead_code)]
    pub fn endpoint_url(&self) -> &str {
        &self.endpoint_url
    }

    /// Set session ID for stateful connections.
    #[allow(dead_code)]
    pub async fn set_session_id(&self, id: String) {
        *self.session_id.lock().await = Some(id);
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        let request_id = request.id;
        let json = serde_json::to_string(&request)?;

        let mut req_builder = self
            .http_client
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        // Add session ID header if present (for stateful servers)
        if let Some(session_id) = self.session_id.lock().await.as_ref() {
            req_builder = req_builder.header("Mcp-Session-Id", session_id.clone());
        }

        let response = req_builder
            .body(json)
            .send()
            .await
            .map_err(|e| McpError::Connection(format!("HTTP request failed: {}", e)))?;

        // Check for session ID in response (server may establish one)
        if let Some(session_id) = response.headers().get("Mcp-Session-Id") {
            if let Ok(id) = session_id.to_str() {
                *self.session_id.lock().await = Some(id.to_string());
            }
        }

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(McpError::Protocol(format!(
                "HTTP {} - {}",
                status.as_u16(),
                body
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| McpError::Protocol(format!("Failed to read response: {}", e)))?;

        let mcp_response: McpResponse = serde_json::from_str(&body)
            .map_err(|e| McpError::Protocol(format!("Invalid JSON-RPC response: {}", e)))?;

        // Validate response ID matches request ID
        if mcp_response.id != request_id {
            return Err(McpError::Protocol(format!(
                "Response ID {} does not match request ID {}",
                mcp_response.id, request_id
            )));
        }

        Ok(mcp_response)
    }

    async fn notify(&self, notification: McpNotification) -> Result<(), McpError> {
        let json = serde_json::to_string(&notification)?;

        let mut req_builder = self
            .http_client
            .post(&self.endpoint_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json");

        // Add session ID header if present
        if let Some(session_id) = self.session_id.lock().await.as_ref() {
            req_builder = req_builder.header("Mcp-Session-Id", session_id.clone());
        }

        let response = req_builder
            .body(json)
            .send()
            .await
            .map_err(|e| McpError::Connection(format!("HTTP request failed: {}", e)))?;

        // Check response status (notifications should still succeed)
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(McpError::Protocol(format!(
                "HTTP {} on notify - {}",
                status.as_u16(),
                body
            )));
        }

        Ok(())
    }

    async fn shutdown(&self) -> Result<(), McpError> {
        // HTTP is stateless - nothing to clean up
        // Clear session ID
        *self.session_id.lock().await = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_valid_url() {
        // Should succeed - just validates URL, doesn't actually connect
        let result = HttpTransport::connect("http://localhost:8080/mcp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_connect_invalid_url() {
        let result = HttpTransport::connect("not-a-valid-url").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_connect_https_url() {
        let result = HttpTransport::connect("https://api.example.com/mcp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_endpoint_url() {
        let transport = HttpTransport::connect("http://localhost:8080/mcp")
            .await
            .unwrap();
        assert_eq!(transport.endpoint_url(), "http://localhost:8080/mcp");
    }

    #[tokio::test]
    async fn test_session_id() {
        let transport = HttpTransport::connect("http://localhost:8080/mcp")
            .await
            .unwrap();

        // Initially no session ID
        assert!(transport.session_id.lock().await.is_none());

        // Set session ID
        transport.set_session_id("test-session-123".to_string()).await;
        assert_eq!(
            transport.session_id.lock().await.as_ref().unwrap(),
            "test-session-123"
        );

        // Shutdown clears session ID
        transport.shutdown().await.unwrap();
        assert!(transport.session_id.lock().await.is_none());
    }
}
