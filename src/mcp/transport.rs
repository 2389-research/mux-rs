// ABOUTME: Transport abstraction for MCP communication.
// ABOUTME: Supports stdio (subprocess), SSE, and HTTP transports.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

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

/// Stdio transport - spawns a subprocess and communicates via JSON-RPC over stdin/stdout.
pub struct StdioTransport {
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>>,
    reader_handle: Mutex<Option<JoinHandle<()>>>,
}

impl StdioTransport {
    /// Create a new stdio transport by spawning a subprocess.
    pub async fn connect(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(env.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Connection(e.to_string()))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Connection("Failed to open stdin".into()))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Connection("Failed to open stdout".into()))?;

        let pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Spawn reader task
        let pending_clone = pending.clone();
        let reader_handle = tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            loop {
                match reader.next_line().await {
                    Ok(Some(line)) => {
                        if let Ok(response) = serde_json::from_str::<McpResponse>(&line) {
                            let mut pending = pending_clone.lock().await;
                            if let Some(tx) = pending.remove(&response.id) {
                                let _ = tx.send(response).await;
                            }
                        }
                    }
                    Ok(None) => break,
                    Err(_) => break,
                }
            }
            pending_clone.lock().await.clear();
        });

        Ok(Self {
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(Some(stdin)),
            pending,
            reader_handle: Mutex::new(Some(reader_handle)),
        })
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        let id = request.id;

        let (tx, mut rx) = mpsc::channel(1);
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        // Write request
        let write_result = async {
            let json = serde_json::to_string(&request)?;
            let mut stdin = self.stdin.lock().await;
            let stdin_ref = stdin
                .as_mut()
                .ok_or_else(|| McpError::Connection("Server connection closed".into()))?;
            stdin_ref.write_all(json.as_bytes()).await?;
            stdin_ref.write_all(b"\n").await?;
            stdin_ref.flush().await?;
            Ok::<_, McpError>(())
        }
        .await;

        if let Err(e) = write_result {
            self.pending.lock().await.remove(&id);
            return Err(e);
        }

        // Wait for response with timeout
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await {
            Ok(Some(response)) => Ok(response),
            Ok(None) => Err(McpError::Protocol("No response received".into())),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::Protocol("Request timed out".into()))
            }
        }
    }

    async fn notify(&self, notification: McpNotification) -> Result<(), McpError> {
        let json = serde_json::to_string(&notification)?;
        let mut stdin = self.stdin.lock().await;
        let stdin_ref = stdin
            .as_mut()
            .ok_or_else(|| McpError::Connection("Server connection closed".into()))?;
        stdin_ref.write_all(json.as_bytes()).await?;
        stdin_ref.write_all(b"\n").await?;
        stdin_ref.flush().await?;
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), McpError> {
        self.stdin.lock().await.take();

        if let Some(handle) = self.reader_handle.lock().await.take() {
            handle.abort();
        }

        if let Some(mut child) = self.child.lock().await.take() {
            match tokio::time::timeout(std::time::Duration::from_millis(500), child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    let _ = child.kill().await;
                }
            }
        }

        Ok(())
    }
}

/// SSE transport - connects to HTTP endpoints using Server-Sent Events.
///
/// MCP over SSE uses:
/// - GET with Accept: text/event-stream for server->client messages
/// - POST for client->server messages
pub struct SseTransport {
    endpoint_url: String,
    messages_url: Mutex<Option<String>>,
    http_client: reqwest::Client,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>>,
    sse_handle: Mutex<Option<JoinHandle<()>>>,
    shutdown_tx: Mutex<Option<mpsc::Sender<()>>>,
}

impl SseTransport {
    /// Connect to an SSE MCP server.
    pub async fn connect(url: &str) -> Result<Self, McpError> {
        let http_client = reqwest::Client::new();

        let pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let (messages_tx, mut messages_rx) = mpsc::channel::<String>(1);

        // Start SSE connection
        let sse_url = url.to_string();
        let pending_clone = pending.clone();
        let client_clone = http_client.clone();

        let sse_handle = tokio::spawn(async move {
            let response = match client_clone
                .get(&sse_url)
                .header("Accept", "text/event-stream")
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(_) => return,
            };

            let mut stream = response.bytes_stream();

            // Buffer for SSE parsing
            let mut buffer = String::new();
            let mut event_type = String::new();
            let mut event_data = String::new();

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    chunk = stream.next() => {
                        match chunk {
                            Some(Ok(bytes)) => {
                                buffer.push_str(&String::from_utf8_lossy(&bytes));

                                // Process complete lines
                                while let Some(pos) = buffer.find('\n') {
                                    let line = buffer[..pos].trim_end_matches('\r').to_string();
                                    buffer = buffer[pos + 1..].to_string();

                                    if line.is_empty() {
                                        // End of event
                                        if event_type == "endpoint" {
                                            // Server is telling us where to POST messages
                                            let _ = messages_tx.send(event_data.clone()).await;
                                        } else if event_type == "message" {
                                            // Parse JSON-RPC response
                                            if let Ok(response) = serde_json::from_str::<McpResponse>(&event_data) {
                                                let mut pending = pending_clone.lock().await;
                                                if let Some(tx) = pending.remove(&response.id) {
                                                    let _ = tx.send(response).await;
                                                }
                                            }
                                        }
                                        event_type.clear();
                                        event_data.clear();
                                    } else if let Some(data) = line.strip_prefix("event: ") {
                                        event_type = data.to_string();
                                    } else if let Some(data) = line.strip_prefix("data: ") {
                                        if !event_data.is_empty() {
                                            event_data.push('\n');
                                        }
                                        event_data.push_str(data);
                                    }
                                }
                            }
                            Some(Err(_)) | None => break,
                        }
                    }
                }
            }
        });

        // Wait for endpoint message with timeout
        let messages_url = tokio::time::timeout(std::time::Duration::from_secs(10), messages_rx.recv())
            .await
            .map_err(|_| McpError::Connection("Timeout waiting for endpoint message".into()))?
            .ok_or_else(|| McpError::Connection("No endpoint message received".into()))?;

        // If the endpoint is relative, resolve it against the base URL
        let full_messages_url = if messages_url.starts_with("http") {
            messages_url
        } else {
            // Parse base URL and resolve
            let base = reqwest::Url::parse(url)
                .map_err(|e| McpError::Connection(format!("Invalid base URL: {}", e)))?;
            base.join(&messages_url)
                .map_err(|e| McpError::Connection(format!("Failed to resolve endpoint: {}", e)))?
                .to_string()
        };

        Ok(Self {
            endpoint_url: url.to_string(),
            messages_url: Mutex::new(Some(full_messages_url)),
            http_client,
            pending,
            sse_handle: Mutex::new(Some(sse_handle)),
            shutdown_tx: Mutex::new(Some(shutdown_tx)),
        })
    }

    /// Get the endpoint URL.
    pub fn endpoint_url(&self) -> &str {
        &self.endpoint_url
    }
}

#[async_trait]
impl Transport for SseTransport {
    async fn send(&self, request: McpRequest) -> Result<McpResponse, McpError> {
        let id = request.id;

        let (tx, mut rx) = mpsc::channel(1);
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        // Get the messages URL
        let messages_url = {
            let guard = self.messages_url.lock().await;
            guard
                .as_ref()
                .ok_or_else(|| McpError::Connection("No messages endpoint".into()))?
                .clone()
        };

        // Send request via POST
        let json = serde_json::to_string(&request)?;
        let post_result = self
            .http_client
            .post(&messages_url)
            .header("Content-Type", "application/json")
            .body(json)
            .send()
            .await;

        if let Err(e) = post_result {
            self.pending.lock().await.remove(&id);
            return Err(McpError::Connection(format!("POST failed: {}", e)));
        }

        // Wait for response via SSE
        match tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await {
            Ok(Some(response)) => Ok(response),
            Ok(None) => Err(McpError::Protocol("No response received".into())),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::Protocol("Request timed out".into()))
            }
        }
    }

    async fn notify(&self, notification: McpNotification) -> Result<(), McpError> {
        let messages_url = {
            let guard = self.messages_url.lock().await;
            guard
                .as_ref()
                .ok_or_else(|| McpError::Connection("No messages endpoint".into()))?
                .clone()
        };

        let json = serde_json::to_string(&notification)?;
        self.http_client
            .post(&messages_url)
            .header("Content-Type", "application/json")
            .body(json)
            .send()
            .await
            .map_err(|e| McpError::Connection(format!("POST failed: {}", e)))?;

        Ok(())
    }

    async fn shutdown(&self) -> Result<(), McpError> {
        // Signal SSE task to stop
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(()).await;
        }

        // Abort SSE handle
        if let Some(handle) = self.sse_handle.lock().await.take() {
            handle.abort();
        }

        Ok(())
    }
}

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
    async fn test_stdio_transport_connect_nonexistent() {
        let result = StdioTransport::connect("/nonexistent/binary", &[], &HashMap::new()).await;

        match result {
            Err(McpError::Connection(_)) => {}
            _ => panic!("Expected McpError::Connection"),
        }
    }

    #[tokio::test]
    async fn test_sse_transport_connect_invalid_url() {
        let result = SseTransport::connect("http://localhost:99999/nonexistent").await;

        // Should fail to connect
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_http_transport_connect_valid_url() {
        // Should succeed - just validates URL, doesn't actually connect
        let result = HttpTransport::connect("http://localhost:8080/mcp").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_http_transport_connect_invalid_url() {
        let result = HttpTransport::connect("not-a-valid-url").await;
        assert!(result.is_err());
    }
}
