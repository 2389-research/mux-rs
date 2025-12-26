// ABOUTME: MCP client for connecting to MCP servers via stdio.
// ABOUTME: Spawns server process and communicates via JSON-RPC.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use super::{McpRequest, McpResponse, McpServerConfig, McpTransport};
use crate::error::McpError;

/// Default timeout for MCP requests in seconds.
const REQUEST_TIMEOUT_SECS: u64 = 30;

/// Client for communicating with an MCP server.
pub struct McpClient {
    config: McpServerConfig,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>>,
    reader_handle: Mutex<Option<JoinHandle<()>>>,
}

impl McpClient {
    /// Connect to an MCP server.
    pub async fn connect(config: McpServerConfig) -> Result<Self, McpError> {
        let McpTransport::Stdio { command, args, env } = &config.transport else {
            return Err(McpError::Connection(
                "Only stdio transport supported".into(),
            ));
        };

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
                        // Notifications and unparseable messages are silently ignored
                    }
                    Ok(None) => break, // EOF
                    Err(_) => break,   // I/O error
                }
            }
            // Clean up any pending requests on disconnect - dropping senders
            // will cause receivers to return None
            pending_clone.lock().await.clear();
        });

        Ok(Self {
            config,
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(Some(stdin)),
            pending,
            reader_handle: Mutex::new(Some(reader_handle)),
        })
    }

    /// Send a request and wait for response with timeout.
    pub async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<McpResponse, McpError> {
        let request = McpRequest::new(method, params);
        let id = request.id;

        let (tx, mut rx) = mpsc::channel(1);
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        // Write request to stdin, cleaning up pending on failure
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
        match tokio::time::timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS), rx.recv()).await {
            Ok(Some(response)) => Ok(response),
            Ok(None) => Err(McpError::Protocol("No response received".into())),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(McpError::Protocol("Request timed out".into()))
            }
        }
    }

    /// Get the server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Initialize the MCP connection.
    pub async fn initialize(&self) -> Result<(), McpError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "mux-rs",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let response = self.request("initialize", Some(params)).await?;

        if let Some(error) = response.error {
            return Err(McpError::Rpc {
                code: error.code,
                message: error.message,
            });
        }

        // Send initialized notification (no response expected)
        let notification = super::McpNotification::new("notifications/initialized", None);
        let json = serde_json::to_string(&notification)?;
        {
            let mut stdin = self.stdin.lock().await;
            let stdin_ref = stdin
                .as_mut()
                .ok_or_else(|| McpError::Connection("Server connection closed".into()))?;
            stdin_ref.write_all(json.as_bytes()).await?;
            stdin_ref.write_all(b"\n").await?;
            stdin_ref.flush().await?;
        }

        Ok(())
    }

    /// List available tools from the server.
    pub async fn list_tools(&self) -> Result<Vec<super::McpToolInfo>, McpError> {
        let response = self.request("tools/list", None).await?;

        if let Some(error) = response.error {
            return Err(McpError::Rpc {
                code: error.code,
                message: error.message,
            });
        }

        let result = response
            .result
            .ok_or_else(|| McpError::Protocol("No result in response".into()))?;

        let tools: Vec<super::McpToolInfo> = serde_json::from_value(result["tools"].clone())?;
        Ok(tools)
    }

    /// Call a tool on the server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<super::McpToolResult, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let response = self.request("tools/call", Some(params)).await?;

        if let Some(error) = response.error {
            return Err(McpError::Rpc {
                code: error.code,
                message: error.message,
            });
        }

        let result = response
            .result
            .ok_or_else(|| McpError::Protocol("No result in response".into()))?;

        Ok(serde_json::from_value(result)?)
    }

    /// Shutdown the server connection gracefully.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        // Close stdin to signal EOF to the server
        self.stdin.lock().await.take();

        // Abort the reader task
        if let Some(handle) = self.reader_handle.lock().await.take() {
            handle.abort();
        }

        // Kill the child process if still running
        if let Some(mut child) = self.child.lock().await.take() {
            // Give the process a moment to exit gracefully, then force kill
            match tokio::time::timeout(Duration::from_millis(500), child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    let _ = child.kill().await;
                }
            }
        }

        Ok(())
    }
}
