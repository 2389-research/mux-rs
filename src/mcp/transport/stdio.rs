// ABOUTME: Stdio transport for MCP communication.
// ABOUTME: Spawns a subprocess and communicates via JSON-RPC over stdin/stdout.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;

use super::Transport;
use crate::error::McpError;
use crate::mcp::{McpNotification, McpRequest, McpResponse};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_nonexistent_binary() {
        let result = StdioTransport::connect("/nonexistent/binary", &[], &HashMap::new()).await;

        match result {
            Err(McpError::Connection(_)) => {}
            _ => panic!("Expected McpError::Connection"),
        }
    }

    /// Returns the appropriate echo-back command for the current platform.
    /// On Unix, `cat` reads stdin and echoes it back. On Windows, `findstr "^"` does the same.
    fn echo_command() -> &'static str {
        if cfg!(target_os = "windows") {
            "findstr"
        } else {
            "cat"
        }
    }

    fn echo_args() -> Vec<String> {
        if cfg!(target_os = "windows") {
            vec!["^".to_string()]
        } else {
            vec![]
        }
    }

    #[tokio::test]
    async fn test_connect_with_args() {
        let result =
            StdioTransport::connect(echo_command(), &echo_args(), &HashMap::new()).await;

        // Should succeed in spawning
        assert!(result.is_ok());

        // Clean up
        let transport = result.unwrap();
        let _ = transport.shutdown().await;
    }

    #[tokio::test]
    async fn test_shutdown_cleans_up() {
        let transport =
            StdioTransport::connect(echo_command(), &echo_args(), &HashMap::new())
                .await
                .unwrap();

        // Shutdown should succeed
        let result = transport.shutdown().await;
        assert!(result.is_ok());

        // Double shutdown should also be fine
        let result = transport.shutdown().await;
        assert!(result.is_ok());
    }
}
