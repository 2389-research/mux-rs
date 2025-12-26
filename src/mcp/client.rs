// ABOUTME: MCP client for connecting to MCP servers via stdio.
// ABOUTME: Spawns server process and communicates via JSON-RPC.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex};

use super::{McpRequest, McpResponse, McpServerConfig, McpTransport};
use crate::error::McpError;

/// Client for communicating with an MCP server.
pub struct McpClient {
    config: McpServerConfig,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>>,
}

impl McpClient {
    /// Connect to an MCP server.
    pub async fn connect(config: McpServerConfig) -> Result<Self, McpError> {
        let McpTransport::Stdio { command, args, env } = &config.transport else {
            return Err(McpError::Connection("Only stdio transport supported".into()));
        };

        let mut cmd = Command::new(command);
        cmd.args(args)
            .envs(env.iter())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        let mut child = cmd.spawn().map_err(|e| McpError::Connection(e.to_string()))?;

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
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(response) = serde_json::from_str::<McpResponse>(&line) {
                    let mut pending = pending_clone.lock().await;
                    if let Some(tx) = pending.remove(&response.id) {
                        let _ = tx.send(response).await;
                    }
                }
            }
        });

        Ok(Self {
            config,
            child: Mutex::new(Some(child)),
            stdin: Mutex::new(Some(stdin)),
            pending,
        })
    }

    /// Send a request and wait for response.
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

        let json = serde_json::to_string(&request)?;
        {
            let mut stdin = self.stdin.lock().await;
            if let Some(ref mut stdin) = *stdin {
                stdin.write_all(json.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await?;
            }
        }

        rx.recv()
            .await
            .ok_or(McpError::Protocol("No response received".into()))
    }

    /// Get the server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }
}
