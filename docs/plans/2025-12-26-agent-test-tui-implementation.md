# agent-test-tui Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a simple REPL for testing mux-rs with Claude, tools, and MCP servers.

**Architecture:** First implement McpClient (stdio transport), then build the TUI binary that uses it.

**Tech Stack:** tokio (process spawning, async IO), rustyline (readline), serde_json

---

## Part 1: McpClient Implementation

### Task 1: Add McpClient struct with stdio transport

**Files:**
- Create: `src/mcp/client.rs`
- Modify: `src/mcp/mod.rs`

**Step 1: Create the client module**

```rust
// ABOUTME: MCP client for connecting to MCP servers via stdio.
// ABOUTME: Spawns server process and communicates via JSON-RPC.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex};

use super::{McpRequest, McpResponse, McpServerConfig, McpToolInfo, McpTransport};
use crate::error::McpError;

/// Client for communicating with an MCP server.
pub struct McpClient {
    config: McpServerConfig,
    child: Mutex<Option<Child>>,
    stdin: Mutex<Option<tokio::process::ChildStdin>>,
    pending: Arc<Mutex<HashMap<u64, mpsc::Sender<McpResponse>>>>,
    next_id: AtomicU64,
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

        let stdin = child.stdin.take().ok_or_else(|| {
            McpError::Connection("Failed to open stdin".into())
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            McpError::Connection("Failed to open stdout".into())
        })?;

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
            next_id: AtomicU64::new(1),
        })
    }

    /// Send a request and wait for response.
    async fn request(&self, method: &str, params: Option<serde_json::Value>) -> Result<McpResponse, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let request = McpRequest::new(id, method, params);

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

        rx.recv().await.ok_or(McpError::Protocol("No response received".into()))
    }

    /// Get the server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }
}
```

**Step 2: Update mod.rs**

```rust
mod client;
pub use client::McpClient;
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(mcp): add McpClient with stdio transport"
```

---

### Task 2: Add initialize and list_tools methods

**Files:**
- Modify: `src/mcp/client.rs`

**Step 1: Add the methods**

```rust
impl McpClient {
    // ... existing code ...

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
        let notification = McpRequest::notification("notifications/initialized");
        let json = serde_json::to_string(&notification)?;
        {
            let mut stdin = self.stdin.lock().await;
            if let Some(ref mut stdin) = *stdin {
                stdin.write_all(json.as_bytes()).await?;
                stdin.write_all(b"\n").await?;
                stdin.flush().await?;
            }
        }

        Ok(())
    }

    /// List available tools from the server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        let response = self.request("tools/list", None).await?;

        if let Some(error) = response.error {
            return Err(McpError::Rpc {
                code: error.code,
                message: error.message,
            });
        }

        let result = response.result.ok_or_else(|| {
            McpError::Protocol("No result in response".into())
        })?;

        let tools: Vec<McpToolInfo> = serde_json::from_value(result["tools"].clone())?;
        Ok(tools)
    }

    /// Call a tool on the server.
    pub async fn call_tool(&self, name: &str, arguments: serde_json::Value) -> Result<super::McpToolResult, McpError> {
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

        let result = response.result.ok_or_else(|| {
            McpError::Protocol("No result in response".into())
        })?;

        Ok(serde_json::from_value(result)?)
    }

    /// Shutdown the server connection.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        if let Some(mut child) = self.child.lock().await.take() {
            let _ = child.kill().await;
        }
        Ok(())
    }
}
```

**Step 2: Add notification helper to McpRequest**

In `src/mcp/types.rs`, add:

```rust
impl McpRequest {
    /// Create a notification (no id, no response expected).
    pub fn notification(method: &str) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: 0, // Will be skipped in serialization for notifications
            method: method.to_string(),
            params: None,
        }
    }
}
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(mcp): add initialize, list_tools, call_tool methods"
```

---

### Task 3: Add McpProxyTool

**Files:**
- Create: `src/mcp/proxy.rs`
- Modify: `src/mcp/mod.rs`

**Step 1: Create the proxy tool**

```rust
// ABOUTME: McpProxyTool wraps an MCP server tool for use in the registry.
// ABOUTME: Forwards tool calls to the MCP server.

use std::sync::Arc;

use async_trait::async_trait;

use super::{McpClient, McpToolInfo};
use crate::tool::{Tool, ToolResult};

/// A tool that proxies calls to an MCP server.
pub struct McpProxyTool {
    client: Arc<McpClient>,
    info: McpToolInfo,
    prefixed_name: String,
}

impl McpProxyTool {
    /// Create a new proxy tool.
    pub fn new(client: Arc<McpClient>, info: McpToolInfo, prefix: Option<&str>) -> Self {
        let prefixed_name = match prefix {
            Some(p) => format!("{}_{}", p, info.name),
            None => info.name.clone(),
        };
        Self {
            client,
            info,
            prefixed_name,
        }
    }
}

#[async_trait]
impl Tool for McpProxyTool {
    fn name(&self) -> &str {
        &self.prefixed_name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn schema(&self) -> serde_json::Value {
        self.info.input_schema.clone()
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let result = self.client.call_tool(&self.info.name, params).await?;

        // Convert MCP result to ToolResult
        let content = result.content
            .iter()
            .filter_map(|c| {
                if c.content_type == "text" {
                    c.text.clone()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Ok(ToolResult::error(content))
        } else {
            Ok(ToolResult::text(content))
        }
    }
}
```

**Step 2: Update mod.rs**

```rust
mod proxy;
pub use proxy::McpProxyTool;
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(mcp): add McpProxyTool for registry integration"
```

---

### Task 4: Add Registry::merge_mcp method

**Files:**
- Modify: `src/tool/registry.rs`

**Step 1: Add the merge method**

```rust
use crate::mcp::{McpClient, McpProxyTool};

impl Registry {
    // ... existing code ...

    /// Merge tools from an MCP client into the registry.
    pub async fn merge_mcp(
        &self,
        client: Arc<McpClient>,
        prefix: Option<&str>,
    ) -> Result<usize, crate::error::McpError> {
        let tools = client.list_tools().await?;
        let count = tools.len();

        for info in tools {
            let proxy = McpProxyTool::new(client.clone(), info, prefix);
            self.register(proxy).await;
        }

        Ok(count)
    }
}
```

**Step 2: Add Arc import**

```rust
use std::sync::Arc;
```

**Step 3: Run cargo check**

Run: `cargo check`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(tool): add Registry::merge_mcp for MCP integration"
```

---

### Task 5: Update prelude with MCP exports

**Files:**
- Modify: `src/prelude.rs`

**Step 1: Add MCP client exports**

```rust
pub use crate::mcp::{McpClient, McpProxyTool, McpServerConfig, McpToolInfo, McpToolResult, McpTransport};
```

**Step 2: Run tests**

Run: `cargo test`
Expected: All tests pass

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add MCP client to prelude"
```

---

## Part 2: agent-test-tui Binary

### Task 6: Create binary crate scaffold

**Files:**
- Modify: `Cargo.toml` (workspace)
- Create: `agent-test-tui/Cargo.toml`
- Create: `agent-test-tui/src/main.rs`

**Step 1: Update workspace Cargo.toml**

Add to members:
```toml
[workspace]
members = [".", "agent-test-tui"]
```

**Step 2: Create agent-test-tui/Cargo.toml**

```toml
[package]
name = "agent-test-tui"
version = "0.1.0"
edition = "2024"

[dependencies]
mux = { path = ".." }
tokio = { version = "1", features = ["full"] }
rustyline = "14"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
```

**Step 3: Create minimal main.rs**

```rust
// ABOUTME: Simple REPL for testing mux-rs with Claude and MCP tools.
// ABOUTME: Reads .mcp.json, connects to servers, runs agentic loop.

fn main() {
    println!("agent-test-tui placeholder");
}
```

**Step 4: Run cargo check**

Run: `cargo check --workspace`
Expected: Compiles

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: scaffold agent-test-tui binary crate"
```

---

### Task 7: Implement config loading

**Files:**
- Modify: `agent-test-tui/src/main.rs`

**Step 1: Add config types and loading**

```rust
// ABOUTME: Simple REPL for testing mux-rs with Claude and MCP tools.
// ABOUTME: Reads .mcp.json, connects to servers, runs agentic loop.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

use mux::mcp::{McpServerConfig, McpTransport};

#[derive(Debug, Deserialize)]
struct McpConfig {
    #[serde(rename = "mcpServers")]
    mcp_servers: HashMap<String, McpServerEntry>,
}

#[derive(Debug, Deserialize)]
struct McpServerEntry {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

fn find_config() -> Option<PathBuf> {
    // Try .mcp.json in current directory
    let local = PathBuf::from(".mcp.json");
    if local.exists() {
        return Some(local);
    }

    // Try ~/.mcp.json
    if let Some(home) = dirs::home_dir() {
        let global = home.join(".mcp.json");
        if global.exists() {
            return Some(global);
        }
    }

    None
}

fn load_config() -> Result<Vec<McpServerConfig>> {
    let Some(path) = find_config() else {
        return Ok(vec![]);
    };

    let content = std::fs::read_to_string(&path)?;
    let config: McpConfig = serde_json::from_str(&content)?;

    let servers = config
        .mcp_servers
        .into_iter()
        .map(|(name, entry)| McpServerConfig {
            name,
            transport: McpTransport::Stdio {
                command: entry.command,
                args: entry.args,
                env: entry.env,
            },
        })
        .collect();

    Ok(servers)
}

fn main() -> Result<()> {
    let configs = load_config()?;
    println!("Loaded {} MCP server configs", configs.len());
    for config in &configs {
        println!("  - {}", config.name);
    }
    Ok(())
}
```

**Step 2: Add dirs dependency**

Run: `cd agent-test-tui && cargo add dirs`

**Step 3: Run cargo check**

Run: `cargo check --workspace`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(tui): add .mcp.json config loading"
```

---

### Task 8: Implement MCP connection and registry setup

**Files:**
- Modify: `agent-test-tui/src/main.rs`

**Step 1: Add async main with MCP connections**

```rust
use std::sync::Arc;

use mux::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Load MCP configs
    let configs = load_config()?;

    // Create registry
    let registry = Registry::new();

    // Connect to MCP servers
    let mut mcp_clients = Vec::new();
    for config in configs {
        let name = config.name.clone();
        match McpClient::connect(config).await {
            Ok(client) => {
                if let Err(e) = client.initialize().await {
                    eprintln!("Warning: Failed to initialize {}: {}", name, e);
                    continue;
                }
                let client = Arc::new(client);
                match registry.merge_mcp(client.clone(), Some(&name)).await {
                    Ok(count) => {
                        println!("Connected to {} ({} tools)", name, count);
                        mcp_clients.push(client);
                    }
                    Err(e) => eprintln!("Warning: Failed to list tools from {}: {}", name, e),
                }
            }
            Err(e) => eprintln!("Warning: Failed to connect to {}: {}", name, e),
        }
    }

    let tool_count = registry.list().await.len();
    println!("\nReady with {} tools.", tool_count);

    // TODO: Run agent loop

    // Cleanup
    for client in mcp_clients {
        let _ = client.shutdown().await;
    }

    Ok(())
}
```

**Step 2: Run cargo check**

Run: `cargo check --workspace`
Expected: Compiles

**Step 3: Commit**

```bash
git add -A
git commit -m "feat(tui): add MCP server connection and registry setup"
```

---

### Task 9: Implement the agent loop

**Files:**
- Modify: `agent-test-tui/src/main.rs`

**Step 1: Add the agent loop**

```rust
use rustyline::DefaultEditor;

async fn run_agent_loop(registry: &Registry) -> Result<()> {
    let client = AnthropicClient::from_env()?;
    let mut history: Vec<Message> = Vec::new();
    let mut rl = DefaultEditor::new()?;

    println!("Type 'quit' to exit.\n");

    loop {
        let line = match rl.readline("> ") {
            Ok(line) => line,
            Err(_) => break,
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }

        rl.add_history_entry(line)?;
        history.push(Message::user(line));

        // Agent loop: keep calling Claude until no more tool calls
        loop {
            let request = Request::new("claude-sonnet-4-20250514")
                .messages(history.clone())
                .tools(registry.to_definitions().await)
                .max_tokens(4096);

            let response = client.create_message(&request).await?;

            // Check for tool calls
            if response.has_tool_use() {
                let mut tool_results = Vec::new();

                for block in response.content.iter() {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        println!("\nClaude wants to call: {}", name);
                        println!("Parameters: {}", serde_json::to_string_pretty(input)?);
                        print!("\nAllow? [y/n]: ");
                        std::io::Write::flush(&mut std::io::stdout())?;

                        let mut answer = String::new();
                        std::io::stdin().read_line(&mut answer)?;

                        if answer.trim().to_lowercase() == "y" {
                            println!("[Executing {}...]", name);
                            match registry.get(name).await {
                                Some(tool) => {
                                    match tool.execute(input.clone()).await {
                                        Ok(result) => {
                                            println!("Result: {}\n", result.content);
                                            if result.is_error {
                                                tool_results.push(ContentBlock::tool_error(id, &result.content));
                                            } else {
                                                tool_results.push(ContentBlock::tool_result(id, &result.content));
                                            }
                                        }
                                        Err(e) => {
                                            let error = format!("Error: {}", e);
                                            println!("{}\n", error);
                                            tool_results.push(ContentBlock::tool_error(id, error));
                                        }
                                    }
                                }
                                None => {
                                    let error = format!("Tool not found: {}", name);
                                    println!("{}\n", error);
                                    tool_results.push(ContentBlock::tool_error(id, error));
                                }
                            }
                        } else {
                            println!("[Denied]\n");
                            tool_results.push(ContentBlock::tool_error(id, "Tool call denied by user"));
                        }
                    }
                }

                // Add assistant message (with tool calls) and user message (with results) to history
                history.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });
                history.push(Message::tool_results(tool_results));

                // Continue the loop to call Claude again
                continue;
            }

            // No tool calls - print response and break inner loop
            let text = response.text();
            if !text.is_empty() {
                println!("\n{}\n", text);
            }
            history.push(Message {
                role: Role::Assistant,
                content: response.content,
            });
            break;
        }
    }

    Ok(())
}
```

**Step 2: Update main to call the loop**

Replace the TODO comment with:
```rust
run_agent_loop(&registry).await?;
```

**Step 3: Run cargo check**

Run: `cargo check --workspace`
Expected: Compiles

**Step 4: Commit**

```bash
git add -A
git commit -m "feat(tui): implement agent loop with tool approval"
```

---

### Task 10: Final testing and cleanup

**Files:**
- Various

**Step 1: Run all tests**

Run: `cargo test --workspace`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: No errors

**Step 3: Format**

Run: `cargo fmt --all`

**Step 4: Test the binary (manual)**

Run: `cargo run -p agent-test-tui`
Expected: Prints tool count and shows prompt

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: cleanup and format"
```
