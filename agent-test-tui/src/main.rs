// ABOUTME: Simple REPL for testing mux-rs with Claude and MCP tools.
// ABOUTME: Reads .mcp.json, connects to servers, runs agentic loop.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use rustyline::DefaultEditor;
use serde::Deserialize;

use mux::prelude::*;

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

        let _ = rl.add_history_entry(line);
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
                        std::io::stdout().flush()?;

                        let mut answer = String::new();
                        std::io::stdin().read_line(&mut answer)?;

                        if answer.trim().to_lowercase() == "y" {
                            println!("[Executing {}...]", name);
                            match registry.get(name).await {
                                Some(tool) => match tool.execute(input.clone()).await {
                                    Ok(result) => {
                                        println!("Result: {}\n", result.content);
                                        if result.is_error {
                                            tool_results.push(ContentBlock::tool_error(
                                                id,
                                                &result.content,
                                            ));
                                        } else {
                                            tool_results.push(ContentBlock::tool_result(
                                                id,
                                                &result.content,
                                            ));
                                        }
                                    }
                                    Err(e) => {
                                        let error = format!("Error: {}", e);
                                        println!("{}\n", error);
                                        tool_results.push(ContentBlock::tool_error(id, error));
                                    }
                                },
                                None => {
                                    let error = format!("Tool not found: {}", name);
                                    println!("{}\n", error);
                                    tool_results.push(ContentBlock::tool_error(id, error));
                                }
                            }
                        } else {
                            println!("[Denied]\n");
                            tool_results
                                .push(ContentBlock::tool_error(id, "Tool call denied by user"));
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

    // Run agent loop, ensuring cleanup runs even on error
    let result = run_agent_loop(&registry).await;

    // Cleanup MCP clients (always runs)
    for client in mcp_clients {
        let _ = client.shutdown().await;
    }

    result
}
