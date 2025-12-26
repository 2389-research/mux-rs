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
