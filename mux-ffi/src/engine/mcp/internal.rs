// ABOUTME: Internal async connection and tool-resolution helpers for MCP server management.
// ABOUTME: These methods back the public FFI API in api.rs but are not themselves exported.

use super::McpClientHandle;
use crate::engine::MuxEngine;
use crate::engine::helpers;
use crate::types::{McpServerConfig, McpTransportType};
use mux::prelude::{
    McpClient, McpServerConfig as MuxMcpServerConfig, McpTransport, Tool, ToolDefinition,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// MCP client management methods
impl MuxEngine {
    /// Acquire the per-workspace MCP lifecycle lock.
    ///
    /// Connect and disconnect operations for the same workspace serialize
    /// through this lock so they cannot interleave. Different workspaces
    /// remain independent.
    async fn lifecycle_guard(&self, workspace_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let mutex = {
            let mut guards = self.workspace_lifecycle.write();
            guards
                .entry(workspace_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
                .clone()
        };
        mutex.lock_owned().await
    }

    /// Shut down and remove every connected MCP client for the workspace.
    /// Caller must already hold the `lifecycle_guard` for this workspace.
    async fn shutdown_workspace_clients(&self, workspace_id: &str) {
        let clients = self.mcp_clients.write().remove(workspace_id);

        if let Some(clients) = clients {
            for (name, handle) in clients {
                let client = handle.client.lock().await;
                if let Err(e) = client.shutdown().await {
                    eprintln!("Error shutting down MCP server '{}': {}", name, e);
                }
            }
        }
    }

    /// Connect to all enabled MCP servers for a workspace.
    ///
    /// Idempotent: if the workspace already has live MCP clients, they are
    /// shut down before reconnecting. This prevents orphaned stdio child
    /// processes / SSE sessions when the same workspace is reconnected
    /// (e.g. after a config change). `McpClientHandle` has no `Drop` that
    /// triggers `shutdown()`, so we must do this explicitly.
    pub(super) async fn do_connect_workspace_servers(
        &self,
        workspace_id: String,
    ) -> Result<(), String> {
        let _guard = self.lifecycle_guard(&workspace_id).await;

        // Get enabled MCP server configs for this workspace
        let server_configs: Vec<McpServerConfig> = {
            let workspaces = self.workspaces.read();
            let workspace = workspaces
                .get(&workspace_id)
                .ok_or_else(|| format!("Workspace not found: {}", workspace_id))?;
            workspace
                .mcp_servers
                .iter()
                .filter(|s| s.enabled)
                .cloned()
                .collect()
        };

        // Always tear down any prior connections for this workspace before
        // (re)connecting, so the active set reflects the current config.
        // (Use the unlocked helper — we already hold the lifecycle guard.)
        self.shutdown_workspace_clients(&workspace_id).await;

        if server_configs.is_empty() {
            return Ok(());
        }

        let mut workspace_clients: HashMap<String, McpClientHandle> = HashMap::new();

        for config in server_configs {
            match self.connect_single_server(&config).await {
                Ok(handle) => {
                    eprintln!(
                        "Connected to MCP server '{}' with {} tools, {} resources, {} prompts",
                        config.name,
                        handle.tools.len(),
                        handle.resources.len(),
                        handle.prompts.len()
                    );
                    workspace_clients.insert(config.name.clone(), handle);
                }
                Err(e) => {
                    eprintln!("Failed to connect to MCP server '{}': {}", config.name, e);
                }
            }
        }

        // Store the connected clients
        self.mcp_clients
            .write()
            .insert(workspace_id, workspace_clients);

        Ok(())
    }

    /// Connect to a single MCP server.
    async fn connect_single_server(
        &self,
        config: &McpServerConfig,
    ) -> Result<McpClientHandle, String> {
        // Convert FFI config to mux config
        let transport = match config.transport_type {
            McpTransportType::Stdio => {
                let command = config
                    .command
                    .as_ref()
                    .ok_or_else(|| "Stdio transport requires command".to_string())?;
                McpTransport::Stdio {
                    command: command.clone(),
                    args: config.args.clone(),
                    env: HashMap::new(),
                }
            }
            McpTransportType::Sse => {
                let url = config
                    .url
                    .as_ref()
                    .ok_or_else(|| "SSE transport requires URL".to_string())?;
                McpTransport::Sse { url: url.clone() }
            }
        };

        let mux_config = MuxMcpServerConfig {
            name: config.name.clone(),
            transport,
        };

        // Connect and initialize
        let mut client = McpClient::connect(mux_config)
            .await
            .map_err(|e| e.to_string())?;

        client.initialize().await.map_err(|e| e.to_string())?;

        // Fetch available tools
        let tools = client.list_tools().await.map_err(|e| e.to_string())?;

        // Pagination safety limit to prevent infinite loops from buggy servers
        const MAX_PAGES: usize = 100;

        // Fetch resources (with pagination)
        let mut resources = Vec::new();
        let mut cursor: Option<String> = None;
        let mut pages = 0;
        loop {
            if pages >= MAX_PAGES {
                eprintln!(
                    "Warning: Hit pagination limit for resources on server {}",
                    config.name
                );
                break;
            }
            let result = client
                .list_resources(cursor.as_deref())
                .await
                .map_err(|e| e.to_string())?;
            resources.extend(result.resources);
            cursor = result.next_cursor;
            pages += 1;
            if cursor.is_none() {
                break;
            }
        }

        // Fetch resource templates (with pagination)
        let mut resource_templates = Vec::new();
        cursor = None;
        pages = 0;
        loop {
            if pages >= MAX_PAGES {
                eprintln!(
                    "Warning: Hit pagination limit for resource templates on server {}",
                    config.name
                );
                break;
            }
            let result = client
                .list_resource_templates(cursor.as_deref())
                .await
                .map_err(|e| e.to_string())?;
            resource_templates.extend(result.resource_templates);
            cursor = result.next_cursor;
            pages += 1;
            if cursor.is_none() {
                break;
            }
        }

        // Fetch prompts (with pagination)
        let mut prompts = Vec::new();
        cursor = None;
        pages = 0;
        loop {
            if pages >= MAX_PAGES {
                eprintln!(
                    "Warning: Hit pagination limit for prompts on server {}",
                    config.name
                );
                break;
            }
            let result = client
                .list_prompts(cursor.as_deref())
                .await
                .map_err(|e| e.to_string())?;
            prompts.extend(result.prompts);
            cursor = result.next_cursor;
            pages += 1;
            if cursor.is_none() {
                break;
            }
        }

        Ok(McpClientHandle {
            client: Arc::new(TokioMutex::new(client)),
            tools,
            resources,
            resource_templates,
            prompts,
            server_name: config.name.clone(),
        })
    }

    /// Disconnect all MCP servers for a workspace.
    pub(super) async fn do_disconnect_workspace_servers(&self, workspace_id: &str) {
        let _guard = self.lifecycle_guard(workspace_id).await;
        self.shutdown_workspace_clients(workspace_id).await;
    }

    /// Get all tools available for a workspace as ToolDefinitions for the LLM.
    /// Includes built-in mux tools, custom tools, and any connected MCP server tools.
    // Catalog/inspection API kept alongside the executable Registry built by
    // `messaging::build_tool_registry`. Currently consumed only by tests;
    // retained as a candidate FFI surface for hosts that want to enumerate
    // available tools without running a chat turn.
    #[allow(dead_code)]
    pub(super) fn get_workspace_tools(&self, workspace_id: &str) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();

        // Add built-in tools (always available, no prefix)
        for tool in &self.builtin_tools {
            tools.push(ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.schema(),
                cache_control: None,
            });
        }

        // Add custom tools registered from Swift
        {
            let custom_tools = self.custom_tools.read();
            for tool in custom_tools.values() {
                tools.push(ToolDefinition {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    input_schema: tool.schema(),
                    cache_control: None,
                });
            }
        }

        // Add TaskTool if subagent event handler is registered
        if self.subagent_event_handler.read().is_some() {
            tools.push(ToolDefinition {
                name: "task".to_string(),
                description: "Spawn a subagent to handle a specific task. Use a registered agent_type OR provide a custom system_prompt for ad-hoc agents.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "agent_type": {
                            "type": "string",
                            "description": "The type of agent to spawn (must be registered). Mutually exclusive with system_prompt."
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "Custom system prompt for an ad-hoc agent. Use this instead of agent_type for one-off specialized tasks."
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use. REQUIRED for ad-hoc agents (system_prompt). For registered agents, uses the model from AgentConfig."
                        },
                        "task": {
                            "type": "string",
                            "description": "The task description to give to the subagent"
                        },
                        "description": {
                            "type": "string",
                            "description": "A short (3-5 word) description of what the agent will do"
                        },
                        "resume_agent_id": {
                            "type": "string",
                            "description": "Optional: ID of a previous agent to resume from its transcript"
                        }
                    },
                    "required": ["task", "description"]
                }),
                cache_control: None,
            });
        }

        // Add MCP tools (prefixed with server name)
        let clients = self.mcp_clients.read();
        if let Some(workspace_clients) = clients.get(workspace_id) {
            for handle in workspace_clients.values() {
                for mcp_tool in &handle.tools {
                    tools.push(ToolDefinition {
                        name: format!("{}:{}", handle.server_name, mcp_tool.name),
                        description: mcp_tool.description.clone(),
                        input_schema: mcp_tool.input_schema.clone(),
                        cache_control: None,
                    });
                }
            }
        }

        tools
    }

    /// Find the MCP client and tool name for a qualified tool name (server:tool).
    // Tool dispatch goes through the executable Registry (keyed by full
    // qualified name), so this helper is unused in production today. Kept
    // alongside `get_workspace_tools` as a candidate FFI surface for hosts
    // that want to introspect or route tool calls directly.
    #[allow(dead_code)]
    pub(super) fn parse_tool_name(&self, qualified_name: &str) -> Option<(String, String)> {
        helpers::parse_qualified_tool_name(qualified_name)
    }
}
