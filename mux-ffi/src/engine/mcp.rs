// ABOUTME: MCP (Model Context Protocol) server management for MuxEngine.
// ABOUTME: Handles connection, disconnection, and tool execution for MCP servers.

use super::MuxEngine;
use super::helpers;
use crate::MuxFfiError;
use crate::types::{
    ApprovalDecision, McpPromptArgument, McpPromptInfo, McpPromptMessage, McpPromptResult,
    McpResourceContent, McpResourceInfo, McpResourceTemplate, McpServerConfig, McpTransportType,
    PromptArgumentValue,
};
use mux::mcp::{
    McpPromptContent, McpPromptInfo as MuxMcpPromptInfo,
    McpResourceContent as MuxMcpResourceContent, McpResourceInfo as MuxMcpResourceInfo,
    McpResourceTemplate as MuxMcpResourceTemplate,
};
use mux::prelude::{
    McpClient, McpContentBlock, McpServerConfig as MuxMcpServerConfig, McpToolInfo, McpTransport,
    Tool, ToolDefinition,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex as TokioMutex;

/// Holds a connected MCP client and its cached capabilities.
pub(super) struct McpClientHandle {
    pub client: Arc<TokioMutex<McpClient>>,
    pub tools: Vec<McpToolInfo>,
    pub resources: Vec<MuxMcpResourceInfo>,
    pub resource_templates: Vec<MuxMcpResourceTemplate>,
    pub prompts: Vec<MuxMcpPromptInfo>,
    pub server_name: String,
}

/// MCP server configuration methods
#[uniffi::export]
impl MuxEngine {
    pub fn add_mcp_server(
        &self,
        workspace_id: String,
        config: McpServerConfig,
    ) -> Result<(), MuxFfiError> {
        let mut workspaces = self.workspaces.write();
        let workspace = workspaces
            .get_mut(&workspace_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            })?;

        // Check if server with same name already exists
        if workspace.mcp_servers.iter().any(|s| s.name == config.name) {
            return Err(MuxFfiError::Engine {
                message: format!("MCP server '{}' already exists in workspace", config.name),
            });
        }

        workspace.mcp_servers.push(config);
        drop(workspaces);

        self.save_workspaces();
        Ok(())
    }

    pub fn remove_mcp_server(
        &self,
        workspace_id: String,
        server_name: String,
    ) -> Result<(), MuxFfiError> {
        let mut workspaces = self.workspaces.write();
        let workspace = workspaces
            .get_mut(&workspace_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            })?;

        let original_len = workspace.mcp_servers.len();
        workspace.mcp_servers.retain(|s| s.name != server_name);

        if workspace.mcp_servers.len() == original_len {
            return Err(MuxFfiError::Engine {
                message: format!("MCP server '{}' not found in workspace", server_name),
            });
        }

        drop(workspaces);

        self.save_workspaces();
        Ok(())
    }

    pub fn list_mcp_servers(&self, workspace_id: String) -> Vec<McpServerConfig> {
        self.workspaces
            .read()
            .get(&workspace_id)
            .map(|ws| ws.mcp_servers.clone())
            .unwrap_or_default()
    }

    pub fn update_mcp_server(
        &self,
        workspace_id: String,
        config: McpServerConfig,
    ) -> Result<(), MuxFfiError> {
        let mut workspaces = self.workspaces.write();
        let workspace = workspaces
            .get_mut(&workspace_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            })?;

        // Find and update the server
        let server = workspace
            .mcp_servers
            .iter_mut()
            .find(|s| s.name == config.name)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("MCP server '{}' not found in workspace", config.name),
            })?;

        *server = config;
        drop(workspaces);

        self.save_workspaces();
        Ok(())
    }

    /// Connect to all enabled MCP servers for a workspace.
    /// This should be called when entering a workspace to establish connections.
    pub fn connect_workspace_servers(self: Arc<Self>, workspace_id: String) {
        let engine = self.clone();
        std::thread::spawn(move || {
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("Failed to create async runtime for MCP connection: {}", e);
                    return;
                }
            };
            rt.block_on(async move {
                if let Err(e) = engine.do_connect_workspace_servers(workspace_id).await {
                    eprintln!("Failed to connect workspace servers: {}", e);
                }
            });
        });
    }

    /// Respond to a tool approval request.
    /// This is called by Swift when the user approves/denies a tool use.
    pub fn respond_to_tool_approval(&self, tool_use_id: String, decision: ApprovalDecision) {
        let mut pending = self.pending_approvals.write();
        if let Some(sender) = pending.remove(&tool_use_id) {
            let _ = sender.send(decision);
        }
    }

    /// Disconnect all MCP servers for a workspace.
    /// This should be called when leaving a workspace.
    pub fn disconnect_workspace_servers(self: Arc<Self>, workspace_id: String) {
        let engine = self.clone();
        std::thread::spawn(move || {
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!(
                        "Failed to create async runtime for MCP disconnection: {}",
                        e
                    );
                    return;
                }
            };
            rt.block_on(async move {
                engine.do_disconnect_workspace_servers(&workspace_id).await;
            });
        });
    }

    /// List all MCP resources available across connected servers in a workspace.
    pub fn list_mcp_resources(&self, workspace_id: String) -> Vec<McpResourceInfo> {
        let clients = self.mcp_clients.read();
        let Some(workspace_clients) = clients.get(&workspace_id) else {
            return Vec::new();
        };

        workspace_clients
            .values()
            .flat_map(|handle| {
                handle.resources.iter().map(|r| McpResourceInfo {
                    uri: r.uri.clone(),
                    name: r.name.clone(),
                    description: r.description.clone(),
                    mime_type: r.mime_type.clone(),
                    server_name: handle.server_name.clone(),
                })
            })
            .collect()
    }

    /// List all MCP resource templates available across connected servers in a workspace.
    pub fn list_mcp_resource_templates(&self, workspace_id: String) -> Vec<McpResourceTemplate> {
        let clients = self.mcp_clients.read();
        let Some(workspace_clients) = clients.get(&workspace_id) else {
            return Vec::new();
        };

        workspace_clients
            .values()
            .flat_map(|handle| {
                handle
                    .resource_templates
                    .iter()
                    .map(|t| McpResourceTemplate {
                        uri_template: t.uri_template.clone(),
                        name: t.name.clone(),
                        description: t.description.clone(),
                        mime_type: t.mime_type.clone(),
                        server_name: handle.server_name.clone(),
                    })
            })
            .collect()
    }

    /// List all MCP prompts available across connected servers in a workspace.
    pub fn list_mcp_prompts(&self, workspace_id: String) -> Vec<McpPromptInfo> {
        let clients = self.mcp_clients.read();
        let Some(workspace_clients) = clients.get(&workspace_id) else {
            return Vec::new();
        };

        workspace_clients
            .values()
            .flat_map(|handle| {
                handle.prompts.iter().map(|p| McpPromptInfo {
                    name: p.name.clone(),
                    description: p.description.clone(),
                    arguments: p
                        .arguments
                        .iter()
                        .map(|a| McpPromptArgument {
                            name: a.name.clone(),
                            description: a.description.clone(),
                            required: a.required,
                        })
                        .collect(),
                    server_name: handle.server_name.clone(),
                })
            })
            .collect()
    }

    /// Read the content of an MCP resource from a specific server.
    pub fn read_mcp_resource(
        self: Arc<Self>,
        workspace_id: String,
        server_name: String,
        uri: String,
    ) -> Result<Vec<McpResourceContent>, MuxFfiError> {
        // Validate URI
        if uri.is_empty() {
            return Err(MuxFfiError::Engine {
                message: "Resource URI cannot be empty".to_string(),
            });
        }

        // Get client handle
        let client = {
            let clients = self.mcp_clients.read();
            clients
                .get(&workspace_id)
                .and_then(|ws| ws.get(&server_name))
                .map(|h| h.client.clone())
                .ok_or_else(|| MuxFfiError::Engine {
                    message: format!("MCP server '{}' not connected", server_name),
                })?
        };

        // Spawn blocking thread for async work
        let handle = std::thread::spawn(move || {
            let rt = Runtime::new().map_err(|e| MuxFfiError::Engine {
                message: format!("Failed to create runtime: {}", e),
            })?;

            rt.block_on(async move {
                let locked = client.lock().await;
                let contents =
                    locked
                        .read_resource(&uri)
                        .await
                        .map_err(|e| MuxFfiError::Engine {
                            message: e.to_string(),
                        })?;

                // Convert mux types to FFI types
                Ok(contents
                    .into_iter()
                    .map(|c| match c {
                        MuxMcpResourceContent::Text {
                            uri,
                            mime_type,
                            text,
                        } => McpResourceContent::Text {
                            uri,
                            mime_type,
                            text,
                        },
                        MuxMcpResourceContent::Blob {
                            uri,
                            mime_type,
                            blob,
                        } => McpResourceContent::Blob {
                            uri,
                            mime_type,
                            blob,
                        },
                    })
                    .collect())
            })
        });

        handle.join().map_err(|e| MuxFfiError::Engine {
            message: format!("Thread panicked: {:?}", e),
        })?
    }

    /// Get an MCP prompt from a specific server with the given arguments.
    ///
    /// NOTE: Prompt content is simplified to text for v1. Image content is converted
    /// to a placeholder string like "[Image: 1234 bytes, type: image/png]" and binary
    /// resource content becomes "[Binary data: N bytes]". The actual binary data is
    /// not preserved. If you need raw image/binary support, file a feature request.
    pub fn get_mcp_prompt(
        self: Arc<Self>,
        workspace_id: String,
        server_name: String,
        name: String,
        arguments: Vec<PromptArgumentValue>,
    ) -> Result<McpPromptResult, MuxFfiError> {
        // Get client handle
        let client = {
            let clients = self.mcp_clients.read();
            clients
                .get(&workspace_id)
                .and_then(|ws| ws.get(&server_name))
                .map(|h| h.client.clone())
                .ok_or_else(|| MuxFfiError::Engine {
                    message: format!("MCP server '{}' not connected", server_name),
                })?
        };

        // Convert Vec<PromptArgumentValue> to HashMap<String, String>
        let args_map: Option<HashMap<String, String>> = if arguments.is_empty() {
            None
        } else {
            Some(arguments.into_iter().map(|a| (a.name, a.value)).collect())
        };

        // Spawn blocking thread for async work
        let handle = std::thread::spawn(move || {
            let rt = Runtime::new().map_err(|e| MuxFfiError::Engine {
                message: format!("Failed to create runtime: {}", e),
            })?;

            rt.block_on(async move {
                let locked = client.lock().await;
                let result =
                    locked
                        .get_prompt(&name, args_map)
                        .await
                        .map_err(|e| MuxFfiError::Engine {
                            message: e.to_string(),
                        })?;

                // Convert to FFI type - simplify content to text only for v1
                Ok(McpPromptResult {
                    description: result.description,
                    messages: result
                        .messages
                        .into_iter()
                        .map(|m| McpPromptMessage {
                            role: m.role,
                            content: match m.content {
                                McpPromptContent::Text { text } => text,
                                McpPromptContent::Image { data, mime_type } => {
                                    format!("[Image: {} bytes, type: {}]", data.len(), mime_type)
                                }
                                McpPromptContent::Resource { resource } => match resource {
                                    MuxMcpResourceContent::Text { text, .. } => text,
                                    MuxMcpResourceContent::Blob { blob, .. } => {
                                        format!("[Binary data: {} bytes]", blob.len())
                                    }
                                },
                            },
                        })
                        .collect(),
                })
            })
        });

        handle.join().map_err(|e| MuxFfiError::Engine {
            message: format!("Thread panicked: {:?}", e),
        })?
    }
}

/// MCP client management methods
impl MuxEngine {
    /// Connect to all enabled MCP servers for a workspace.
    pub(super) async fn do_connect_workspace_servers(
        &self,
        workspace_id: String,
    ) -> Result<(), String> {
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

    /// Get all tools available for a workspace as ToolDefinitions for the LLM.
    /// Includes built-in mux tools, custom tools, and any connected MCP server tools.
    pub(super) fn get_workspace_tools(&self, workspace_id: &str) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();

        // Add built-in tools (always available, no prefix)
        for tool in &self.builtin_tools {
            tools.push(ToolDefinition {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.schema(),
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
                    });
                }
            }
        }

        tools
    }

    /// Find the MCP client and tool name for a qualified tool name (server:tool).
    pub(super) fn parse_tool_name(&self, qualified_name: &str) -> Option<(String, String)> {
        helpers::parse_qualified_tool_name(qualified_name)
    }

    /// Execute a tool call using pre-captured MCP client references.
    /// This is immune to race conditions from workspace disconnection during message processing.
    pub(super) async fn execute_tool_with_captured_client(
        captured_clients: &HashMap<String, Arc<TokioMutex<McpClient>>>,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let client_arc = captured_clients
            .get(server_name)
            .ok_or_else(|| format!("Server '{}' not available", server_name))?;

        let client = client_arc.lock().await;
        let result = client
            .call_tool(tool_name, arguments)
            .await
            .map_err(|e| e.to_string())?;

        // Convert McpToolResult to string
        let content_text: String = result
            .content
            .iter()
            .map(|block| match block {
                McpContentBlock::Text { text } => text.clone(),
                McpContentBlock::Image { data, mime_type } => {
                    format!("[Image: {} bytes, type: {}]", data.len(), mime_type)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Err(content_text)
        } else {
            Ok(content_text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::MuxEngine;
    use crate::types::McpServerConfig;

    fn test_dir(name: &str) -> String {
        std::env::temp_dir()
            .join(name)
            .to_string_lossy()
            .to_string()
    }

    fn create_test_engine() -> std::sync::Arc<MuxEngine> {
        MuxEngine::new(test_dir("mux-test-mcp")).unwrap()
    }

    fn create_stdio_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport_type: McpTransportType::Stdio,
            command: Some(if cfg!(target_os = "windows") {
                "cmd.exe".to_string()
            } else {
                "/usr/bin/echo".to_string()
            }),
            args: if cfg!(target_os = "windows") {
                vec!["/C".to_string(), "echo".to_string(), "hello".to_string()]
            } else {
                vec!["hello".to_string()]
            },
            url: None,
            enabled: true,
        }
    }

    fn create_sse_config(name: &str) -> McpServerConfig {
        McpServerConfig {
            name: name.to_string(),
            transport_type: McpTransportType::Sse,
            command: None,
            args: vec![],
            url: Some("http://localhost:8080".to_string()),
            enabled: false,
        }
    }

    #[test]
    fn test_add_mcp_server() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Test".to_string(), None)
            .unwrap();

        let config = create_stdio_config("test-server");
        engine.add_mcp_server(ws.id.clone(), config).unwrap();

        let servers = engine.list_mcp_servers(ws.id.clone());
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "test-server");

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_add_mcp_server_duplicate_name() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Dup Test".to_string(), None)
            .unwrap();

        let config1 = create_stdio_config("dup-server");
        engine.add_mcp_server(ws.id.clone(), config1).unwrap();

        let config2 = create_stdio_config("dup-server");
        let result = engine.add_mcp_server(ws.id.clone(), config2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_add_mcp_server_workspace_not_found() {
        let engine = create_test_engine();
        let config = create_stdio_config("orphan-server");
        let result = engine.add_mcp_server("nonexistent-ws".to_string(), config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Workspace not found")
        );
    }

    #[test]
    fn test_remove_mcp_server() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Remove Test".to_string(), None)
            .unwrap();

        let config = create_stdio_config("to-remove");
        engine.add_mcp_server(ws.id.clone(), config).unwrap();
        assert_eq!(engine.list_mcp_servers(ws.id.clone()).len(), 1);

        engine
            .remove_mcp_server(ws.id.clone(), "to-remove".to_string())
            .unwrap();
        assert_eq!(engine.list_mcp_servers(ws.id.clone()).len(), 0);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_remove_mcp_server_not_found() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Remove NF".to_string(), None)
            .unwrap();

        let result = engine.remove_mcp_server(ws.id.clone(), "ghost".to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in workspace")
        );

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_remove_mcp_server_workspace_not_found() {
        let engine = create_test_engine();
        let result = engine.remove_mcp_server("fake-ws".to_string(), "any".to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Workspace not found")
        );
    }

    #[test]
    fn test_list_mcp_servers_empty() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Empty".to_string(), None)
            .unwrap();

        let servers = engine.list_mcp_servers(ws.id.clone());
        assert!(servers.is_empty());

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_list_mcp_servers_nonexistent_workspace() {
        let engine = create_test_engine();
        let servers = engine.list_mcp_servers("no-such-ws".to_string());
        assert!(servers.is_empty());
    }

    #[test]
    fn test_list_mcp_servers_multiple() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Multi".to_string(), None)
            .unwrap();

        engine
            .add_mcp_server(ws.id.clone(), create_stdio_config("server-a"))
            .unwrap();
        engine
            .add_mcp_server(ws.id.clone(), create_sse_config("server-b"))
            .unwrap();
        engine
            .add_mcp_server(ws.id.clone(), create_stdio_config("server-c"))
            .unwrap();

        let servers = engine.list_mcp_servers(ws.id.clone());
        assert_eq!(servers.len(), 3);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_update_mcp_server() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Update".to_string(), None)
            .unwrap();

        let config = create_stdio_config("updatable");
        engine.add_mcp_server(ws.id.clone(), config).unwrap();

        // Update to SSE transport
        let updated = McpServerConfig {
            name: "updatable".to_string(),
            transport_type: McpTransportType::Sse,
            command: None,
            args: vec![],
            url: Some("http://new-url:9000".to_string()),
            enabled: false,
        };
        engine.update_mcp_server(ws.id.clone(), updated).unwrap();

        let servers = engine.list_mcp_servers(ws.id.clone());
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].transport_type, McpTransportType::Sse);
        assert_eq!(servers[0].url, Some("http://new-url:9000".to_string()));
        assert!(!servers[0].enabled);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_update_mcp_server_not_found() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MCP Update NF".to_string(), None)
            .unwrap();

        let config = create_stdio_config("ghost-update");
        let result = engine.update_mcp_server(ws.id.clone(), config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in workspace")
        );

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_update_mcp_server_workspace_not_found() {
        let engine = create_test_engine();
        let config = create_stdio_config("any");
        let result = engine.update_mcp_server("fake-ws".to_string(), config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Workspace not found")
        );
    }

    #[test]
    fn test_respond_to_tool_approval_with_pending() {
        let engine = create_test_engine();
        let (tx, rx) = tokio::sync::oneshot::channel();

        // Insert a pending approval
        engine
            .pending_approvals
            .write()
            .insert("test-tool-use-id".to_string(), tx);

        // Respond to it
        engine.respond_to_tool_approval("test-tool-use-id".to_string(), ApprovalDecision::Allow);

        // Verify it was received
        let rt = tokio::runtime::Runtime::new().unwrap();
        let decision = rt.block_on(async { rx.await.unwrap() });
        assert_eq!(decision, ApprovalDecision::Allow);
    }

    #[test]
    fn test_respond_to_tool_approval_denied() {
        let engine = create_test_engine();
        let (tx, rx) = tokio::sync::oneshot::channel();

        engine
            .pending_approvals
            .write()
            .insert("deny-id".to_string(), tx);
        engine.respond_to_tool_approval("deny-id".to_string(), ApprovalDecision::Deny);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let decision = rt.block_on(async { rx.await.unwrap() });
        assert_eq!(decision, ApprovalDecision::Deny);
    }

    #[test]
    fn test_respond_to_tool_approval_no_pending() {
        let engine = create_test_engine();
        // This should not panic - just silently ignore
        engine.respond_to_tool_approval("nonexistent".to_string(), ApprovalDecision::Allow);
    }

    #[test]
    fn test_get_workspace_tools_builtin_only() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Tools Test".to_string(), None)
            .unwrap();

        let tools = engine.get_workspace_tools(&ws.id);

        // Should have built-in tools: ReadFileTool, WriteFileTool, ListFilesTool, SearchTool, BashTool
        assert!(tools.len() >= 5);
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"read_file"));
        assert!(tool_names.contains(&"write_file"));
        assert!(tool_names.contains(&"list_files"));
        assert!(tool_names.contains(&"search"));
        assert!(tool_names.contains(&"bash"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_get_workspace_tools_no_task_tool_without_handler() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("No Task".to_string(), None)
            .unwrap();

        let tools = engine.get_workspace_tools(&ws.id);
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // Task tool should NOT be present without a subagent event handler
        assert!(!tool_names.contains(&"task"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_get_workspace_tools_with_task_tool() {
        use crate::callback::SubagentEventHandler;

        struct DummyHandler;
        impl SubagentEventHandler for DummyHandler {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
            fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
            fn on_stream_usage(&self, _subagent_id: String, _input_tokens: u32, _output_tokens: u32) {}
        }

        let engine = create_test_engine();
        let ws = engine
            .create_workspace("With Task".to_string(), None)
            .unwrap();

        engine.set_subagent_event_handler(Box::new(DummyHandler));

        let tools = engine.get_workspace_tools(&ws.id);
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

        // Task tool SHOULD be present with a handler
        assert!(tool_names.contains(&"task"));

        engine.clear_subagent_event_handler();
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_parse_tool_name_delegates_to_helpers() {
        let engine = create_test_engine();

        // Valid qualified name
        let result = engine.parse_tool_name("server:tool");
        assert_eq!(result, Some(("server".to_string(), "tool".to_string())));

        // No colon - returns None
        let result = engine.parse_tool_name("builtin_tool");
        assert!(result.is_none());
    }

    // ========================================================================
    // MCP Resources Tests
    // ========================================================================

    #[test]
    fn test_list_mcp_resources_empty_workspace() {
        let engine = create_test_engine();
        // Non-existent workspace returns empty list
        let resources = engine.list_mcp_resources("nonexistent".to_string());
        assert!(resources.is_empty());
    }

    #[test]
    fn test_list_mcp_resources_no_connected_servers() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Resources Empty".to_string(), None)
            .unwrap();

        // Workspace exists but no MCP servers connected - returns empty
        let resources = engine.list_mcp_resources(ws.id.clone());
        assert!(resources.is_empty());

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_list_mcp_resource_templates_empty_workspace() {
        let engine = create_test_engine();
        let templates = engine.list_mcp_resource_templates("nonexistent".to_string());
        assert!(templates.is_empty());
    }

    #[test]
    fn test_read_mcp_resource_server_not_connected() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Read Resource".to_string(), None)
            .unwrap();

        let result = engine.clone().read_mcp_resource(
            ws.id.clone(),
            "unknown_server".to_string(),
            "file:///test.txt".to_string(),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not connected"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_read_mcp_resource_empty_uri() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Empty URI".to_string(), None)
            .unwrap();

        let result = engine.clone().read_mcp_resource(
            ws.id.clone(),
            "any_server".to_string(),
            "".to_string(), // Empty URI
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("URI cannot be empty")
        );

        engine.delete_workspace(ws.id).unwrap();
    }

    // ========================================================================
    // MCP Prompts Tests
    // ========================================================================

    #[test]
    fn test_list_mcp_prompts_empty_workspace() {
        let engine = create_test_engine();
        let prompts = engine.list_mcp_prompts("nonexistent".to_string());
        assert!(prompts.is_empty());
    }

    #[test]
    fn test_list_mcp_prompts_no_connected_servers() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Prompts Empty".to_string(), None)
            .unwrap();

        let prompts = engine.list_mcp_prompts(ws.id.clone());
        assert!(prompts.is_empty());

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_get_mcp_prompt_server_not_connected() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Get Prompt".to_string(), None)
            .unwrap();

        let result = engine.clone().get_mcp_prompt(
            ws.id.clone(),
            "unknown_server".to_string(),
            "test_prompt".to_string(),
            vec![],
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not connected"));

        engine.delete_workspace(ws.id).unwrap();
    }
}
