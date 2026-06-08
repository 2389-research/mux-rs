// ABOUTME: UniFFI-exported MCP operations on MuxEngine (server configuration, resource/prompt access).
// ABOUTME: This entire impl block is exported as one unit to preserve the binding contract.

use crate::MuxFfiError;
use crate::engine::MuxEngine;
use crate::types::{
    ApprovalDecision, McpPromptArgument, McpPromptInfo, McpPromptMessage, McpPromptResult,
    McpResourceContent, McpResourceInfo, McpResourceTemplate, McpServerConfig, PromptArgumentValue,
};
use mux::mcp::{McpPromptContent, McpResourceContent as MuxMcpResourceContent};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Reject MCP server names that would break the "server:tool" qualified-name
/// scheme (parsed via `splitn(2, ':')` in `engine::helpers::parse_qualified_tool_name`).
fn validate_server_name(name: &str) -> Result<(), MuxFfiError> {
    if name.is_empty() {
        return Err(MuxFfiError::Engine {
            message: "MCP server name cannot be empty".to_string(),
        });
    }
    if name.contains(':') {
        return Err(MuxFfiError::Engine {
            message: format!(
                "MCP server name '{}' must not contain ':' (reserved as the server:tool separator)",
                name
            ),
        });
    }
    Ok(())
}

/// MCP server configuration methods
#[uniffi::export]
impl MuxEngine {
    pub fn add_mcp_server(
        &self,
        workspace_id: String,
        config: McpServerConfig,
    ) -> Result<(), MuxFfiError> {
        validate_server_name(&config.name)?;
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
        validate_server_name(&config.name)?;
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
