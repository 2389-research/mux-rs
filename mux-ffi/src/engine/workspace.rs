// ABOUTME: Workspace and conversation CRUD operations for MuxEngine.
// ABOUTME: Handles creation, listing, and deletion of workspaces and conversations.

use super::MuxEngine;
use crate::MuxFfiError;
use crate::types::{Conversation, Workspace, WorkspaceSummary};

/// Workspace and Conversation CRUD operations
#[uniffi::export]
impl MuxEngine {
    pub fn create_workspace(
        &self,
        name: String,
        path: Option<String>,
    ) -> Result<Workspace, MuxFfiError> {
        let workspace = Workspace::new(name, path);
        let id = workspace.id.clone();

        self.workspaces
            .write()
            .insert(id.clone(), workspace.clone());
        self.conversations.write().insert(id, Vec::new());

        // Persist to disk
        self.save_workspaces();
        self.save_conversations();

        Ok(workspace)
    }

    pub fn list_workspaces(&self) -> Vec<WorkspaceSummary> {
        let workspaces = self.workspaces.read();
        let conversations = self.conversations.read();

        workspaces
            .values()
            .map(|ws| WorkspaceSummary {
                id: ws.id.clone(),
                name: ws.name.clone(),
                path: ws.path.clone(),
                conversation_count: conversations
                    .get(&ws.id)
                    .map(|c| c.len() as u32)
                    .unwrap_or(0),
            })
            .collect()
    }

    pub fn delete_workspace(&self, workspace_id: String) -> Result<(), MuxFfiError> {
        // Get conversation IDs before removing them (for message cleanup)
        let conversation_ids: Vec<String> = self
            .conversations
            .read()
            .get(&workspace_id)
            .map(|convs| convs.iter().map(|c| c.id.clone()).collect())
            .unwrap_or_default();

        // Remove any connected MCP clients for this workspace (fire and forget shutdown).
        // We remove from the map synchronously to prevent new tool calls,
        // but shutdown happens in the background to avoid blocking.
        // NOTE: This is intentionally fire-and-forget. Callers should not expect
        // synchronous cleanup. If the app exits immediately after deletion, some
        // MCP servers may not receive clean shutdown signals.
        if let Some(clients) = self.mcp_clients.write().remove(&workspace_id) {
            std::thread::spawn(move || {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        eprintln!("Failed to create runtime for MCP cleanup: {}", e);
                        return;
                    }
                };
                rt.block_on(async {
                    for (name, handle) in clients {
                        let client = handle.client.lock().await;
                        if let Err(e) = client.shutdown().await {
                            eprintln!(
                                "Error shutting down MCP server '{}' on workspace delete: {}",
                                name, e
                            );
                        }
                    }
                });
            });
        }

        self.workspaces.write().remove(&workspace_id);
        self.conversations.write().remove(&workspace_id);

        // Remove message history for all conversations in this workspace
        {
            let mut history = self.message_history.write();
            for conv_id in &conversation_ids {
                history.remove(conv_id);
            }
        }

        // Persist to disk
        self.save_workspaces();
        self.save_conversations();

        // Delete message files for conversations in this workspace
        for conv_id in conversation_ids {
            self.delete_message_file(&conv_id);
        }

        Ok(())
    }

    pub fn create_conversation(
        &self,
        workspace_id: String,
        title: String,
    ) -> Result<Conversation, MuxFfiError> {
        let conversation = Conversation::new(workspace_id.clone(), title);

        let mut conversations = self.conversations.write();
        conversations
            .entry(workspace_id)
            .or_insert_with(Vec::new)
            .push(conversation.clone());

        // Initialize empty message history for this conversation
        self.message_history
            .write()
            .insert(conversation.id.clone(), Vec::new());

        // Persist to disk
        drop(conversations); // Release lock before saving
        self.save_conversations();

        Ok(conversation)
    }

    pub fn list_conversations(&self, workspace_id: String) -> Vec<Conversation> {
        self.conversations
            .read()
            .get(&workspace_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Set a custom system prompt for a workspace.
    /// Tool guidance is automatically appended to this prompt.
    /// Pass None to reset to the default prompt.
    pub fn set_system_prompt(
        &self,
        workspace_id: String,
        prompt: Option<String>,
    ) -> Result<(), MuxFfiError> {
        let mut workspaces = self.workspaces.write();
        let workspace = workspaces
            .get_mut(&workspace_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            })?;

        workspace.system_prompt = prompt;
        drop(workspaces);

        self.save_workspaces();
        Ok(())
    }

    /// Get the current system prompt for a workspace.
    /// Returns None if using the default prompt.
    pub fn get_system_prompt(&self, workspace_id: String) -> Option<String> {
        self.workspaces
            .read()
            .get(&workspace_id)
            .and_then(|ws| ws.system_prompt.clone())
    }
}
