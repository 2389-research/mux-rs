// ABOUTME: BuddyEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::types::*;
use crate::MuxFfiError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(uniffi::Object)]
pub struct BuddyEngine {
    #[allow(dead_code)]
    data_dir: PathBuf,
    workspaces: Arc<RwLock<HashMap<String, Workspace>>>,
    conversations: Arc<RwLock<HashMap<String, Vec<Conversation>>>>,
}

#[uniffi::export]
impl BuddyEngine {
    #[uniffi::constructor]
    pub fn new(data_dir: String) -> Result<Arc<Self>, MuxFfiError> {
        let path = PathBuf::from(&data_dir);

        std::fs::create_dir_all(&path).map_err(|e| MuxFfiError::Engine {
            message: format!("Failed to create data directory: {}", e),
        })?;

        Ok(Arc::new(Self {
            data_dir: path,
            workspaces: Arc::new(RwLock::new(HashMap::new())),
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }))
    }

    pub fn create_workspace(
        &self,
        name: String,
        path: Option<String>,
    ) -> Result<Workspace, MuxFfiError> {
        let workspace = Workspace::new(name, path);
        let id = workspace.id.clone();

        self.workspaces.write().insert(id.clone(), workspace.clone());
        self.conversations.write().insert(id, Vec::new());

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
        self.workspaces.write().remove(&workspace_id);
        self.conversations.write().remove(&workspace_id);
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

        Ok(conversation)
    }

    pub fn list_conversations(&self, workspace_id: String) -> Vec<Conversation> {
        self.conversations
            .read()
            .get(&workspace_id)
            .cloned()
            .unwrap_or_default()
    }
}
