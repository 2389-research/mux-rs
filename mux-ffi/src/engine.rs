// ABOUTME: BuddyEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::callback::{ChatCallback, ChatResult};
use crate::types::*;
use crate::MuxFfiError;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Arc as StdArc;
use tokio::runtime::Runtime;

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

    pub fn set_api_key(&self, provider: Provider, key: String) {
        let provider_key = match provider {
            Provider::Anthropic => "ANTHROPIC_API_KEY",
            Provider::OpenAI => "OPENAI_API_KEY",
        };
        // SAFETY: We're setting environment variables at initialization time,
        // before spawning threads that might read them concurrently.
        unsafe {
            std::env::set_var(provider_key, key);
        }
    }

    pub fn send_message(
        &self,
        conversation_id: String,
        content: String,
        callback: Box<dyn ChatCallback>,
    ) {
        let callback = StdArc::new(callback);
        let cb = callback.clone();

        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async {
                match Self::do_send_message(conversation_id.clone(), content).await {
                    Ok(result) => cb.on_complete(result),
                    Err(e) => cb.on_error(e),
                }
            });
        });
    }
}

impl BuddyEngine {
    async fn do_send_message(
        conversation_id: String,
        content: String,
    ) -> Result<ChatResult, String> {
        // Stub implementation that echoes back - real LLM client integration comes later
        Ok(ChatResult {
            conversation_id,
            final_text: format!("Echo: {}", content),
            tool_use_count: 0,
            input_tokens: 0,
            output_tokens: 0,
        })
    }
}
