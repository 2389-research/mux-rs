// ABOUTME: BuddyEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::callback::{ChatCallback, ChatResult};
use crate::types::{Conversation, McpServerConfig, Provider, Workspace, WorkspaceSummary};
use crate::MuxFfiError;
use futures::StreamExt;
use mux::prelude::{AnthropicClient, ContentBlock, LlmClient, Message, Request, Role, StreamEvent};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Stored message for conversation history
#[derive(Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: Role,
    content: String,
}

/// File names for persistence
const WORKSPACES_FILE: &str = "workspaces.json";
const CONVERSATIONS_FILE: &str = "conversations.json";
const MESSAGES_DIR: &str = "messages";

#[derive(uniffi::Object)]
pub struct BuddyEngine {
    data_dir: PathBuf,
    workspaces: Arc<RwLock<HashMap<String, Workspace>>>,
    conversations: Arc<RwLock<HashMap<String, Vec<Conversation>>>>,
    /// Conversation history for LLM context
    message_history: Arc<RwLock<HashMap<String, Vec<StoredMessage>>>>,
    /// Thread-safe API key storage (avoids unsafe env::set_var)
    api_keys: Arc<RwLock<HashMap<Provider, String>>>,
}

#[uniffi::export]
impl BuddyEngine {
    #[uniffi::constructor]
    pub fn new(data_dir: String) -> Result<Arc<Self>, MuxFfiError> {
        let path = PathBuf::from(&data_dir);

        fs::create_dir_all(&path).map_err(|e| MuxFfiError::Engine {
            message: format!("Failed to create data directory: {}", e),
        })?;

        // Create messages directory if it doesn't exist
        let messages_dir = path.join(MESSAGES_DIR);
        fs::create_dir_all(&messages_dir).map_err(|e| MuxFfiError::Engine {
            message: format!("Failed to create messages directory: {}", e),
        })?;

        // Load existing data from disk
        let workspaces = Self::load_workspaces(&path);
        let conversations = Self::load_conversations(&path);
        let message_history = Self::load_all_messages(&path, &conversations);

        Ok(Arc::new(Self {
            data_dir: path,
            workspaces: Arc::new(RwLock::new(workspaces)),
            conversations: Arc::new(RwLock::new(conversations)),
            message_history: Arc::new(RwLock::new(message_history)),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
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

    pub fn set_api_key(&self, provider: Provider, key: String) {
        // Store API key in thread-safe storage
        self.api_keys.write().insert(provider, key);
    }

    /// Get stored API key for a provider
    pub fn get_api_key(&self, provider: Provider) -> Option<String> {
        self.api_keys.read().get(&provider).cloned()
    }

    pub fn send_message(
        self: Arc<Self>,
        conversation_id: String,
        content: String,
        callback: Box<dyn ChatCallback>,
    ) {
        let engine = self.clone();
        let callback = Arc::new(callback);
        let cb = callback.clone();

        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                match engine
                    .do_send_message(conversation_id.clone(), content, cb.clone())
                    .await
                {
                    Ok(result) => cb.on_complete(result),
                    Err(e) => cb.on_error(e),
                }
            });
        });
    }

    pub fn add_mcp_server(
        &self,
        workspace_id: String,
        config: McpServerConfig,
    ) -> Result<(), MuxFfiError> {
        let mut workspaces = self.workspaces.write();
        let workspace = workspaces.get_mut(&workspace_id).ok_or_else(|| {
            MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            }
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
        let workspace = workspaces.get_mut(&workspace_id).ok_or_else(|| {
            MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            }
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
        let workspace = workspaces.get_mut(&workspace_id).ok_or_else(|| {
            MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            }
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
}

impl BuddyEngine {
    /// Get the configured model for a conversation's workspace
    fn get_model_for_conversation(&self, conversation_id: &str) -> Option<String> {
        // Find workspace for this conversation
        let conversations = self.conversations.read();
        let workspace_id = conversations
            .iter()
            .find_map(|(ws_id, convs)| {
                if convs.iter().any(|c| c.id == conversation_id) {
                    Some(ws_id.clone())
                } else {
                    None
                }
            })?;
        drop(conversations);

        // Get workspace's LLM config
        let workspaces = self.workspaces.read();
        let workspace = workspaces.get(&workspace_id)?;
        workspace.llm_config.as_ref().map(|cfg| cfg.model.clone())
    }

    async fn do_send_message(
        &self,
        conversation_id: String,
        content: String,
        callback: Arc<Box<dyn ChatCallback>>,
    ) -> Result<ChatResult, String> {
        // Add user message to history
        {
            let mut history = self.message_history.write();
            let messages = history
                .entry(conversation_id.clone())
                .or_insert_with(Vec::new);
            messages.push(StoredMessage {
                role: Role::User,
                content: content.clone(),
            });
        }

        // Persist user message to disk
        self.save_messages(&conversation_id);

        // Get API key from thread-safe storage
        let api_key = match self.api_keys.read().get(&Provider::Anthropic).cloned() {
            Some(key) if !key.is_empty() => key,
            _ => {
                // Fallback to echo if no API key
                let echo_text = format!("(No API key set) Echo: {}", content);
                callback.on_text_delta(echo_text.clone());

                // Store assistant response
                {
                    let mut history = self.message_history.write();
                    if let Some(messages) = history.get_mut(&conversation_id) {
                        messages.push(StoredMessage {
                            role: Role::Assistant,
                            content: echo_text.clone(),
                        });
                    }
                }

                // Persist to disk
                self.save_messages(&conversation_id);

                return Ok(ChatResult {
                    conversation_id,
                    final_text: echo_text,
                    tool_use_count: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                });
            }
        };

        // Create client with stored API key
        let client = AnthropicClient::new(&api_key);

        // Build message history for context
        let messages: Vec<Message> = {
            let history = self.message_history.read();
            history
                .get(&conversation_id)
                .map(|msgs| {
                    msgs.iter()
                        .map(|m| Message {
                            role: m.role,
                            content: vec![ContentBlock::text(m.content.clone())],
                        })
                        .collect()
                })
                .unwrap_or_default()
        };

        // Get model from workspace config, falling back to default
        let model = self.get_model_for_conversation(&conversation_id)
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

        // Create request
        let request = Request::new(&model)
            .messages(messages)
            .max_tokens(4096);

        // Send and stream response
        let mut stream = client.create_message_stream(&request);

        let mut full_text = String::new();
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;

        while let Some(event) = stream.next().await {
            match event {
                Ok(StreamEvent::ContentBlockDelta { text, .. }) => {
                    full_text.push_str(&text);
                    callback.on_text_delta(text);
                }
                Ok(StreamEvent::MessageStart { .. }) => {
                    // MessageStart doesn't carry usage in the StreamEvent enum
                }
                Ok(StreamEvent::MessageDelta { usage, .. }) => {
                    input_tokens = usage.input_tokens;
                    output_tokens = usage.output_tokens;
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(e.to_string());
                }
            }
        }

        // Store assistant response in history
        {
            let mut history = self.message_history.write();
            if let Some(messages) = history.get_mut(&conversation_id) {
                messages.push(StoredMessage {
                    role: Role::Assistant,
                    content: full_text.clone(),
                });
            }
        }

        // Persist to disk
        self.save_messages(&conversation_id);

        Ok(ChatResult {
            conversation_id,
            final_text: full_text,
            tool_use_count: 0,
            input_tokens,
            output_tokens,
        })
    }
}

/// Persistence helper methods
impl BuddyEngine {
    /// Load workspaces from disk. Returns empty HashMap if file doesn't exist or is invalid.
    fn load_workspaces(data_dir: &PathBuf) -> HashMap<String, Workspace> {
        let path = data_dir.join(WORKSPACES_FILE);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("Failed to parse {}: {}", path.display(), e);
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    }

    /// Load conversations from disk. Returns empty HashMap if file doesn't exist or is invalid.
    fn load_conversations(data_dir: &PathBuf) -> HashMap<String, Vec<Conversation>> {
        let path = data_dir.join(CONVERSATIONS_FILE);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("Failed to parse {}: {}", path.display(), e);
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    }

    /// Load all message histories from disk based on known conversations.
    fn load_all_messages(
        data_dir: &PathBuf,
        conversations: &HashMap<String, Vec<Conversation>>,
    ) -> HashMap<String, Vec<StoredMessage>> {
        let messages_dir = data_dir.join(MESSAGES_DIR);
        let mut all_messages = HashMap::new();

        for convs in conversations.values() {
            for conv in convs {
                let path = messages_dir.join(format!("{}.json", conv.id));
                if let Ok(contents) = fs::read_to_string(&path) {
                    if let Ok(msgs) = serde_json::from_str::<Vec<StoredMessage>>(&contents) {
                        all_messages.insert(conv.id.clone(), msgs);
                    }
                }
                // Initialize with empty vec if not loaded
                all_messages.entry(conv.id.clone()).or_insert_with(Vec::new);
            }
        }

        all_messages
    }

    /// Save workspaces to disk.
    fn save_workspaces(&self) {
        let path = self.data_dir.join(WORKSPACES_FILE);
        let workspaces = self.workspaces.read();
        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&*workspaces).unwrap_or_default()) {
            eprintln!("Failed to save workspaces to {}: {}", path.display(), e);
        }
    }

    /// Save conversations to disk.
    fn save_conversations(&self) {
        let path = self.data_dir.join(CONVERSATIONS_FILE);
        let conversations = self.conversations.read();
        if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&*conversations).unwrap_or_default()) {
            eprintln!("Failed to save conversations to {}: {}", path.display(), e);
        }
    }

    /// Save message history for a specific conversation to disk.
    fn save_messages(&self, conversation_id: &str) {
        let messages_dir = self.data_dir.join(MESSAGES_DIR);
        let path = messages_dir.join(format!("{}.json", conversation_id));
        let history = self.message_history.read();
        if let Some(messages) = history.get(conversation_id) {
            if let Err(e) = fs::write(&path, serde_json::to_string_pretty(messages).unwrap_or_default()) {
                eprintln!("Failed to save messages to {}: {}", path.display(), e);
            }
        }
    }

    /// Delete message file for a conversation.
    fn delete_message_file(&self, conversation_id: &str) {
        let messages_dir = self.data_dir.join(MESSAGES_DIR);
        let path = messages_dir.join(format!("{}.json", conversation_id));
        if let Err(e) = fs::remove_file(&path) {
            // Only log if it's not a "file not found" error
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("Failed to delete message file {}: {}", path.display(), e);
            }
        }
    }
}
