// ABOUTME: BuddyEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::callback::{ChatCallback, ChatResult};
use crate::types::*;
use crate::MuxFfiError;
use futures::StreamExt;
use mux::prelude::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Stored message for conversation history
#[derive(Clone)]
struct StoredMessage {
    role: Role,
    content: String,
}

#[derive(uniffi::Object)]
pub struct BuddyEngine {
    #[allow(dead_code)]
    data_dir: PathBuf,
    workspaces: Arc<RwLock<HashMap<String, Workspace>>>,
    conversations: Arc<RwLock<HashMap<String, Vec<Conversation>>>>,
    /// Conversation history for LLM context
    message_history: Arc<RwLock<HashMap<String, Vec<StoredMessage>>>>,
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
            message_history: Arc::new(RwLock::new(HashMap::new())),
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

        // Initialize empty message history for this conversation
        self.message_history
            .write()
            .insert(conversation.id.clone(), Vec::new());

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
}

impl BuddyEngine {
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

        // Try to create Anthropic client
        let client = match AnthropicClient::from_env() {
            Ok(c) => c,
            Err(_) => {
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

                return Ok(ChatResult {
                    conversation_id,
                    final_text: echo_text,
                    tool_use_count: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                });
            }
        };

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

        // Create request
        let request = Request::new("claude-sonnet-4-20250514")
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

        Ok(ChatResult {
            conversation_id,
            final_text: full_text,
            tool_use_count: 0,
            input_tokens,
            output_tokens,
        })
    }
}
