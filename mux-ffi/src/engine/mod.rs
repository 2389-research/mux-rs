// ABOUTME: MuxEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

mod context_mgmt;
mod helpers;
mod mcp;
mod messaging;
mod persistence;
mod subagent;
mod workspace;

use crate::bridge::FfiToolBridge;
use crate::callback::{
    ChatCallback, CustomTool, HookHandler, LlmProvider, SubagentCallback, SubagentEventHandler,
};
use crate::callback_client::CallbackLlmClient;
use crate::context::ModelContextConfig;
use crate::types::{AgentConfig, ApprovalDecision, Conversation, Provider, TranscriptData, Workspace};
use crate::MuxFfiError;
use mux::agent::MemoryTranscriptStore;
#[cfg(test)]
use mux::prelude::{ContentBlock, Role};
use mux::tool::Tool;
use mux::tools::{BashTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Internal configuration for a provider
#[derive(Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: Option<String>,
    /// Default model for this provider. Must be set by Swift - no hardcoded defaults.
    default_model: Option<String>,
}

use mcp::McpClientHandle;
use persistence::{StoredMessage, MESSAGES_DIR};

#[derive(uniffi::Object)]
pub struct MuxEngine {
    data_dir: PathBuf,
    workspaces: Arc<RwLock<HashMap<String, Workspace>>>,
    conversations: Arc<RwLock<HashMap<String, Vec<Conversation>>>>,
    /// Conversation history for LLM context
    message_history: Arc<RwLock<HashMap<String, Vec<StoredMessage>>>>,
    /// Thread-safe API key storage (avoids unsafe env::set_var).
    /// NOTE: Keys are stored in-memory only for security. The Swift app should
    /// persist keys securely (e.g., Keychain) and call set_api_key on each launch.
    /// NOTE: Currently only Anthropic provider is supported for chat. OpenAI support
    /// is planned but not yet implemented.
    api_keys: Arc<RwLock<HashMap<Provider, ProviderConfig>>>,
    /// Connected MCP clients, keyed by workspace_id -> server_name -> handle
    mcp_clients: Arc<RwLock<HashMap<String, HashMap<String, McpClientHandle>>>>,
    /// Pending tool approval requests, keyed by tool_use_id -> oneshot sender
    pending_approvals:
        Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>,
    /// Built-in tools from mux (always available)
    builtin_tools: Vec<Arc<dyn Tool>>,
    /// Registered agent configurations
    agent_configs: Arc<RwLock<HashMap<String, AgentConfig>>>,
    /// Hook handler (optional)
    hook_handler: Arc<RwLock<Option<Box<dyn HookHandler>>>>,
    /// Custom tools registered from Swift
    custom_tools: Arc<RwLock<HashMap<String, Arc<FfiToolBridge>>>>,
    /// Transcript storage for resume capability
    transcript_store: Arc<MemoryTranscriptStore>,
    /// Default provider for new workspaces
    default_provider: Arc<RwLock<Provider>>,
    /// Subagent event handler for TaskTool events
    subagent_event_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>>,
    /// Registered callback LLM providers, keyed by name
    callback_providers: Arc<RwLock<HashMap<String, Arc<CallbackLlmClient>>>>,
    /// Per-model context configuration (context limit, compaction mode, etc.)
    model_context_configs: Arc<RwLock<HashMap<String, ModelContextConfig>>>,
}

#[uniffi::export]
impl MuxEngine {
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

        // Initialize built-in tools
        let builtin_tools: Vec<Arc<dyn Tool>> = vec![
            Arc::new(ReadFileTool),
            Arc::new(WriteFileTool),
            Arc::new(ListFilesTool),
            Arc::new(SearchTool),
            Arc::new(BashTool),
        ];

        Ok(Arc::new(Self {
            data_dir: path,
            workspaces: Arc::new(RwLock::new(workspaces)),
            conversations: Arc::new(RwLock::new(conversations)),
            message_history: Arc::new(RwLock::new(message_history)),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            mcp_clients: Arc::new(RwLock::new(HashMap::new())),
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
            builtin_tools,
            agent_configs: Arc::new(RwLock::new(HashMap::new())),
            hook_handler: Arc::new(RwLock::new(None)),
            custom_tools: Arc::new(RwLock::new(HashMap::new())),
            transcript_store: MemoryTranscriptStore::shared(),
            default_provider: Arc::new(RwLock::new(Provider::Anthropic)),
            subagent_event_handler: Arc::new(RwLock::new(None)),
            callback_providers: Arc::new(RwLock::new(HashMap::new())),
            model_context_configs: Arc::new(RwLock::new(HashMap::new())),
        }))
    }

    pub fn set_api_key(&self, provider: Provider, key: String) {
        // Store API key in thread-safe storage using set_provider_config
        // Note: This preserves existing default_model if any
        let existing_model = self.get_default_model(provider.clone());
        self.set_provider_config(provider, key, None, existing_model);
    }

    /// Get stored API key for a provider
    pub fn get_api_key(&self, provider: Provider) -> Option<String> {
        self.api_keys
            .read()
            .get(&provider)
            .map(|config| config.api_key.clone())
    }

    /// Set provider configuration with API key and optional base URL
    pub fn set_provider_config(
        &self,
        provider: Provider,
        api_key: String,
        base_url: Option<String>,
        default_model: Option<String>,
    ) {
        self.api_keys.write().insert(
            provider,
            ProviderConfig {
                api_key,
                base_url,
                default_model,
            },
        );
    }

    /// Get the default model for a provider
    pub fn get_default_model(&self, provider: Provider) -> Option<String> {
        self.api_keys
            .read()
            .get(&provider)
            .and_then(|c| c.default_model.clone())
    }

    /// Set the default provider for new workspaces
    pub fn set_default_provider(&self, provider: Provider) {
        *self.default_provider.write() = provider;
    }

    /// Get the current default provider
    pub fn get_default_provider(&self) -> Provider {
        self.default_provider.read().clone()
    }

    /// Register an agent configuration
    pub fn register_agent(&self, config: AgentConfig) -> Result<(), MuxFfiError> {
        let name = config.name.clone();
        self.agent_configs.write().insert(name, config);
        Ok(())
    }

    /// List all registered agent names
    pub fn list_agents(&self) -> Vec<String> {
        self.agent_configs.read().keys().cloned().collect()
    }

    /// Unregister an agent by name
    pub fn unregister_agent(&self, name: String) -> Result<(), MuxFfiError> {
        if self.agent_configs.write().remove(&name).is_none() {
            return Err(MuxFfiError::AgentNotFound { name });
        }
        Ok(())
    }

    /// Set the hook handler for intercepting lifecycle events
    pub fn set_hook_handler(&self, handler: Box<dyn HookHandler>) {
        *self.hook_handler.write() = Some(handler);
    }

    /// Clear the current hook handler
    pub fn clear_hook_handler(&self) {
        *self.hook_handler.write() = None;
    }

    /// Set the subagent event handler for TaskTool events.
    /// This handler receives streaming updates when subagents are spawned.
    pub fn set_subagent_event_handler(&self, handler: Box<dyn SubagentEventHandler>) {
        *self.subagent_event_handler.write() = Some(handler);
    }

    /// Clear the current subagent event handler.
    pub fn clear_subagent_event_handler(&self) {
        *self.subagent_event_handler.write() = None;
    }

    /// Register a callback-based LLM provider.
    /// The provider can then be used via `set_default_provider(Provider::Custom { name })`.
    pub fn register_llm_provider(&self, name: String, provider: Box<dyn LlmProvider>) {
        let client = CallbackLlmClient::new(provider);
        self.callback_providers
            .write()
            .insert(name, Arc::new(client));
    }

    /// Unregister a callback LLM provider.
    pub fn unregister_llm_provider(&self, name: String) -> Result<(), MuxFfiError> {
        if self.callback_providers.write().remove(&name).is_none() {
            return Err(MuxFfiError::LlmProviderNotFound { name });
        }
        Ok(())
    }

    /// List registered callback LLM providers.
    pub fn list_llm_providers(&self) -> Vec<String> {
        self.callback_providers.read().keys().cloned().collect()
    }

    /// Register a custom tool from Swift
    pub fn register_custom_tool(
        &self,
        tool: Box<dyn CustomTool>,
    ) -> Result<(), MuxFfiError> {
        let bridge = FfiToolBridge::new(tool).map_err(|e| MuxFfiError::Engine {
            message: e.to_string(),
        })?;
        let name = bridge.name().to_string();
        self.custom_tools.write().insert(name, Arc::new(bridge));
        Ok(())
    }

    /// Unregister a custom tool by name
    pub fn unregister_custom_tool(&self, name: String) -> Result<(), MuxFfiError> {
        if self.custom_tools.write().remove(&name).is_none() {
            return Err(MuxFfiError::ToolNotFound { name });
        }
        Ok(())
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
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    cb.on_error(format!("Failed to create async runtime: {}", e));
                    return;
                }
            };
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

    /// Spawn a subagent to perform a task.
    /// The agent runs asynchronously and results are delivered via the callback.
    pub fn spawn_agent(
        self: Arc<Self>,
        workspace_id: String,
        agent_name: String,
        task: String,
        save_transcript: bool,
        callback: Box<dyn SubagentCallback>,
    ) {
        let engine = self.clone();
        let callback = Arc::new(callback);

        std::thread::spawn(move || {
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    callback.on_error("".to_string(), format!("Failed to create runtime: {}", e));
                    return;
                }
            };

            rt.block_on(async move {
                match engine
                    .do_spawn_agent(
                        workspace_id,
                        agent_name,
                        task,
                        save_transcript,
                        callback.clone(),
                    )
                    .await
                {
                    Ok(result) => callback.on_complete(result),
                    Err(e) => callback.on_error("".to_string(), e),
                }
            });
        });
    }

    /// Resume an agent from a saved transcript.
    /// The agent continues from where it left off, with results delivered via the callback.
    pub fn resume_agent(
        self: Arc<Self>,
        transcript: TranscriptData,
        callback: Box<dyn SubagentCallback>,
    ) {
        let engine = self.clone();
        let callback = Arc::new(callback);
        let agent_id = transcript.agent_id.clone();

        std::thread::spawn(move || {
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    callback
                        .on_error(agent_id.clone(), format!("Failed to create runtime: {}", e));
                    return;
                }
            };

            rt.block_on(async move {
                match engine.do_resume_agent(transcript, callback.clone()).await {
                    Ok(result) => callback.on_complete(result),
                    Err(e) => callback.on_error(agent_id, e),
                }
            });
        });
    }
}

/// Test helper methods - only available in test builds
#[cfg(test)]
impl MuxEngine {
    /// Inject a message into conversation history for testing.
    pub(crate) fn inject_test_message(&self, conversation_id: &str, role: Role, text: &str) {
        let mut history = self.message_history.write();
        let messages = history.entry(conversation_id.to_string()).or_default();
        messages.push(StoredMessage {
            role,
            content: vec![ContentBlock::text(text)],
        });
    }

    /// Get message count for a conversation (for test assertions).
    pub(crate) fn get_message_count(&self, conversation_id: &str) -> usize {
        self.message_history
            .read()
            .get(conversation_id)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Set LLM config for a workspace (for testing context management).
    pub(crate) fn set_workspace_llm_config(
        &self,
        workspace_id: &str,
        model: &str,
    ) {
        let mut workspaces = self.workspaces.write();
        if let Some(ws) = workspaces.get_mut(workspace_id) {
            ws.llm_config = Some(crate::types::LlmConfig {
                provider: crate::types::Provider::Anthropic,
                model: model.to_string(),
                api_key_ref: None,
            });
        }
    }
}
