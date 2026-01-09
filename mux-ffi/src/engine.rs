// ABOUTME: MuxEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::bridge::FfiToolBridge;
use crate::callback::{
    ChatCallback, ChatResult, CustomTool, HookHandler, LlmProvider, SubagentCallback,
    SubagentEventHandler, ToolUseRequest,
};
use crate::callback_client::CallbackLlmClient;
use crate::context::{CompactionMode, ContextUsage, ModelContextConfig, effective_limit, estimate_tokens};
use crate::task_tool::FfiTaskTool;
use crate::types::{
    AgentConfig, ApprovalDecision, Conversation, McpServerConfig, McpTransportType, Provider,
    SubagentResult, TranscriptData, Workspace, WorkspaceSummary,
};
use crate::MuxFfiError;
use mux::agent::{AgentRegistry, MemoryTranscriptStore};
use mux::hook::HookRegistry;
use mux::llm::GeminiClient;
use mux::prelude::{
    AgentDefinition, AnthropicClient, ContentBlock, LlmClient, McpClient, McpContentBlock,
    McpServerConfig as MuxMcpServerConfig, McpToolInfo, McpTransport, Message, OpenAIClient,
    Registry, Request, Role, StopReason, SubAgent, ToolDefinition,
};
use mux::tool::Tool;
use mux::tools::{BashTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex as TokioMutex;

/// Internal configuration for a provider
#[derive(Clone)]
struct ProviderConfig {
    api_key: String,
    base_url: Option<String>,
    /// Default model for this provider. Must be set by Swift - no hardcoded defaults.
    default_model: Option<String>,
}

/// Stored message for conversation history.
/// Uses Vec<ContentBlock> to preserve tool use/result structure.
#[derive(Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: Role,
    content: Vec<ContentBlock>,
}

/// Legacy format (pre-v0.6.2) stored content as String.
/// Used for migration of old conversation files.
#[derive(Clone, Deserialize)]
struct LegacyStoredMessage {
    role: Role,
    content: String,
}

impl From<LegacyStoredMessage> for StoredMessage {
    fn from(legacy: LegacyStoredMessage) -> Self {
        StoredMessage {
            role: legacy.role,
            content: vec![ContentBlock::text(legacy.content)],
        }
    }
}

/// File names for persistence
const WORKSPACES_FILE: &str = "workspaces.json";
const CONVERSATIONS_FILE: &str = "conversations.json";
const MESSAGES_DIR: &str = "messages";

/// Holds a connected MCP client and its available tools.
struct McpClientHandle {
    client: Arc<TokioMutex<McpClient>>,
    tools: Vec<McpToolInfo>,
    server_name: String,
}

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
                            eprintln!("Error shutting down MCP server '{}' on workspace delete: {}", name, e);
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
        let workspace = workspaces.get_mut(&workspace_id).ok_or_else(|| {
            MuxFfiError::Engine {
                message: format!("Workspace not found: {}", workspace_id),
            }
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
                    eprintln!("Failed to create async runtime for MCP disconnection: {}", e);
                    return;
                }
            };
            rt.block_on(async move {
                engine.do_disconnect_workspace_servers(&workspace_id).await;
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

/// MCP client management methods
impl MuxEngine {
    /// Connect to all enabled MCP servers for a workspace.
    async fn do_connect_workspace_servers(&self, workspace_id: String) -> Result<(), String> {
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
                        "Connected to MCP server '{}' with {} tools",
                        config.name,
                        handle.tools.len()
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

        Ok(McpClientHandle {
            client: Arc::new(TokioMutex::new(client)),
            tools,
            server_name: config.name.clone(),
        })
    }

    /// Disconnect all MCP servers for a workspace.
    async fn do_disconnect_workspace_servers(&self, workspace_id: &str) {
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
    fn get_workspace_tools(&self, workspace_id: &str) -> Vec<ToolDefinition> {
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
    fn parse_tool_name(&self, qualified_name: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = qualified_name.splitn(2, ':').collect();
        if parts.len() == 2 {
            Some((parts[0].to_string(), parts[1].to_string()))
        } else {
            None
        }
    }

    /// Execute a tool call using pre-captured MCP client references.
    /// This is immune to race conditions from workspace disconnection during message processing.
    async fn execute_tool_with_captured_client(
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

impl MuxEngine {
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

    /// Get the workspace ID for a conversation.
    fn get_workspace_for_conversation(&self, conversation_id: &str) -> Option<String> {
        let conversations = self.conversations.read();
        conversations.iter().find_map(|(ws_id, convs)| {
            if convs.iter().any(|c| c.id == conversation_id) {
                Some(ws_id.clone())
            } else {
                None
            }
        })
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
                content: vec![ContentBlock::text(content.clone())],
            });
        }

        // Persist user message to disk
        self.save_messages(&conversation_id);

        // Get current provider and create appropriate client
        let provider = self.default_provider.read().clone();

        // Build client based on provider type (same pattern as execute_task_tool)
        let client: Arc<dyn LlmClient> = match &provider {
            Provider::Custom { name } => {
                // For custom providers, use registered callback
                match self.callback_providers.read().get(name).cloned() {
                    Some(callback_client) => callback_client as Arc<dyn LlmClient>,
                    None => {
                        let error = format!(
                            "Custom LLM provider '{}' not registered. Call register_llm_provider first.",
                            name
                        );
                        callback.on_error(error.clone());
                        return Err(error);
                    }
                }
            }
            _ => {
                // For cloud providers, get API key from config
                let config = self.api_keys.read().get(&provider).cloned();
                match config {
                    Some(c) if !c.api_key.is_empty() => {
                        match &provider {
                            Provider::Anthropic => {
                                Arc::new(AnthropicClient::new(&c.api_key))
                            }
                            Provider::OpenAI | Provider::Ollama => {
                                let mut client = OpenAIClient::new(&c.api_key);
                                if let Some(ref url) = c.base_url {
                                    client = client.with_base_url(url);
                                }
                                Arc::new(client)
                            }
                            Provider::Gemini => {
                                Arc::new(GeminiClient::new(&c.api_key))
                            }
                            Provider::Custom { .. } => unreachable!(),
                        }
                    }
                    _ => {
                        // Fallback to echo if no API key for cloud provider
                        let echo_text = format!("(No API key set) Echo: {}", content);
                        callback.on_text_delta(echo_text.clone());

                        // Store assistant response
                        {
                            let mut history = self.message_history.write();
                            if let Some(messages) = history.get_mut(&conversation_id) {
                                messages.push(StoredMessage {
                                    role: Role::Assistant,
                                    content: vec![ContentBlock::text(echo_text.clone())],
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
                            context_usage: Default::default(),
                        });
                    }
                }
            }
        };

        // Get workspace ID for tool lookup
        let workspace_id = self.get_workspace_for_conversation(&conversation_id);

        // Capture MCP clients for the duration of this message processing.
        // This prevents race conditions if user switches workspaces mid-execution.
        // We clone the Arc<TokioMutex<McpClient>> references so they remain valid
        // even if the workspace is disconnected.
        let captured_mcp_clients: HashMap<String, Arc<TokioMutex<McpClient>>> = workspace_id
            .as_ref()
            .map(|ws_id| {
                let clients = self.mcp_clients.read();
                clients
                    .get(ws_id)
                    .map(|workspace_clients| {
                        workspace_clients
                            .iter()
                            .map(|(name, handle)| (name.clone(), Arc::clone(&handle.client)))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .unwrap_or_default();

        // Get available tools for this workspace
        let tools: Vec<ToolDefinition> = workspace_id
            .as_ref()
            .map(|ws_id| self.get_workspace_tools(ws_id))
            .unwrap_or_default();

        // Get model from workspace config, falling back to provider default
        // For custom providers, use a placeholder since the callback handles model selection internally
        let model = match &provider {
            Provider::Custom { name } => {
                // Custom providers handle model selection internally via callback
                // Use provider name as placeholder for logging/tracking
                name.clone()
            }
            _ => {
                self.get_model_for_conversation(&conversation_id)
                    .or_else(|| self.get_default_model(provider.clone()))
                    .ok_or_else(|| {
                        format!(
                            "No model configured. Set default_model via set_provider_config for {:?}",
                            provider
                        )
                    })?
            }
        };

        let mut total_tool_use_count = 0u32;
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;
        let mut full_text = String::new();

        // Agentic loop: keep going while the LLM wants to use tools
        // Limit iterations to prevent runaway loops from misbehaving LLM/MCP servers
        const MAX_AGENTIC_ITERATIONS: u32 = 50;
        let mut iteration_count = 0u32;

        loop {
            iteration_count += 1;
            if iteration_count > MAX_AGENTIC_ITERATIONS {
                let warning = format!(
                    "\n\n[Agentic loop terminated after {} iterations to prevent runaway execution]",
                    MAX_AGENTIC_ITERATIONS
                );
                callback.on_text_delta(warning.clone());
                full_text.push_str(&warning);
                break;
            }
            // Build message history for context - preserves ToolUse/ToolResult structure
            let messages: Vec<Message> = {
                let history = self.message_history.read();
                history
                    .get(&conversation_id)
                    .map(|msgs| {
                        msgs.iter()
                            .map(|m| Message {
                                role: m.role,
                                content: m.content.clone(),
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };

            // Build system prompt with tool guidance
            let (workspace_path, custom_prompt) = workspace_id
                .as_ref()
                .and_then(|ws_id| {
                    self.workspaces.read().get(ws_id).map(|ws| {
                        (
                            ws.path.clone().unwrap_or_else(|| "~".to_string()),
                            ws.system_prompt.clone(),
                        )
                    })
                })
                .unwrap_or_else(|| ("~".to_string(), None));

            let tool_list: String = tools
                .iter()
                .map(|t| format!("- {}: {}", t.name, t.description))
                .collect::<Vec<_>>()
                .join("\n");

            // Use custom prompt if set, otherwise use default
            let base_prompt = custom_prompt.unwrap_or_else(|| {
                "You are a helpful AI assistant with access to local tools.".to_string()
            });

            let system_prompt = format!(
                "{}\n\n\
                Available tools:\n{}\n\n\
                IMPORTANT: When using file tools, always use ABSOLUTE paths (starting with / or ~).\n\
                The workspace directory is: {}\n\
                For example, use '{}/file.txt' instead of just 'file.txt'.",
                base_prompt,
                tool_list,
                workspace_path,
                workspace_path
            );

            // Create request with tools and system prompt
            let mut request = Request::new(&model)
                .system(&system_prompt)
                .messages(messages)
                .max_tokens(4096);

            if !tools.is_empty() {
                request = request.tools(tools.clone());
            }

            // Use non-streaming for tool use to get complete response
            let response = client
                .create_message(&request)
                .await
                .map_err(|e| e.to_string())?;

            total_input_tokens += response.usage.input_tokens;
            total_output_tokens += response.usage.output_tokens;

            // Process the response content
            // Tuple: (tool_use_id, qualified_name "server:tool", arguments)
            let mut tool_uses: Vec<(String, String, serde_json::Value)> = Vec::new();
            let mut response_text = String::new();

            for block in &response.content {
                match block {
                    ContentBlock::Text { text } => {
                        response_text.push_str(text);
                        callback.on_text_delta(text.clone());
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), input.clone()));
                    }
                    ContentBlock::ToolResult { .. } => {
                        // Should not appear in assistant response
                    }
                }
            }

            full_text.push_str(&response_text);

            // If no tool uses, we're done
            if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
                // Store final assistant response
                if !response_text.is_empty() {
                    let mut history = self.message_history.write();
                    if let Some(messages) = history.get_mut(&conversation_id) {
                        messages.push(StoredMessage {
                            role: Role::Assistant,
                            content: vec![ContentBlock::text(response_text)],
                        });
                    }
                }
                self.save_messages(&conversation_id);
                break;
            }

            // Process tool uses
            let mut tool_results: Vec<ContentBlock> = Vec::new();

            for (tool_use_id, tool_name, arguments) in tool_uses {
                total_tool_use_count += 1;

                // Check if this is a built-in tool (no colon in name)
                let builtin_tool = self.builtin_tools.iter().find(|t| t.name() == tool_name);

                // Check if this is a custom tool
                let custom_tool = self.custom_tools.read().get(&tool_name).cloned();

                if let Some(tool) = builtin_tool {
                    // Execute built-in tool
                    callback.on_tool_use(ToolUseRequest {
                        id: tool_use_id.clone(),
                        tool_name: tool_name.clone(),
                        server_name: "builtin".to_string(),
                        arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                    });

                    match tool.execute(arguments).await {
                        Ok(result) => {
                            callback.on_tool_result(tool_use_id.clone(), result.content.clone());
                            if result.is_error {
                                tool_results.push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results.push(ContentBlock::tool_result(&tool_use_id, &result.content));
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Tool execution failed: {}", e);
                            callback.on_tool_result(tool_use_id.clone(), error_msg.clone());
                            tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                        }
                    }
                } else if let Some(tool) = custom_tool {
                    // Execute custom tool from Swift
                    callback.on_tool_use(ToolUseRequest {
                        id: tool_use_id.clone(),
                        tool_name: tool_name.clone(),
                        server_name: "custom".to_string(),
                        arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                    });

                    match tool.execute(arguments).await {
                        Ok(result) => {
                            callback.on_tool_result(tool_use_id.clone(), result.content.clone());
                            if result.is_error {
                                tool_results.push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results.push(ContentBlock::tool_result(&tool_use_id, &result.content));
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("Tool execution failed: {}", e);
                            callback.on_tool_result(tool_use_id.clone(), error_msg.clone());
                            tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                        }
                    }
                } else if tool_name == "task" {
                    // Execute TaskTool for spawning subagents
                    callback.on_tool_use(ToolUseRequest {
                        id: tool_use_id.clone(),
                        tool_name: tool_name.clone(),
                        server_name: "builtin".to_string(),
                        arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                    });

                    match self.execute_task_tool(arguments.clone()).await {
                        Ok(result) => {
                            callback.on_tool_result(tool_use_id.clone(), result.content.clone());
                            if result.is_error {
                                tool_results.push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results.push(ContentBlock::tool_result(&tool_use_id, &result.content));
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("TaskTool execution failed: {}", e);
                            callback.on_tool_result(tool_use_id.clone(), error_msg.clone());
                            tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                        }
                    }
                } else if let Some((server_name, mcp_tool_name)) = self.parse_tool_name(&tool_name) {
                    // Execute MCP tool (server:tool format)
                    callback.on_tool_use(ToolUseRequest {
                        id: tool_use_id.clone(),
                        tool_name: mcp_tool_name.clone(),
                        server_name: server_name.clone(),
                        arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                    });

                    match Self::execute_tool_with_captured_client(
                        &captured_mcp_clients,
                        &server_name,
                        &mcp_tool_name,
                        arguments,
                    )
                    .await
                    {
                        Ok(result) => {
                            callback.on_tool_result(tool_use_id.clone(), result.clone());
                            tool_results.push(ContentBlock::tool_result(&tool_use_id, &result));
                        }
                        Err(e) => {
                            callback.on_tool_result(tool_use_id.clone(), format!("Error: {}", e));
                            tool_results.push(ContentBlock::tool_error(&tool_use_id, &e));
                        }
                    }
                } else {
                    // Unknown tool format
                    let error_msg = format!("Unknown tool: {}", tool_name);
                    tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                    callback.on_tool_result(tool_use_id.clone(), error_msg);
                }
            }

            // Store assistant response with tool uses - preserves ToolUse blocks
            // This ensures Claude recognizes tool calls on subsequent iterations
            {
                let mut history = self.message_history.write();
                if let Some(messages) = history.get_mut(&conversation_id) {
                    // Store the full assistant response including ToolUse blocks
                    messages.push(StoredMessage {
                        role: Role::Assistant,
                        content: response.content.clone(),
                    });

                    // Store tool results as user message with proper ToolResult blocks
                    if !tool_results.is_empty() {
                        messages.push(StoredMessage {
                            role: Role::User,
                            content: tool_results.clone(),
                        });
                    }
                }
            }
            self.save_messages(&conversation_id);
        }

        Ok(ChatResult {
            conversation_id,
            final_text: full_text,
            tool_use_count: total_tool_use_count,
            input_tokens: total_input_tokens,
            output_tokens: total_output_tokens,
            context_usage: Default::default(),
        })
    }

    /// Execute the TaskTool to spawn a subagent.
    /// This creates an FfiTaskTool with the current engine state and event handler.
    async fn execute_task_tool(
        &self,
        params: serde_json::Value,
    ) -> Result<mux::tool::ToolResult, String> {
        // Check if handler is set (required for TaskTool)
        if self.subagent_event_handler.read().is_none() {
            return Ok(mux::tool::ToolResult::error(
                "TaskTool not available: no subagent event handler registered",
            ));
        }

        // Get provider and validate Custom providers exist before proceeding
        let provider = self.default_provider.read().clone();
        if let Provider::Custom { ref name } = provider {
            if !self.callback_providers.read().contains_key(name) {
                return Err(format!(
                    "Custom LLM provider '{}' not registered. Call register_llm_provider first.",
                    name
                ));
            }
        }

        // Build AgentRegistry from registered agent configs
        let agent_registry = AgentRegistry::new();
        let provider_default_model = self.get_default_model(provider.clone());
        {
            let configs = self.agent_configs.read();
            for (name, config) in configs.iter() {
                let model = config
                    .model
                    .clone()
                    .or_else(|| provider_default_model.clone())
                    .ok_or_else(|| {
                        format!(
                            "No model configured for agent '{}'. Set model in AgentConfig or set default_model via set_provider_config",
                            name
                        )
                    })?;

                let mut definition = AgentDefinition::new(name, &config.system_prompt)
                    .model(&model)
                    .max_iterations(config.max_iterations as usize);

                if !config.allowed_tools.is_empty() {
                    definition = definition.allowed_tools(config.allowed_tools.clone());
                }
                if !config.denied_tools.is_empty() {
                    definition = definition.denied_tools(config.denied_tools.clone());
                }

                agent_registry.register(definition).await;
            }
        }

        // Build tool registry with builtin + custom tools
        let tool_registry = Registry::new();
        for tool in &self.builtin_tools {
            tool_registry.register_arc(tool.clone()).await;
        }
        {
            let custom_tools = self.custom_tools.read();
            for tool in custom_tools.values() {
                tool_registry
                    .register_arc(tool.clone() as Arc<dyn Tool>)
                    .await;
            }
        }

        // For Custom providers, we don't need API key config (provider already read above)
        let provider_config = match &provider {
            Provider::Custom { .. } => None,
            _ => Some(
                self.api_keys
                    .read()
                    .get(&provider)
                    .cloned()
                    .ok_or_else(|| format!("Provider not configured: {:?}", provider))?,
            ),
        };

        // Clone what we need for the client factory closure
        let provider_clone = provider.clone();
        let api_key = provider_config.as_ref().map(|c| c.api_key.clone());
        let base_url = provider_config.as_ref().and_then(|c| c.base_url.clone());
        let callback_providers = self.callback_providers.clone();

        // Create client factory
        let client_factory = move |_model: &str| -> Arc<dyn LlmClient> {
            match &provider_clone {
                Provider::Custom { name } => {
                    callback_providers
                        .read()
                        .get(name)
                        .cloned()
                        .map(|c| c as Arc<dyn LlmClient>)
                        .expect("Custom provider was validated at start of execute_task_tool")
                }
                Provider::Anthropic => {
                    Arc::new(AnthropicClient::new(api_key.as_deref().unwrap_or("")))
                }
                Provider::OpenAI | Provider::Ollama => {
                    let mut c = OpenAIClient::new(api_key.as_deref().unwrap_or(""));
                    if let Some(ref url) = base_url {
                        c = c.with_base_url(url);
                    }
                    Arc::new(c)
                }
                Provider::Gemini => Arc::new(GeminiClient::new(api_key.as_deref().unwrap_or(""))),
            }
        };

        // Get the event handler - we need to clone it for FfiTaskTool
        // Since Box<dyn SubagentEventHandler> isn't Clone, we need a workaround.
        // For now, we'll create a simple proxy that forwards to the stored handler.
        let handler_proxy = TaskToolEventProxy {
            engine_handler: self.subagent_event_handler.clone(),
        };

        let task_tool = FfiTaskTool::new(
            agent_registry,
            tool_registry,
            client_factory,
            Box::new(handler_proxy),
        )
        .with_transcript_store(self.transcript_store.clone());

        task_tool.execute(params).await.map_err(|e| e.to_string())
    }
}

/// Proxy that forwards SubagentEventHandler calls to the engine's stored handler.
struct TaskToolEventProxy {
    engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>>,
}

impl SubagentEventHandler for TaskToolEventProxy {
    fn on_agent_started(
        &self,
        subagent_id: String,
        agent_type: String,
        task: String,
        description: String,
    ) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_agent_started(subagent_id, agent_type, task, description);
        }
    }

    fn on_tool_use(&self, subagent_id: String, tool_name: String, arguments_json: String) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_tool_use(subagent_id, tool_name, arguments_json);
        }
    }

    fn on_tool_result(
        &self,
        subagent_id: String,
        tool_name: String,
        result: String,
        is_error: bool,
    ) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_tool_result(subagent_id, tool_name, result, is_error);
        }
    }

    fn on_iteration(&self, subagent_id: String, iteration: u32) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_iteration(subagent_id, iteration);
        }
    }

    fn on_agent_completed(
        &self,
        subagent_id: String,
        content: String,
        tool_use_count: u32,
        iterations: u32,
        transcript_saved: bool,
    ) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_agent_completed(subagent_id, content, tool_use_count, iterations, transcript_saved);
        }
    }

    fn on_agent_error(&self, subagent_id: String, error: String) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_agent_error(subagent_id, error);
        }
    }
}

/// Persistence helper methods
impl MuxEngine {
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
    /// Handles migration from legacy format (String content) to new format (Vec<ContentBlock>).
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
                    // Try new format first
                    if let Ok(msgs) = serde_json::from_str::<Vec<StoredMessage>>(&contents) {
                        all_messages.insert(conv.id.clone(), msgs);
                    }
                    // Fall back to legacy format (String content)
                    else if let Ok(legacy_msgs) =
                        serde_json::from_str::<Vec<LegacyStoredMessage>>(&contents)
                    {
                        let msgs: Vec<StoredMessage> =
                            legacy_msgs.into_iter().map(StoredMessage::from).collect();
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

/// A hook that proxies tool events to the SubagentCallback.
/// This enables streaming tool use/result events to Swift during subagent execution.
struct CallbackProxyHook {
    agent_id: String,
    callback: Arc<Box<dyn SubagentCallback>>,
}

impl CallbackProxyHook {
    fn new(agent_id: String, callback: Arc<Box<dyn SubagentCallback>>) -> Self {
        Self { agent_id, callback }
    }
}

#[async_trait::async_trait]
impl mux::hook::Hook for CallbackProxyHook {
    async fn on_event(
        &self,
        event: &mux::hook::HookEvent,
    ) -> Result<mux::hook::HookAction, anyhow::Error> {
        match event {
            mux::hook::HookEvent::PreToolUse { tool_name, input } => {
                // Notify callback of tool use
                self.callback.on_tool_use(
                    self.agent_id.clone(),
                    ToolUseRequest {
                        id: uuid::Uuid::new_v4().to_string(),
                        tool_name: tool_name.clone(),
                        server_name: "builtin".to_string(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                );
            }
            mux::hook::HookEvent::PostToolUse {
                tool_name: _,
                input: _,
                result,
            } => {
                // Notify callback of tool result
                self.callback.on_tool_result(
                    self.agent_id.clone(),
                    "".to_string(), // tool_id not available from hook
                    result.content.clone(),
                );
            }
            _ => {
                // Other events (AgentStart, AgentStop, Iteration) don't map to callback methods
            }
        }
        Ok(mux::hook::HookAction::Continue)
    }

    fn accepts(&self, _event: &mux::hook::HookEvent) -> bool {
        true
    }
}

/// Subagent implementation methods.
impl MuxEngine {
    /// Internal implementation of spawn_agent.
    async fn do_spawn_agent(
        &self,
        _workspace_id: String,
        agent_name: String,
        task: String,
        save_transcript: bool,
        callback: Arc<Box<dyn SubagentCallback>>,
    ) -> Result<SubagentResult, String> {
        // Get agent config
        let config = self
            .agent_configs
            .read()
            .get(&agent_name)
            .cloned()
            .ok_or_else(|| format!("Agent not found: {}", agent_name))?;

        // Get provider and create LLM client
        let provider = self.default_provider.read().clone();
        let client: Arc<dyn LlmClient> = match &provider {
            Provider::Custom { name } => {
                self.callback_providers
                    .read()
                    .get(name)
                    .cloned()
                    .map(|c| c as Arc<dyn LlmClient>)
                    .ok_or_else(|| format!("Callback provider '{}' not registered", name))?
            }
            _ => {
                let provider_config = self
                    .api_keys
                    .read()
                    .get(&provider)
                    .cloned()
                    .ok_or_else(|| format!("Provider not configured: {:?}", provider))?;

                match provider {
                    Provider::Anthropic => {
                        Arc::new(AnthropicClient::new(&provider_config.api_key))
                    }
                    Provider::OpenAI | Provider::Ollama => {
                        let mut c = OpenAIClient::new(&provider_config.api_key);
                        if let Some(url) = &provider_config.base_url {
                            c = c.with_base_url(url);
                        }
                        Arc::new(c)
                    }
                    Provider::Gemini => Arc::new(GeminiClient::new(&provider_config.api_key)),
                    Provider::Custom { .. } => unreachable!(),
                }
            }
        };

        // Build tool registry with built-in tools
        let registry = Registry::new();
        for tool in &self.builtin_tools {
            registry.register_arc(tool.clone()).await;
        }

        // Add custom tools to registry
        {
            let custom_tools = self.custom_tools.read();
            for tool in custom_tools.values() {
                registry.register_arc(tool.clone() as Arc<dyn Tool>).await;
            }
        }

        // Create agent definition from config with allowed/denied tools
        // Get default model from provider config (not available for Custom providers)
        let default_model = self.get_default_model(provider.clone());
        let model = config
            .model
            .clone()
            .or(default_model)
            .ok_or_else(|| {
                format!(
                    "No model configured for agent '{}'. Set model in AgentConfig or set default_model via set_provider_config",
                    agent_name
                )
            })?;
        let mut definition = AgentDefinition::new(&agent_name, &config.system_prompt)
            .model(&model)
            .max_iterations(config.max_iterations as usize);

        // Apply allowed tools (empty means all allowed, so only set if non-empty)
        if !config.allowed_tools.is_empty() {
            definition = definition.allowed_tools(config.allowed_tools.clone());
        }

        // Apply denied tools
        if !config.denied_tools.is_empty() {
            definition = definition.denied_tools(config.denied_tools.clone());
        }

        // Create subagent
        let mut subagent = SubAgent::new(definition, client, registry);
        let agent_id = subagent.agent_id().to_string();

        // Wire up callback via hook for tool events
        // We always want to proxy tool events to the callback, regardless of whether
        // a user hook handler is set
        let hook_registry = HookRegistry::new();
        let proxy_hook = CallbackProxyHook::new(agent_id.clone(), callback.clone());
        hook_registry.register(proxy_hook).await;

        // Note: User-provided hook handlers are stored in self.hook_handler, but
        // we can't clone Box<dyn HookHandler> to pass to the subagent. The user's
        // hook functionality is available via the callback interface for now.
        // Future: Support cloneable hook handlers or use a different pattern.
        let _ = &self.hook_handler; // Acknowledge hook_handler exists

        subagent = subagent.with_hooks(Arc::new(hook_registry));

        let result = subagent.run(&task).await.map_err(|e| e.to_string())?;

        // Optionally save transcript
        let transcript_json = if save_transcript {
            Some(serde_json::to_string(subagent.transcript()).unwrap_or_default())
        } else {
            None
        };

        Ok(SubagentResult {
            agent_id: agent_id.clone(),
            content: result.content,
            tool_use_count: result.tool_use_count as u32,
            iterations: result.iterations as u32,
            transcript_json,
        })
    }

    /// Internal implementation of resume_agent.
    async fn do_resume_agent(
        &self,
        transcript: TranscriptData,
        callback: Arc<Box<dyn SubagentCallback>>,
    ) -> Result<SubagentResult, String> {
        // Parse transcript messages
        let messages: Vec<Message> = serde_json::from_str(&transcript.messages_json)
            .map_err(|e| format!("Invalid transcript JSON: {}", e))?;

        // Get provider and create LLM client (all providers supported for resume)
        let provider = self.default_provider.read().clone();
        let client: Arc<dyn LlmClient> = match &provider {
            Provider::Custom { name } => {
                self.callback_providers
                    .read()
                    .get(name)
                    .cloned()
                    .map(|c| c as Arc<dyn LlmClient>)
                    .ok_or_else(|| format!("Callback provider '{}' not registered", name))?
            }
            _ => {
                let provider_config = self
                    .api_keys
                    .read()
                    .get(&provider)
                    .cloned()
                    .ok_or_else(|| format!("Provider not configured: {:?}", provider))?;

                match provider {
                    Provider::Anthropic => {
                        Arc::new(AnthropicClient::new(&provider_config.api_key))
                    }
                    Provider::OpenAI | Provider::Ollama => {
                        let mut c = OpenAIClient::new(&provider_config.api_key);
                        if let Some(url) = &provider_config.base_url {
                            c = c.with_base_url(url);
                        }
                        Arc::new(c)
                    }
                    Provider::Gemini => Arc::new(GeminiClient::new(&provider_config.api_key)),
                    Provider::Custom { .. } => unreachable!(),
                }
            }
        };

        // Build tool registry with built-in tools
        let registry = Registry::new();
        for tool in &self.builtin_tools {
            registry.register_arc(tool.clone()).await;
        }

        // Add custom tools to registry
        {
            let custom_tools = self.custom_tools.read();
            for tool in custom_tools.values() {
                registry.register_arc(tool.clone() as Arc<dyn Tool>).await;
            }
        }

        // Create definition for resume
        let definition = AgentDefinition::new("resumed", "You are a helpful assistant.")
            .max_iterations(10);

        // Resume agent with transcript
        let mut subagent = SubAgent::resume(
            transcript.agent_id.clone(),
            definition,
            client,
            registry,
            messages,
        );

        // Wire up callback via hook for tool events
        let hook_registry = HookRegistry::new();
        let proxy_hook = CallbackProxyHook::new(transcript.agent_id.clone(), callback.clone());
        hook_registry.register(proxy_hook).await;
        subagent = subagent.with_hooks(Arc::new(hook_registry));

        let result = subagent
            .run("Continue from where you left off.")
            .await
            .map_err(|e| e.to_string())?;

        Ok(SubagentResult {
            agent_id: result.agent_id,
            content: result.content,
            tool_use_count: result.tool_use_count as u32,
            iterations: result.iterations as u32,
            transcript_json: Some(
                serde_json::to_string(subagent.transcript()).unwrap_or_default(),
            ),
        })
    }
}
