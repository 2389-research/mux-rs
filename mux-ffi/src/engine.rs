// ABOUTME: BuddyEngine - the main entry point for the FFI layer.
// ABOUTME: Manages workspaces, conversations, and bridges to mux core.

use crate::callback::{ChatCallback, ChatResult, ToolUseRequest};
use crate::types::{
    ApprovalDecision, Conversation, McpServerConfig, McpTransportType, Provider, Workspace,
    WorkspaceSummary,
};
use crate::MuxFfiError;
use mux::prelude::{
    AnthropicClient, ContentBlock, LlmClient, McpClient, McpContentBlock,
    McpServerConfig as MuxMcpServerConfig, McpToolInfo, McpTransport, Message, Request, Role,
    StopReason, ToolDefinition,
};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex as TokioMutex;

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

/// Holds a connected MCP client and its available tools.
struct McpClientHandle {
    client: Arc<TokioMutex<McpClient>>,
    tools: Vec<McpToolInfo>,
    server_name: String,
}

#[derive(uniffi::Object)]
pub struct BuddyEngine {
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
    api_keys: Arc<RwLock<HashMap<Provider, String>>>,
    /// Connected MCP clients, keyed by workspace_id -> server_name -> handle
    mcp_clients: Arc<RwLock<HashMap<String, HashMap<String, McpClientHandle>>>>,
    /// Pending tool approval requests, keyed by tool_use_id -> oneshot sender
    pending_approvals:
        Arc<RwLock<HashMap<String, tokio::sync::oneshot::Sender<ApprovalDecision>>>>,
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
            mcp_clients: Arc::new(RwLock::new(HashMap::new())),
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
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

    /// Connect to all enabled MCP servers for a workspace.
    /// This should be called when entering a workspace to establish connections.
    pub fn connect_workspace_servers(self: Arc<Self>, workspace_id: String) {
        let engine = self.clone();
        std::thread::spawn(move || {
            let rt = Runtime::new().unwrap();
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
            let rt = Runtime::new().unwrap();
            rt.block_on(async move {
                engine.do_disconnect_workspace_servers(&workspace_id).await;
            });
        });
    }
}

/// MCP client management methods
impl BuddyEngine {
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
    fn get_workspace_tools(&self, workspace_id: &str) -> Vec<ToolDefinition> {
        let clients = self.mcp_clients.read();
        let workspace_clients = match clients.get(workspace_id) {
            Some(c) => c,
            None => return Vec::new(),
        };

        workspace_clients
            .values()
            .flat_map(|handle| {
                handle.tools.iter().map(|tool| ToolDefinition {
                    name: format!("{}:{}", handle.server_name, tool.name),
                    description: tool.description.clone(),
                    input_schema: tool.input_schema.clone(),
                })
            })
            .collect()
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

    /// Execute a tool call on an MCP server.
    async fn execute_tool(
        &self,
        workspace_id: &str,
        server_name: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, String> {
        let clients = self.mcp_clients.read();
        let workspace_clients = clients
            .get(workspace_id)
            .ok_or_else(|| "Workspace not connected".to_string())?;
        let handle = workspace_clients
            .get(server_name)
            .ok_or_else(|| format!("Server '{}' not connected", server_name))?;

        let client = handle.client.lock().await;
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

        // Get workspace ID for tool lookup
        let workspace_id = self.get_workspace_for_conversation(&conversation_id);

        // Get available tools for this workspace
        let tools: Vec<ToolDefinition> = workspace_id
            .as_ref()
            .map(|ws_id| self.get_workspace_tools(ws_id))
            .unwrap_or_default();

        // Get model from workspace config, falling back to default
        let model = self
            .get_model_for_conversation(&conversation_id)
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

        let mut total_tool_use_count = 0u32;
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;
        let mut full_text = String::new();

        // Agentic loop: keep going while the LLM wants to use tools
        loop {
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

            // Create request with tools
            let mut request = Request::new(&model).messages(messages).max_tokens(4096);

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
            let mut tool_uses: Vec<(String, String, String, serde_json::Value)> = Vec::new();
            let mut response_text = String::new();

            for block in &response.content {
                match block {
                    ContentBlock::Text { text } => {
                        response_text.push_str(text);
                        callback.on_text_delta(text.clone());
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), name.clone(), input.clone()));
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
                            content: response_text,
                        });
                    }
                }
                self.save_messages(&conversation_id);
                break;
            }

            // Process tool uses
            let mut tool_results: Vec<ContentBlock> = Vec::new();

            for (tool_use_id, qualified_name, _display_name, arguments) in tool_uses {
                total_tool_use_count += 1;

                // Parse server:tool format
                let (server_name, tool_name) = match self.parse_tool_name(&qualified_name) {
                    Some((s, t)) => (s, t),
                    None => {
                        let error_msg = format!("Invalid tool name format: {}", qualified_name);
                        tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                        callback.on_tool_result(tool_use_id.clone(), error_msg);
                        continue;
                    }
                };

                // Notify callback about tool use
                callback.on_tool_use(ToolUseRequest {
                    id: tool_use_id.clone(),
                    tool_name: tool_name.clone(),
                    server_name: server_name.clone(),
                    arguments: serde_json::to_string(&arguments).unwrap_or_default(),
                });

                // For MVP: Auto-approve all tool calls
                // In future, we could wait for approval via pending_approvals

                // Execute the tool
                if let Some(ws_id) = &workspace_id {
                    match self
                        .execute_tool(ws_id, &server_name, &tool_name, arguments)
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
                    let error_msg = "No workspace context for tool execution";
                    callback.on_tool_result(tool_use_id.clone(), error_msg.to_string());
                    tool_results.push(ContentBlock::tool_error(&tool_use_id, error_msg));
                }
            }

            // Store assistant response with tool uses in history as text representation
            {
                let mut history = self.message_history.write();
                if let Some(messages) = history.get_mut(&conversation_id) {
                    // Store assistant message
                    if !response_text.is_empty() {
                        messages.push(StoredMessage {
                            role: Role::Assistant,
                            content: response_text,
                        });
                    }
                    // Store tool results as user message
                    let tool_results_text: String = tool_results
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content,
                                is_error,
                            } => Some(format!(
                                "[Tool Result {}{}]: {}",
                                tool_use_id,
                                if *is_error { " (error)" } else { "" },
                                content
                            )),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !tool_results_text.is_empty() {
                        messages.push(StoredMessage {
                            role: Role::User,
                            content: tool_results_text,
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
