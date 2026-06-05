// ABOUTME: Message handling using SubAgent for unified agentic execution.
// ABOUTME: All tool execution goes through SubAgent with hooks for callbacks.

use super::MuxEngine;
use super::persistence::StoredMessage;
use super::subagent::TaskToolEventProxy;
use super::tool_wrappers::{CustomToolWrapper, McpToolWrapper};
use crate::callback::{ChatCallback, ChatResult, ToolUseRequest};
use crate::task_tool::FfiTaskTool;
use crate::types::Provider;
use async_trait::async_trait;
use mux::agent::{AgentDefinition, AgentRegistry, SubAgent};
use mux::hook::{Hook, HookAction, HookEvent, HookRegistry};
use mux::llm::{GeminiClient, MediaSource};
use mux::prelude::{
    AnthropicClient, ContentBlock, LlmClient, McpClient, Message, OpenAIClient, Registry, Role,
};
use mux::tool::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// Hook that proxies SubAgent events to ChatCallback for streaming UI updates.
struct ChatCallbackHook {
    callback: Arc<Box<dyn ChatCallback>>,
}

impl ChatCallbackHook {
    fn new(callback: Arc<Box<dyn ChatCallback>>) -> Self {
        Self { callback }
    }
}

#[async_trait]
impl Hook for ChatCallbackHook {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        let callback = self.callback.clone();

        match event {
            HookEvent::ResponseReceived {
                text, tool_uses, ..
            } => {
                // Stream text to callback
                if !text.is_empty() {
                    let text = text.clone();
                    tokio::task::spawn_blocking(move || {
                        callback.on_text_delta(text);
                    })
                    .await
                    .ok();
                }

                // Notify about tool uses
                for (name, id, input) in tool_uses {
                    let callback = self.callback.clone();
                    let request = ToolUseRequest {
                        id: id.clone(),
                        tool_name: name.clone(),
                        server_name: String::new(), // Not an MCP tool
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    };
                    tokio::task::spawn_blocking(move || {
                        callback.on_tool_use(request);
                    })
                    .await
                    .ok();
                }
            }
            HookEvent::PostToolUse {
                tool_use_id,
                result,
                ..
            } => {
                let callback = self.callback.clone();
                let tool_id = tool_use_id.clone();
                let content = result.content.clone();

                tokio::task::spawn_blocking(move || {
                    callback.on_tool_result(tool_id, content);
                })
                .await
                .ok();
            }
            _ => {}
        }

        Ok(HookAction::Continue)
    }

    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(
            event,
            HookEvent::ResponseReceived { .. } | HookEvent::PostToolUse { .. }
        )
    }
}

/// Convert `FfiMedia` attachments into core `ContentBlock::Media` blocks,
/// resolving any `MediaSource::Path` to `MediaSource::Base64` so the blocks
/// are portable (no host-dependent filesystem paths) when persisted to
/// conversation history. Non-Path sources pass through unchanged.
///
/// Consolidating the Path -> Base64 step at the FFI boundary means the
/// resolved bytes are shared by both the persisted user message and the LLM
/// request, avoiding double reads.
async fn ffi_media_into_blocks(
    media: Vec<crate::media::FfiMedia>,
) -> Result<Vec<ContentBlock>, String> {
    let mut blocks = Vec::with_capacity(media.len());
    // `resolve_to_base64` takes a `&reqwest::Client` but does not use it for
    // Path sources — a throwaway client is fine here.
    let http = reqwest::Client::new();
    for m in media {
        let block = m.into_content_block();
        let block = match block {
            ContentBlock::Media {
                kind,
                source: MediaSource::Path(ref p),
                ref mime_type,
            } => {
                let path_src = MediaSource::Path(p.clone());
                let mime_hint = mime_type.clone();
                let (data, mime) = mux::llm::resolve_to_base64(&path_src, &mime_hint, &http)
                    .await
                    .map_err(|e| format!("media resolution failed: {}", e))?;
                ContentBlock::Media {
                    kind,
                    source: MediaSource::Base64(data),
                    mime_type: mime,
                }
            }
            other => other,
        };
        blocks.push(block);
    }
    Ok(blocks)
}

/// Messaging implementation using SubAgent for unified agentic execution.
impl MuxEngine {
    /// Build a tool Registry containing all available tools for this conversation.
    async fn build_tool_registry(
        &self,
        workspace_id: &Option<String>,
        captured_mcp_clients: &HashMap<String, Arc<TokioMutex<McpClient>>>,
    ) -> Registry {
        let registry = Registry::new();

        // Add built-in tools (already Arc-wrapped)
        for tool in &self.builtin_tools {
            registry.register_arc(tool.clone()).await;
        }

        // Collect MCP tool wrappers while holding lock, then register after releasing
        let mcp_wrappers: Vec<McpToolWrapper> = if let Some(ws_id) = workspace_id {
            let clients = self.mcp_clients.read();
            clients
                .get(ws_id)
                .map(|workspace_clients| {
                    workspace_clients
                        .iter()
                        .flat_map(|(server_name, handle)| {
                            handle.tools.iter().filter_map(|tool_def| {
                                captured_mcp_clients.get(server_name).map(|client| {
                                    McpToolWrapper::new(
                                        server_name.clone(),
                                        tool_def.name.clone(),
                                        tool_def.description.clone(),
                                        tool_def.input_schema.clone(),
                                        client.clone(),
                                    )
                                })
                            })
                        })
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        // Lock released, now register
        for wrapper in mcp_wrappers {
            registry.register(wrapper).await;
        }

        // Collect custom tool wrappers while holding lock, then register after releasing
        let custom_wrappers: Vec<CustomToolWrapper> = {
            let custom_tools = self.custom_tools.read();
            custom_tools
                .values()
                .map(|bridge| CustomToolWrapper::new(bridge.clone()))
                .collect()
        };
        // Lock released, now register
        for wrapper in custom_wrappers {
            registry.register(wrapper).await;
        }

        registry
    }

    pub(super) async fn do_send_message(
        &self,
        conversation_id: String,
        content: String,
        media: Vec<crate::media::FfiMedia>,
        callback: Arc<Box<dyn ChatCallback>>,
    ) -> Result<ChatResult, String> {
        // Resolve any MediaSource::Path attachments to Base64 at the FFI
        // boundary so the Vec<ContentBlock> we persist into conversation
        // history (and hand to the LLM) never embeds host-dependent filesystem
        // paths. Done once up front so echo-fallback and real-LLM paths share
        // the same resolved blocks.
        let media_blocks = match ffi_media_into_blocks(media).await {
            Ok(blocks) => blocks,
            Err(e) => {
                callback.on_error(e.clone());
                return Err(e);
            }
        };

        // Get current provider and create appropriate client
        let provider = self.default_provider.read().clone();

        // Build client based on provider type
        let client: Arc<dyn LlmClient> = match &provider {
            Provider::Custom { name } => match self.callback_providers.read().get(name).cloned() {
                Some(callback_client) => callback_client as Arc<dyn LlmClient>,
                None => {
                    let error = format!(
                        "Custom LLM provider '{}' not registered. Call register_llm_provider first.",
                        name
                    );
                    callback.on_error(error.clone());
                    return Err(error);
                }
            },
            _ => {
                let config = self.api_keys.read().get(&provider).cloned();
                match config {
                    Some(c) if !c.api_key.is_empty() => match &provider {
                        Provider::Anthropic => Arc::new(AnthropicClient::new(&c.api_key)),
                        Provider::OpenAI | Provider::Ollama => {
                            let mut client = OpenAIClient::new(&c.api_key);
                            if let Some(ref url) = c.base_url {
                                client = client.with_base_url(url);
                            }
                            Arc::new(client)
                        }
                        Provider::Gemini => Arc::new(GeminiClient::new(&c.api_key)),
                        Provider::Custom { .. } => unreachable!(),
                    },
                    _ => {
                        // Fallback to echo if no API key
                        let echo_text = format!("(No API key set) Echo: {}", content);
                        callback.on_text_delta(echo_text.clone());

                        // Build user content blocks with any attached media.
                        // media_blocks was resolved up front; any Path sources
                        // have already been normalized to Base64.
                        let mut user_blocks: Vec<ContentBlock> =
                            Vec::with_capacity(1 + media_blocks.len());
                        if !content.is_empty() {
                            user_blocks.push(ContentBlock::text(content.clone()));
                        }
                        for block in media_blocks {
                            user_blocks.push(block);
                        }

                        // Store in history
                        {
                            let mut history = self.message_history.write();
                            let messages = history.entry(conversation_id.clone()).or_default();
                            messages.push(StoredMessage {
                                role: Role::User,
                                content: user_blocks,
                            });
                            messages.push(StoredMessage {
                                role: Role::Assistant,
                                content: vec![ContentBlock::text(echo_text.clone())],
                            });
                        }
                        self.save_messages(&conversation_id);
                        self.check_and_warn_context(&conversation_id, callback.as_ref().as_ref());

                        return Ok(ChatResult {
                            conversation_id: conversation_id.clone(),
                            final_text: echo_text,
                            tool_use_count: 0,
                            input_tokens: 0,
                            output_tokens: 0,
                            context_usage: self
                                .get_context_usage(conversation_id.clone())
                                .unwrap_or_default(),
                        });
                    }
                }
            }
        };

        // Get workspace and model configuration
        let workspace_id = self.get_workspace_for_conversation(&conversation_id);
        let model = match &provider {
            Provider::Custom { name } => name.clone(),
            _ => self
                .get_model_for_conversation(&conversation_id)
                .or_else(|| self.get_default_model(provider.clone()))
                .ok_or_else(|| {
                    format!(
                        "No model configured. Set default_model via set_provider_config for {:?}",
                        provider
                    )
                })?,
        };

        // Capture MCP clients to prevent race conditions
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

        // Build tool Registry with all available tools
        let tool_registry = self
            .build_tool_registry(&workspace_id, &captured_mcp_clients)
            .await;

        // Build system prompt
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

        let tool_list: String = tool_registry
            .to_definitions()
            .await
            .iter()
            .map(|t| format!("- {}: {}", t.name, t.description))
            .collect::<Vec<_>>()
            .join("\n");

        let base_prompt = custom_prompt.unwrap_or_else(|| {
            "You are a helpful AI assistant with access to local tools.".to_string()
        });

        let system_prompt = format!(
            "{}\n\n\
            Available tools:\n{}\n\n\
            IMPORTANT: When using file tools, always use ABSOLUTE paths (starting with / or ~).\n\
            The workspace directory is: {}\n\
            For example, use '{}/file.txt' instead of just 'file.txt'.",
            base_prompt, tool_list, workspace_path, workspace_path
        );

        // Create AgentDefinition with iteration limit
        const MAX_AGENTIC_ITERATIONS: usize = 50;
        let definition = AgentDefinition::new("chat", &system_prompt)
            .model(&model)
            .max_iterations(MAX_AGENTIC_ITERATIONS);

        // Get existing conversation history
        let existing_messages: Vec<Message> = {
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

        // Create SubAgent with conversation history
        let mut subagent = if existing_messages.is_empty() {
            SubAgent::new(definition, client, tool_registry)
        } else {
            // Use a unique ID for this conversation's agent
            SubAgent::resume(
                conversation_id.clone(),
                definition,
                client,
                tool_registry,
                existing_messages,
            )
        };

        // Attach hook registry with ChatCallbackHook for streaming
        let hook_registry = Arc::new(HookRegistry::new());
        hook_registry
            .register(ChatCallbackHook::new(callback.clone()))
            .await;
        subagent = subagent.with_hooks(hook_registry);

        // Run the agent with the user's message. If media attachments are
        // present, dispatch through `run_with_blocks` so the media blocks
        // reach the provider alongside the text; otherwise use the plain
        // text path. media_blocks was resolved once up front (Path -> Base64)
        // and is reused here.
        let run_result = if media_blocks.is_empty() {
            subagent.run(&content).await
        } else {
            let mut blocks: Vec<ContentBlock> = Vec::with_capacity(1 + media_blocks.len());
            if !content.is_empty() {
                blocks.push(ContentBlock::text(content.clone()));
            }
            for block in media_blocks {
                blocks.push(block);
            }
            subagent.run_with_blocks(blocks).await
        };
        let result = match run_result {
            Ok(result) => result,
            Err(e) => {
                let error_str = e.to_string();
                // Check if this is a max iterations error - handle gracefully
                if error_str.contains("exceeded max iterations") {
                    // Capture actual usage and tool count from the subagent
                    let actual_usage = subagent.usage().clone();
                    let actual_tool_count = subagent.tool_use_count();
                    let termination_msg = format!(
                        "Agent loop terminated after {} iterations to prevent infinite loops.",
                        MAX_AGENTIC_ITERATIONS
                    );
                    mux::agent::SubAgentResult {
                        agent_id: subagent.agent_id().to_string(),
                        content: termination_msg,
                        tool_use_count: actual_tool_count,
                        usage: actual_usage,
                        iterations: MAX_AGENTIC_ITERATIONS,
                    }
                } else {
                    // On other errors, return without saving transcript.
                    // This means the failed attempt is lost, but the conversation
                    // remains consistent - user can retry with the same message.
                    let error_msg = format!("Agent error: {}", e);
                    callback.on_error(error_msg.clone());
                    return Err(error_msg);
                }
            }
        };

        // Extract transcript and save to history
        let transcript = subagent.transcript();
        {
            let mut history = self.message_history.write();
            let messages = history.entry(conversation_id.clone()).or_default();
            messages.clear();
            for msg in transcript {
                messages.push(StoredMessage {
                    role: msg.role,
                    content: msg.content.clone(),
                });
            }
        }
        self.save_messages(&conversation_id);

        // Check context warning
        self.check_and_warn_context(&conversation_id, callback.as_ref().as_ref());

        // Return result
        let context_usage = self
            .get_context_usage(conversation_id.clone())
            .unwrap_or_default();
        Ok(ChatResult {
            conversation_id,
            final_text: result.content,
            tool_use_count: result.tool_use_count as u32,
            input_tokens: result.usage.input_tokens,
            output_tokens: result.usage.output_tokens,
            context_usage,
        })
    }

    /// Execute the TaskTool to spawn a subagent.
    /// This creates an FfiTaskTool with the current engine state and event handler.
    // Unwired: the task/subagent tool is implemented and tested but not yet dispatched from the
    // production chat loop. Retained until wired. See #9.
    // (lock-across-await at the registry loops rides with this unwired method; see #9)
    #[allow(dead_code)]
    #[allow(clippy::await_holding_lock)]
    pub(super) async fn execute_task_tool(
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
        if let Provider::Custom { ref name } = provider
            && !self.callback_providers.read().contains_key(name)
        {
            return Err(format!(
                "Custom LLM provider '{}' not registered. Call register_llm_provider first.",
                name
            ));
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

        // For custom providers, capture the client Arc upfront to avoid race conditions
        // (provider could be unregistered between validation and factory execution)
        let captured_custom_client: Option<Arc<dyn LlmClient>> = match &provider {
            Provider::Custom { name } => self
                .callback_providers
                .read()
                .get(name)
                .cloned()
                .map(|c| c as Arc<dyn LlmClient>),
            _ => None,
        };

        // Create client factory
        let client_factory = move |_model: &str| -> Arc<dyn LlmClient> {
            match &provider_clone {
                Provider::Custom { .. } => captured_custom_client
                    .clone()
                    .expect("Custom provider was captured at start of execute_task_tool"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::callback::{ChatCallback, SubagentEventHandler};
    use crate::context::ContextUsage;
    use crate::types::AgentConfig;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn test_dir(name: &str) -> String {
        std::env::temp_dir()
            .join(name)
            .to_string_lossy()
            .to_string()
    }

    fn create_test_engine() -> Arc<MuxEngine> {
        MuxEngine::new(test_dir("mux-test-messaging")).unwrap()
    }

    // Mock callback that tracks calls
    struct TrackingCallback {
        text_received: std::sync::Mutex<String>,
        error_received: std::sync::Mutex<Option<String>>,
        complete_called: AtomicBool,
    }

    impl TrackingCallback {
        fn new() -> Self {
            Self {
                text_received: std::sync::Mutex::new(String::new()),
                error_received: std::sync::Mutex::new(None),
                complete_called: AtomicBool::new(false),
            }
        }
    }

    impl ChatCallback for TrackingCallback {
        fn on_text_delta(&self, text: String) {
            self.text_received.lock().unwrap().push_str(&text);
        }

        fn on_tool_use(&self, _request: ToolUseRequest) {}

        fn on_tool_result(&self, _tool_use_id: String, _result: String) {}

        fn on_complete(&self, _result: ChatResult) {
            self.complete_called.store(true, Ordering::SeqCst);
        }

        fn on_error(&self, error: String) {
            *self.error_received.lock().unwrap() = Some(error);
        }

        fn on_context_warning(&self, _usage: ContextUsage) {}
    }

    #[test]
    fn test_do_send_message_no_api_key_echo_fallback() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Msg Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Don't set any API key - should trigger echo fallback
        let callback = Arc::new(TrackingCallback::new());
        let _cb: Arc<Box<dyn ChatCallback>> = Arc::new(Box::new(TrackingCallback::new()));

        // Need to run async code
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Hello world".to_string(),
            Vec::new(),
            Arc::new(Box::new({
                struct Wrapper(Arc<TrackingCallback>);
                impl ChatCallback for Wrapper {
                    fn on_text_delta(&self, text: String) {
                        self.0.on_text_delta(text);
                    }
                    fn on_tool_use(&self, r: ToolUseRequest) {
                        self.0.on_tool_use(r);
                    }
                    fn on_tool_result(&self, id: String, result: String) {
                        self.0.on_tool_result(id, result);
                    }
                    fn on_complete(&self, r: ChatResult) {
                        self.0.on_complete(r);
                    }
                    fn on_error(&self, e: String) {
                        self.0.on_error(e);
                    }
                    fn on_context_warning(&self, u: ContextUsage) {
                        self.0.on_context_warning(u);
                    }
                }
                Wrapper(callback.clone())
            })),
        ));

        assert!(result.is_ok());
        let chat_result = result.unwrap();

        // Should return echo message
        assert!(chat_result.final_text.contains("Echo: Hello world"));
        assert!(chat_result.final_text.contains("No API key set"));
        assert_eq!(chat_result.tool_use_count, 0);
        assert_eq!(chat_result.input_tokens, 0);
        assert_eq!(chat_result.output_tokens, 0);

        // Callback should have received the text
        let text = callback.text_received.lock().unwrap();
        assert!(text.contains("Echo: Hello world"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_custom_provider_not_registered() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Custom Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Set custom provider but don't register it
        engine.set_default_provider(Provider::Custom {
            name: "my-custom-llm".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Hello".to_string(),
            Vec::new(),
            Arc::new(Box::new({
                struct Wrapper(Arc<TrackingCallback>);
                impl ChatCallback for Wrapper {
                    fn on_text_delta(&self, text: String) {
                        self.0.on_text_delta(text);
                    }
                    fn on_tool_use(&self, r: ToolUseRequest) {
                        self.0.on_tool_use(r);
                    }
                    fn on_tool_result(&self, id: String, result: String) {
                        self.0.on_tool_result(id, result);
                    }
                    fn on_complete(&self, r: ChatResult) {
                        self.0.on_complete(r);
                    }
                    fn on_error(&self, e: String) {
                        self.0.on_error(e);
                    }
                    fn on_context_warning(&self, u: ContextUsage) {
                        self.0.on_context_warning(u);
                    }
                }
                Wrapper(callback.clone())
            })),
        ));

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("my-custom-llm"));
        assert!(error.contains("not registered"));

        // Callback should have received the error
        let error_received = callback.error_received.lock().unwrap();
        assert!(error_received.is_some());
        assert!(error_received.as_ref().unwrap().contains("my-custom-llm"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_media_attached_to_user_message_echo_fallback() {
        // Verifies the `media` parameter plumbed through send_message ->
        // do_send_message is actually attached to the stored user message.
        // Uses the echo-fallback path (no API key) which is the Message::user
        // construction site inside do_send_message itself.
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Media Attach Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        let media = vec![crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Base64 {
                data: "aGVsbG8=".to_string(),
            },
            mime_type: "image/png".to_string(),
        }];

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Describe this image".to_string(),
            media,
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        assert!(result.is_ok());

        // The stored user message should contain both the text block and the
        // media block, confirming `media` reached the Message construction.
        let history = engine.message_history.read();
        let messages = history.get(&conv.id).expect("history entry for conv");
        let user_msg = messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .expect("user message stored");
        assert_eq!(
            user_msg.content.len(),
            2,
            "user message should have text + media blocks"
        );
        assert!(matches!(user_msg.content[0], ContentBlock::Text { .. }));
        assert!(matches!(
            user_msg.content[1],
            ContentBlock::Media {
                kind: mux::llm::MediaKind::Image,
                ..
            }
        ));
        drop(history);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_execute_task_tool_no_handler() {
        let engine = create_test_engine();

        // Don't set subagent event handler
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
            "agent_type": "test",
            "task": "do something",
            "description": "test task"
        })));

        assert!(result.is_ok());
        let tool_result = result.unwrap();
        assert!(tool_result.is_error);
        assert!(tool_result.content.contains("no subagent event handler"));
    }

    #[test]
    fn test_execute_task_tool_custom_provider_not_registered() {
        let engine = create_test_engine();

        // Set up handler but use unregistered custom provider
        struct DummyHandler;
        impl SubagentEventHandler for DummyHandler {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
            fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
            fn on_stream_usage(
                &self,
                _subagent_id: String,
                _input_tokens: u32,
                _output_tokens: u32,
            ) {
            }
        }

        engine.set_subagent_event_handler(Box::new(DummyHandler));
        engine.set_default_provider(Provider::Custom {
            name: "unregistered-provider".to_string(),
        });

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
            "agent_type": "test",
            "task": "do something",
            "description": "test task"
        })));

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("unregistered-provider"));
        assert!(error.contains("not registered"));

        engine.clear_subagent_event_handler();
    }

    #[test]
    fn test_execute_task_tool_no_model_for_agent() {
        let engine = create_test_engine();

        struct DummyHandler;
        impl SubagentEventHandler for DummyHandler {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
            fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
            fn on_stream_usage(
                &self,
                _subagent_id: String,
                _input_tokens: u32,
                _output_tokens: u32,
            ) {
            }
        }

        engine.set_subagent_event_handler(Box::new(DummyHandler));

        // Register an agent WITHOUT a model, and don't set provider default model
        engine
            .register_agent(AgentConfig::new(
                "no-model-agent".to_string(),
                "You are a test agent.".to_string(),
            ))
            .unwrap();

        // Set Anthropic provider with API key but NO default model
        engine.set_api_key(Provider::Anthropic, "sk-test".to_string());

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
            "agent_type": "no-model-agent",
            "task": "do something",
            "description": "test task"
        })));

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("No model configured"));
        assert!(error.contains("no-model-agent"));

        engine.clear_subagent_event_handler();
    }

    #[test]
    fn test_execute_task_tool_provider_not_configured() {
        let engine = create_test_engine();

        struct DummyHandler;
        impl SubagentEventHandler for DummyHandler {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
            fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
            fn on_stream_usage(
                &self,
                _subagent_id: String,
                _input_tokens: u32,
                _output_tokens: u32,
            ) {
            }
        }

        engine.set_subagent_event_handler(Box::new(DummyHandler));

        // Register an agent with a model
        let mut config = AgentConfig::new(
            "configured-agent".to_string(),
            "You are a test agent.".to_string(),
        );
        config.model = Some("claude-3-opus".to_string());
        engine.register_agent(config).unwrap();

        // Set Anthropic as provider but DON'T configure API key
        engine.set_default_provider(Provider::Anthropic);

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
            "agent_type": "configured-agent",
            "task": "do something",
            "description": "test task"
        })));

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Provider not configured"));

        engine.clear_subagent_event_handler();
    }

    // ========================================================================
    // Mock LLM Provider for testing the agentic loop
    // ========================================================================

    use crate::callback::LlmProvider;
    use crate::types::{ChatRole, LlmRequest, LlmResponse, LlmToolCall, LlmUsage};
    use std::sync::atomic::AtomicU32;

    /// Mock LLM provider that returns canned responses.
    /// Can be configured to return text-only or tool calls.
    struct MockLlmProvider {
        responses: std::sync::Mutex<Vec<LlmResponse>>,
        call_count: AtomicU32,
    }

    impl MockLlmProvider {
        fn new(responses: Vec<LlmResponse>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                call_count: AtomicU32::new(0),
            }
        }

        /// Create a simple text-only response
        fn text_response(text: &str) -> LlmResponse {
            LlmResponse {
                text: text.to_string(),
                tool_calls: vec![],
                usage: LlmUsage {
                    input_tokens: 10,
                    output_tokens: 20,
                },
                error: None,
            }
        }

        /// Create a response with a tool call
        fn tool_call_response(tool_name: &str, args: &str) -> LlmResponse {
            LlmResponse {
                text: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: format!("tool_{}", uuid::Uuid::new_v4()),
                    name: tool_name.to_string(),
                    arguments: args.to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                error: None,
            }
        }

        #[allow(dead_code)]
        fn get_call_count(&self) -> u32 {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl LlmProvider for MockLlmProvider {
        fn generate(&self, _request: LlmRequest) -> LlmResponse {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            let responses = self.responses.lock().unwrap();

            if (count as usize) < responses.len() {
                responses[count as usize].clone()
            } else if !responses.is_empty() {
                // Return last response if we've exhausted the list
                responses.last().unwrap().clone()
            } else {
                // Default: return simple text
                MockLlmProvider::text_response("Default mock response")
            }
        }

        fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
            false
        }
    }

    // Helper to create a wrapper struct for ChatCallback forwarding
    struct CallbackWrapper(Arc<TrackingCallback>);

    impl ChatCallback for CallbackWrapper {
        fn on_text_delta(&self, text: String) {
            self.0.on_text_delta(text);
        }
        fn on_tool_use(&self, r: ToolUseRequest) {
            self.0.on_tool_use(r);
        }
        fn on_tool_result(&self, id: String, result: String) {
            self.0.on_tool_result(id, result);
        }
        fn on_complete(&self, r: ChatResult) {
            self.0.on_complete(r);
        }
        fn on_error(&self, e: String) {
            self.0.on_error(e);
        }
        fn on_context_warning(&self, u: ContextUsage) {
            self.0.on_context_warning(u);
        }
    }

    #[test]
    fn test_do_send_message_with_mock_llm_simple_text() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MockLLM Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Register mock LLM provider
        let mock_provider =
            MockLlmProvider::new(vec![MockLlmProvider::text_response("Hello from mock LLM!")]);
        engine.register_llm_provider("mock-llm".to_string(), Box::new(mock_provider));

        // Set custom provider
        engine.set_default_provider(Provider::Custom {
            name: "mock-llm".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Hi there".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        assert!(result.is_ok());
        let chat_result = result.unwrap();

        // Verify the response
        assert_eq!(chat_result.final_text, "Hello from mock LLM!");
        assert_eq!(chat_result.tool_use_count, 0);
        assert_eq!(chat_result.input_tokens, 10);
        assert_eq!(chat_result.output_tokens, 20);

        // Verify callback received the text
        let text = callback.text_received.lock().unwrap();
        assert!(text.contains("Hello from mock LLM!"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_with_mock_llm_tool_use() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("MockLLM Tool Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Register mock LLM that first calls a tool, then responds with text
        let mock_provider = MockLlmProvider::new(vec![
            // First call: request read_file tool
            MockLlmProvider::tool_call_response("read_file", r#"{"path": "/tmp/test.txt"}"#),
            // Second call: after tool result, return final text
            MockLlmProvider::text_response("I read the file successfully!"),
        ]);
        engine.register_llm_provider("mock-tool-llm".to_string(), Box::new(mock_provider));

        engine.set_default_provider(Provider::Custom {
            name: "mock-tool-llm".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Read that file for me".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        assert!(result.is_ok());
        let chat_result = result.unwrap();

        // Should have used one tool
        assert_eq!(chat_result.tool_use_count, 1);
        assert_eq!(chat_result.final_text, "I read the file successfully!");

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_accumulates_tokens() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Token Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Multiple iterations accumulate tokens
        let mock_provider = MockLlmProvider::new(vec![
            LlmResponse {
                text: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: "tool_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: r#"{"path": "/tmp/a.txt"}"#.to_string(),
                }],
                usage: LlmUsage {
                    input_tokens: 100,
                    output_tokens: 50,
                },
                error: None,
            },
            LlmResponse {
                text: "Done!".to_string(),
                tool_calls: vec![],
                usage: LlmUsage {
                    input_tokens: 200,
                    output_tokens: 75,
                },
                error: None,
            },
        ]);
        engine.register_llm_provider("token-test".to_string(), Box::new(mock_provider));
        engine.set_default_provider(Provider::Custom {
            name: "token-test".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Do something".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        assert!(result.is_ok());
        let chat_result = result.unwrap();

        // Tokens should be accumulated: 100+200 input, 50+75 output
        assert_eq!(chat_result.input_tokens, 300);
        assert_eq!(chat_result.output_tokens, 125);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_max_iterations_limit() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Max Iter Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Mock that always returns tool calls - should hit MAX_AGENTIC_ITERATIONS
        let mock_provider = MockLlmProvider::new(vec![MockLlmProvider::tool_call_response(
            "read_file",
            r#"{"path": "/tmp/loop.txt"}"#,
        )]);
        engine.register_llm_provider("loop-test".to_string(), Box::new(mock_provider));
        engine.set_default_provider(Provider::Custom {
            name: "loop-test".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Loop forever".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        assert!(result.is_ok());
        let chat_result = result.unwrap();

        // Should have hit the limit (50 iterations)
        assert!(chat_result.tool_use_count >= 49); // At least 49 tool uses
        assert!(chat_result.final_text.contains("terminated after"));
        assert!(chat_result.final_text.contains("50 iterations"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_llm_error_response() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Error Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Mock that returns an error
        let mock_provider = MockLlmProvider::new(vec![LlmResponse {
            text: String::new(),
            tool_calls: vec![],
            usage: LlmUsage::default(),
            error: Some("Rate limit exceeded".to_string()),
        }]);
        engine.register_llm_provider("error-test".to_string(), Box::new(mock_provider));
        engine.set_default_provider(Provider::Custom {
            name: "error-test".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Trigger error".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));

        // LLM errors are propagated
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.contains("Rate limit exceeded"));

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_preserves_conversation_history() {
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("History Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        // Track how many messages have been sent
        let call_count = Arc::new(AtomicU32::new(0));
        let call_count_clone = call_count.clone();

        struct HistoryCheckingProvider {
            call_count: Arc<AtomicU32>,
        }

        impl LlmProvider for HistoryCheckingProvider {
            fn generate(&self, request: LlmRequest) -> LlmResponse {
                let count = self.call_count.fetch_add(1, Ordering::SeqCst);

                // First call should have 1 message, second call should have 3
                // (user, assistant from first call, new user)
                if count == 0 {
                    assert_eq!(request.messages.len(), 1);
                } else if count == 1 {
                    // After first response: user + assistant + new user = 3
                    assert!(request.messages.len() >= 2);
                }

                LlmResponse {
                    text: format!("Response {}", count),
                    tool_calls: vec![],
                    usage: LlmUsage {
                        input_tokens: 10,
                        output_tokens: 10,
                    },
                    error: None,
                }
            }

            fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
                false
            }
        }

        let provider = HistoryCheckingProvider {
            call_count: call_count_clone,
        };
        engine.register_llm_provider("history-test".to_string(), Box::new(provider));
        engine.set_default_provider(Provider::Custom {
            name: "history-test".to_string(),
        });

        let rt = tokio::runtime::Runtime::new().unwrap();

        // First message
        let callback1 = Arc::new(TrackingCallback::new());
        let result1 = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "First message".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback1.clone()))),
        ));
        assert!(result1.is_ok());

        // Second message - should include history
        let callback2 = Arc::new(TrackingCallback::new());
        let result2 = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Second message".to_string(),
            Vec::new(),
            Arc::new(Box::new(CallbackWrapper(callback2.clone()))),
        ));
        assert!(result2.is_ok());

        // Verify we made 2 calls total
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_ffi_media_into_blocks_resolves_path_to_base64() {
        // The helper should resolve MediaSource::Path attachments to
        // MediaSource::Base64 so persisted conversations stay portable.
        let dir = tempfile::tempdir().unwrap();
        let png_path = dir.path().join("fixture.png");
        std::fs::write(&png_path, b"fake-png-bytes").unwrap();

        let media = vec![crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Path {
                path: png_path.to_string_lossy().into_owned(),
            },
            mime_type: "image/png".to_string(),
        }];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let blocks = rt.block_on(ffi_media_into_blocks(media)).unwrap();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Media {
                kind,
                source,
                mime_type,
            } => {
                assert_eq!(*kind, mux::llm::MediaKind::Image);
                assert_eq!(mime_type, "image/png");
                // Base64 of b"fake-png-bytes"
                match source {
                    MediaSource::Base64(data) => {
                        assert_eq!(data, "ZmFrZS1wbmctYnl0ZXM=");
                    }
                    other => panic!("expected Base64 source, got {:?}", other),
                }
            }
            other => panic!("expected Media block, got {:?}", other),
        }
    }

    #[test]
    fn test_ffi_media_into_blocks_passes_through_non_path_sources() {
        // Base64 and Url sources should be forwarded without modification.
        let media = vec![
            crate::media::FfiMedia {
                kind: crate::media::FfiMediaKind::Image,
                source: crate::media::FfiMediaSource::Base64 {
                    data: "aGVsbG8=".to_string(),
                },
                mime_type: "image/png".to_string(),
            },
            crate::media::FfiMedia {
                kind: crate::media::FfiMediaKind::Image,
                source: crate::media::FfiMediaSource::Url {
                    url: "https://example.com/x.png".to_string(),
                },
                mime_type: "image/png".to_string(),
            },
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        let blocks = rt.block_on(ffi_media_into_blocks(media)).unwrap();
        assert_eq!(blocks.len(), 2);
        match &blocks[0] {
            ContentBlock::Media {
                source: MediaSource::Base64(data),
                ..
            } => assert_eq!(data, "aGVsbG8="),
            other => panic!("expected Base64 source, got {:?}", other),
        }
        match &blocks[1] {
            ContentBlock::Media {
                source: MediaSource::Url(url),
                ..
            } => assert_eq!(url, "https://example.com/x.png"),
            other => panic!("expected Url source, got {:?}", other),
        }
    }

    #[test]
    fn test_do_send_message_path_media_normalized_to_base64_in_history() {
        // End-to-end check: a MediaSource::Path attachment passed through
        // do_send_message should be persisted in history as MediaSource::Base64.
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Path Normalize Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let png_path = dir.path().join("fixture.png");
        std::fs::write(&png_path, b"fake-png-bytes").unwrap();

        let media = vec![crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Path {
                path: png_path.to_string_lossy().into_owned(),
            },
            mime_type: "image/png".to_string(),
        }];

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Describe this image".to_string(),
            media,
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));
        assert!(result.is_ok(), "do_send_message failed: {:?}", result);

        // History's user message should have the media block normalized to
        // Base64 — NOT Path — so saved conversations stay portable.
        let history = engine.message_history.read();
        let messages = history.get(&conv.id).expect("history entry for conv");
        let user_msg = messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .expect("user message stored");
        let media_block = user_msg
            .content
            .iter()
            .find(|b| matches!(b, ContentBlock::Media { .. }))
            .expect("user message should contain a media block");
        match media_block {
            ContentBlock::Media { source, .. } => {
                assert!(
                    matches!(source, MediaSource::Base64(_)),
                    "expected Base64 source in history, got {:?}",
                    source
                );
            }
            _ => unreachable!(),
        }
        drop(history);

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_do_send_message_media_reaches_real_llm_provider() {
        // Verifies that media attachments plumbed through do_send_message reach
        // the provider on the real-LLM path (via SubAgent::run_with_blocks),
        // not just the echo fallback path.
        let engine = create_test_engine();
        let ws = engine
            .create_workspace("Media RealLLM Test".to_string(), None)
            .unwrap();
        let conv = engine
            .create_conversation(ws.id.clone(), "Test Conv".to_string())
            .unwrap();

        /// Provider that captures the most recent request's user-message media count.
        struct CapturingProvider {
            captured_media_count: Arc<std::sync::Mutex<Option<usize>>>,
        }

        impl LlmProvider for CapturingProvider {
            fn generate(&self, request: LlmRequest) -> LlmResponse {
                // Find the user message and record how many media blocks it has.
                let user_media = request
                    .messages
                    .iter()
                    .find(|m| matches!(m.role, ChatRole::User))
                    .map(|m| m.media.len())
                    .unwrap_or(0);
                *self.captured_media_count.lock().unwrap() = Some(user_media);

                LlmResponse {
                    text: "ack".to_string(),
                    tool_calls: vec![],
                    usage: LlmUsage {
                        input_tokens: 1,
                        output_tokens: 1,
                    },
                    error: None,
                }
            }

            fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
                true
            }
        }

        let captured: Arc<std::sync::Mutex<Option<usize>>> = Arc::new(std::sync::Mutex::new(None));
        let provider = CapturingProvider {
            captured_media_count: captured.clone(),
        };
        engine.register_llm_provider("media-capture".to_string(), Box::new(provider));
        engine.set_default_provider(Provider::Custom {
            name: "media-capture".to_string(),
        });

        let media = vec![crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Base64 {
                data: "aGVsbG8=".to_string(),
            },
            mime_type: "image/png".to_string(),
        }];

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Describe this image".to_string(),
            media,
            Arc::new(Box::new(CallbackWrapper(callback.clone()))),
        ));
        assert!(result.is_ok(), "do_send_message failed: {:?}", result);

        // Confirm the provider observed the media on the user message.
        let captured = captured.lock().unwrap();
        assert_eq!(
            *captured,
            Some(1),
            "provider should have seen 1 media block on the user message"
        );

        engine.delete_workspace(ws.id).unwrap();
    }
}
