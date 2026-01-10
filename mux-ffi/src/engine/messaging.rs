// ABOUTME: Message handling and agentic loop for MuxEngine.
// ABOUTME: Contains do_send_message and execute_task_tool for LLM interactions.

use super::persistence::StoredMessage;
use super::subagent::TaskToolEventProxy;
use super::MuxEngine;
use crate::callback::{ChatCallback, ChatResult, ToolUseRequest};
use crate::task_tool::FfiTaskTool;
use crate::types::Provider;
use mux::agent::AgentRegistry;
use mux::llm::GeminiClient;
use mux::prelude::{
    AgentDefinition, AnthropicClient, ContentBlock, LlmClient, McpClient, Message, OpenAIClient,
    Registry, Request, Role, StopReason, ToolDefinition,
};
use mux::tool::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// Messaging implementation
impl MuxEngine {
    pub(super) async fn do_send_message(
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

                        // Check context warning before completing
                        self.check_and_warn_context(&conversation_id, callback.as_ref().as_ref());

                        return Ok(ChatResult {
                            conversation_id: conversation_id.clone(),
                            final_text: echo_text,
                            tool_use_count: 0,
                            input_tokens: 0,
                            output_tokens: 0,
                            context_usage: self
                                .get_context_usage(conversation_id)
                                .unwrap_or_default(),
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
                base_prompt, tool_list, workspace_path, workspace_path
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
                                tool_results
                                    .push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results
                                    .push(ContentBlock::tool_result(&tool_use_id, &result.content));
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
                                tool_results
                                    .push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results
                                    .push(ContentBlock::tool_result(&tool_use_id, &result.content));
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
                                tool_results
                                    .push(ContentBlock::tool_error(&tool_use_id, &result.content));
                            } else {
                                tool_results
                                    .push(ContentBlock::tool_result(&tool_use_id, &result.content));
                            }
                        }
                        Err(e) => {
                            let error_msg = format!("TaskTool execution failed: {}", e);
                            callback.on_tool_result(tool_use_id.clone(), error_msg.clone());
                            tool_results.push(ContentBlock::tool_error(&tool_use_id, &error_msg));
                        }
                    }
                } else if let Some((server_name, mcp_tool_name)) = self.parse_tool_name(&tool_name)
                {
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

        // Check context warning before completing
        self.check_and_warn_context(&conversation_id, callback.as_ref().as_ref());

        Ok(ChatResult {
            conversation_id: conversation_id.clone(),
            final_text: full_text,
            tool_use_count: total_tool_use_count,
            input_tokens: total_input_tokens,
            output_tokens: total_output_tokens,
            context_usage: self.get_context_usage(conversation_id).unwrap_or_default(),
        })
    }

    /// Execute the TaskTool to spawn a subagent.
    /// This creates an FfiTaskTool with the current engine state and event handler.
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
                Provider::Custom { name } => callback_providers
                    .read()
                    .get(name)
                    .cloned()
                    .map(|c| c as Arc<dyn LlmClient>)
                    .expect("Custom provider was validated at start of execute_task_tool"),
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

    fn create_test_engine() -> Arc<MuxEngine> {
        MuxEngine::new("/tmp/mux-test-messaging".to_string()).unwrap()
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
    use crate::types::{LlmRequest, LlmResponse, LlmToolCall, LlmUsage};
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
        let mock_provider = MockLlmProvider::new(vec![
            MockLlmProvider::text_response("Hello from mock LLM!"),
        ]);
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
        let mock_provider = MockLlmProvider::new(vec![
            MockLlmProvider::tool_call_response("read_file", r#"{"path": "/tmp/loop.txt"}"#),
        ]);
        engine.register_llm_provider("loop-test".to_string(), Box::new(mock_provider));
        engine.set_default_provider(Provider::Custom {
            name: "loop-test".to_string(),
        });

        let callback = Arc::new(TrackingCallback::new());
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Loop forever".to_string(),
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
            Arc::new(Box::new(CallbackWrapper(callback1.clone()))),
        ));
        assert!(result1.is_ok());

        // Second message - should include history
        let callback2 = Arc::new(TrackingCallback::new());
        let result2 = rt.block_on(engine.do_send_message(
            conv.id.clone(),
            "Second message".to_string(),
            Arc::new(Box::new(CallbackWrapper(callback2.clone()))),
        ));
        assert!(result2.is_ok());

        // Verify we made 2 calls total
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        engine.delete_workspace(ws.id).unwrap();
    }
}
