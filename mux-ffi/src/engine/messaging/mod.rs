// ABOUTME: Engine messaging path: chat send, streaming, media-bearing turns.
// ABOUTME: ChatCallbackHook lives in callback_hook.rs.

mod callback_hook;
use callback_hook::ChatCallbackHook;

use super::MuxEngine;
use super::persistence::StoredMessage;
use super::subagent::TaskToolEventProxy;
use super::tool_wrappers::{CustomToolWrapper, McpToolWrapper};
#[cfg(test)]
use crate::callback::ToolUseRequest;
use crate::callback::{ChatCallback, ChatResult};
use crate::task_tool::FfiTaskTool;
use crate::types::Provider;
use mux::agent::{AgentDefinition, AgentRegistry, SubAgent};
use mux::hook::HookRegistry;
use mux::llm::{GeminiClient, MediaSource};
use mux::prelude::{
    AnthropicClient, ContentBlock, LlmClient, McpClient, Message, OpenAIClient, Registry, Role,
};
use mux::tool::Tool;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

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

        // Register the task tool when a subagent event handler is set, so the
        // LLM running this turn can dispatch subagents via the `task` tool and
        // events stream back to the host. Without a handler the tool is
        // silently absent: the host opts in by calling `set_subagent_event_handler`.
        if self.subagent_event_handler.read().is_some() {
            match self.try_build_ffi_task_tool().await {
                Ok(task_tool) => registry.register(task_tool).await,
                Err(e) => {
                    eprintln!("Skipping task tool registration for this turn: {}", e)
                }
            }
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
    /// Used directly by unit tests; the production chat loop reaches the same
    /// implementation by registering the tool into `build_tool_registry`.
    #[cfg(test)]
    pub(super) async fn execute_task_tool(
        &self,
        params: serde_json::Value,
    ) -> Result<mux::tool::ToolResult, String> {
        if self.subagent_event_handler.read().is_none() {
            return Ok(mux::tool::ToolResult::error(
                "TaskTool not available: no subagent event handler registered",
            ));
        }
        let tool = self.try_build_ffi_task_tool().await?;
        tool.execute(params).await.map_err(|e| e.to_string())
    }

    /// Build an `FfiTaskTool` from the engine's current state.
    ///
    /// Returns `Err` if a precondition can't be satisfied at build time
    /// (no agent model resolvable, custom provider unregistered, primary
    /// provider unconfigured). Callers gate on `subagent_event_handler`
    /// being present; this helper does not re-check it.
    ///
    /// The constructed tool can be registered into the chat loop's tool
    /// `Registry` so the LLM can dispatch subagents directly, or executed
    /// once via `execute_task_tool` for unit testing.
    pub(super) async fn try_build_ffi_task_tool(&self) -> Result<FfiTaskTool, String> {
        let provider = self.default_provider.read().clone();

        // Capture the custom provider's client atomically with validation so a
        // concurrent `unregister_llm_provider` between here and the factory's
        // first call cannot turn a build-time Ok into a runtime panic. If the
        // lookup fails we fail the build cleanly; the factory closure then
        // operates on a guaranteed-Some `captured_custom_client`.
        let captured_custom_client: Option<Arc<dyn LlmClient>> = match &provider {
            Provider::Custom { name } => Some(
                self.callback_providers
                    .read()
                    .get(name)
                    .cloned()
                    .map(|c| c as Arc<dyn LlmClient>)
                    .ok_or_else(|| {
                        format!(
                            "Custom LLM provider '{}' not registered. Call register_llm_provider first.",
                            name
                        )
                    })?,
            ),
            _ => None,
        };

        // Snapshot agent configs before any `.await` so the registration
        // loop doesn't hold the read guard across await points.
        let agent_configs_snapshot: Vec<_> = self
            .agent_configs
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let agent_registry = AgentRegistry::new();
        let provider_default_model = self.get_default_model(provider.clone());
        for (name, config) in agent_configs_snapshot {
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

            let mut definition = AgentDefinition::new(&name, &config.system_prompt)
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

        // Snapshot custom tools before any `.await`.
        let custom_tools_snapshot: Vec<Arc<dyn Tool>> = self
            .custom_tools
            .read()
            .values()
            .map(|t| t.clone() as Arc<dyn Tool>)
            .collect();

        let tool_registry = Registry::new();
        for tool in &self.builtin_tools {
            tool_registry.register_arc(tool.clone()).await;
        }
        for tool in custom_tools_snapshot {
            tool_registry.register_arc(tool).await;
        }

        // For Custom providers we don't need API key config — the captured
        // callback client is enough. For everything else, fail fast if the
        // provider isn't configured.
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

        let provider_clone = provider.clone();
        let api_key = provider_config.as_ref().map(|c| c.api_key.clone());
        let base_url = provider_config.as_ref().and_then(|c| c.base_url.clone());

        let client_factory = move |_model: &str| -> Arc<dyn LlmClient> {
            match &provider_clone {
                Provider::Custom { .. } => captured_custom_client
                    .clone()
                    // Invariant: when `provider` is `Custom`, the match above
                    // populated `captured_custom_client` with `Some(_)` or
                    // returned `Err` before reaching this point.
                    .expect("invariant: custom provider validated and captured at build"),
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

        // Forward subagent events through a proxy that late-binds against
        // the engine's currently-registered handler. The handler may be
        // swapped via `set_subagent_event_handler` between build and use;
        // the proxy reads it at event time.
        let handler_proxy = TaskToolEventProxy {
            engine_handler: self.subagent_event_handler.clone(),
        };

        Ok(FfiTaskTool::new(
            agent_registry,
            tool_registry,
            client_factory,
            Box::new(handler_proxy),
        )
        .with_transcript_store(self.transcript_store.clone()))
    }
}

#[cfg(test)]
#[path = "messaging_test.rs"]
mod tests;
