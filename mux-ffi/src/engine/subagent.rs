// ABOUTME: Subagent management for MuxEngine.
// ABOUTME: Handles spawning, resuming agents, and proxy types for event forwarding.

use super::MuxEngine;
use crate::callback::{SubagentCallback, SubagentEventHandler, ToolUseRequest};
use crate::types::{Provider, SubagentResult, TranscriptData};
use mux::hook::HookRegistry;
use mux::llm::GeminiClient;
use mux::prelude::{
    AgentDefinition, AnthropicClient, LlmClient, Message, OpenAIClient, Registry, SubAgent,
};
use mux::tool::Tool;
use parking_lot::RwLock;
use std::sync::Arc;

/// Proxy that forwards SubagentEventHandler calls to the engine's stored handler.
/// Used by execute_task_tool to forward events from the TaskTool.
pub(super) struct TaskToolEventProxy {
    pub engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>>,
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
            handler.on_agent_completed(
                subagent_id,
                content,
                tool_use_count,
                iterations,
                transcript_saved,
            );
        }
    }

    fn on_agent_error(&self, subagent_id: String, error: String) {
        if let Some(handler) = self.engine_handler.read().as_ref() {
            handler.on_agent_error(subagent_id, error);
        }
    }
}

/// A hook that proxies tool events to the SubagentCallback.
/// This enables streaming tool use/result events to Swift during subagent execution.
pub(super) struct CallbackProxyHook {
    agent_id: String,
    callback: Arc<Box<dyn SubagentCallback>>,
}

impl CallbackProxyHook {
    pub fn new(agent_id: String, callback: Arc<Box<dyn SubagentCallback>>) -> Self {
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
                tool_use_id,
                input: _,
                result,
            } => {
                // Notify callback of tool result
                self.callback.on_tool_result(
                    self.agent_id.clone(),
                    tool_use_id.clone(),
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
    pub(super) async fn do_spawn_agent(
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
            Provider::Custom { name } => self
                .callback_providers
                .read()
                .get(name)
                .cloned()
                .map(|c| c as Arc<dyn LlmClient>)
                .ok_or_else(|| format!("Callback provider '{}' not registered", name))?,
            _ => {
                let provider_config = self
                    .api_keys
                    .read()
                    .get(&provider)
                    .cloned()
                    .ok_or_else(|| format!("Provider not configured: {:?}", provider))?;

                match provider {
                    Provider::Anthropic => Arc::new(AnthropicClient::new(&provider_config.api_key)),
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
        let model = config.model.clone().or(default_model).ok_or_else(|| {
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
    pub(super) async fn do_resume_agent(
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
            Provider::Custom { name } => self
                .callback_providers
                .read()
                .get(name)
                .cloned()
                .map(|c| c as Arc<dyn LlmClient>)
                .ok_or_else(|| format!("Callback provider '{}' not registered", name))?,
            _ => {
                let provider_config = self
                    .api_keys
                    .read()
                    .get(&provider)
                    .cloned()
                    .ok_or_else(|| format!("Provider not configured: {:?}", provider))?;

                match provider {
                    Provider::Anthropic => Arc::new(AnthropicClient::new(&provider_config.api_key)),
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
        let definition =
            AgentDefinition::new("resumed", "You are a helpful assistant.").max_iterations(10);

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
            transcript_json: Some(serde_json::to_string(subagent.transcript()).unwrap_or_default()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    // Mock handler that tracks calls
    struct TrackingHandler {
        started_count: AtomicU32,
        tool_use_count: AtomicU32,
        tool_result_count: AtomicU32,
        iteration_count: AtomicU32,
        completed_count: AtomicU32,
        error_count: AtomicU32,
    }

    impl TrackingHandler {
        fn new() -> Self {
            Self {
                started_count: AtomicU32::new(0),
                tool_use_count: AtomicU32::new(0),
                tool_result_count: AtomicU32::new(0),
                iteration_count: AtomicU32::new(0),
                completed_count: AtomicU32::new(0),
                error_count: AtomicU32::new(0),
            }
        }
    }

    impl SubagentEventHandler for TrackingHandler {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {
            self.started_count.fetch_add(1, Ordering::SeqCst);
        }
        fn on_tool_use(&self, _: String, _: String, _: String) {
            self.tool_use_count.fetch_add(1, Ordering::SeqCst);
        }
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {
            self.tool_result_count.fetch_add(1, Ordering::SeqCst);
        }
        fn on_iteration(&self, _: String, _: u32) {
            self.iteration_count.fetch_add(1, Ordering::SeqCst);
        }
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {
            self.completed_count.fetch_add(1, Ordering::SeqCst);
        }
        fn on_agent_error(&self, _: String, _: String) {
            self.error_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_agent_started() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(Some(Box::new(TrackingHandler::new()))));

        // Replace with our tracking handler
        *engine_handler.write() = Some(Box::new({
            struct ForwardToArc(Arc<TrackingHandler>);
            impl SubagentEventHandler for ForwardToArc {
                fn on_agent_started(&self, a: String, b: String, c: String, d: String) {
                    self.0.on_agent_started(a, b, c, d);
                }
                fn on_tool_use(&self, a: String, b: String, c: String) {
                    self.0.on_tool_use(a, b, c);
                }
                fn on_tool_result(&self, a: String, b: String, c: String, d: bool) {
                    self.0.on_tool_result(a, b, c, d);
                }
                fn on_iteration(&self, a: String, b: u32) {
                    self.0.on_iteration(a, b);
                }
                fn on_agent_completed(&self, a: String, b: String, c: u32, d: u32, e: bool) {
                    self.0.on_agent_completed(a, b, c, d, e);
                }
                fn on_agent_error(&self, a: String, b: String) {
                    self.0.on_agent_error(a, b);
                }
            }
            ForwardToArc(handler.clone())
        }));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_agent_started("id".into(), "type".into(), "task".into(), "desc".into());
        assert_eq!(handler.started_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_tool_use() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        struct ForwardToArc(Arc<TrackingHandler>);
        impl SubagentEventHandler for ForwardToArc {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, a: String, b: String, c: String) {
                self.0.on_tool_use(a, b, c);
            }
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
        }
        *engine_handler.write() = Some(Box::new(ForwardToArc(handler.clone())));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_tool_use("id".into(), "tool".into(), "{}".into());
        assert_eq!(handler.tool_use_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_tool_result() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        struct ForwardToArc(Arc<TrackingHandler>);
        impl SubagentEventHandler for ForwardToArc {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, a: String, b: String, c: String, d: bool) {
                self.0.on_tool_result(a, b, c, d);
            }
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
        }
        *engine_handler.write() = Some(Box::new(ForwardToArc(handler.clone())));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_tool_result("id".into(), "tool".into(), "result".into(), false);
        assert_eq!(handler.tool_result_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_iteration() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        struct ForwardToArc(Arc<TrackingHandler>);
        impl SubagentEventHandler for ForwardToArc {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, a: String, b: u32) {
                self.0.on_iteration(a, b);
            }
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, _: String, _: String) {}
        }
        *engine_handler.write() = Some(Box::new(ForwardToArc(handler.clone())));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_iteration("id".into(), 5);
        assert_eq!(handler.iteration_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_agent_completed() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        struct ForwardToArc(Arc<TrackingHandler>);
        impl SubagentEventHandler for ForwardToArc {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, a: String, b: String, c: u32, d: u32, e: bool) {
                self.0.on_agent_completed(a, b, c, d, e);
            }
            fn on_agent_error(&self, _: String, _: String) {}
        }
        *engine_handler.write() = Some(Box::new(ForwardToArc(handler.clone())));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_agent_completed("id".into(), "content".into(), 3, 2, true);
        assert_eq!(handler.completed_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_forwards_on_agent_error() {
        let handler = Arc::new(TrackingHandler::new());
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        struct ForwardToArc(Arc<TrackingHandler>);
        impl SubagentEventHandler for ForwardToArc {
            fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: String, _: String) {}
            fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
            fn on_iteration(&self, _: String, _: u32) {}
            fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
            fn on_agent_error(&self, a: String, b: String) {
                self.0.on_agent_error(a, b);
            }
        }
        *engine_handler.write() = Some(Box::new(ForwardToArc(handler.clone())));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        proxy.on_agent_error("id".into(), "error message".into());
        assert_eq!(handler.error_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_task_tool_event_proxy_no_handler() {
        // When no handler is set, calls should not panic
        let engine_handler: Arc<RwLock<Option<Box<dyn SubagentEventHandler>>>> =
            Arc::new(RwLock::new(None));

        let proxy = TaskToolEventProxy {
            engine_handler: engine_handler.clone(),
        };

        // None of these should panic
        proxy.on_agent_started("id".into(), "type".into(), "task".into(), "desc".into());
        proxy.on_tool_use("id".into(), "tool".into(), "{}".into());
        proxy.on_tool_result("id".into(), "tool".into(), "result".into(), false);
        proxy.on_iteration("id".into(), 1);
        proxy.on_agent_completed("id".into(), "content".into(), 0, 0, false);
        proxy.on_agent_error("id".into(), "error".into());
    }

    #[test]
    fn test_callback_proxy_hook_new() {
        use crate::callback::SubagentCallback;

        struct DummyCallback;
        impl SubagentCallback for DummyCallback {
            fn on_text_delta(&self, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: ToolUseRequest) {}
            fn on_tool_result(&self, _: String, _: String, _: String) {}
            fn on_complete(&self, _: SubagentResult) {}
            fn on_error(&self, _: String, _: String) {}
        }

        let callback: Arc<Box<dyn SubagentCallback>> = Arc::new(Box::new(DummyCallback));
        let hook = CallbackProxyHook::new("agent-123".to_string(), callback);

        assert_eq!(hook.agent_id, "agent-123");
    }

    #[test]
    fn test_callback_proxy_hook_accepts_all() {
        use crate::callback::SubagentCallback;
        use mux::hook::{Hook, HookEvent};
        use std::collections::HashMap;

        struct DummyCallback;
        impl SubagentCallback for DummyCallback {
            fn on_text_delta(&self, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: ToolUseRequest) {}
            fn on_tool_result(&self, _: String, _: String, _: String) {}
            fn on_complete(&self, _: SubagentResult) {}
            fn on_error(&self, _: String, _: String) {}
        }

        let callback: Arc<Box<dyn SubagentCallback>> = Arc::new(Box::new(DummyCallback));
        let hook = CallbackProxyHook::new("agent-123".to_string(), callback);

        // Test that accepts returns true for various events
        let pre_tool = HookEvent::PreToolUse {
            tool_name: "test".to_string(),
            input: serde_json::json!({}),
        };
        assert!(hook.accepts(&pre_tool));

        let post_tool = HookEvent::PostToolUse {
            tool_name: "test".to_string(),
            tool_use_id: "toolu_123".to_string(),
            input: serde_json::json!({}),
            result: mux::tool::ToolResult {
                content: "result".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        };
        assert!(hook.accepts(&post_tool));

        let iteration = HookEvent::Iteration {
            agent_id: "agent-123".to_string(),
            iteration: 1,
        };
        assert!(hook.accepts(&iteration));
    }

    #[test]
    fn test_callback_proxy_hook_on_event_pre_tool_use() {
        use crate::callback::SubagentCallback;
        use mux::hook::{Hook, HookAction, HookEvent};
        use std::sync::atomic::AtomicBool;

        struct TrackingCallback {
            tool_use_called: AtomicBool,
        }
        impl SubagentCallback for TrackingCallback {
            fn on_text_delta(&self, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: ToolUseRequest) {
                self.tool_use_called.store(true, Ordering::SeqCst);
            }
            fn on_tool_result(&self, _: String, _: String, _: String) {}
            fn on_complete(&self, _: SubagentResult) {}
            fn on_error(&self, _: String, _: String) {}
        }

        let callback = Arc::new(TrackingCallback {
            tool_use_called: AtomicBool::new(false),
        });
        let boxed: Arc<Box<dyn SubagentCallback>> = Arc::new(Box::new({
            struct Wrapper(Arc<TrackingCallback>);
            impl SubagentCallback for Wrapper {
                fn on_text_delta(&self, _: String, _: String) {}
                fn on_tool_use(&self, a: String, b: ToolUseRequest) {
                    self.0.on_tool_use(a, b);
                }
                fn on_tool_result(&self, _: String, _: String, _: String) {}
                fn on_complete(&self, _: SubagentResult) {}
                fn on_error(&self, _: String, _: String) {}
            }
            Wrapper(callback.clone())
        }));

        let hook = CallbackProxyHook::new("agent-456".to_string(), boxed);

        let event = HookEvent::PreToolUse {
            tool_name: "read_file".to_string(),
            input: serde_json::json!({"path": "/tmp/test"}),
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(hook.on_event(&event)).unwrap();

        assert!(matches!(result, HookAction::Continue));
        assert!(callback.tool_use_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_proxy_hook_on_event_post_tool_use() {
        use crate::callback::SubagentCallback;
        use mux::hook::{Hook, HookAction, HookEvent};
        use std::collections::HashMap;
        use std::sync::atomic::AtomicBool;

        struct TrackingCallback {
            tool_result_called: AtomicBool,
        }
        impl SubagentCallback for TrackingCallback {
            fn on_text_delta(&self, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: ToolUseRequest) {}
            fn on_tool_result(&self, _: String, _: String, _: String) {
                self.tool_result_called.store(true, Ordering::SeqCst);
            }
            fn on_complete(&self, _: SubagentResult) {}
            fn on_error(&self, _: String, _: String) {}
        }

        let callback = Arc::new(TrackingCallback {
            tool_result_called: AtomicBool::new(false),
        });
        let boxed: Arc<Box<dyn SubagentCallback>> = Arc::new(Box::new({
            struct Wrapper(Arc<TrackingCallback>);
            impl SubagentCallback for Wrapper {
                fn on_text_delta(&self, _: String, _: String) {}
                fn on_tool_use(&self, _: String, _: ToolUseRequest) {}
                fn on_tool_result(&self, a: String, b: String, c: String) {
                    self.0.on_tool_result(a, b, c);
                }
                fn on_complete(&self, _: SubagentResult) {}
                fn on_error(&self, _: String, _: String) {}
            }
            Wrapper(callback.clone())
        }));

        let hook = CallbackProxyHook::new("agent-789".to_string(), boxed);

        let event = HookEvent::PostToolUse {
            tool_name: "bash".to_string(),
            tool_use_id: "toolu_456".to_string(),
            input: serde_json::json!({"command": "ls"}),
            result: mux::tool::ToolResult {
                content: "file1.txt\nfile2.txt".to_string(),
                is_error: false,
                metadata: HashMap::new(),
            },
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(hook.on_event(&event)).unwrap();

        assert!(matches!(result, HookAction::Continue));
        assert!(callback.tool_result_called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_proxy_hook_on_event_other_events() {
        use crate::callback::SubagentCallback;
        use mux::hook::{Hook, HookAction, HookEvent};

        struct DummyCallback;
        impl SubagentCallback for DummyCallback {
            fn on_text_delta(&self, _: String, _: String) {}
            fn on_tool_use(&self, _: String, _: ToolUseRequest) {}
            fn on_tool_result(&self, _: String, _: String, _: String) {}
            fn on_complete(&self, _: SubagentResult) {}
            fn on_error(&self, _: String, _: String) {}
        }

        let callback: Arc<Box<dyn SubagentCallback>> = Arc::new(Box::new(DummyCallback));
        let hook = CallbackProxyHook::new("agent-abc".to_string(), callback);

        // Other events should just return Continue without side effects
        let iteration = HookEvent::Iteration {
            agent_id: "agent-abc".to_string(),
            iteration: 5,
        };

        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(hook.on_event(&iteration)).unwrap();

        assert!(matches!(result, HookAction::Continue));
    }
}
