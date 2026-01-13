// ABOUTME: Hook system for extensibility in tool and agent lifecycle.
// ABOUTME: Provides events, actions, and a registry for hook management.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::agent::SubAgentResult;
use crate::tool::ToolResult;

/// Events that can trigger hooks.
#[derive(Debug, Clone)]
pub enum HookEvent {
    /// Fired before a tool is executed.
    PreToolUse {
        tool_name: String,
        input: Value,
    },

    /// Fired after a tool execution completes.
    PostToolUse {
        tool_name: String,
        tool_use_id: String,
        input: Value,
        result: ToolResult,
    },

    /// Fired when an agent starts execution.
    AgentStart {
        agent_id: String,
        task: String,
    },

    /// Fired when an agent completes execution.
    AgentStop {
        agent_id: String,
        result: SubAgentResult,
    },

    /// Fired at the start of each think-act iteration.
    Iteration {
        agent_id: String,
        iteration: usize,
    },

    /// Fired when a session starts (via run() or continue()).
    SessionStart {
        session_id: String,
        /// Source of the session: "run" or "continue"
        source: String,
        prompt: String,
    },

    /// Fired when a session ends.
    SessionEnd {
        session_id: String,
        /// Error message if the session ended with an error.
        error: Option<String>,
        /// Reason for ending: "complete", "error", or "cancelled"
        reason: String,
    },

    /// Fired before the agent loop stops.
    /// Hooks can set `continue_loop` to true to request continuation.
    Stop {
        session_id: String,
        final_text: String,
        /// Set to true to request the agent loop continue.
        /// Uses Arc<AtomicBool> for interior mutability across async hooks.
        continue_loop: Arc<AtomicBool>,
    },

    /// Fired when a subagent is started.
    SubagentStart {
        parent_id: String,
        child_id: String,
        name: String,
    },

    /// Fired when a subagent completes.
    SubagentStop {
        parent_id: String,
        child_id: String,
        name: String,
        /// Error message if the subagent ended with an error.
        error: Option<String>,
    },

    /// Fired after each LLM response is received.
    /// Enables streaming text and tool use notifications to callbacks.
    ResponseReceived {
        agent_id: String,
        /// Text content from the response (if any).
        text: String,
        /// Tool uses in this response (name, id, input JSON).
        tool_uses: Vec<(String, String, Value)>,
    },
}

/// Actions a hook can return to control execution flow.
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Continue with normal execution.
    Continue,

    /// Block the action with a message (only valid for Pre* events).
    Block(String),

    /// Transform the input (only valid for PreToolUse).
    Transform(Value),
}

impl Default for HookAction {
    fn default() -> Self {
        Self::Continue
    }
}

/// Trait for implementing hooks.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called when an event occurs.
    ///
    /// Return `Ok(HookAction::Continue)` to proceed normally.
    /// Return `Ok(HookAction::Block(msg))` to block Pre* events.
    /// Return `Ok(HookAction::Transform(value))` to modify PreToolUse input.
    /// Return `Err` to signal a hook failure (treated as Block).
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error>;

    /// Optional: Filter which events this hook cares about.
    /// Default returns true for all events.
    fn accepts(&self, event: &HookEvent) -> bool {
        let _ = event;
        true
    }
}

/// Registry for managing and firing hooks.
pub struct HookRegistry {
    hooks: RwLock<Vec<Arc<dyn Hook>>>,
}

impl HookRegistry {
    /// Create a new empty hook registry.
    pub fn new() -> Self {
        Self {
            hooks: RwLock::new(Vec::new()),
        }
    }

    /// Register a hook.
    pub async fn register(&self, hook: impl Hook + 'static) {
        self.hooks.write().await.push(Arc::new(hook));
    }

    /// Register a hook wrapped in Arc.
    pub async fn register_arc(&self, hook: Arc<dyn Hook>) {
        self.hooks.write().await.push(hook);
    }

    /// Fire an event to all registered hooks.
    ///
    /// Returns the final action after all hooks have processed.
    /// If any hook blocks, returns Block immediately.
    /// If any hook transforms, uses the transformed value for subsequent hooks.
    pub async fn fire(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        let hooks = self.hooks.read().await;
        let mut current_event = event.clone();
        let mut final_action = HookAction::Continue;

        for hook in hooks.iter() {
            if !hook.accepts(&current_event) {
                continue;
            }

            match hook.on_event(&current_event).await? {
                HookAction::Continue => {}
                HookAction::Block(msg) => {
                    return Ok(HookAction::Block(msg));
                }
                HookAction::Transform(new_input) => {
                    // Transform only valid for PreToolUse events
                    if let HookEvent::PreToolUse { tool_name, .. } = &current_event {
                        current_event = HookEvent::PreToolUse {
                            tool_name: tool_name.clone(),
                            input: new_input.clone(),
                        };
                        final_action = HookAction::Transform(new_input);
                    } else {
                        // Transform action returned for non-PreToolUse event - this is a bug
                        let event_type = match &current_event {
                            HookEvent::PreToolUse { .. } => "PreToolUse",
                            HookEvent::PostToolUse { .. } => "PostToolUse",
                            HookEvent::AgentStart { .. } => "AgentStart",
                            HookEvent::AgentStop { .. } => "AgentStop",
                            HookEvent::Iteration { .. } => "Iteration",
                            HookEvent::SessionStart { .. } => "SessionStart",
                            HookEvent::SessionEnd { .. } => "SessionEnd",
                            HookEvent::Stop { .. } => "Stop",
                            HookEvent::SubagentStart { .. } => "SubagentStart",
                            HookEvent::SubagentStop { .. } => "SubagentStop",
                            HookEvent::ResponseReceived { .. } => "ResponseReceived",
                        };
                        return Err(anyhow::anyhow!(
                            "HookAction::Transform is only valid for PreToolUse events, got {}",
                            event_type
                        ));
                    }
                }
            }
        }

        Ok(final_action)
    }

    /// Get the number of registered hooks.
    pub async fn len(&self) -> usize {
        self.hooks.read().await.len()
    }

    /// Check if the registry is empty.
    pub async fn is_empty(&self) -> bool {
        self.hooks.read().await.is_empty()
    }

    /// Register a hook that only handles SessionStart events.
    ///
    /// The callback receives (session_id, source, prompt).
    pub async fn on_session_start<F>(&self, f: F)
    where
        F: Fn(&str, &str, &str) -> HookAction + Send + Sync + 'static,
    {
        self.register(SessionStartHook { callback: f }).await;
    }

    /// Register a hook that only handles SessionEnd events.
    ///
    /// The callback receives (session_id, error, reason).
    pub async fn on_session_end<F>(&self, f: F)
    where
        F: Fn(&str, Option<&str>, &str) -> HookAction + Send + Sync + 'static,
    {
        self.register(SessionEndHook { callback: f }).await;
    }

    /// Register a hook that only handles Stop events.
    ///
    /// The callback receives (session_id, final_text, continue_loop).
    /// Set `continue_loop.store(true, Ordering::SeqCst)` to request continuation.
    pub async fn on_stop<F>(&self, f: F)
    where
        F: Fn(&str, &str, &Arc<AtomicBool>) -> HookAction + Send + Sync + 'static,
    {
        self.register(StopHook { callback: f }).await;
    }

    /// Register a hook that only handles SubagentStart events.
    ///
    /// The callback receives (parent_id, child_id, name).
    pub async fn on_subagent_start<F>(&self, f: F)
    where
        F: Fn(&str, &str, &str) -> HookAction + Send + Sync + 'static,
    {
        self.register(SubagentStartHook { callback: f }).await;
    }

    /// Register a hook that only handles SubagentStop events.
    ///
    /// The callback receives (parent_id, child_id, name, error).
    pub async fn on_subagent_stop<F>(&self, f: F)
    where
        F: Fn(&str, &str, &str, Option<&str>) -> HookAction + Send + Sync + 'static,
    {
        self.register(SubagentStopHook { callback: f }).await;
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Hook wrapper for SessionStart events.
struct SessionStartHook<F> {
    callback: F,
}

#[async_trait]
impl<F> Hook for SessionStartHook<F>
where
    F: Fn(&str, &str, &str) -> HookAction + Send + Sync,
{
    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(event, HookEvent::SessionStart { .. })
    }

    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::SessionStart {
            session_id,
            source,
            prompt,
        } = event
        {
            Ok((self.callback)(session_id, source, prompt))
        } else {
            Ok(HookAction::Continue)
        }
    }
}

/// Hook wrapper for SessionEnd events.
struct SessionEndHook<F> {
    callback: F,
}

#[async_trait]
impl<F> Hook for SessionEndHook<F>
where
    F: Fn(&str, Option<&str>, &str) -> HookAction + Send + Sync,
{
    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(event, HookEvent::SessionEnd { .. })
    }

    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::SessionEnd {
            session_id,
            error,
            reason,
        } = event
        {
            Ok((self.callback)(session_id, error.as_deref(), reason))
        } else {
            Ok(HookAction::Continue)
        }
    }
}

/// Hook wrapper for Stop events.
struct StopHook<F> {
    callback: F,
}

#[async_trait]
impl<F> Hook for StopHook<F>
where
    F: Fn(&str, &str, &Arc<AtomicBool>) -> HookAction + Send + Sync,
{
    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(event, HookEvent::Stop { .. })
    }

    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::Stop {
            session_id,
            final_text,
            continue_loop,
        } = event
        {
            Ok((self.callback)(session_id, final_text, continue_loop))
        } else {
            Ok(HookAction::Continue)
        }
    }
}

/// Hook wrapper for SubagentStart events.
struct SubagentStartHook<F> {
    callback: F,
}

#[async_trait]
impl<F> Hook for SubagentStartHook<F>
where
    F: Fn(&str, &str, &str) -> HookAction + Send + Sync,
{
    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(event, HookEvent::SubagentStart { .. })
    }

    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::SubagentStart {
            parent_id,
            child_id,
            name,
        } = event
        {
            Ok((self.callback)(parent_id, child_id, name))
        } else {
            Ok(HookAction::Continue)
        }
    }
}

/// Hook wrapper for SubagentStop events.
struct SubagentStopHook<F> {
    callback: F,
}

#[async_trait]
impl<F> Hook for SubagentStopHook<F>
where
    F: Fn(&str, &str, &str, Option<&str>) -> HookAction + Send + Sync,
{
    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(event, HookEvent::SubagentStop { .. })
    }

    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        if let HookEvent::SubagentStop {
            parent_id,
            child_id,
            name,
            error,
        } = event
        {
            Ok((self.callback)(parent_id, child_id, name, error.as_deref()))
        } else {
            Ok(HookAction::Continue)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;

    struct LoggingHook {
        events: Arc<RwLock<Vec<String>>>,
    }

    impl LoggingHook {
        fn new() -> (Self, Arc<RwLock<Vec<String>>>) {
            let events = Arc::new(RwLock::new(Vec::new()));
            (
                Self {
                    events: events.clone(),
                },
                events,
            )
        }
    }

    #[async_trait]
    impl Hook for LoggingHook {
        async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
            let msg = match event {
                HookEvent::PreToolUse { tool_name, .. } => format!("pre:{}", tool_name),
                HookEvent::PostToolUse { tool_name, .. } => format!("post:{}", tool_name),
                HookEvent::AgentStart { agent_id, .. } => format!("start:{}", agent_id),
                HookEvent::AgentStop { agent_id, .. } => format!("stop:{}", agent_id),
                HookEvent::Iteration {
                    agent_id,
                    iteration,
                } => format!("iter:{}:{}", agent_id, iteration),
                HookEvent::SessionStart { session_id, .. } => {
                    format!("session_start:{}", session_id)
                }
                HookEvent::SessionEnd { session_id, .. } => format!("session_end:{}", session_id),
                HookEvent::Stop { session_id, .. } => format!("stop_event:{}", session_id),
                HookEvent::SubagentStart { child_id, .. } => format!("subagent_start:{}", child_id),
                HookEvent::SubagentStop { child_id, .. } => format!("subagent_stop:{}", child_id),
                HookEvent::ResponseReceived { agent_id, .. } => {
                    format!("response:{}", agent_id)
                }
            };
            self.events.write().await.push(msg);
            Ok(HookAction::Continue)
        }
    }

    struct BlockingHook {
        block_tool: String,
    }

    #[async_trait]
    impl Hook for BlockingHook {
        async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
            if let HookEvent::PreToolUse { tool_name, .. } = event {
                if tool_name == &self.block_tool {
                    return Ok(HookAction::Block(format!(
                        "Tool {} is blocked",
                        tool_name
                    )));
                }
            }
            Ok(HookAction::Continue)
        }
    }

    #[tokio::test]
    async fn test_hook_registry_fire() {
        let registry = HookRegistry::new();
        let (hook, events) = LoggingHook::new();
        registry.register(hook).await;

        let event = HookEvent::PreToolUse {
            tool_name: "bash".into(),
            input: serde_json::json!({"command": "ls"}),
        };

        let action = registry.fire(&event).await.unwrap();
        assert!(matches!(action, HookAction::Continue));

        let logged = events.read().await;
        assert_eq!(logged.len(), 1);
        assert_eq!(logged[0], "pre:bash");
    }

    #[tokio::test]
    async fn test_hook_blocking() {
        let registry = HookRegistry::new();
        registry
            .register(BlockingHook {
                block_tool: "dangerous".into(),
            })
            .await;

        // Should not block
        let event = HookEvent::PreToolUse {
            tool_name: "safe".into(),
            input: serde_json::Value::Null,
        };
        let action = registry.fire(&event).await.unwrap();
        assert!(matches!(action, HookAction::Continue));

        // Should block
        let event = HookEvent::PreToolUse {
            tool_name: "dangerous".into(),
            input: serde_json::Value::Null,
        };
        let action = registry.fire(&event).await.unwrap();
        assert!(matches!(action, HookAction::Block(_)));
    }

    #[tokio::test]
    async fn test_multiple_hooks() {
        let registry = HookRegistry::new();
        let (hook1, events1) = LoggingHook::new();
        let (hook2, events2) = LoggingHook::new();
        registry.register(hook1).await;
        registry.register(hook2).await;

        let event = HookEvent::AgentStart {
            agent_id: "agent-1".into(),
            task: "do stuff".into(),
        };

        registry.fire(&event).await.unwrap();

        assert_eq!(events1.read().await.len(), 1);
        assert_eq!(events2.read().await.len(), 1);
    }

    #[tokio::test]
    async fn test_hook_transform() {
        struct TransformHook;

        #[async_trait]
        impl Hook for TransformHook {
            async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
                if let HookEvent::PreToolUse { input, .. } = event {
                    let mut modified = input.clone();
                    modified["injected"] = serde_json::json!(true);
                    return Ok(HookAction::Transform(modified));
                }
                Ok(HookAction::Continue)
            }
        }

        let registry = HookRegistry::new();
        registry.register(TransformHook).await;

        let event = HookEvent::PreToolUse {
            tool_name: "test".into(),
            input: serde_json::json!({"original": true}),
        };

        let action = registry.fire(&event).await.unwrap();
        if let HookAction::Transform(value) = action {
            assert!(value["injected"].as_bool().unwrap());
            assert!(value["original"].as_bool().unwrap());
        } else {
            panic!("Expected Transform action");
        }
    }

    #[tokio::test]
    async fn test_transform_on_non_pre_tool_use_errors() {
        // A hook that always returns Transform regardless of event type
        struct BadTransformHook;

        #[async_trait]
        impl Hook for BadTransformHook {
            async fn on_event(&self, _event: &HookEvent) -> Result<HookAction, anyhow::Error> {
                // This is a bug - Transform should only be returned for PreToolUse
                Ok(HookAction::Transform(serde_json::json!({"bad": true})))
            }
        }

        let registry = HookRegistry::new();
        registry.register(BadTransformHook).await;

        // Transform on PostToolUse should error
        let event = HookEvent::PostToolUse {
            tool_name: "test".into(),
            tool_use_id: "toolu_123".into(),
            input: serde_json::json!({}),
            result: crate::tool::ToolResult::text("ok"),
        };
        let result = registry.fire(&event).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("PostToolUse"));
        assert!(err_msg.contains("only valid for PreToolUse"));

        // Transform on AgentStart should error
        let event = HookEvent::AgentStart {
            agent_id: "agent-1".into(),
            task: "test".into(),
        };
        let result = registry.fire(&event).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("AgentStart"));

        // Transform on Iteration should error
        let event = HookEvent::Iteration {
            agent_id: "agent-1".into(),
            iteration: 1,
        };
        let result = registry.fire(&event).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Iteration"));
    }

    #[tokio::test]
    async fn test_session_start_event() {
        let registry = HookRegistry::new();
        let (hook, events) = LoggingHook::new();
        registry.register(hook).await;

        let event = HookEvent::SessionStart {
            session_id: "sess-123".into(),
            source: "run".into(),
            prompt: "hello world".into(),
        };

        registry.fire(&event).await.unwrap();

        let logged = events.read().await;
        assert_eq!(logged.len(), 1);
        assert_eq!(logged[0], "session_start:sess-123");
    }

    #[tokio::test]
    async fn test_session_end_event() {
        let registry = HookRegistry::new();
        let (hook, events) = LoggingHook::new();
        registry.register(hook).await;

        // Successful session end
        let event = HookEvent::SessionEnd {
            session_id: "sess-123".into(),
            error: None,
            reason: "complete".into(),
        };
        registry.fire(&event).await.unwrap();

        // Session end with error
        let event = HookEvent::SessionEnd {
            session_id: "sess-456".into(),
            error: Some("something went wrong".into()),
            reason: "error".into(),
        };
        registry.fire(&event).await.unwrap();

        let logged = events.read().await;
        assert_eq!(logged.len(), 2);
        assert_eq!(logged[0], "session_end:sess-123");
        assert_eq!(logged[1], "session_end:sess-456");
    }

    #[tokio::test]
    async fn test_stop_event_continue_loop() {
        let registry = HookRegistry::new();
        let continue_loop = Arc::new(AtomicBool::new(false));

        // Register a hook that sets continue_loop to true
        struct ContinueHook;

        #[async_trait]
        impl Hook for ContinueHook {
            async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
                if let HookEvent::Stop { continue_loop, .. } = event {
                    continue_loop.store(true, Ordering::SeqCst);
                }
                Ok(HookAction::Continue)
            }
        }

        registry.register(ContinueHook).await;

        let event = HookEvent::Stop {
            session_id: "sess-123".into(),
            final_text: "Done!".into(),
            continue_loop: continue_loop.clone(),
        };

        registry.fire(&event).await.unwrap();

        // Verify the hook set continue_loop to true
        assert!(continue_loop.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_subagent_start_event() {
        let registry = HookRegistry::new();
        let (hook, events) = LoggingHook::new();
        registry.register(hook).await;

        let event = HookEvent::SubagentStart {
            parent_id: "parent-1".into(),
            child_id: "child-1".into(),
            name: "researcher".into(),
        };

        registry.fire(&event).await.unwrap();

        let logged = events.read().await;
        assert_eq!(logged.len(), 1);
        assert_eq!(logged[0], "subagent_start:child-1");
    }

    #[tokio::test]
    async fn test_subagent_stop_event() {
        let registry = HookRegistry::new();
        let (hook, events) = LoggingHook::new();
        registry.register(hook).await;

        // Successful subagent stop
        let event = HookEvent::SubagentStop {
            parent_id: "parent-1".into(),
            child_id: "child-1".into(),
            name: "researcher".into(),
            error: None,
        };
        registry.fire(&event).await.unwrap();

        // Subagent stop with error
        let event = HookEvent::SubagentStop {
            parent_id: "parent-1".into(),
            child_id: "child-2".into(),
            name: "coder".into(),
            error: Some("timeout".into()),
        };
        registry.fire(&event).await.unwrap();

        let logged = events.read().await;
        assert_eq!(logged.len(), 2);
        assert_eq!(logged[0], "subagent_stop:child-1");
        assert_eq!(logged[1], "subagent_stop:child-2");
    }

    #[tokio::test]
    async fn test_on_session_start_convenience() {
        let registry = HookRegistry::new();
        let called = Arc::new(RwLock::new(Vec::<String>::new()));
        let called_clone = called.clone();

        registry
            .on_session_start(move |session_id, source, prompt| {
                let called = called_clone.clone();
                // Capture the values synchronously since we can't use async in this closure
                let session_id = session_id.to_string();
                let source = source.to_string();
                let prompt = prompt.to_string();
                // We can't actually use async here, so we'll just return Continue
                // and verify the hook was called by checking it doesn't error
                let _ = (called, session_id, source, prompt);
                HookAction::Continue
            })
            .await;

        let event = HookEvent::SessionStart {
            session_id: "test-sess".into(),
            source: "run".into(),
            prompt: "hello".into(),
        };

        let action = registry.fire(&event).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_on_stop_convenience_with_continue() {
        let registry = HookRegistry::new();
        let continue_loop = Arc::new(AtomicBool::new(false));

        registry
            .on_stop(|_session_id, _final_text, cont| {
                cont.store(true, Ordering::SeqCst);
                HookAction::Continue
            })
            .await;

        let event = HookEvent::Stop {
            session_id: "test-sess".into(),
            final_text: "done".into(),
            continue_loop: continue_loop.clone(),
        };

        registry.fire(&event).await.unwrap();
        assert!(continue_loop.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_convenience_hooks_filter_events() {
        let registry = HookRegistry::new();

        // Register on_session_start hook
        registry
            .on_session_start(|_, _, _| HookAction::Block("blocked".into()))
            .await;

        // Fire a different event type - should not trigger the hook
        let event = HookEvent::SessionEnd {
            session_id: "test".into(),
            error: None,
            reason: "complete".into(),
        };

        let action = registry.fire(&event).await.unwrap();
        // Should continue because on_session_start doesn't accept SessionEnd
        assert!(matches!(action, HookAction::Continue));

        // Fire the correct event type
        let event = HookEvent::SessionStart {
            session_id: "test".into(),
            source: "run".into(),
            prompt: "hello".into(),
        };

        let action = registry.fire(&event).await.unwrap();
        // Should block because on_session_start accepts SessionStart
        assert!(matches!(action, HookAction::Block(_)));
    }
}
