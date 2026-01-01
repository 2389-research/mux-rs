// ABOUTME: Hook system for extensibility in tool and agent lifecycle.
// ABOUTME: Provides events, actions, and a registry for hook management.

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
                        return Err(anyhow::anyhow!(
                            "HookAction::Transform is only valid for PreToolUse events, got {:?}",
                            std::mem::discriminant(&current_event)
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
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
