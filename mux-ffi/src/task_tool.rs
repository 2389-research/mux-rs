// ABOUTME: FfiTaskTool - FFI wrapper for TaskTool that proxies events to Swift.
// ABOUTME: Bridges mux SubAgent hook events to SubagentEventHandler callbacks.

use std::sync::Arc;

use async_trait::async_trait;

use crate::callback::SubagentEventHandler;
use mux::agent::{AgentDefinition, AgentRegistry, SubAgent, TranscriptStore};
use mux::hook::{Hook, HookAction, HookEvent, HookRegistry};
use mux::llm::LlmClient;
use mux::tool::{Registry, Tool, ToolResult};

/// A hook that proxies SubAgent events to Swift's SubagentEventHandler.
struct SubagentEventProxyHook {
    agent_id: String,
    handler: Arc<Box<dyn SubagentEventHandler>>,
}

impl SubagentEventProxyHook {
    fn new(agent_id: String, handler: Arc<Box<dyn SubagentEventHandler>>) -> Self {
        Self { agent_id, handler }
    }
}

#[async_trait]
impl Hook for SubagentEventProxyHook {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        // Clone handler to move into spawn_blocking
        let handler = self.handler.clone();
        let agent_id = self.agent_id.clone();

        match event {
            HookEvent::PreToolUse { tool_name, input } => {
                let tool_name = tool_name.clone();
                let arguments_json =
                    serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());

                tokio::task::spawn_blocking(move || {
                    handler.on_tool_use(agent_id, tool_name, arguments_json);
                })
                .await
                .ok();
            }
            HookEvent::PostToolUse {
                tool_name, result, ..
            } => {
                let tool_name = tool_name.clone();
                let result_content = result.content.clone();
                let is_error = result.is_error;

                tokio::task::spawn_blocking(move || {
                    handler.on_tool_result(agent_id, tool_name, result_content, is_error);
                })
                .await
                .ok();
            }
            HookEvent::Iteration { iteration, .. } => {
                let iteration = *iteration as u32;

                tokio::task::spawn_blocking(move || {
                    handler.on_iteration(agent_id, iteration);
                })
                .await
                .ok();
            }
            HookEvent::AgentStart { .. } | HookEvent::AgentStop { .. } => {
                // These are handled at the FfiTaskTool level, not via hooks
            }
        }

        Ok(HookAction::Continue)
    }

    fn accepts(&self, _event: &HookEvent) -> bool {
        true
    }
}

/// FFI TaskTool that spawns subagents with event streaming to Swift.
pub struct FfiTaskTool {
    /// Registry of available agent definitions.
    agent_registry: AgentRegistry,

    /// Registry of tools available to subagents.
    tool_registry: Registry,

    /// Factory function to create LLM clients for subagents.
    client_factory: Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>,

    /// Optional transcript store for agent resume.
    transcript_store: Option<Arc<dyn TranscriptStore>>,

    /// Event handler for streaming updates to Swift.
    event_handler: Arc<Box<dyn SubagentEventHandler>>,
}

impl FfiTaskTool {
    /// Create a new FfiTaskTool.
    pub fn new<F>(
        agent_registry: AgentRegistry,
        tool_registry: Registry,
        client_factory: F,
        event_handler: Box<dyn SubagentEventHandler>,
    ) -> Self
    where
        F: Fn(&str) -> Arc<dyn LlmClient> + Send + Sync + 'static,
    {
        Self {
            agent_registry,
            tool_registry,
            client_factory: Arc::new(client_factory),
            transcript_store: None,
            event_handler: Arc::new(event_handler),
        }
    }

    /// Set a transcript store for agent resume capability.
    pub fn with_transcript_store(mut self, store: Arc<dyn TranscriptStore>) -> Self {
        self.transcript_store = Some(store);
        self
    }
}

#[async_trait]
impl Tool for FfiTaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific task. Use a registered agent_type OR provide a custom \
         system_prompt for ad-hoc agents. Optionally resume from a previous transcript."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
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
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        // Parse required parameters
        let task = params
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: task"))?;

        let description = params
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("subagent task");

        let resume_agent_id = params.get("resume_agent_id").and_then(|v| v.as_str());

        // Parse optional parameters for registered vs ad-hoc agents
        let agent_type = params.get("agent_type").and_then(|v| v.as_str());
        let system_prompt = params.get("system_prompt").and_then(|v| v.as_str());
        let model_param = params.get("model").and_then(|v| v.as_str());

        // Get or create agent definition
        let (definition, agent_type_name) = if let Some(agent_type) = agent_type {
            // Use registered agent
            match self.agent_registry.get(agent_type).await {
                Some(def) => (def, agent_type.to_string()),
                None => {
                    let available = self.agent_registry.list().await;
                    return Ok(ToolResult::error(format!(
                        "Agent type '{}' not found. Available types: {}",
                        agent_type,
                        available.join(", ")
                    )));
                }
            }
        } else if let Some(prompt) = system_prompt {
            // Create ad-hoc agent - model is required
            let model = match model_param {
                Some(m) => m,
                None => {
                    return Ok(ToolResult::error(
                        "Ad-hoc agents require a 'model' parameter (e.g., 'claude-sonnet-4-5-20250929')",
                    ));
                }
            };
            let def = AgentDefinition::new("adhoc", prompt).model(model);
            (def, "adhoc".to_string())
        } else {
            return Ok(ToolResult::error(
                "Must provide either 'agent_type' (registered agent) or 'system_prompt' (ad-hoc agent)",
            ));
        };

        // Determine which model/client to use - model must be set
        let model = match definition.model.clone() {
            Some(m) => m,
            None => {
                return Ok(ToolResult::error(
                    "Agent definition has no model configured. Set model in AgentConfig or provide 'model' parameter.",
                ));
            }
        };
        let client = (self.client_factory)(&model);

        // Create the subagent (new or resumed)
        let subagent = if let Some(agent_id) = resume_agent_id {
            // Try to load transcript for resume
            let transcript = if let Some(store) = &self.transcript_store {
                match store.load(agent_id).await {
                    Ok(Some(transcript)) => transcript,
                    Ok(None) => {
                        return Ok(ToolResult::error(format!(
                            "No transcript found for agent_id '{}'",
                            agent_id
                        )));
                    }
                    Err(e) => {
                        return Ok(ToolResult::error(format!(
                            "Failed to load transcript: {}",
                            e
                        )));
                    }
                }
            } else {
                return Ok(ToolResult::error(
                    "Cannot resume: no transcript store configured",
                ));
            };

            SubAgent::resume(
                agent_id.to_string(),
                definition,
                client,
                self.tool_registry.clone(),
                transcript,
            )
        } else {
            SubAgent::new(definition, client, self.tool_registry.clone())
        };

        let agent_id = subagent.agent_id().to_string();

        // Fire on_agent_started via spawn_blocking
        {
            let handler = self.event_handler.clone();
            let agent_id_clone = agent_id.clone();
            let agent_type_clone = agent_type_name.clone();
            let task_clone = task.to_string();
            let description = description.to_string();

            tokio::task::spawn_blocking(move || {
                handler.on_agent_started(agent_id_clone, agent_type_clone, task_clone, description);
            })
            .await
            .ok();
        }

        // Create hook registry with our proxy hook
        let hook_registry = Arc::new(HookRegistry::new());
        hook_registry
            .register(SubagentEventProxyHook::new(
                agent_id.clone(),
                self.event_handler.clone(),
            ))
            .await;

        // Attach hooks and run
        let mut subagent = subagent.with_hooks(hook_registry);

        match subagent.run(task).await {
            Ok(result) => {
                // Save transcript for future resume
                let transcript_saved = if let Some(store) = &self.transcript_store {
                    match store.save(&result.agent_id, subagent.transcript()).await {
                        Ok(()) => true,
                        Err(e) => {
                            eprintln!("Warning: failed to save transcript: {}", e);
                            false
                        }
                    }
                } else {
                    false
                };

                // Fire on_agent_completed via spawn_blocking
                {
                    let handler = self.event_handler.clone();
                    let agent_id = result.agent_id.clone();
                    let content = result.content.clone();
                    let tool_use_count = result.tool_use_count as u32;
                    let iterations = result.iterations as u32;

                    tokio::task::spawn_blocking(move || {
                        handler.on_agent_completed(
                            agent_id,
                            content,
                            tool_use_count,
                            iterations,
                            transcript_saved,
                        );
                    })
                    .await
                    .ok();
                }

                let output = serde_json::json!({
                    "agent_id": result.agent_id,
                    "content": result.content,
                    "tool_use_count": result.tool_use_count,
                    "iterations": result.iterations,
                    "tokens": {
                        "input": result.usage.input_tokens,
                        "output": result.usage.output_tokens
                    },
                    "transcript_saved": transcript_saved
                });
                Ok(ToolResult::text(serde_json::to_string_pretty(&output)?))
            }
            Err(e) => {
                // Fire on_agent_error via spawn_blocking
                {
                    let handler = self.event_handler.clone();
                    let agent_id = agent_id.clone();
                    let error = e.to_string();

                    tokio::task::spawn_blocking(move || {
                        handler.on_agent_error(agent_id, error);
                    })
                    .await
                    .ok();
                }

                Ok(ToolResult::error(format!("Subagent error: {}", e)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mux::prelude::AgentDefinition;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Mock event handler for testing
    struct MockEventHandler {
        started_count: AtomicU32,
        completed_count: AtomicU32,
        error_count: AtomicU32,
    }

    impl MockEventHandler {
        fn new() -> Self {
            Self {
                started_count: AtomicU32::new(0),
                completed_count: AtomicU32::new(0),
                error_count: AtomicU32::new(0),
            }
        }
    }

    impl SubagentEventHandler for MockEventHandler {
        fn on_agent_started(
            &self,
            _subagent_id: String,
            _agent_type: String,
            _task: String,
            _description: String,
        ) {
            self.started_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_tool_use(&self, _subagent_id: String, _tool_name: String, _arguments_json: String) {}

        fn on_tool_result(
            &self,
            _subagent_id: String,
            _tool_name: String,
            _result: String,
            _is_error: bool,
        ) {
        }

        fn on_iteration(&self, _subagent_id: String, _iteration: u32) {}

        fn on_agent_completed(
            &self,
            _subagent_id: String,
            _content: String,
            _tool_use_count: u32,
            _iterations: u32,
            _transcript_saved: bool,
        ) {
            self.completed_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_agent_error(&self, _subagent_id: String, _error: String) {
            self.error_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_ffi_task_tool_schema() {
        let agent_registry = AgentRegistry::new();
        let tool_registry = Registry::new();
        let handler = Box::new(MockEventHandler::new());

        let tool = FfiTaskTool::new(agent_registry, tool_registry, |_| {
            panic!("Should not be called in schema test")
        }, handler);

        let schema = tool.schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("agent_type").is_some());
        assert!(schema["properties"].get("task").is_some());
        assert!(schema["properties"].get("description").is_some());
    }

    #[tokio::test]
    async fn test_ffi_task_tool_missing_agent_type() {
        let agent_registry = AgentRegistry::new();
        agent_registry
            .register(AgentDefinition::new("researcher", "You research things"))
            .await;

        let tool_registry = Registry::new();
        let handler = Box::new(MockEventHandler::new());

        let tool = FfiTaskTool::new(agent_registry, tool_registry, |_| {
            panic!("Should not be called")
        }, handler);

        let result = tool
            .execute(serde_json::json!({
                "agent_type": "nonexistent",
                "task": "do something",
                "description": "test"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
        assert!(result.content.contains("researcher"));
    }
}
