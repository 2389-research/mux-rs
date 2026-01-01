// ABOUTME: TaskTool - a tool that spawns subagents to handle delegated tasks.
// ABOUTME: Supports resume from previous transcripts for long-running tasks.

use std::sync::Arc;

use async_trait::async_trait;

use super::definition::AgentRegistry;
use super::runner::SubAgent;
use super::transcript::TranscriptStore;
use crate::llm::LlmClient;
use crate::tool::{Registry, Tool, ToolResult};

/// A tool that spawns subagents to handle delegated tasks.
///
/// When an LLM calls this tool, it spawns a subagent of the specified type,
/// runs it on the given task, and returns the result. Optionally supports
/// resuming agents from previous transcripts.
pub struct TaskTool {
    /// Registry of available agent definitions.
    agent_registry: AgentRegistry,

    /// Registry of tools available to subagents.
    tool_registry: Registry,

    /// Factory function to create LLM clients for subagents.
    client_factory: Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>,

    /// Optional transcript store for agent resume.
    transcript_store: Option<Arc<dyn TranscriptStore>>,
}

impl TaskTool {
    /// Create a new TaskTool.
    ///
    /// # Arguments
    /// * `agent_registry` - Registry of available agent definitions
    /// * `tool_registry` - Registry of tools available to subagents
    /// * `client_factory` - Function that creates an LLM client for a given model name
    pub fn new<F>(agent_registry: AgentRegistry, tool_registry: Registry, client_factory: F) -> Self
    where
        F: Fn(&str) -> Arc<dyn LlmClient> + Send + Sync + 'static,
    {
        Self {
            agent_registry,
            tool_registry,
            client_factory: Arc::new(client_factory),
            transcript_store: None,
        }
    }

    /// Create with a default client (ignores model parameter).
    pub fn with_default_client(
        agent_registry: AgentRegistry,
        tool_registry: Registry,
        client: Arc<dyn LlmClient>,
    ) -> Self {
        let client_clone = client.clone();
        Self::new(agent_registry, tool_registry, move |_| client_clone.clone())
    }

    /// Set a transcript store for agent resume capability.
    pub fn with_transcript_store(mut self, store: Arc<dyn TranscriptStore>) -> Self {
        self.transcript_store = Some(store);
        self
    }
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "task"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific task. Use this to delegate work to specialized agents. \
         Optionally resume from a previous agent's transcript."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "description": "The type of agent to spawn (must be registered in the agent registry)"
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
            "required": ["agent_type", "task", "description"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        // Parse parameters
        let agent_type = params
            .get("agent_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: agent_type"))?;

        let task = params
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing required parameter: task"))?;

        let resume_agent_id = params.get("resume_agent_id").and_then(|v| v.as_str());

        // Get agent definition
        let definition = match self.agent_registry.get(agent_type).await {
            Some(def) => def,
            None => {
                let available = self.agent_registry.list().await;
                return Ok(ToolResult::error(format!(
                    "Agent type '{}' not found. Available types: {}",
                    agent_type,
                    available.join(", ")
                )));
            }
        };

        // Determine which model/client to use
        let model = definition
            .model
            .clone()
            .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());
        let client = (self.client_factory)(&model);

        // Create the subagent (new or resumed)
        let mut subagent = if let Some(agent_id) = resume_agent_id {
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

        match subagent.run(task).await {
            Ok(result) => {
                // Save transcript for future resume
                if let Some(store) = &self.transcript_store {
                    if let Err(e) = store.save(&result.agent_id, subagent.transcript()).await {
                        // Log but don't fail - transcript save is best-effort
                        eprintln!("Warning: failed to save transcript: {}", e);
                    }
                }

                let output = serde_json::json!({
                    "agent_id": result.agent_id,
                    "content": result.content,
                    "tool_use_count": result.tool_use_count,
                    "iterations": result.iterations,
                    "tokens": {
                        "input": result.usage.input_tokens,
                        "output": result.usage.output_tokens
                    }
                });
                Ok(ToolResult::text(serde_json::to_string_pretty(&output)?))
            }
            Err(e) => Ok(ToolResult::error(format!("Subagent error: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentDefinition;

    #[tokio::test]
    async fn test_task_tool_schema() {
        let agent_registry = AgentRegistry::new();
        let tool_registry = Registry::new();

        // Mock client factory
        let tool = TaskTool::new(agent_registry, tool_registry, |_| {
            panic!("Should not be called in schema test")
        });

        let schema = tool.schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["properties"].get("agent_type").is_some());
        assert!(schema["properties"].get("task").is_some());
    }

    #[tokio::test]
    async fn test_task_tool_missing_agent_type() {
        let agent_registry = AgentRegistry::new();
        agent_registry
            .register(AgentDefinition::new("researcher", "You research things"))
            .await;

        let tool_registry = Registry::new();

        let tool = TaskTool::new(agent_registry, tool_registry, |_| {
            panic!("Should not be called")
        });

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
