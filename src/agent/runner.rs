// ABOUTME: SubAgent runner - executes the think-act loop for a spawned agent.
// ABOUTME: Handles tool execution, conversation management, and result aggregation.

use std::sync::Arc;

use uuid::Uuid;

use super::definition::AgentDefinition;
use super::filter::FilteredRegistry;
use crate::error::LlmError;
use crate::llm::{ContentBlock, LlmClient, Message, Request, Role, Usage};
use crate::tool::Registry;

/// Result from running a subagent.
#[derive(Debug, Clone)]
pub struct SubAgentResult {
    /// Unique identifier for this agent run.
    pub agent_id: String,

    /// Final text content from the agent.
    pub content: String,

    /// Number of tool calls made during execution.
    pub tool_use_count: usize,

    /// Total token usage across all LLM calls.
    pub usage: Usage,

    /// Number of iterations in the think-act loop.
    pub iterations: usize,
}

/// A subagent that can be spawned to handle a specific task.
pub struct SubAgent {
    /// Unique identifier for this agent instance.
    agent_id: String,

    /// The agent definition (type, system prompt, etc.).
    definition: AgentDefinition,

    /// The LLM client to use.
    client: Arc<dyn LlmClient>,

    /// Filtered tool registry for this agent.
    tools: FilteredRegistry,

    /// Conversation history.
    messages: Vec<Message>,

    /// Running total of tool calls.
    tool_use_count: usize,

    /// Running total of token usage.
    usage: Usage,
}

impl SubAgent {
    /// Create a new subagent from a definition.
    pub fn new(
        definition: AgentDefinition,
        client: Arc<dyn LlmClient>,
        registry: Registry,
    ) -> Self {
        let tools = FilteredRegistry::new(registry)
            .allowed(definition.allowed_tools.clone())
            .denied(definition.denied_tools.clone());

        Self {
            agent_id: Uuid::new_v4().to_string(),
            definition,
            client,
            tools,
            messages: Vec::new(),
            tool_use_count: 0,
            usage: Usage::default(),
        }
    }

    /// Get the agent ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Fork conversation context from a parent agent.
    pub fn fork_messages(&mut self, parent_messages: Vec<Message>) {
        self.messages = parent_messages;
    }

    /// Run the agent on a task and return the result.
    pub async fn run(&mut self, task: &str) -> Result<SubAgentResult, LlmError> {
        // Add the task as a user message
        self.messages.push(Message::user(task));

        let mut iterations = 0;

        // Think-act loop
        loop {
            iterations += 1;

            if iterations > self.definition.max_iterations {
                return Err(LlmError::Api {
                    status: 0,
                    message: format!(
                        "Agent exceeded max iterations ({})",
                        self.definition.max_iterations
                    ),
                });
            }

            // Build the request
            let model = self
                .definition
                .model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

            let request = Request::new(&model)
                .system(&self.definition.system_prompt)
                .messages(self.messages.clone())
                .tools(self.tools.to_definitions().await)
                .max_tokens(4096);

            // Call the LLM
            let response = self.client.create_message(&request).await?;

            // Aggregate usage
            self.usage.input_tokens += response.usage.input_tokens;
            self.usage.output_tokens += response.usage.output_tokens;

            // Check for tool use
            if response.has_tool_use() {
                // Add assistant response to history
                self.messages.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });

                // Execute each tool
                let mut tool_results = Vec::new();

                for block in &response.content {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        self.tool_use_count += 1;

                        let result = match self.tools.get(name).await {
                            Some(tool) => {
                                match tool.execute(input.clone()).await {
                                    Ok(r) => {
                                        if r.is_error {
                                            ContentBlock::tool_error(id, &r.content)
                                        } else {
                                            ContentBlock::tool_result(id, &r.content)
                                        }
                                    }
                                    Err(e) => ContentBlock::tool_error(id, e.to_string()),
                                }
                            }
                            None => ContentBlock::tool_error(
                                id,
                                format!("Tool '{}' not found or not allowed", name),
                            ),
                        };

                        tool_results.push(result);
                    }
                }

                // Add tool results to history
                self.messages.push(Message::tool_results(tool_results));

                // Continue the loop
                continue;
            }

            // No tool use - agent is done
            let content = response.text();

            return Ok(SubAgentResult {
                agent_id: self.agent_id.clone(),
                content,
                tool_use_count: self.tool_use_count,
                usage: self.usage.clone(),
                iterations,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subagent_result() {
        let result = SubAgentResult {
            agent_id: "test-123".into(),
            content: "Done".into(),
            tool_use_count: 3,
            usage: Usage {
                input_tokens: 100,
                output_tokens: 50,
            },
            iterations: 2,
        };

        assert_eq!(result.agent_id, "test-123");
        assert_eq!(result.tool_use_count, 3);
        assert_eq!(result.iterations, 2);
    }
}
