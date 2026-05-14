// ABOUTME: SubAgent runner - executes the think-act loop for a spawned agent.
// ABOUTME: Handles tool execution, conversation management, hooks, and result aggregation.

use std::sync::Arc;

use uuid::Uuid;

use super::definition::AgentDefinition;
use super::filter::FilteredRegistry;
use futures::StreamExt;

use crate::error::LlmError;
use crate::hook::{HookAction, HookEvent, HookRegistry};
use crate::llm::stream_accumulator::StreamAccumulator;
use crate::llm::{ContentBlock, LlmClient, Message, Request, Response, Role, StreamEvent, Usage};
use crate::permission::{ApprovalContext, ApprovalHandler};
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

    /// Optional hook registry for lifecycle events.
    hooks: Option<Arc<HookRegistry>>,

    /// Optional approval handler for tools requiring user approval.
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
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
            hooks: None,
            approval_handler: None,
        }
    }

    /// Resume a subagent from a previous transcript.
    ///
    /// This creates an agent that continues from where a previous run left off,
    /// preserving the conversation history and agent ID.
    ///
    /// Note: `tool_use_count` and `usage` are reset to zero for the resumed run.
    /// The returned `SubAgentResult` will only reflect metrics from this run,
    /// not cumulative totals across all runs. To track totals, accumulate the
    /// results from each run externally.
    pub fn resume(
        agent_id: String,
        definition: AgentDefinition,
        client: Arc<dyn LlmClient>,
        registry: Registry,
        transcript: Vec<Message>,
    ) -> Self {
        let tools = FilteredRegistry::new(registry)
            .allowed(definition.allowed_tools.clone())
            .denied(definition.denied_tools.clone());

        Self {
            agent_id,
            definition,
            client,
            tools,
            messages: transcript,
            tool_use_count: 0,
            usage: Usage::default(),
            hooks: None,
            approval_handler: None,
        }
    }

    /// Set the hook registry for lifecycle events.
    pub fn with_hooks(mut self, hooks: Arc<HookRegistry>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Set the approval handler for tools requiring user approval.
    pub fn with_approval_handler(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    /// Get the agent ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Get the current transcript (conversation history).
    ///
    /// Use this to save the conversation for later resumption.
    pub fn transcript(&self) -> &[Message] {
        &self.messages
    }

    /// Get the current accumulated token usage.
    ///
    /// Useful for retrieving partial usage after an error (e.g., max iterations).
    pub fn usage(&self) -> &Usage {
        &self.usage
    }

    /// Get the current tool use count.
    pub fn tool_use_count(&self) -> usize {
        self.tool_use_count
    }

    /// Fork conversation context from a parent agent.
    pub fn fork_messages(&mut self, parent_messages: Vec<Message>) {
        self.messages = parent_messages;
    }

    /// Fire a hook event and handle the result.
    async fn fire_hook(&self, event: HookEvent) -> Result<HookAction, LlmError> {
        if let Some(hooks) = &self.hooks {
            hooks.fire(&event).await.map_err(|e| LlmError::Api {
                status: 0,
                message: format!("Hook error: {}", e),
            })
        } else {
            Ok(HookAction::Continue)
        }
    }

    /// Run the agent on a task and return the result.
    pub async fn run(&mut self, task: &str) -> Result<SubAgentResult, LlmError> {
        self.run_internal(task.to_string(), Message::user(task))
            .await
    }

    /// Run the agent on a pre-built user message containing content blocks
    /// (text, media, etc.). Use this when the user turn includes attachments
    /// that need to reach the provider alongside plain text.
    ///
    /// The derived task string used for logging/hooks is the concatenation
    /// of any text blocks in `blocks`; non-text blocks are not stringified.
    pub async fn run_with_blocks(
        &mut self,
        blocks: Vec<ContentBlock>,
    ) -> Result<SubAgentResult, LlmError> {
        // Derive a logical task string from text blocks for the AgentStart
        // hook. Non-text blocks (media, etc.) are not stringified; they
        // reach the provider via the Message itself.
        let task = blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        self.run_internal(task, Message::user_with(blocks)).await
    }

    /// Shared body for `run` and `run_with_blocks`.
    async fn run_internal(
        &mut self,
        task: String,
        user_message: Message,
    ) -> Result<SubAgentResult, LlmError> {
        // Fire AgentStart hook
        self.fire_hook(HookEvent::AgentStart {
            agent_id: self.agent_id.clone(),
            task,
        })
        .await?;

        // Add the task as a user message
        self.messages.push(user_message);

        let mut iterations = 0;

        // Think-act loop
        let result = loop {
            iterations += 1;

            // Fire Iteration hook
            self.fire_hook(HookEvent::Iteration {
                agent_id: self.agent_id.clone(),
                iteration: iterations,
            })
            .await?;

            if iterations > self.definition.max_iterations {
                return Err(LlmError::Api {
                    status: 0,
                    message: format!(
                        "Agent exceeded max iterations ({})",
                        self.definition.max_iterations
                    ),
                });
            }

            // Build the request - model must be configured
            let model = self.definition.model.clone().ok_or_else(|| {
                LlmError::Configuration(
                    "Agent definition has no model configured. Set model in AgentDefinition."
                        .to_string(),
                )
            })?;

            // System prompt: prefer system_blocks (cacheable) when present,
            // otherwise fall back to the legacy single-string system_prompt.
            let mut request = Request::new(&model);
            if !self.definition.system_blocks.is_empty() {
                request = request.system_blocks(self.definition.system_blocks.clone());
            } else {
                request = request.system(&self.definition.system_prompt);
            }

            // Tools: optionally mark cache_control: ephemeral on the LAST tool
            // definition. Anthropic prompt-caching uses cache breakpoints —
            // a single cache_control marker caches EVERYTHING BEFORE IT (in
            // request order) as one cached unit. Marking every tool would
            // create N breakpoints; Anthropic caps at 4 per request and
            // rejects with HTTP 400. Marking just the last tool caches the
            // entire tool block with a single breakpoint.
            //
            // ORDER MATTERS for caching: the underlying tool registry is
            // backed by a HashMap, so `to_definitions()` returns tools in
            // non-deterministic order. We sort by name here so the same
            // tool ends up "last" across calls — without this, the cache
            // breakpoint position shifts between requests and the cache
            // never hits (every call appears to be cacheable-content-changed
            // to Anthropic's deduplicator).
            let mut tool_defs = self.tools.to_definitions().await;
            tool_defs.sort_by(|a, b| a.name.cmp(&b.name));
            if self.definition.cache_tools
                && let Some(last) = tool_defs.last_mut()
            {
                last.cache_control = Some(crate::llm::CacheControl::ephemeral());
            }

            let request = request
                .messages(self.messages.clone())
                .tools(tool_defs)
                .max_tokens(4096);

            // Call the LLM
            let response = self.call_llm(&request).await?;

            // Aggregate usage
            self.usage.input_tokens += response.usage.input_tokens;
            self.usage.output_tokens += response.usage.output_tokens;
            self.usage.cache_read_tokens += response.usage.cache_read_tokens;
            self.usage.cache_write_tokens += response.usage.cache_write_tokens;

            // Fire ResponseReceived hook for streaming callbacks
            let response_text = response.text();
            let tool_uses: Vec<(String, String, serde_json::Value)> = response
                .content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::ToolUse { name, id, input } = block {
                        Some((name.clone(), id.clone(), input.clone()))
                    } else {
                        None
                    }
                })
                .collect();

            self.fire_hook(HookEvent::ResponseReceived {
                agent_id: self.agent_id.clone(),
                text: response_text.clone(),
                tool_uses: tool_uses.clone(),
            })
            .await?;

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

                        // Fire PreToolUse hook
                        let hook_action = self
                            .fire_hook(HookEvent::PreToolUse {
                                tool_name: name.clone(),
                                input: input.clone(),
                            })
                            .await?;

                        // Check if hook blocked the tool, track effective input
                        let (tool_result, effective_input) = match hook_action {
                            HookAction::Block(msg) => (
                                crate::tool::ToolResult::error(format!("Blocked by hook: {}", msg)),
                                input.clone(),
                            ),
                            HookAction::Transform(new_input) => {
                                // Use transformed input
                                let result = self.execute_tool(name, new_input.clone()).await;
                                (result, new_input)
                            }
                            HookAction::Continue => {
                                let result = self.execute_tool(name, input.clone()).await;
                                (result, input.clone())
                            }
                        };

                        // Fire PostToolUse hook with the effective input (after any transform)
                        self.fire_hook(HookEvent::PostToolUse {
                            tool_name: name.clone(),
                            tool_use_id: id.clone(),
                            input: effective_input,
                            result: tool_result.clone(),
                        })
                        .await?;

                        let result_block = if tool_result.is_error {
                            ContentBlock::tool_error(id, &tool_result.content)
                        } else {
                            ContentBlock::tool_result(id, &tool_result.content)
                        };

                        tool_results.push(result_block);
                    }
                }

                // Add tool results to history
                self.messages.push(Message::tool_results(tool_results));

                // Continue the loop
                continue;
            }

            // No tool use - agent is done
            let content = response.text();

            break SubAgentResult {
                agent_id: self.agent_id.clone(),
                content,
                tool_use_count: self.tool_use_count,
                usage: self.usage.clone(),
                iterations,
            };
        };

        // Fire AgentStop hook
        self.fire_hook(HookEvent::AgentStop {
            agent_id: self.agent_id.clone(),
            result: result.clone(),
        })
        .await?;

        Ok(result)
    }

    /// Call the LLM, using streaming or non-streaming based on definition.
    ///
    /// When streaming is enabled, fires `StreamDelta` hooks for text tokens
    /// and `StreamUsage` hooks for MessageDelta events, then assembles the
    /// final `Response` from accumulated stream events.
    async fn call_llm(&self, request: &Request) -> Result<Response, LlmError> {
        if !self.definition.streaming {
            return self.client.create_message(request).await;
        }

        // Streaming path
        let mut stream = self.client.create_message_stream(request);
        let mut accumulator = StreamAccumulator::new();
        let mut message_id = String::new();
        let mut model = String::new();
        let mut stop_reason = None;
        let mut usage = Usage::default();

        while let Some(event_result) = stream.next().await {
            let event = event_result?;

            match &event {
                StreamEvent::MessageStart {
                    id: msg_id,
                    model: msg_model,
                    usage: start_usage,
                } => {
                    message_id = msg_id.clone();
                    model = msg_model.clone();
                    // Anthropic carries the cache_creation_input_tokens and
                    // cache_read_input_tokens in message_start, with
                    // output_tokens=0. MessageDelta later updates output_tokens
                    // and re-sends input/cache. We seed usage here so we don't
                    // lose cache info if a provider only sends it in start.
                    usage = start_usage.clone();
                }
                StreamEvent::ContentBlockDelta { text, .. } if !accumulator.in_tool_use() => {
                    // Fire StreamDelta hook for text tokens
                    self.fire_hook(HookEvent::StreamDelta {
                        agent_id: self.agent_id.clone(),
                        text: text.clone(),
                    })
                    .await?;
                }
                StreamEvent::MessageDelta {
                    stop_reason: sr,
                    usage: delta_usage,
                } => {
                    stop_reason = *sr;
                    // Merge with usage from MessageStart: take MessageDelta
                    // values that are non-zero (it has the final output_tokens
                    // and may or may not re-send cache fields), preserve any
                    // non-zero values from MessageStart otherwise.
                    let mut merged = delta_usage.clone();
                    if merged.input_tokens == 0 {
                        merged.input_tokens = usage.input_tokens;
                    }
                    if merged.cache_read_tokens == 0 {
                        merged.cache_read_tokens = usage.cache_read_tokens;
                    }
                    if merged.cache_write_tokens == 0 {
                        merged.cache_write_tokens = usage.cache_write_tokens;
                    }
                    usage = merged;
                    // Fire StreamUsage hook
                    self.fire_hook(HookEvent::StreamUsage {
                        agent_id: self.agent_id.clone(),
                        usage: usage.clone(),
                    })
                    .await?;
                }
                _ => {}
            }

            accumulator.handle_event(&event);
        }

        Ok(Response {
            id: message_id,
            content: accumulator.into_content(),
            stop_reason: stop_reason.unwrap_or(crate::llm::StopReason::EndTurn),
            model,
            usage,
        })
    }

    /// Execute a tool and return the result.
    ///
    /// If the tool requires approval and an approval handler is set,
    /// this will request approval before executing. If denied, returns
    /// an error result without executing the tool.
    async fn execute_tool(&self, name: &str, input: serde_json::Value) -> crate::tool::ToolResult {
        match self.tools.get(name).await {
            Some(tool) => {
                // Check if tool requires approval
                if tool.requires_approval(&input) {
                    if let Some(handler) = &self.approval_handler {
                        let context = ApprovalContext {
                            tool_description: tool.description().to_string(),
                            request_id: Uuid::new_v4().to_string(),
                        };

                        match handler.request_approval(name, &input, &context).await {
                            Ok(true) => {
                                // Approved - continue to execution
                            }
                            Ok(false) => {
                                return crate::tool::ToolResult::error(format!(
                                    "Tool '{}' execution denied by approval handler",
                                    name
                                ));
                            }
                            Err(e) => {
                                return crate::tool::ToolResult::error(format!(
                                    "Approval handler error for '{}': {}",
                                    name, e
                                ));
                            }
                        }
                    }
                    // No approval handler but tool requires approval - block by default
                    else {
                        return crate::tool::ToolResult::error(format!(
                            "Tool '{}' requires approval but no approval handler is set",
                            name
                        ));
                    }
                }

                // Execute the tool
                match tool.execute(input).await {
                    Ok(r) => r,
                    Err(e) => crate::tool::ToolResult::error(e.to_string()),
                }
            }
            None => {
                crate::tool::ToolResult::error(format!("Tool '{}' not found or not allowed", name))
            }
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
                cache_read_tokens: 20,
                cache_write_tokens: 10,
            },
            iterations: 2,
        };

        assert_eq!(result.agent_id, "test-123");
        assert_eq!(result.tool_use_count, 3);
        assert_eq!(result.iterations, 2);
        assert_eq!(result.usage.cache_read_tokens, 20);
        assert_eq!(result.usage.cache_write_tokens, 10);
    }
}
