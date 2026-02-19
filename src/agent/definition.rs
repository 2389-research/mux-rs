// ABOUTME: Agent definition types - configuration for spawning subagents.
// ABOUTME: AgentRegistry holds available agent types that can be spawned.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

/// Definition of an agent type that can be spawned.
#[derive(Debug, Clone)]
pub struct AgentDefinition {
    /// Unique identifier for this agent type.
    pub agent_type: String,

    /// Model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o").
    /// If None, inherits from parent or uses default.
    pub model: Option<String>,

    /// System prompt for this agent.
    pub system_prompt: String,

    /// Tools this agent is allowed to use (allowlist).
    /// If None, inherits all tools from parent registry.
    pub allowed_tools: Option<Vec<String>>,

    /// Tools this agent is denied from using (denylist).
    /// Takes precedence over allowed_tools.
    pub denied_tools: Vec<String>,

    /// Whether to fork parent conversation context.
    /// If true, subagent sees parent's message history.
    /// If false, subagent starts fresh with only the task prompt.
    pub fork_context: bool,

    /// Maximum iterations for the think-act loop.
    pub max_iterations: usize,

    /// Whether to use streaming for LLM calls.
    /// When true, the agent uses `create_message_stream()` and fires
    /// `StreamDelta` / `StreamUsage` hooks for real-time token delivery.
    pub streaming: bool,
}

impl AgentDefinition {
    /// Create a new agent definition with required fields.
    pub fn new(agent_type: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self {
            agent_type: agent_type.into(),
            model: None,
            system_prompt: system_prompt.into(),
            allowed_tools: None,
            denied_tools: Vec::new(),
            fork_context: false,
            max_iterations: 10,
            streaming: false,
        }
    }

    /// Set the model for this agent.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set allowed tools (allowlist).
    pub fn allowed_tools(mut self, tools: Vec<String>) -> Self {
        self.allowed_tools = Some(tools);
        self
    }

    /// Set denied tools (denylist).
    pub fn denied_tools(mut self, tools: Vec<String>) -> Self {
        self.denied_tools = tools;
        self
    }

    /// Enable context forking from parent.
    pub fn fork_context(mut self, fork: bool) -> Self {
        self.fork_context = fork;
        self
    }

    /// Set maximum iterations.
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Enable or disable streaming for LLM calls.
    pub fn streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }
}

/// Registry of available agent definitions.
#[derive(Default)]
pub struct AgentRegistry {
    agents: Arc<RwLock<HashMap<String, AgentDefinition>>>,
}

impl AgentRegistry {
    /// Create a new empty agent registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an agent definition.
    pub async fn register(&self, definition: AgentDefinition) {
        let mut agents = self.agents.write().await;
        agents.insert(definition.agent_type.clone(), definition);
    }

    /// Get an agent definition by type.
    pub async fn get(&self, agent_type: &str) -> Option<AgentDefinition> {
        let agents = self.agents.read().await;
        agents.get(agent_type).cloned()
    }

    /// List all registered agent types.
    pub async fn list(&self) -> Vec<String> {
        let agents = self.agents.read().await;
        let mut types: Vec<_> = agents.keys().cloned().collect();
        types.sort();
        types
    }
}

impl Clone for AgentRegistry {
    fn clone(&self) -> Self {
        Self {
            agents: Arc::clone(&self.agents),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_definition_builder() {
        let def = AgentDefinition::new("researcher", "You are a research assistant.")
            .model("claude-sonnet-4-20250514")
            .allowed_tools(vec!["search".into(), "read_file".into()])
            .denied_tools(vec!["write_file".into()])
            .fork_context(true)
            .max_iterations(5);

        assert_eq!(def.agent_type, "researcher");
        assert_eq!(def.model, Some("claude-sonnet-4-20250514".into()));
        assert_eq!(
            def.allowed_tools,
            Some(vec!["search".into(), "read_file".into()])
        );
        assert_eq!(def.denied_tools, vec!["write_file".to_string()]);
        assert!(def.fork_context);
        assert_eq!(def.max_iterations, 5);
    }

    #[tokio::test]
    async fn test_agent_registry() {
        let registry = AgentRegistry::new();

        let def = AgentDefinition::new("coder", "You write code.");
        registry.register(def).await;

        let retrieved = registry.get("coder").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().agent_type, "coder");

        let missing = registry.get("missing").await;
        assert!(missing.is_none());

        let list = registry.list().await;
        assert_eq!(list, vec!["coder"]);
    }
}
