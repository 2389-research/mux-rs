// ABOUTME: UniFFI bindings for mux - exposes Rust agentic library to Swift/Kotlin.
// ABOUTME: This crate provides the MuxEngine interface for GUI applications.

uniffi::setup_scaffolding!();

mod types;
pub use types::*;

mod engine;
pub use engine::*;

mod callback;
pub use callback::*;

mod bridge;
pub use bridge::*;

mod task_tool;
pub use task_tool::*;

mod callback_client;
pub use callback_client::*;

mod context;
pub use context::*;

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MuxFfiError {
    #[error("Engine error: {message}")]
    Engine { message: String },
    #[error("Agent not found: {name}")]
    AgentNotFound { name: String },
    #[error("Tool not found: {name}")]
    ToolNotFound { name: String },
    #[error("Provider not configured: {provider}")]
    ProviderNotConfigured { provider: String },
    #[error("LLM provider not found: {name}")]
    LlmProviderNotFound { name: String },
    #[error("Invalid transcript: {reason}")]
    TranscriptInvalid { reason: String },
    #[error("Hook failed: {reason}")]
    HookFailed { reason: String },
}

#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{CompactionMode, ModelContextConfig};
    use crate::types::AgentConfig;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }

    #[test]
    fn test_workspace_creation() {
        let ws = Workspace::new("test-workspace".to_string(), None);
        assert_eq!(ws.name, "test-workspace");
        assert!(ws.path.is_none());
        assert!(!ws.id.is_empty());
    }

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::anthropic("claude-sonnet-4-20250514".to_string());
        assert_eq!(config.provider, Provider::Anthropic);
    }

    #[test]
    fn test_engine_creation() {
        let engine = MuxEngine::new("/tmp/mux-test".to_string()).unwrap();
        assert!(engine.list_workspaces().is_empty());
    }

    #[test]
    fn test_workspace_crud() {
        let engine = MuxEngine::new("/tmp/mux-test-crud".to_string()).unwrap();

        // Create
        let ws = engine.create_workspace("Test Project".to_string(), None).unwrap();
        assert_eq!(ws.name, "Test Project");

        // List
        let workspaces = engine.list_workspaces();
        assert_eq!(workspaces.len(), 1);

        // Delete
        engine.delete_workspace(ws.id.clone()).unwrap();
        assert!(engine.list_workspaces().is_empty());
    }

    #[test]
    fn test_system_prompt() {
        let engine = MuxEngine::new("/tmp/mux-test-prompt".to_string()).unwrap();
        let ws = engine.create_workspace("Prompt Test".to_string(), None).unwrap();

        // Default is None
        assert!(engine.get_system_prompt(ws.id.clone()).is_none());

        // Set custom prompt
        engine
            .set_system_prompt(ws.id.clone(), Some("You are a pirate assistant.".to_string()))
            .unwrap();
        assert_eq!(
            engine.get_system_prompt(ws.id.clone()),
            Some("You are a pirate assistant.".to_string())
        );

        // Reset to default
        engine.set_system_prompt(ws.id.clone(), None).unwrap();
        assert!(engine.get_system_prompt(ws.id.clone()).is_none());

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_agent_registration() {
        let engine = MuxEngine::new("/tmp/mux-test-agents".to_string()).unwrap();

        // Register agent
        let config = AgentConfig::new(
            "researcher".to_string(),
            "You are a research assistant.".to_string(),
        );
        engine.register_agent(config).unwrap();

        // List agents
        let agents = engine.list_agents();
        assert!(agents.contains(&"researcher".to_string()));

        // Unregister
        engine.unregister_agent("researcher".to_string()).unwrap();
        assert!(engine.list_agents().is_empty());
    }

    #[test]
    fn test_provider_config() {
        let engine = MuxEngine::new("/tmp/mux-test-providers".to_string()).unwrap();

        // Set provider config with default models
        engine.set_provider_config(
            Provider::Anthropic,
            "sk-test".to_string(),
            None,
            Some("claude-sonnet-4-5-20250929".to_string()),
        );
        engine.set_provider_config(
            Provider::OpenAI,
            "sk-openai".to_string(),
            Some("https://api.openai.com".to_string()),
            Some("gpt-4o".to_string()),
        );

        // Verify API keys
        assert_eq!(engine.get_api_key(Provider::Anthropic), Some("sk-test".to_string()));
        assert_eq!(engine.get_api_key(Provider::OpenAI), Some("sk-openai".to_string()));

        // Verify default models
        assert_eq!(
            engine.get_default_model(Provider::Anthropic),
            Some("claude-sonnet-4-5-20250929".to_string())
        );
        assert_eq!(
            engine.get_default_model(Provider::OpenAI),
            Some("gpt-4o".to_string())
        );

        // Set default provider
        engine.set_default_provider(Provider::Gemini);
        assert_eq!(engine.get_default_provider(), Provider::Gemini);
    }

    #[test]
    fn test_unregister_nonexistent_agent() {
        let engine = MuxEngine::new("/tmp/mux-test-unreg".to_string()).unwrap();
        let result = engine.unregister_agent("nonexistent".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_model_context_config() {
        let engine = MuxEngine::new("/tmp/mux-test-context".to_string()).unwrap();

        // Default config
        let default_config = engine.get_model_context_config("unknown-model".to_string());
        assert_eq!(default_config.context_limit, 0);
        assert_eq!(default_config.compaction_mode, CompactionMode::Summarize);

        // Set custom config
        let config = ModelContextConfig {
            model: "foundation-3b".to_string(),
            context_limit: 4096,
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.8,
        };
        engine.set_model_context_config(config.clone());

        let retrieved = engine.get_model_context_config("foundation-3b".to_string());
        assert_eq!(retrieved.context_limit, 4096);
        assert_eq!(retrieved.compaction_mode, CompactionMode::TruncateOldest);
    }

    #[test]
    fn test_clear_context() {
        let engine = MuxEngine::new("/tmp/mux-test-clear".to_string()).unwrap();
        let ws = engine.create_workspace("Test".to_string(), None).unwrap();
        let conv = engine.create_conversation(ws.id.clone(), "Test Conv".to_string()).unwrap();

        // Clear should work on empty conversation
        engine.clear_context(conv.id.clone()).unwrap();

        // Clear nonexistent should fail
        let result = engine.clear_context("nonexistent".to_string());
        assert!(result.is_err());

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_context_workflow() {
        let engine = MuxEngine::new("/tmp/mux-test-context-flow".to_string()).unwrap();

        // Create workspace with model config
        let ws = engine.create_workspace("Context Test".to_string(), None).unwrap();

        // Configure small context model
        engine.set_model_context_config(ModelContextConfig {
            model: "test-small-model".to_string(),
            context_limit: 100, // Very small for testing
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.5,
        });

        // Create conversation
        let conv = engine
            .create_conversation(ws.id.clone(), "Context Test".to_string())
            .unwrap();

        // Check initial usage
        let usage = engine.get_context_usage(conv.id.clone()).unwrap();
        assert_eq!(usage.message_count, 0);
        assert_eq!(usage.estimated_tokens, 0);

        // Clear context works
        engine.clear_context(conv.id.clone()).unwrap();

        // Compact context works (no-op on empty)
        let usage = engine.compact_context(conv.id.clone()).unwrap();
        assert_eq!(usage.message_count, 0);

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }
}
