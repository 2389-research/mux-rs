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
            compaction_model: None,
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
            compaction_model: None,
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

    #[test]
    fn test_context_usage_with_messages() {
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-ctx-msg".to_string()).unwrap();
        let ws = engine.create_workspace("Test".to_string(), None).unwrap();
        let conv = engine.create_conversation(ws.id.clone(), "Test Conv".to_string()).unwrap();

        // Inject some messages
        engine.inject_test_message(&conv.id, Role::User, "Hello, how are you?");
        engine.inject_test_message(&conv.id, Role::Assistant, "I'm doing well, thanks for asking!");

        // Check usage reflects messages
        let usage = engine.get_context_usage(conv.id.clone()).unwrap();
        assert_eq!(usage.message_count, 2);
        assert!(usage.estimated_tokens > 0);

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_truncate_oldest_via_compact() {
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-truncate".to_string()).unwrap();
        let ws = engine.create_workspace("Truncate Test".to_string(), None).unwrap();

        // Set workspace LLM config so model is known
        engine.set_workspace_llm_config(&ws.id, "truncate-test-model");

        // Configure very small context limit
        engine.set_model_context_config(ModelContextConfig {
            model: "truncate-test-model".to_string(),
            context_limit: 50, // Very small - forces truncation
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.5,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();

        // Add many messages to exceed limit
        for i in 0..10 {
            engine.inject_test_message(
                &conv.id,
                Role::User,
                &format!("This is message number {} with some extra text to use tokens", i),
            );
        }

        let before_count = engine.get_message_count(&conv.id);
        assert_eq!(before_count, 10);

        // Compact should truncate oldest
        let usage = engine.compact_context(conv.id.clone()).unwrap();

        let after_count = engine.get_message_count(&conv.id);
        assert!(after_count < before_count, "Messages should be truncated");
        assert!(usage.estimated_tokens <= 40, "Should be under effective limit (50 * 0.8)");

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_get_workspace_for_conversation() {
        let engine = MuxEngine::new("/tmp/mux-test-ws-conv".to_string()).unwrap();

        let ws1 = engine.create_workspace("Workspace 1".to_string(), None).unwrap();
        let ws2 = engine.create_workspace("Workspace 2".to_string(), None).unwrap();

        let conv1 = engine.create_conversation(ws1.id.clone(), "Conv 1".to_string()).unwrap();
        let conv2 = engine.create_conversation(ws2.id.clone(), "Conv 2".to_string()).unwrap();

        // Verify correct workspace mapping
        let usage1 = engine.get_context_usage(conv1.id.clone()).unwrap();
        let usage2 = engine.get_context_usage(conv2.id.clone()).unwrap();
        assert_eq!(usage1.message_count, 0);
        assert_eq!(usage2.message_count, 0);

        // Nonexistent conversation returns error
        let result = engine.get_context_usage("nonexistent".to_string());
        assert!(result.is_err());

        // Cleanup
        engine.delete_workspace(ws1.id).unwrap();
        engine.delete_workspace(ws2.id).unwrap();
    }

    #[test]
    fn test_auto_compaction_strategy_threshold() {
        use crate::context::SMALL_CONTEXT_THRESHOLD;

        // Verify the threshold is 8K tokens
        assert_eq!(SMALL_CONTEXT_THRESHOLD, 8192);
    }

    #[test]
    fn test_auto_compaction_small_context_uses_truncation() {
        // Small context models (<=8K) should use truncation, not summarization
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-auto-small".to_string()).unwrap();
        let ws = engine.create_workspace("Auto Small Test".to_string(), None).unwrap();

        engine.set_workspace_llm_config(&ws.id, "small-model");

        // Configure model with context_limit <= SMALL_CONTEXT_THRESHOLD (8192)
        // This should auto-select truncation
        engine.set_model_context_config(ModelContextConfig {
            model: "small-model".to_string(),
            context_limit: 4096, // Small context - will use truncation
            compaction_mode: CompactionMode::Summarize, // This is ignored - auto-selects based on limit
            warning_threshold: 0.8,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();

        // Add messages
        for i in 0..5 {
            engine.inject_test_message(
                &conv.id,
                Role::User,
                &format!("Message {} with content to consume tokens", i),
            );
        }

        let before = engine.get_message_count(&conv.id);

        // Compact should succeed with truncation (no LLM call needed)
        let result = engine.compact_context(conv.id.clone());
        assert!(result.is_ok(), "Small context compaction should succeed via truncation");

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_auto_compaction_large_context_needs_api_key() {
        // Large context models (>8K) should attempt summarization
        // Without an API key, this should fail with a clear error
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-auto-large".to_string()).unwrap();
        let ws = engine.create_workspace("Auto Large Test".to_string(), None).unwrap();

        engine.set_workspace_llm_config(&ws.id, "large-model");

        // Configure model with context_limit > SMALL_CONTEXT_THRESHOLD (8192)
        // This should auto-select summarization
        engine.set_model_context_config(ModelContextConfig {
            model: "large-model".to_string(),
            context_limit: 100000, // Large context - will attempt summarization
            compaction_mode: CompactionMode::TruncateOldest, // This is ignored - auto-selects based on limit
            warning_threshold: 0.8,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();

        // Add a message
        engine.inject_test_message(
            &conv.id,
            Role::User,
            "Test message",
        );

        // Compact should fail because no API key is set for summarization
        let result = engine.compact_context(conv.id.clone());
        assert!(result.is_err(), "Large context compaction should fail without API key");
        let error = result.unwrap_err().to_string();
        assert!(error.contains("API key") || error.contains("summarization"),
            "Error should mention API key or summarization: {}", error);

        // Cleanup
        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_auto_compaction_boundary_at_threshold() {
        // Test the exact boundary: 8192 should use truncation, 8193 should use summarization
        use crate::context::SMALL_CONTEXT_THRESHOLD;
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-boundary".to_string()).unwrap();
        let ws = engine.create_workspace("Boundary Test".to_string(), None).unwrap();

        // Test at exactly threshold (should use truncation - succeeds without API key)
        engine.set_workspace_llm_config(&ws.id, "boundary-model");
        engine.set_model_context_config(ModelContextConfig {
            model: "boundary-model".to_string(),
            context_limit: SMALL_CONTEXT_THRESHOLD, // Exactly 8192
            compaction_mode: CompactionMode::Summarize,
            warning_threshold: 0.8,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();
        engine.inject_test_message(&conv.id, Role::User, "Test");

        // At threshold, should use truncation (no API key needed)
        let result = engine.compact_context(conv.id.clone());
        assert!(result.is_ok(), "At threshold (8192) should use truncation and succeed");

        // Test just above threshold (should try summarization - fails without API key)
        engine.set_model_context_config(ModelContextConfig {
            model: "boundary-model".to_string(),
            context_limit: SMALL_CONTEXT_THRESHOLD + 1, // 8193
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.8,
            compaction_model: None,
        });

        // Above threshold, should try summarization (needs API key)
        let result2 = engine.compact_context(conv.id.clone());
        assert!(result2.is_err(), "Above threshold (8193) should try summarization and fail without API key");

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_truncation_with_tiny_target() {
        // Edge case: target_tokens smaller than any single message
        use mux::prelude::Role;

        let engine = MuxEngine::new("/tmp/mux-test-tiny".to_string()).unwrap();
        let ws = engine.create_workspace("Tiny Test".to_string(), None).unwrap();

        engine.set_workspace_llm_config(&ws.id, "tiny-model");

        // Context limit of 10 tokens (effective = 8 after 0.8 safety margin)
        engine.set_model_context_config(ModelContextConfig {
            model: "tiny-model".to_string(),
            context_limit: 10,
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.8,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();

        // Add a message that's larger than the effective limit (8 tokens)
        // "This message is definitely longer than eight tokens" = ~12 tokens
        engine.inject_test_message(
            &conv.id,
            Role::User,
            "This message is definitely longer than eight tokens of text",
        );

        let before = engine.get_message_count(&conv.id);
        assert_eq!(before, 1);

        // Compact - the message won't fit within the tiny target
        let result = engine.compact_context(conv.id.clone());
        assert!(result.is_ok());

        // With truncation, if no messages fit within target, ALL are removed.
        // This is by design - truncation is meant for small context models where
        // keeping oversized messages defeats the purpose. For large context models,
        // summarization would be used instead (which compresses rather than removes).
        let after = engine.get_message_count(&conv.id);
        assert_eq!(after, 0, "Truncation removes all messages when none fit within target");

        engine.delete_workspace(ws.id).unwrap();
    }

    #[test]
    fn test_compaction_on_empty_conversation() {
        // Edge case: compact an empty conversation
        let engine = MuxEngine::new("/tmp/mux-test-empty".to_string()).unwrap();
        let ws = engine.create_workspace("Empty Test".to_string(), None).unwrap();

        engine.set_workspace_llm_config(&ws.id, "empty-model");
        engine.set_model_context_config(ModelContextConfig {
            model: "empty-model".to_string(),
            context_limit: 4096,
            compaction_mode: CompactionMode::TruncateOldest,
            warning_threshold: 0.8,
            compaction_model: None,
        });

        let conv = engine.create_conversation(ws.id.clone(), "Test".to_string()).unwrap();

        // Compact empty conversation should succeed (no-op)
        let result = engine.compact_context(conv.id.clone());
        assert!(result.is_ok(), "Compacting empty conversation should succeed");

        let usage = result.unwrap();
        assert_eq!(usage.message_count, 0);
        assert_eq!(usage.estimated_tokens, 0);

        engine.delete_workspace(ws.id).unwrap();
    }
}
