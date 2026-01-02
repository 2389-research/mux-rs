// ABOUTME: UniFFI bindings for mux - exposes Rust agentic library to Swift/Kotlin.
// ABOUTME: This crate provides the MuxEngine interface for GUI applications.

uniffi::setup_scaffolding!();

mod types;
pub use types::*;

mod engine;
pub use engine::*;

mod callback;
pub use callback::*;

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
}
