// ABOUTME: UniFFI bindings for mux - exposes Rust agentic library to Swift/Kotlin.
// ABOUTME: This crate provides the BuddyEngine interface for GUI applications.

uniffi::setup_scaffolding!();

mod types;
pub use types::*;

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MuxFfiError {
    #[error("Engine error: {message}")]
    Engine { message: String },
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
}
