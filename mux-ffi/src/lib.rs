// ABOUTME: UniFFI bindings for mux - exposes Rust agentic library to Swift/Kotlin.
// ABOUTME: This crate provides the BuddyEngine interface for GUI applications.

uniffi::setup_scaffolding!();

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
}
