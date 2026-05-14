// ABOUTME: Defines the LlmClient trait - the abstraction layer that allows
// ABOUTME: mux to work with any LLM provider (Anthropic, OpenAI, etc.)

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use super::{Request, Response};
use crate::error::LlmError;

/// Event types for streaming responses.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// Message creation started. Anthropic includes initial usage with
    /// `input_tokens` + cache fields here (and `output_tokens: 0`); the
    /// `output_tokens` get updated in subsequent `MessageDelta` events.
    /// Providers that don't surface initial usage emit a default Usage.
    MessageStart {
        id: String,
        model: String,
        usage: super::Usage,
    },

    /// A content block started.
    ContentBlockStart {
        index: usize,
        block: super::ContentBlock,
    },

    /// Delta for a content block (text content only).
    /// Text deltas should be concatenated to build the complete text.
    ContentBlockDelta { index: usize, text: String },

    /// Delta for tool input JSON arguments.
    /// These arrive after `ContentBlockStart` for a `ToolUse` block.
    /// Accumulate `partial_json` values and parse as JSON at `ContentBlockStop`.
    ///
    /// Event order for tool calls:
    /// 1. `ContentBlockStart` with `ToolUse { id, name, input: {} }`
    /// 2. Zero or more `InputJsonDelta` with partial JSON fragments
    /// 3. `ContentBlockStop`
    ///
    /// The `index` matches the `ContentBlockStart` index for the tool block.
    InputJsonDelta { index: usize, partial_json: String },

    /// A content block finished.
    ContentBlockStop { index: usize },

    /// Message metadata update.
    MessageDelta {
        stop_reason: Option<super::StopReason>,
        usage: super::Usage,
    },

    /// Message complete.
    MessageStop,
}

/// Trait for LLM client implementations.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Create a message (non-streaming).
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError>;

    /// Create a message with streaming response.
    fn create_message_stream(
        &self,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>>;

    /// Does this client support the given media kind as user input?
    ///
    /// Default is `false`; providers override to declare their capability set.
    /// Frontends should call this to hide/disable attachment UI for unsupported kinds.
    fn supports_media(&self, _kind: super::MediaKind) -> bool {
        false
    }
}

#[cfg(test)]
mod client_test {
    use super::*;
    use crate::llm::MediaKind;

    struct DefaultClient;

    #[async_trait]
    impl LlmClient for DefaultClient {
        async fn create_message(&self, _: &Request) -> Result<Response, crate::error::LlmError> {
            unimplemented!()
        }
        fn create_message_stream(
            &self,
            _: &Request,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, crate::error::LlmError>> + Send + 'static>>
        {
            unimplemented!()
        }
    }

    #[test]
    fn default_supports_media_is_false() {
        let c = DefaultClient;
        assert!(!c.supports_media(MediaKind::Image));
        assert!(!c.supports_media(MediaKind::Document));
        assert!(!c.supports_media(MediaKind::Audio));
        assert!(!c.supports_media(MediaKind::Video));
    }
}
