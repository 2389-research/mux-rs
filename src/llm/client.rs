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
    /// Message creation started.
    MessageStart { id: String, model: String },

    /// A content block started.
    ContentBlockStart {
        index: usize,
        block: super::ContentBlock,
    },

    /// Delta for a content block (usually text).
    ContentBlockDelta { index: usize, text: String },

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
}
