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
}
