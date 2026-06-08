// ABOUTME: Response parsing for the OpenAI Chat Completions API.
// ABOUTME: Converts OpenAIResponse into crate Response and parses SSE streaming lines.

use super::types::{
    OpenAIChoice, OpenAIResponse, OpenAIResponseMessage, OpenAIStreamChunk, OpenAIUsage,
};
use crate::llm::{ContentBlock, Response, StopReason, Usage};

pub(super) fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

impl From<OpenAIResponse> for Response {
    fn from(resp: OpenAIResponse) -> Self {
        let choice = resp.choices.into_iter().next().unwrap_or(OpenAIChoice {
            index: 0,
            message: OpenAIResponseMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: None,
            },
            finish_reason: None,
        });

        let mut content = Vec::new();

        // Add text content if present
        if let Some(text) = choice.message.content
            && !text.is_empty()
        {
            content.push(ContentBlock::Text { text });
        }

        // Add tool calls if present
        if let Some(tool_calls) = choice.message.tool_calls {
            for call in tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&call.function.arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse {
                    id: call.id,
                    name: call.function.name,
                    input,
                });
            }
        }

        let usage = resp.usage.unwrap_or(OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

        Response {
            id: resp.id,
            content,
            stop_reason: parse_stop_reason(choice.finish_reason.as_deref()),
            model: resp.model,
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                ..Default::default()
            },
        }
    }
}

/// Parse an SSE line into an OpenAI stream chunk.
/// Used internally by OpenAI-compatible clients.
pub fn parse_sse_line(line: &str) -> Option<OpenAIStreamChunk> {
    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None;
    }
    serde_json::from_str(data).ok()
}
