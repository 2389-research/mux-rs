// ABOUTME: Adapts Swift's LlmProvider callback to Rust's LlmClient trait.
// ABOUTME: Enables on-device models to integrate with Mux orchestration.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use mux::error::LlmError;
use mux::llm::{ContentBlock, LlmClient, Request, Response, StreamEvent, StopReason, Usage};

use crate::callback::LlmProvider;
use crate::types::{ChatMessage, ChatRole, FfiToolDefinition, LlmRequest};

/// Adapter that wraps a Swift-provided LlmProvider as a Rust LlmClient.
pub struct CallbackLlmClient {
    provider: Arc<Box<dyn LlmProvider>>,
}

impl CallbackLlmClient {
    pub fn new(provider: Box<dyn LlmProvider>) -> Self {
        Self {
            provider: Arc::new(provider),
        }
    }

    /// Convert mux Request to FFI LlmRequest
    fn convert_request(req: &Request) -> LlmRequest {
        let messages: Vec<ChatMessage> = req
            .messages
            .iter()
            .map(|m| {
                // Collect text content from message blocks
                let content: String = m
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                ChatMessage {
                    role: match m.role {
                        mux::llm::Role::User => ChatRole::User,
                        mux::llm::Role::Assistant => ChatRole::Assistant,
                    },
                    content,
                }
            })
            .collect();

        let tools: Vec<FfiToolDefinition> = req
            .tools
            .iter()
            .map(|t| FfiToolDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema_json: serde_json::to_string(&t.input_schema).unwrap_or_default(),
            })
            .collect();

        LlmRequest {
            messages,
            tools,
            system_prompt: req.system.clone(),
            max_tokens: req.max_tokens,
        }
    }
}

#[async_trait]
impl LlmClient for CallbackLlmClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let llm_request = Self::convert_request(req);

        // Call Swift provider (blocking call)
        let provider = self.provider.clone();
        let llm_response = tokio::task::spawn_blocking(move || provider.generate(llm_request))
            .await
            .map_err(|e| LlmError::Api {
                status: 0,
                message: format!("Provider task failed: {}", e),
            })?;

        // Check for error
        if let Some(error) = llm_response.error {
            return Err(LlmError::Api {
                status: 0,
                message: error,
            });
        }

        // Build content blocks
        let mut content: Vec<ContentBlock> = Vec::new();

        if !llm_response.text.is_empty() {
            content.push(ContentBlock::Text {
                text: llm_response.text,
            });
        }

        for tool_call in llm_response.tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&tool_call.arguments).unwrap_or_default();
            content.push(ContentBlock::ToolUse {
                id: tool_call.id,
                name: tool_call.name,
                input,
            });
        }

        // Determine stop reason
        let stop_reason = if content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        Ok(Response {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            stop_reason,
            model: "callback-provider".to_string(),
            usage: Usage {
                input_tokens: llm_response.usage.input_tokens,
                output_tokens: llm_response.usage.output_tokens,
            },
        })
    }

    fn create_message_stream(
        &self,
        _req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        // Streaming not supported for callback providers - return empty stream
        // The blocking generate() call is used instead
        Box::pin(futures::stream::empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LlmResponse, LlmToolCall, LlmUsage};
    use mux::llm::Message;

    struct EchoProvider;

    impl LlmProvider for EchoProvider {
        fn generate(&self, request: LlmRequest) -> LlmResponse {
            // Echo back the last message
            let response = request
                .messages
                .last()
                .map(|m| format!("Echo: {}", m.content))
                .unwrap_or_else(|| "No message".to_string());

            LlmResponse {
                text: response,
                tool_calls: Vec::new(),
                usage: LlmUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                error: None,
            }
        }
    }

    struct ToolUseProvider;

    impl LlmProvider for ToolUseProvider {
        fn generate(&self, _request: LlmRequest) -> LlmResponse {
            LlmResponse {
                text: String::new(),
                tool_calls: vec![LlmToolCall {
                    id: "call_123".to_string(),
                    name: "read_file".to_string(),
                    arguments: r#"{"path": "/tmp/test.txt"}"#.to_string(),
                }],
                usage: LlmUsage::default(),
                error: None,
            }
        }
    }

    #[tokio::test]
    async fn test_callback_client_echo() {
        let client = CallbackLlmClient::new(Box::new(EchoProvider));
        let request = Request::new("test-model").message(Message::user("Hello"));

        let response = client.create_message(&request).await.unwrap();

        assert_eq!(response.text(), "Echo: Hello");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn test_callback_client_tool_use() {
        let client = CallbackLlmClient::new(Box::new(ToolUseProvider));
        let request = Request::new("test-model").message(Message::user("Read a file"));

        let response = client.create_message(&request).await.unwrap();

        assert!(response.has_tool_use());
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        let tool_uses = response.tool_uses();
        assert_eq!(tool_uses.len(), 1);
    }
}
