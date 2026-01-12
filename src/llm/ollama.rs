// ABOUTME: Ollama API client wrapping OpenAI-compatible API for local LLM inference.
// ABOUTME: Connects to Ollama server (default localhost:11434) with dummy API key.

use super::client::StreamEvent;
use super::openai::{parse_sse_line, OpenAIError, OpenAIRequest, OpenAIResponse};
use super::{ContentBlock, Request, Response, StopReason, Usage};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Base URL for Ollama's OpenAI-compatible API.
pub const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";

/// Default model when none is specified.
pub const OLLAMA_DEFAULT_MODEL: &str = "llama3.2";

/// Client for Ollama API.
/// Ollama runs LLMs locally and exposes an OpenAI-compatible API.
#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    http: reqwest::Client,
    default_model: String,
}

impl OllamaClient {
    /// Create a new Ollama client connecting to localhost:11434.
    pub fn new(model: &str) -> Self {
        Self {
            base_url: OLLAMA_BASE_URL.to_string(),
            http: reqwest::Client::new(),
            default_model: if model.is_empty() {
                OLLAMA_DEFAULT_MODEL.to_string()
            } else {
                model.to_string()
            },
        }
    }

    /// Create a new Ollama client with a custom base URL.
    ///
    /// # Arguments
    /// * `base_url` - The base URL of the Ollama server (e.g., "http://remote-server:11434/v1")
    /// * `model` - The default model to use (e.g., "llama3.2", "mistral", "codellama")
    pub fn with_base_url(base_url: &str, model: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            http: reqwest::Client::new(),
            default_model: if model.is_empty() {
                OLLAMA_DEFAULT_MODEL.to_string()
            } else {
                model.to_string()
            },
        }
    }

    /// Set the default model to use when none is specified in the request.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }
}

impl Default for OllamaClient {
    fn default() -> Self {
        Self::new(OLLAMA_DEFAULT_MODEL)
    }
}

fn parse_stop_reason(s: Option<&str>) -> StopReason {
    match s {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

#[async_trait]
impl super::client::LlmClient for OllamaClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let mut openai_req = OpenAIRequest::from(req);

        // Use default model if none specified
        if openai_req.model.is_empty() {
            openai_req.model = self.default_model.clone();
        }

        let url = format!("{}/chat/completions", self.base_url);

        // Ollama ignores API key but HTTP client might require it
        let response = self
            .http
            .post(&url)
            .header("Authorization", "Bearer ollama")
            .header("Content-Type", "application/json")
            .json(&openai_req)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error: OpenAIError = response.json().await?;
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: error.error.message,
            });
        }

        let openai_resp: OpenAIResponse = response.json().await?;
        Ok(Response::from(openai_resp))
    }

    fn create_message_stream(
        &self,
        req: &Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send + 'static>> {
        let mut openai_req = OpenAIRequest::from(req);

        // Use default model if none specified
        if openai_req.model.is_empty() {
            openai_req.model = self.default_model.clone();
        }

        openai_req.stream = Some(true);

        let base_url = self.base_url.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let url = format!("{}/chat/completions", base_url);
            let response = http
                .post(&url)
                .header("Authorization", "Bearer ollama")
                .header("Content-Type", "application/json")
                .json(&openai_req)
                .send()
                .await?;

            let status = response.status();
            if !status.is_success() {
                let error_text = response.text().await?;
                let error: OpenAIError = serde_json::from_str(&error_text)?;
                Err(LlmError::Api {
                    status: status.as_u16(),
                    message: error.error.message,
                })?;
                return;
            }

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut message_started = false;
            let mut text_block_index: Option<usize> = None;
            let mut next_block_index = 0usize;
            // Track tool calls: (id, name, args, block_index, block_started)
            let mut current_tool_calls: Vec<(String, String, String, usize, bool)> = Vec::new();

            while let Some(chunk) = futures::StreamExt::next(&mut stream).await {
                let chunk = chunk?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }

                    if let Some(chunk) = parse_sse_line(&line) {
                        if !message_started {
                            yield StreamEvent::MessageStart {
                                id: chunk.id.clone(),
                                model: chunk.model.clone(),
                            };
                            message_started = true;
                        }

                        for choice in chunk.choices {
                            // Handle text content
                            if let Some(text) = choice.delta.content {
                                // Emit ContentBlockStart for text on first text delta
                                if text_block_index.is_none() {
                                    let idx = next_block_index;
                                    next_block_index += 1;
                                    yield StreamEvent::ContentBlockStart {
                                        index: idx,
                                        block: ContentBlock::Text { text: String::new() },
                                    };
                                    text_block_index = Some(idx);
                                }
                                yield StreamEvent::ContentBlockDelta {
                                    index: text_block_index.unwrap(),
                                    text,
                                };
                            }

                            // Handle tool calls
                            if let Some(tool_calls) = choice.delta.tool_calls {
                                for tc in tool_calls {
                                    let tc_idx = tc.index;

                                    // Ensure we have space for this tool call
                                    while current_tool_calls.len() <= tc_idx {
                                        current_tool_calls.push((String::new(), String::new(), String::new(), 0, false));
                                    }

                                    // Accumulate tool call data
                                    if let Some(id) = tc.id {
                                        current_tool_calls[tc_idx].0 = id;
                                    }
                                    if let Some(func) = tc.function {
                                        if let Some(name) = func.name {
                                            current_tool_calls[tc_idx].1 = name;
                                        }

                                        // Emit ContentBlockStart when we have id and name (before JSON deltas)
                                        let (ref id, ref name, _, ref mut block_idx, ref mut started) = current_tool_calls[tc_idx];
                                        if !*started && !id.is_empty() && !name.is_empty() {
                                            *block_idx = next_block_index;
                                            next_block_index += 1;
                                            yield StreamEvent::ContentBlockStart {
                                                index: *block_idx,
                                                block: ContentBlock::ToolUse {
                                                    id: id.clone(),
                                                    name: name.clone(),
                                                    input: serde_json::Value::Object(serde_json::Map::new()),
                                                },
                                            };
                                            *started = true;
                                        }

                                        if let Some(args) = func.arguments {
                                            current_tool_calls[tc_idx].2.push_str(&args);
                                            // Yield as input JSON delta for tool argument accumulation
                                            yield StreamEvent::InputJsonDelta {
                                                index: current_tool_calls[tc_idx].3,
                                                partial_json: args,
                                            };
                                        }
                                    }
                                }
                            }

                            // Handle finish reason
                            if let Some(reason) = choice.finish_reason {
                                // Close text block if started
                                if let Some(idx) = text_block_index {
                                    yield StreamEvent::ContentBlockStop { index: idx };
                                }

                                // Close tool call blocks
                                for (id, _, _, block_idx, started) in current_tool_calls.iter() {
                                    if *started && !id.is_empty() {
                                        yield StreamEvent::ContentBlockStop { index: *block_idx };
                                    }
                                }

                                yield StreamEvent::MessageDelta {
                                    stop_reason: Some(parse_stop_reason(Some(&reason))),
                                    usage: Usage::default(),
                                };
                                yield StreamEvent::MessageStop;
                            }
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod ollama_test {
    use super::*;

    #[test]
    fn test_client_new() {
        let client = OllamaClient::new("llama3.2");
        assert_eq!(client.base_url, OLLAMA_BASE_URL);
        assert_eq!(client.default_model, "llama3.2");
    }

    #[test]
    fn test_client_new_empty_model() {
        let client = OllamaClient::new("");
        assert_eq!(client.default_model, OLLAMA_DEFAULT_MODEL);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = OllamaClient::with_base_url("http://remote:11434/v1", "mistral");
        assert_eq!(client.base_url, "http://remote:11434/v1");
        assert_eq!(client.default_model, "mistral");
    }

    #[test]
    fn test_client_with_base_url_empty_model() {
        let client = OllamaClient::with_base_url("http://remote:11434/v1", "");
        assert_eq!(client.default_model, OLLAMA_DEFAULT_MODEL);
    }

    #[test]
    fn test_client_with_default_model() {
        let client = OllamaClient::new("llama3.2").with_default_model("codellama");
        assert_eq!(client.default_model, "codellama");
    }

    #[test]
    fn test_client_default() {
        let client = OllamaClient::default();
        assert_eq!(client.base_url, OLLAMA_BASE_URL);
        assert_eq!(client.default_model, OLLAMA_DEFAULT_MODEL);
    }

    #[test]
    fn test_constants() {
        assert_eq!(OLLAMA_BASE_URL, "http://localhost:11434/v1");
        assert_eq!(OLLAMA_DEFAULT_MODEL, "llama3.2");
    }
}
