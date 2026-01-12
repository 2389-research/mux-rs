// ABOUTME: OpenRouter API client wrapping OpenAI-compatible API.
// ABOUTME: Supports custom HTTP-Referer and X-Title headers for app identification.

use super::client::StreamEvent;
use super::openai::{parse_sse_line, OpenAIError, OpenAIRequest, OpenAIResponse};
use super::{ContentBlock, Request, Response, StopReason, Usage};
use crate::error::LlmError;
use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue};
use std::pin::Pin;

/// Base URL for OpenRouter's OpenAI-compatible API.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Default model when none is specified.
pub const OPENROUTER_DEFAULT_MODEL: &str = "anthropic/claude-3.5-sonnet";

/// Client for OpenRouter API.
/// OpenRouter provides a unified API that routes to various LLM providers.
#[derive(Debug, Clone)]
pub struct OpenRouterClient {
    api_key: String,
    http: reqwest::Client,
    default_model: String,
}

impl OpenRouterClient {
    /// Create a new OpenRouter client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_headers(api_key, None, None)
    }

    /// Create a new OpenRouter client from the OPENROUTER_API_KEY environment variable.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "OPENROUTER_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::new(api_key))
    }

    /// Create a new OpenRouter client with custom headers for app identification.
    ///
    /// # Arguments
    /// * `api_key` - OpenRouter API key
    /// * `referer` - HTTP-Referer header (your app's URL, helps OpenRouter track usage)
    /// * `title` - X-Title header (your app's name, displayed in OpenRouter dashboard)
    pub fn with_headers(
        api_key: impl Into<String>,
        referer: Option<&str>,
        title: Option<&str>,
    ) -> Self {
        let mut headers = HeaderMap::new();

        if let Some(referer) = referer {
            if let Ok(value) = HeaderValue::from_str(referer) {
                headers.insert("HTTP-Referer", value);
            }
        }

        if let Some(title) = title {
            if let Ok(value) = HeaderValue::from_str(title) {
                headers.insert("X-Title", value);
            }
        }

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            api_key: api_key.into(),
            http,
            default_model: OPENROUTER_DEFAULT_MODEL.to_string(),
        }
    }

    /// Set the default model to use when none is specified in the request.
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
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
impl super::client::LlmClient for OpenRouterClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let mut openai_req = OpenAIRequest::from(req);

        // Use default model if none specified
        if openai_req.model.is_empty() {
            openai_req.model = self.default_model.clone();
        }

        let url = format!("{}/chat/completions", OPENROUTER_BASE_URL);

        let response = self
            .http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
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

        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let url = format!("{}/chat/completions", OPENROUTER_BASE_URL);
            let response = http
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
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
mod openrouter_test {
    use super::*;

    #[test]
    fn test_client_from_env_missing() {
        // SAFETY: This test runs in isolation and only affects this process
        unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        }
        let result = OpenRouterClient::from_env();
        assert!(result.is_err());
    }

    #[test]
    fn test_client_new() {
        let client = OpenRouterClient::new("test-key");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.default_model, OPENROUTER_DEFAULT_MODEL);
    }

    #[test]
    fn test_client_with_headers() {
        let client =
            OpenRouterClient::with_headers("test-key", Some("https://myapp.com"), Some("MyApp"));
        assert_eq!(client.api_key, "test-key");
        // Headers are set on the http client internally
    }

    #[test]
    fn test_client_with_default_model() {
        let client =
            OpenRouterClient::new("test-key").with_default_model("openai/gpt-4-turbo");
        assert_eq!(client.default_model, "openai/gpt-4-turbo");
    }

    #[test]
    fn test_constants() {
        assert_eq!(OPENROUTER_BASE_URL, "https://openrouter.ai/api/v1");
        assert_eq!(OPENROUTER_DEFAULT_MODEL, "anthropic/claude-3.5-sonnet");
    }
}
