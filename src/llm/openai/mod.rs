// ABOUTME: OpenAI provider client and LlmClient implementation.
// ABOUTME: Wire types live in types.rs, request building in convert.rs, parsing in response.rs.
mod convert;
mod response;
mod types;

pub use convert::*;
pub use response::*;
pub use types::*;

use super::client::StreamEvent;
use super::media::resolve_request_media;
use crate::error::LlmError;
use crate::llm::{ContentBlock, Request, Response, Usage};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Client for OpenAI and OpenAI-compatible APIs (OpenRouter, Ollama, etc.).
#[derive(Debug, Clone)]
pub struct OpenAIClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl OpenAIClient {
    /// Create a new OpenAI client with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_DEFAULT_BASE_URL.to_string(),
            http: reqwest::Client::new(),
        }
    }

    /// Create a new OpenAI client from the OPENAI_API_KEY environment variable.
    pub fn from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "OPENAI_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::new(api_key))
    }

    /// Override the base URL for OpenAI-compatible APIs.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Create an OpenRouter client with the given API key.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::new(api_key).with_base_url("https://openrouter.ai/api/v1")
    }

    /// Create an OpenRouter client from the OPENROUTER_API_KEY environment variable.
    pub fn openrouter_from_env() -> Result<Self, LlmError> {
        let api_key = std::env::var("OPENROUTER_API_KEY").map_err(|_| LlmError::Api {
            status: 0,
            message: "OPENROUTER_API_KEY environment variable not set".to_string(),
        })?;
        Ok(Self::openrouter(api_key))
    }

    /// Create an Ollama client connecting to localhost:11434.
    pub fn ollama() -> Self {
        Self::new("ollama").with_base_url("http://localhost:11434/v1")
    }

    /// Create an Ollama client connecting to a custom host.
    pub fn ollama_at(host: impl Into<String>) -> Self {
        Self::new("ollama").with_base_url(format!("{}/v1", host.into()))
    }
}

#[async_trait]
impl super::client::LlmClient for OpenAIClient {
    async fn create_message(&self, req: &Request) -> Result<Response, LlmError> {
        let resolved = resolve_request_media(req, &self.http).await?;
        let openai_req = try_into_openai_request(&resolved)?;
        let url = format!("{}/chat/completions", self.base_url);

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
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let http = self.http.clone();
        let req = req.clone();

        Box::pin(async_stream::try_stream! {
            let resolved = resolve_request_media(&req, &http).await?;
            let mut openai_req = try_into_openai_request(&resolved)?;
            openai_req.stream = Some(true);

            let url = format!("{}/chat/completions", base_url);
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
                                usage: crate::llm::Usage::default(),
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
                                    stop_reason: Some(response::parse_stop_reason(Some(&reason))),
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

    fn supports_media(&self, kind: super::MediaKind) -> bool {
        use super::MediaKind;
        matches!(
            kind,
            MediaKind::Image | MediaKind::Document | MediaKind::Audio
        )
    }
}

#[cfg(test)]
mod openai_test;
