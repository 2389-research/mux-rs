// ABOUTME: Request building functions that convert crate domain types into OpenAI wire types.
// ABOUTME: Includes From<&ToolDefinition>, try_openai_messages, and try_into_openai_request.

use super::types::{
    OpenAIContent, OpenAIContentPart, OpenAIFile, OpenAIFunction, OpenAIFunctionCall,
    OpenAIImageUrl, OpenAIInputAudio, OpenAIMessage, OpenAIRequest, OpenAITool, OpenAIToolCall,
};
use crate::error::LlmError;
use crate::llm::{ContentBlock, MediaKind, MediaSource, Message, Request, Role, ToolDefinition};

impl From<&ToolDefinition> for OpenAITool {
    fn from(tool: &ToolDefinition) -> Self {
        OpenAITool {
            tool_type: "function".to_string(),
            function: OpenAIFunction {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        }
    }
}

fn try_openai_messages(messages: &[Message]) -> Result<Vec<OpenAIMessage>, LlmError> {
    let mut result = Vec::new();
    for msg in messages {
        // Split out tool-result blocks — each becomes its own `role: "tool"` message
        let tool_results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => Some((tool_use_id.clone(), content.clone())),
                _ => None,
            })
            .collect();

        if !tool_results.is_empty() {
            if msg.content.len() != tool_results.len() {
                return Err(LlmError::Configuration(
                    "ToolResult blocks cannot be mixed with text, media, or tool calls in the same message".into(),
                ));
            }
            for (tool_use_id, content) in tool_results {
                result.push(OpenAIMessage {
                    role: "tool".to_string(),
                    content: Some(OpenAIContent::String(content)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id),
                });
            }
            continue;
        }

        // For non-tool-result messages: build content (string or parts) + tool_calls
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
        .to_string();

        let has_media = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::Media { .. }));

        let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                tool_calls.push(OpenAIToolCall {
                    id: id.clone(),
                    call_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).map_err(|e| {
                            LlmError::Configuration(format!(
                                "failed to serialize tool '{}' input as JSON: {}",
                                name, e
                            ))
                        })?,
                    },
                });
            }
        }

        let content = if has_media {
            // Build parts array
            let mut parts: Vec<OpenAIContentPart> = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => {
                        parts.push(OpenAIContentPart::Text { text: text.clone() });
                    }
                    ContentBlock::Media {
                        kind,
                        source,
                        mime_type,
                    } => {
                        parts.push(try_media_part(*kind, source, mime_type)?);
                    }
                    // ToolUse is handled by the tool_calls field above; skip here
                    ContentBlock::ToolUse { .. } => {}
                    ContentBlock::ToolResult { .. } => {
                        // Shouldn't be reachable here (tool results are split above)
                        return Err(LlmError::Configuration(
                            "ToolResult mixed with non-tool-result content in same message".into(),
                        ));
                    }
                }
            }
            Some(OpenAIContent::Parts(parts))
        } else {
            // Collect text only
            let text: String = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            if text.is_empty() {
                None
            } else {
                Some(OpenAIContent::String(text))
            }
        };

        result.push(OpenAIMessage {
            role,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        });
    }
    Ok(result)
}

fn try_media_part(
    kind: MediaKind,
    source: &MediaSource,
    mime_type: &str,
) -> Result<OpenAIContentPart, LlmError> {
    match kind {
        MediaKind::Image => {
            let url = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "image_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    format!("data:{};base64,{}", mime_type, data)
                }
                MediaSource::Url(url) => url.clone(),
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved before serialization".into(),
                    ));
                }
            };
            Ok(OpenAIContentPart::ImageUrl {
                image_url: OpenAIImageUrl { url },
            })
        }
        MediaKind::Document => {
            // Chat Completions expects the file_data as a data URL
            // ("data:{mime};base64,{data}") — not plain base64.
            let data_url = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "document_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    format!("data:{};base64,{}", mime_type, data)
                }
                MediaSource::Url(_) => {
                    // OpenAI's file input does NOT accept URLs; only file_data (base64) or file_id.
                    // file_id belongs to the Files API — out of scope for this task.
                    return Err(LlmError::Configuration(
                        "openai file input requires base64 data; URL sources are not supported"
                            .into(),
                    ));
                }
                MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "MediaSource::Path must be resolved before serialization".into(),
                    ));
                }
            };
            Ok(OpenAIContentPart::File {
                file: OpenAIFile {
                    filename: None,
                    file_data: data_url,
                },
            })
        }
        MediaKind::Audio => {
            let data = match source {
                MediaSource::Base64(data) => {
                    if mime_type.is_empty() {
                        return Err(LlmError::Configuration(
                            "audio_base64 requires a non-empty mime_type".into(),
                        ));
                    }
                    data.clone()
                }
                MediaSource::Url(_) | MediaSource::Path(_) => {
                    return Err(LlmError::Configuration(
                        "openai input_audio requires base64 data (URL/Path must be resolved)"
                            .into(),
                    ));
                }
            };
            let format = audio_format_from_mime(mime_type)?;
            Ok(OpenAIContentPart::InputAudio {
                input_audio: OpenAIInputAudio { data, format },
            })
        }
        MediaKind::Video => Err(LlmError::UnsupportedMedia {
            provider: "openai",
            kind: MediaKind::Video,
        }),
    }
}

fn audio_format_from_mime(mime: &str) -> Result<String, LlmError> {
    let format = match mime {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        _ => {
            return Err(LlmError::Configuration(format!(
                "unsupported input_audio mime_type: '{}' (only audio/wav, audio/x-wav, audio/mpeg, audio/mp3 accepted)",
                mime
            )));
        }
    };
    Ok(format.to_string())
}

/// Check if a model requires max_completion_tokens instead of max_tokens.
fn uses_max_completion_tokens(model: &str) -> bool {
    let model_lower = model.to_lowercase();
    model_lower.starts_with("o1")
        || model_lower.starts_with("o3")
        || model_lower.starts_with("gpt-5")
}

pub fn try_into_openai_request(req: &Request) -> Result<OpenAIRequest, LlmError> {
    let mut messages: Vec<OpenAIMessage> = Vec::new();

    if let Some(system) = req.effective_system() {
        messages.push(OpenAIMessage {
            role: "system".to_string(),
            content: Some(OpenAIContent::String(system)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    messages.extend(try_openai_messages(&req.messages)?);

    let (max_tokens, max_completion_tokens) = if uses_max_completion_tokens(&req.model) {
        (None, req.max_tokens)
    } else {
        (req.max_tokens, None)
    };

    Ok(OpenAIRequest {
        model: req.model.clone(),
        messages,
        max_tokens,
        max_completion_tokens,
        temperature: req.temperature,
        tools: req.tools.iter().map(OpenAITool::from).collect(),
        stream: None,
    })
}
