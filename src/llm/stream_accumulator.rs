// ABOUTME: Utility that accumulates StreamEvents into Vec<ContentBlock>.
// ABOUTME: Handles text deltas, tool use JSON fragments, and block lifecycle.

use super::{ContentBlock, StreamEvent};

/// Accumulates streaming events into finalized content blocks.
///
/// Feed events via [`handle_event`](Self::handle_event) and call
/// [`into_content`](Self::into_content) to retrieve the assembled blocks.
pub struct StreamAccumulator {
    content_blocks: Vec<ContentBlock>,
    current_text: String,
    current_tool_id: String,
    current_tool_name: String,
    current_tool_input: String,
}

impl StreamAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self {
            content_blocks: Vec::new(),
            current_text: String::new(),
            current_tool_id: String::new(),
            current_tool_name: String::new(),
            current_tool_input: String::new(),
        }
    }

    /// Process a single stream event.
    pub fn handle_event(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart { block, .. } => match block {
                ContentBlock::Text { .. } => {
                    self.current_text = String::new();
                }
                ContentBlock::ToolUse { id, name, .. } => {
                    self.current_tool_id = id.clone();
                    self.current_tool_name = name.clone();
                    self.current_tool_input = String::new();
                }
                _ => {}
            },
            StreamEvent::ContentBlockDelta { text, .. } => {
                self.current_text.push_str(text);
            }
            StreamEvent::InputJsonDelta { partial_json, .. } => {
                self.current_tool_input.push_str(partial_json);
            }
            StreamEvent::ContentBlockStop { .. } => {
                if !self.current_tool_id.is_empty() {
                    // Finalize tool use block
                    let input = serde_json::from_str(&self.current_tool_input)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    self.content_blocks.push(ContentBlock::ToolUse {
                        id: std::mem::take(&mut self.current_tool_id),
                        name: std::mem::take(&mut self.current_tool_name),
                        input,
                    });
                    self.current_tool_input.clear();
                } else if !self.current_text.is_empty() {
                    // Finalize text block
                    self.content_blocks
                        .push(ContentBlock::text(std::mem::take(&mut self.current_text)));
                }
            }
            _ => {}
        }
    }

    /// Returns true if the accumulator is currently inside a tool use block.
    pub fn in_tool_use(&self) -> bool {
        !self.current_tool_id.is_empty()
    }

    /// Consume the accumulator and return the finalized content blocks.
    pub fn into_content(self) -> Vec<ContentBlock> {
        self.content_blocks
    }
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::StopReason;

    #[test]
    fn test_accumulate_text_only() {
        let mut acc = StreamAccumulator::new();

        acc.handle_event(&StreamEvent::MessageStart {
            id: "msg_1".into(),
            model: "test".into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 0,
            block: ContentBlock::text(""),
        });
        acc.handle_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            text: "Hello".into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            text: " world".into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStop { index: 0 });
        acc.handle_event(&StreamEvent::MessageDelta {
            stop_reason: Some(StopReason::EndTurn),
            usage: crate::llm::Usage::default(),
        });
        acc.handle_event(&StreamEvent::MessageStop);

        let blocks = acc.into_content();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello world"),
            _ => panic!("Expected Text block"),
        }
    }

    #[test]
    fn test_accumulate_tool_use() {
        let mut acc = StreamAccumulator::new();

        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 0,
            block: ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            },
        });
        acc.handle_event(&StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: r#"{"com"#.into(),
        });
        acc.handle_event(&StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: r#"mand": "ls"}"#.into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStop { index: 0 });

        let blocks = acc.into_content();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_1");
                assert_eq!(name, "bash");
                assert_eq!(input, &serde_json::json!({"command": "ls"}));
            }
            _ => panic!("Expected ToolUse block"),
        }
    }

    #[test]
    fn test_accumulate_mixed_text_and_tool() {
        let mut acc = StreamAccumulator::new();

        // Text block
        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 0,
            block: ContentBlock::text(""),
        });
        acc.handle_event(&StreamEvent::ContentBlockDelta {
            index: 0,
            text: "Let me check.".into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStop { index: 0 });

        // Tool use block
        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 1,
            block: ContentBlock::ToolUse {
                id: "toolu_2".into(),
                name: "read".into(),
                input: serde_json::json!({}),
            },
        });
        acc.handle_event(&StreamEvent::InputJsonDelta {
            index: 1,
            partial_json: r#"{"path": "foo.rs"}"#.into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStop { index: 1 });

        let blocks = acc.into_content();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me check."));
        assert!(matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "read"));
    }

    #[test]
    fn test_in_tool_use() {
        let mut acc = StreamAccumulator::new();
        assert!(!acc.in_tool_use());

        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 0,
            block: ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            },
        });
        assert!(acc.in_tool_use());

        acc.handle_event(&StreamEvent::ContentBlockStop { index: 0 });
        assert!(!acc.in_tool_use());
    }

    #[test]
    fn test_invalid_json_fallback() {
        let mut acc = StreamAccumulator::new();

        acc.handle_event(&StreamEvent::ContentBlockStart {
            index: 0,
            block: ContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "bash".into(),
                input: serde_json::json!({}),
            },
        });
        acc.handle_event(&StreamEvent::InputJsonDelta {
            index: 0,
            partial_json: "not valid json".into(),
        });
        acc.handle_event(&StreamEvent::ContentBlockStop { index: 0 });

        let blocks = acc.into_content();
        assert_eq!(blocks.len(), 1);
        // Should fall back to empty object
        match &blocks[0] {
            ContentBlock::ToolUse { input, .. } => {
                assert_eq!(input, &serde_json::json!({}));
            }
            _ => panic!("Expected ToolUse block"),
        }
    }
}
