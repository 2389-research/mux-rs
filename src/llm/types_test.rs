// ABOUTME: Tests for LLM types - serialization, deserialization, helpers.
// ABOUTME: Verifies JSON format matches provider APIs.

use super::*;

#[test]
fn test_role_serialization() {
    assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
    assert_eq!(
        serde_json::to_string(&Role::Assistant).unwrap(),
        "\"assistant\""
    );
}

#[test]
fn test_role_deserialization() {
    assert_eq!(
        serde_json::from_str::<Role>("\"user\"").unwrap(),
        Role::User
    );
    assert_eq!(
        serde_json::from_str::<Role>("\"assistant\"").unwrap(),
        Role::Assistant
    );
}

#[test]
fn test_content_block_text_serialization() {
    let block = ContentBlock::text("Hello");
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "text");
    assert_eq!(json["text"], "Hello");
}

#[test]
fn test_content_block_tool_use_deserialization() {
    let json = r#"{
        "type": "tool_use",
        "id": "123",
        "name": "read_file",
        "input": {"path": "/tmp/test.txt"}
    }"#;
    let block: ContentBlock = serde_json::from_str(json).unwrap();
    match block {
        ContentBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "123");
            assert_eq!(name, "read_file");
            assert_eq!(input["path"], "/tmp/test.txt");
        }
        _ => panic!("Expected ToolUse"),
    }
}

#[test]
fn test_content_block_tool_result_serialization() {
    let block = ContentBlock::tool_result("123", "file contents");
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_result");
    assert_eq!(json["tool_use_id"], "123");
    assert_eq!(json["content"], "file contents");
    assert_eq!(json["is_error"], false);
}

#[test]
fn test_content_block_tool_error_serialization() {
    let block = ContentBlock::tool_error("123", "file not found");
    let json = serde_json::to_value(&block).unwrap();
    assert_eq!(json["type"], "tool_result");
    assert_eq!(json["is_error"], true);
}

#[test]
fn test_message_user_helper() {
    let msg = Message::user("Hello");
    assert_eq!(msg.role, Role::User);
    assert_eq!(msg.content.len(), 1);
    match &msg.content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "Hello"),
        _ => panic!("Expected Text"),
    }
}

#[test]
fn test_request_builder() {
    let req = Request::new("claude-sonnet-4-20250514")
        .message(Message::user("Hi"))
        .system("You are helpful")
        .max_tokens(1024)
        .temperature(0.7);

    assert_eq!(req.model, "claude-sonnet-4-20250514");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.system, Some("You are helpful".to_string()));
    assert_eq!(req.max_tokens, Some(1024));
    assert_eq!(req.temperature, Some(0.7));
}

#[test]
fn test_response_has_tool_use() {
    let response = Response {
        id: "123".to_string(),
        content: vec![
            ContentBlock::text("I'll read that file"),
            ContentBlock::ToolUse {
                id: "456".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": "/tmp/test.txt"}),
            },
        ],
        stop_reason: StopReason::ToolUse,
        model: "claude-sonnet-4-20250514".to_string(),
        usage: Usage::default(),
    };

    assert!(response.has_tool_use());
    assert_eq!(response.tool_uses().len(), 1);
    assert_eq!(response.text(), "I'll read that file");
}

#[test]
fn test_response_no_tool_use() {
    let response = Response {
        id: "123".to_string(),
        content: vec![ContentBlock::text("Hello!")],
        stop_reason: StopReason::EndTurn,
        model: "claude-sonnet-4-20250514".to_string(),
        usage: Usage::default(),
    };

    assert!(!response.has_tool_use());
    assert!(response.tool_uses().is_empty());
}

#[test]
fn test_stop_reason_serialization() {
    assert_eq!(
        serde_json::to_string(&StopReason::EndTurn).unwrap(),
        "\"end_turn\""
    );
    assert_eq!(
        serde_json::to_string(&StopReason::ToolUse).unwrap(),
        "\"tool_use\""
    );
    assert_eq!(
        serde_json::to_string(&StopReason::MaxTokens).unwrap(),
        "\"max_tokens\""
    );
}

// --- Token Usage Tests ---

#[test]
fn test_usage_with_cache_tokens() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
    };

    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.cache_read_tokens, 20);
    assert_eq!(usage.cache_write_tokens, 10);
}

#[test]
fn test_usage_default_cache_tokens_are_zero() {
    let usage = Usage::default();

    assert_eq!(usage.cache_read_tokens, 0);
    assert_eq!(usage.cache_write_tokens, 0);
}

#[test]
fn test_usage_serialization_omits_zero_cache_tokens() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
    };

    let json = serde_json::to_value(&usage).unwrap();
    assert_eq!(json["input_tokens"], 100);
    assert_eq!(json["output_tokens"], 50);
    // Cache tokens should be omitted when zero
    assert!(json.get("cache_read_tokens").is_none());
    assert!(json.get("cache_write_tokens").is_none());
}

#[test]
fn test_usage_serialization_includes_nonzero_cache_tokens() {
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
    };

    let json = serde_json::to_value(&usage).unwrap();
    assert_eq!(json["cache_read_tokens"], 20);
    assert_eq!(json["cache_write_tokens"], 10);
}

#[test]
fn test_usage_deserialization_with_cache_tokens() {
    let json = r#"{
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10
    }"#;

    let usage: Usage = serde_json::from_str(json).unwrap();
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.cache_read_tokens, 20);
    assert_eq!(usage.cache_write_tokens, 10);
}

#[test]
fn test_usage_deserialization_without_cache_tokens() {
    let json = r#"{
        "input_tokens": 100,
        "output_tokens": 50
    }"#;

    let usage: Usage = serde_json::from_str(json).unwrap();
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.cache_read_tokens, 0);
    assert_eq!(usage.cache_write_tokens, 0);
}

// --- TokenUsage Accumulator Tests ---

#[test]
fn test_token_usage_new() {
    let tracker = TokenUsage::new();
    assert_eq!(tracker.total(), 0);
}

#[test]
fn test_token_usage_add() {
    let tracker = TokenUsage::new();
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
    };

    tracker.add(&usage);

    let snapshot = tracker.snapshot();
    assert_eq!(snapshot.input_tokens, 100);
    assert_eq!(snapshot.output_tokens, 50);
    assert_eq!(snapshot.cache_read_tokens, 20);
    assert_eq!(snapshot.cache_write_tokens, 10);
    assert_eq!(snapshot.request_count, 1);
}

#[test]
fn test_token_usage_add_multiple() {
    let tracker = TokenUsage::new();

    tracker.add(&Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
    });

    tracker.add(&Usage {
        input_tokens: 200,
        output_tokens: 100,
        cache_read_tokens: 30,
        cache_write_tokens: 0,
    });

    let snapshot = tracker.snapshot();
    assert_eq!(snapshot.input_tokens, 300);
    assert_eq!(snapshot.output_tokens, 150);
    assert_eq!(snapshot.cache_read_tokens, 30);
    assert_eq!(snapshot.request_count, 2);
}

#[test]
fn test_token_usage_add_with_cache() {
    let tracker = TokenUsage::new();
    let usage = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
    };

    tracker.add_with_cache(&usage, 20, 10);

    let snapshot = tracker.snapshot();
    assert_eq!(snapshot.input_tokens, 100);
    assert_eq!(snapshot.output_tokens, 50);
    assert_eq!(snapshot.cache_read_tokens, 20);
    assert_eq!(snapshot.cache_write_tokens, 10);
    assert_eq!(snapshot.request_count, 1);
}

#[test]
fn test_token_usage_total() {
    let tracker = TokenUsage::new();
    tracker.add(&Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
    });

    assert_eq!(tracker.total(), 150);
}

#[test]
fn test_token_usage_reset() {
    let tracker = TokenUsage::new();
    tracker.add(&Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
    });

    tracker.reset();

    let snapshot = tracker.snapshot();
    assert_eq!(snapshot.input_tokens, 0);
    assert_eq!(snapshot.output_tokens, 0);
    assert_eq!(snapshot.cache_read_tokens, 0);
    assert_eq!(snapshot.cache_write_tokens, 0);
    assert_eq!(snapshot.request_count, 0);
}

#[test]
fn test_token_usage_clone() {
    let tracker = TokenUsage::new();
    tracker.add(&Usage {
        input_tokens: 100,
        output_tokens: 50,
        ..Default::default()
    });

    let cloned = tracker.clone();

    // Both should see the same data (Arc sharing)
    tracker.add(&Usage {
        input_tokens: 100,
        output_tokens: 50,
        ..Default::default()
    });

    assert_eq!(cloned.total(), 300);
}

#[test]
fn test_token_usage_default() {
    let tracker = TokenUsage::default();
    assert_eq!(tracker.total(), 0);
}

// --- TokenUsageSnapshot Tests ---

#[test]
fn test_token_usage_snapshot_total() {
    let snapshot = TokenUsageSnapshot {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        request_count: 1,
    };

    assert_eq!(snapshot.total(), 150);
}

#[test]
fn test_token_usage_snapshot_display() {
    let snapshot = TokenUsageSnapshot {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        request_count: 1,
    };

    assert_eq!(
        snapshot.to_string(),
        "100 input + 50 output = 150 total (1 requests)"
    );
}

#[test]
fn test_token_usage_snapshot_display_multiple_requests() {
    let snapshot = TokenUsageSnapshot {
        input_tokens: 500,
        output_tokens: 250,
        cache_read_tokens: 100,
        cache_write_tokens: 50,
        request_count: 5,
    };

    assert_eq!(
        snapshot.to_string(),
        "500 input + 250 output = 750 total (5 requests)"
    );
}

#[test]
fn test_token_usage_snapshot_serialization_omits_zero_cache() {
    let snapshot = TokenUsageSnapshot {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        request_count: 1,
    };

    let json = serde_json::to_value(&snapshot).unwrap();
    assert!(json.get("cache_read_tokens").is_none());
    assert!(json.get("cache_write_tokens").is_none());
}

#[test]
fn test_token_usage_snapshot_serialization_includes_nonzero_cache() {
    let snapshot = TokenUsageSnapshot {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
        request_count: 1,
    };

    let json = serde_json::to_value(&snapshot).unwrap();
    assert_eq!(json["cache_read_tokens"], 20);
    assert_eq!(json["cache_write_tokens"], 10);
}

#[test]
fn test_token_usage_snapshot_deserialization() {
    let json = r#"{
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
        "request_count": 3
    }"#;

    let snapshot: TokenUsageSnapshot = serde_json::from_str(json).unwrap();
    assert_eq!(snapshot.input_tokens, 100);
    assert_eq!(snapshot.output_tokens, 50);
    assert_eq!(snapshot.cache_read_tokens, 20);
    assert_eq!(snapshot.cache_write_tokens, 10);
    assert_eq!(snapshot.request_count, 3);
}

#[test]
fn test_token_usage_snapshot_deserialization_without_cache() {
    let json = r#"{
        "input_tokens": 100,
        "output_tokens": 50,
        "request_count": 1
    }"#;

    let snapshot: TokenUsageSnapshot = serde_json::from_str(json).unwrap();
    assert_eq!(snapshot.cache_read_tokens, 0);
    assert_eq!(snapshot.cache_write_tokens, 0);
}

#[test]
fn test_token_usage_snapshot_equality() {
    let a = TokenUsageSnapshot {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: 20,
        cache_write_tokens: 10,
        request_count: 1,
    };
    let b = a.clone();

    assert_eq!(a, b);
}

#[test]
fn test_token_usage_snapshot_default() {
    let snapshot = TokenUsageSnapshot::default();
    assert_eq!(snapshot.input_tokens, 0);
    assert_eq!(snapshot.output_tokens, 0);
    assert_eq!(snapshot.cache_read_tokens, 0);
    assert_eq!(snapshot.cache_write_tokens, 0);
    assert_eq!(snapshot.request_count, 0);
}
