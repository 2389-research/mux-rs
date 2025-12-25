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
