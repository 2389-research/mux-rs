// ABOUTME: Tests for Anthropic client type conversions.
// ABOUTME: Verifies serialization matches Anthropic API format.

use super::*;

#[test]
fn test_request_serialization() {
    let req = Request::new("claude-sonnet-4-20250514")
        .message(Message::user("Hello"))
        .system("You are helpful")
        .max_tokens(1024);

    let anthropic_req = AnthropicRequest::from(&req);

    assert_eq!(anthropic_req.model, "claude-sonnet-4-20250514");
    assert_eq!(anthropic_req.max_tokens, 1024);
    assert_eq!(anthropic_req.system, Some("You are helpful".to_string()));
    assert_eq!(anthropic_req.messages.len(), 1);
    assert_eq!(anthropic_req.messages[0].role, "user");
}

#[test]
fn test_request_json_format() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user("Hello"));

    let anthropic_req = AnthropicRequest::from(&req);
    let json = serde_json::to_value(&anthropic_req).unwrap();

    assert_eq!(json["model"], "claude-sonnet-4-20250514");
    assert_eq!(json["messages"][0]["role"], "user");
    assert_eq!(json["messages"][0]["content"][0]["type"], "text");
    assert_eq!(json["messages"][0]["content"][0]["text"], "Hello");
}

#[test]
fn test_tool_serialization() {
    let tool = ToolDefinition {
        name: "greet".to_string(),
        description: "Greet someone".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }),
    };

    let anthropic_tool = AnthropicTool::from(&tool);
    let json = serde_json::to_value(&anthropic_tool).unwrap();

    assert_eq!(json["name"], "greet");
    assert_eq!(json["description"], "Greet someone");
    assert!(json["input_schema"]["properties"]["name"].is_object());
}

#[test]
fn test_response_deserialization() {
    let json = r#"{
        "id": "msg_123",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": "end_turn",
        "model": "claude-sonnet-4-20250514",
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }"#;

    let anthropic_resp: AnthropicResponse = serde_json::from_str(json).unwrap();
    let response = Response::from(anthropic_resp);

    assert_eq!(response.id, "msg_123");
    assert_eq!(response.text(), "Hello!");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert_eq!(response.usage.input_tokens, 10);
}

#[test]
fn test_tool_use_response() {
    let json = r#"{
        "id": "msg_456",
        "content": [
            {"type": "text", "text": "Let me greet you."},
            {"type": "tool_use", "id": "tu_1", "name": "greet", "input": {"name": "Alice"}}
        ],
        "stop_reason": "tool_use",
        "model": "claude-sonnet-4-20250514",
        "usage": {"input_tokens": 20, "output_tokens": 15}
    }"#;

    let anthropic_resp: AnthropicResponse = serde_json::from_str(json).unwrap();
    let response = Response::from(anthropic_resp);

    assert!(response.has_tool_use());
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.tool_uses().len(), 1);
}

#[test]
fn test_tool_result_message() {
    let msg = Message::tool_results(vec![ContentBlock::tool_result("tu_1", "Hello, Alice!")]);

    let anthropic_msg = AnthropicMessage::from(&msg);
    let json = serde_json::to_value(&anthropic_msg).unwrap();

    assert_eq!(json["role"], "user");
    assert_eq!(json["content"][0]["type"], "tool_result");
    assert_eq!(json["content"][0]["tool_use_id"], "tu_1");
    assert_eq!(json["content"][0]["content"], "Hello, Alice!");
}

#[test]
fn test_client_from_env_missing() {
    // Temporarily unset the env var if it exists
    let original = std::env::var("ANTHROPIC_API_KEY").ok();
    unsafe {
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    let result = AnthropicClient::from_env();
    assert!(result.is_err());

    // Restore if it was set
    if let Some(val) = original {
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", val);
        }
    }
}
