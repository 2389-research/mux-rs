// ABOUTME: Tests for Anthropic client type conversions.
// ABOUTME: Verifies serialization matches Anthropic API format.

use super::*;

#[test]
fn test_request_serialization() {
    let req = Request::new("claude-sonnet-4-20250514")
        .message(Message::user("Hello"))
        .system("You are helpful")
        .max_tokens(1024);

    let anthropic_req = try_into_anthropic_request(&req).unwrap();

    assert_eq!(anthropic_req.model, "claude-sonnet-4-20250514");
    assert_eq!(anthropic_req.max_tokens, 1024);
    assert_eq!(anthropic_req.system, Some("You are helpful".to_string()));
    assert_eq!(anthropic_req.messages.len(), 1);
    assert_eq!(anthropic_req.messages[0].role, "user");
}

#[test]
fn test_request_json_format() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user("Hello"));

    let anthropic_req = try_into_anthropic_request(&req).unwrap();
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

    let anthropic_msg = try_anthropic_message(&msg).unwrap();
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

#[test]
fn test_anthropic_image_base64_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::text("what is this?"),
        ContentBlock::image_base64("image/png", "aGVsbG8="),
    ]));
    let ar = try_into_anthropic_request(&req).unwrap();
    let json = serde_json::to_value(&ar).unwrap();
    let content = &json["messages"][0]["content"];
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image");
    assert_eq!(content[1]["source"]["type"], "base64");
    assert_eq!(content[1]["source"]["media_type"], "image/png");
    assert_eq!(content[1]["source"]["data"], "aGVsbG8=");
}

#[test]
fn test_anthropic_image_url_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::image_url("https://example.com/a.png"),
    ]));
    let ar = try_into_anthropic_request(&req).unwrap();
    let json = serde_json::to_value(&ar).unwrap();
    let src = &json["messages"][0]["content"][0]["source"];
    assert_eq!(src["type"], "url");
    assert_eq!(src["url"], "https://example.com/a.png");
}

#[test]
fn test_anthropic_document_serialization() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::document_base64("application/pdf", "JVBE"),
    ]));
    let ar = try_into_anthropic_request(&req).unwrap();
    let json = serde_json::to_value(&ar).unwrap();
    assert_eq!(json["messages"][0]["content"][0]["type"], "document");
    assert_eq!(
        json["messages"][0]["content"][0]["source"]["type"],
        "base64"
    );
    assert_eq!(
        json["messages"][0]["content"][0]["source"]["media_type"],
        "application/pdf"
    );
}

#[test]
fn test_anthropic_audio_errors() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::audio_base64("audio/wav", "UklGR"),
    ]));
    let result = try_into_anthropic_request(&req);
    assert!(matches!(
        result,
        Err(crate::error::LlmError::UnsupportedMedia {
            kind: MediaKind::Audio,
            ..
        })
    ));
}

#[test]
fn test_anthropic_video_errors() {
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::video_base64("video/mp4", "AAAAG"),
    ]));
    let result = try_into_anthropic_request(&req);
    assert!(matches!(
        result,
        Err(crate::error::LlmError::UnsupportedMedia {
            kind: MediaKind::Video,
            ..
        })
    ));
}

#[test]
fn test_anthropic_path_without_resolution_errors() {
    // try_into_anthropic_request expects paths to be pre-resolved. If a Path
    // is still present at serialize time, it's a programmer error.
    let req = Request::new("claude-sonnet-4-20250514").message(Message::user_with(vec![
        ContentBlock::image_path(std::path::PathBuf::from("/tmp/x.png")),
    ]));
    let result = try_into_anthropic_request(&req);
    assert!(result.is_err());
}

#[test]
fn test_anthropic_supports_media() {
    let client = AnthropicClient::new("fake");
    assert!(client.supports_media(MediaKind::Image));
    assert!(client.supports_media(MediaKind::Document));
    assert!(!client.supports_media(MediaKind::Audio));
    assert!(!client.supports_media(MediaKind::Video));
}
