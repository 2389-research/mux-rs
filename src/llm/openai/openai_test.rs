// ABOUTME: Tests for the OpenAI client (request building, response parsing, media).
// ABOUTME: Extracted verbatim from openai.rs; behavior unchanged.
use super::*;
use crate::llm::{ContentBlock, MediaKind, Message, Request, Role, ToolDefinition};

#[test]
fn test_client_from_env_var_missing() {
    // Use a synthetic variable name guaranteed not to be set so the test
    // exercises the missing-key path without racing against other parallel
    // tests by mutating the process-global OPENAI_API_KEY.
    let result = OpenAIClient::from_env_var("MUX_TEST_NONEXISTENT_API_KEY_FOR_OPENAI");
    let err = result.expect_err("missing env var must produce an error");
    assert!(matches!(err, crate::error::LlmError::Configuration(_)));
}

#[test]
fn test_request_serialization() {
    let req = Request::new("gpt-4o")
        .message(Message::user("Hello"))
        .system("Be helpful")
        .max_tokens(100);

    let openai_req = try_into_openai_request(&req).unwrap();
    assert_eq!(openai_req.model, "gpt-4o");
    assert_eq!(openai_req.messages.len(), 2); // system + user
    assert_eq!(openai_req.messages[0].role, "system");
    assert_eq!(openai_req.messages[1].role, "user");
}

#[test]
fn test_tool_definition_conversion() {
    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the weather".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }),
        cache_control: None,
    };

    let openai_tool = OpenAITool::from(&tool);
    assert_eq!(openai_tool.tool_type, "function");
    assert_eq!(openai_tool.function.name, "get_weather");
}

#[test]
fn test_openai_image_base64_becomes_data_url() {
    let req = Request::new("gpt-4o").message(Message::user_with(vec![
        ContentBlock::text("what?"),
        ContentBlock::image_base64("image/png", "aGVsbG8="),
    ]));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    let content = &json["messages"][0]["content"];
    assert!(
        content.is_array(),
        "content should be parts array when media is present"
    );
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "what?");
    assert_eq!(content[1]["type"], "image_url");
    let url = content[1]["image_url"]["url"].as_str().unwrap();
    assert!(
        url.starts_with("data:image/png;base64,"),
        "got url: {}",
        url
    );
    assert!(url.ends_with("aGVsbG8="));
}

#[test]
fn test_openai_image_url_passthrough() {
    let req = Request::new("gpt-4o").message(Message::user_with(vec![ContentBlock::image_url(
        "https://example.com/cat.png",
    )]));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    assert_eq!(
        json["messages"][0]["content"][0]["image_url"]["url"],
        "https://example.com/cat.png"
    );
}

#[test]
fn test_openai_document_as_file() {
    let req =
        Request::new("gpt-4o").message(Message::user_with(vec![ContentBlock::document_base64(
            "application/pdf",
            "JVBERi0=",
        )]));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    let part = &json["messages"][0]["content"][0];
    assert_eq!(part["type"], "file");
    let file_data = part["file"]["file_data"]
        .as_str()
        .expect("file_data string");
    assert!(file_data.starts_with("data:application/pdf;base64,"));
    assert!(file_data.ends_with("JVBERi0="));
}

#[test]
fn test_openai_audio_with_format_inference() {
    let req = Request::new("gpt-4o-audio-preview").message(Message::user_with(vec![
        ContentBlock::audio_base64("audio/wav", "UklGR"),
    ]));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    let part = &json["messages"][0]["content"][0];
    assert_eq!(part["type"], "input_audio");
    assert_eq!(part["input_audio"]["data"], "UklGR");
    assert_eq!(part["input_audio"]["format"], "wav");
}

#[test]
fn test_openai_video_errors() {
    let req = Request::new("gpt-4o").message(Message::user_with(vec![ContentBlock::video_base64(
        "video/mp4",
        "AAAAG",
    )]));
    let result = try_into_openai_request(&req);
    assert!(matches!(
        result,
        Err(crate::error::LlmError::UnsupportedMedia {
            kind: MediaKind::Video,
            ..
        })
    ));
}

#[test]
fn test_openai_text_only_keeps_string_content() {
    // Ensure we don't regress backwards compat for pure-text messages.
    let req = Request::new("gpt-4o").message(Message::user("hello"));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    assert!(
        json["messages"][0]["content"].is_string(),
        "expected string content when no media present; got {:?}",
        json["messages"][0]["content"]
    );
    assert_eq!(json["messages"][0]["content"], "hello");
}

#[test]
fn test_openai_tool_result_message_preserved() {
    // Tool result messages use the legacy {"role":"tool","content":"...","tool_call_id":"..."} shape.
    // Ensure that path still works after the refactor.
    let req =
        Request::new("gpt-4o").message(Message::tool_results(vec![ContentBlock::tool_result(
            "tu_1",
            "Hello, Alice!",
        )]));
    let oa = try_into_openai_request(&req).unwrap();
    let json = serde_json::to_value(&oa).unwrap();
    assert_eq!(json["messages"][0]["role"], "tool");
    assert_eq!(json["messages"][0]["tool_call_id"], "tu_1");
    assert_eq!(json["messages"][0]["content"], "Hello, Alice!");
}

#[test]
fn test_openai_mixed_tool_result_and_text_errors() {
    let req = Request::new("gpt-4o").message(Message {
        role: Role::User,
        content: vec![
            ContentBlock::text("accompanying note"),
            ContentBlock::tool_result("tu_1", "result"),
        ],
    });
    let result = try_into_openai_request(&req);
    assert!(matches!(
        result,
        Err(crate::error::LlmError::Configuration(_))
    ));
}

#[test]
fn test_openai_audio_unknown_mime_errors() {
    let req = Request::new("gpt-4o-audio-preview").message(Message::user_with(vec![
        ContentBlock::audio_base64("audio/ogg", "xxx"),
    ]));
    let result = try_into_openai_request(&req);
    assert!(matches!(
        result,
        Err(crate::error::LlmError::Configuration(_))
    ));
}

#[test]
fn test_openai_supports_media() {
    use super::super::client::LlmClient;
    let c = OpenAIClient::new("fake");
    assert!(c.supports_media(MediaKind::Image));
    assert!(c.supports_media(MediaKind::Document));
    assert!(c.supports_media(MediaKind::Audio));
    assert!(!c.supports_media(MediaKind::Video));
}

#[test]
fn test_response_malformed_tool_arguments_propagates_error() {
    use crate::error::LlmError;
    use crate::llm::Response;

    let resp = OpenAIResponse {
        id: "resp_123".to_string(),
        model: "gpt-4o".to_string(),
        choices: vec![OpenAIChoice {
            index: 0,
            message: OpenAIResponseMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![OpenAIToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: "{not valid json".to_string(),
                    },
                }]),
            },
            finish_reason: Some("tool_calls".to_string()),
        }],
        usage: None,
    };

    let result = Response::try_from(resp);
    let err = result.expect_err("malformed tool arguments must propagate as error");
    let message = err.to_string();
    assert!(matches!(err, LlmError::Configuration(_)));
    assert!(message.contains("get_weather"), "error must reference tool name, got: {}", message);
}
