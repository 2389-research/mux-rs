// ABOUTME: Tests for the engine messaging path (chat send, streaming, subagent task tool).
// ABOUTME: Extracted verbatim from messaging.rs; behavior unchanged.
use super::*;
use crate::callback::{ChatCallback, SubagentEventHandler};
use crate::context::ContextUsage;
use crate::types::AgentConfig;
use std::sync::atomic::{AtomicBool, Ordering};

fn test_dir(name: &str) -> String {
    std::env::temp_dir()
        .join(name)
        .to_string_lossy()
        .to_string()
}

fn create_test_engine() -> Arc<MuxEngine> {
    MuxEngine::new(test_dir("mux-test-messaging")).unwrap()
}

// Mock callback that tracks calls
struct TrackingCallback {
    text_received: std::sync::Mutex<String>,
    error_received: std::sync::Mutex<Option<String>>,
    complete_called: AtomicBool,
}

impl TrackingCallback {
    fn new() -> Self {
        Self {
            text_received: std::sync::Mutex::new(String::new()),
            error_received: std::sync::Mutex::new(None),
            complete_called: AtomicBool::new(false),
        }
    }
}

impl ChatCallback for TrackingCallback {
    fn on_text_delta(&self, text: String) {
        self.text_received.lock().unwrap().push_str(&text);
    }

    fn on_tool_use(&self, _request: ToolUseRequest) {}

    fn on_tool_result(&self, _tool_use_id: String, _result: String) {}

    fn on_complete(&self, _result: ChatResult) {
        self.complete_called.store(true, Ordering::SeqCst);
    }

    fn on_error(&self, error: String) {
        *self.error_received.lock().unwrap() = Some(error);
    }

    fn on_context_warning(&self, _usage: ContextUsage) {}
}

#[test]
fn test_do_send_message_no_api_key_echo_fallback() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Msg Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Don't set any API key - should trigger echo fallback
    let callback = Arc::new(TrackingCallback::new());
    let _cb: Arc<Box<dyn ChatCallback>> = Arc::new(Box::new(TrackingCallback::new()));

    // Need to run async code
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Hello world".to_string(),
        Vec::new(),
        Arc::new(Box::new({
            struct Wrapper(Arc<TrackingCallback>);
            impl ChatCallback for Wrapper {
                fn on_text_delta(&self, text: String) {
                    self.0.on_text_delta(text);
                }
                fn on_tool_use(&self, r: ToolUseRequest) {
                    self.0.on_tool_use(r);
                }
                fn on_tool_result(&self, id: String, result: String) {
                    self.0.on_tool_result(id, result);
                }
                fn on_complete(&self, r: ChatResult) {
                    self.0.on_complete(r);
                }
                fn on_error(&self, e: String) {
                    self.0.on_error(e);
                }
                fn on_context_warning(&self, u: ContextUsage) {
                    self.0.on_context_warning(u);
                }
            }
            Wrapper(callback.clone())
        })),
    ));

    assert!(result.is_ok());
    let chat_result = result.unwrap();

    // Should return echo message
    assert!(chat_result.final_text.contains("Echo: Hello world"));
    assert!(chat_result.final_text.contains("No API key set"));
    assert_eq!(chat_result.tool_use_count, 0);
    assert_eq!(chat_result.input_tokens, 0);
    assert_eq!(chat_result.output_tokens, 0);

    // Callback should have received the text
    let text = callback.text_received.lock().unwrap();
    assert!(text.contains("Echo: Hello world"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_custom_provider_not_registered() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Custom Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Set custom provider but don't register it
    engine.set_default_provider(Provider::Custom {
        name: "my-custom-llm".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Hello".to_string(),
        Vec::new(),
        Arc::new(Box::new({
            struct Wrapper(Arc<TrackingCallback>);
            impl ChatCallback for Wrapper {
                fn on_text_delta(&self, text: String) {
                    self.0.on_text_delta(text);
                }
                fn on_tool_use(&self, r: ToolUseRequest) {
                    self.0.on_tool_use(r);
                }
                fn on_tool_result(&self, id: String, result: String) {
                    self.0.on_tool_result(id, result);
                }
                fn on_complete(&self, r: ChatResult) {
                    self.0.on_complete(r);
                }
                fn on_error(&self, e: String) {
                    self.0.on_error(e);
                }
                fn on_context_warning(&self, u: ContextUsage) {
                    self.0.on_context_warning(u);
                }
            }
            Wrapper(callback.clone())
        })),
    ));

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("my-custom-llm"));
    assert!(error.contains("not registered"));

    // Callback should have received the error
    let error_received = callback.error_received.lock().unwrap();
    assert!(error_received.is_some());
    assert!(error_received.as_ref().unwrap().contains("my-custom-llm"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_media_attached_to_user_message_echo_fallback() {
    // Verifies the `media` parameter plumbed through send_message ->
    // do_send_message is actually attached to the stored user message.
    // Uses the echo-fallback path (no API key) which is the Message::user
    // construction site inside do_send_message itself.
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Media Attach Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    let media = vec![crate::media::FfiMedia {
        kind: crate::media::FfiMediaKind::Image,
        source: crate::media::FfiMediaSource::Base64 {
            data: "aGVsbG8=".to_string(),
        },
        mime_type: "image/png".to_string(),
    }];

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Describe this image".to_string(),
        media,
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    assert!(result.is_ok());

    // The stored user message should contain both the text block and the
    // media block, confirming `media` reached the Message construction.
    let history = engine.message_history.read();
    let messages = history.get(&conv.id).expect("history entry for conv");
    let user_msg = messages
        .iter()
        .find(|m| matches!(m.role, Role::User))
        .expect("user message stored");
    assert_eq!(
        user_msg.content.len(),
        2,
        "user message should have text + media blocks"
    );
    assert!(matches!(user_msg.content[0], ContentBlock::Text { .. }));
    assert!(matches!(
        user_msg.content[1],
        ContentBlock::Media {
            kind: mux::llm::MediaKind::Image,
            ..
        }
    ));
    drop(history);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_execute_task_tool_no_handler() {
    let engine = create_test_engine();

    // Don't set subagent event handler
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
        "agent_type": "test",
        "task": "do something",
        "description": "test task"
    })));

    assert!(result.is_ok());
    let tool_result = result.unwrap();
    assert!(tool_result.is_error);
    assert!(tool_result.content.contains("no subagent event handler"));
}

#[test]
fn test_execute_task_tool_custom_provider_not_registered() {
    let engine = create_test_engine();

    // Set up handler but use unregistered custom provider
    struct DummyHandler;
    impl SubagentEventHandler for DummyHandler {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
        fn on_tool_use(&self, _: String, _: String, _: String) {}
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
        fn on_iteration(&self, _: String, _: u32) {}
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
        fn on_agent_error(&self, _: String, _: String) {}
        fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
        fn on_stream_usage(&self, _subagent_id: String, _input_tokens: u32, _output_tokens: u32) {}
    }

    engine.set_subagent_event_handler(Box::new(DummyHandler));
    engine.set_default_provider(Provider::Custom {
        name: "unregistered-provider".to_string(),
    });

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
        "agent_type": "test",
        "task": "do something",
        "description": "test task"
    })));

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("unregistered-provider"));
    assert!(error.contains("not registered"));

    engine.clear_subagent_event_handler();
}

#[test]
fn test_execute_task_tool_no_model_for_agent() {
    let engine = create_test_engine();

    struct DummyHandler;
    impl SubagentEventHandler for DummyHandler {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
        fn on_tool_use(&self, _: String, _: String, _: String) {}
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
        fn on_iteration(&self, _: String, _: u32) {}
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
        fn on_agent_error(&self, _: String, _: String) {}
        fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
        fn on_stream_usage(&self, _subagent_id: String, _input_tokens: u32, _output_tokens: u32) {}
    }

    engine.set_subagent_event_handler(Box::new(DummyHandler));

    // Register an agent WITHOUT a model, and don't set provider default model
    engine
        .register_agent(AgentConfig::new(
            "no-model-agent".to_string(),
            "You are a test agent.".to_string(),
        ))
        .unwrap();

    // Set Anthropic provider with API key but NO default model
    engine.set_api_key(Provider::Anthropic, "sk-test".to_string());

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
        "agent_type": "no-model-agent",
        "task": "do something",
        "description": "test task"
    })));

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("No model configured"));
    assert!(error.contains("no-model-agent"));

    engine.clear_subagent_event_handler();
}

#[test]
fn test_execute_task_tool_provider_not_configured() {
    let engine = create_test_engine();

    struct DummyHandler;
    impl SubagentEventHandler for DummyHandler {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
        fn on_tool_use(&self, _: String, _: String, _: String) {}
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
        fn on_iteration(&self, _: String, _: u32) {}
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
        fn on_agent_error(&self, _: String, _: String) {}
        fn on_stream_delta(&self, _subagent_id: String, _text: String) {}
        fn on_stream_usage(&self, _subagent_id: String, _input_tokens: u32, _output_tokens: u32) {}
    }

    engine.set_subagent_event_handler(Box::new(DummyHandler));

    // Register an agent with a model
    let mut config = AgentConfig::new(
        "configured-agent".to_string(),
        "You are a test agent.".to_string(),
    );
    config.model = Some("claude-3-opus".to_string());
    engine.register_agent(config).unwrap();

    // Set Anthropic as provider but DON'T configure API key
    engine.set_default_provider(Provider::Anthropic);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.execute_task_tool(serde_json::json!({
        "agent_type": "configured-agent",
        "task": "do something",
        "description": "test task"
    })));

    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("Provider not configured"));

    engine.clear_subagent_event_handler();
}

// ========================================================================
// Mock LLM Provider for testing the agentic loop
// ========================================================================

use crate::callback::LlmProvider;
use crate::types::{ChatRole, LlmRequest, LlmResponse, LlmToolCall, LlmUsage};
use std::sync::atomic::AtomicU32;

/// Mock LLM provider that returns canned responses.
/// Can be configured to return text-only or tool calls.
struct MockLlmProvider {
    responses: std::sync::Mutex<Vec<LlmResponse>>,
    call_count: AtomicU32,
}

impl MockLlmProvider {
    fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
            call_count: AtomicU32::new(0),
        }
    }

    /// Create a simple text-only response
    fn text_response(text: &str) -> LlmResponse {
        LlmResponse {
            text: text.to_string(),
            tool_calls: vec![],
            usage: LlmUsage {
                input_tokens: 10,
                output_tokens: 20,
            },
            error: None,
        }
    }

    /// Create a response with a tool call
    fn tool_call_response(tool_name: &str, args: &str) -> LlmResponse {
        LlmResponse {
            text: String::new(),
            tool_calls: vec![LlmToolCall {
                id: format!("tool_{}", uuid::Uuid::new_v4()),
                name: tool_name.to_string(),
                arguments: args.to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
            error: None,
        }
    }

    #[allow(dead_code)]
    fn get_call_count(&self) -> u32 {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl LlmProvider for MockLlmProvider {
    fn generate(&self, _request: LlmRequest) -> LlmResponse {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        let responses = self.responses.lock().unwrap();

        if (count as usize) < responses.len() {
            responses[count as usize].clone()
        } else if !responses.is_empty() {
            // Return last response if we've exhausted the list
            responses.last().unwrap().clone()
        } else {
            // Default: return simple text
            MockLlmProvider::text_response("Default mock response")
        }
    }

    fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
        false
    }
}

// Helper to create a wrapper struct for ChatCallback forwarding
struct CallbackWrapper(Arc<TrackingCallback>);

impl ChatCallback for CallbackWrapper {
    fn on_text_delta(&self, text: String) {
        self.0.on_text_delta(text);
    }
    fn on_tool_use(&self, r: ToolUseRequest) {
        self.0.on_tool_use(r);
    }
    fn on_tool_result(&self, id: String, result: String) {
        self.0.on_tool_result(id, result);
    }
    fn on_complete(&self, r: ChatResult) {
        self.0.on_complete(r);
    }
    fn on_error(&self, e: String) {
        self.0.on_error(e);
    }
    fn on_context_warning(&self, u: ContextUsage) {
        self.0.on_context_warning(u);
    }
}

#[test]
fn test_do_send_message_with_mock_llm_simple_text() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MockLLM Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Register mock LLM provider
    let mock_provider =
        MockLlmProvider::new(vec![MockLlmProvider::text_response("Hello from mock LLM!")]);
    engine.register_llm_provider("mock-llm".to_string(), Box::new(mock_provider));

    // Set custom provider
    engine.set_default_provider(Provider::Custom {
        name: "mock-llm".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Hi there".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    assert!(result.is_ok());
    let chat_result = result.unwrap();

    // Verify the response
    assert_eq!(chat_result.final_text, "Hello from mock LLM!");
    assert_eq!(chat_result.tool_use_count, 0);
    assert_eq!(chat_result.input_tokens, 10);
    assert_eq!(chat_result.output_tokens, 20);

    // Verify callback received the text
    let text = callback.text_received.lock().unwrap();
    assert!(text.contains("Hello from mock LLM!"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_with_mock_llm_tool_use() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MockLLM Tool Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Register mock LLM that first calls a tool, then responds with text
    let mock_provider = MockLlmProvider::new(vec![
        // First call: request read_file tool
        MockLlmProvider::tool_call_response("read_file", r#"{"path": "/tmp/test.txt"}"#),
        // Second call: after tool result, return final text
        MockLlmProvider::text_response("I read the file successfully!"),
    ]);
    engine.register_llm_provider("mock-tool-llm".to_string(), Box::new(mock_provider));

    engine.set_default_provider(Provider::Custom {
        name: "mock-tool-llm".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Read that file for me".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    assert!(result.is_ok());
    let chat_result = result.unwrap();

    // Should have used one tool
    assert_eq!(chat_result.tool_use_count, 1);
    assert_eq!(chat_result.final_text, "I read the file successfully!");

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_accumulates_tokens() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Token Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Multiple iterations accumulate tokens
    let mock_provider = MockLlmProvider::new(vec![
        LlmResponse {
            text: String::new(),
            tool_calls: vec![LlmToolCall {
                id: "tool_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path": "/tmp/a.txt"}"#.to_string(),
            }],
            usage: LlmUsage {
                input_tokens: 100,
                output_tokens: 50,
            },
            error: None,
        },
        LlmResponse {
            text: "Done!".to_string(),
            tool_calls: vec![],
            usage: LlmUsage {
                input_tokens: 200,
                output_tokens: 75,
            },
            error: None,
        },
    ]);
    engine.register_llm_provider("token-test".to_string(), Box::new(mock_provider));
    engine.set_default_provider(Provider::Custom {
        name: "token-test".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Do something".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    assert!(result.is_ok());
    let chat_result = result.unwrap();

    // Tokens should be accumulated: 100+200 input, 50+75 output
    assert_eq!(chat_result.input_tokens, 300);
    assert_eq!(chat_result.output_tokens, 125);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_max_iterations_limit() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Max Iter Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Mock that always returns tool calls - should hit MAX_AGENTIC_ITERATIONS
    let mock_provider = MockLlmProvider::new(vec![MockLlmProvider::tool_call_response(
        "read_file",
        r#"{"path": "/tmp/loop.txt"}"#,
    )]);
    engine.register_llm_provider("loop-test".to_string(), Box::new(mock_provider));
    engine.set_default_provider(Provider::Custom {
        name: "loop-test".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Loop forever".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    assert!(result.is_ok());
    let chat_result = result.unwrap();

    // Should have hit the limit (50 iterations)
    assert!(chat_result.tool_use_count >= 49); // At least 49 tool uses
    assert!(chat_result.final_text.contains("terminated after"));
    assert!(chat_result.final_text.contains("50 iterations"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_llm_error_response() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Error Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Mock that returns an error
    let mock_provider = MockLlmProvider::new(vec![LlmResponse {
        text: String::new(),
        tool_calls: vec![],
        usage: LlmUsage::default(),
        error: Some("Rate limit exceeded".to_string()),
    }]);
    engine.register_llm_provider("error-test".to_string(), Box::new(mock_provider));
    engine.set_default_provider(Provider::Custom {
        name: "error-test".to_string(),
    });

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Trigger error".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));

    // LLM errors are propagated
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("Rate limit exceeded"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_preserves_conversation_history() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("History Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    // Track how many messages have been sent
    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = call_count.clone();

    struct HistoryCheckingProvider {
        call_count: Arc<AtomicU32>,
    }

    impl LlmProvider for HistoryCheckingProvider {
        fn generate(&self, request: LlmRequest) -> LlmResponse {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);

            // First call should have 1 message, second call should have 3
            // (user, assistant from first call, new user)
            if count == 0 {
                assert_eq!(request.messages.len(), 1);
            } else if count == 1 {
                // After first response: user + assistant + new user = 3
                assert!(request.messages.len() >= 2);
            }

            LlmResponse {
                text: format!("Response {}", count),
                tool_calls: vec![],
                usage: LlmUsage {
                    input_tokens: 10,
                    output_tokens: 10,
                },
                error: None,
            }
        }

        fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
            false
        }
    }

    let provider = HistoryCheckingProvider {
        call_count: call_count_clone,
    };
    engine.register_llm_provider("history-test".to_string(), Box::new(provider));
    engine.set_default_provider(Provider::Custom {
        name: "history-test".to_string(),
    });

    let rt = tokio::runtime::Runtime::new().unwrap();

    // First message
    let callback1 = Arc::new(TrackingCallback::new());
    let result1 = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "First message".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback1.clone()))),
    ));
    assert!(result1.is_ok());

    // Second message - should include history
    let callback2 = Arc::new(TrackingCallback::new());
    let result2 = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Second message".to_string(),
        Vec::new(),
        Arc::new(Box::new(CallbackWrapper(callback2.clone()))),
    ));
    assert!(result2.is_ok());

    // Verify we made 2 calls total
    assert_eq!(call_count.load(Ordering::SeqCst), 2);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_ffi_media_into_blocks_resolves_path_to_base64() {
    // The helper should resolve MediaSource::Path attachments to
    // MediaSource::Base64 so persisted conversations stay portable.
    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("fixture.png");
    std::fs::write(&png_path, b"fake-png-bytes").unwrap();

    let media = vec![crate::media::FfiMedia {
        kind: crate::media::FfiMediaKind::Image,
        source: crate::media::FfiMediaSource::Path {
            path: png_path.to_string_lossy().into_owned(),
        },
        mime_type: "image/png".to_string(),
    }];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let blocks = rt.block_on(ffi_media_into_blocks(media)).unwrap();
    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::Media {
            kind,
            source,
            mime_type,
        } => {
            assert_eq!(*kind, mux::llm::MediaKind::Image);
            assert_eq!(mime_type, "image/png");
            // Base64 of b"fake-png-bytes"
            match source {
                MediaSource::Base64(data) => {
                    assert_eq!(data, "ZmFrZS1wbmctYnl0ZXM=");
                }
                other => panic!("expected Base64 source, got {:?}", other),
            }
        }
        other => panic!("expected Media block, got {:?}", other),
    }
}

#[test]
fn test_ffi_media_into_blocks_passes_through_non_path_sources() {
    // Base64 and Url sources should be forwarded without modification.
    let media = vec![
        crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Base64 {
                data: "aGVsbG8=".to_string(),
            },
            mime_type: "image/png".to_string(),
        },
        crate::media::FfiMedia {
            kind: crate::media::FfiMediaKind::Image,
            source: crate::media::FfiMediaSource::Url {
                url: "https://example.com/x.png".to_string(),
            },
            mime_type: "image/png".to_string(),
        },
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let blocks = rt.block_on(ffi_media_into_blocks(media)).unwrap();
    assert_eq!(blocks.len(), 2);
    match &blocks[0] {
        ContentBlock::Media {
            source: MediaSource::Base64(data),
            ..
        } => assert_eq!(data, "aGVsbG8="),
        other => panic!("expected Base64 source, got {:?}", other),
    }
    match &blocks[1] {
        ContentBlock::Media {
            source: MediaSource::Url(url),
            ..
        } => assert_eq!(url, "https://example.com/x.png"),
        other => panic!("expected Url source, got {:?}", other),
    }
}

#[test]
fn test_do_send_message_path_media_normalized_to_base64_in_history() {
    // End-to-end check: a MediaSource::Path attachment passed through
    // do_send_message should be persisted in history as MediaSource::Base64.
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Path Normalize Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let png_path = dir.path().join("fixture.png");
    std::fs::write(&png_path, b"fake-png-bytes").unwrap();

    let media = vec![crate::media::FfiMedia {
        kind: crate::media::FfiMediaKind::Image,
        source: crate::media::FfiMediaSource::Path {
            path: png_path.to_string_lossy().into_owned(),
        },
        mime_type: "image/png".to_string(),
    }];

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Describe this image".to_string(),
        media,
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));
    assert!(result.is_ok(), "do_send_message failed: {:?}", result);

    // History's user message should have the media block normalized to
    // Base64 — NOT Path — so saved conversations stay portable.
    let history = engine.message_history.read();
    let messages = history.get(&conv.id).expect("history entry for conv");
    let user_msg = messages
        .iter()
        .find(|m| matches!(m.role, Role::User))
        .expect("user message stored");
    let media_block = user_msg
        .content
        .iter()
        .find(|b| matches!(b, ContentBlock::Media { .. }))
        .expect("user message should contain a media block");
    match media_block {
        ContentBlock::Media { source, .. } => {
            assert!(
                matches!(source, MediaSource::Base64(_)),
                "expected Base64 source in history, got {:?}",
                source
            );
        }
        _ => unreachable!(),
    }
    drop(history);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_do_send_message_media_reaches_real_llm_provider() {
    // Verifies that media attachments plumbed through do_send_message reach
    // the provider on the real-LLM path (via SubAgent::run_with_blocks),
    // not just the echo fallback path.
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Media RealLLM Test".to_string(), None)
        .unwrap();
    let conv = engine
        .create_conversation(ws.id.clone(), "Test Conv".to_string())
        .unwrap();

    /// Provider that captures the most recent request's user-message media count.
    struct CapturingProvider {
        captured_media_count: Arc<std::sync::Mutex<Option<usize>>>,
    }

    impl LlmProvider for CapturingProvider {
        fn generate(&self, request: LlmRequest) -> LlmResponse {
            // Find the user message and record how many media blocks it has.
            let user_media = request
                .messages
                .iter()
                .find(|m| matches!(m.role, ChatRole::User))
                .map(|m| m.media.len())
                .unwrap_or(0);
            *self.captured_media_count.lock().unwrap() = Some(user_media);

            LlmResponse {
                text: "ack".to_string(),
                tool_calls: vec![],
                usage: LlmUsage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
                error: None,
            }
        }

        fn supports_media(&self, _kind: crate::media::FfiMediaKind) -> bool {
            true
        }
    }

    let captured: Arc<std::sync::Mutex<Option<usize>>> = Arc::new(std::sync::Mutex::new(None));
    let provider = CapturingProvider {
        captured_media_count: captured.clone(),
    };
    engine.register_llm_provider("media-capture".to_string(), Box::new(provider));
    engine.set_default_provider(Provider::Custom {
        name: "media-capture".to_string(),
    });

    let media = vec![crate::media::FfiMedia {
        kind: crate::media::FfiMediaKind::Image,
        source: crate::media::FfiMediaSource::Base64 {
            data: "aGVsbG8=".to_string(),
        },
        mime_type: "image/png".to_string(),
    }];

    let callback = Arc::new(TrackingCallback::new());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(engine.do_send_message(
        conv.id.clone(),
        "Describe this image".to_string(),
        media,
        Arc::new(Box::new(CallbackWrapper(callback.clone()))),
    ));
    assert!(result.is_ok(), "do_send_message failed: {:?}", result);

    // Confirm the provider observed the media on the user message.
    let captured = captured.lock().unwrap();
    assert_eq!(
        *captured,
        Some(1),
        "provider should have seen 1 media block on the user message"
    );

    engine.delete_workspace(ws.id).unwrap();
}

// =============================================================================
// build_tool_registry — task tool wiring regression (#9)
// =============================================================================
//
// These tests pin the wiring change that closes #9: the production chat loop's
// tool registry must include the `task` tool when (and only when) a subagent
// event handler is registered. Together with the existing
// `test_execute_task_tool_*` and the FfiTaskTool unit tests, they cover the
// full path from "host registers handler" to "LLM sees `task` in its
// inventory".

fn build_tool_registry_for_test(engine: &Arc<MuxEngine>) -> Registry {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        engine
            .build_tool_registry(&None, &std::collections::HashMap::new())
            .await
    })
}

fn assert_builtins_present(names: &[String]) {
    for expected in ["read_file", "write_file", "list_files", "search", "bash"] {
        assert!(
            names.iter().any(|n| n == expected),
            "expected builtin '{}' in registry; got {:?}",
            expected,
            names
        );
    }
}

#[test]
fn test_build_tool_registry_omits_task_when_no_handler() {
    let engine = create_test_engine();
    // Make sure no leftover handler from a previous test pollutes state.
    engine.clear_subagent_event_handler();
    engine.set_api_key(Provider::Anthropic, "sk-test".to_string());

    let registry = build_tool_registry_for_test(&engine);
    let names = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(registry.list());

    assert_builtins_present(&names);
    assert!(
        !names.iter().any(|n| n == "task"),
        "task tool must NOT be registered without a handler; got {:?}",
        names
    );
}

#[test]
fn test_build_tool_registry_includes_task_when_handler_set() {
    let engine = create_test_engine();
    engine.set_api_key(Provider::Anthropic, "sk-test".to_string());

    struct CountingHandler;
    impl SubagentEventHandler for CountingHandler {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
        fn on_tool_use(&self, _: String, _: String, _: String) {}
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
        fn on_iteration(&self, _: String, _: u32) {}
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
        fn on_agent_error(&self, _: String, _: String) {}
        fn on_stream_delta(&self, _: String, _: String) {}
        fn on_stream_usage(&self, _: String, _: u32, _: u32) {}
    }
    engine.set_subagent_event_handler(Box::new(CountingHandler));

    let registry = build_tool_registry_for_test(&engine);
    let names = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(registry.list());

    assert_builtins_present(&names);
    assert!(
        names.iter().any(|n| n == "task"),
        "task tool must be registered when a handler is set; got {:?}",
        names
    );

    engine.clear_subagent_event_handler();
}

#[test]
fn test_build_tool_registry_drops_task_after_handler_cleared() {
    // Idempotency check: registering then clearing the handler reverts the
    // tool inventory. Each `send_message` turn rebuilds the registry, so a
    // host that calls `clear_subagent_event_handler` mid-conversation gets
    // a turn without `task` immediately afterward.
    let engine = create_test_engine();
    engine.set_api_key(Provider::Anthropic, "sk-test".to_string());

    struct H;
    impl SubagentEventHandler for H {
        fn on_agent_started(&self, _: String, _: String, _: String, _: String) {}
        fn on_tool_use(&self, _: String, _: String, _: String) {}
        fn on_tool_result(&self, _: String, _: String, _: String, _: bool) {}
        fn on_iteration(&self, _: String, _: u32) {}
        fn on_agent_completed(&self, _: String, _: String, _: u32, _: u32, _: bool) {}
        fn on_agent_error(&self, _: String, _: String) {}
        fn on_stream_delta(&self, _: String, _: String) {}
        fn on_stream_usage(&self, _: String, _: u32, _: u32) {}
    }
    engine.set_subagent_event_handler(Box::new(H));

    // First build: handler is set, task is present.
    let names_before = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(build_tool_registry_for_test(&engine).list());
    assert!(names_before.iter().any(|n| n == "task"));

    // Clear and rebuild: task is gone.
    engine.clear_subagent_event_handler();
    let names_after = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(build_tool_registry_for_test(&engine).list());
    assert!(!names_after.iter().any(|n| n == "task"));
}
