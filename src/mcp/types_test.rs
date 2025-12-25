// ABOUTME: Tests for MCP types - serialization, deserialization.
// ABOUTME: Verifies JSON format matches MCP protocol.

use super::*;

#[test]
fn test_request_serialization() {
    let req = McpRequest::new("tools/list", None);
    let json = serde_json::to_value(&req).unwrap();

    assert_eq!(json["jsonrpc"], "2.0");
    assert_eq!(json["method"], "tools/list");
    assert!(json["id"].as_u64().is_some());
}

#[test]
fn test_request_with_params() {
    let params = serde_json::json!({"name": "read_file", "arguments": {"path": "/tmp"}});
    let req = McpRequest::new("tools/call", Some(params.clone()));
    let json = serde_json::to_value(&req).unwrap();

    assert_eq!(json["params"], params);
}

#[test]
fn test_response_deserialization_success() {
    let json = r#"{
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": []}
    }"#;

    let resp: McpResponse = serde_json::from_str(json).unwrap();
    assert!(resp.result.is_some());
    assert!(resp.error.is_none());
}

#[test]
fn test_response_deserialization_error() {
    let json = r#"{
        "jsonrpc": "2.0",
        "id": 1,
        "error": {
            "code": -32600,
            "message": "Invalid Request"
        }
    }"#;

    let resp: McpResponse = serde_json::from_str(json).unwrap();
    assert!(resp.result.is_none());
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, -32600);
}

#[test]
fn test_tool_info_deserialization() {
    let json = r#"{
        "name": "read_file",
        "description": "Read a file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }
    }"#;

    let info: McpToolInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.name, "read_file");
    assert_eq!(info.description, "Read a file");
}

#[test]
fn test_tool_result_deserialization() {
    let json = r#"{
        "content": [
            {"type": "text", "text": "file contents here"}
        ],
        "isError": false
    }"#;

    let result: McpToolResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.content.len(), 1);
    assert!(!result.is_error);

    match &result.content[0] {
        McpContentBlock::Text { text } => assert_eq!(text, "file contents here"),
        _ => panic!("Expected text block"),
    }
}

#[test]
fn test_initialize_params_serialization() {
    let params = McpInitializeParams {
        protocol_version: "2024-11-05".to_string(),
        capabilities: serde_json::json!({}),
        client_info: McpClientInfo {
            name: "mux".to_string(),
            version: "0.1.0".to_string(),
        },
    };

    let json = serde_json::to_value(&params).unwrap();
    assert_eq!(json["protocolVersion"], "2024-11-05");
    assert_eq!(json["clientInfo"]["name"], "mux");
}

#[test]
fn test_request_ids_increment() {
    let req1 = McpRequest::new("test1", None);
    let req2 = McpRequest::new("test2", None);

    assert!(req2.id > req1.id);
}
