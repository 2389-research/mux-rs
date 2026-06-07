// ABOUTME: Tests for the engine MCP integration (connect, list, call tools).
// ABOUTME: Extracted verbatim from mcp.rs; behavior unchanged.
use super::*;
use crate::engine::MuxEngine;
use crate::types::McpServerConfig;

fn test_dir(name: &str) -> String {
    std::env::temp_dir()
        .join(name)
        .to_string_lossy()
        .to_string()
}

fn create_test_engine() -> std::sync::Arc<MuxEngine> {
    MuxEngine::new(test_dir("mux-test-mcp")).unwrap()
}

fn create_stdio_config(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport_type: McpTransportType::Stdio,
        command: Some(if cfg!(target_os = "windows") {
            "cmd.exe".to_string()
        } else {
            "/usr/bin/echo".to_string()
        }),
        args: if cfg!(target_os = "windows") {
            vec!["/C".to_string(), "echo".to_string(), "hello".to_string()]
        } else {
            vec!["hello".to_string()]
        },
        url: None,
        enabled: true,
    }
}

fn create_sse_config(name: &str) -> McpServerConfig {
    McpServerConfig {
        name: name.to_string(),
        transport_type: McpTransportType::Sse,
        command: None,
        args: vec![],
        url: Some("http://localhost:8080".to_string()),
        enabled: false,
    }
}

#[test]
fn test_add_mcp_server() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Test".to_string(), None)
        .unwrap();

    let config = create_stdio_config("test-server");
    engine.add_mcp_server(ws.id.clone(), config).unwrap();

    let servers = engine.list_mcp_servers(ws.id.clone());
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].name, "test-server");

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_add_mcp_server_duplicate_name() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Dup Test".to_string(), None)
        .unwrap();

    let config1 = create_stdio_config("dup-server");
    engine.add_mcp_server(ws.id.clone(), config1).unwrap();

    let config2 = create_stdio_config("dup-server");
    let result = engine.add_mcp_server(ws.id.clone(), config2);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_add_mcp_server_workspace_not_found() {
    let engine = create_test_engine();
    let config = create_stdio_config("orphan-server");
    let result = engine.add_mcp_server("nonexistent-ws".to_string(), config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Workspace not found")
    );
}

#[test]
fn test_remove_mcp_server() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Remove Test".to_string(), None)
        .unwrap();

    let config = create_stdio_config("to-remove");
    engine.add_mcp_server(ws.id.clone(), config).unwrap();
    assert_eq!(engine.list_mcp_servers(ws.id.clone()).len(), 1);

    engine
        .remove_mcp_server(ws.id.clone(), "to-remove".to_string())
        .unwrap();
    assert_eq!(engine.list_mcp_servers(ws.id.clone()).len(), 0);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_remove_mcp_server_not_found() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Remove NF".to_string(), None)
        .unwrap();

    let result = engine.remove_mcp_server(ws.id.clone(), "ghost".to_string());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in workspace")
    );

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_remove_mcp_server_workspace_not_found() {
    let engine = create_test_engine();
    let result = engine.remove_mcp_server("fake-ws".to_string(), "any".to_string());
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Workspace not found")
    );
}

#[test]
fn test_list_mcp_servers_empty() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Empty".to_string(), None)
        .unwrap();

    let servers = engine.list_mcp_servers(ws.id.clone());
    assert!(servers.is_empty());

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_list_mcp_servers_nonexistent_workspace() {
    let engine = create_test_engine();
    let servers = engine.list_mcp_servers("no-such-ws".to_string());
    assert!(servers.is_empty());
}

#[test]
fn test_list_mcp_servers_multiple() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Multi".to_string(), None)
        .unwrap();

    engine
        .add_mcp_server(ws.id.clone(), create_stdio_config("server-a"))
        .unwrap();
    engine
        .add_mcp_server(ws.id.clone(), create_sse_config("server-b"))
        .unwrap();
    engine
        .add_mcp_server(ws.id.clone(), create_stdio_config("server-c"))
        .unwrap();

    let servers = engine.list_mcp_servers(ws.id.clone());
    assert_eq!(servers.len(), 3);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_update_mcp_server() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Update".to_string(), None)
        .unwrap();

    let config = create_stdio_config("updatable");
    engine.add_mcp_server(ws.id.clone(), config).unwrap();

    // Update to SSE transport
    let updated = McpServerConfig {
        name: "updatable".to_string(),
        transport_type: McpTransportType::Sse,
        command: None,
        args: vec![],
        url: Some("http://new-url:9000".to_string()),
        enabled: false,
    };
    engine.update_mcp_server(ws.id.clone(), updated).unwrap();

    let servers = engine.list_mcp_servers(ws.id.clone());
    assert_eq!(servers.len(), 1);
    assert_eq!(servers[0].transport_type, McpTransportType::Sse);
    assert_eq!(servers[0].url, Some("http://new-url:9000".to_string()));
    assert!(!servers[0].enabled);

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_update_mcp_server_not_found() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("MCP Update NF".to_string(), None)
        .unwrap();

    let config = create_stdio_config("ghost-update");
    let result = engine.update_mcp_server(ws.id.clone(), config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in workspace")
    );

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_update_mcp_server_workspace_not_found() {
    let engine = create_test_engine();
    let config = create_stdio_config("any");
    let result = engine.update_mcp_server("fake-ws".to_string(), config);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Workspace not found")
    );
}

#[test]
fn test_respond_to_tool_approval_with_pending() {
    let engine = create_test_engine();
    let (tx, rx) = tokio::sync::oneshot::channel();

    // Insert a pending approval
    engine
        .pending_approvals
        .write()
        .insert("test-tool-use-id".to_string(), tx);

    // Respond to it
    engine.respond_to_tool_approval("test-tool-use-id".to_string(), ApprovalDecision::Allow);

    // Verify it was received
    let rt = tokio::runtime::Runtime::new().unwrap();
    let decision = rt.block_on(async { rx.await.unwrap() });
    assert_eq!(decision, ApprovalDecision::Allow);
}

#[test]
fn test_respond_to_tool_approval_denied() {
    let engine = create_test_engine();
    let (tx, rx) = tokio::sync::oneshot::channel();

    engine
        .pending_approvals
        .write()
        .insert("deny-id".to_string(), tx);
    engine.respond_to_tool_approval("deny-id".to_string(), ApprovalDecision::Deny);

    let rt = tokio::runtime::Runtime::new().unwrap();
    let decision = rt.block_on(async { rx.await.unwrap() });
    assert_eq!(decision, ApprovalDecision::Deny);
}

#[test]
fn test_respond_to_tool_approval_no_pending() {
    let engine = create_test_engine();
    // This should not panic - just silently ignore
    engine.respond_to_tool_approval("nonexistent".to_string(), ApprovalDecision::Allow);
}

#[test]
fn test_get_workspace_tools_builtin_only() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Tools Test".to_string(), None)
        .unwrap();

    let tools = engine.get_workspace_tools(&ws.id);

    // Should have built-in tools: ReadFileTool, WriteFileTool, ListFilesTool, SearchTool, BashTool
    assert!(tools.len() >= 5);
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"read_file"));
    assert!(tool_names.contains(&"write_file"));
    assert!(tool_names.contains(&"list_files"));
    assert!(tool_names.contains(&"search"));
    assert!(tool_names.contains(&"bash"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_get_workspace_tools_no_task_tool_without_handler() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("No Task".to_string(), None)
        .unwrap();

    let tools = engine.get_workspace_tools(&ws.id);
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

    // Task tool should NOT be present without a subagent event handler
    assert!(!tool_names.contains(&"task"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_get_workspace_tools_with_task_tool() {
    use crate::callback::SubagentEventHandler;

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

    let engine = create_test_engine();
    let ws = engine
        .create_workspace("With Task".to_string(), None)
        .unwrap();

    engine.set_subagent_event_handler(Box::new(DummyHandler));

    let tools = engine.get_workspace_tools(&ws.id);
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();

    // Task tool SHOULD be present with a handler
    assert!(tool_names.contains(&"task"));

    engine.clear_subagent_event_handler();
    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_parse_tool_name_delegates_to_helpers() {
    let engine = create_test_engine();

    // Valid qualified name
    let result = engine.parse_tool_name("server:tool");
    assert_eq!(result, Some(("server".to_string(), "tool".to_string())));

    // No colon - returns None
    let result = engine.parse_tool_name("builtin_tool");
    assert!(result.is_none());
}

// ========================================================================
// MCP Resources Tests
// ========================================================================

#[test]
fn test_list_mcp_resources_empty_workspace() {
    let engine = create_test_engine();
    // Non-existent workspace returns empty list
    let resources = engine.list_mcp_resources("nonexistent".to_string());
    assert!(resources.is_empty());
}

#[test]
fn test_list_mcp_resources_no_connected_servers() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Resources Empty".to_string(), None)
        .unwrap();

    // Workspace exists but no MCP servers connected - returns empty
    let resources = engine.list_mcp_resources(ws.id.clone());
    assert!(resources.is_empty());

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_list_mcp_resource_templates_empty_workspace() {
    let engine = create_test_engine();
    let templates = engine.list_mcp_resource_templates("nonexistent".to_string());
    assert!(templates.is_empty());
}

#[test]
fn test_read_mcp_resource_server_not_connected() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Read Resource".to_string(), None)
        .unwrap();

    let result = engine.clone().read_mcp_resource(
        ws.id.clone(),
        "unknown_server".to_string(),
        "file:///test.txt".to_string(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not connected"));

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_read_mcp_resource_empty_uri() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Empty URI".to_string(), None)
        .unwrap();

    let result = engine.clone().read_mcp_resource(
        ws.id.clone(),
        "any_server".to_string(),
        "".to_string(), // Empty URI
    );
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("URI cannot be empty")
    );

    engine.delete_workspace(ws.id).unwrap();
}

// ========================================================================
// MCP Prompts Tests
// ========================================================================

#[test]
fn test_list_mcp_prompts_empty_workspace() {
    let engine = create_test_engine();
    let prompts = engine.list_mcp_prompts("nonexistent".to_string());
    assert!(prompts.is_empty());
}

#[test]
fn test_list_mcp_prompts_no_connected_servers() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Prompts Empty".to_string(), None)
        .unwrap();

    let prompts = engine.list_mcp_prompts(ws.id.clone());
    assert!(prompts.is_empty());

    engine.delete_workspace(ws.id).unwrap();
}

#[test]
fn test_get_mcp_prompt_server_not_connected() {
    let engine = create_test_engine();
    let ws = engine
        .create_workspace("Get Prompt".to_string(), None)
        .unwrap();

    let result = engine.clone().get_mcp_prompt(
        ws.id.clone(),
        "unknown_server".to_string(),
        "test_prompt".to_string(),
        vec![],
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not connected"));

    engine.delete_workspace(ws.id).unwrap();
}
