// ABOUTME: Compile-only guard pinning the mux public API surface.
// ABOUTME: If a prelude/public path is renamed or removed, this test fails to compile.
#![allow(unused_imports)]

// Each `use ... as _` pins one public path. Renames/removals break compilation.
use mux::MuxError as _;
use mux::agent::{
    AgentDefinition as _, AgentRegistry as _, FilteredRegistry as _, MemoryTranscriptStore as _,
    Preset as _, RunHandle as _, RunStatus as _, SubAgent as _, SubAgentResult as _,
    TaskTool as _, TranscriptStore as _,
};
use mux::error::{
    LlmError as _, McpError as _, MuxError as _, PermissionError as _, ToolError as _,
};
use mux::llm::{
    AnthropicClient as _, ContentBlock as _, LlmClient as _, Message as _, OpenAIClient as _,
    Request as _, Response as _, Role as _, StopReason as _, StreamEvent as _,
    ToolDefinition as _, Usage as _,
};
use mux::mcp::{
    HttpTransport as _, McpClient as _, McpContentBlock as _, McpLogLevel as _,
    McpPromptGetResult as _, McpPromptInfo as _, McpPromptsListResult as _, McpProxyTool as _,
    McpResourceContent as _, McpResourceInfo as _, McpResourcesListResult as _, McpRoot as _,
    McpSamplingParams as _, McpSamplingResult as _, McpServerCapabilities as _,
    McpServerConfig as _, McpToolInfo as _, McpToolResult as _, McpTransport as _,
    SseTransport as _, StdioTransport as _, Transport as _,
};
use mux::permission::{
    AlwaysApprove as _, AlwaysReject as _, ApprovalContext as _, ApprovalHandler as _,
    Decision as _, Policy as _, PolicyBuilder as _,
};
use mux::prelude::*;
use mux::tool::{Registry as _, Tool as _, ToolExecute as _, ToolResult as _};
use mux::tools::{
    BashTool as _, EditTool as _, ListFilesTool as _, ReadFileTool as _, SearchResult as _,
    SearchTool as _, WebFetchTool as _, WebSearchTool as _, WriteFileTool as _,
};

#[test]
fn public_api_surface_is_stable() {
    // Compilation of the `use … as _` imports above is the assertion for types,
    // traits, and enums. Statics and free functions cannot use `as _` syntax, so
    // they are pinned by reference here instead.
    let _ = &mux::agent::EXPLORER;
    let _ = &mux::agent::PLANNER;
    let _ = &mux::agent::RESEARCHER;
    let _ = &mux::agent::REVIEWER;
    let _ = &mux::agent::WRITER;
    let _ = mux::agent::all_presets;
    let _ = mux::agent::get_preset as fn(&str) -> _;
}
