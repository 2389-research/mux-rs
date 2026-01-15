// ABOUTME: Core data types for mux-ffi, exposed to Swift via UniFFI.
// ABOUTME: These mirror the design doc's data model.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    Ollama,
    /// Custom callback-based provider (e.g., Apple Foundation Models)
    Custom { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct LlmConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key_ref: Option<String>,
}

impl LlmConfig {
    pub fn anthropic(model: String) -> Self {
        Self {
            provider: Provider::Anthropic,
            model,
            api_key_ref: None,
        }
    }

    pub fn openai(model: String) -> Self {
        Self {
            provider: Provider::OpenAI,
            model,
            api_key_ref: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub path: Option<String>,
    pub llm_config: Option<LlmConfig>,
    pub mcp_servers: Vec<McpServerConfig>,
    /// Custom system prompt for this workspace. If None, a default is used.
    /// Tool guidance is automatically appended to whatever prompt is set.
    pub system_prompt: Option<String>,
}

impl Workspace {
    pub fn new(name: String, path: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            path,
            llm_config: None,
            mcp_servers: Vec::new(),
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct WorkspaceSummary {
    pub id: String,
    pub name: String,
    pub path: Option<String>,
    pub conversation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct Conversation {
    pub id: String,
    pub workspace_id: String,
    pub title: String,
    pub created_at: u64,
}

impl Conversation {
    pub fn new(workspace_id: String, title: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            workspace_id,
            title,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, uniffi::Enum)]
pub enum McpTransportType {
    Stdio,
    Sse,
}

#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct McpServerConfig {
    pub name: String,
    pub transport_type: McpTransportType,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub url: Option<String>,
    pub enabled: bool,
}

impl McpServerConfig {
    pub fn stdio(name: String, command: String, args: Vec<String>) -> Self {
        Self {
            name,
            transport_type: McpTransportType::Stdio,
            command: Some(command),
            args,
            url: None,
            enabled: true,
        }
    }

    pub fn sse(name: String, url: String) -> Self {
        Self {
            name,
            transport_type: McpTransportType::Sse,
            command: None,
            args: Vec::new(),
            url: Some(url),
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub server_name: String,
}

// ============================================================================
// MCP Resource Types
// ============================================================================

/// Information about an MCP resource.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpResourceInfo {
    pub uri: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub server_name: String,
}

/// Content returned when reading an MCP resource.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum McpResourceContent {
    Text {
        uri: String,
        mime_type: Option<String>,
        text: String,
    },
    Blob {
        uri: String,
        mime_type: Option<String>,
        blob: String, // base64 encoded
    },
}

/// A parameterized resource template from an MCP server.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpResourceTemplate {
    pub uri_template: String,
    pub name: String,
    pub description: Option<String>,
    pub mime_type: Option<String>,
    pub server_name: String,
}

// ============================================================================
// MCP Prompt Types
// ============================================================================

/// Information about an MCP prompt.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpPromptInfo {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Vec<McpPromptArgument>,
    pub server_name: String,
}

/// An argument definition for an MCP prompt.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpPromptArgument {
    pub name: String,
    pub description: Option<String>,
    pub required: bool,
}

/// A message returned from an MCP prompt.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: String,
}

/// Result of getting an MCP prompt.
#[derive(Debug, Clone, uniffi::Record)]
pub struct McpPromptResult {
    pub description: Option<String>,
    pub messages: Vec<McpPromptMessage>,
}

/// FFI-friendly argument value for prompt execution (alternative to HashMap).
/// Distinct from McpPromptArgument which defines what arguments a prompt accepts.
#[derive(Debug, Clone, uniffi::Record)]
pub struct PromptArgumentValue {
    pub name: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum ApprovalDecision {
    Allow,
    AlwaysAllow,
    Deny,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentConfig {
    pub name: String,
    pub system_prompt: String,
    pub model: Option<String>,
    pub allowed_tools: Vec<String>,
    pub denied_tools: Vec<String>,
    pub max_iterations: u32,
}

impl AgentConfig {
    pub fn new(name: String, system_prompt: String) -> Self {
        Self {
            name,
            system_prompt,
            model: None,
            allowed_tools: Vec::new(),
            denied_tools: Vec::new(),
            max_iterations: 10,
        }
    }
}

#[derive(Debug, Clone, uniffi::Enum)]
pub enum HookEventType {
    PreToolUse { tool_name: String, input: String },
    PostToolUse { tool_name: String, input: String, result: String },
    AgentStart { agent_id: String, task: String },
    AgentStop { agent_id: String },
    Iteration { agent_id: String, iteration: u32 },
}

#[derive(Debug, Clone, uniffi::Enum)]
pub enum HookResponse {
    Continue,
    Block { reason: String },
    Transform { new_input: String },
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct TranscriptData {
    pub agent_id: String,
    pub messages_json: String,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct SubagentResult {
    pub agent_id: String,
    pub content: String,
    pub tool_use_count: u32,
    pub iterations: u32,
    pub transcript_json: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct ToolExecutionResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolExecutionResult {
    pub fn success(content: String) -> Self {
        Self { content, is_error: false }
    }
    pub fn error(content: String) -> Self {
        Self { content, is_error: true }
    }
}

// ============================================================================
// Callback LLM Provider Types
// ============================================================================

/// Role of a chat message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ChatRole {
    User,
    Assistant,
}

/// A chat message for LLM context.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Tool definition for FFI - UniFFI-safe version of ToolDefinition.
#[derive(Debug, Clone, uniffi::Record)]
pub struct FfiToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON string of the input schema (serde_json::Value isn't FFI-safe)
    pub input_schema_json: String,
}

/// Usage statistics from LLM generation.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Request for LLM generation via callback provider.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LlmRequest {
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<FfiToolDefinition>,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
}

/// Tool call from the LLM.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LlmToolCall {
    pub id: String,
    pub name: String,
    /// JSON string of the tool arguments
    pub arguments: String,
}

/// Response from LLM generation via callback provider.
#[derive(Debug, Clone, uniffi::Record)]
pub struct LlmResponse {
    pub text: String,
    pub tool_calls: Vec<LlmToolCall>,
    pub usage: LlmUsage,
    /// Error message if generation failed
    pub error: Option<String>,
}
