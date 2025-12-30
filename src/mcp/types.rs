// ABOUTME: Defines MCP protocol types - JSON-RPC 2.0 messages, tool info,
// ABOUTME: and server configuration structures.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl McpRequest {
    /// Create a new request with an auto-incrementing ID.
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: REQUEST_ID.fetch_add(1, Ordering::SeqCst),
            method: method.into(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 notification (no id, no response expected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpNotification {
    pub jsonrpc: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl McpNotification {
    /// Create a new notification.
    pub fn new(method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpRpcError>,
}

/// A JSON-RPC 2.0 error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Information about an MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Result of listing tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolsListResult {
    pub tools: Vec<McpToolInfo>,
}

/// Parameters for calling a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolCallParams {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
}

/// Content block in a tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

/// Result of calling a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub content: Vec<McpContentBlock>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

/// Transport configuration for MCP.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Stdio transport - spawn a subprocess.
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    /// SSE transport - connect to HTTP endpoint with Server-Sent Events.
    Sse { url: String },
    /// HTTP transport - simple request/response over HTTP (Streamable HTTP).
    Http { url: String },
}

/// Configuration for an MCP server.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: McpTransport,
}

/// Client info for MCP handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientInfo {
    pub name: String,
    pub version: String,
}

/// Parameters for MCP initialize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpInitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: serde_json::Value,
    #[serde(rename = "clientInfo")]
    pub client_info: McpClientInfo,
}

/// Server capabilities returned from initialize.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub resources: Option<serde_json::Value>,
    #[serde(default)]
    pub prompts: Option<serde_json::Value>,
}

/// Initialize result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpInitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: McpServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: Option<McpClientInfo>,
}

// ============================================================================
// Resources
// ============================================================================

/// Information about an MCP resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceInfo {
    /// URI identifying this resource.
    pub uri: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional MIME type.
    #[serde(rename = "mimeType", default)]
    pub mime_type: Option<String>,
}

/// Result of listing resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourcesListResult {
    pub resources: Vec<McpResourceInfo>,
    #[serde(rename = "nextCursor", default)]
    pub next_cursor: Option<String>,
}

/// Content of a resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpResourceContent {
    #[serde(rename = "text")]
    Text {
        uri: String,
        #[serde(rename = "mimeType", default)]
        mime_type: Option<String>,
        text: String,
    },
    #[serde(rename = "blob")]
    Blob {
        uri: String,
        #[serde(rename = "mimeType", default)]
        mime_type: Option<String>,
        blob: String, // base64 encoded
    },
}

/// Result of reading a resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceReadResult {
    pub contents: Vec<McpResourceContent>,
}

/// Resource template for dynamic resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceTemplate {
    /// URI template (RFC 6570).
    #[serde(rename = "uriTemplate")]
    pub uri_template: String,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional MIME type.
    #[serde(rename = "mimeType", default)]
    pub mime_type: Option<String>,
}

/// Result of listing resource templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceTemplatesListResult {
    #[serde(rename = "resourceTemplates")]
    pub resource_templates: Vec<McpResourceTemplate>,
    #[serde(rename = "nextCursor", default)]
    pub next_cursor: Option<String>,
}

// ============================================================================
// Prompts
// ============================================================================

/// Information about an MCP prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptInfo {
    /// Unique identifier for this prompt.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Arguments the prompt accepts.
    #[serde(default)]
    pub arguments: Vec<McpPromptArgument>,
}

/// Argument for a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptArgument {
    /// Argument name.
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// Whether this argument is required.
    #[serde(default)]
    pub required: bool,
}

/// Result of listing prompts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptsListResult {
    pub prompts: Vec<McpPromptInfo>,
    #[serde(rename = "nextCursor", default)]
    pub next_cursor: Option<String>,
}

/// Message in a prompt result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptMessage {
    pub role: String,
    pub content: McpPromptContent,
}

/// Content in a prompt message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpPromptContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        data: String,
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
    #[serde(rename = "resource")]
    Resource { resource: McpResourceContent },
}

/// Result of getting a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPromptGetResult {
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// The prompt messages.
    pub messages: Vec<McpPromptMessage>,
}

// ============================================================================
// Roots
// ============================================================================

/// A root directory that the client has access to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRoot {
    /// The root URI (usually file://).
    pub uri: String,
    /// Optional human-readable name.
    #[serde(default)]
    pub name: Option<String>,
}

/// Result of listing roots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRootsListResult {
    pub roots: Vec<McpRoot>,
}

// ============================================================================
// Logging
// ============================================================================

/// Log level for MCP logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpLogLevel {
    Debug,
    Info,
    Notice,
    Warning,
    Error,
    Critical,
    Alert,
    Emergency,
}

/// A log message from the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpLogMessage {
    pub level: McpLogLevel,
    #[serde(default)]
    pub logger: Option<String>,
    pub data: serde_json::Value,
}

// ============================================================================
// Progress
// ============================================================================

/// Progress notification for long-running operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpProgress {
    /// Progress token identifying the operation.
    #[serde(rename = "progressToken")]
    pub progress_token: serde_json::Value,
    /// Progress value (0.0 to 1.0 or count).
    pub progress: f64,
    /// Optional total value for the operation.
    #[serde(default)]
    pub total: Option<f64>,
}

// ============================================================================
// Sampling (for servers that support model sampling)
// ============================================================================

/// Message for sampling request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingMessage {
    pub role: String,
    pub content: McpPromptContent,
}

/// Parameters for a sampling request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingParams {
    pub messages: Vec<McpSamplingMessage>,
    #[serde(rename = "modelPreferences", default)]
    pub model_preferences: Option<McpModelPreferences>,
    #[serde(rename = "systemPrompt", default)]
    pub system_prompt: Option<String>,
    #[serde(rename = "includeContext", default)]
    pub include_context: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u32,
    #[serde(rename = "stopSequences", default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Model preferences for sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelPreferences {
    #[serde(default)]
    pub hints: Option<Vec<McpModelHint>>,
    #[serde(rename = "costPriority", default)]
    pub cost_priority: Option<f64>,
    #[serde(rename = "speedPriority", default)]
    pub speed_priority: Option<f64>,
    #[serde(rename = "intelligencePriority", default)]
    pub intelligence_priority: Option<f64>,
}

/// Hint for model selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpModelHint {
    #[serde(default)]
    pub name: Option<String>,
}

/// Result of a sampling request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpSamplingResult {
    pub role: String,
    pub content: McpPromptContent,
    pub model: String,
    #[serde(rename = "stopReason", default)]
    pub stop_reason: Option<String>,
}
