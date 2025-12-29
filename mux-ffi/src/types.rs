// ABOUTME: Core data types for mux-ffi, exposed to Swift via UniFFI.
// ABOUTME: These mirror the design doc's data model.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,
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
}

impl Workspace {
    pub fn new(name: String, path: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            path,
            llm_config: None,
            mcp_servers: Vec::new(),
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

#[derive(Debug, Clone, uniffi::Enum)]
pub enum ApprovalDecision {
    Allow,
    AlwaysAllow,
    Deny,
}
