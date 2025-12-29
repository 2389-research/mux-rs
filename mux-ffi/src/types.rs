// ABOUTME: Core data types for mux-ffi, exposed to Swift via UniFFI.
// ABOUTME: These mirror the design doc's data model.

use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum Provider {
    Anthropic,
    OpenAI,
}

#[derive(Debug, Clone, uniffi::Record)]
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

#[derive(Debug, Clone, uniffi::Record)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub path: Option<String>,
    pub llm_config: Option<LlmConfig>,
}

impl Workspace {
    pub fn new(name: String, path: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            path,
            llm_config: None,
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

#[derive(Debug, Clone, uniffi::Record)]
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

#[derive(Debug, Clone, uniffi::Record)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub enabled: bool,
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
