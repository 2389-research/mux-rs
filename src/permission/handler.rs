// ABOUTME: Defines the ApprovalHandler trait for async user approval.
// ABOUTME: Called when policy returns Decision::Ask.

use async_trait::async_trait;

/// Context provided to approval handlers.
#[derive(Debug, Clone)]
pub struct ApprovalContext {
    /// Description of the tool being executed.
    pub tool_description: String,

    /// Unique identifier for this approval request.
    pub request_id: String,
}

/// Trait for handling approval requests.
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Request approval for a tool execution.
    ///
    /// Returns `Ok(true)` if approved, `Ok(false)` if rejected.
    async fn request_approval(
        &self,
        tool: &str,
        params: &serde_json::Value,
        context: &ApprovalContext,
    ) -> Result<bool, anyhow::Error>;
}

/// An approval handler that always approves.
pub struct AlwaysApprove;

#[async_trait]
impl ApprovalHandler for AlwaysApprove {
    async fn request_approval(
        &self,
        _tool: &str,
        _params: &serde_json::Value,
        _context: &ApprovalContext,
    ) -> Result<bool, anyhow::Error> {
        Ok(true)
    }
}

/// An approval handler that always rejects.
pub struct AlwaysReject;

#[async_trait]
impl ApprovalHandler for AlwaysReject {
    async fn request_approval(
        &self,
        _tool: &str,
        _params: &serde_json::Value,
        _context: &ApprovalContext,
    ) -> Result<bool, anyhow::Error> {
        Ok(false)
    }
}
