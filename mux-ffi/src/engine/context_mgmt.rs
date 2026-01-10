// ABOUTME: Context management for MuxEngine - token estimation, compaction, usage tracking.
// ABOUTME: Supports small context models by tracking and managing conversation size.

use super::MuxEngine;
use crate::callback::ChatCallback;
use crate::context::{effective_limit, estimate_tokens, CompactionMode, ContextUsage, ModelContextConfig};
use crate::MuxFfiError;

/// Context configuration and usage methods
impl MuxEngine {
    /// Set context configuration for a specific model.
    /// Call this during app init to configure small context models.
    #[uniffi::method]
    pub fn set_model_context_config(&self, config: ModelContextConfig) {
        self.model_context_configs
            .write()
            .insert(config.model.clone(), config);
    }

    /// Get context configuration for a model.
    /// Returns a default config if not explicitly set.
    #[uniffi::method]
    pub fn get_model_context_config(&self, model: String) -> ModelContextConfig {
        self.model_context_configs
            .read()
            .get(&model)
            .cloned()
            .unwrap_or_else(|| ModelContextConfig {
                model,
                context_limit: 0, // 0 means no limit
                compaction_mode: CompactionMode::Summarize,
                warning_threshold: 0.8,
            })
    }

    /// Get current context usage for a conversation.
    #[uniffi::method]
    pub fn get_context_usage(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError> {
        // Check conversation exists
        let conversations = self.conversations.read();
        let exists = conversations
            .values()
            .flatten()
            .any(|c| c.id == conversation_id);
        if !exists {
            return Err(MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            });
        }
        drop(conversations);

        let (message_count, estimated_tokens) = self.estimate_conversation_tokens(&conversation_id);

        // Get workspace to find model - extract model name first, then release lock
        let workspace_id = self.get_workspace_for_conversation(&conversation_id);
        let model_name = workspace_id.as_ref().and_then(|ws_id| {
            let workspaces = self.workspaces.read();
            workspaces
                .get(ws_id)
                .and_then(|ws| ws.llm_config.as_ref().map(|c| c.model.clone()))
        });

        // Now get context limit from model config (separate lock acquisition)
        let context_limit = model_name.and_then(|model| {
            let model_configs = self.model_context_configs.read();
            model_configs.get(&model).map(|c| c.context_limit)
        });

        Ok(ContextUsage::new(message_count, estimated_tokens, context_limit))
    }

    /// Clear conversation context (start fresh, keep conversation ID).
    /// Useful when context is full and user wants to start over.
    #[uniffi::method]
    pub fn clear_context(&self, conversation_id: String) -> Result<(), MuxFfiError> {
        // Check conversation exists
        let conversations = self.conversations.read();
        let exists = conversations
            .values()
            .flatten()
            .any(|c| c.id == conversation_id);
        if !exists {
            return Err(MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            });
        }
        drop(conversations);

        // Clear message history
        {
            let mut history = self.message_history.write();
            if let Some(messages) = history.get_mut(&conversation_id) {
                messages.clear();
            }
        }

        // Persist cleared state
        self.save_messages(&conversation_id);

        Ok(())
    }

    /// Manually trigger context compaction.
    /// Uses the configured compaction mode for the conversation's model.
    /// For TruncateOldest: drops oldest messages to fit in limit.
    /// For Summarize: (TODO) asks LLM to summarize old messages - currently falls back to truncate.
    #[uniffi::method]
    pub fn compact_context(&self, conversation_id: String) -> Result<ContextUsage, MuxFfiError> {
        // Check conversation exists and get workspace
        let workspace_id = self
            .get_workspace_for_conversation(&conversation_id)
            .ok_or_else(|| MuxFfiError::Engine {
                message: format!("Conversation not found: {}", conversation_id),
            })?;

        // Get model config
        let (_model, config) = {
            let workspaces = self.workspaces.read();
            let ws = workspaces
                .get(&workspace_id)
                .ok_or_else(|| MuxFfiError::Engine {
                    message: "Workspace not found".to_string(),
                })?;

            let model = ws
                .llm_config
                .as_ref()
                .map(|c| c.model.clone())
                .unwrap_or_default();

            let model_configs = self.model_context_configs.read();
            let config = model_configs.get(&model).cloned();
            (model, config)
        };

        // If no config or no limit, nothing to compact
        let config = match config {
            Some(c) if c.context_limit > 0 => c,
            _ => return self.get_context_usage(conversation_id),
        };

        let target_tokens = effective_limit(config.context_limit);

        match config.compaction_mode {
            CompactionMode::TruncateOldest => {
                self.truncate_oldest(&conversation_id, target_tokens);
                self.save_messages(&conversation_id);
            }
            CompactionMode::Summarize => {
                // TODO: Implement LLM-based summarization
                // For now, fall back to truncate with warning
                eprintln!(
                    "Warning: CompactionMode::Summarize not yet implemented, falling back to TruncateOldest"
                );
                self.truncate_oldest(&conversation_id, target_tokens);
                self.save_messages(&conversation_id);
            }
        }

        self.get_context_usage(conversation_id)
    }
}

/// Context helper methods
impl MuxEngine {
    /// Estimate total tokens in a conversation's message history
    pub(super) fn estimate_conversation_tokens(&self, conversation_id: &str) -> (u32, u32) {
        let history = self.message_history.read();
        if let Some(messages) = history.get(conversation_id) {
            let message_count = messages.len() as u32;
            let total_tokens: u32 = messages
                .iter()
                .map(|m| {
                    m.content
                        .iter()
                        .map(|block| match block {
                            mux::llm::ContentBlock::Text { text } => estimate_tokens(text),
                            mux::llm::ContentBlock::ToolUse { input, .. } => {
                                estimate_tokens(&input.to_string())
                            }
                            mux::llm::ContentBlock::ToolResult { content, .. } => {
                                estimate_tokens(content)
                            }
                        })
                        .sum::<u32>()
                })
                .sum();
            (message_count, total_tokens)
        } else {
            (0, 0)
        }
    }

    /// Get the configured model for a conversation's workspace
    pub(super) fn get_model_for_conversation(&self, conversation_id: &str) -> Option<String> {
        let workspace_id = self.get_workspace_for_conversation(conversation_id)?;
        let workspaces = self.workspaces.read();
        workspaces
            .get(&workspace_id)
            .and_then(|ws| ws.llm_config.as_ref())
            .map(|cfg| cfg.model.clone())
    }

    /// Get the workspace ID for a conversation.
    pub(super) fn get_workspace_for_conversation(&self, conversation_id: &str) -> Option<String> {
        let conversations = self.conversations.read();
        conversations.iter().find_map(|(ws_id, convs)| {
            if convs.iter().any(|c| c.id == conversation_id) {
                Some(ws_id.clone())
            } else {
                None
            }
        })
    }

    /// Check context usage and fire warning callback if above threshold.
    /// Returns true if warning was fired.
    pub(super) fn check_and_warn_context(
        &self,
        conversation_id: &str,
        callback: &dyn ChatCallback,
    ) -> bool {
        let usage = match self.get_context_usage(conversation_id.to_string()) {
            Ok(u) => u,
            Err(_) => return false,
        };

        // Get model config for threshold
        let threshold = self.get_warning_threshold_for_conversation(conversation_id);

        if let Some(percent) = usage.usage_percent {
            if percent >= threshold * 100.0 {
                callback.on_context_warning(usage);
                return true;
            }
        }

        false
    }

    /// Get warning threshold for a conversation's configured model.
    fn get_warning_threshold_for_conversation(&self, conversation_id: &str) -> f32 {
        let model = self.get_model_for_conversation(conversation_id);
        model
            .and_then(|m| {
                self.model_context_configs
                    .read()
                    .get(&m)
                    .map(|c| c.warning_threshold)
            })
            .unwrap_or(0.8)
    }

    /// Truncate oldest messages to fit within token limit.
    /// Keeps most recent messages, drops oldest.
    ///
    /// NOTE: This simple truncation may break ToolUse/ToolResult pairs if the cut
    /// happens mid-sequence. For small context models where truncation is the primary
    /// compaction strategy, consider keeping conversations short or using clear_context
    /// between major tasks to avoid orphaned tool calls.
    pub(super) fn truncate_oldest(&self, conversation_id: &str, target_tokens: u32) {
        let mut history = self.message_history.write();
        if let Some(messages) = history.get_mut(conversation_id) {
            // Calculate tokens from end, keep messages that fit
            let mut kept_tokens = 0u32;
            let mut keep_from = messages.len();

            for (i, msg) in messages.iter().enumerate().rev() {
                let msg_tokens: u32 = msg
                    .content
                    .iter()
                    .map(|block| match block {
                        mux::llm::ContentBlock::Text { text } => estimate_tokens(text),
                        mux::llm::ContentBlock::ToolUse { input, .. } => {
                            estimate_tokens(&input.to_string())
                        }
                        mux::llm::ContentBlock::ToolResult { content, .. } => {
                            estimate_tokens(content)
                        }
                    })
                    .sum();

                if kept_tokens + msg_tokens <= target_tokens {
                    kept_tokens += msg_tokens;
                    keep_from = i;
                } else {
                    break;
                }
            }

            // Remove messages before keep_from
            if keep_from > 0 {
                messages.drain(0..keep_from);
            }
        }
    }
}
