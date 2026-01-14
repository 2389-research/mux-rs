// ABOUTME: Context management types and token estimation utilities.
// ABOUTME: Supports small context models like Apple Foundation Models (4K).

/// Approximate bytes per token for estimation (conservative)
pub const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Safety margin to avoid hitting exact limit
pub const SAFETY_MARGIN: f32 = 0.8;

/// Threshold for auto-selecting compaction strategy.
/// Models with context_limit <= this use truncation (fast, no LLM overhead).
/// Models with context_limit > this use summarization (intelligent compression).
pub const SMALL_CONTEXT_THRESHOLD: u32 = 8192;

/// System prompt for LLM summarization during compaction.
pub const SUMMARIZATION_PROMPT: &str = r#"You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."#;

/// Prefix added to summaries when storing as assistant message.
pub const SUMMARY_PREFIX: &str = r#"Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"#;

/// Context usage statistics for a conversation
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct ContextUsage {
    pub message_count: u32,
    pub estimated_tokens: u32,
    pub context_limit: Option<u32>,
    pub usage_percent: Option<f32>,
}

impl ContextUsage {
    pub fn new(message_count: u32, estimated_tokens: u32, context_limit: Option<u32>) -> Self {
        let usage_percent = context_limit.map(|limit| {
            if limit > 0 {
                (estimated_tokens as f32 / limit as f32) * 100.0
            } else {
                0.0
            }
        });
        Self {
            message_count,
            estimated_tokens,
            context_limit,
            usage_percent,
        }
    }
}

/// How to handle context overflow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, uniffi::Enum)]
pub enum CompactionMode {
    /// Ask LLM to summarize old messages (default for large context)
    #[default]
    Summarize,
    /// Drop oldest messages, keep recent (for small context models)
    TruncateOldest,
}

/// Per-model context configuration
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelContextConfig {
    pub model: String,
    pub context_limit: u32,
    /// Deprecated: compaction mode is now auto-selected based on context_limit.
    /// Small context (<=8K) uses truncation, large context uses summarization.
    pub compaction_mode: CompactionMode,
    pub warning_threshold: f32,
    /// Optional model to use for summarization (e.g., cheaper model like Haiku).
    /// If None, uses the conversation's configured model.
    ///
    /// CAVEAT: The compaction_model uses the engine's default provider, not a
    /// per-model provider. If you set compaction_model to a model from a different
    /// provider (e.g., "claude-3-haiku" when default provider is OpenAI), the
    /// summarization will fail. Ensure the compaction_model is compatible with
    /// your configured provider.
    pub compaction_model: Option<String>,
}

impl ModelContextConfig {
    pub fn new(model: String, context_limit: u32) -> Self {
        Self {
            model,
            context_limit,
            compaction_mode: CompactionMode::default(),
            warning_threshold: 0.8,
            compaction_model: None,
        }
    }

    /// Deprecated: compaction mode is now auto-selected based on context_limit.
    pub fn with_compaction_mode(mut self, mode: CompactionMode) -> Self {
        self.compaction_mode = mode;
        self
    }

    pub fn with_warning_threshold(mut self, threshold: f32) -> Self {
        self.warning_threshold = threshold;
        self
    }

    /// Set a separate model for summarization (e.g., cheaper model like Haiku).
    pub fn with_compaction_model(mut self, model: String) -> Self {
        self.compaction_model = Some(model);
        self
    }
}

/// Estimate token count from text using byte-based heuristic
pub fn estimate_tokens(text: &str) -> u32 {
    let bytes = text.len();
    ((bytes + APPROX_BYTES_PER_TOKEN - 1) / APPROX_BYTES_PER_TOKEN) as u32
}

/// Calculate effective limit with safety margin.
/// Intentionally truncates (not rounds) to be conservative - we'd rather
/// have slightly more headroom than risk hitting the exact limit.
pub fn effective_limit(limit: u32) -> u32 {
    (limit as f32 * SAFETY_MARGIN) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_short() {
        // 4 bytes = 1 token
        assert_eq!(estimate_tokens("test"), 1);
    }

    #[test]
    fn test_estimate_tokens_longer() {
        // 12 bytes = 3 tokens
        assert_eq!(estimate_tokens("hello world!"), 3);
    }

    #[test]
    fn test_effective_limit() {
        // 4096 * 0.8 = 3276
        assert_eq!(effective_limit(4096), 3276);
    }

    #[test]
    fn test_context_usage_percent() {
        let usage = ContextUsage::new(5, 2000, Some(4096));
        assert!(usage.usage_percent.is_some());
        let percent = usage
            .usage_percent
            .expect("usage_percent should be Some when context_limit > 0");
        assert!(percent > 48.0 && percent < 49.0);
    }

    #[test]
    fn test_context_usage_no_limit() {
        let usage = ContextUsage::new(5, 2000, None);
        assert!(usage.usage_percent.is_none());
    }
}
