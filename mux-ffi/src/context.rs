// ABOUTME: Context management types and token estimation utilities.
// ABOUTME: Supports small context models like Apple Foundation Models (4K).

/// Approximate bytes per token for estimation (conservative)
pub const APPROX_BYTES_PER_TOKEN: usize = 4;

/// Safety margin to avoid hitting exact limit
pub const SAFETY_MARGIN: f32 = 0.8;

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
    pub compaction_mode: CompactionMode,
    pub warning_threshold: f32,
}

impl ModelContextConfig {
    pub fn new(model: String, context_limit: u32) -> Self {
        Self {
            model,
            context_limit,
            compaction_mode: CompactionMode::default(),
            warning_threshold: 0.8,
        }
    }

    pub fn with_compaction_mode(mut self, mode: CompactionMode) -> Self {
        self.compaction_mode = mode;
        self
    }

    pub fn with_warning_threshold(mut self, threshold: f32) -> Self {
        self.warning_threshold = threshold;
        self
    }
}

/// Estimate token count from text using byte-based heuristic
pub fn estimate_tokens(text: &str) -> u32 {
    let bytes = text.len();
    ((bytes + APPROX_BYTES_PER_TOKEN - 1) / APPROX_BYTES_PER_TOKEN) as u32
}

/// Calculate effective limit with safety margin
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
        let percent = usage.usage_percent.unwrap();
        assert!(percent > 48.0 && percent < 49.0);
    }

    #[test]
    fn test_context_usage_no_limit() {
        let usage = ContextUsage::new(5, 2000, None);
        assert!(usage.usage_percent.is_none());
    }
}
