// ABOUTME: Defines the policy engine - rules, decisions, and evaluation.
// ABOUTME: Supports globs, conditionals, and default policies.

use std::sync::Arc;

/// The decision made by a policy rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Allow the tool execution.
    Allow,
    /// Deny the tool execution.
    Deny,
    /// Ask the user for approval.
    Ask,
}

/// A condition function for conditional rules.
pub type ConditionFn = Arc<dyn Fn(&serde_json::Value) -> Decision + Send + Sync>;

/// A rule in the policy.
pub enum PolicyRule {
    /// Allow a specific tool by exact name.
    Allow(String),

    /// Deny a specific tool by exact name.
    Deny(String),

    /// Allow tools matching a glob pattern.
    AllowPattern(glob::Pattern),

    /// Deny tools matching a glob pattern.
    DenyPattern(glob::Pattern),

    /// Conditional rule based on parameters.
    Conditional {
        tool: String,
        condition: ConditionFn,
    },
}

/// A policy that evaluates tool execution requests.
pub struct Policy {
    rules: Vec<PolicyRule>,
    default: Decision,
}

impl Policy {
    /// Create a new policy builder.
    pub fn builder() -> PolicyBuilder {
        PolicyBuilder::new()
    }

    /// Evaluate whether a tool should be allowed.
    pub fn evaluate(&self, tool: &str, params: &serde_json::Value) -> Decision {
        for rule in &self.rules {
            match rule {
                PolicyRule::Allow(name) if name == tool => return Decision::Allow,
                PolicyRule::Deny(name) if name == tool => return Decision::Deny,
                PolicyRule::AllowPattern(pattern) if pattern.matches(tool) => {
                    return Decision::Allow
                }
                PolicyRule::DenyPattern(pattern) if pattern.matches(tool) => return Decision::Deny,
                PolicyRule::Conditional { tool: t, condition } if t == tool => {
                    return condition(params)
                }
                _ => continue,
            }
        }
        self.default
    }
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            default: Decision::Deny,
        }
    }
}

/// Builder for constructing policies.
pub struct PolicyBuilder {
    rules: Vec<PolicyRule>,
    default: Decision,
}

impl Default for PolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyBuilder {
    /// Create a new builder with default deny.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default: Decision::Deny,
        }
    }

    /// Allow a tool by exact name.
    pub fn allow(mut self, tool: impl Into<String>) -> Self {
        self.rules.push(PolicyRule::Allow(tool.into()));
        self
    }

    /// Deny a tool by exact name.
    pub fn deny(mut self, tool: impl Into<String>) -> Self {
        self.rules.push(PolicyRule::Deny(tool.into()));
        self
    }

    /// Allow tools matching a glob pattern.
    pub fn allow_pattern(mut self, pattern: &str) -> Self {
        if let Ok(p) = glob::Pattern::new(pattern) {
            self.rules.push(PolicyRule::AllowPattern(p));
        }
        self
    }

    /// Deny tools matching a glob pattern.
    pub fn deny_pattern(mut self, pattern: &str) -> Self {
        if let Ok(p) = glob::Pattern::new(pattern) {
            self.rules.push(PolicyRule::DenyPattern(p));
        }
        self
    }

    /// Add a conditional rule.
    pub fn conditional<F>(mut self, tool: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&serde_json::Value) -> Decision + Send + Sync + 'static,
    {
        self.rules.push(PolicyRule::Conditional {
            tool: tool.into(),
            condition: Arc::new(condition),
        });
        self
    }

    /// Set the default decision for unmatched tools.
    pub fn default(mut self, decision: Decision) -> Self {
        self.default = decision;
        self
    }

    /// Build the policy.
    pub fn build(self) -> Policy {
        Policy {
            rules: self.rules,
            default: self.default,
        }
    }
}
