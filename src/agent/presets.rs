// ABOUTME: Provides preset agent configurations for common patterns.
// ABOUTME: Includes Explorer, Planner, Researcher, Writer, and Reviewer presets with sensible defaults.

use super::AgentDefinition;

/// Pre-configured agent template with sensible defaults.
#[derive(Debug, Clone)]
pub struct Preset {
    /// Default name for agents using this preset.
    pub name: &'static str,

    /// Specialized system prompt for this agent type.
    pub system_prompt: &'static str,

    /// Restricts the agent to these tools only.
    /// None means all tools are allowed.
    pub allowed_tools: Option<&'static [&'static str]>,

    /// Prevents the agent from using these tools.
    pub denied_tools: &'static [&'static str],

    /// Override for the default iteration limit.
    pub max_iterations: usize,
}

impl Preset {
    /// Apply preset defaults to an AgentDefinition.
    /// Only overrides fields that are empty/default in the AgentDefinition.
    /// If AgentDefinition already has a value, keeps it (preset is just defaults).
    pub fn apply(&self, mut def: AgentDefinition) -> AgentDefinition {
        // Apply name if agent_type is empty
        if def.agent_type.is_empty() {
            def.agent_type = self.name.to_string();
        }

        // Apply system prompt if empty
        if def.system_prompt.is_empty() {
            def.system_prompt = self.system_prompt.to_string();
        }

        // Apply allowed_tools if not set in definition and preset has them
        if let (None, Some(tools)) = (&def.allowed_tools, self.allowed_tools) {
            def.allowed_tools = Some(tools.iter().map(|s| s.to_string()).collect());
        }

        // Apply denied_tools if definition's list is empty and preset has them
        if def.denied_tools.is_empty() && !self.denied_tools.is_empty() {
            def.denied_tools = self.denied_tools.iter().map(|s| s.to_string()).collect();
        }

        // Apply max_iterations if definition uses default (10).
        // The value 10 is coupled to AgentDefinition::default()'s max_iterations.
        if def.max_iterations == 10 {
            def.max_iterations = self.max_iterations;
        }

        def
    }
}

/// Read-only tools available to exploration and analysis presets.
const READONLY_TOOLS: &[&str] = &[
    "read_file",
    "read",
    "glob",
    "grep",
    "search",
    "list_files",
    "list_directory",
    "ls",
];

/// Researcher tools - includes readonly plus web access.
const RESEARCHER_TOOLS: &[&str] = &[
    "read_file",
    "read",
    "glob",
    "grep",
    "search",
    "list_files",
    "list_directory",
    "ls",
    "web_search",
    "web_fetch",
    "fetch_url",
];

/// Explorer preset - optimized for codebase exploration.
/// Uses read-only tools and a system prompt focused on finding information.
pub static EXPLORER: Preset = Preset {
    name: "explorer",
    system_prompt: r#"You are a codebase explorer. Your job is to find information efficiently.

Guidelines:
- Use search tools (grep, glob) to locate relevant files
- Read files to understand structure and patterns
- Report findings clearly and concisely
- Do not modify any files
- Focus on answering the specific question asked"#,
    allowed_tools: Some(READONLY_TOOLS),
    denied_tools: &[],
    max_iterations: 20,
};

/// Planner preset - optimized for planning and design.
/// Uses read-only tools and a system prompt focused on architecture.
pub static PLANNER: Preset = Preset {
    name: "planner",
    system_prompt: r#"You are a software architect and planner. Your job is to design implementation approaches.

Guidelines:
- Analyze existing code structure before proposing changes
- Create step-by-step implementation plans
- Consider edge cases and error handling
- Identify dependencies and potential conflicts
- Do not implement - only plan
- Output structured, actionable plans"#,
    allowed_tools: Some(READONLY_TOOLS),
    denied_tools: &[],
    max_iterations: 30,
};

/// Researcher preset - optimized for multi-source research.
/// Can access web and codebase for comprehensive research.
pub static RESEARCHER: Preset = Preset {
    name: "researcher",
    system_prompt: r#"You are a researcher. Your job is to gather and synthesize information from multiple sources.

Guidelines:
- Search the codebase for existing patterns and implementations
- Use web search for external documentation and best practices
- Cross-reference findings from multiple sources
- Cite sources for your findings
- Summarize key findings clearly
- Note any contradictions or uncertainties"#,
    allowed_tools: Some(RESEARCHER_TOOLS),
    denied_tools: &[],
    max_iterations: 40,
};

/// Writer preset - optimized for writing and editing files.
/// Has full write access with a focused system prompt.
pub static WRITER: Preset = Preset {
    name: "writer",
    system_prompt: r#"You are a code writer. Your job is to implement changes based on specifications.

Guidelines:
- Follow the existing code style and patterns
- Write clean, readable, well-documented code
- Handle errors appropriately
- Write tests for new functionality
- Make minimal, focused changes
- Explain significant design decisions"#,
    allowed_tools: None, // Unrestricted - needs write access
    denied_tools: &[],
    max_iterations: 50,
};

/// Reviewer preset - optimized for code review.
/// Read-only access with a review-focused system prompt.
pub static REVIEWER: Preset = Preset {
    name: "reviewer",
    system_prompt: r#"You are a code reviewer. Your job is to analyze code for issues and improvements.

Guidelines:
- Check for bugs, security issues, and performance problems
- Verify code follows project conventions
- Look for edge cases and error handling gaps
- Suggest specific improvements with examples
- Be constructive and actionable
- Prioritize issues by severity"#,
    allowed_tools: Some(READONLY_TOOLS),
    denied_tools: &[],
    max_iterations: 30,
};

/// Returns all available presets.
pub fn all_presets() -> [&'static Preset; 5] {
    [&EXPLORER, &PLANNER, &RESEARCHER, &WRITER, &REVIEWER]
}

/// Look up a preset by name.
pub fn get_preset(name: &str) -> Option<&'static Preset> {
    match name {
        "explorer" => Some(&EXPLORER),
        "planner" => Some(&PLANNER),
        "researcher" => Some(&RESEARCHER),
        "writer" => Some(&WRITER),
        "reviewer" => Some(&REVIEWER),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_apply_empty_definition() {
        let def = AgentDefinition::new("", "");
        let result = EXPLORER.apply(def);

        assert_eq!(result.agent_type, "explorer");
        assert!(!result.system_prompt.is_empty());
        assert!(result.allowed_tools.is_some());
        assert_eq!(result.allowed_tools.as_ref().unwrap().len(), 8);
        assert_eq!(result.max_iterations, 20);
    }

    #[test]
    fn test_preset_apply_with_overrides() {
        let def = AgentDefinition::new("custom-name", "custom prompt").max_iterations(100);
        let result = EXPLORER.apply(def);

        // Custom values should be preserved
        assert_eq!(result.agent_type, "custom-name");
        assert_eq!(result.system_prompt, "custom prompt");
        assert_eq!(result.max_iterations, 100);
        // Allowed tools should still come from preset since base didn't specify
        assert!(result.allowed_tools.is_some());
    }

    #[test]
    fn test_preset_apply_partial_overrides() {
        let def = AgentDefinition::new("my-explorer", "");
        let result = EXPLORER.apply(def);

        // Name should be kept, prompt should come from preset
        assert_eq!(result.agent_type, "my-explorer");
        assert!(!result.system_prompt.is_empty());
        assert!(result.system_prompt.contains("codebase explorer"));
    }

    #[test]
    fn test_preset_apply_preserves_allowed_tools() {
        let def = AgentDefinition::new("", "").allowed_tools(vec!["only_this_tool".to_string()]);
        let result = EXPLORER.apply(def);

        // Definition's allowed_tools should be preserved
        assert_eq!(
            result.allowed_tools,
            Some(vec!["only_this_tool".to_string()])
        );
    }

    #[test]
    fn test_preset_apply_preserves_denied_tools() {
        let def = AgentDefinition::new("", "").denied_tools(vec!["blocked_tool".to_string()]);
        let result = EXPLORER.apply(def);

        // Definition's denied_tools should be preserved
        assert_eq!(result.denied_tools, vec!["blocked_tool".to_string()]);
    }

    #[test]
    fn test_explorer_preset() {
        assert_eq!(EXPLORER.name, "explorer");
        assert!(!EXPLORER.system_prompt.is_empty());
        assert!(EXPLORER.allowed_tools.is_some());
        assert!(!EXPLORER.allowed_tools.unwrap().is_empty());
        assert_eq!(EXPLORER.max_iterations, 20);
    }

    #[test]
    fn test_planner_preset() {
        assert_eq!(PLANNER.name, "planner");
        assert!(!PLANNER.system_prompt.is_empty());
        assert!(PLANNER.system_prompt.contains("architect"));
        assert_eq!(PLANNER.max_iterations, 30);
    }

    #[test]
    fn test_researcher_preset() {
        assert_eq!(RESEARCHER.name, "researcher");
        assert!(!RESEARCHER.system_prompt.is_empty());
        // Researcher should have web access
        let tools = RESEARCHER.allowed_tools.unwrap();
        assert!(tools.contains(&"web_search"));
        assert!(tools.contains(&"web_fetch"));
        assert_eq!(RESEARCHER.max_iterations, 40);
    }

    #[test]
    fn test_writer_preset() {
        assert_eq!(WRITER.name, "writer");
        assert!(!WRITER.system_prompt.is_empty());
        // Writer should not restrict tools (needs write access)
        assert!(WRITER.allowed_tools.is_none());
        assert_eq!(WRITER.max_iterations, 50);
    }

    #[test]
    fn test_reviewer_preset() {
        assert_eq!(REVIEWER.name, "reviewer");
        assert!(!REVIEWER.system_prompt.is_empty());
        assert!(REVIEWER.system_prompt.contains("code reviewer"));
        assert_eq!(REVIEWER.max_iterations, 30);
    }

    #[test]
    fn test_all_presets() {
        let presets = all_presets();
        assert_eq!(presets.len(), 5);
    }

    #[test]
    fn test_get_preset() {
        assert!(get_preset("explorer").is_some());
        assert!(get_preset("planner").is_some());
        assert!(get_preset("researcher").is_some());
        assert!(get_preset("writer").is_some());
        assert!(get_preset("reviewer").is_some());
        assert!(get_preset("nonexistent").is_none());
    }

    #[test]
    fn test_preset_iterations_table() {
        // Verify iterations match the spec table
        assert_eq!(EXPLORER.max_iterations, 20);
        assert_eq!(PLANNER.max_iterations, 30);
        assert_eq!(RESEARCHER.max_iterations, 40);
        assert_eq!(WRITER.max_iterations, 50);
        assert_eq!(REVIEWER.max_iterations, 30);
    }
}
