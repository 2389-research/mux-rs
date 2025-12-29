// ABOUTME: Subagent orchestration module - spawn and manage child agents.
// ABOUTME: Provides TaskTool, AgentDefinition, FilteredRegistry, and SubAgent runner.

mod definition;
mod filter;
mod runner;
mod task;

pub use definition::{AgentDefinition, AgentRegistry};
pub use filter::FilteredRegistry;
pub use runner::{SubAgent, SubAgentResult};
pub use task::TaskTool;
