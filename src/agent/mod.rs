// ABOUTME: Subagent orchestration module - spawn and manage child agents.
// ABOUTME: Provides TaskTool, AgentDefinition, FilteredRegistry, SubAgent runner, and transcript storage.

mod definition;
mod filter;
mod runner;
mod task;
mod transcript;

pub use definition::{AgentDefinition, AgentRegistry};
pub use filter::FilteredRegistry;
pub use runner::{SubAgent, SubAgentResult};
pub use task::TaskTool;
pub use transcript::{MemoryTranscriptStore, TranscriptStore};
