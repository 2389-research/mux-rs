// ABOUTME: Subagent orchestration module - spawn and manage child agents.
// ABOUTME: Provides TaskTool, AgentDefinition, FilteredRegistry, SubAgent runner, and transcript storage.

mod async_handle;
mod definition;
mod filter;
mod presets;
mod runner;
mod task;
mod transcript;

pub use async_handle::{RunHandle, RunStatus};
pub use definition::{AgentDefinition, AgentRegistry};
pub use filter::FilteredRegistry;
pub use presets::{
    EXPLORER, PLANNER, Preset, RESEARCHER, REVIEWER, WRITER, all_presets, get_preset,
};
pub use runner::{SubAgent, SubAgentResult};
pub use task::TaskTool;
pub use transcript::{MemoryTranscriptStore, TranscriptStore};
