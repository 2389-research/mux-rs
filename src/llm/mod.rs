// ABOUTME: LLM module - client abstraction for language model providers.
// ABOUTME: Defines types, traits, and provider implementations.

mod anthropic;
mod client;
mod gemini;
mod ollama;
mod openai;
mod openrouter;
pub mod stream_accumulator;
mod types;

pub use anthropic::*;
pub use client::*;
pub use gemini::*;
pub use ollama::*;
pub use openai::*;
pub use openrouter::*;
pub use types::*;

#[cfg(test)]
mod types_test;

#[cfg(test)]
mod anthropic_test;
