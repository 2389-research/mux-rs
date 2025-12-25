// ABOUTME: Permission module - policy engine for tool execution control.
// ABOUTME: Supports rules, patterns, conditionals, and approval handlers.

mod handler;
mod policy;

pub use handler::*;
pub use policy::*;

#[cfg(test)]
mod policy_test;
