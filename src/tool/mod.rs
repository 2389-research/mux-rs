// ABOUTME: Tool module - defines tools, registry, and execution.
// ABOUTME: Core abstraction for agent capabilities.

mod registry;
mod result;
mod traits;

pub use registry::*;
pub use result::*;
pub use traits::*;

#[cfg(test)]
mod registry_test;
#[cfg(test)]
mod result_test;
