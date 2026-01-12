// ABOUTME: Coordinator module for managing agent execution resources.
// ABOUTME: Contains rate limiting and other coordination primitives.

mod coordinator;
mod rate_limiter;

pub use coordinator::{Coordinator, LockError, ResourceLock};
pub use rate_limiter::RateLimiter;

#[cfg(test)]
mod coordinator_test;
#[cfg(test)]
mod rate_limiter_test;
