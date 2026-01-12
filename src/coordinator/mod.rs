// ABOUTME: Coordinator module for managing agent execution resources.
// ABOUTME: Contains rate limiting and other coordination primitives.

mod rate_limiter;

pub use rate_limiter::RateLimiter;

#[cfg(test)]
mod rate_limiter_test;
