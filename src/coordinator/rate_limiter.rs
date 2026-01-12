// ABOUTME: Token bucket rate limiter for API call throttling.
// ABOUTME: Allows bursts up to capacity while maintaining average rate.

use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Error returned when a rate limiter operation is cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "operation cancelled")
    }
}

impl std::error::Error for Cancelled {}

/// Mutable state for the rate limiter, protected by a single mutex.
struct RateLimiterState {
    tokens: f64,
    last_refill: Instant,
}

/// Token bucket rate limiter for API call throttling.
///
/// The token bucket algorithm allows bursting up to `capacity` tokens,
/// then refills at `refill_rate` tokens per second. This provides
/// smooth rate limiting while allowing short bursts of activity.
pub struct RateLimiter {
    state: Mutex<RateLimiterState>,
    capacity: f64,
    refill_rate: f64,
}

impl RateLimiter {
    /// Create a new token bucket rate limiter.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum tokens (burst size). The bucket starts full.
    /// * `refill_rate` - Tokens added per second.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` or `refill_rate` is not positive.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        assert!(capacity > 0.0, "capacity must be positive");
        assert!(refill_rate > 0.0, "refill_rate must be positive");

        Self {
            state: Mutex::new(RateLimiterState {
                tokens: capacity,
                last_refill: Instant::now(),
            }),
            capacity,
            refill_rate,
        }
    }

    /// Take tokens from the bucket, waiting if necessary.
    ///
    /// Returns `Ok(())` when the tokens have been consumed.
    /// Returns `Err(Cancelled)` if the cancellation token is triggered.
    ///
    /// # Arguments
    ///
    /// * `tokens` - Number of tokens to consume.
    /// * `cancel` - Cancellation token. When this future completes, the operation is cancelled.
    pub async fn take<F>(&self, tokens: f64, cancel: F) -> Result<(), Cancelled>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(cancel);

        loop {
            let wait_time = self.try_take(tokens).await;
            if wait_time.is_zero() {
                return Ok(());
            }

            // Enforce a minimum wait time of 10ms to prevent CPU-intensive busy loops.
            // This is necessary because:
            // 1. Clock/timer granularity varies across systems (typically 1-15ms)
            // 2. Very short waits (<10ms) can lead to excessive CPU usage from
            //    rapid lock contention and context switching
            // 3. The 10ms minimum ensures predictable behavior across platforms
            //    while still providing responsive rate limiting for most use cases
            let wait_time = wait_time.max(Duration::from_millis(10));

            tokio::select! {
                biased;
                () = &mut cancel => {
                    return Err(Cancelled);
                }
                () = tokio::time::sleep(wait_time) => {
                    // Retry after calculated wait time
                }
            }
        }
    }

    /// Attempt to take tokens without waiting.
    ///
    /// Returns `Duration::ZERO` if successful, otherwise returns the
    /// estimated wait time until tokens are available.
    async fn try_take(&self, tokens: f64) -> Duration {
        let mut state = self.state.lock().await;

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.last_refill = now;

        state.tokens += elapsed * self.refill_rate;
        if state.tokens > self.capacity {
            state.tokens = self.capacity;
        }

        // Try to take tokens
        if state.tokens >= tokens {
            state.tokens -= tokens;
            return Duration::ZERO;
        }

        // Calculate wait time for needed tokens
        let needed = tokens - state.tokens;
        let wait_seconds = needed / self.refill_rate;
        Duration::from_secs_f64(wait_seconds)
    }

    /// Get the current number of tokens available (for testing/monitoring).
    ///
    /// Note: This method performs a token refill as a side effect, updating
    /// the internal state to account for time elapsed since the last operation.
    pub async fn available(&self) -> f64 {
        let mut state = self.state.lock().await;

        // Refill tokens based on elapsed time
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.last_refill = now;

        state.tokens += elapsed * self.refill_rate;
        if state.tokens > self.capacity {
            state.tokens = self.capacity;
        }

        state.tokens
    }
}
