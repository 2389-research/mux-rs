// ABOUTME: Tests for the token bucket rate limiter.
// ABOUTME: Covers basic operations, refill behavior, and cancellation.

use std::time::{Duration, Instant};

use super::rate_limiter::{Cancelled, RateLimiter};

#[tokio::test]
async fn test_new_limiter_starts_full() {
    let limiter = RateLimiter::new(10.0, 1.0);
    let available = limiter.available().await;
    // Allow small floating point variance
    assert!(
        (available - 10.0).abs() < 0.1,
        "Expected ~10 tokens, got {}",
        available
    );
}

#[tokio::test]
async fn test_take_immediate_when_tokens_available() {
    let limiter = RateLimiter::new(10.0, 1.0);
    let never_cancel = std::future::pending::<()>();

    let start = Instant::now();
    let result = limiter.take(5.0, never_cancel).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    // Should be nearly instant
    assert!(
        elapsed < Duration::from_millis(50),
        "Take should be instant, took {:?}",
        elapsed
    );

    let available = limiter.available().await;
    assert!(
        (available - 5.0).abs() < 0.1,
        "Expected ~5 tokens remaining, got {}",
        available
    );
}

#[tokio::test]
async fn test_take_waits_when_insufficient_tokens() {
    let limiter = RateLimiter::new(1.0, 10.0); // 1 token capacity, refills at 10/sec
    let never_cancel = std::future::pending::<()>();

    // Take the initial token
    limiter
        .take(1.0, std::future::pending::<()>())
        .await
        .unwrap();

    // Now taking 0.5 tokens should wait ~50ms (0.5 tokens / 10 tokens per second)
    let start = Instant::now();
    let result = limiter.take(0.5, never_cancel).await;
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    // Should wait at least 10ms (minimum wait), but less than 200ms
    assert!(
        elapsed >= Duration::from_millis(10),
        "Should wait at least 10ms, waited {:?}",
        elapsed
    );
    assert!(
        elapsed < Duration::from_millis(200),
        "Should not wait too long, waited {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_take_cancelled() {
    let limiter = RateLimiter::new(1.0, 0.1); // Very slow refill

    // Take the initial token
    limiter
        .take(1.0, std::future::pending::<()>())
        .await
        .unwrap();

    // Try to take another token but cancel quickly
    let cancel = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
    };

    let result = limiter.take(1.0, cancel).await;
    assert_eq!(result, Err(Cancelled));
}

#[tokio::test]
async fn test_refill_caps_at_capacity() {
    let limiter = RateLimiter::new(5.0, 100.0); // High refill rate

    // Wait a bit to accumulate tokens
    tokio::time::sleep(Duration::from_millis(100)).await;

    let available = limiter.available().await;
    // Should be capped at capacity (5.0)
    assert!(
        (available - 5.0).abs() < 0.1,
        "Expected ~5 tokens (capacity), got {}",
        available
    );
}

#[tokio::test]
async fn test_multiple_takes_drain_bucket() {
    let limiter = RateLimiter::new(10.0, 1.0);

    // Take tokens multiple times
    for _ in 0..5 {
        limiter
            .take(2.0, std::future::pending::<()>())
            .await
            .unwrap();
    }

    let available = limiter.available().await;
    // Should have ~0 tokens left (took 10 total)
    assert!(available < 0.5, "Expected near 0 tokens, got {}", available);
}

#[tokio::test]
async fn test_burst_then_throttle() {
    let limiter = RateLimiter::new(3.0, 10.0); // Burst of 3, refill at 10/sec

    // Burst: take all 3 tokens instantly
    let start = Instant::now();
    for _ in 0..3 {
        limiter
            .take(1.0, std::future::pending::<()>())
            .await
            .unwrap();
    }
    let burst_time = start.elapsed();
    assert!(
        burst_time < Duration::from_millis(50),
        "Burst should be fast, took {:?}",
        burst_time
    );

    // Now we should be throttled
    let throttle_start = Instant::now();
    limiter
        .take(1.0, std::future::pending::<()>())
        .await
        .unwrap();
    let throttle_time = throttle_start.elapsed();

    // Should wait at least 10ms (minimum wait)
    assert!(
        throttle_time >= Duration::from_millis(10),
        "Should be throttled, waited {:?}",
        throttle_time
    );
}

#[tokio::test]
async fn test_minimum_wait_time_enforced() {
    // Slow refill rate so calculated wait is measurable but less than 10ms
    // With refill_rate = 100 tokens/sec and needing 0.5 tokens,
    // calculated wait = 0.5 / 100 = 5ms, but minimum is 10ms
    let limiter = RateLimiter::new(0.5, 100.0);

    // Drain the bucket completely
    limiter
        .take(0.5, std::future::pending::<()>())
        .await
        .unwrap();

    // Now we need 0.5 tokens. Calculated wait = 5ms, but minimum is 10ms.
    let start = Instant::now();
    limiter
        .take(0.5, std::future::pending::<()>())
        .await
        .unwrap();
    let elapsed = start.elapsed();

    // Should wait at least 10ms even though calculated wait is only 5ms
    assert!(
        elapsed >= Duration::from_millis(10),
        "Minimum wait should be enforced, waited {:?}",
        elapsed
    );
    // But shouldn't wait too long (less than 30ms should be reasonable)
    assert!(
        elapsed < Duration::from_millis(30),
        "Should not wait excessively, waited {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_cancelled_error_display() {
    let err = Cancelled;
    assert_eq!(err.to_string(), "operation cancelled");
}

#[tokio::test]
async fn test_concurrent_takes() {
    use std::sync::Arc;

    let limiter = Arc::new(RateLimiter::new(10.0, 100.0));
    let mut handles = Vec::new();

    // Spawn 5 concurrent tasks that each take 2 tokens
    for _ in 0..5 {
        let limiter = limiter.clone();
        handles.push(tokio::spawn(async move {
            limiter.take(2.0, std::future::pending::<()>()).await
        }));
    }

    // All should complete successfully
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}
