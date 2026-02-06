// ABOUTME: Tests for the resource coordinator lock semantics.
// ABOUTME: Covers acquire, release, ownership verification, and idempotency.

use super::coordinator::{Coordinator, LockError};

#[tokio::test]
async fn test_acquire_succeeds_when_resource_unlocked() {
    let coordinator = Coordinator::new();
    let result = coordinator.acquire("agent-1", "resource-a").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_acquire_idempotent_when_same_owner() {
    let coordinator = Coordinator::new();

    // Acquire once
    coordinator.acquire("agent-1", "resource-a").await.unwrap();

    // Acquire again with same owner should succeed
    let result = coordinator.acquire("agent-1", "resource-a").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_acquire_fails_when_locked_by_another() {
    let coordinator = Coordinator::new();

    // Agent 1 acquires
    coordinator.acquire("agent-1", "resource-a").await.unwrap();

    // Agent 2 tries to acquire same resource
    let result = coordinator.acquire("agent-2", "resource-a").await;
    assert!(result.is_err());

    match result.unwrap_err() {
        LockError::AlreadyLocked {
            resource_id,
            owner_id,
        } => {
            assert_eq!(resource_id, "resource-a");
            assert_eq!(owner_id, "agent-1");
        }
        other => panic!("Expected AlreadyLocked, got {:?}", other),
    }
}

#[tokio::test]
async fn test_release_succeeds_when_owner() {
    let coordinator = Coordinator::new();

    coordinator.acquire("agent-1", "resource-a").await.unwrap();
    let result = coordinator.release("agent-1", "resource-a").await;
    assert!(result.is_ok());

    // Resource should now be available
    let result = coordinator.acquire("agent-2", "resource-a").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_release_fails_when_not_owner() {
    let coordinator = Coordinator::new();

    coordinator.acquire("agent-1", "resource-a").await.unwrap();

    // Agent 2 tries to release
    let result = coordinator.release("agent-2", "resource-a").await;
    assert!(result.is_err());

    match result.unwrap_err() {
        LockError::NotOwner {
            resource_id,
            owner_id,
            requester_id,
        } => {
            assert_eq!(resource_id, "resource-a");
            assert_eq!(owner_id, "agent-1");
            assert_eq!(requester_id, "agent-2");
        }
        other => panic!("Expected NotOwner, got {:?}", other),
    }
}

#[tokio::test]
async fn test_release_idempotent_when_not_locked() {
    let coordinator = Coordinator::new();

    // Release a resource that was never locked should succeed
    let result = coordinator.release("agent-1", "resource-a").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_release_all_releases_all_locks_for_agent() {
    let coordinator = Coordinator::new();

    // Agent 1 acquires multiple resources
    coordinator.acquire("agent-1", "resource-a").await.unwrap();
    coordinator.acquire("agent-1", "resource-b").await.unwrap();
    coordinator.acquire("agent-1", "resource-c").await.unwrap();

    // Agent 2 acquires one resource
    coordinator.acquire("agent-2", "resource-d").await.unwrap();

    // Release all for agent 1
    coordinator.release_all("agent-1").await;

    // Agent 2 should be able to acquire agent 1's former resources
    assert!(coordinator.acquire("agent-2", "resource-a").await.is_ok());
    assert!(coordinator.acquire("agent-2", "resource-b").await.is_ok());
    assert!(coordinator.acquire("agent-2", "resource-c").await.is_ok());

    // But agent 2's resource should still be locked
    assert!(coordinator.acquire("agent-1", "resource-d").await.is_err());
}

#[tokio::test]
async fn test_release_all_idempotent_when_no_locks() {
    let coordinator = Coordinator::new();

    // Should not panic or error
    coordinator.release_all("agent-1").await;
    coordinator.release_all("agent-1").await; // Call twice
}

#[tokio::test]
async fn test_acquire_cancelled() {
    let coordinator = Coordinator::new();

    // Create a cancel future that completes immediately
    let cancel = async {};

    let result = coordinator
        .acquire_with_cancel("agent-1", "resource-a", cancel)
        .await;
    assert!(result.is_err());

    match result.unwrap_err() {
        LockError::Cancelled => {}
        other => panic!("Expected Cancelled, got {:?}", other),
    }
}

#[tokio::test]
async fn test_multiple_resources_independent() {
    let coordinator = Coordinator::new();

    // Different resources can be locked by different agents
    coordinator.acquire("agent-1", "resource-a").await.unwrap();
    coordinator.acquire("agent-2", "resource-b").await.unwrap();

    // Both should be able to release their own
    assert!(coordinator.release("agent-1", "resource-a").await.is_ok());
    assert!(coordinator.release("agent-2", "resource-b").await.is_ok());
}

#[tokio::test]
async fn test_lock_error_display() {
    let err = LockError::AlreadyLocked {
        resource_id: "res".to_string(),
        owner_id: "owner".to_string(),
    };
    assert!(err.to_string().contains("res"));
    assert!(err.to_string().contains("owner"));

    let err = LockError::NotOwner {
        resource_id: "res".to_string(),
        owner_id: "owner".to_string(),
        requester_id: "req".to_string(),
    };
    assert!(err.to_string().contains("res"));
    assert!(err.to_string().contains("owner"));
    assert!(err.to_string().contains("req"));

    let err = LockError::Cancelled;
    assert!(err.to_string().contains("cancelled"));
}

#[tokio::test]
async fn test_concurrent_acquire_same_resource() {
    use std::sync::Arc;

    let coordinator = Arc::new(Coordinator::new());
    let mut handles = Vec::new();

    // Spawn 10 tasks trying to acquire the same resource
    for i in 0..10 {
        let coordinator = coordinator.clone();
        let agent_id = format!("agent-{}", i);
        handles.push(tokio::spawn(async move {
            coordinator.acquire(&agent_id, "shared-resource").await
        }));
    }

    // Exactly one should succeed
    let mut success_count = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            success_count += 1;
        }
    }

    assert_eq!(
        success_count, 1,
        "Exactly one agent should acquire the lock"
    );
}
