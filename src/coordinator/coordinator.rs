// ABOUTME: Resource coordinator for multi-agent synchronization.
// ABOUTME: Provides mutex-style locking for shared resources across agents.

use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::Mutex;

/// Error type for resource lock operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockError {
    /// Resource is already locked by another agent.
    AlreadyLocked {
        resource_id: String,
        owner_id: String,
    },
    /// Agent does not own the lock it's trying to release.
    NotOwner {
        resource_id: String,
        owner_id: String,
        requester_id: String,
    },
    /// Operation was cancelled.
    Cancelled,
}

impl std::fmt::Display for LockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LockError::AlreadyLocked { resource_id, owner_id } => {
                write!(f, "resource '{}' is locked by agent '{}'", resource_id, owner_id)
            }
            LockError::NotOwner { resource_id, owner_id, requester_id } => {
                write!(
                    f,
                    "agent '{}' does not own lock on '{}' (owned by '{}')",
                    requester_id, resource_id, owner_id
                )
            }
            LockError::Cancelled => {
                write!(f, "operation cancelled")
            }
        }
    }
}

impl std::error::Error for LockError {}

/// Information about a held resource lock.
#[derive(Debug, Clone)]
pub struct ResourceLock {
    /// The locked resource's identifier.
    pub resource_id: String,
    /// The agent that owns the lock.
    pub owner_id: String,
    /// When the lock was acquired.
    pub acquired_at: Instant,
}

/// Resource coordinator for multi-agent synchronization.
///
/// The coordinator allows multiple agents to coordinate access to shared
/// resources (files, APIs, etc). This prevents conflicts when running
/// parallel agents.
///
/// # Lock Semantics
///
/// - **Idempotent acquire:** If an agent already owns a resource, `acquire()` returns `Ok`.
/// - **Ownership verification:** `release()` validates that the requesting agent owns the lock.
/// - **Idempotent release_all:** Returns no error even if the agent holds no locks.
pub struct Coordinator {
    locks: Mutex<HashMap<String, ResourceLock>>,
}

impl Default for Coordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl Coordinator {
    /// Create a new resource coordinator.
    pub fn new() -> Self {
        Self {
            locks: Mutex::new(HashMap::new()),
        }
    }

    /// Acquire a lock on a resource.
    ///
    /// Returns `Ok(())` if the lock was acquired or if the agent already owns it.
    /// Returns `Err(LockError::AlreadyLocked)` if another agent owns the lock.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The ID of the agent requesting the lock.
    /// * `resource_id` - The ID of the resource to lock.
    pub async fn acquire(&self, agent_id: &str, resource_id: &str) -> Result<(), LockError> {
        self.acquire_with_cancel(agent_id, resource_id, std::future::pending::<()>())
            .await
    }

    /// Acquire a lock on a resource with cancellation support.
    ///
    /// Returns `Ok(())` if the lock was acquired or if the agent already owns it.
    /// Returns `Err(LockError::AlreadyLocked)` if another agent owns the lock.
    /// Returns `Err(LockError::Cancelled)` if the cancel future completes first.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The ID of the agent requesting the lock.
    /// * `resource_id` - The ID of the resource to lock.
    /// * `cancel` - A future that, when completed, cancels the operation.
    pub async fn acquire_with_cancel<F>(
        &self,
        agent_id: &str,
        resource_id: &str,
        cancel: F,
    ) -> Result<(), LockError>
    where
        F: std::future::Future<Output = ()>,
    {
        tokio::pin!(cancel);

        // Check for cancellation first
        tokio::select! {
            biased;
            () = &mut cancel => {
                return Err(LockError::Cancelled);
            }
            result = self.try_acquire(agent_id, resource_id) => {
                return result;
            }
        }
    }

    /// Internal acquire implementation.
    async fn try_acquire(&self, agent_id: &str, resource_id: &str) -> Result<(), LockError> {
        let mut locks = self.locks.lock().await;

        if let Some(lock) = locks.get(resource_id) {
            if lock.owner_id == agent_id {
                // Idempotent: already owned by this agent
                return Ok(());
            }
            return Err(LockError::AlreadyLocked {
                resource_id: resource_id.to_string(),
                owner_id: lock.owner_id.clone(),
            });
        }

        locks.insert(
            resource_id.to_string(),
            ResourceLock {
                resource_id: resource_id.to_string(),
                owner_id: agent_id.to_string(),
                acquired_at: Instant::now(),
            },
        );

        Ok(())
    }

    /// Release a lock on a resource.
    ///
    /// Returns `Ok(())` if the lock was released or if the resource wasn't locked.
    /// Returns `Err(LockError::NotOwner)` if the resource is locked by a different agent.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The ID of the agent releasing the lock.
    /// * `resource_id` - The ID of the resource to unlock.
    pub async fn release(&self, agent_id: &str, resource_id: &str) -> Result<(), LockError> {
        let mut locks = self.locks.lock().await;

        if let Some(lock) = locks.get(resource_id) {
            if lock.owner_id != agent_id {
                return Err(LockError::NotOwner {
                    resource_id: resource_id.to_string(),
                    owner_id: lock.owner_id.clone(),
                    requester_id: agent_id.to_string(),
                });
            }
            locks.remove(resource_id);
        }

        // Idempotent: not locked is fine
        Ok(())
    }

    /// Release all locks held by an agent.
    ///
    /// This is idempotent - calling it when the agent holds no locks is not an error.
    ///
    /// # Arguments
    ///
    /// * `agent_id` - The ID of the agent whose locks should be released.
    pub async fn release_all(&self, agent_id: &str) {
        let mut locks = self.locks.lock().await;

        locks.retain(|_, lock| lock.owner_id != agent_id);
    }
}
