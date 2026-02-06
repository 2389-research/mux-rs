// ABOUTME: Implements RunHandle for async agent execution with status polling.
// ABOUTME: Provides status tracking, waiting, timeout, duration, and cancellation support.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use crate::error::MuxError;

/// Represents the current state of an async agent run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunStatus {
    /// The agent has not started yet.
    Pending = 0,
    /// The agent is currently executing.
    Running = 1,
    /// The agent completed successfully.
    Completed = 2,
    /// The agent encountered an error.
    Failed = 3,
    /// The agent was cancelled.
    Cancelled = 4,
}

impl RunStatus {
    /// Convert from u8 value to RunStatus.
    fn from_u8(value: u8) -> Self {
        match value {
            0 => RunStatus::Pending,
            1 => RunStatus::Running,
            2 => RunStatus::Completed,
            3 => RunStatus::Failed,
            4 => RunStatus::Cancelled,
            _ => RunStatus::Failed, // Unknown status treated as failed
        }
    }
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Pending => write!(f, "pending"),
            RunStatus::Running => write!(f, "running"),
            RunStatus::Completed => write!(f, "completed"),
            RunStatus::Failed => write!(f, "failed"),
            RunStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Handle for async agent execution with status polling and waiting.
///
/// A RunHandle allows callers to:
/// - Check if an agent is still running (polling)
/// - Wait for completion (blocking)
/// - Wait with timeout
/// - Get duration of execution
/// - Cancel execution
pub struct RunHandle {
    /// Atomic status for lock-free reads.
    status: Arc<AtomicU8>,
    /// Error from execution (if any).
    error: Arc<Mutex<Option<MuxError>>>,
    /// Notification for async waiting.
    done: Arc<Notify>,
    /// When the run started.
    start_time: Instant,
    /// When the run ended (if complete).
    end_time: Arc<Mutex<Option<Instant>>>,
}

impl RunHandle {
    /// Create a new RunHandle in Pending state.
    pub fn new() -> Self {
        Self {
            status: Arc::new(AtomicU8::new(RunStatus::Pending as u8)),
            error: Arc::new(Mutex::new(None)),
            done: Arc::new(Notify::new()),
            start_time: Instant::now(),
            end_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Get the current execution status.
    pub fn status(&self) -> RunStatus {
        RunStatus::from_u8(self.status.load(Ordering::SeqCst))
    }

    /// Get the error from execution, if any.
    ///
    /// Returns None if still running or completed successfully.
    /// The error is cloned to allow multiple reads.
    pub fn err(&self) -> Option<String> {
        let guard = self.error.lock().unwrap();
        guard.as_ref().map(|e| e.to_string())
    }

    /// Wait for execution to complete.
    ///
    /// Returns Ok(()) if completed successfully, or the error if failed.
    pub async fn wait(&self) -> Result<(), String> {
        // If already complete, return immediately
        if self.is_complete() {
            return self.result();
        }

        // Wait for notification
        self.done.notified().await;
        self.result()
    }

    /// Wait for execution to complete with a timeout.
    ///
    /// Returns Ok(()) if completed successfully within the timeout.
    /// Returns an error if failed or timeout expired.
    pub async fn wait_with_timeout(&self, timeout: Duration) -> Result<(), String> {
        // If already complete, return immediately
        if self.is_complete() {
            return self.result();
        }

        // Wait with timeout
        match tokio::time::timeout(timeout, self.done.notified()).await {
            Ok(()) => self.result(),
            Err(_) => Err("timeout waiting for completion".to_string()),
        }
    }

    /// Poll the current status and error without blocking.
    ///
    /// If the agent is still running, error will be None.
    pub fn poll(&self) -> (RunStatus, Option<String>) {
        (self.status(), self.err())
    }

    /// Returns true if the agent has finished (success, failure, or cancelled).
    pub fn is_complete(&self) -> bool {
        let status = self.status();
        matches!(
            status,
            RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled
        )
    }

    /// Get how long the agent has been running (or ran for if complete).
    pub fn duration(&self) -> Duration {
        let end_guard = self.end_time.lock().unwrap();
        match *end_guard {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Attempt to cancel the running agent.
    ///
    /// Returns true if the agent was still running when cancel was called.
    pub fn cancel(&self) -> bool {
        let current = self.status.load(Ordering::SeqCst);
        if current == RunStatus::Pending as u8 || current == RunStatus::Running as u8 {
            // Attempt to set status to Cancelled
            let result = self
                .status
                .compare_exchange(
                    current,
                    RunStatus::Cancelled as u8,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok();

            if result {
                // Set end time
                let mut end_guard = self.end_time.lock().unwrap();
                *end_guard = Some(Instant::now());

                // Notify waiters
                self.done.notify_waiters();
            }

            result
        } else {
            false
        }
    }

    // Internal methods for managing state

    /// Set status to Running.
    pub(crate) fn set_running(&self) {
        self.status
            .store(RunStatus::Running as u8, Ordering::SeqCst);
    }

    /// Mark as completed successfully.
    pub(crate) fn set_completed(&self) {
        {
            let mut end_guard = self.end_time.lock().unwrap();
            *end_guard = Some(Instant::now());
        }
        self.status
            .store(RunStatus::Completed as u8, Ordering::SeqCst);
        self.done.notify_waiters();
    }

    /// Mark as failed with an error.
    pub(crate) fn set_failed(&self, error: MuxError) {
        {
            let mut end_guard = self.end_time.lock().unwrap();
            *end_guard = Some(Instant::now());
        }
        {
            let mut err_guard = self.error.lock().unwrap();
            *err_guard = Some(error);
        }
        self.status.store(RunStatus::Failed as u8, Ordering::SeqCst);
        self.done.notify_waiters();
    }

    /// Get the result based on current status.
    fn result(&self) -> Result<(), String> {
        match self.status() {
            RunStatus::Completed => Ok(()),
            RunStatus::Cancelled => Err("cancelled".to_string()),
            RunStatus::Failed => Err(self.err().unwrap_or_else(|| "unknown error".to_string())),
            _ => Err("not complete".to_string()),
        }
    }
}

impl Default for RunHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_run_status_display() {
        assert_eq!(RunStatus::Pending.to_string(), "pending");
        assert_eq!(RunStatus::Running.to_string(), "running");
        assert_eq!(RunStatus::Completed.to_string(), "completed");
        assert_eq!(RunStatus::Failed.to_string(), "failed");
        assert_eq!(RunStatus::Cancelled.to_string(), "cancelled");
    }

    #[test]
    fn test_run_status_from_u8() {
        assert_eq!(RunStatus::from_u8(0), RunStatus::Pending);
        assert_eq!(RunStatus::from_u8(1), RunStatus::Running);
        assert_eq!(RunStatus::from_u8(2), RunStatus::Completed);
        assert_eq!(RunStatus::from_u8(3), RunStatus::Failed);
        assert_eq!(RunStatus::from_u8(4), RunStatus::Cancelled);
        // Unknown value defaults to Failed
        assert_eq!(RunStatus::from_u8(255), RunStatus::Failed);
    }

    #[test]
    fn test_new_handle_is_pending() {
        let handle = RunHandle::new();
        assert_eq!(handle.status(), RunStatus::Pending);
        assert!(!handle.is_complete());
        assert!(handle.err().is_none());
    }

    #[test]
    fn test_set_running() {
        let handle = RunHandle::new();
        handle.set_running();
        assert_eq!(handle.status(), RunStatus::Running);
        assert!(!handle.is_complete());
    }

    #[test]
    fn test_set_completed() {
        let handle = RunHandle::new();
        handle.set_running();
        handle.set_completed();
        assert_eq!(handle.status(), RunStatus::Completed);
        assert!(handle.is_complete());
        assert!(handle.err().is_none());
    }

    #[test]
    fn test_set_failed() {
        let handle = RunHandle::new();
        handle.set_running();
        handle.set_failed(MuxError::Llm(crate::error::LlmError::StreamClosed));
        assert_eq!(handle.status(), RunStatus::Failed);
        assert!(handle.is_complete());
        assert!(handle.err().is_some());
        assert!(handle.err().unwrap().contains("Stream closed"));
    }

    #[test]
    fn test_cancel_running() {
        let handle = RunHandle::new();
        handle.set_running();
        let cancelled = handle.cancel();
        assert!(cancelled);
        assert_eq!(handle.status(), RunStatus::Cancelled);
        assert!(handle.is_complete());
    }

    #[test]
    fn test_cancel_pending() {
        let handle = RunHandle::new();
        let cancelled = handle.cancel();
        assert!(cancelled);
        assert_eq!(handle.status(), RunStatus::Cancelled);
    }

    #[test]
    fn test_cancel_completed_returns_false() {
        let handle = RunHandle::new();
        handle.set_completed();
        let cancelled = handle.cancel();
        assert!(!cancelled);
        assert_eq!(handle.status(), RunStatus::Completed);
    }

    #[test]
    fn test_poll() {
        let handle = RunHandle::new();
        let (status, err) = handle.poll();
        assert_eq!(status, RunStatus::Pending);
        assert!(err.is_none());

        handle.set_running();
        let (status, err) = handle.poll();
        assert_eq!(status, RunStatus::Running);
        assert!(err.is_none());

        handle.set_failed(MuxError::Llm(crate::error::LlmError::StreamClosed));
        let (status, err) = handle.poll();
        assert_eq!(status, RunStatus::Failed);
        assert!(err.is_some());
    }

    #[test]
    fn test_duration_increases() {
        let handle = RunHandle::new();
        let d1 = handle.duration();
        std::thread::sleep(Duration::from_millis(10));
        let d2 = handle.duration();
        assert!(d2 > d1);
    }

    #[test]
    fn test_duration_freezes_on_complete() {
        let handle = RunHandle::new();
        std::thread::sleep(Duration::from_millis(10));
        handle.set_completed();
        let d1 = handle.duration();
        std::thread::sleep(Duration::from_millis(10));
        let d2 = handle.duration();
        // Duration should be the same after completion
        assert_eq!(d1, d2);
    }

    #[tokio::test]
    async fn test_wait_immediate_completion() {
        let handle = RunHandle::new();
        handle.set_completed();
        let result = handle.wait().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_completion() {
        let handle = RunHandle::new();
        let handle_clone = RunHandle {
            status: handle.status.clone(),
            error: handle.error.clone(),
            done: handle.done.clone(),
            start_time: handle.start_time,
            end_time: handle.end_time.clone(),
        };

        // Spawn a task that completes after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            handle_clone.set_completed();
        });

        let result = handle.wait().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_with_timeout_success() {
        let handle = RunHandle::new();
        handle.set_completed();
        let result = handle.wait_with_timeout(Duration::from_millis(100)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_with_timeout_expired() {
        let handle = RunHandle::new();
        let result = handle.wait_with_timeout(Duration::from_millis(10)).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timeout"));
    }

    #[tokio::test]
    async fn test_wait_returns_error_on_failure() {
        let handle = RunHandle::new();
        handle.set_failed(MuxError::Llm(crate::error::LlmError::StreamClosed));
        let result = handle.wait().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wait_returns_cancelled() {
        let handle = RunHandle::new();
        handle.cancel();
        let result = handle.wait().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cancelled"));
    }

    #[test]
    fn test_is_complete_variants() {
        let handle = RunHandle::new();
        assert!(!handle.is_complete()); // Pending

        handle.set_running();
        assert!(!handle.is_complete()); // Running

        let handle2 = RunHandle::new();
        handle2.set_completed();
        assert!(handle2.is_complete()); // Completed

        let handle3 = RunHandle::new();
        handle3.set_failed(MuxError::Llm(crate::error::LlmError::StreamClosed));
        assert!(handle3.is_complete()); // Failed

        let handle4 = RunHandle::new();
        handle4.cancel();
        assert!(handle4.is_complete()); // Cancelled
    }

    #[test]
    fn test_default_impl() {
        let handle = RunHandle::default();
        assert_eq!(handle.status(), RunStatus::Pending);
    }
}
