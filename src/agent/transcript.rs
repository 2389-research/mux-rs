// ABOUTME: Transcript storage for agent conversations.
// ABOUTME: Enables agent resume by persisting conversation history.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::llm::Message;

/// Trait for storing and retrieving agent transcripts.
///
/// Implement this trait to provide custom storage backends
/// (file system, database, etc.) for agent conversation history.
#[async_trait]
pub trait TranscriptStore: Send + Sync {
    /// Save a transcript for an agent.
    async fn save(&self, agent_id: &str, messages: &[Message]) -> Result<(), anyhow::Error>;

    /// Load a transcript for an agent.
    /// Returns None if no transcript exists for the given agent_id.
    async fn load(&self, agent_id: &str) -> Result<Option<Vec<Message>>, anyhow::Error>;

    /// Delete a transcript.
    async fn delete(&self, agent_id: &str) -> Result<(), anyhow::Error>;

    /// List all stored agent IDs.
    async fn list(&self) -> Result<Vec<String>, anyhow::Error>;
}

/// In-memory transcript store.
///
/// Stores transcripts in memory. Useful for testing and short-lived
/// sessions where persistence is not required.
pub struct MemoryTranscriptStore {
    transcripts: RwLock<HashMap<String, Vec<Message>>>,
}

impl MemoryTranscriptStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            transcripts: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new store wrapped in Arc for sharing.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }
}

impl Default for MemoryTranscriptStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TranscriptStore for MemoryTranscriptStore {
    async fn save(&self, agent_id: &str, messages: &[Message]) -> Result<(), anyhow::Error> {
        self.transcripts
            .write()
            .await
            .insert(agent_id.to_string(), messages.to_vec());
        Ok(())
    }

    async fn load(&self, agent_id: &str) -> Result<Option<Vec<Message>>, anyhow::Error> {
        Ok(self.transcripts.read().await.get(agent_id).cloned())
    }

    async fn delete(&self, agent_id: &str) -> Result<(), anyhow::Error> {
        self.transcripts.write().await.remove(agent_id);
        Ok(())
    }

    async fn list(&self) -> Result<Vec<String>, anyhow::Error> {
        Ok(self.transcripts.read().await.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ContentBlock, Role};

    fn sample_messages() -> Vec<Message> {
        vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::text("Hello")],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::text("Hi there!")],
            },
        ]
    }

    #[tokio::test]
    async fn test_memory_store_save_load() {
        let store = MemoryTranscriptStore::new();
        let messages = sample_messages();

        store.save("agent-1", &messages).await.unwrap();

        let loaded = store.load("agent-1").await.unwrap();
        assert!(loaded.is_some());

        let loaded = loaded.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_store_load_nonexistent() {
        let store = MemoryTranscriptStore::new();
        let loaded = store.load("nonexistent").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_memory_store_delete() {
        let store = MemoryTranscriptStore::new();
        store.save("agent-1", &sample_messages()).await.unwrap();

        store.delete("agent-1").await.unwrap();

        let loaded = store.load("agent-1").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_memory_store_list() {
        let store = MemoryTranscriptStore::new();
        store.save("agent-1", &sample_messages()).await.unwrap();
        store.save("agent-2", &sample_messages()).await.unwrap();

        let mut list = store.list().await.unwrap();
        list.sort();
        assert_eq!(list, vec!["agent-1", "agent-2"]);
    }

    #[tokio::test]
    async fn test_memory_store_overwrite() {
        let store = MemoryTranscriptStore::new();

        let messages1 = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("First")],
        }];

        let messages2 = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("Second")],
        }];

        store.save("agent-1", &messages1).await.unwrap();
        store.save("agent-1", &messages2).await.unwrap();

        let loaded = store.load("agent-1").await.unwrap().unwrap();
        assert_eq!(loaded.len(), 1);
        if let ContentBlock::Text { text } = &loaded[0].content[0] {
            assert_eq!(text, "Second");
        } else {
            panic!("Expected text block");
        }
    }
}
