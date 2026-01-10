// ABOUTME: Persistence layer for MuxEngine - load/save workspaces, conversations, messages.
// ABOUTME: Handles disk I/O and legacy format migration.

use super::MuxEngine;
use crate::types::Conversation;
use mux::prelude::{ContentBlock, Role};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// File names for persistence
pub(super) const WORKSPACES_FILE: &str = "workspaces.json";
pub(super) const CONVERSATIONS_FILE: &str = "conversations.json";
pub(super) const MESSAGES_DIR: &str = "messages";

/// Stored message for conversation history.
/// Uses Vec<ContentBlock> to preserve tool use/result structure.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct StoredMessage {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// Legacy format (pre-v0.6.2) stored content as String.
/// Used for migration of old conversation files.
#[derive(Clone, Deserialize)]
struct LegacyStoredMessage {
    role: Role,
    content: String,
}

impl From<LegacyStoredMessage> for StoredMessage {
    fn from(legacy: LegacyStoredMessage) -> Self {
        StoredMessage {
            role: legacy.role,
            content: vec![ContentBlock::text(legacy.content)],
        }
    }
}

/// Persistence helper methods
impl MuxEngine {
    /// Load workspaces from disk. Returns empty HashMap if file doesn't exist or is invalid.
    pub(super) fn load_workspaces(
        data_dir: &PathBuf,
    ) -> HashMap<String, crate::types::Workspace> {
        let path = data_dir.join(WORKSPACES_FILE);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("Failed to parse {}: {}", path.display(), e);
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    }

    /// Load conversations from disk. Returns empty HashMap if file doesn't exist or is invalid.
    pub(super) fn load_conversations(
        data_dir: &PathBuf,
    ) -> HashMap<String, Vec<Conversation>> {
        let path = data_dir.join(CONVERSATIONS_FILE);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("Failed to parse {}: {}", path.display(), e);
                HashMap::new()
            }),
            Err(_) => HashMap::new(),
        }
    }

    /// Load all message histories from disk based on known conversations.
    /// Handles migration from legacy format (String content) to new format (Vec<ContentBlock>).
    pub(super) fn load_all_messages(
        data_dir: &PathBuf,
        conversations: &HashMap<String, Vec<Conversation>>,
    ) -> HashMap<String, Vec<StoredMessage>> {
        let messages_dir = data_dir.join(MESSAGES_DIR);
        let mut all_messages = HashMap::new();

        for convs in conversations.values() {
            for conv in convs {
                let path = messages_dir.join(format!("{}.json", conv.id));
                if let Ok(contents) = fs::read_to_string(&path) {
                    // Try new format first
                    if let Ok(msgs) = serde_json::from_str::<Vec<StoredMessage>>(&contents) {
                        all_messages.insert(conv.id.clone(), msgs);
                    }
                    // Fall back to legacy format (String content)
                    else if let Ok(legacy_msgs) =
                        serde_json::from_str::<Vec<LegacyStoredMessage>>(&contents)
                    {
                        let msgs: Vec<StoredMessage> =
                            legacy_msgs.into_iter().map(StoredMessage::from).collect();
                        all_messages.insert(conv.id.clone(), msgs);
                    }
                }
                // Initialize with empty vec if not loaded
                all_messages.entry(conv.id.clone()).or_insert_with(Vec::new);
            }
        }

        all_messages
    }

    /// Save workspaces to disk.
    pub(super) fn save_workspaces(&self) {
        let path = self.data_dir.join(WORKSPACES_FILE);
        let workspaces = self.workspaces.read();
        if let Err(e) =
            fs::write(&path, serde_json::to_string_pretty(&*workspaces).unwrap_or_default())
        {
            eprintln!("Failed to save workspaces to {}: {}", path.display(), e);
        }
    }

    /// Save conversations to disk.
    pub(super) fn save_conversations(&self) {
        let path = self.data_dir.join(CONVERSATIONS_FILE);
        let conversations = self.conversations.read();
        if let Err(e) = fs::write(
            &path,
            serde_json::to_string_pretty(&*conversations).unwrap_or_default(),
        ) {
            eprintln!("Failed to save conversations to {}: {}", path.display(), e);
        }
    }

    /// Save message history for a specific conversation to disk.
    pub(super) fn save_messages(&self, conversation_id: &str) {
        let messages_dir = self.data_dir.join(MESSAGES_DIR);
        let path = messages_dir.join(format!("{}.json", conversation_id));
        let history = self.message_history.read();
        if let Some(messages) = history.get(conversation_id) {
            if let Err(e) =
                fs::write(&path, serde_json::to_string_pretty(messages).unwrap_or_default())
            {
                eprintln!("Failed to save messages to {}: {}", path.display(), e);
            }
        }
    }

    /// Delete message file for a conversation.
    pub(super) fn delete_message_file(&self, conversation_id: &str) {
        let messages_dir = self.data_dir.join(MESSAGES_DIR);
        let path = messages_dir.join(format!("{}.json", conversation_id));
        if let Err(e) = fs::remove_file(&path) {
            // Only log if it's not a "file not found" error
            if e.kind() != std::io::ErrorKind::NotFound {
                eprintln!("Failed to delete message file {}: {}", path.display(), e);
            }
        }
    }
}
