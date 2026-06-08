// ABOUTME: ChatCallbackHook proxies SubAgent hook events to ChatCallback for streaming UI updates.
// ABOUTME: Used by messaging/mod.rs to wire streaming callbacks into the SubAgent hook registry.

use crate::callback::{ChatCallback, ToolUseRequest};
use async_trait::async_trait;
use mux::hook::{Hook, HookAction, HookEvent};
use std::sync::Arc;

/// Hook that proxies SubAgent events to ChatCallback for streaming UI updates.
pub(super) struct ChatCallbackHook {
    callback: Arc<Box<dyn ChatCallback>>,
}

impl ChatCallbackHook {
    pub(super) fn new(callback: Arc<Box<dyn ChatCallback>>) -> Self {
        Self { callback }
    }
}

#[async_trait]
impl Hook for ChatCallbackHook {
    async fn on_event(&self, event: &HookEvent) -> Result<HookAction, anyhow::Error> {
        let callback = self.callback.clone();

        match event {
            HookEvent::ResponseReceived {
                text, tool_uses, ..
            } => {
                // Stream text to callback
                if !text.is_empty() {
                    let text = text.clone();
                    tokio::task::spawn_blocking(move || {
                        callback.on_text_delta(text);
                    })
                    .await
                    .ok();
                }

                // Notify about tool uses
                for (name, id, input) in tool_uses {
                    let callback = self.callback.clone();
                    let request = ToolUseRequest {
                        id: id.clone(),
                        tool_name: name.clone(),
                        server_name: String::new(), // Not an MCP tool
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    };
                    tokio::task::spawn_blocking(move || {
                        callback.on_tool_use(request);
                    })
                    .await
                    .ok();
                }
            }
            HookEvent::PostToolUse {
                tool_use_id,
                result,
                ..
            } => {
                let callback = self.callback.clone();
                let tool_id = tool_use_id.clone();
                let content = result.content.clone();

                tokio::task::spawn_blocking(move || {
                    callback.on_tool_result(tool_id, content);
                })
                .await
                .ok();
            }
            _ => {}
        }

        Ok(HookAction::Continue)
    }

    fn accepts(&self, event: &HookEvent) -> bool {
        matches!(
            event,
            HookEvent::ResponseReceived { .. } | HookEvent::PostToolUse { .. }
        )
    }
}
