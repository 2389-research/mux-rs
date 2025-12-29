// ABOUTME: WriteFileTool - writes content to a file.
// ABOUTME: Creates parent directories if needed, overwrites existing files.

use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for writing content to files.
pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            path: String,
            content: String,
        }
        let params: Params = serde_json::from_value(params)?;

        // Create parent directories if needed
        if let Some(parent) = Path::new(&params.path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        match std::fs::write(&params.path, &params.content) {
            Ok(()) => Ok(ToolResult::text(format!(
                "Successfully wrote {} bytes to {}",
                params.content.len(),
                params.path
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_file_success() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");

        let tool = WriteFileTool;
        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "content": "Hello, world!"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Successfully wrote"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "Hello, world!");
    }

    #[tokio::test]
    async fn test_write_file_creates_directories() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("dir").join("test.txt");

        let tool = WriteFileTool;
        let result = tool
            .execute(serde_json::json!({
                "path": path.to_str().unwrap(),
                "content": "Nested content"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(path.exists());
    }
}
