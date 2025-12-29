// ABOUTME: ReadFileTool - reads file contents as text.
// ABOUTME: Returns file contents or error message if file cannot be read.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for reading file contents.
pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the file contents as text."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            path: String,
        }
        let params: Params = serde_json::from_value(params)?;

        match std::fs::read_to_string(&params.path) {
            Ok(content) => Ok(ToolResult::text(content)),
            Err(e) => Ok(ToolResult::error(format!("Failed to read file: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_read_file_success() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, world!").unwrap();

        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({
                "path": file.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let tool = ReadFileTool;
        let result = tool
            .execute(serde_json::json!({
                "path": "/nonexistent/file.txt"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Failed to read file"));
    }
}
