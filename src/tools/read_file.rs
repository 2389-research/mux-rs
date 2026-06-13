// ABOUTME: ReadFileTool - reads file contents as text.
// ABOUTME: Returns file contents or error message if file cannot be read.

use std::io::Read;

use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for reading file contents.
#[derive(Default)]
pub struct ReadFileTool {
    root: Option<RootedFs>,
}

impl ReadFileTool {
    /// Create an unconfined reader (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a reader confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}

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

        let content = match &self.root {
            Some(jail) => match jail.open_read(&params.path) {
                Ok(mut file) => {
                    let mut buf = String::new();
                    match file.read_to_string(&mut buf) {
                        Ok(_) => buf,
                        Err(e) => {
                            return Ok(ToolResult::error(format!("Failed to read file: {}", e)));
                        }
                    }
                }
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => match std::fs::read_to_string(&params.path) {
                Ok(content) => content,
                Err(e) => return Ok(ToolResult::error(format!("Failed to read file: {}", e))),
            },
        };
        Ok(ToolResult::text(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file_success() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "Hello, world!").unwrap();

        let tool = ReadFileTool::new();
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
        let tool = ReadFileTool::new();
        let result = tool
            .execute(serde_json::json!({
                "path": "/nonexistent/file.txt"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Failed to read file"));
    }

    #[tokio::test]
    async fn test_read_file_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "in-root secret").unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ReadFileTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({ "path": "inside.txt" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert!(ok.content.contains("in-root secret"));

        let blocked = tool
            .execute(serde_json::json!({ "path": "/etc/passwd" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
    }
}
