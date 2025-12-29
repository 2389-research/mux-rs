// ABOUTME: ListFilesTool - lists files matching a glob pattern.
// ABOUTME: Shows directories with [dir] prefix.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for listing files in a directory with glob patterns.
pub struct ListFilesTool;

#[async_trait]
impl Tool for ListFilesTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn description(&self) -> &str {
        "List files in a directory matching a glob pattern."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory to list (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to match (default: *)"
                }
            }
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize, Default)]
        struct Params {
            path: Option<String>,
            glob: Option<String>,
        }
        let params: Params = serde_json::from_value(params).unwrap_or_default();

        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let glob_pattern = params.glob.unwrap_or_else(|| "*".to_string());
        let full_pattern = format!("{}/{}", base_path, glob_pattern);

        let mut files = Vec::new();
        for entry in glob::glob(&full_pattern).unwrap_or_else(|_| glob::glob("").unwrap()) {
            if let Ok(path) = entry {
                let prefix = if path.is_dir() { "[dir] " } else { "" };
                files.push(format!("{}{}", prefix, path.display()));
            }
        }

        if files.is_empty() {
            Ok(ToolResult::text("No files found"))
        } else {
            Ok(ToolResult::text(files.join("\n")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_list_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file1.txt"), "").unwrap();
        std::fs::write(dir.path().join("file2.txt"), "").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(serde_json::json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("file1.txt"));
        assert!(result.content.contains("file2.txt"));
        assert!(result.content.contains("[dir]"));
    }

    #[tokio::test]
    async fn test_list_files_with_glob() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file1.txt"), "").unwrap();
        std::fs::write(dir.path().join("file2.rs"), "").unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(serde_json::json!({
                "path": dir.path().to_str().unwrap(),
                "glob": "*.txt"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("file1.txt"));
        assert!(!result.content.contains("file2.rs"));
    }

    #[tokio::test]
    async fn test_list_files_empty() {
        let dir = TempDir::new().unwrap();

        let tool = ListFilesTool;
        let result = tool
            .execute(serde_json::json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("No files found"));
    }
}
