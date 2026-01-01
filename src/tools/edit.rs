// ABOUTME: EditTool - precise string replacement in files.
// ABOUTME: Requires unique matches to prevent accidental overwrites.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for precise string replacement in files.
///
/// Unlike WriteFileTool which overwrites entire files, EditTool performs
/// targeted string replacement. It requires the old_string to be unique
/// in the file (unless replace_all is true) to prevent accidental changes.
pub struct EditTool;

#[derive(Deserialize)]
struct EditParams {
    file_path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing a specific string with new content. \
         The old_string must be unique in the file unless replace_all is true. \
         Use this for precise edits instead of rewriting entire files."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact string to find and replace (must be unique unless replace_all is true)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The string to replace old_string with"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace all occurrences. Default false (requires unique match).",
                    "default": false
                }
            },
            "required": ["file_path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let params: EditParams = serde_json::from_value(params)?;

        // Read the file
        let content = match std::fs::read_to_string(&params.file_path) {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolResult::error(format!(
                    "Failed to read file '{}': {}",
                    params.file_path, e
                )));
            }
        };

        // Count occurrences
        let occurrences = content.matches(&params.old_string).count();

        if occurrences == 0 {
            return Ok(ToolResult::error(format!(
                "String not found in file '{}'. Make sure old_string matches exactly, \
                 including whitespace and indentation.",
                params.file_path
            )));
        }

        if occurrences > 1 && !params.replace_all {
            return Ok(ToolResult::error(format!(
                "String appears {} times in file '{}'. Either:\n\
                 1. Add more surrounding context to make old_string unique, or\n\
                 2. Set replace_all: true to replace all occurrences",
                occurrences, params.file_path
            )));
        }

        // Perform the replacement
        let new_content = if params.replace_all {
            content.replace(&params.old_string, &params.new_string)
        } else {
            content.replacen(&params.old_string, &params.new_string, 1)
        };

        // Write the file
        match std::fs::write(&params.file_path, &new_content) {
            Ok(()) => {
                let msg = if params.replace_all && occurrences > 1 {
                    format!(
                        "Replaced {} occurrences in '{}'",
                        occurrences, params.file_path
                    )
                } else {
                    format!("Successfully edited '{}'", params.file_path)
                };
                Ok(ToolResult::text(msg))
            }
            Err(e) => Ok(ToolResult::error(format!(
                "Failed to write file '{}': {}",
                params.file_path, e
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_edit_unique_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "Hello, world!").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "world",
                "new_string": "Rust"
            }))
            .await
            .unwrap();

        assert!(!result.is_error, "Error: {}", result.content);
        assert!(result.content.contains("Successfully edited"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "Hello, Rust!");
    }

    #[tokio::test]
    async fn test_edit_string_not_found() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "Hello, world!").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "nonexistent",
                "new_string": "replacement"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_edit_non_unique_fails() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo baz foo").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "qux"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("3 times"));
        assert!(result.content.contains("replace_all"));

        // File should be unchanged
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "foo bar foo baz foo");
    }

    #[tokio::test]
    async fn test_edit_replace_all() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "foo bar foo baz foo").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "foo",
                "new_string": "qux",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("3 occurrences"));

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "qux bar qux baz qux");
    }

    #[tokio::test]
    async fn test_edit_file_not_found() {
        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": "/nonexistent/path/file.txt",
                "old_string": "foo",
                "new_string": "bar"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Failed to read"));
    }

    #[tokio::test]
    async fn test_edit_preserves_whitespace() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.rs");
        let content = "fn main() {\n    println!(\"Hello\");\n}\n";
        std::fs::write(&path, content).unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "    println!(\"Hello\");",
                "new_string": "    println!(\"Goodbye\");"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);

        let new_content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(new_content, "fn main() {\n    println!(\"Goodbye\");\n}\n");
    }

    #[tokio::test]
    async fn test_edit_multiline() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "line1\nline2\nline3\n").unwrap();

        let tool = EditTool;
        let result = tool
            .execute(serde_json::json!({
                "file_path": path.to_str().unwrap(),
                "old_string": "line1\nline2",
                "new_string": "first\nsecond"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content, "first\nsecond\nline3\n");
    }
}
