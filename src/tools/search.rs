// ABOUTME: SearchTool - grep-like content search in files.
// ABOUTME: Supports regex patterns and glob file matching.

use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for searching file contents with regex patterns.
#[derive(Default)]
pub struct SearchTool {
    root: Option<RootedFs>,
}

impl SearchTool {
    /// Create an unconfined search tool (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a search tool confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> &str {
        "Search for a pattern in files. Supports glob patterns for file matching and regex for content matching."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for in file contents"
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern for files to search (default: **/*)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            pattern: String,
            path: Option<String>,
            glob: Option<String>,
        }
        let params: Params = serde_json::from_value(params)?;

        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let base_path = match &self.root {
            Some(jail) => match jail.resolve(&base_path) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => base_path,
        };
        let glob_pattern = params.glob.unwrap_or_else(|| "**/*".to_string());
        let full_pattern = format!("{}/{}", base_path, glob_pattern);

        let mut results = Vec::new();
        let regex = match Regex::new(&params.pattern) {
            Ok(r) => r,
            Err(e) => return Ok(ToolResult::error(format!("Invalid regex: {}", e))),
        };

        for path in glob::glob(&full_pattern)
            .unwrap_or_else(|_| glob::glob("").unwrap())
            .flatten()
        {
            // A glob can expand through a symlink to outside the root; drop those.
            if let Some(jail) = &self.root
                && jail.resolve(&path).is_err()
            {
                continue;
            }
            if path.is_file()
                && let Ok(content) = std::fs::read_to_string(&path)
            {
                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        results.push(format!(
                            "{}:{}: {}",
                            path.display(),
                            line_num + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }

        if results.is_empty() {
            Ok(ToolResult::text("No matches found"))
        } else {
            Ok(ToolResult::text(format!(
                "Found {} matches:\n{}",
                results.len(),
                results.join("\n")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_search_finds_matches() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(file, "Hello, world!").unwrap();
        writeln!(file, "Goodbye, world!").unwrap();
        writeln!(file, "Hello again!").unwrap();

        let tool = SearchTool::new();
        let result = tool
            .execute(serde_json::json!({
                "pattern": "Hello",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Found 2 matches"));
    }

    #[tokio::test]
    async fn test_search_no_matches() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "Hello, world!").unwrap();

        let tool = SearchTool::new();
        let result = tool
            .execute(serde_json::json!({
                "pattern": "foobar",
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("No matches found"));
    }

    #[tokio::test]
    async fn test_search_invalid_regex() {
        let tool = SearchTool::new();
        let result = tool
            .execute(serde_json::json!({
                "pattern": "[invalid"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("Invalid regex"));
    }

    #[tokio::test]
    async fn test_search_rooted_excludes_outside_matches() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "FINDME inside").unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "FINDME OUTSIDE_SECRET").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();

        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = SearchTool::rooted(jail);

        let result = tool
            .execute(serde_json::json!({ "pattern": "FINDME" }))
            .await
            .unwrap();
        assert!(!result.is_error, "Error: {}", result.content);
        assert!(result.content.contains("inside"));
        // A symlink-escaped hit must not leak content from outside the root.
        assert!(!result.content.contains("OUTSIDE_SECRET"));
    }

    #[tokio::test]
    async fn test_search_rooted_blocks_outside_base_path() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = SearchTool::rooted(jail);
        let result = tool
            .execute(serde_json::json!({ "pattern": "x", "path": "/etc" }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("escapes") || result.content.contains("confinement"),
            "expected a confinement rejection, got: {}",
            result.content
        );
    }
}
