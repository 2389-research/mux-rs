// ABOUTME: ListFilesTool - lists files matching a glob pattern.
// ABOUTME: Shows directories with [dir] prefix.

use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for listing files in a directory with glob patterns.
#[derive(Default)]
pub struct ListFilesTool {
    root: Option<RootedFs>,
}

impl ListFilesTool {
    /// Create an unconfined lister (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lister confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}

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
        let base_path = match &self.root {
            Some(jail) => match jail.resolve(&base_path) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => base_path,
        };
        let glob_pattern = params.glob.unwrap_or_else(|| "*".to_string());
        let full_pattern = std::path::Path::new(&base_path)
            .join(&glob_pattern)
            .to_string_lossy()
            .to_string();

        let mut files = Vec::new();
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
            let prefix = if path.is_dir() { "[dir] " } else { "" };
            files.push(format!("{}{}", prefix, path.display()));
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

        let tool = ListFilesTool::new();
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

        let tool = ListFilesTool::new();
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

        let tool = ListFilesTool::new();
        let result = tool
            .execute(serde_json::json!({
                "path": dir.path().to_str().unwrap()
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("No files found"));
    }

    #[tokio::test]
    async fn test_list_files_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "").unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ListFilesTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({ "path": ".", "glob": "*" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert!(ok.content.contains("inside.txt"));

        // Listing a base path outside the root is refused.
        let blocked = tool
            .execute(serde_json::json!({ "path": "/etc" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
        assert!(
            blocked.content.contains("escapes") || blocked.content.contains("confinement"),
            "expected a confinement rejection, got: {}",
            blocked.content
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_list_files_rooted_blocks_symlinked_base() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ListFilesTool::rooted(jail);

        let blocked = tool
            .execute(serde_json::json!({ "path": "link" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
        assert!(
            blocked.content.contains("escapes") || blocked.content.contains("confinement"),
            "expected a confinement rejection, got: {}",
            blocked.content
        );
    }
}
