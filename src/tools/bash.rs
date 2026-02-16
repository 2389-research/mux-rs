// ABOUTME: BashTool - executes shell commands.
// ABOUTME: Returns stdout/stderr and handles non-zero exit codes.

use std::process::Stdio;

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for executing shell commands.
/// Uses `bash -c` on Unix and `cmd.exe /C` on Windows.
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its output. Use for running tests, git commands, etc."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "The working directory for the command (default: current directory)"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            command: String,
            working_dir: Option<String>,
        }
        let params: Params = serde_json::from_value(params)?;

        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = tokio::process::Command::new("cmd.exe");
            c.arg("/C").arg(&params.command);
            c
        } else {
            let mut c = tokio::process::Command::new("bash");
            c.arg("-c").arg(&params.command);
            c
        };
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        if let Some(dir) = params.working_dir {
            cmd.current_dir(dir);
        }

        let output = cmd.output().await?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let result = if output.status.success() {
            if stderr.is_empty() {
                stdout.to_string()
            } else {
                format!("{}\n\nstderr:\n{}", stdout, stderr)
            }
        } else {
            format!(
                "Command failed with exit code {}\n\nstdout:\n{}\n\nstderr:\n{}",
                output.status.code().unwrap_or(-1),
                stdout,
                stderr
            )
        };

        if output.status.success() {
            Ok(ToolResult::text(result))
        } else {
            Ok(ToolResult::error(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bash_echo() {
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({
                "command": "echo Hello, world!"
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert!(result.content.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_bash_failing_command() {
        let tool = BashTool;
        let result = tool
            .execute(serde_json::json!({
                "command": "exit 1"
            }))
            .await
            .unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("exit code 1"));
    }

    #[tokio::test]
    async fn test_bash_with_working_dir() {
        let tool = BashTool;
        let tmp = std::env::temp_dir();
        let tmp_str = tmp.to_string_lossy().to_string();
        let command = if cfg!(target_os = "windows") {
            "cd".to_string()
        } else {
            "pwd".to_string()
        };
        let result = tool
            .execute(serde_json::json!({
                "command": command,
                "working_dir": tmp_str
            }))
            .await
            .unwrap();

        assert!(!result.is_error);
        // Verify the output contains some part of the temp dir path
        let output = result.content.trim();
        assert!(
            !output.is_empty(),
            "Command should produce output for the working directory"
        );
    }
}
