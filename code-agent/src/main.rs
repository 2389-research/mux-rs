// ABOUTME: Code editing agent with read, write, search, and bash tools.
// ABOUTME: Demonstrates building custom tools with mux-rs.

use std::path::Path;
use std::process::Stdio;

use anyhow::Result;
use async_trait::async_trait;
use rustyline::DefaultEditor;
use serde::Deserialize;

use mux::prelude::*;

// ============================================================================
// Read File Tool
// ============================================================================

struct ReadFileTool;

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

// ============================================================================
// Write File Tool
// ============================================================================

struct WriteFileTool;

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

// ============================================================================
// Search Tool (grep-like)
// ============================================================================

struct SearchTool;

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
        let glob_pattern = params.glob.unwrap_or_else(|| "**/*".to_string());
        let full_pattern = format!("{}/{}", base_path, glob_pattern);

        let mut results = Vec::new();
        let regex = match regex::Regex::new(&params.pattern) {
            Ok(r) => r,
            Err(e) => return Ok(ToolResult::error(format!("Invalid regex: {}", e))),
        };

        for entry in glob::glob(&full_pattern).unwrap_or_else(|_| glob::glob("").unwrap()) {
            if let Ok(path) = entry {
                if path.is_file() {
                    if let Ok(content) = std::fs::read_to_string(&path) {
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

// ============================================================================
// List Files Tool
// ============================================================================

struct ListFilesTool;

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

// ============================================================================
// Bash Tool
// ============================================================================

struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output. Use for running tests, git commands, etc."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
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

        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c").arg(&params.command);
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

// ============================================================================
// Agent Loop
// ============================================================================

async fn run_agent_loop(registry: &Registry) -> Result<()> {
    let client = AnthropicClient::from_env()?;
    let mut history: Vec<Message> = Vec::new();
    let mut rl = DefaultEditor::new()?;

    println!("Code Agent - Type 'quit' to exit.\n");

    loop {
        let line = match rl.readline("> ") {
            Ok(line) => line,
            Err(_) => break,
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "quit" || line == "exit" {
            break;
        }

        let _ = rl.add_history_entry(line);
        history.push(Message::user(line));

        // Agent loop: keep calling Claude until no more tool calls
        loop {
            let request = Request::new("claude-sonnet-4-20250514")
                .messages(history.clone())
                .tools(registry.to_definitions().await)
                .system("You are a helpful coding assistant. You have access to tools for reading files, writing files, searching code, listing files, and running bash commands. Use these tools to help the user with their coding tasks. Be concise in your responses.")
                .max_tokens(4096);

            let response = client.create_message(&request).await?;

            // Check for tool calls
            if response.has_tool_use() {
                let mut tool_results = Vec::new();

                for block in response.content.iter() {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        println!("\n[Calling {}...]", name);

                        match registry.get(name).await {
                            Some(tool) => match tool.execute(input.clone()).await {
                                Ok(result) => {
                                    // Truncate long outputs for display
                                    let display = if result.content.len() > 500 {
                                        format!("{}...\n[truncated, {} bytes total]",
                                            &result.content[..500],
                                            result.content.len())
                                    } else {
                                        result.content.clone()
                                    };
                                    println!("{}\n", display);

                                    if result.is_error {
                                        tool_results
                                            .push(ContentBlock::tool_error(id, &result.content));
                                    } else {
                                        tool_results
                                            .push(ContentBlock::tool_result(id, &result.content));
                                    }
                                }
                                Err(e) => {
                                    let error = format!("Error: {}", e);
                                    println!("{}\n", error);
                                    tool_results.push(ContentBlock::tool_error(id, error));
                                }
                            },
                            None => {
                                let error = format!("Tool not found: {}", name);
                                println!("{}\n", error);
                                tool_results.push(ContentBlock::tool_error(id, error));
                            }
                        }
                    }
                }

                // Add messages to history
                history.push(Message {
                    role: Role::Assistant,
                    content: response.content.clone(),
                });
                history.push(Message::tool_results(tool_results));

                continue;
            }

            // No tool calls - print response and break inner loop
            let text = response.text();
            if !text.is_empty() {
                println!("\n{}\n", text);
            }
            history.push(Message {
                role: Role::Assistant,
                content: response.content,
            });
            break;
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Create registry with built-in tools
    let registry = Registry::new();
    registry.register(ReadFileTool).await;
    registry.register(WriteFileTool).await;
    registry.register(SearchTool).await;
    registry.register(ListFilesTool).await;
    registry.register(BashTool).await;

    let tools: Vec<_> = registry.list().await.iter().map(|t| t.to_string()).collect();
    println!("Tools: {}\n", tools.join(", "));

    run_agent_loop(&registry).await
}
