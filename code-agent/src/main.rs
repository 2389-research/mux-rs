// ABOUTME: Code editing agent with read, write, search, and bash tools.
// ABOUTME: Demonstrates using mux-rs built-in tools.

use std::io::Write;

use anyhow::Result;
use futures::StreamExt;
use rustyline::DefaultEditor;

use mux::prelude::*;

// ============================================================================
// Agent Loop (with streaming)
// ============================================================================

/// Accumulator for building content blocks from stream events.
#[derive(Default)]
struct StreamAccumulator {
    content_blocks: Vec<ContentBlock>,
    current_text: String,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_input: String,
}

impl StreamAccumulator {
    fn handle_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::ContentBlockStart { block, .. } => {
                // Start a new block
                match &block {
                    ContentBlock::Text { .. } => {
                        self.current_text.clear();
                    }
                    ContentBlock::ToolUse { id, name, .. } => {
                        self.current_tool_id = Some(id.clone());
                        self.current_tool_name = Some(name.clone());
                        self.current_tool_input.clear();
                    }
                    _ => {}
                }
            }
            StreamEvent::ContentBlockDelta { text, .. } => {
                // Could be text delta or tool input delta
                if self.current_tool_id.is_some() {
                    // Accumulating tool input JSON
                    self.current_tool_input.push_str(&text);
                } else {
                    // Accumulating text
                    self.current_text.push_str(&text);
                }
            }
            StreamEvent::ContentBlockStop { .. } => {
                // Finalize the current block
                if let (Some(id), Some(name)) =
                    (self.current_tool_id.take(), self.current_tool_name.take())
                {
                    // Parse the accumulated JSON input
                    let input: serde_json::Value =
                        serde_json::from_str(&self.current_tool_input).unwrap_or_default();
                    self.content_blocks
                        .push(ContentBlock::ToolUse { id, name, input });
                    self.current_tool_input.clear();
                } else if !self.current_text.is_empty() {
                    self.content_blocks.push(ContentBlock::Text {
                        text: std::mem::take(&mut self.current_text),
                    });
                }
            }
            _ => {}
        }
    }

    fn into_content(self) -> Vec<ContentBlock> {
        self.content_blocks
    }
}

async fn run_agent_loop(registry: &Registry) -> Result<()> {
    let client = AnthropicClient::from_env()?;
    let mut history: Vec<Message> = Vec::new();
    let mut rl = DefaultEditor::new()?;

    println!("Code Agent (streaming) - Type 'quit' to exit.\n");

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

            // Use streaming API
            let mut stream = client.create_message_stream(&request);
            let mut accumulator = StreamAccumulator::default();
            let mut printed_newline = false;

            while let Some(event) = stream.next().await {
                let event = event?;

                // Print text deltas as they arrive
                if let StreamEvent::ContentBlockDelta { ref text, .. } = event {
                    if accumulator.current_tool_id.is_none() {
                        // Only print if we're in a text block, not tool input
                        if !printed_newline {
                            println!();
                            printed_newline = true;
                        }
                        print!("{}", text);
                        std::io::stdout().flush()?;
                    }
                }

                // Show when tool calls start
                if let StreamEvent::ContentBlockStart {
                    block: ContentBlock::ToolUse { name, .. },
                    ..
                } = &event
                {
                    println!("\n[Calling {}...]", name);
                }

                accumulator.handle_event(event);
            }

            if printed_newline {
                println!("\n");
            }

            let content = accumulator.into_content();

            // Check for tool calls
            if content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
            {
                let mut tool_results = Vec::new();

                for block in content.iter() {
                    if let ContentBlock::ToolUse { id, name, input } = block {
                        match registry.get(name).await {
                            Some(tool) => match tool.execute(input.clone()).await {
                                Ok(result) => {
                                    // Truncate long outputs for display
                                    let display = if result.content.len() > 500 {
                                        format!(
                                            "{}...\n[truncated, {} bytes total]",
                                            &result.content[..500],
                                            result.content.len()
                                        )
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
                    content: content.clone(),
                });
                history.push(Message::tool_results(tool_results));

                continue;
            }

            // No tool calls - add to history and break inner loop
            history.push(Message {
                role: Role::Assistant,
                content,
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

    // Create registry with built-in tools from mux library
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
