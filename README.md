# mux

Agentic infrastructure for Rust. Tool execution, MCP integration, permission-gated approval flows, and orchestration.

## Installation

```toml
[dependencies]
mux = "0.1"
```

## Features

- **Tool Execution**: Define and execute tools with structured input/output handling
- **MCP Integration**: Model Context Protocol client for connecting to external tool servers
- **Permission-Gated Approvals**: Policy engine with patterns, conditionals, and async approval handlers
- **Type Safety**: Leverages Rust's type system for reliable tool definitions
- **Async-First**: Built on tokio for high-performance async operations

## Quick Start

```rust
use mux::prelude::*;

// Define a tool by implementing the Tool trait
struct GreetTool;

#[async_trait::async_trait]
impl Tool for GreetTool {
    fn name(&self) -> &str { "greet" }
    fn description(&self) -> &str { "Greet a person by name" }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Name to greet" }
            },
            "required": ["name"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        let name = params["name"].as_str().unwrap_or("World");
        Ok(ToolResult::text(format!("Hello, {}!", name)))
    }
}

#[tokio::main]
async fn main() -> Result<(), mux::MuxError> {
    // Register tools
    let registry = Registry::new();
    registry.register(GreetTool).await;

    // Set up permissions
    let policy = Policy::builder()
        .allow("greet")
        .deny_pattern("dangerous_*")
        .default(Decision::Deny)
        .build();

    // Execute a tool with policy check
    let tool = registry.get("greet").await.unwrap();
    let params = serde_json::json!({"name": "Alice"});

    if policy.evaluate(tool.name(), &params) == Decision::Allow {
        let result = tool.execute(params).await?;
        println!("{}", result.content);
    }

    Ok(())
}
```

## Status

Under active development. API subject to change.

## License

MIT
