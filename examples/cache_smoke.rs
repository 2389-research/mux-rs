//! End-to-end smoke test: construct a mux Request with a cacheable
//! SystemBlock, call Anthropic twice, verify cache_write on call 1 and
//! cache_read on call 2.
//!
//! Run with:
//!   ANTHROPIC_API_KEY=sk-... cargo run --example cache_smoke

use mux::llm::{AnthropicClient, LlmClient, Message, Request, SystemBlock};

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");
    let client = AnthropicClient::new(api_key);

    // Pad to comfortably exceed Anthropic's 1024-token Sonnet cache minimum.
    let large_system = format!(
        "You are a helpful assistant. Be concise.\n\n{}",
        "This is padding text to ensure the system block exceeds the 1024-token minimum cache size that Anthropic requires for Sonnet 4.5 prompt caching to take effect. We want headroom above the threshold. "
            .repeat(80)
    );

    let req = Request::new("claude-sonnet-4-5-20250929")
        .system_block(SystemBlock::cached(large_system))
        .message(Message::user("Say 'hello' and nothing else."))
        .max_tokens(100);

    println!("Call 1 (expect cache_write > 0):");
    let r1 = LlmClient::create_message(&client, &req)
        .await
        .expect("call 1 failed");
    println!(
        "  in={} out={} cache_write={} cache_read={}",
        r1.usage.input_tokens,
        r1.usage.output_tokens,
        r1.usage.cache_write_tokens,
        r1.usage.cache_read_tokens
    );

    println!("\nCall 2 (expect cache_read > 0):");
    let r2 = LlmClient::create_message(&client, &req)
        .await
        .expect("call 2 failed");
    println!(
        "  in={} out={} cache_write={} cache_read={}",
        r2.usage.input_tokens,
        r2.usage.output_tokens,
        r2.usage.cache_write_tokens,
        r2.usage.cache_read_tokens
    );

    println!();
    if r1.usage.cache_write_tokens == 0 {
        eprintln!("✗ Call 1 did not write to cache");
        std::process::exit(2);
    }
    if r2.usage.cache_read_tokens == 0 {
        eprintln!("✗ Call 2 did not read from cache");
        std::process::exit(2);
    }
    println!(
        "✓ End-to-end through mux works. cache_write={} (call 1) → cache_read={} (call 2)",
        r1.usage.cache_write_tokens, r2.usage.cache_read_tokens
    );
}
