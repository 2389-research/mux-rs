// ABOUTME: Integration tests for multimodal input across LLM providers.
// ABOUTME: Each test is gated on the relevant API key env var and skips if unset.

use mux::llm::*;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn skip_unless(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => {
            eprintln!("skip: {} not set", key);
            None
        }
    }
}

// ---------- Anthropic ----------

#[tokio::test]
async fn anthropic_image() {
    let Some(key) = skip_unless("ANTHROPIC_API_KEY") else {
        return;
    };
    let client = AnthropicClient::new(key);
    let req = Request::new("claude-haiku-4-5-20251001")
        .message(Message::user_with(vec![
            ContentBlock::text("Describe this image in one short sentence."),
            ContentBlock::image_path(fixture("tiny.png")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("anthropic api call");
    assert!(!resp.text().is_empty(), "expected non-empty response");
}

#[tokio::test]
async fn anthropic_document() {
    let Some(key) = skip_unless("ANTHROPIC_API_KEY") else {
        return;
    };
    let client = AnthropicClient::new(key);
    let req = Request::new("claude-haiku-4-5-20251001")
        .message(Message::user_with(vec![
            ContentBlock::text("Summarize this document in one sentence."),
            ContentBlock::media_path(MediaKind::Document, "application/pdf", fixture("tiny.pdf")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("anthropic doc api call");
    assert!(!resp.text().is_empty());
}

// ---------- OpenAI ----------

#[tokio::test]
async fn openai_image() {
    let Some(key) = skip_unless("OPENAI_API_KEY") else {
        return;
    };
    let client = OpenAIClient::new(key);
    let req = Request::new("gpt-4o-mini")
        .message(Message::user_with(vec![
            ContentBlock::text("Describe this image briefly."),
            ContentBlock::image_path(fixture("tiny.png")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("openai image api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn openai_document() {
    let Some(key) = skip_unless("OPENAI_API_KEY") else {
        return;
    };
    let client = OpenAIClient::new(key);
    let req = Request::new("gpt-4o-mini")
        .message(Message::user_with(vec![
            ContentBlock::text("Summarize this PDF in one sentence."),
            ContentBlock::media_path(MediaKind::Document, "application/pdf", fixture("tiny.pdf")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("openai doc api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn openai_audio() {
    let Some(key) = skip_unless("OPENAI_API_KEY") else {
        return;
    };
    let client = OpenAIClient::new(key);
    // Audio input requires a specific model.
    let req = Request::new("gpt-4o-audio-preview")
        .message(Message::user_with(vec![
            ContentBlock::text("What's in this audio clip? Keep it short."),
            ContentBlock::media_path(MediaKind::Audio, "audio/wav", fixture("tiny.wav")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("openai audio api call");
    assert!(!resp.text().is_empty());
}

// ---------- Gemini ----------

fn gemini_key() -> Option<String> {
    // Gemini accepts either env var per GeminiClient::from_env behavior.
    std::env::var("GEMINI_API_KEY")
        .ok()
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .filter(|v| !v.is_empty())
        .or_else(|| {
            eprintln!("skip: neither GEMINI_API_KEY nor GOOGLE_API_KEY set");
            None
        })
}

#[tokio::test]
async fn gemini_image() {
    let Some(key) = gemini_key() else {
        return;
    };
    let client = GeminiClient::new(key);
    let req = Request::new("gemini-1.5-flash")
        .message(Message::user_with(vec![
            ContentBlock::text("Describe this image briefly."),
            ContentBlock::image_path(fixture("tiny.png")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("gemini image api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn gemini_document() {
    let Some(key) = gemini_key() else {
        return;
    };
    let client = GeminiClient::new(key);
    let req = Request::new("gemini-1.5-flash")
        .message(Message::user_with(vec![
            ContentBlock::text("Summarize this PDF briefly."),
            ContentBlock::media_path(MediaKind::Document, "application/pdf", fixture("tiny.pdf")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("gemini doc api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn gemini_audio() {
    let Some(key) = gemini_key() else {
        return;
    };
    let client = GeminiClient::new(key);
    let req = Request::new("gemini-1.5-flash")
        .message(Message::user_with(vec![
            ContentBlock::text("What's in this audio? Keep it short."),
            ContentBlock::media_path(MediaKind::Audio, "audio/wav", fixture("tiny.wav")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("gemini audio api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn gemini_video() {
    let Some(key) = gemini_key() else {
        return;
    };
    let client = GeminiClient::new(key);
    let req = Request::new("gemini-1.5-flash")
        .message(Message::user_with(vec![
            ContentBlock::text("Describe this video briefly."),
            ContentBlock::media_path(MediaKind::Video, "video/mp4", fixture("tiny.mp4")),
        ]))
        .max_tokens(128);
    let resp = client
        .create_message(&req)
        .await
        .expect("gemini video api call");
    assert!(!resp.text().is_empty());
}

#[tokio::test]
async fn gemini_rejects_url_source_preflight() {
    // Gemini requires inline bytes — the library never fetches URLs for it.
    // No API key needed: the validator rejects before any network call.
    let client = GeminiClient::new("fake-key-not-actually-used");
    let req = Request::new("gemini-1.5-flash")
        .message(Message::user_with(vec![ContentBlock::image_url(
            "https://example.com/x.png",
        )]))
        .max_tokens(64);
    let result = client.create_message(&req).await;
    assert!(matches!(
        result,
        Err(mux::error::LlmError::UnsupportedSource {
            provider: "gemini",
            ..
        })
    ));
}
