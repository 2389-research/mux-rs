// ABOUTME: Helpers for resolving MediaSource variants to base64-encoded bytes
// ABOUTME: and inferring mime types from URL/path extensions.

use super::MediaSource;
use crate::error::LlmError;
use base64::{Engine as _, engine::general_purpose::STANDARD};

/// Maximum size of a local media file the library will read and encode inline.
/// Provider-side limits vary (Anthropic images 5MB, OpenAI/Gemini 20MB),
/// so we cap at 20MB here — callers with larger assets should pass base64
/// directly or use provider file-ID APIs.
pub const MAX_MEDIA_BYTES: u64 = 20 * 1024 * 1024;

/// Resolve a `MediaSource` to base64-encoded data plus an effective mime type.
///
/// - `Base64`: returned as-is (`mime_hint` passed through).
/// - `Path`: bytes read from disk (up to `MAX_MEDIA_BYTES`), encoded. Mime
///   inferred from extension if `mime_hint` is empty.
/// - `Url`: the library never fetches URLs. Callers that want inline bytes for
///   a URL must fetch themselves and pass `MediaSource::Base64`. If a `Url`
///   reaches this function, a `Configuration` error is returned as a defensive
///   check; providers that accept URL sources natively should pass through
///   before calling here, and providers that require inline bytes should
///   pre-flight reject with `LlmError::UnsupportedSource`.
///
/// The `http` parameter is unused today — kept for signature stability with
/// the previous URL-fetch-capable version and for potential future use.
pub async fn resolve_to_base64(
    source: &MediaSource,
    mime_hint: &str,
    _http: &reqwest::Client,
) -> Result<(String, String), LlmError> {
    match source {
        MediaSource::Base64(data) => Ok((data.clone(), mime_hint.to_string())),
        MediaSource::Path(p) => {
            use tokio::io::AsyncReadExt;
            // Single-handle bounded stream read avoids the TOCTOU window that
            // a separate `metadata()` + `read()` would introduce. We read up
            // to MAX_MEDIA_BYTES + 1 bytes; if we got more than MAX, reject.
            let file = tokio::fs::File::open(p).await?;
            let mut bytes = Vec::new();
            let n = file
                .take(MAX_MEDIA_BYTES + 1)
                .read_to_end(&mut bytes)
                .await?;
            if (n as u64) > MAX_MEDIA_BYTES {
                return Err(LlmError::MediaTooLarge {
                    limit: MAX_MEDIA_BYTES as usize,
                    actual: n as u64,
                });
            }
            let mime = if !mime_hint.is_empty() {
                mime_hint.to_string()
            } else {
                mime_from_path(p)
            };
            Ok((STANDARD.encode(bytes), mime))
        }
        MediaSource::Url(_) => Err(LlmError::Configuration(
            "resolve_to_base64 received a Url source — the library no longer fetches URLs; \
             callers must either pass the URL through to a provider that accepts URL sources \
             natively, or fetch the bytes themselves and pass MediaSource::Base64"
                .to_string(),
        )),
    }
}

fn mime_from_path(p: &std::path::Path) -> String {
    mime_from_ext(p.extension().and_then(|e| e.to_str()).unwrap_or(""))
}

fn mime_from_ext(ext: &str) -> String {
    match ext.to_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "pdf" => "application/pdf",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        _ => "application/octet-stream",
    }
    .to_string()
}

/// Walk a request, replacing any `MediaSource::Path` with `MediaSource::Base64`
/// by reading the file from disk and encoding. `MediaSource::Url` and
/// `MediaSource::Base64` pass through unchanged.
///
/// Returns a new `Request` — caller's copy is unchanged. Providers that accept
/// URL sources natively (Anthropic, OpenAI, OpenRouter) pass them through as-is;
/// providers that require inline bytes (Gemini) must pre-flight reject URL
/// sources with `LlmError::UnsupportedSource` before invoking this helper.
pub async fn resolve_request_media(
    req: &crate::llm::Request,
    http: &reqwest::Client,
) -> Result<crate::llm::Request, LlmError> {
    use crate::llm::ContentBlock;
    let mut out = req.clone();
    for msg in out.messages.iter_mut() {
        for block in msg.content.iter_mut() {
            if let ContentBlock::Media {
                source, mime_type, ..
            } = block
                && matches!(source, MediaSource::Path(_))
            {
                let (data, mime) = resolve_to_base64(source, mime_type, http).await?;
                *source = MediaSource::Base64(data);
                *mime_type = mime;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_base64_passthrough() {
        let src = MediaSource::Base64("aGVsbG8=".to_string());
        let http = reqwest::Client::new();
        let (data, mime) = resolve_to_base64(&src, "image/png", &http).await.unwrap();
        assert_eq!(data, "aGVsbG8=");
        assert_eq!(mime, "image/png");
    }

    #[tokio::test]
    async fn test_path_reads_and_encodes() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"hello").unwrap();
        let src = MediaSource::Path(tmp.path().to_path_buf());
        let http = reqwest::Client::new();
        let (data, _) = resolve_to_base64(&src, "image/png", &http).await.unwrap();
        assert_eq!(data, "aGVsbG8=");
    }

    #[tokio::test]
    async fn test_path_infers_mime_from_extension() {
        let dir = tempfile::tempdir().unwrap();
        let png_path = dir.path().join("fixture.png");
        std::fs::write(&png_path, b"fake-png-bytes").unwrap();
        let src = MediaSource::Path(png_path);
        let http = reqwest::Client::new();
        let (_, mime) = resolve_to_base64(&src, "", &http).await.unwrap();
        assert_eq!(mime, "image/png");
    }

    #[tokio::test]
    async fn test_url_source_returns_configuration_error() {
        // Defense-in-depth: resolve_to_base64 should never be called with a Url
        // source (providers either pass URLs through natively or pre-flight
        // reject), but if it happens we surface a Configuration error rather
        // than attempting a fetch.
        let src = MediaSource::Url("https://example.com/a.png".to_string());
        let http = reqwest::Client::new();
        let result = resolve_to_base64(&src, "image/png", &http).await;
        assert!(matches!(result, Err(LlmError::Configuration(_))));
    }

    #[tokio::test]
    async fn test_path_exceeds_max_media_bytes() {
        // Use set_len to create a sparse file that's MAX_MEDIA_BYTES+1 without
        // actually writing 20MB to disk.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(tmp.path())
            .await
            .unwrap();
        file.set_len(MAX_MEDIA_BYTES + 1).await.unwrap();
        drop(file);
        let src = MediaSource::Path(tmp.path().to_path_buf());
        let http = reqwest::Client::new();
        let result = resolve_to_base64(&src, "image/png", &http).await;
        assert!(
            matches!(result, Err(LlmError::MediaTooLarge { .. })),
            "expected MediaTooLarge, got {:?}",
            result
        );
    }

    #[test]
    fn test_mime_inference_basic() {
        assert_eq!(mime_from_ext("png"), "image/png");
        assert_eq!(mime_from_ext("PNG"), "image/png");
        assert_eq!(mime_from_ext("pdf"), "application/pdf");
        assert_eq!(mime_from_ext("unknown"), "application/octet-stream");
        assert_eq!(mime_from_ext(""), "application/octet-stream");
    }
}
