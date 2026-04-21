// ABOUTME: Helpers for resolving MediaSource variants to base64-encoded bytes
// ABOUTME: and inferring mime types from URL/path extensions.

use super::MediaSource;
use crate::error::LlmError;
use base64::{Engine as _, engine::general_purpose::STANDARD};

/// Resolve a `MediaSource` to base64-encoded data plus an effective mime type.
///
/// - `Base64`: returned as-is (`mime_hint` passed through).
/// - `Path`: bytes read from disk, encoded. Mime inferred from extension if
///   `mime_hint` is empty.
/// - `Url`: fetched via the provided client, encoded. Mime inferred from the
///   response's Content-Type header or URL extension if `mime_hint` is empty.
///
/// If `mime_hint` is non-empty it wins for all variants; otherwise mime is
/// inferred from Content-Type (URL) or extension (URL/Path).
pub async fn resolve_to_base64(
    source: &MediaSource,
    mime_hint: &str,
    http: &reqwest::Client,
) -> Result<(String, String), LlmError> {
    match source {
        MediaSource::Base64(data) => Ok((data.clone(), mime_hint.to_string())),
        MediaSource::Path(p) => {
            let bytes = std::fs::read(p)?;
            let mime = if !mime_hint.is_empty() {
                mime_hint.to_string()
            } else {
                mime_from_path(p)
            };
            Ok((STANDARD.encode(bytes), mime))
        }
        MediaSource::Url(url) => {
            // The caller's `http` client may not have a timeout configured.
            // Apply a per-request timeout so a slow remote can't hang the
            // entire LLM call. `RequestBuilder::timeout` overrides the
            // client's default for just this request.
            let resp = http
                .get(url)
                .timeout(std::time::Duration::from_secs(30))
                .send()
                .await?;
            let resp = resp.error_for_status()?;
            let mime = if !mime_hint.is_empty() {
                mime_hint.to_string()
            } else {
                resp.headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|h| h.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| mime_from_url(url))
            };
            let bytes = resp.bytes().await?;
            Ok((STANDARD.encode(bytes), mime))
        }
    }
}

fn mime_from_path(p: &std::path::Path) -> String {
    mime_from_ext(p.extension().and_then(|e| e.to_str()).unwrap_or(""))
}

fn mime_from_url(url: &str) -> String {
    // Strip query string, then take the final extension after the last dot.
    let without_query = url.split('?').next().unwrap_or(url);
    let ext = without_query.rsplit('.').next().unwrap_or("");
    mime_from_ext(ext)
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

#[derive(Copy, Clone)]
enum ResolvePolicy {
    PathsOnly,
    PathsAndUrls,
}

async fn resolve_request_media_inner(
    req: &crate::llm::Request,
    http: &reqwest::Client,
    policy: ResolvePolicy,
) -> Result<crate::llm::Request, LlmError> {
    use crate::llm::{ContentBlock, MediaSource};
    let mut out = req.clone();
    for msg in out.messages.iter_mut() {
        for block in msg.content.iter_mut() {
            if let ContentBlock::Media {
                source, mime_type, ..
            } = block
            {
                let should_resolve = !matches!(
                    (policy, &*source),
                    (_, MediaSource::Base64(_)) | (ResolvePolicy::PathsOnly, MediaSource::Url(_))
                );
                if should_resolve {
                    let (data, mime) = resolve_to_base64(source, mime_type, http).await?;
                    *source = MediaSource::Base64(data);
                    *mime_type = mime;
                }
            }
        }
    }
    Ok(out)
}

/// Walk a request, replacing any `MediaSource::Path` with `MediaSource::Base64`
/// by reading the file from disk and encoding.
///
/// Returns a new `Request` — caller's copy is unchanged. Paths are the only
/// variant resolved; URLs pass through unchanged (providers that need base64
/// for URL sources should handle that inside their serialization).
pub async fn resolve_request_media(
    req: &crate::llm::Request,
    http: &reqwest::Client,
) -> Result<crate::llm::Request, LlmError> {
    resolve_request_media_inner(req, http, ResolvePolicy::PathsOnly).await
}

/// Like `resolve_request_media`, but also fetches any `MediaSource::Url` and
/// inlines it as `Base64`. Use for providers (e.g. Gemini) that don't accept
/// arbitrary URL sources natively.
pub async fn resolve_request_media_fully(
    req: &crate::llm::Request,
    http: &reqwest::Client,
) -> Result<crate::llm::Request, LlmError> {
    resolve_request_media_inner(req, http, ResolvePolicy::PathsAndUrls).await
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

    #[test]
    fn test_mime_inference_basic() {
        assert_eq!(mime_from_ext("png"), "image/png");
        assert_eq!(mime_from_ext("PNG"), "image/png");
        assert_eq!(mime_from_ext("pdf"), "application/pdf");
        assert_eq!(mime_from_ext("unknown"), "application/octet-stream");
        assert_eq!(mime_from_ext(""), "application/octet-stream");
    }

    #[test]
    fn test_mime_from_url_strips_query() {
        assert_eq!(mime_from_url("https://example.com/a.png?v=1"), "image/png");
        assert_eq!(mime_from_url("https://example.com/a.jpg"), "image/jpeg");
        assert_eq!(
            mime_from_url("https://example.com/no-ext"),
            "application/octet-stream"
        );
    }

    #[tokio::test]
    async fn test_url_error_status_returns_error() {
        // Use a bogus local URL that will 404 or fail to connect.
        // Purpose: confirm error_for_status() surfaces the failure
        // rather than silently encoding an error response body.
        let src = MediaSource::Url("http://127.0.0.1:1/nonexistent".to_string());
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let result = resolve_to_base64(&src, "image/png", &http).await;
        assert!(
            result.is_err(),
            "expected an error for an unreachable URL, got Ok"
        );
    }

    #[tokio::test]
    async fn test_resolve_fully_errors_on_bad_url() {
        use crate::llm::{ContentBlock, Message, Request};
        let req = Request::new("x").message(Message::user_with(vec![ContentBlock::image_url(
            "http://127.0.0.1:1/nonexistent",
        )]));
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let result = resolve_request_media_fully(&req, &http).await;
        assert!(result.is_err());
    }
}
