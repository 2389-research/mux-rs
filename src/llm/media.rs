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
            let resp = http.get(url).send().await?;
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
}
