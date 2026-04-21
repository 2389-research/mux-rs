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
            let bytes = tokio::fs::read(p).await?;
            let mime = if !mime_hint.is_empty() {
                mime_hint.to_string()
            } else {
                mime_from_path(p)
            };
            Ok((STANDARD.encode(bytes), mime))
        }
        MediaSource::Url(url) => {
            validate_fetchable_url(url)?;
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

/// Validate a URL before issuing an outbound fetch. Rejects non-http(s)
/// schemes and literal-IP hosts that fall inside loopback, private, link-local,
/// multicast, or IPv6 ULA ranges (including AWS IMDS at 169.254.169.254).
///
/// KNOWN LIMITATION: hostname-based SSRF (a malicious DNS name that resolves
/// to a private IP) is NOT covered — the fix would require resolving the host
/// here and re-checking post-DNS, which introduces TOCTOU concerns and is out
/// of scope. Callers should only pass URLs from trusted provenance.
fn validate_fetchable_url(url: &str) -> Result<(), LlmError> {
    let parsed = reqwest::Url::parse(url)
        .map_err(|e| LlmError::Configuration(format!("invalid media URL: {}", e)))?;

    // Only http/https. Reject file://, data://, ftp://, etc.
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(LlmError::Configuration(format!(
            "media URL scheme '{}' not allowed; only http/https",
            parsed.scheme()
        )));
    }

    // If the host is a literal IP, block private/loopback/link-local/metadata.
    // reqwest::Url::host_str() returns IPv6 literals wrapped in brackets
    // (e.g. "[::1]"); strip them before IpAddr parsing.
    if let Some(host) = parsed.host_str() {
        let ip_candidate = host
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(host);
        if let Ok(ip) = ip_candidate.parse::<std::net::IpAddr>()
            && !is_public_ip(&ip)
        {
            return Err(LlmError::Configuration(format!(
                "media URL host '{}' is not a public IP",
                host
            )));
        }
        // If it's a DNS name, we don't resolve here — callers should use
        // trusted URLs. Documented limitation.
    }

    Ok(())
}

fn is_public_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                // 169.254.169.254 is link-local, caught above, but be explicit.
                || v4.octets() == [169, 254, 169, 254])
        }
        std::net::IpAddr::V6(v6) => {
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // IPv6 ULA fc00::/7
                || (v6.segments()[0] & 0xfe00) == 0xfc00)
        }
    }
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
        // Use a loopback URL. With the SSRF validator in place, this should
        // surface a Configuration error before any network call is made.
        let src = MediaSource::Url("http://127.0.0.1:1/nonexistent".to_string());
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap();
        let result = resolve_to_base64(&src, "image/png", &http).await;
        assert!(
            matches!(result, Err(LlmError::Configuration(_))),
            "expected Configuration error from SSRF validator, got {:?}",
            result
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
        assert!(matches!(result, Err(LlmError::Configuration(_))));
    }

    #[test]
    fn test_validate_rejects_file_scheme() {
        assert!(validate_fetchable_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_rejects_loopback_ipv4() {
        assert!(validate_fetchable_url("http://127.0.0.1/x").is_err());
    }

    #[test]
    fn test_validate_rejects_metadata_ip() {
        assert!(validate_fetchable_url("http://169.254.169.254/latest/meta-data").is_err());
    }

    #[test]
    fn test_validate_rejects_rfc1918() {
        assert!(validate_fetchable_url("http://10.0.0.1/x").is_err());
        assert!(validate_fetchable_url("http://192.168.1.1/x").is_err());
        assert!(validate_fetchable_url("http://172.16.0.1/x").is_err());
    }

    #[test]
    fn test_validate_rejects_loopback_ipv6() {
        assert!(validate_fetchable_url("http://[::1]/x").is_err());
    }

    #[test]
    fn test_validate_allows_public_host() {
        assert!(validate_fetchable_url("https://example.com/a.png").is_ok());
        assert!(validate_fetchable_url("http://1.1.1.1/x").is_ok());
    }
}
