// ABOUTME: WebFetchTool - fetches content from URLs.
// ABOUTME: Returns page content with optional HTML-to-text conversion.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for fetching web content from URLs.
pub struct WebFetchTool {
    client: reqwest::Client,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    /// Create a new WebFetchTool with default settings.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("mux-rs/0.2.0")
            .build()
            .expect("Failed to create HTTP client");
        Self { client }
    }

    /// Create with a custom reqwest client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Simple HTML to text conversion - strips tags and decodes entities.
    fn html_to_text(html: &str) -> String {
        // Remove script and style tags with their contents
        let mut result = html.to_string();

        // Remove script tags
        while let Some(start) = result.find("<script") {
            if let Some(end) = result[start..].find("</script>") {
                result = format!("{}{}", &result[..start], &result[start + end + 9..]);
            } else {
                break;
            }
        }

        // Remove style tags
        while let Some(start) = result.find("<style") {
            if let Some(end) = result[start..].find("</style>") {
                result = format!("{}{}", &result[..start], &result[start + end + 8..]);
            } else {
                break;
            }
        }

        // Replace common block elements with newlines
        for tag in &[
            "</p>", "</div>", "</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "<br>",
            "<br/>", "</li>", "</tr>",
        ] {
            result = result.replace(tag, &format!("{}\n", tag));
        }

        // Strip remaining HTML tags
        let mut text = String::new();
        let mut in_tag = false;
        for ch in result.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => text.push(ch),
                _ => {}
            }
        }

        // Decode common HTML entities
        let text = text
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'");

        // Collapse whitespace and trim
        let mut collapsed = String::new();
        let mut prev_whitespace = false;
        let mut prev_newline = false;

        for ch in text.chars() {
            if ch == '\n' {
                if !prev_newline {
                    collapsed.push('\n');
                    prev_newline = true;
                }
                prev_whitespace = true;
            } else if ch.is_whitespace() {
                if !prev_whitespace {
                    collapsed.push(' ');
                    prev_whitespace = true;
                }
                prev_newline = false;
            } else {
                collapsed.push(ch);
                prev_whitespace = false;
                prev_newline = false;
            }
        }

        collapsed.trim().to_string()
    }
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL. Returns the page content as text, optionally converting HTML to plain text."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "convert_html": {
                    "type": "boolean",
                    "description": "Convert HTML to plain text (default: true)",
                    "default": true
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum content length to return (default: 50000)",
                    "default": 50000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            url: String,
            #[serde(default = "default_convert_html")]
            convert_html: bool,
            #[serde(default = "default_max_length")]
            max_length: usize,
        }

        fn default_convert_html() -> bool {
            true
        }
        fn default_max_length() -> usize {
            50000
        }

        let params: Params = serde_json::from_value(params)?;

        // Validate URL
        let url = if !params.url.starts_with("http://") && !params.url.starts_with("https://") {
            format!("https://{}", params.url)
        } else {
            params.url
        };

        // Fetch content
        let response = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => return Ok(ToolResult::error(format!("Failed to fetch URL: {}", e))),
        };

        // Check status
        let status = response.status();
        if !status.is_success() {
            return Ok(ToolResult::error(format!(
                "HTTP error: {} {}",
                status.as_u16(),
                status.canonical_reason().unwrap_or("Unknown")
            )));
        }

        // Get content type
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        // Get body
        let body = match response.text().await {
            Ok(text) => text,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read response: {}", e))),
        };

        // Convert if HTML and requested
        let content = if params.convert_html && content_type.contains("text/html") {
            Self::html_to_text(&body)
        } else {
            body
        };

        // Truncate if needed
        let content = if content.len() > params.max_length {
            format!(
                "{}...\n\n[Content truncated at {} characters, total {} characters]",
                &content[..params.max_length],
                params.max_length,
                content.len()
            )
        } else {
            content
        };

        Ok(ToolResult::text(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_html_to_text() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b>!</p></body></html>";
        let text = WebFetchTool::html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_html_to_text_strips_scripts() {
        let html = "<html><script>alert('xss')</script><body>Content</body></html>";
        let text = WebFetchTool::html_to_text(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("xss"));
    }

    #[test]
    fn test_html_entities() {
        let html = "&lt;tag&gt; &amp; &quot;quoted&quot;";
        let text = WebFetchTool::html_to_text(html);
        assert!(text.contains("<tag>"));
        assert!(text.contains("&"));
        assert!(text.contains("\"quoted\""));
    }

    #[tokio::test]
    async fn test_invalid_url() {
        let tool = WebFetchTool::new();
        let result = tool
            .execute(serde_json::json!({
                "url": "not-a-valid-url-at-all"
            }))
            .await
            .unwrap();

        // Should fail to connect
        assert!(result.is_error);
    }
}
