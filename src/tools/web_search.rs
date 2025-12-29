// ABOUTME: WebSearchTool - performs web searches.
// ABOUTME: Uses DuckDuckGo HTML search or configurable search API.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// A single search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

/// Tool for performing web searches.
pub struct WebSearchTool {
    client: reqwest::Client,
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchTool {
    /// Create a new WebSearchTool.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("Mozilla/5.0 (compatible; mux-rs/0.2.0)")
            .build()
            .expect("Failed to create HTTP client");
        Self { client }
    }

    /// Create with a custom reqwest client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// Parse DuckDuckGo HTML search results.
    fn parse_ddg_results(html: &str) -> Vec<SearchResult> {
        let mut results = Vec::new();

        // Look for result divs - DuckDuckGo uses class="result" or similar
        // This is a simple parser that extracts links and text
        let mut remaining = html;

        while let Some(result_start) = remaining.find("class=\"result__a\"") {
            remaining = &remaining[result_start..];

            // Extract URL from href
            let url = if let Some(href_start) = remaining.find("href=\"") {
                let href_content = &remaining[href_start + 6..];
                if let Some(href_end) = href_content.find('"') {
                    let raw_url = &href_content[..href_end];
                    // DDG wraps URLs, extract actual URL
                    if raw_url.contains("uddg=") {
                        if let Some(uddg_pos) = raw_url.find("uddg=") {
                            let encoded = &raw_url[uddg_pos + 5..];
                            if let Some(amp_pos) = encoded.find('&') {
                                urlencoding::decode(&encoded[..amp_pos])
                                    .unwrap_or_default()
                                    .to_string()
                            } else {
                                urlencoding::decode(encoded).unwrap_or_default().to_string()
                            }
                        } else {
                            raw_url.to_string()
                        }
                    } else {
                        raw_url.to_string()
                    }
                } else {
                    remaining = &remaining[1..];
                    continue;
                }
            } else {
                remaining = &remaining[1..];
                continue;
            };

            // Extract title (text between > and </a>)
            let title = if let Some(gt_pos) = remaining.find('>') {
                let after_gt = &remaining[gt_pos + 1..];
                if let Some(end_a) = after_gt.find("</a>") {
                    Self::strip_tags(&after_gt[..end_a])
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            // Extract snippet from result__snippet class
            let snippet = if let Some(snippet_start) = remaining.find("class=\"result__snippet\"") {
                let snippet_content = &remaining[snippet_start..];
                if let Some(gt_pos) = snippet_content.find('>') {
                    let after_gt = &snippet_content[gt_pos + 1..];
                    if let Some(end_div) = after_gt.find("</") {
                        Self::strip_tags(&after_gt[..end_div])
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                }
            } else {
                String::new()
            };

            if !url.is_empty() && !title.is_empty() {
                results.push(SearchResult {
                    title: title.trim().to_string(),
                    url,
                    snippet: snippet.trim().to_string(),
                });
            }

            // Move past this result
            if let Some(next) = remaining.get(1..) {
                remaining = next;
            } else {
                break;
            }
        }

        results
    }

    fn strip_tags(html: &str) -> String {
        let mut text = String::new();
        let mut in_tag = false;
        for ch in html.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => text.push(ch),
                _ => {}
            }
        }
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information. Returns a list of search results with titles, URLs, and snippets."
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> Result<ToolResult, anyhow::Error> {
        #[derive(Deserialize)]
        struct Params {
            query: String,
            #[serde(default = "default_max_results")]
            max_results: usize,
        }

        fn default_max_results() -> usize {
            10
        }

        let params: Params = serde_json::from_value(params)?;

        // Use DuckDuckGo HTML search
        let encoded_query = urlencoding::encode(&params.query);
        let url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

        let response = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => return Ok(ToolResult::error(format!("Search failed: {}", e))),
        };

        if !response.status().is_success() {
            return Ok(ToolResult::error(format!(
                "Search failed with status: {}",
                response.status()
            )));
        }

        let html = match response.text().await {
            Ok(text) => text,
            Err(e) => return Ok(ToolResult::error(format!("Failed to read response: {}", e))),
        };

        let results = Self::parse_ddg_results(&html);
        let results: Vec<_> = results.into_iter().take(params.max_results).collect();

        if results.is_empty() {
            return Ok(ToolResult::text("No results found.".to_string()));
        }

        // Format results
        let mut output = format!("Found {} results for \"{}\":\n\n", results.len(), params.query);
        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   {}\n   {}\n\n",
                i + 1,
                result.title,
                result.url,
                if result.snippet.is_empty() {
                    "(no snippet)"
                } else {
                    &result.snippet
                }
            ));
        }

        Ok(ToolResult::text(output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_tags() {
        let html = "<b>Bold</b> and <i>italic</i>";
        let text = WebSearchTool::strip_tags(html);
        assert_eq!(text, "Bold and italic");
    }

    #[test]
    fn test_parse_empty_results() {
        let results = WebSearchTool::parse_ddg_results("<html><body>No results</body></html>");
        assert!(results.is_empty());
    }

    // Live test - commented out by default
    // #[tokio::test]
    // async fn test_live_search() {
    //     let tool = WebSearchTool::new();
    //     let result = tool
    //         .execute(serde_json::json!({
    //             "query": "rust programming language",
    //             "max_results": 3
    //         }))
    //         .await
    //         .unwrap();
    //     println!("{}", result.content);
    //     assert!(!result.is_error);
    // }
}
