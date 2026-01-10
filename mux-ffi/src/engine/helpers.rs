// ABOUTME: Shared helper utilities for MuxEngine modules.
// ABOUTME: Contains pure functions that are used across multiple engine submodules.

/// Parse a qualified tool name (server:tool) into its components.
/// Returns None if the name doesn't contain a colon separator.
pub(crate) fn parse_qualified_tool_name(qualified_name: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = qualified_name.splitn(2, ':').collect();
    if parts.len() == 2 {
        Some((parts[0].to_string(), parts[1].to_string()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qualified_tool_name_valid() {
        let result = parse_qualified_tool_name("server:tool");
        assert_eq!(result, Some(("server".to_string(), "tool".to_string())));
    }

    #[test]
    fn test_parse_qualified_tool_name_with_colons() {
        // Tool name with colons (splitn(2) keeps rest together)
        let result = parse_qualified_tool_name("mcp:read:file");
        assert_eq!(result, Some(("mcp".to_string(), "read:file".to_string())));
    }

    #[test]
    fn test_parse_qualified_tool_name_empty_server() {
        let result = parse_qualified_tool_name(":tool");
        assert_eq!(result, Some(("".to_string(), "tool".to_string())));
    }

    #[test]
    fn test_parse_qualified_tool_name_no_colon() {
        let result = parse_qualified_tool_name("read_file");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_qualified_tool_name_empty() {
        let result = parse_qualified_tool_name("");
        assert_eq!(result, None);
    }
}
