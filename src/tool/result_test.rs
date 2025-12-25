// ABOUTME: Tests for ToolResult - constructors, metadata, defaults.
// ABOUTME: Verifies result structure works correctly.

use super::*;

#[test]
fn test_text_result() {
    let result = ToolResult::text("Hello, world!");
    assert_eq!(result.content, "Hello, world!");
    assert!(!result.is_error);
    assert!(result.metadata.is_empty());
}

#[test]
fn test_error_result() {
    let result = ToolResult::error("Something went wrong");
    assert_eq!(result.content, "Something went wrong");
    assert!(result.is_error);
}

#[test]
fn test_with_metadata() {
    let result = ToolResult::text("output")
        .with_metadata("bytes_read", 1024)
        .with_metadata("cached", true);

    assert_eq!(result.metadata["bytes_read"], 1024);
    assert_eq!(result.metadata["cached"], true);
}

#[test]
fn test_default() {
    let result = ToolResult::default();
    assert_eq!(result.content, "");
    assert!(!result.is_error);
}
