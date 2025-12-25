// ABOUTME: Tests for Policy - rules, patterns, conditionals, defaults.
// ABOUTME: Verifies policy evaluation works correctly.

use super::*;

#[test]
fn test_allow_exact() {
    let policy = Policy::builder().allow("read_file").build();

    assert_eq!(
        policy.evaluate("read_file", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("write_file", &serde_json::json!({})),
        Decision::Deny
    );
}

#[test]
fn test_deny_exact() {
    let policy = Policy::builder()
        .deny("dangerous_tool")
        .default(Decision::Allow)
        .build();

    assert_eq!(
        policy.evaluate("dangerous_tool", &serde_json::json!({})),
        Decision::Deny
    );
    assert_eq!(
        policy.evaluate("safe_tool", &serde_json::json!({})),
        Decision::Allow
    );
}

#[test]
fn test_allow_pattern() {
    let policy = Policy::builder().allow_pattern("read_*").build();

    assert_eq!(
        policy.evaluate("read_file", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("read_dir", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("write_file", &serde_json::json!({})),
        Decision::Deny
    );
}

#[test]
fn test_deny_pattern() {
    let policy = Policy::builder()
        .deny_pattern("dangerous_*")
        .default(Decision::Allow)
        .build();

    assert_eq!(
        policy.evaluate("dangerous_delete", &serde_json::json!({})),
        Decision::Deny
    );
    assert_eq!(
        policy.evaluate("safe_operation", &serde_json::json!({})),
        Decision::Allow
    );
}

#[test]
fn test_conditional() {
    let policy = Policy::builder()
        .conditional("bash", |params| {
            let cmd = params["command"].as_str().unwrap_or("");
            if cmd.contains("rm -rf") {
                Decision::Ask
            } else {
                Decision::Allow
            }
        })
        .build();

    assert_eq!(
        policy.evaluate("bash", &serde_json::json!({"command": "ls -la"})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("bash", &serde_json::json!({"command": "rm -rf /"})),
        Decision::Ask
    );
}

#[test]
fn test_rule_order() {
    // First matching rule wins
    let policy = Policy::builder()
        .allow("tool")
        .deny("tool") // Should not be reached
        .build();

    assert_eq!(
        policy.evaluate("tool", &serde_json::json!({})),
        Decision::Allow
    );
}

#[test]
fn test_default_decision() {
    let allow_default = Policy::builder().default(Decision::Allow).build();
    let ask_default = Policy::builder().default(Decision::Ask).build();

    assert_eq!(
        allow_default.evaluate("any", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        ask_default.evaluate("any", &serde_json::json!({})),
        Decision::Ask
    );
}

#[test]
fn test_complex_policy() {
    let policy = Policy::builder()
        .allow("read_file")
        .allow("list_dir")
        .deny_pattern("dangerous_*")
        .allow_pattern("mcp_*")
        .conditional("bash", |params| {
            let cmd = params["command"].as_str().unwrap_or("");
            if cmd.starts_with("sudo") {
                Decision::Deny
            } else if cmd.contains("rm") {
                Decision::Ask
            } else {
                Decision::Allow
            }
        })
        .default(Decision::Ask)
        .build();

    assert_eq!(
        policy.evaluate("read_file", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("dangerous_delete", &serde_json::json!({})),
        Decision::Deny
    );
    assert_eq!(
        policy.evaluate("mcp_fetch", &serde_json::json!({})),
        Decision::Allow
    );
    assert_eq!(
        policy.evaluate("bash", &serde_json::json!({"command": "sudo rm -rf /"})),
        Decision::Deny
    );
    assert_eq!(
        policy.evaluate("bash", &serde_json::json!({"command": "rm file.txt"})),
        Decision::Ask
    );
    assert_eq!(
        policy.evaluate("unknown_tool", &serde_json::json!({})),
        Decision::Ask
    );
}
