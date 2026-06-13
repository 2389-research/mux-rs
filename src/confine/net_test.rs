// ABOUTME: Tests for the SSRF guard - the is_globally_routable deny-list truth table,
// ABOUTME: UrlPolicy host checks, and guarded web_fetch over real loopback sockets.

use crate::confine::{ConfinementError, UrlPolicy, is_globally_routable};
use crate::tool::Tool;
use crate::tools::WebFetchTool;
use std::net::IpAddr;

fn ip(s: &str) -> IpAddr {
    s.parse().unwrap()
}

#[test]
fn globally_routable_truth_table() {
    // Public addresses are routable.
    assert!(is_globally_routable(ip("1.1.1.1")));
    assert!(is_globally_routable(ip("8.8.8.8")));
    assert!(is_globally_routable(ip("2606:4700:4700::1111")));

    // Everything internal is refused.
    assert!(!is_globally_routable(ip("0.0.0.0")));
    assert!(!is_globally_routable(ip("127.0.0.1")));
    assert!(!is_globally_routable(ip("10.0.0.1")));
    assert!(!is_globally_routable(ip("172.16.0.1")));
    assert!(!is_globally_routable(ip("192.168.1.1")));
    assert!(!is_globally_routable(ip("169.254.169.254")));
    assert!(!is_globally_routable(ip("100.64.0.1")));
    assert!(!is_globally_routable(ip("255.255.255.255")));
    assert!(!is_globally_routable(ip("::1")));
    assert!(!is_globally_routable(ip("fe80::1")));
    assert!(!is_globally_routable(ip("fc00::1")));

    // IPv4-mapped IPv6 forms are judged by the embedded v4 address.
    assert!(!is_globally_routable(ip("::ffff:127.0.0.1")));
    assert!(is_globally_routable(ip("::ffff:1.1.1.1")));

    // Deprecated IPv4-compatible IPv6 forms (::a.b.c.d) are likewise judged by
    // the embedded v4 address, so an internal host cannot hide behind one.
    assert!(!is_globally_routable(ip("::127.0.0.1")));
    assert!(!is_globally_routable(ip("::169.254.169.254")));
    assert!(!is_globally_routable(ip("::10.0.0.1")));
    // ...while a public embedded v4 stays routable — the widening must not over-block.
    assert!(is_globally_routable(ip("::1.1.1.1")));
}

#[tokio::test]
async fn check_host_blocks_loopback_literal() {
    let policy = UrlPolicy::public_only();
    let err = policy.check_host("127.0.0.1").await.unwrap_err();
    assert!(
        matches!(err, ConfinementError::BlockedAddress { ip: blocked, .. } if blocked == ip("127.0.0.1"))
    );
}

#[tokio::test]
async fn check_host_blocks_ipv6_loopback_literal_with_brackets() {
    let policy = UrlPolicy::public_only();
    let err = policy.check_host("[::1]").await.unwrap_err();
    assert!(
        matches!(err, ConfinementError::BlockedAddress { ip: blocked, .. } if blocked == ip("::1"))
    );
}

#[tokio::test]
async fn check_host_allows_public_literal() {
    let policy = UrlPolicy::public_only();
    assert!(policy.check_host("1.1.1.1").await.is_ok());
}

#[tokio::test]
async fn custom_policy_predicate_is_honored() {
    use std::net::Ipv4Addr;
    // Allow loopback, deny 10.0.0.1.
    let policy = UrlPolicy::custom(|ip| ip != IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
    assert!(policy.check_host("127.0.0.1").await.is_ok());
    let err = policy.check_host("10.0.0.1").await.unwrap_err();
    assert!(
        matches!(err, ConfinementError::BlockedAddress { ip: blocked, .. } if blocked == ip("10.0.0.1"))
    );
}

/// Spawn a one-shot HTTP/1.1 server on 127.0.0.1 that writes `response` to the
/// first connection, then returns the bound port. Real socket, no mock.
fn spawn_http_once(response: &'static str) -> u16 {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf); // drain the request line/headers
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    port
}

#[tokio::test]
async fn guarded_fetch_blocks_loopback_literal() {
    let tool = WebFetchTool::guarded();
    let result = tool
        .execute(serde_json::json!({ "url": "http://127.0.0.1:9/" }))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.content.contains("blocked"));
}

#[tokio::test]
async fn guarded_fetch_blocks_private_redirect_hop() {
    use std::net::{IpAddr, Ipv4Addr};
    // Server responds with a redirect to a private address.
    let port = spawn_http_once(
        "HTTP/1.1 302 Found\r\nLocation: http://10.0.0.1/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );
    // Allow loopback so the first hop reaches the test server, but deny 10.0.0.1.
    let policy = UrlPolicy::custom(|ip| ip != IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
    let tool = WebFetchTool::with_url_policy(policy);
    let result = tool
        .execute(serde_json::json!({ "url": format!("http://127.0.0.1:{}/", port) }))
        .await
        .unwrap();
    assert!(result.is_error);
    // The error must be a policy block on the redirect target, not merely a
    // connection error that happens to mention the address (which would pass
    // even if per-hop re-validation were removed).
    assert!(
        result.content.contains("blocked by policy"),
        "expected a policy block, got: {}",
        result.content
    );
    assert!(result.content.contains("10.0.0.1"));
}

#[tokio::test]
async fn unguarded_fetch_still_works() {
    let port = spawn_http_once(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello there",
    );
    let tool = WebFetchTool::new();
    let result = tool
        .execute(serde_json::json!({ "url": format!("http://127.0.0.1:{}/", port) }))
        .await
        .unwrap();
    assert!(!result.is_error, "Error: {}", result.content);
    assert!(result.content.contains("hello there"));
}
