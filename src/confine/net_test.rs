// ABOUTME: Tests for the SSRF deny-list (is_globally_routable truth table).
// ABOUTME: All assertions use parsed IpAddr values; no I/O or network calls.

use crate::confine::{ConfinementError, UrlPolicy, is_globally_routable};
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
