// ABOUTME: Tests for the SSRF deny-list (is_globally_routable truth table).
// ABOUTME: All assertions use parsed IpAddr values; no I/O or network calls.

use crate::confine::is_globally_routable;
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
