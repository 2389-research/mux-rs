// ABOUTME: SSRF guard - a deny-list of non-public IP ranges and a UrlPolicy that
// ABOUTME: resolves hosts and refuses any address a confined fetch must not reach.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Returns false for any address a confined fetch must refuse: unspecified,
/// loopback, RFC1918 private, link-local, CGNAT/shared (100.64/10), IPv6
/// unique-local (fc00::/7), broadcast, documentation ranges, and the
/// IPv4-mapped IPv6 forms of any of the above.
pub fn is_globally_routable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_globally_routable_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_globally_routable_v4(v4),
            None => is_globally_routable_v6(v6),
        },
    }
}

fn is_globally_routable_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    // 0.0.0.0/8 (includes the unspecified address).
    if o[0] == 0 {
        return false;
    }
    if ip.is_loopback() || ip.is_private() || ip.is_link_local() || ip.is_broadcast() {
        return false;
    }
    if ip.is_documentation() {
        return false;
    }
    // 100.64.0.0/10 — carrier-grade NAT / shared address space.
    if o[0] == 100 && (o[1] & 0xc0) == 0x40 {
        return false;
    }
    true
}

fn is_globally_routable_v6(ip: Ipv6Addr) -> bool {
    if ip.is_unspecified() || ip.is_loopback() {
        return false;
    }
    let seg = ip.segments();
    // fc00::/7 — unique-local addresses.
    if (seg[0] & 0xfe00) == 0xfc00 {
        return false;
    }
    // fe80::/10 — link-local unicast.
    if (seg[0] & 0xffc0) == 0xfe80 {
        return false;
    }
    true
}
