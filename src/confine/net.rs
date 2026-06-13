// ABOUTME: SSRF guard - a deny-list of non-public IP ranges and a UrlPolicy that
// ABOUTME: resolves hosts and refuses any address a confined fetch must not reach.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Returns false for any address a confined fetch must refuse: unspecified,
/// loopback, RFC1918 private, link-local, CGNAT/shared (100.64/10), IPv6
/// unique-local (fc00::/7), broadcast, documentation ranges, and the
/// IPv4-mapped (`::ffff:a.b.c.d`) and deprecated IPv4-compatible (`::a.b.c.d`)
/// IPv6 forms of any of the above.
pub fn is_globally_routable(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_globally_routable_v4(v4),
        // `to_ipv4()` covers both the IPv4-mapped (`::ffff:a.b.c.d`) and the
        // deprecated IPv4-compatible (`::a.b.c.d`) forms; judge either by its
        // embedded IPv4 address so a loopback/private host cannot slip through
        // wearing IPv6 clothing. Addresses outside `::/96` and `::ffff:0:0/96`
        // (genuine global unicast, link-local, unique-local) have
        // `to_ipv4() == None` and fall through to the v6 check. `::` and `::1`
        // do take the `Some` arm (embedded v4 0.0.0.0 / 0.0.0.1) but stay
        // blocked by the `0.0.0.0/8` rule there.
        IpAddr::V6(v6) => match v6.to_ipv4() {
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
