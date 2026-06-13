// ABOUTME: SSRF guard - a deny-list of non-public IP ranges and a UrlPolicy that
// ABOUTME: resolves hosts and refuses any address a confined fetch must not reach.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

use crate::confine::ConfinementError;

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

/// A policy deciding which resolved IP addresses a confined fetch may reach.
#[derive(Clone)]
pub struct UrlPolicy {
    predicate: Arc<dyn Fn(IpAddr) -> bool + Send + Sync>,
}

impl UrlPolicy {
    /// The default policy: allow only globally-routable (public) addresses.
    pub fn public_only() -> Self {
        Self::custom(is_globally_routable)
    }

    /// A policy with a caller-supplied predicate over resolved IP addresses.
    pub fn custom(f: impl Fn(IpAddr) -> bool + Send + Sync + 'static) -> Self {
        Self {
            predicate: Arc::new(f),
        }
    }

    /// Whether a single resolved address is allowed.
    pub fn allows(&self, ip: IpAddr) -> bool {
        (self.predicate)(ip)
    }

    /// Resolve `host` and ensure every resolved address is allowed. IP-literal
    /// hosts (optionally bracketed for IPv6) are checked directly without DNS.
    pub async fn check_host(&self, host: &str) -> Result<(), ConfinementError> {
        let bare = host
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(host);

        if let Ok(ip) = bare.parse::<IpAddr>() {
            return self.check_ip(host, ip);
        }

        // Port is irrelevant to the IP deny-list; 0 is fine for resolution.
        let addrs = tokio::net::lookup_host((bare, 0u16))
            .await
            .map_err(|source| ConfinementError::Resolve {
                host: host.to_string(),
                source,
            })?;

        let mut saw_any = false;
        for addr in addrs {
            saw_any = true;
            self.check_ip(host, addr.ip())?;
        }
        if !saw_any {
            return Err(ConfinementError::Resolve {
                host: host.to_string(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "host resolved to no addresses",
                ),
            });
        }
        Ok(())
    }

    fn check_ip(&self, host: &str, ip: IpAddr) -> Result<(), ConfinementError> {
        if self.allows(ip) {
            Ok(())
        } else {
            Err(ConfinementError::BlockedAddress {
                host: host.to_string(),
                ip,
            })
        }
    }
}
