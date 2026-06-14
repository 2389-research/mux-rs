// ABOUTME: Opt-in confinement primitives for the built-in tools.
// ABOUTME: Re-exports RootedFs (filesystem jail) and defines ConfinementError.

mod fs;
mod net;

pub use fs::RootedFs;
pub use net::{UrlPolicy, is_globally_routable};

use std::net::IpAddr;
use std::path::PathBuf;

/// Error returned when a tool-supplied path or URL is refused by a confinement.
#[derive(Debug, thiserror::Error)]
pub enum ConfinementError {
    #[error("path {candidate:?} escapes the confinement root {root:?}")]
    EscapesRoot { candidate: PathBuf, root: PathBuf },

    #[error("path {0:?} is not valid within the confinement root")]
    InvalidPath(PathBuf),

    #[error("address {ip} for host {host:?} is blocked by policy")]
    BlockedAddress { host: String, ip: IpAddr },

    #[error("failed to resolve host {host:?}: {source}")]
    Resolve {
        host: String,
        #[source]
        source: std::io::Error,
    },

    #[error("unsupported URL scheme {0:?} (only http/https allowed)")]
    UnsupportedScheme(String),

    #[error("invalid URL: {0}")]
    InvalidUrl(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod fs_test;

#[cfg(test)]
mod net_test;
