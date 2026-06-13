// ABOUTME: Opt-in confinement primitives for the built-in tools.
// ABOUTME: Re-exports RootedFs (filesystem jail) and defines ConfinementError.

mod fs;

pub use fs::RootedFs;

use std::path::PathBuf;

/// Error returned when a tool-supplied path or URL is refused by a confinement.
#[derive(Debug, thiserror::Error)]
pub enum ConfinementError {
    #[error("path {candidate:?} escapes the confinement root {root:?}")]
    EscapesRoot { candidate: PathBuf, root: PathBuf },

    #[error("path {0:?} is not valid within the confinement root")]
    InvalidPath(PathBuf),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod fs_test;
