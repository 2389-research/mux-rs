// ABOUTME: RootedFs - a canonicalized filesystem root that paths are confined to.
// ABOUTME: Resolves tool-supplied paths safely, rejecting `..` and symlink escapes.

use std::path::{Path, PathBuf};

/// A canonicalized filesystem root that paths are confined to.
#[derive(Clone, Debug)]
pub struct RootedFs {
    /// Canonicalized, guaranteed to exist and be a directory.
    root: PathBuf,
}

impl RootedFs {
    /// Canonicalize `root` once. Errors if it does not exist or is not a directory.
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "confinement root is not a directory",
            ));
        }
        Ok(Self { root })
    }

    /// The canonicalized root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}
