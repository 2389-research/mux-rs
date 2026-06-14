// ABOUTME: RootedFs - a canonicalized filesystem root that paths are confined to.
// ABOUTME: Resolves tool-supplied paths safely, rejecting `..` and symlink escapes.

use crate::confine::ConfinementError;
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

    /// Resolve a tool-supplied path against the root, returning the safe absolute
    /// path to use. Relative paths are joined onto the root; absolute paths must
    /// already be within it. `..` traversal and symlink escapes are rejected. A
    /// not-yet-existing leaf is allowed if its longest existing ancestor
    /// canonicalizes within the root.
    pub fn resolve(&self, candidate: impl AsRef<Path>) -> Result<PathBuf, ConfinementError> {
        let candidate = candidate.as_ref();
        let joined = if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            self.root.join(candidate)
        };

        // Walk from the leaf upward to the longest existing ancestor, recording
        // the non-existing remainder. `file_name()` returns None when a component
        // is `..`/`.`/root, so a `..` in the non-existing tail is rejected here.
        let mut probe = joined.clone();
        let mut remainder: Vec<std::ffi::OsString> = Vec::new();
        while !probe.exists() {
            let name = probe
                .file_name()
                .ok_or_else(|| ConfinementError::InvalidPath(joined.clone()))?
                .to_os_string();
            remainder.push(name);
            match probe.parent() {
                Some(parent) => probe = parent.to_path_buf(),
                None => return Err(ConfinementError::InvalidPath(joined.clone())),
            }
        }

        // Canonicalize the existing portion (fully resolves any symlinks in it),
        // then re-append the non-existing remainder, which cannot be a symlink.
        let mut resolved = probe.canonicalize()?;
        for name in remainder.iter().rev() {
            resolved.push(name);
        }

        // `Path::starts_with` is component-wise, so it cannot be fooled by a
        // sibling like `/root-evil` against root `/root`.
        if !resolved.starts_with(&self.root) {
            return Err(ConfinementError::EscapesRoot {
                candidate: joined,
                root: self.root.clone(),
            });
        }
        Ok(resolved)
    }

    /// Open a file for reading, re-verifying containment at open time. This closes
    /// the resolve→open window against a symlink swapped in after `resolve`. Note
    /// the residual race against an attacker actively swapping a component between
    /// open and the post-open canonicalize is best-effort; fully closing it needs
    /// platform-specific `openat2(RESOLVE_BENEATH)`, which is out of scope.
    pub fn open_read(
        &self,
        candidate: impl AsRef<Path>,
    ) -> Result<std::fs::File, ConfinementError> {
        let safe = self.resolve(candidate)?;
        let file = std::fs::File::open(&safe)?;
        // Re-verify: the file now exists, so canonicalize resolves any leaf symlink
        // planted between resolve() and open(). This guards the post-resolve race;
        // resolve() already rejects a symlink that exists at resolve time, so this
        // branch is only reachable under a real TOCTOU race (not single-threaded tests).
        let real = safe.canonicalize()?;
        if !real.starts_with(&self.root) {
            return Err(ConfinementError::EscapesRoot {
                candidate: safe,
                root: self.root.clone(),
            });
        }
        Ok(file)
    }
}
