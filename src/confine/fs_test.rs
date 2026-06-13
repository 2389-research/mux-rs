// ABOUTME: Tests for RootedFs path-resolution and containment guarantees.
// ABOUTME: Uses real temp directories and symlinks; no mocks.

use crate::confine::RootedFs;
use tempfile::TempDir;

#[test]
fn new_accepts_existing_directory() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    // root() is canonicalized, so it equals the canonical temp path.
    assert_eq!(jail.root(), dir.path().canonicalize().unwrap());
}

#[test]
fn new_rejects_missing_path() {
    let err = RootedFs::new("/no/such/path/at/all").unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

#[test]
fn new_rejects_a_file() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("a_file.txt");
    std::fs::write(&file, "x").unwrap();
    let err = RootedFs::new(&file).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}
