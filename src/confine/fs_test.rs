// ABOUTME: Tests for RootedFs path-resolution and containment guarantees.
// ABOUTME: Uses real temp directories and symlinks; no mocks.

use crate::confine::ConfinementError;
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

#[test]
fn resolve_relative_path_inside_root() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let resolved = jail.resolve("sub/file.txt").unwrap();
    assert!(resolved.starts_with(jail.root()));
    assert!(resolved.ends_with("sub/file.txt"));
}

#[test]
fn resolve_absolute_path_inside_root_ok() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let inside = jail.root().join("a.txt");
    let resolved = jail.resolve(&inside).unwrap();
    assert_eq!(resolved, inside);
}

#[test]
fn resolve_absolute_path_outside_root_escapes() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let err = jail.resolve("/etc/passwd").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}

#[test]
fn resolve_absolute_dotdot_escapes() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    // An absolute candidate containing `..` is canonicalized, then containment-checked.
    let err = jail.resolve("/etc/../etc/passwd").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}

#[test]
fn resolve_dotdot_traversal_escapes() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    // Existing-portion `..` is resolved by the OS and caught by the containment check.
    let err = jail.resolve("../../../../etc/passwd").unwrap_err();
    assert!(matches!(
        err,
        ConfinementError::EscapesRoot { .. } | ConfinementError::InvalidPath(_)
    ));
}

#[cfg(unix)]
#[test]
fn resolve_symlink_escape_is_rejected() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let outside = TempDir::new().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
    // A symlink inside the root pointing at an outside directory.
    std::os::unix::fs::symlink(outside.path(), jail.root().join("link")).unwrap();
    let err = jail.resolve("link/secret.txt").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}

#[test]
fn resolve_nonexisting_leaf_with_inside_ancestor_ok() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    std::fs::create_dir(jail.root().join("existing")).unwrap();
    // Leaf does not exist yet (write case); its existing ancestor is inside root.
    let resolved = jail.resolve("existing/new_file.txt").unwrap();
    assert!(resolved.starts_with(jail.root()));
    assert!(resolved.ends_with("existing/new_file.txt"));
}

#[cfg(unix)]
#[test]
fn resolve_nonexisting_leaf_with_outside_ancestor_escapes() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let outside = TempDir::new().unwrap();
    std::os::unix::fs::symlink(outside.path(), jail.root().join("link")).unwrap();
    // Leaf does not exist, but its existing ancestor (link) canonicalizes outside root.
    let err = jail.resolve("link/new_file.txt").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}

#[test]
fn open_read_reads_in_root_file() {
    use std::io::Read;
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    std::fs::write(jail.root().join("hello.txt"), "hi there").unwrap();
    let mut file = jail.open_read("hello.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    assert_eq!(contents, "hi there");
}

#[cfg(unix)]
#[test]
fn open_read_rejects_symlink_escape() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    let outside = TempDir::new().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("secret.txt"),
        jail.root().join("escape.txt"),
    )
    .unwrap();
    // The escaping symlink exists at resolve time, so resolve() rejects it before
    // open(); the post-open re-check covers the post-resolve race (see open_read).
    let err = jail.open_read("escape.txt").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}

#[test]
fn open_read_reports_io_error_for_missing_file() {
    let dir = TempDir::new().unwrap();
    let jail = RootedFs::new(dir.path()).unwrap();
    // Resolves inside the root, but the file does not exist: surfaces as an Io error
    // from File::open, exercising open_read's `?` propagation on the open call.
    let err = jail.open_read("not_here.txt").unwrap_err();
    assert!(matches!(err, ConfinementError::Io(_)));
}
