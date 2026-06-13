# Opt-In Tool Confinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two opt-in, off-by-default guardrails to mux's built-in tools — a rooted-filesystem jail for the five filesystem tools and an SSRF guard for `web_fetch` — without changing behavior for any existing (non-opted-in) caller.

**Architecture:** A new `mux::confine` module holds two independent mechanisms: `RootedFs` (safe path resolution against a canonicalized root) and `UrlPolicy` + `is_globally_routable` (per-hop IP deny-list for fetches). The five FS tools and `WebFetchTool` change from unit structs to one-field structs carrying an `Option<...>`; their existing `new()` constructors keep current behavior, and new `rooted(...)` / `guarded()` constructors opt into confinement. The FFI layer gains an additive `MuxEngine::new_confined` constructor. The mechanism lives in the library because pre-execution hooks cannot canonicalize a path the tool re-opens itself, nor see per-hop redirect IPs resolved inside `reqwest`.

**Tech Stack:** Rust (edition 2024), `thiserror` for errors, `std::fs`/`std::path` for the jail, `reqwest` 0.12 + `tokio::net::lookup_host` for the guarded fetch, `tempfile` + raw `std::net::TcpListener` for real-filesystem / real-socket tests (no mocks). UniFFI 0.30 for the FFI constructor.

**Reference spec:** `docs/superpowers/specs/2026-06-12-tool-confinement-design.md`

---

## File Structure

**New files:**
- `src/confine/mod.rs` — module root: re-exports + `ConfinementError`.
- `src/confine/fs.rs` — `RootedFs` (filesystem jail mechanism).
- `src/confine/fs_test.rs` — `RootedFs` unit tests (sibling-test convention, gated `#[cfg(test)] mod fs_test;`).
- `src/confine/net.rs` — `UrlPolicy` + `is_globally_routable` (SSRF mechanism).
- `src/confine/net_test.rs` — net unit tests + real-socket `web_fetch` guard tests.
- `docs/confining-mux.md` — operator guide (Phase 5).

**Modified files:**
- `src/lib.rs` — add `pub mod confine;`.
- `src/prelude.rs` — re-export the new public symbols.
- `src/tools/{read_file,write_file,edit,search,list_files}.rs` — `Option<RootedFs>` field, `new()`/`rooted()` constructors, `execute()` integration, inline-test constructor updates.
- `src/tools/web_fetch.rs` — `Option<UrlPolicy>` field, `guarded()`/`with_url_policy()` constructors, manual per-hop redirect loop.
- `mux-ffi/src/engine/mod.rs` — additive `new_confined` constructor + shared private `build` helper.
- `code-agent/src/main.rs` — update the four FS-tool construction calls (`ReadFileTool` → `ReadFileTool::new()`, etc.).
- `tests/public_api_surface.rs` — pin the new `confine` symbols.
- `CHANGELOG.md` — `### Added` entry under a new `## [Unreleased]` section.

**Construction-site note (verified against `main`):** The FS tools are constructed by *name* in exactly two non-test places — `mux-ffi/src/engine/mod.rs:113-119` (the `builtin_tools` vec) and `code-agent/src/main.rs:220-224`. The other FFI sites (`engine/messaging/mod.rs:79` & `:536`, `engine/subagent.rs:212` & `:336`) only iterate `&self.builtin_tools` and clone `Arc`s, so the unit→one-field-struct change does not touch them. Each tool's own inline `#[cfg(test)] mod tests` also constructs the unit value and must be updated in that tool's task.

---

## Conventions for every task

These apply to **every** task below. They are not repeated per-step.

- **Format before every commit.** Run `cargo fmt --all` before `git add`. This repo has **no** pre-commit hook, and the final CI gate is `cargo fmt --all --check` — unformatted code fails CI even when it compiles and every test passes.
- **Never bypass commit checks.** Do not pass `--no-verify`, `--no-hooks`, or `--no-pre-commit-hook` to `git commit`.
- **Confinement violations are values, not panics.** A blocked path or address returns `Ok(ToolResult::error(message))` — never `panic!`, `unwrap`, or a hard `Err`. Every confinement test asserts on `result.is_error` (and usually a substring of `result.content`), not on a Rust panic.
- **The unconfined path stays byte-for-byte unchanged.** The `None` / `::new()` branch of every tool must behave exactly as it does on `main`; the pre-existing tests covering it must stay green untouched. This is what makes "opt-in, zero behavior change" mechanically true.
- **New files start with two `// ABOUTME:` lines** describing what the file does.
- **Per-task gates must be clean before you commit.** Each task's `cargo clippy -p mux --all-targets` must report no warnings, and the task's own `cargo test` must pass, before the commit step.

---

## Phase 1 — `RootedFs` + `ConfinementError`

Builds the filesystem-jail mechanism and its tests in isolation. No tool behavior changes yet.

### Task 1: Module skeleton, `ConfinementError`, `RootedFs::new`

**Files:**
- Create: `src/confine/mod.rs`
- Create: `src/confine/fs.rs`
- Create: `src/confine/fs_test.rs`
- Modify: `src/lib.rs:4-5` (add module declaration)

- [ ] **Step 1: Wire the module into the crate**

Edit `src/lib.rs`, inserting `confine` between `agent` and `coordinator` (the list is alphabetical):

```rust
pub mod agent;
pub mod confine;
pub mod coordinator;
```

- [ ] **Step 2: Create the module root with the error type**

Create `src/confine/mod.rs`:

```rust
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
```

- [ ] **Step 3: Write the failing test for `RootedFs::new`**

Create `src/confine/fs_test.rs`:

```rust
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
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cargo test -p mux confine::fs_test::new_`
Expected: FAIL — does not compile: `cannot find type RootedFs` / `cannot find function new`.

- [ ] **Step 5: Implement `RootedFs::new` + `root()`**

Create `src/confine/fs.rs`:

```rust
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
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cargo test -p mux confine::fs_test::new_`
Expected: PASS (3 tests).

- [ ] **Step 7: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings on the new files.

```bash
git add src/lib.rs src/confine/mod.rs src/confine/fs.rs src/confine/fs_test.rs
git commit -m "feat(confine): add RootedFs::new and ConfinementError"
```

---

### Task 2: `RootedFs::resolve`

**Files:**
- Modify: `src/confine/fs.rs` (add `resolve`)
- Modify: `src/confine/fs_test.rs` (add resolve tests)

- [ ] **Step 1: Write the failing tests**

Append to `src/confine/fs_test.rs`:

```rust
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
```

Add `use crate::confine::ConfinementError;` to the top of `src/confine/fs_test.rs` (next to `use crate::confine::RootedFs;`) — the resolve assertions above are the first place it is used (via `matches!`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mux confine::fs_test::resolve_`
Expected: FAIL — does not compile: `no method named resolve`.

- [ ] **Step 3: Implement `resolve`**

First add the error-type import to the top of `src/confine/fs.rs` (next to `use std::path::{Path, PathBuf};`) — `resolve` is the first method to use it:

```rust
use crate::confine::ConfinementError;
```

Then add this method to the `impl RootedFs` block in `src/confine/fs.rs`:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p mux confine::fs_test::resolve_`
Expected: PASS (7 tests).

- [ ] **Step 5: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/confine/fs.rs src/confine/fs_test.rs
git commit -m "feat(confine): implement RootedFs::resolve with symlink/.. rejection"
```

---

### Task 3: `RootedFs::open_read` (TOCTOU re-check)

**Files:**
- Modify: `src/confine/fs.rs` (add `open_read`)
- Modify: `src/confine/fs_test.rs` (add open_read tests)

- [ ] **Step 1: Write the failing tests**

Append to `src/confine/fs_test.rs`:

```rust
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
    let err = jail.open_read("escape.txt").unwrap_err();
    assert!(matches!(err, ConfinementError::EscapesRoot { .. }));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p mux confine::fs_test::open_read_`
Expected: FAIL — does not compile: `no method named open_read`.

- [ ] **Step 3: Implement `open_read`**

Add this method to the `impl RootedFs` block in `src/confine/fs.rs`:

```rust
    /// Open a file for reading, re-verifying containment at open time. This closes
    /// the resolve→open window against a symlink swapped in after `resolve`. Note
    /// the residual race against an attacker actively swapping a component between
    /// open and the post-open canonicalize is best-effort; fully closing it needs
    /// platform-specific `openat2(RESOLVE_BENEATH)`, which is out of scope.
    pub fn open_read(&self, candidate: impl AsRef<Path>) -> Result<std::fs::File, ConfinementError> {
        let safe = self.resolve(candidate)?;
        let file = std::fs::File::open(&safe)?;
        // Re-verify: the file now exists, so canonicalize resolves any leaf symlink
        // planted between resolve() and open().
        let real = safe.canonicalize()?;
        if !real.starts_with(&self.root) {
            return Err(ConfinementError::EscapesRoot {
                candidate: safe,
                root: self.root.clone(),
            });
        }
        Ok(file)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p mux confine::fs_test::open_read_`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full `confine` suite**

Run: `cargo test -p mux confine::`
Expected: PASS (12 tests across fs_test).

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/confine/fs.rs src/confine/fs_test.rs
git commit -m "feat(confine): add RootedFs::open_read with open-time re-check"
```

---

### Task 4: Export `RootedFs` + `ConfinementError` and pin the API surface

**Files:**
- Modify: `src/prelude.rs:19-22` (add a `confine` re-export)
- Modify: `tests/public_api_surface.rs:34-37` (pin new symbols)

- [ ] **Step 1: Write the failing API-surface assertion**

Edit `tests/public_api_surface.rs`. After the `use mux::tools::{ ... };` block (ends line 37), add:

```rust
use mux::confine::{ConfinementError as _, RootedFs as _};
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux --test public_api_surface`
Expected: FAIL — does not compile: `RootedFs` / `ConfinementError` not found in the prelude path is fine (they resolve via `mux::confine`), but the prelude re-export below is what consumers use; add it next. If this already compiles (the `mux::confine` path exists from Task 1), proceed to Step 3 to add the prelude convenience export.

- [ ] **Step 3: Add the prelude re-export**

Edit `src/prelude.rs`. After the `pub use crate::agent::{ ... };` block (lines 4-6), add a new line:

```rust
pub use crate::confine::{ConfinementError, RootedFs};
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p mux --test public_api_surface`
Expected: PASS.

- [ ] **Step 5: Run the whole `mux` test suite to confirm nothing regressed**

Run: `cargo test -p mux`
Expected: PASS (all pre-existing tests plus the 12 new confine tests).

- [ ] **Step 6: Commit**

```bash
git add src/prelude.rs tests/public_api_surface.rs
git commit -m "feat(confine): export RootedFs and ConfinementError in prelude"
```

---

## Phase 2 — Wire the five filesystem tools

Each task converts one tool from a unit struct to a one-field struct, adds `new()`/`rooted()`, integrates confinement into `execute()` (the `None` branch preserves current behavior byte-for-byte), updates that tool's inline tests, and fixes any external construction site so the workspace keeps compiling.

### Task 5: `ReadFileTool`

**Files:**
- Modify: `src/tools/read_file.rs` (struct, constructors, execute, tests)
- Modify: `mux-ffi/src/engine/mod.rs:114` (construction site)
- Modify: `code-agent/src/main.rs:220` (construction site)

- [ ] **Step 1: Write the failing test**

In `src/tools/read_file.rs`, inside `mod tests`, add:

```rust
    #[tokio::test]
    async fn test_read_file_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "in-root secret").unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ReadFileTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({ "path": "inside.txt" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert!(ok.content.contains("in-root secret"));

        let blocked = tool
            .execute(serde_json::json!({ "path": "/etc/passwd" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux read_file::tests::test_read_file_rooted`
Expected: FAIL — does not compile: no function `rooted`.

- [ ] **Step 3: Convert the struct, add constructors, integrate `execute`**

In `src/tools/read_file.rs`, change the imports and struct. Replace:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for reading file contents.
pub struct ReadFileTool;
```

with:

```rust
use std::io::Read;

use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for reading file contents.
#[derive(Default)]
pub struct ReadFileTool {
    root: Option<RootedFs>,
}

impl ReadFileTool {
    /// Create an unconfined reader (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a reader confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}
```

Then replace the body of `execute` (the `match std::fs::read_to_string(...)` block, lines 42-45) with:

```rust
        let content = match &self.root {
            Some(jail) => match jail.open_read(&params.path) {
                Ok(mut file) => {
                    let mut buf = String::new();
                    match file.read_to_string(&mut buf) {
                        Ok(_) => buf,
                        Err(e) => {
                            return Ok(ToolResult::error(format!("Failed to read file: {}", e)));
                        }
                    }
                }
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => match std::fs::read_to_string(&params.path) {
                Ok(content) => content,
                Err(e) => return Ok(ToolResult::error(format!("Failed to read file: {}", e))),
            },
        };
        Ok(ToolResult::text(content))
```

- [ ] **Step 4: Update the inline tests' constructor**

In `src/tools/read_file.rs` `mod tests`, replace both occurrences of `let tool = ReadFileTool;` with `let tool = ReadFileTool::new();`.

- [ ] **Step 5: Update the two external construction sites**

In `mux-ffi/src/engine/mod.rs`, line 114, replace `Arc::new(ReadFileTool),` with `Arc::new(ReadFileTool::new()),`.

In `code-agent/src/main.rs`, line 220, replace `registry.register(ReadFileTool).await;` with `registry.register(ReadFileTool::new()).await;`.

- [ ] **Step 6: Run to verify it passes (and the workspace compiles)**

Run: `cargo test -p mux read_file`
Expected: PASS (3 tests: success, not_found, rooted).

Run: `cargo build --workspace`
Expected: compiles (mux-ffi and code-agent construction sites updated).

- [ ] **Step 7: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/tools/read_file.rs mux-ffi/src/engine/mod.rs code-agent/src/main.rs
git commit -m "feat(confine): add opt-in rooted confinement to read_file"
```

---

### Task 6: `WriteFileTool`

**Files:**
- Modify: `src/tools/write_file.rs` (struct, constructors, execute, tests)
- Modify: `mux-ffi/src/engine/mod.rs:115` (construction site)
- Modify: `code-agent/src/main.rs:221` (construction site)

- [ ] **Step 1: Write the failing test**

In `src/tools/write_file.rs`, inside `mod tests`, add:

```rust
    #[tokio::test]
    async fn test_write_file_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = WriteFileTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({ "path": "out.txt", "content": "hello" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hello"
        );

        let blocked = tool
            .execute(serde_json::json!({ "path": "/tmp/mux_confine_escape.txt", "content": "x" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
        assert!(!std::path::Path::new("/tmp/mux_confine_escape.txt").exists());
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux write_file::tests::test_write_file_rooted`
Expected: FAIL — does not compile: no function `rooted`.

- [ ] **Step 3: Convert the struct, add constructors, integrate `execute`**

In `src/tools/write_file.rs`, replace:

```rust
use std::path::Path;

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for writing content to files.
pub struct WriteFileTool;
```

with:

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for writing content to files.
#[derive(Default)]
pub struct WriteFileTool {
    root: Option<RootedFs>,
}

impl WriteFileTool {
    /// Create an unconfined writer (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a writer confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}
```

Then, in `execute`, replace the directory-creation and write block (lines 49-63) with a version that resolves through the jail first:

```rust
        let path: PathBuf = match &self.root {
            Some(jail) => match jail.resolve(&params.path) {
                Ok(p) => p,
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => PathBuf::from(&params.path),
        };

        // Create parent directories if needed
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        match std::fs::write(&path, &params.content) {
            Ok(()) => Ok(ToolResult::text(format!(
                "Successfully wrote {} bytes to {}",
                params.content.len(),
                params.path
            ))),
            Err(e) => Ok(ToolResult::error(format!("Failed to write file: {}", e))),
        }
```

- [ ] **Step 4: Update the inline tests' constructor**

In `src/tools/write_file.rs` `mod tests`, replace both occurrences of `let tool = WriteFileTool;` with `let tool = WriteFileTool::new();`.

- [ ] **Step 5: Update the two external construction sites**

In `mux-ffi/src/engine/mod.rs`, line 115, replace `Arc::new(WriteFileTool),` with `Arc::new(WriteFileTool::new()),`.

In `code-agent/src/main.rs`, line 221, replace `registry.register(WriteFileTool).await;` with `registry.register(WriteFileTool::new()).await;`.

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -p mux write_file`
Expected: PASS (3 tests).

Run: `cargo build --workspace`
Expected: compiles.

- [ ] **Step 7: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/tools/write_file.rs mux-ffi/src/engine/mod.rs code-agent/src/main.rs
git commit -m "feat(confine): add opt-in rooted confinement to write_file"
```

---

### Task 7: `EditTool`

**Files:**
- Modify: `src/tools/edit.rs` (struct, constructors, execute, tests)

Note: `EditTool` is **not** an FFI builtin and **not** registered in `code-agent`, so there are no external construction sites to update — only its own inline tests.

- [ ] **Step 1: Write the failing test**

In `src/tools/edit.rs`, inside `mod tests`, add:

```rust
    #[tokio::test]
    async fn test_edit_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("inside.txt");
        std::fs::write(&path, "Hello, world!").unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = EditTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({
                "file_path": "inside.txt",
                "old_string": "world",
                "new_string": "Rust"
            }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Hello, Rust!");

        let blocked = tool
            .execute(serde_json::json!({
                "file_path": "/etc/passwd",
                "old_string": "root",
                "new_string": "pwned"
            }))
            .await
            .unwrap();
        assert!(blocked.is_error);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux edit::tests::test_edit_rooted`
Expected: FAIL — does not compile: no function `rooted`.

- [ ] **Step 3: Convert the struct, add constructors, integrate `execute`**

In `src/tools/edit.rs`, replace:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for precise string replacement in files.
///
/// Unlike WriteFileTool which overwrites entire files, EditTool performs
/// targeted string replacement. It requires the old_string to be unique
/// in the file (unless replace_all is true) to prevent accidental changes.
pub struct EditTool;
```

with:

```rust
use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for precise string replacement in files.
///
/// Unlike WriteFileTool which overwrites entire files, EditTool performs
/// targeted string replacement. It requires the old_string to be unique
/// in the file (unless replace_all is true) to prevent accidental changes.
#[derive(Default)]
pub struct EditTool {
    root: Option<RootedFs>,
}

impl EditTool {
    /// Create an unconfined editor (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an editor confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}
```

Then, in `execute`, immediately after `let params: EditParams = serde_json::from_value(params)?;` (line 64) and before `// Read the file`, insert a resolution step and use the resolved path for both the read and the write. Replace:

```rust
        let params: EditParams = serde_json::from_value(params)?;

        // Read the file
        let content = match std::fs::read_to_string(&params.file_path) {
```

with:

```rust
        let params: EditParams = serde_json::from_value(params)?;

        let file_path: PathBuf = match &self.root {
            Some(jail) => match jail.resolve(&params.file_path) {
                Ok(p) => p,
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => PathBuf::from(&params.file_path),
        };

        // Read the file
        let content = match std::fs::read_to_string(&file_path) {
```

Then replace the write call (line 105) `match std::fs::write(&params.file_path, &new_content) {` with `match std::fs::write(&file_path, &new_content) {`.

The user-facing messages (which interpolate `params.file_path`) are intentionally left unchanged so the model sees the path it supplied.

- [ ] **Step 4: Update the inline tests' constructor**

In `src/tools/edit.rs` `mod tests`, replace all occurrences of `let tool = EditTool;` with `let tool = EditTool::new();` (there are 7).

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p mux edit`
Expected: PASS (8 tests: the 7 existing plus the new rooted test).

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/tools/edit.rs
git commit -m "feat(confine): add opt-in rooted confinement to edit"
```

---

### Task 8: `SearchTool`

**Files:**
- Modify: `src/tools/search.rs` (struct, constructors, execute, tests)
- Modify: `mux-ffi/src/engine/mod.rs:117` (construction site)
- Modify: `code-agent/src/main.rs:222` (construction site)

- [ ] **Step 1: Write the failing test**

In `src/tools/search.rs`, inside `mod tests`, add:

```rust
    #[tokio::test]
    async fn test_search_rooted_excludes_outside_matches() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "FINDME inside").unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "FINDME OUTSIDE_SECRET").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();

        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = SearchTool::rooted(jail);

        let result = tool
            .execute(serde_json::json!({ "pattern": "FINDME" }))
            .await
            .unwrap();
        assert!(!result.is_error, "Error: {}", result.content);
        assert!(result.content.contains("inside"));
        // A symlink-escaped hit must not leak content from outside the root.
        assert!(!result.content.contains("OUTSIDE_SECRET"));
    }

    #[tokio::test]
    async fn test_search_rooted_blocks_outside_base_path() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = SearchTool::rooted(jail);
        let result = tool
            .execute(serde_json::json!({ "pattern": "x", "path": "/etc" }))
            .await
            .unwrap();
        assert!(result.is_error);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux search::tests::test_search_rooted`
Expected: FAIL — does not compile: no function `rooted`.

- [ ] **Step 3: Convert the struct, add constructors, integrate `execute`**

In `src/tools/search.rs`, replace:

```rust
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for searching file contents with regex patterns.
pub struct SearchTool;
```

with:

```rust
use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for searching file contents with regex patterns.
#[derive(Default)]
pub struct SearchTool {
    root: Option<RootedFs>,
}

impl SearchTool {
    /// Create an unconfined search tool (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a search tool confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}
```

Then replace the base-path / pattern setup (lines 53-55):

```rust
        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let glob_pattern = params.glob.unwrap_or_else(|| "**/*".to_string());
        let full_pattern = format!("{}/{}", base_path, glob_pattern);
```

with a jail-resolved base path:

```rust
        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let base_path = match &self.root {
            Some(jail) => match jail.resolve(&base_path) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => base_path,
        };
        let glob_pattern = params.glob.unwrap_or_else(|| "**/*".to_string());
        let full_pattern = format!("{}/{}", base_path, glob_pattern);
```

Finally, replace the **entire** glob loop with a version that drops hits resolving outside the root. The new `if let Some(jail) = &self.root && jail.resolve(&path).is_err()` guard adds one nesting level (the flat `if let Ok(path) = entry && path.is_file() && ...` chain becomes nested), so replace this complete loop:

```rust
        for entry in glob::glob(&full_pattern).unwrap_or_else(|_| glob::glob("").unwrap()) {
            if let Ok(path) = entry
                && path.is_file()
                && let Ok(content) = std::fs::read_to_string(&path)
            {
                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        results.push(format!(
                            "{}:{}: {}",
                            path.display(),
                            line_num + 1,
                            line.trim()
                        ));
                    }
                }
            }
        }
```

with this complete loop:

```rust
        for entry in glob::glob(&full_pattern).unwrap_or_else(|_| glob::glob("").unwrap()) {
            if let Ok(path) = entry {
                // A glob can expand through a symlink to outside the root; drop those.
                if let Some(jail) = &self.root
                    && jail.resolve(&path).is_err()
                {
                    continue;
                }
                if path.is_file()
                    && let Ok(content) = std::fs::read_to_string(&path)
                {
                    for (line_num, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            results.push(format!(
                                "{}:{}: {}",
                                path.display(),
                                line_num + 1,
                                line.trim()
                            ));
                        }
                    }
                }
            }
        }
```

- [ ] **Step 4: Update the inline tests' constructor**

In `src/tools/search.rs` `mod tests`, replace all occurrences of `let tool = SearchTool;` with `let tool = SearchTool::new();` (there are 3).

- [ ] **Step 5: Update the two external construction sites**

In `mux-ffi/src/engine/mod.rs`, line 117, replace `Arc::new(SearchTool),` with `Arc::new(SearchTool::new()),`.

In `code-agent/src/main.rs`, line 222, replace `registry.register(SearchTool).await;` with `registry.register(SearchTool::new()).await;`.

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -p mux search`
Expected: PASS (5 tests).

Run: `cargo build --workspace`
Expected: compiles.

- [ ] **Step 7: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/tools/search.rs mux-ffi/src/engine/mod.rs code-agent/src/main.rs
git commit -m "feat(confine): add opt-in rooted confinement to search"
```

---

### Task 9: `ListFilesTool`

**Files:**
- Modify: `src/tools/list_files.rs` (struct, constructors, execute, tests)
- Modify: `mux-ffi/src/engine/mod.rs:116` (construction site)
- Modify: `code-agent/src/main.rs:223` (construction site)

- [ ] **Step 1: Write the failing test**

In `src/tools/list_files.rs`, inside `mod tests`, add:

```rust
    #[tokio::test]
    async fn test_list_files_rooted_allows_inside_blocks_outside() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("inside.txt"), "").unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ListFilesTool::rooted(jail);

        let ok = tool
            .execute(serde_json::json!({ "path": ".", "glob": "*" }))
            .await
            .unwrap();
        assert!(!ok.is_error, "Error: {}", ok.content);
        assert!(ok.content.contains("inside.txt"));

        // Listing a base path outside the root is refused.
        let blocked = tool
            .execute(serde_json::json!({ "path": "/etc" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_list_files_rooted_blocks_symlinked_base() {
        use crate::confine::RootedFs;
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
        let jail = RootedFs::new(dir.path()).unwrap();
        let tool = ListFilesTool::rooted(jail);

        let blocked = tool
            .execute(serde_json::json!({ "path": "link" }))
            .await
            .unwrap();
        assert!(blocked.is_error);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux list_files::tests::test_list_files_rooted`
Expected: FAIL — does not compile: no function `rooted`.

- [ ] **Step 3: Convert the struct, add constructors, integrate `execute`**

In `src/tools/list_files.rs`, replace:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for listing files in a directory with glob patterns.
pub struct ListFilesTool;
```

with:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::RootedFs;
use crate::tool::{Tool, ToolResult};

/// Tool for listing files in a directory with glob patterns.
#[derive(Default)]
pub struct ListFilesTool {
    root: Option<RootedFs>,
}

impl ListFilesTool {
    /// Create an unconfined lister (current behavior).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lister confined to `root`.
    pub fn rooted(root: RootedFs) -> Self {
        Self { root: Some(root) }
    }
}
```

Then replace the base-path / pattern setup (lines 46-51):

```rust
        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let glob_pattern = params.glob.unwrap_or_else(|| "*".to_string());
        let full_pattern = std::path::Path::new(&base_path)
            .join(&glob_pattern)
            .to_string_lossy()
            .to_string();
```

with a jail-resolved base path:

```rust
        let base_path = params.path.unwrap_or_else(|| ".".to_string());
        let base_path = match &self.root {
            Some(jail) => match jail.resolve(&base_path) {
                Ok(p) => p.to_string_lossy().into_owned(),
                Err(e) => return Ok(ToolResult::error(e.to_string())),
            },
            None => base_path,
        };
        let glob_pattern = params.glob.unwrap_or_else(|| "*".to_string());
        let full_pattern = std::path::Path::new(&base_path)
            .join(&glob_pattern)
            .to_string_lossy()
            .to_string();
```

Then, in the glob loop, drop entries that escape the root. Replace:

```rust
        for path in glob::glob(&full_pattern)
            .unwrap_or_else(|_| glob::glob("").unwrap())
            .flatten()
        {
            let prefix = if path.is_dir() { "[dir] " } else { "" };
            files.push(format!("{}{}", prefix, path.display()));
        }
```

with:

```rust
        for path in glob::glob(&full_pattern)
            .unwrap_or_else(|_| glob::glob("").unwrap())
            .flatten()
        {
            if let Some(jail) = &self.root
                && jail.resolve(&path).is_err()
            {
                continue;
            }
            let prefix = if path.is_dir() { "[dir] " } else { "" };
            files.push(format!("{}{}", prefix, path.display()));
        }
```

- [ ] **Step 4: Update the inline tests' constructor**

In `src/tools/list_files.rs` `mod tests`, replace all occurrences of `let tool = ListFilesTool;` with `let tool = ListFilesTool::new();` (there are 3).

- [ ] **Step 5: Update the two external construction sites**

In `mux-ffi/src/engine/mod.rs`, line 116, replace `Arc::new(ListFilesTool),` with `Arc::new(ListFilesTool::new()),`.

In `code-agent/src/main.rs`, line 223, replace `registry.register(ListFilesTool).await;` with `registry.register(ListFilesTool::new()).await;`.

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -p mux list_files`
Expected: PASS (5 tests).

- [ ] **Step 7: Verify the whole workspace builds and tests green**

Run: `cargo build --workspace`
Expected: compiles (all six construction sites now use `::new()`).

Run: `cargo test --workspace`
Expected: PASS across `mux`, `mux-ffi`, `code-agent`, `agent-test-tui`.

- [ ] **Step 8: Lint and commit**

Run: `cargo clippy --workspace --all-targets`
Expected: no warnings.

```bash
git add src/tools/list_files.rs mux-ffi/src/engine/mod.rs code-agent/src/main.rs
git commit -m "feat(confine): add opt-in rooted confinement to list_files"
```

---

## Phase 3 — `confine::net` + guarded `web_fetch`

### Task 10: `is_globally_routable` + net error variants

**Files:**
- Modify: `src/confine/mod.rs` (add `mod net;`, re-exports, net error variants, `use std::net::IpAddr;`)
- Create: `src/confine/net.rs` (`is_globally_routable`)
- Create: `src/confine/net_test.rs` (truth-table tests)

- [ ] **Step 1: Write the failing truth-table test**

Create `src/confine/net_test.rs`:

```rust
// ABOUTME: Tests for the SSRF deny-list and the guarded web_fetch redirect loop.
// ABOUTME: Uses real local sockets (std::net::TcpListener); no mocks.

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
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux confine::net_test::globally_routable`
Expected: FAIL — does not compile: `cannot find function is_globally_routable`.

- [ ] **Step 3: Create `net.rs` with `is_globally_routable`**

Create `src/confine/net.rs`:

```rust
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
```

- [ ] **Step 4: Wire `net` into the module and add net error variants**

Edit `src/confine/mod.rs`. Change the top:

```rust
mod fs;

pub use fs::RootedFs;

use std::path::PathBuf;
```

to:

```rust
mod fs;
mod net;

pub use fs::RootedFs;
pub use net::{UrlPolicy, is_globally_routable};

use std::net::IpAddr;
use std::path::PathBuf;
```

(`UrlPolicy` is created in Task 11; declaring the re-export now is fine because `net.rs` will define it before this phase's tests that need it. If you prefer strictly-compiling intermediate steps, add `pub use net::is_globally_routable;` here and extend it to include `UrlPolicy` in Task 11. Either way, end state is the line shown above.)

Then extend the `ConfinementError` enum with the network variants:

```rust
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
```

Finally, add the test-module declaration at the bottom of `src/confine/mod.rs`, after `mod fs_test;`:

```rust
#[cfg(test)]
mod net_test;
```

For this task, the re-export of `UrlPolicy` references a type defined in Task 11. To keep Task 10 self-contained and compiling, temporarily re-export only `is_globally_routable`:

```rust
pub use net::is_globally_routable;
```

and add `UrlPolicy` to the re-export in Task 11.

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p mux confine::net_test::globally_routable`
Expected: PASS (1 test).

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/confine/mod.rs src/confine/net.rs src/confine/net_test.rs
git commit -m "feat(confine): add is_globally_routable SSRF deny-list"
```

---

### Task 11: `UrlPolicy`

**Files:**
- Modify: `src/confine/net.rs` (add `UrlPolicy`)
- Modify: `src/confine/mod.rs` (extend the re-export to include `UrlPolicy`)
- Modify: `src/confine/net_test.rs` (add `check_host` tests)

- [ ] **Step 1: Write the failing tests**

Append to `src/confine/net_test.rs`:

```rust
use crate::confine::{ConfinementError, UrlPolicy};

#[tokio::test]
async fn check_host_blocks_loopback_literal() {
    let policy = UrlPolicy::public_only();
    let err = policy.check_host("127.0.0.1").await.unwrap_err();
    assert!(matches!(err, ConfinementError::BlockedAddress { .. }));
}

#[tokio::test]
async fn check_host_blocks_ipv6_loopback_literal_with_brackets() {
    let policy = UrlPolicy::public_only();
    let err = policy.check_host("[::1]").await.unwrap_err();
    assert!(matches!(err, ConfinementError::BlockedAddress { .. }));
}

#[tokio::test]
async fn check_host_allows_public_literal() {
    let policy = UrlPolicy::public_only();
    assert!(policy.check_host("1.1.1.1").await.is_ok());
}

#[tokio::test]
async fn custom_policy_predicate_is_honored() {
    use std::net::{IpAddr, Ipv4Addr};
    // Allow loopback, deny 10.0.0.1.
    let policy = UrlPolicy::custom(|ip| ip != IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
    assert!(policy.check_host("127.0.0.1").await.is_ok());
    let err = policy.check_host("10.0.0.1").await.unwrap_err();
    assert!(matches!(err, ConfinementError::BlockedAddress { .. }));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux confine::net_test::check_host`
Expected: FAIL — does not compile: `cannot find type UrlPolicy`.

- [ ] **Step 3: Implement `UrlPolicy`**

Add to the top of `src/confine/net.rs`, after the existing `use` line:

```rust
use std::sync::Arc;

use crate::confine::ConfinementError;
```

Then append to `src/confine/net.rs`:

```rust
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
```

- [ ] **Step 4: Extend the module re-export**

In `src/confine/mod.rs`, change `pub use net::is_globally_routable;` to:

```rust
pub use net::{UrlPolicy, is_globally_routable};
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p mux confine::net_test::check_host confine::net_test::custom_policy`
Expected: PASS (5 tests). Note: `check_host_allows_public_literal` performs no network I/O (literal `1.1.1.1` is parsed, not resolved), so it is hermetic.

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/confine/net.rs src/confine/mod.rs src/confine/net_test.rs
git commit -m "feat(confine): add UrlPolicy with per-host IP resolution checks"
```

---

### Task 12: Guarded `WebFetchTool`

**Files:**
- Modify: `src/tools/web_fetch.rs` (field, constructors, manual redirect loop, execute branch)
- Modify: `src/confine/net_test.rs` (real-socket guard tests)

- [ ] **Step 1: Write the failing tests**

Append to `src/confine/net_test.rs`:

```rust
use crate::tools::WebFetchTool;
use crate::tool::Tool;

/// Spawn a one-shot HTTP/1.1 server on 127.0.0.1 that writes `response` to the
/// first connection, then returns the bound port. Real socket, no mock.
fn spawn_http_once(response: &'static str) -> u16 {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf); // drain the request line/headers
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    port
}

#[tokio::test]
async fn guarded_fetch_blocks_loopback_literal() {
    let tool = WebFetchTool::guarded();
    let result = tool
        .execute(serde_json::json!({ "url": "http://127.0.0.1:9/" }))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.content.contains("blocked"));
}

#[tokio::test]
async fn guarded_fetch_blocks_private_redirect_hop() {
    use std::net::{IpAddr, Ipv4Addr};
    // Server responds with a redirect to a private address.
    let port = spawn_http_once(
        "HTTP/1.1 302 Found\r\nLocation: http://10.0.0.1/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );
    // Allow loopback so the first hop reaches the test server, but deny 10.0.0.1.
    let policy = UrlPolicy::custom(|ip| ip != IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
    let tool = WebFetchTool::with_url_policy(policy);
    let result = tool
        .execute(serde_json::json!({ "url": format!("http://127.0.0.1:{}/", port) }))
        .await
        .unwrap();
    assert!(result.is_error);
    assert!(result.content.contains("10.0.0.1"));
}

#[tokio::test]
async fn unguarded_fetch_still_works() {
    let port = spawn_http_once(
        "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 11\r\nConnection: close\r\n\r\nhello there",
    );
    let tool = WebFetchTool::new();
    let result = tool
        .execute(serde_json::json!({ "url": format!("http://127.0.0.1:{}/", port) }))
        .await
        .unwrap();
    assert!(!result.is_error, "Error: {}", result.content);
    assert!(result.content.contains("hello there"));
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux confine::net_test::guarded_fetch`
Expected: FAIL — does not compile: no function `guarded` / `with_url_policy`.

- [ ] **Step 3: Add the field and constructors**

In `src/tools/web_fetch.rs`, replace the imports and struct:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolResult};

/// Tool for fetching web content from URLs.
pub struct WebFetchTool {
    client: reqwest::Client,
}
```

with:

```rust
use async_trait::async_trait;
use serde::Deserialize;

use crate::confine::UrlPolicy;
use crate::tool::{Tool, ToolResult};

/// Tool for fetching web content from URLs.
pub struct WebFetchTool {
    client: reqwest::Client,
    /// When set, every request (and every redirect hop) is checked against this
    /// policy and redirects are followed manually. When `None`, redirects are
    /// followed automatically by reqwest with no SSRF check (current behavior).
    policy: Option<UrlPolicy>,
}
```

Then replace the `new` / `with_client` constructors (lines 20-36):

```rust
impl WebFetchTool {
    /// Create a new WebFetchTool with default settings.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("mux-rs/0.2.0")
            .build()
            // Safe: reqwest client construction only fails on catastrophic TLS-backend
            // init. Returning Result here would change the public `new()` signature.
            .expect("Failed to create HTTP client");
        Self { client }
    }

    /// Create with a custom reqwest client.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
```

with:

```rust
impl WebFetchTool {
    /// Create a new WebFetchTool with default settings (unconfined, auto-redirect).
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("mux-rs/0.2.0")
            .build()
            // Safe: reqwest client construction only fails on catastrophic TLS-backend
            // init. Returning Result here would change the public `new()` signature.
            .expect("Failed to create HTTP client");
        Self {
            client,
            policy: None,
        }
    }

    /// Create with a custom reqwest client (unconfined).
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            client,
            policy: None,
        }
    }

    /// Create a guarded fetcher that refuses private/internal addresses
    /// (`UrlPolicy::public_only`) and re-validates every redirect hop.
    pub fn guarded() -> Self {
        Self::with_url_policy(UrlPolicy::public_only())
    }

    /// Create a guarded fetcher with a caller-supplied URL policy. Redirects are
    /// followed manually so each hop's resolved IPs can be re-checked.
    pub fn with_url_policy(policy: UrlPolicy) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("mux-rs/0.2.0")
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("Failed to create HTTP client");
        Self {
            client,
            policy: Some(policy),
        }
    }

    /// Follow redirects manually, re-validating every hop's host against the
    /// policy before connecting. Returns the final non-redirect response, or an
    /// error message suitable for `ToolResult::error`.
    async fn fetch_guarded(
        &self,
        url: &str,
        policy: &UrlPolicy,
    ) -> Result<reqwest::Response, String> {
        let mut current = reqwest::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;
        for _hop in 0..10 {
            match current.scheme() {
                "http" | "https" => {}
                other => return Err(format!("Unsupported URL scheme '{}'", other)),
            }
            let host = current
                .host_str()
                .ok_or_else(|| "URL has no host".to_string())?;
            if let Err(e) = policy.check_host(host).await {
                return Err(e.to_string());
            }
            let resp = self
                .client
                .get(current.clone())
                .send()
                .await
                .map_err(|e| format!("Failed to fetch URL: {}", e))?;
            if resp.status().is_redirection() {
                let location = resp
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .ok_or_else(|| "Redirect response had no usable Location header".to_string())?;
                current = current
                    .join(location)
                    .map_err(|e| format!("Invalid redirect URL: {}", e))?;
                continue;
            }
            return Ok(resp);
        }
        Err("Too many redirects (max 10)".to_string())
    }
```

- [ ] **Step 4: Branch `execute` on the policy**

In `src/tools/web_fetch.rs` `execute`, replace the fetch block (lines 179-183):

```rust
        // Fetch content
        let response = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => return Ok(ToolResult::error(format!("Failed to fetch URL: {}", e))),
        };
```

with:

```rust
        // Fetch content. Guarded mode follows redirects manually and re-validates
        // every hop's resolved IPs; unguarded mode keeps reqwest's auto-redirect.
        let response = match &self.policy {
            Some(policy) => match self.fetch_guarded(&url, policy).await {
                Ok(resp) => resp,
                Err(msg) => return Ok(ToolResult::error(msg)),
            },
            None => match self.client.get(&url).send().await {
                Ok(resp) => resp,
                Err(e) => return Ok(ToolResult::error(format!("Failed to fetch URL: {}", e))),
            },
        };
```

The remainder of `execute` (status check, content-type, body, HTML conversion, truncation) is unchanged and shared by both modes.

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p mux confine::net_test::guarded_fetch confine::net_test::unguarded_fetch`
Expected: PASS (3 tests).

Run: `cargo test -p mux web_fetch`
Expected: PASS (the 4 pre-existing web_fetch tests still green).

- [ ] **Step 6: Lint and commit**

Run: `cargo clippy -p mux --all-targets`
Expected: no warnings.

```bash
git add src/tools/web_fetch.rs src/confine/net_test.rs
git commit -m "feat(confine): add opt-in SSRF guard with per-hop redirect checks to web_fetch"
```

---

### Task 13: Export net symbols and pin the API surface

**Files:**
- Modify: `src/prelude.rs` (add net re-exports)
- Modify: `tests/public_api_surface.rs` (pin `UrlPolicy`, `is_globally_routable`)

- [ ] **Step 1: Update the API-surface pin (failing assertion)**

In `tests/public_api_surface.rs`, change the line added in Task 4:

```rust
use mux::confine::{ConfinementError as _, RootedFs as _};
```

to:

```rust
use mux::confine::{ConfinementError as _, RootedFs as _, UrlPolicy as _};
```

Then, inside `fn public_api_surface_is_stable`, add a pin for the free function (free functions cannot use `as _`):

```rust
    let _ = mux::confine::is_globally_routable as fn(std::net::IpAddr) -> bool;
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux --test public_api_surface`
Expected: FAIL — does not compile: `UrlPolicy` not yet in the prelude is irrelevant here (path is `mux::confine::UrlPolicy`, which exists); this should actually compile already. If it compiles, proceed — the pin is in place. Then add the prelude convenience exports in Step 3 for consumers.

- [ ] **Step 3: Add prelude re-exports**

In `src/prelude.rs`, change the line added in Task 4:

```rust
pub use crate::confine::{ConfinementError, RootedFs};
```

to:

```rust
pub use crate::confine::{ConfinementError, RootedFs, UrlPolicy, is_globally_routable};
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p mux --test public_api_surface`
Expected: PASS.

Run: `cargo test -p mux`
Expected: PASS (full `mux` suite, including all confine tests).

- [ ] **Step 5: Commit**

```bash
git add src/prelude.rs tests/public_api_surface.rs
git commit -m "feat(confine): export UrlPolicy and is_globally_routable in prelude"
```

---

## Phase 4 — FFI `new_confined`

### Task 14: Additive confined engine constructor

**Files:**
- Modify: `mux-ffi/src/engine/mod.rs` (import `RootedFs`, extract `build`, add `new_confined`, add test)

- [ ] **Step 1: Write the failing test**

At the bottom of `mux-ffi/src/engine/mod.rs`, add a test module:

```rust
#[cfg(test)]
mod confine_tests {
    use super::*;

    #[tokio::test]
    async fn new_confined_read_file_refuses_outside_root() {
        let data = tempfile::tempdir().unwrap();
        let root = tempfile::tempdir().unwrap();
        let engine = MuxEngine::new_confined(
            data.path().to_string_lossy().to_string(),
            root.path().to_string_lossy().to_string(),
        )
        .unwrap();

        let read = engine
            .builtin_tools
            .iter()
            .find(|t| t.name() == "read_file")
            .expect("read_file builtin present");

        let result = read
            .execute(serde_json::json!({ "path": "/etc/passwd" }))
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn new_confined_rejects_missing_root() {
        let data = tempfile::tempdir().unwrap();
        let err = MuxEngine::new_confined(
            data.path().to_string_lossy().to_string(),
            "/no/such/confinement/root".to_string(),
        );
        assert!(err.is_err());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p mux-ffi confine_tests`
Expected: FAIL — does not compile: no function `new_confined`.

- [ ] **Step 3: Import `RootedFs`**

In `mux-ffi/src/engine/mod.rs`, change the import (line 27):

```rust
use mux::tools::{BashTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};
```

to add the confine import on the following line:

```rust
use mux::confine::RootedFs;
use mux::tools::{BashTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};
```

- [ ] **Step 4: Extract a shared `build` helper and add `new_confined`**

In the `#[uniffi::export] impl MuxEngine` block, replace the entire current `new` constructor (lines 93-140, from `#[uniffi::constructor]` through the closing `}` of `new`) with two thin constructors:

```rust
    #[uniffi::constructor]
    pub fn new(data_dir: String) -> Result<Arc<Self>, MuxFfiError> {
        Self::build(data_dir, None)
    }

    /// Build an engine whose built-in filesystem tools are confined to `root`.
    /// Additive and non-breaking: `new` is unchanged. `bash` is still registered
    /// — a confined filesystem jail is moot unless the deployment also drops or
    /// OS-sandboxes `bash` (see docs/confining-mux.md).
    #[uniffi::constructor]
    pub fn new_confined(data_dir: String, root: String) -> Result<Arc<Self>, MuxFfiError> {
        let jail = RootedFs::new(&root).map_err(|e| MuxFfiError::Engine {
            message: format!("Invalid confinement root '{}': {}", root, e),
        })?;
        Self::build(data_dir, Some(jail))
    }
```

Then add a **separate, non-exported** `impl MuxEngine` block (immediately after the `#[uniffi::export] impl MuxEngine { ... }` block closes) containing the shared builder, which is the old `new` body with the `builtin_tools` construction made conditional:

```rust
impl MuxEngine {
    /// Shared constructor body for `new` / `new_confined`. When `root` is set, the
    /// filesystem built-ins are confined to it; `bash` is always registered.
    fn build(data_dir: String, root: Option<RootedFs>) -> Result<Arc<Self>, MuxFfiError> {
        let path = PathBuf::from(&data_dir);

        fs::create_dir_all(&path).map_err(|e| MuxFfiError::Engine {
            message: format!("Failed to create data directory: {}", e),
        })?;

        // Create messages directory if it doesn't exist
        let messages_dir = path.join(MESSAGES_DIR);
        fs::create_dir_all(&messages_dir).map_err(|e| MuxFfiError::Engine {
            message: format!("Failed to create messages directory: {}", e),
        })?;

        // Load existing data from disk
        let workspaces = Self::load_workspaces(&path);
        let conversations = Self::load_conversations(&path);
        let message_history = Self::load_all_messages(&path, &conversations);

        // Initialize built-in tools. When confined, the filesystem tools are
        // rooted; bash is always added (its presence defeats the jail — see docs).
        let builtin_tools: Vec<Arc<dyn Tool>> = match &root {
            Some(jail) => vec![
                Arc::new(ReadFileTool::rooted(jail.clone())),
                Arc::new(WriteFileTool::rooted(jail.clone())),
                Arc::new(ListFilesTool::rooted(jail.clone())),
                Arc::new(SearchTool::rooted(jail.clone())),
                Arc::new(BashTool),
            ],
            None => vec![
                Arc::new(ReadFileTool::new()),
                Arc::new(WriteFileTool::new()),
                Arc::new(ListFilesTool::new()),
                Arc::new(SearchTool::new()),
                Arc::new(BashTool),
            ],
        };

        Ok(Arc::new(Self {
            data_dir: path,
            workspaces: Arc::new(RwLock::new(workspaces)),
            conversations: Arc::new(RwLock::new(conversations)),
            message_history: Arc::new(RwLock::new(message_history)),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            mcp_clients: Arc::new(RwLock::new(HashMap::new())),
            workspace_lifecycle: Arc::new(RwLock::new(HashMap::new())),
            pending_approvals: Arc::new(RwLock::new(HashMap::new())),
            builtin_tools,
            agent_configs: Arc::new(RwLock::new(HashMap::new())),
            hook_handler: Arc::new(RwLock::new(None)),
            custom_tools: Arc::new(RwLock::new(HashMap::new())),
            transcript_store: MemoryTranscriptStore::shared(),
            default_provider: Arc::new(RwLock::new(Provider::Anthropic)),
            subagent_event_handler: Arc::new(RwLock::new(None)),
            callback_providers: Arc::new(RwLock::new(HashMap::new())),
            model_context_configs: Arc::new(RwLock::new(HashMap::new())),
        }))
    }
}
```

Note: `ReadFileTool::new()` etc. were already applied to the previous `Arc::new(ReadFileTool)` form during Phase 2 (Tasks 5/6/8/9 edited these exact lines). This task moves them into the `None` arm of `build` and adds the confined `Some` arm. Confirm the final `None` arm matches the post–Phase-2 state (`::new()` forms).

**Copy the struct literal verbatim from the real `new`.** The `Self { ... }` shown above mirrors the current constructor, but the authoritative field list lives in `mux-ffi/src/engine/mod.rs`. Before writing `build`, open that file and copy the existing `new` body's `Self { ... }` exactly — the *only* change is that `builtin_tools` becomes the `match &root { ... }`-built vec shown above. Do not hand-transcribe field names from this plan.

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p mux-ffi confine_tests`
Expected: PASS (2 tests).

- [ ] **Step 6: Verify the FFI bindings still generate and the workspace is green**

Run: `cargo build -p mux-ffi`
Expected: compiles (UniFFI processes the new `#[uniffi::constructor]`; `build` lives in a non-exported impl block so it is not part of the FFI surface).

Run: `cargo test --workspace`
Expected: PASS.

- [ ] **Step 7: Lint and commit**

Run: `cargo clippy --workspace --all-targets`
Expected: no warnings.

```bash
git add mux-ffi/src/engine/mod.rs
git commit -m "feat(mux-ffi): add additive MuxEngine::new_confined for a rooted FS toolset"
```

---

## Phase 5 — Docs + CHANGELOG

### Task 15: Operator guide and changelog entry

**Files:**
- Create: `docs/confining-mux.md`
- Modify: `CHANGELOG.md` (new `## [Unreleased]` section)

- [ ] **Step 1: Write the operator guide**

Create `docs/confining-mux.md`:

````markdown
# Confining mux's built-in tools

mux's built-in tools accept any path and any URL by default. That is the right
default for a trusted, locally-run agent, but a confused or prompt-injected model
can turn `read_file` into `~/.ssh/id_rsa` exfiltration, `write_file` into
arbitrary disk writes, or `web_fetch` into a request to a cloud-metadata endpoint
like `http://169.254.169.254/`.

mux ships **two opt-in guardrails**. Both are **off by default** — existing
constructors are unchanged.

> **This is not a sandbox.** It is in-process defense-in-depth against a *confused
> or injected model*, not against native-code execution. A model that can run an
> un-jailed `bash` can `cat ~/.ssh/id_rsa` or `curl 169.254.169.254` straight
> through the shell, bypassing everything below. **The filesystem jail only has
> teeth in a deployment that also drops or OS-sandboxes `bash`.**

## 1. The filesystem jail (`RootedFs`)

`RootedFs` is a canonicalized root that paths are confined to. Construct the
filesystem tools with `::rooted(...)` instead of `::new()`:

```rust
use mux::confine::RootedFs;
use mux::tools::{EditFileTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};

let jail = RootedFs::new("/srv/agent/workspace")?; // must exist; canonicalized once
let read = ReadFileTool::rooted(jail.clone());
let write = WriteFileTool::rooted(jail.clone());
let edit = EditTool::rooted(jail.clone());
let search = SearchTool::rooted(jail.clone());
let list = ListFilesTool::rooted(jail);
```

A rooted tool resolves every tool-supplied path against the root:

- relative paths are joined onto the root;
- absolute paths must already be inside the root;
- `..` traversal and symlink escapes are rejected;
- a not-yet-existing leaf (for writes) is allowed if its existing ancestor
  canonicalizes within the root;
- `search` / `list_files` additionally drop glob hits that resolve outside the
  root, so a symlink inside the root cannot leak outside content.

Violations are returned as a tool **error result** (not a hard failure), so the
model sees "path … escapes the confinement root" and can adapt.

### Residual race (TOCTOU)

`read_file` re-verifies containment at open time. Against the confused-model
threat this is sufficient. Against a *local attacker actively swapping a symlink*
between resolution and open, it is best-effort: fully closing the race needs
platform-specific `openat2(RESOLVE_BENEATH)` (Linux), which is not portable and
out of scope.

## 2. The `web_fetch` SSRF guard (`UrlPolicy`)

`WebFetchTool::guarded()` refuses to fetch private/internal addresses and
re-validates the host on **every redirect hop**:

```rust
use mux::tools::WebFetchTool;

let fetch = WebFetchTool::guarded(); // UrlPolicy::public_only()
```

The default `UrlPolicy::public_only()` denies unspecified, loopback, RFC1918
private, link-local (including `169.254.169.254`), CGNAT/shared (`100.64/10`),
IPv6 unique-local (`fc00::/7`), broadcast, documentation ranges, and the
IPv4-mapped IPv6 forms of all of these. Supply your own predicate with
`UrlPolicy::custom(|ip| ...)` and `WebFetchTool::with_url_policy(policy)`.

Guarded mode disables reqwest's automatic redirect following and runs a manual
loop (max 10 hops), resolving and re-checking each hop's host before connecting.

### Residual race (DNS rebinding)

reqwest connects by hostname, so a DNS-rebinding attacker could return a public
IP to our resolver and a private IP to the actual connect. Pinning the connection
to the resolved IP is future hardening; for the confused-model threat, per-hop
resolution plus the deny-list is the right level.

## 3. Vetoing `bash` (and other tools) — use `ApprovalHandler`

mux does **not** ship a `bash` jail, because the existing `ApprovalHandler` hook
(wired into the agent runner) already lets a consumer veto any tool call by
name and raw parameters before it executes:

```rust
use async_trait::async_trait;
use mux::permission::{ApprovalContext, ApprovalHandler};

struct DenyShell;

#[async_trait]
impl ApprovalHandler for DenyShell {
    async fn request_approval(
        &self,
        tool: &str,
        params: &serde_json::Value,
        _context: &ApprovalContext,
    ) -> anyhow::Result<bool> {
        if tool == "bash" {
            return Ok(false); // refuse all shell commands
        }
        Ok(true)
    }
}
```

The cleanest hardening is to simply **not register** `bash` in a confined
deployment. The filesystem jail's guarantees are only real once an
un-jailed shell is out of the model's reach.

## 4. FFI (Swift/Kotlin)

The filesystem jail reaches the FFI consumer through an additive constructor:

```swift
let engine = try MuxEngine.newConfined(dataDir: dataDir, root: workspaceRoot)
```

`new_confined` builds the filesystem built-ins rooted at `root`. `bash` is still
registered; drop it or sandbox it at the OS level for the jail to mean anything.
The `web_fetch` SSRF guard is a Rust-side construct (`web_fetch` is not an FFI
built-in this round).
````

Note: in the code sample above, correct the tool name to the actual exported type. Use `EditTool` (not `EditFileTool`) and remove the stray `EditFileTool` from the `use` line — the exported names are `ReadFileTool`, `WriteFileTool`, `EditTool`, `SearchTool`, `ListFilesTool`. Final `use` line:

```rust
use mux::tools::{EditTool, ListFilesTool, ReadFileTool, SearchTool, WriteFileTool};
```

- [ ] **Step 2: Add the changelog entry**

In `CHANGELOG.md`, insert a new section immediately after the intro line (after line 3, `All notable changes to this project are documented in this file.`) and before `## [0.14.0] - 2026-06-09`:

```markdown
## [Unreleased]

### Added

- **Opt-in tool confinement (`mux::confine`).** Two off-by-default guardrails for the built-in tools, with no behavior change for existing callers.
  - **Filesystem jail.** `RootedFs::new(root)` plus `ReadFileTool::rooted`, `WriteFileTool::rooted`, `EditTool::rooted`, `SearchTool::rooted`, and `ListFilesTool::rooted` confine the five filesystem tools to a canonicalized root, rejecting `..` traversal and symlink escapes and dropping glob hits that resolve outside the root. Violations return a tool error result rather than aborting the run.
  - **`web_fetch` SSRF guard.** `WebFetchTool::guarded()` (and `with_url_policy`) deny unspecified/loopback/RFC1918/link-local/CGNAT/ULA addresses via `UrlPolicy::public_only()` / `is_globally_routable`, re-validating every redirect hop with manual redirect following.
  - **FFI.** Additive `MuxEngine::new_confined(data_dir, root)` builds a rooted filesystem toolset for the Swift/Kotlin consumer; the existing `new` is unchanged.
  - New public symbols `RootedFs`, `UrlPolicy`, `ConfinementError`, and `is_globally_routable` are exported from `mux::confine` and the prelude. See `docs/confining-mux.md`. This is in-process defense against a confused/injected model — not a sandbox; the filesystem jail is moot unless `bash` is also dropped or OS-sandboxed.
```

- [ ] **Step 3: Verify the build and full suite once more**

Run: `cargo test --workspace`
Expected: PASS.

Run: `cargo clippy --workspace --all-targets`
Expected: no warnings.

- [ ] **Step 4: Commit**

```bash
git add docs/confining-mux.md CHANGELOG.md
git commit -m "docs(confine): add Confining mux guide and changelog entry"
```

---

## Final verification (before opening the PR)

- [ ] Run the full workspace suite: `cargo test --workspace` → all green.
- [ ] Run the linter as CI does: `cargo clippy --workspace --all-targets` → no warnings (the workspace lints set `clippy::all = "warn"` and `unused = "warn"`; pre-commit enforces).
- [ ] Run the formatter check: `cargo fmt --all --check` → clean.
- [ ] Confirm no behavior change for non-opted-in callers: `cargo test -p mux read_file write_file edit search list_files web_fetch` → all pre-existing tests still pass unchanged.
- [ ] Open the PR for issue #20 via `superpowers:finishing-a-development-branch`.

---

## Self-Review

**Spec coverage** — every spec section maps to a task:

- Mechanism 1 `RootedFs` (type/API, `resolve` algorithm, TOCTOU `open_read`, `ConfinementError`) → Tasks 1-3.
- Tool integration for all five FS tools (one-field struct, `::rooted`, `execute` integration, glob-escape rejection for search/list) → Tasks 5-9.
- Back-compat constructor churn (unit → one-field; update construction sites + inline tests) → Tasks 5-9, plus the two real construction sites identified (`engine/mod.rs`, `code-agent/main.rs`).
- Mechanism 2 `web_fetch` SSRF (default IP policy, `UrlPolicy` API, guarded fetch flow, per-hop re-validation) → Tasks 10-12.
- FFI exposure (`new_confined`, additive, root as `String`, bash still added + warned) → Task 14.
- Public API / prelude changes + `public_api_surface.rs` → Tasks 4 and 13.
- Testing strategy (RootedFs unit tests, per-tool tests, SSRF truth table + guarded/redirect/unguarded, FFI test; all real fs/sockets, no mocks) → covered across Tasks 1-3, 5-12, 14.
- Error behavior (`Ok(ToolResult::error(...))`, never hard `Err`) → every tool task returns error results; verified in tests via `assert!(result.is_error)`.
- Docs (`Confining mux` guide incl. the `ApprovalHandler` bash-veto pattern and the loud "jail is moot unless bash dropped" caveat) + CHANGELOG → Task 15.

**Open items from the spec, resolved here:** `new_confined` keeps `bash` registered and warns loudly in the docs (Task 14 + Task 15), rather than omitting it — the additive, least-surprising choice. DNS-rebinding connection-pinning remains documented-as-residual (Task 15), not built.

**Placeholder scan:** No "TBD"/"implement later"/"add error handling" — every code step shows complete code and exact commands. The one doc-sample correction (the `EditFileTool` → `EditTool` fix in Task 15) is called out explicitly with the corrected code.

**Type consistency:** `RootedFs::{new, resolve, open_read, root}`, `ConfinementError::{EscapesRoot, InvalidPath, BlockedAddress, Resolve, UnsupportedScheme, InvalidUrl, Io}`, `UrlPolicy::{public_only, custom, allows, check_host}`, `is_globally_routable`, and the tool constructors `::new()`/`::rooted()`/`::guarded()`/`::with_url_policy()` are named identically everywhere they appear. `check_host` is `async` throughout (the spec's signature elided `async` for brevity). `MuxEngine::{new, new_confined, build}` are consistent across Task 14.

**TDD discipline:** every task is red (failing/compile-error test) → green (minimal impl) → commit, with the unconfined `None`/`new()` branches preserved byte-for-byte so the "no behavior change" guarantee is mechanically enforced by the pre-existing tests staying green.
