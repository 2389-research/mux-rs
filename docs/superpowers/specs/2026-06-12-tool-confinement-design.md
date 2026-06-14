# Opt-In Tool Confinement — Design Spec

- **Issue:** #20
- **Date:** 2026-06-12
- **Status:** Proposed
- **Scope decision:** Build the rooted-filesystem jail + `web_fetch` SSRF guard, both opt-in. Skip a dedicated `bash` veto hook (the existing `ApprovalHandler` already covers it; document the pattern instead).

---

## Summary

mux's built-in tools accept any path and any URL with zero validation. A confused or
prompt-injected model can make `read_file` exfiltrate `~/.ssh/id_rsa`, `write_file`
scribble anywhere on disk, or `web_fetch` reach cloud-metadata endpoints like
`http://169.254.169.254/...`. This spec adds two **opt-in** guardrails:

1. **A rooted-filesystem jail** for `read_file`, `write_file`, `edit`, `search`,
   `list_files` — the tools refuse to touch anything outside a caller-chosen root,
   including `..` traversal and symlink escapes.
2. **An SSRF guard** for `web_fetch` — it refuses to fetch private/internal addresses
   and re-validates on every redirect hop.

Both are **off by default**: existing constructors keep their current behavior, so no
runtime behavior changes for anyone who does not opt in.

## Why this lives in the library (mechanism vs. policy)

The codebase already lets a consumer make *decisions* about tool calls: `ApprovalHandler`
(wired into `runner.rs:486-534`) sees the tool name + raw params before `execute()` and
can veto. So a `bash` command veto is already possible today — that is why issue #20
itself calls it "not a true sandbox," and why we are **not** building a bash hook.

What a consumer **cannot** do from that hook:

- **Filesystem:** the hook sees the raw param string and returns a bool. It cannot rewrite
  the path, and the tool then opens the *raw* string itself (`read_file.rs:42`,
  `write_file.rs:56`, `edit.rs:67/105`, glob in `search.rs`). Even a perfect
  canonicalization in the hook is ignored — the tool reopens the original path. Confining
  the built-ins requires changing the tools.
- **Network:** a redirect to `169.254.169.254` happens *inside* `web_fetch`'s `execute()`,
  per hop, invisible to any pre-execution check.

So the **mechanism** (safe path resolution, per-hop IP checks) must live in the library;
only the **policy** (which root, which IP ranges) is the consumer's, and we ship sensible
defaults the consumer can override.

## Goals

- Opt-in confinement for the five filesystem tools and `web_fetch`.
- No behavior change for existing (non-opted-in) callers.
- The hard, easy-to-get-wrong parts (longest-ancestor canonicalization, symlink/`..`
  rejection, per-hop IP re-resolution) implemented once, with tests.
- Batteries included: a one-liner gives you a safe default; power users can override.
- Reachable from the FFI/Swift consumer for the filesystem jail (the in-process consumer
  with no OS sandbox is the reason we are building this).

## Non-Goals

- **Not a true sandbox.** This is in-process, defense-in-depth against a *confused or
  injected model*, not against native code execution. A determined attacker with code
  execution (e.g. via an un-jailed `bash`) is out of scope.
- **The jail is moot if `bash` is still registered.** A model can `cat ~/.ssh/id_rsa` or
  `curl 169.254.169.254` straight through the shell. The docs must say this plainly: the
  filesystem jail only has teeth in a deployment that also drops or OS-sandboxes `bash`.
- **No new trait on `Tool`,** no change to the `execute(params)` signature, no async
  approval rework. Confinement is constructor-level configuration on the tools.
- Bash confinement (documented, not built).

## Threat Model

| Actor | In scope? | Mitigation |
|---|---|---|
| Confused/prompt-injected model issuing tool calls | **Yes** | fs jail + SSRF guard |
| Model using an un-jailed `bash` to bypass the jail | No (documented) | drop/OS-sandbox bash |
| Local attacker actively racing symlinks (TOCTOU) | Best-effort only | re-check at open; see residual-race note |
| Native-code attacker / memory-safety exploits | No | OS-level isolation |

---

## Architecture

Two independent mechanisms in a new top-level `confine` module, each consumed by the
relevant tools. Nothing else in the crate changes shape.

```
src/confine/
  mod.rs    — re-exports; ConfinementError
  fs.rs     — RootedFs (filesystem jail mechanism)
  net.rs    — UrlPolicy + is_globally_routable (SSRF mechanism)
```

- `src/tools/{read_file,write_file,edit,search,list_files}.rs` — gain an
  `Option<RootedFs>` and a `::rooted(...)` constructor; call `RootedFs::resolve` (+ safe
  open) at the top of `execute()`.
- `src/tools/web_fetch.rs` — gains an `Option<UrlPolicy>` and a guarded constructor;
  replaces reqwest's automatic redirect following with a manual per-hop loop.
- `mux-ffi/src/engine/mod.rs` — gains an additive, non-breaking confined entry point that
  builds the four FS builtins with a root.

---

## Mechanism 1: `RootedFs` (filesystem jail)

### Type & API (`src/confine/fs.rs`)

```rust
/// A canonicalized filesystem root that paths are confined to.
#[derive(Clone, Debug)]
pub struct RootedFs {
    root: std::path::PathBuf, // canonicalized, guaranteed to exist
}

impl RootedFs {
    /// Canonicalize `root` once. Errors if it does not exist / is not a dir.
    pub fn new(root: impl AsRef<std::path::Path>) -> std::io::Result<Self>;

    /// Resolve a tool-supplied path against the root.
    /// - relative paths are joined onto the root,
    /// - absolute paths must already be within the root,
    /// - `..` traversal and symlink escapes are rejected,
    /// - a not-yet-existing leaf (for writes) is allowed if its existing
    ///   ancestor canonicalizes within the root.
    /// Returns the safe absolute path to use.
    pub fn resolve(&self, candidate: impl AsRef<std::path::Path>)
        -> Result<std::path::PathBuf, ConfinementError>;

    /// Open a file for reading, re-verifying containment at open time (TOCTOU re-check).
    pub fn open_read(&self, candidate: impl AsRef<std::path::Path>)
        -> Result<std::fs::File, ConfinementError>;

    pub fn root(&self) -> &std::path::Path;
}
```

### `resolve` algorithm

1. If `candidate` is relative, join onto `self.root`; if absolute, take as-is.
2. Walk from the leaf upward to find the **longest existing ancestor**; `canonicalize`
   that ancestor (this fully resolves any symlinks in the existing portion).
3. Re-append the non-existing remainder component-by-component, rejecting any `..` or
   absolute reset; the remainder cannot contain symlinks because it does not exist yet.
4. Assert the resulting absolute path is `self.root` or starts with `self.root` +
   `MAIN_SEPARATOR`. Otherwise → `ConfinementError::EscapesRoot`.
5. Return the safe path.

### TOCTOU handling & residual race (stated honestly)

`open_read` re-runs containment at open time and opens through the validated path. Against
the *confused-model* threat this is sufficient. Against a *local attacker actively
swapping a symlink* between resolve and open, this is best-effort: fully closing the race
needs platform-specific `openat2(RESOLVE_BENEATH)` (Linux) / equivalent, which is not
portable to macOS and is out of scope. The docs state this limitation.

### Error type (`src/confine/mod.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum ConfinementError {
    #[error("path {candidate:?} escapes the confinement root {root:?}")]
    EscapesRoot { candidate: PathBuf, root: PathBuf },
    #[error("path {0:?} is not valid within the confinement root")]
    InvalidPath(PathBuf),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

### Tool integration

Each FS tool changes from a unit struct to a one-field struct, preserving a no-root
constructor:

```rust
#[derive(Default)]
pub struct ReadFileTool { root: Option<RootedFs> }

impl ReadFileTool {
    pub fn new() -> Self { Self::default() }            // unconfined (current behavior)
    pub fn rooted(root: RootedFs) -> Self { Self { root: Some(root) } }
}
```

In `execute()`, before any fs access:

```rust
let path = match &self.root {
    Some(jail) => match jail.resolve(&params.path) {
        Ok(p) => p,
        Err(e) => return Ok(ToolResult::error(e.to_string())),
    },
    None => PathBuf::from(&params.path), // unconfined path unchanged
};
```

`search` / `list_files` resolve their `base_path` through the jail and additionally
**reject glob results that fall outside the root** (the glob can expand to symlinked
escapes), so `path:"/" glob:"**/.ssh/id_*"` yields nothing under a jail.

### Back-compat note (one real wrinkle)

The FS tools are currently **unit structs** used as `Arc::new(ReadFileTool)`. Converting
them to one-field structs means `Arc::new(ReadFileTool)` (unit value) no longer compiles;
callers use `ReadFileTool::new()` instead. This is a **source-level** change at the 6
internal registration sites + `tests/public_api_surface.rs`, which we own and update. The
*behavioral* contract ("unconfined by default") is unchanged. `#[derive(Default)]` keeps
`ReadFileTool::default()` working too. This is called out so reviewers expect the
constructor churn.

---

## Mechanism 2: `web_fetch` SSRF guard

### Default IP policy (`src/confine/net.rs`)

```rust
/// Returns false for any address a confined fetch must refuse.
/// Denied: unspecified (0.0.0.0/::), loopback (127/8, ::1),
/// private (10/8, 172.16/12, 192.168/16), link-local (169.254/16, fe80::/10),
/// shared/CGNAT (100.64/10), unique-local (fc00::/7), and IPv4-mapped IPv6
/// forms of the above.
pub fn is_globally_routable(ip: std::net::IpAddr) -> bool;

#[derive(Clone)]
pub struct UrlPolicy { /* predicate over resolved IPs; default = is_globally_routable */ }

impl UrlPolicy {
    pub fn public_only() -> Self;                       // the default
    pub fn custom(f: impl Fn(IpAddr) -> bool + Send + Sync + 'static) -> Self;
    pub fn check_host(&self, host: &str) -> Result<(), ConfinementError>; // resolves + checks all IPs
}
```

### `WebFetchTool` API

```rust
impl WebFetchTool {
    pub fn new() -> Self;                       // unchanged: follows redirects, no SSRF check
    pub fn guarded() -> Self;                    // UrlPolicy::public_only(), manual redirect loop
    pub fn with_url_policy(policy: UrlPolicy) -> Self;
}
```

### Guarded fetch flow

Guarded mode builds its reqwest client with `redirect(reqwest::redirect::Policy::none())`
and runs a manual loop (max 10 hops):

1. Parse the URL; extract host. Reject non-`http(s)` schemes.
2. Resolve the host (`tokio::net::lookup_host`); if **any** resolved IP fails the policy →
   `ToolResult::error` (blocked). IP-literal hosts are checked directly.
3. Issue the request with redirects disabled.
4. On a 3xx + `Location`, resolve the next URL (relative to current) and repeat from (1) —
   **re-validating the new host's IPs every hop.**
5. Otherwise return the body (current behavior).

**Residual race (stated honestly):** reqwest connects by hostname, so a DNS-rebinding
attacker could return a public IP to our `lookup_host` and a private IP to the actual
connect. Pinning the connection to the resolved IP is a future hardening; for the
confused-model threat, per-hop resolution + the deny-list is the right level. Documented.

---

## Error behavior

All confinement violations return `Ok(ToolResult::error(message))` — never a hard `Err`.
This matches the existing tool convention (`runner.rs:525-527` already maps tool `Err` to
an error result) and, more importantly, lets the **model see the refusal and adapt** ("path
outside allowed root", "address blocked") rather than aborting the run.

---

## FFI exposure (final phase)

Today `MuxEngine::new(data_dir)` hardcodes a 5-tool builtin set
(`mux-ffi/src/engine/mod.rs:113-119`): read/write/list/search/bash. `web_fetch` is **not**
an FFI builtin, so the SSRF guard is Rust-side only this round — no FFI work for it.

For the filesystem jail, add an **additive, non-breaking** confined entry point so the
Swift/Kotlin app can boot a jailed FS tool set:

```rust
// new constructor alongside the existing one (existing `new` untouched)
impl MuxEngine {
    pub fn new_confined(data_dir: String, root: String) -> Result<Arc<Self>, ...>;
}
```

When a root is set, the four FS builtins are constructed via `::rooted(RootedFs::new(root)?)`;
`bash` is still added but the docs warn it defeats the jail (the confined-engine path may
later choose to omit bash — flagged for the plan, not decided here). The root crosses
UniFFI as a plain `String`; a *custom* policy object across FFI is a deferred power-user
path.

---

## Public API / prelude changes

- New: `mux::confine::{RootedFs, UrlPolicy, ConfinementError, is_globally_routable}`,
  re-exported in `src/prelude.rs`.
- FS tools + `WebFetchTool` gain constructors (above); their exported names are unchanged.
- `tests/public_api_surface.rs` updated for the new symbols and constructor forms.

---

## Testing strategy (TDD throughout)

**`RootedFs` unit tests** (`src/confine/fs_test.rs`):
- relative path inside root → resolves under root
- absolute path inside root → ok; absolute path outside root → `EscapesRoot`
- `../` traversal escaping root → `EscapesRoot`
- symlink (created via `tempfile`) pointing outside root → `EscapesRoot`
- not-yet-existing leaf whose ancestor is inside root → ok (write case)
- not-yet-existing leaf whose ancestor is outside root → `EscapesRoot`
- `open_read` re-check rejects a path swapped to a symlink-escape after `resolve`

**Per-tool tests** (real files via `tempfile`, no mocks):
- each FS tool unconfined → unchanged behavior
- each FS tool rooted → in-root op succeeds, out-of-root op returns `ToolResult::error`
- `search`/`list_files` rooted → glob escape yields nothing

**SSRF tests** (`src/confine/net_test.rs` + web_fetch tests):
- `is_globally_routable` truth table across v4/v6 (loopback, private, link-local, CGNAT,
  ULA, mapped, and a public control like `1.1.1.1`)
- guarded `web_fetch` to a loopback/private literal → blocked
- guarded `web_fetch` whose redirect `Location` points at a private literal → blocked at
  the redirect hop (per-hop re-validation), using a real local test server
- unguarded `web_fetch` → unchanged

**FFI test** (`mux-ffi`): `new_confined` builds an engine whose `read_file` refuses a path
outside the root.

All tests use real filesystem temp dirs and real local sockets — **no mock mode**.

---

## Implementation phases (the PR)

Each phase is its own set of commits; the whole branch becomes one PR for #20.

1. **`confine::RootedFs` + `ConfinementError`** — mechanism + unit tests. (`src/confine/`)
2. **Wire the 5 FS tools** — constructors + `execute()` integration + per-tool tests +
   update the 6 registration sites + `public_api_surface.rs`.
3. **`confine::net` + guarded `web_fetch`** — `is_globally_routable`, `UrlPolicy`, manual
   redirect loop + tests.
4. **FFI `new_confined`** — additive engine constructor + FFI test.
5. **Docs + CHANGELOG** — "Confining mux" guide (tool selection, the jail, the SSRF guard,
   the `ApprovalHandler` bash-veto pattern, and the loud "jail is moot unless bash is
   dropped" caveat); `CHANGELOG.md` under *Added*.

Phases 1–2 deliver the headline value; 3–5 complete the scope. Version bump handled at
release time per the existing tag-triggered process (not in this PR).

## Open items for the plan (not blocking design)

- Whether `new_confined` omits `bash` by default or just warns (Phase 4 detail).
- Exact connection-pinning hardening for DNS rebinding (future, documented as residual).
