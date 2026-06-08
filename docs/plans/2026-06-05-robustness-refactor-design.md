# mux Robustness & Cleanup Refactor — Design

**Status:** Draft for review — 2026-06-05
**Branch:** `refactor/robustness-pass`
**Author:** pairing session (Doctor Biz + Claude)

## Goal

Bring the `mux` workspace to a "present-it-on-stage" bar of robustness, clarity, and
maintainability — **without changing any public API or observable behavior.** The
library should become more trustworthy through *enforced quality gates and proof
harnesses*, not through altered runtime behavior.

## Hard constraints (non-negotiable)

1. **Public API frozen, byte-identical.** Both surfaces:
   - The Rust public surface (`mux` crate: `lib.rs` modules, `prelude`, every `pub` item).
   - The UniFFI surface consumed by Swift/Kotlin (`mux-ffi` `#[uniffi::export]` items).
2. **Observable behavior unchanged.** No change to what a caller sees on any input,
   success or failure path. (Notably, the panic-site work in §5 turns out to be
   behavior-*preserving* — see that section.)
3. **No new features, no dependency changes, no renames of public items.**
4. **Phased execution.** Each phase ends green (`test` + `clippy -D warnings` + `fmt`)
   and is a review checkpoint. Hand edits respect the working agreement of ≤5 files per
   phase; tool-applied mechanical passes (`cargo fmt`, `cargo clippy --fix`) are applied
   workspace-wide as a single verifiable step.

## Non-goals

- API ergonomics improvements, builder additions, trait reshaping.
- Performance optimization (unless it falls out of a mechanical clippy fix).
- Touching the `.scratch/` scenarios (gitignored scratch space).
- Replacing idiomatic `Mutex::lock().unwrap()` — see §5.

## Scope

Entire workspace: `mux` (core), `mux-ffi` (Swift/Kotlin bindings), `agent-test-tui`,
`code-agent`. The two demo crates are in scope for lint/fmt cleanup and double as
real consumers of the public API (see verification spine).

---

## The verification spine — proof of zero change

Three independent proofs, run at the end of every phase. This is the contract.

1. **Test suite is the floor.** `cargo test --workspace` is green before and after every
   phase, and the **total test count never drops.** Extracted/moved tests still compile
   and run.

2. **Rust public API is compiler-proven.** The integration tests (`tests/`) and both demo
   crates (`agent-test-tui`, `code-agent`) are real consumers of `mux`'s public API; if a
   public path moves or disappears, **they fail to compile.** We add one belt-and-suspenders
   guard: a compile-only test that names every `prelude` export and the documented public
   types, so a removed/renamed export is caught immediately. (`cargo-public-api` would be
   ideal but needs a nightly toolchain that isn't installed; the consumer-compile proof is
   equivalent for our purposes and free.)

3. **FFI contract is snapshot-proven.** `uniffi-bindgen generate` (already installed,
   already used by `release-xcframework.yml`) produces the Swift **and** Kotlin bindings.
   We snapshot them at Phase 0 and re-generate at the end of every structural phase; the
   diff must be **empty.** This is the strongest possible proof that the Swift/Kotlin
   shipping contract is byte-identical.

The only deliberate source changes that could *conceivably* touch behavior are the
streaming-unwrap restructures in §5 — and those are provably behavior-preserving, with new
tests pinning the streamed output.

---

## Workstream §1 — CI + lint posture

- Add a single **`[workspace.lints]`** table to the root `Cargo.toml` (edition-2024
  idiomatic): `rust` and `clippy` groups at `warn`. No `#![deny(warnings)]` in source
  (that is a downstream footgun); enforcement happens in CI.
- Each crate opts in with `[lints] workspace = true`.
- **CI gains teeth** (`.github/workflows/ci.yml`): add, alongside the existing 3-OS
  build+test matrix, a lint job running:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets -- -D warnings`

## Workstream §2 — Mechanical cleanup (zero behavior change)

- `cargo fmt --all` (today `examples/cache_smoke.rs` and others fail `--check`).
- `cargo clippy --fix` for the 9 auto-fixable warnings; manual fixes for the rest:
  - `type_complexity` on `Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>`
    (`mux-ffi/src/task_tool.rs:115`) → introduce a named `type` alias.
  - manual `div_ceil` (`mux-ffi/src/context.rs:118`) → `.div_ceil(..)`.
  - `or_insert_with(Vec::new)` → `or_default()` (`workspace.rs:120`).
  - `unnecessary_filter_map` (`tool_wrappers.rs:15`) → `map`.
- **Remove confirmed dead code:** `execute_tool_with_captured_client`
  (`mux-ffi/src/engine/mcp.rs:691`) — `pub(super)`, zero call sites (verified by full-repo
  grep). Not part of the public API. A fresh rename-safety sweep (calls, strings, dynamic
  dispatch) precedes removal.
- **Tighten a hollow test:** `mux-ffi/src/lib.rs:450` computes `before` but never asserts
  on it (hence the `unused_variable` warning). Make the test actually assert truncation
  (`after < before`). This strengthens the test; library behavior is untouched.

## Workstream §3 — Test extraction (structural, near-zero risk)

Key insight: the giant files aren't big — their **inline `#[cfg(test)]` modules** are. The
codebase already uses sibling `#[cfg(test)] mod xxx_test;` files (`anthropic_test.rs`,
`types_test.rs`, `registry_test.rs`, …). We follow that exact convention.

| File | Now | After test-extract |
|---|---|---|
| `mux-ffi/src/engine/messaging.rs` | 1670 | ~636 (tests → `messaging_test.rs`, ~1033 lines) |
| `mux-ffi/src/engine/mcp.rs` | 1224 | ~727 (tests → `mcp_test.rs`, ~496 lines) |
| `src/llm/openai.rs` | 1008 | ~808 (tests → `openai_test.rs`, ~200 lines) |

Test code moves verbatim; the parent gets `#[cfg(test)] mod <name>_test;`. Test count
unchanged. Any other file whose inline tests push it over ~600 LOC is eligible by the same
rule.

## Workstream §4 — Production splits (structural, paths preserved)

After test extraction, split the remaining production code by responsibility. Public paths
are preserved via re-exports; the FFI snapshot and consumer-compile proof guard them.

- **`src/llm/openai.rs`** → `src/llm/openai/` layered by concern:
  - `types.rs` — wire structs/enums (`OpenAIRequest`, `OpenAIMessage`, deltas, …).
  - `convert.rs` — request building (`try_openai_messages`, `try_media_part`,
    `try_into_openai_request`, `From<&ToolDefinition>`).
  - `response.rs` — `From<OpenAIResponse>`, `parse_stop_reason`, `parse_sse_line`.
  - `mod.rs` — `OpenAIClient` + the `LlmClient` impl; `pub use` re-exports so
    `crate::llm::openai::*` and `crate::llm::OpenAIClient` resolve unchanged.
- **`mux-ffi/src/engine/messaging.rs`** → lift `ChatCallbackHook` and
  `ffi_media_into_blocks` into a focused submodule; keep the `impl MuxEngine` messaging
  block cohesive. (Exact boundary finalized in the plan after a full read.)
- **`mux-ffi/src/engine/mcp.rs`** → split the two `impl MuxEngine` concern-groups
  (connection/lifecycle vs tool execution) into sibling files within the `engine` module.

Exact module boundaries for `messaging`/`mcp` are finalized in the implementation plan
after reading each file end-to-end; the principle is "one clear responsibility per file,
public paths unchanged."

## Workstream §5 — Panic-site audit & behavior-preserving fixes

Full audit of every production (non-test) `unwrap`/`expect`/`panic` and its disposition.
**Dispositions reflect the frozen-behavior constraint.**

| Site(s) | Pattern | Disposition |
|---|---|---|
| `src/llm/types.rs` ×6, `src/agent/async_handle.rs` ×6 | `Mutex::lock().unwrap()` | **KEEP** — idiomatic; only fails on a poisoned mutex (another thread already panicked). |
| `src/llm/{openai.rs:721, gemini.rs:590, ollama.rs:228, openrouter.rs:217}` | streaming `text_block_index.unwrap()` | **RESTRUCTURE (behavior-preserving).** The index is set to `Some(idx)` immediately before the `unwrap()`, so it is provably non-`None` — the unwrap cannot panic today. Rewrite to thread `idx` through both branches so the unwrap disappears. Zero behavior change; a new unit test per client pins the emitted `ContentBlockStart`/`Delta` indices. |
| `src/tools/web_fetch.rs:27`, `src/tools/web_search.rs:35` | `.expect("…HTTP client")` in constructor | **KEEP (documented).** Only fails on catastrophic TLS-backend init. Converting requires changing the public `new()`/`Default` signature to return `Result` → public-API change → out of scope. Add a clarifying comment. |
| `mux-ffi/src/engine/messaging.rs:603` | `.expect("Custom provider was captured…")` inside the infallible `client_factory` closure | **KEEP (documented).** The closure type is `Fn(&str) -> Arc<dyn LlmClient>` (infallible by design); the value is captured upfront specifically to dodge the unregister race. Propagating an error would change the closure/`FfiTaskTool` contract → out of scope. Add a comment explaining the invariant. |
| `src/tools/list_files.rs:54`, `src/tools/search.rs:63` | `glob(p).unwrap_or_else(\|_\| glob("").unwrap())` | **KEEP (preserve behavior).** Currently an invalid pattern silently yields empty results. The robust alternative (return an error) is a user-visible behavior change — explicitly excluded by the frozen-behavior constraint. |

Net: **no behavioral changes** in the entire refactor. The streaming restructures remove
fragile-looking-but-currently-safe unwraps; everything else is kept and (where useful)
documented.

## Workstream §6 — Docs & hygiene

- **README fix:** `mux = "0.1"` → matches the real crate version (`0.10.0`). Doc-only.
- **`missing_docs = "warn"`** on the core `mux` crate (warn, not deny) + doc-comments on
  the `prelude`-exported public types. Bounded; guides without gating CI.
- **`CHANGELOG.md`** entry under `[Unreleased]` recording the refactor and stating
  explicitly: no public-API change, no behavioral change.

---

## Phasing & sequencing

- **Phase 0 — Baseline & safety net.** Record green test baseline + counts; generate and
  store Swift+Kotlin binding snapshots; add the public-API guard test; add a local
  `verify` helper (test + clippy + fmt + binding-diff). *No source changes.*
- **Phase 1 — CI + lint posture + mechanical cleanup** (§1, §2).
- **Phase 2 — Test extraction** (§3). Parallelizable across files (subagents).
- **Phase 3 — Production splits** (§4). One file-group at a time, ≤5 files each.
- **Phase 4 — Panic restructure** (§5) + new tests.
- **Phase 5 — Docs & hygiene** (§6).
- **Phase 6 — Final verification & finish.** Full `clippy -D warnings` + `fmt` + `test`;
  empty Swift+Kotlin binding diff; guard test green → PR via
  `finishing-a-development-branch`.

Execution runs through `subagent-driven-development` per the working agreement; panic
restructures (§5) stay sequential and individually reviewed.

## Risks & mitigations

- **A split silently changes a public path.** → Consumer-compile proof + FFI binding diff
  + guard test catch it before commit.
- **`clippy --fix` rewrites something subtly.** → Full test run + binding diff after the
  mechanical pass; review the diff.
- **Test extraction drops a `use`/visibility.** → Compiler + unchanged test count catch it.
- **`[workspace.lints]` surfaces new warnings that tempt scope creep.** → Fix only what the
  enabled lints flag; anything requiring a behavior/API change is logged, not fixed.

## Definition of done

- `cargo fmt --all --check` clean; `cargo clippy --workspace --all-targets -- -D warnings`
  clean; `cargo test --workspace` green with test count ≥ baseline.
- Swift + Kotlin binding diff vs Phase-0 snapshot is **empty.**
- Public-API guard test green; both demo crates + integration tests compile unchanged.
- No giant files left (every source file comfortably screen-sized).
- CHANGELOG + README updated; CI enforces fmt + clippy going forward.
