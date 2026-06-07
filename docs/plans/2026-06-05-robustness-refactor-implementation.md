# mux Robustness & Cleanup Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `mux` workspace robust, clean, and maintainable to a present-on-stage bar, with **zero change to any public API (Rust or UniFFI Swift/Kotlin) and zero change to observable behavior.**

**Architecture:** A pure refactor. Robustness comes from enforced quality gates (`clippy -D warnings` + `fmt --check` in CI, a `[workspace.lints]` table), structural clarity (extract inline test modules to sibling files, split a few giant files by responsibility with paths preserved via re-exports), dead-code removal, and a tightened test — not from altering runtime behavior. Every phase is verified against a three-part proof spine: full test suite green (count never drops), Rust public surface compiler-/grep-proven unchanged, and an empty `uniffi-bindgen` Swift+Kotlin binding diff.

**Tech Stack:** Rust (edition 2024), tokio, UniFFI 0.30, `uniffi-bindgen` (installed at `$HOME/.cargo/bin/uniffi-bindgen`), reqwest, thiserror.

**Source spec:** `docs/plans/2026-06-05-robustness-refactor-design.md`

---

## Conventions used by every task

**The verification spine** (built in Task 2, referenced throughout as "run the spine"):

```bash
./scripts/verify-refactor.sh          # fmt --check + clippy -D warnings + test + binding diff
./scripts/verify-refactor.sh baseline # (re)capture the binding baseline (only when intentionally rebaselining)
```

**Refactor verification model:** This is a refactor, so for most tasks the "test" is the
*existing* suite plus the spine staying green/unchanged (Fowler-style: tests pass before and
after). New tests are added only where genuinely new surface appears (Task 1) or an existing
test is strengthened (Task 7). Do **not** invent mock-based tests — this repo forbids mocks;
the streaming change (Task 15) is verified by provable equivalence + the spine.

**Per-phase expected spine state:**
- Phases 0: `cargo test --workspace` green; bindings captured. (`fmt`/`clippy` NOT yet clean — expected.)
- Phase 1 onward: full spine green and binding diff empty after every task.

**Verified baseline (captured 2026-06-05):** 414 tests pass workspace-wide (275 `mux` lib + 6 + 10
integration + 123 `mux-ffi` lib + doc/empty harnesses). This is the **floor** — the count must
never drop below 414 (Task 1 raises it by 1; no other task lowers it). `uniffi-bindgen 0.30.0`
lives at `$HOME/.cargo/bin/uniffi-bindgen`; the host FFI dylib is `target/debug/libmux_ffi.dylib`.
Kotlin generation prints a harmless `ktlint`-not-found warning — the `.kt` file is still produced
and the diff stays deterministic across runs on this machine.

**Commit discipline:** one commit per task (TDD-style), conventional-commit messages, never `--no-verify`.

---

## Phase 0 — Baseline & safety net

### Task 1: Public-API surface guard test

**Files:**
- Create: `tests/public_api_surface.rs`

This is a compile-only guard. The integration tests and demo crates already catch most
public-path breakage, but the `mux::llm` module uses a glob re-export (`pub use openai::*`),
so a dropped item would silently shrink the API without a compile error *unless* something
references it. This test references every prelude export by exact path.

- [ ] **Step 1: Write the guard (it must compile = it passes)**

```rust
// ABOUTME: Compile-only guard pinning the mux public API surface.
// ABOUTME: If a prelude/public path is renamed or removed, this test fails to compile.
#![allow(unused_imports)]

// Each `use ... as _` pins one public path. Renames/removals break compilation.
use mux::MuxError as _;
use mux::agent::{
    AgentDefinition as _, AgentRegistry as _, FilteredRegistry as _, SubAgent as _,
    SubAgentResult as _, TaskTool as _,
};
use mux::error::{
    LlmError as _, McpError as _, MuxError as _, PermissionError as _, ToolError as _,
};
use mux::llm::{
    AnthropicClient as _, ContentBlock as _, LlmClient as _, Message as _, OpenAIClient as _,
    Request as _, Response as _, Role as _, StopReason as _, StreamEvent as _,
    ToolDefinition as _, Usage as _,
};
use mux::mcp::{
    HttpTransport as _, McpClient as _, McpContentBlock as _, McpLogLevel as _,
    McpPromptGetResult as _, McpPromptInfo as _, McpPromptsListResult as _, McpProxyTool as _,
    McpResourceContent as _, McpResourceInfo as _, McpResourcesListResult as _, McpRoot as _,
    McpSamplingParams as _, McpSamplingResult as _, McpServerCapabilities as _,
    McpServerConfig as _, McpToolInfo as _, McpToolResult as _, McpTransport as _,
    SseTransport as _, StdioTransport as _, Transport as _,
};
use mux::permission::{
    AlwaysApprove as _, AlwaysReject as _, ApprovalContext as _, ApprovalHandler as _,
    Decision as _, Policy as _, PolicyBuilder as _,
};
use mux::prelude::*;
use mux::tool::{Registry as _, Tool as _, ToolExecute as _, ToolResult as _};
use mux::tools::{
    BashTool as _, ListFilesTool as _, ReadFileTool as _, SearchResult as _, SearchTool as _,
    WebFetchTool as _, WebSearchTool as _, WriteFileTool as _,
};

#[test]
fn public_api_surface_is_stable() {
    // Compilation of the imports above is the assertion.
}
```

- [ ] **Step 2: Verify it compiles and runs**

Run: `cargo test --test public_api_surface`
Expected: PASS (`1 passed`). If any path fails to resolve, the prelude/module layout is wrong — fix the source, not this test.

- [ ] **Step 3: Commit**

```bash
git add tests/public_api_surface.rs
git commit -m "test: compile-only guard pinning the public API surface"
```

### Task 2: Verification harness + binding baseline

**Files:**
- Create: `scripts/verify-refactor.sh`

- [ ] **Step 1: Write the harness**

```bash
#!/usr/bin/env bash
# ABOUTME: Dev-only verification harness for the no-behavior-change refactor.
# ABOUTME: Proves the contract: fmt, clippy -D warnings, tests, and UniFFI binding diff.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BINDGEN="${UNIFFI_BINDGEN:-$HOME/.cargo/bin/uniffi-bindgen}"
LIB="target/debug/libmux_ffi.dylib"   # host (macOS) cdylib from crate-type
OUT="target/verify"
MODE="${1:-check}"                      # check | baseline

gen_bindings() {
  local dir="$1"
  rm -rf "$dir"; mkdir -p "$dir"
  cargo build -p mux-ffi --quiet
  "$BINDGEN" generate --library "$LIB" --language swift  --out-dir "$dir" >/dev/null
  "$BINDGEN" generate --library "$LIB" --language kotlin --out-dir "$dir" >/dev/null
}

if [ "$MODE" = "baseline" ]; then
  gen_bindings "$OUT/baseline"
  echo "Baseline UniFFI bindings captured at $OUT/baseline"
  exit 0
fi

echo "== fmt --check ==";  cargo fmt --all --check
echo "== clippy -D ==";    cargo clippy --workspace --all-targets -- -D warnings
echo "== test ==";         cargo test --workspace
echo "== uniffi binding diff =="
gen_bindings "$OUT/current"
if diff -ru "$OUT/baseline" "$OUT/current"; then
  echo "FFI bindings byte-identical OK"
else
  echo "FFI BINDINGS CHANGED — refactor altered the Swift/Kotlin contract"; exit 1
fi
```

- [ ] **Step 2: Make it executable and capture the baseline**

Run:
```bash
chmod +x scripts/verify-refactor.sh
./scripts/verify-refactor.sh baseline
```
Expected: "Baseline UniFFI bindings captured at target/verify/baseline" and files present:
```bash
ls target/verify/baseline   # *.swift and *.kt present
```
(`target/` is gitignored, so the baseline is not committed.)

- [ ] **Step 3: Confirm the test floor**

Run: `cargo test --workspace 2>&1 | rg '(test result|running)'`
Expected: all suites `ok`. Record the total passed count — it is the floor for every later phase.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify-refactor.sh
git commit -m "chore: add dev verification harness for the refactor"
```

---

## Phase 1 — CI + lint posture + mechanical cleanup

### Task 3: Add the workspace lint posture

**Files:**
- Modify: `Cargo.toml` (root), `mux-ffi/Cargo.toml`, `agent-test-tui/Cargo.toml`, `code-agent/Cargo.toml`

- [ ] **Step 1: Add the `[workspace.lints]` table to root `Cargo.toml`**

Append to the root `Cargo.toml` (keep modest — `clippy::all` is the default warn set; do
NOT enable `pedantic`/`unwrap_used`, which would flood `-D warnings` and force behavior/API
changes):

```toml
[workspace.lints.rust]
unused = "warn"

[workspace.lints.clippy]
all = "warn"
```

- [ ] **Step 2: Opt every workspace member in**

Add to each of `Cargo.toml` (root package section), `mux-ffi/Cargo.toml`,
`agent-test-tui/Cargo.toml`, `code-agent/Cargo.toml`:

```toml
[lints]
workspace = true
```

- [ ] **Step 3: Observe (do not yet gate)**

Run: `cargo clippy --workspace --all-targets 2>&1 | rg '^warning|^error' | tail -5`
Expected: the known ~25 warnings still present (fixed in Tasks 4–6). No `error`.

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml mux-ffi/Cargo.toml agent-test-tui/Cargo.toml code-agent/Cargo.toml
git commit -m "chore: add shared [workspace.lints] posture"
```

### Task 4: Format the whole tree

**Files:** workspace-wide (tool-applied; mechanical).

- [ ] **Step 1: Format**

Run: `cargo fmt --all`

- [ ] **Step 2: Verify formatting + tests**

Run: `cargo fmt --all --check && cargo test --workspace 2>&1 | rg 'test result'`
Expected: `--check` exits 0; test counts unchanged from the Task 2 floor.

- [ ] **Step 3: Verify FFI bindings unchanged**

Run: `./scripts/verify-refactor.sh` is not yet fully green (clippy still dirty), so check bindings directly:
```bash
cargo build -p mux-ffi --quiet
"$HOME/.cargo/bin/uniffi-bindgen" generate --library target/debug/libmux_ffi.dylib --language swift --out-dir target/verify/current >/dev/null
"$HOME/.cargo/bin/uniffi-bindgen" generate --library target/debug/libmux_ffi.dylib --language kotlin --out-dir target/verify/current >/dev/null
diff -ru target/verify/baseline target/verify/current && echo "bindings OK"
```
Expected: `bindings OK`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "style: cargo fmt --all"
```

### Task 5: Fix all mechanical/style clippy warnings

**Scope note (full census, captured 2026-06-05):** `cargo clippy --workspace --all-targets`
reports **55** warnings. This task owns the **mechanical/style** subset only. The
**dead-code/unused family** (`dead_code`, `unused_variables`, `unused_mut` where it is the
unused-`before` test var, `await_holding_lock`) is handled by **Task 6** (dead-code
disposition) and **Task 7** (hollow test). After Task 5, clippy must show *only* the
items Tasks 6/7 own — every style lint below must be gone:
`collapsible_if` ×4, `while_let_loop` ×2, `type_complexity` ×2, `unwrap_or_default`,
`unnecessary_lazy_evaluations`, `unnecessary_filter_map`, `ptr_arg`, `module_inception`,
`manual_flatten`, `manual_div_ceil`, `derivable_impls`, `collapsible_match`,
`clone_on_copy`, and the `unused_mut` in `src/llm/media.rs`.

**Files:**
- Modify: `mux-ffi/src/context.rs:118`, `mux-ffi/src/engine/workspace.rs:120`, `mux-ffi/src/engine/tool_wrappers.rs:15`, `mux-ffi/src/task_tool.rs:115`, `src/agent/task.rs:27`, `src/coordinator/mod.rs:4`, plus any auto-fixable sites.

- [ ] **Step 1: Apply the auto-fixable subset**

Run: `cargo clippy --fix --workspace --allow-dirty --allow-staged`

- [ ] **Step 2: Manual fix — `manual_div_ceil` in `context.rs`**

Replace (line ~118):
```rust
    ((bytes + APPROX_BYTES_PER_TOKEN - 1) / APPROX_BYTES_PER_TOKEN) as u32
```
with:
```rust
    bytes.div_ceil(APPROX_BYTES_PER_TOKEN) as u32
```

- [ ] **Step 3: Manual fix — `or_insert_with(Vec::new)` in `workspace.rs`**

Replace (line ~120):
```rust
            .or_insert_with(Vec::new)
```
with:
```rust
            .or_default()
```

- [ ] **Step 4: Manual fix — `unnecessary_filter_map` in `tool_wrappers.rs`**

Replace the `filter_map` block (lines ~14-19) with `map` (both arms return `Some`, so it is a `map`):
```rust
        .map(|block| match block {
            McpContentBlock::Text { text } => text.as_str(),
            McpContentBlock::Image { .. } => "[image]",
        })
```

- [ ] **Step 5: Manual fix — `type_complexity` in `task_tool.rs`**

Add a type alias near the top of `mux-ffi/src/task_tool.rs` (after imports):
```rust
/// Factory that builds an LLM client for a given model name.
type LlmClientFactory = Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>;
```
Then replace the field type (line ~115):
```rust
    client_factory: Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>,
```
with:
```rust
    client_factory: LlmClientFactory,
```
Search for the same `Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>` in any function
signature in this file (e.g. the constructor) and replace those with `LlmClientFactory` too:
```bash
rg -n 'Arc<dyn Fn\(&str\) -> Arc<dyn LlmClient> \+ Send \+ Sync>' mux-ffi/src/task_tool.rs
```

- [ ] **Step 5b: Manual fix — `type_complexity` in `src/agent/task.rs` (core crate)**

Same pattern, but this is the **frozen `mux` crate** — the alias MUST be private (no `pub`),
since it is an internal field type, not public API. Add near the top of `src/agent/task.rs`
(after imports):
```rust
/// Factory that builds an LLM client for a given model name.
type LlmClientFactory = Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>;
```
Then replace the field type (line ~27) `Arc<dyn Fn(&str) -> Arc<dyn LlmClient> + Send + Sync>`
with `LlmClientFactory`, and the same type in the constructor/any signature in this file:
```bash
rg -n 'Arc<dyn Fn\(&str\) -> Arc<dyn LlmClient> \+ Send \+ Sync>' src/agent/task.rs
```

- [ ] **Step 5c: Manual fix — `module_inception` in `src/coordinator/mod.rs`**

Renaming the module would change a path → forbidden by the freeze. Suppress with a documented
`#[allow]`. Read `src/coordinator/mod.rs:4` first; on the `mod coordinator` line add:
```rust
// `coordinator` submodule mirrors the module name by design; renaming would
// change the public path, which the API freeze forbids.
#[allow(clippy::module_inception)]
mod coordinator;
```

- [ ] **Step 6: Verify only the Task 6/7 warnings remain**

Run: `cargo clippy --workspace --all-targets 2>&1 | rg '^warning: ' | sort | uniq -c`
Expected: **every style lint from the census note is gone.** The only remaining warnings are
the dead-code/unused family owned by Task 6 (`dead_code`, `field … never read`,
`… never used`, `await_holding_lock`) and the single `unused_variables: before` owned by
Task 7. `cargo clippy … -- -D warnings` will still fail at this point — that is expected and
becomes clean after Tasks 6 + 7.

- [ ] **Step 7: Verify behavior unchanged**

Run: `cargo test --workspace 2>&1 | rg 'test result'` (counts ≥ floor) and the binding diff (as in Task 4 Step 3).
Expected: tests green, `bindings OK`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "fix(clippy): resolve mechanical/style lint warnings"
```

### Task 6: Dead-code disposition

**Decision (2026-06-05, Doctor Biz approved "Option A"):** the dead-code/unused warnings are
NOT all simple dead code. They split two ways, and each half is handled differently:

1. **Truly orphaned — zero references anywhere, incl. tests → DELETE** (after a rename-safety
   sweep): `execute_tool_with_captured_client` and the `server_name` field + getter.
2. **Tested but not yet wired into production → KEEP + `#[allow(dead_code)]` + comment + tracked
   issue** (deleting would discard tested, probably-intended code; wiring it up would change
   behavior and is forbidden by the freeze). Two clusters:
   - **FFI task/subagent-tool cluster** → tracked in **issue #9**.
   - **Core `RunHandle` status setters** → tracked in **issue #10**.

`await_holding_lock` fires at **4 sites**, in two groups:
- **2 inside `execute_task_tool`** (`messaging.rs:523`, `:557`) — dead/not-yet-wired code; they
  ride with that method's annotations (covered by `#[allow(clippy::await_holding_lock)]` in
  Step 4, noted in #9).
- **2 in LIVE, FFI-reachable code** (`do_spawn_agent` at `subagent.rs:214`, `do_resume_agent` at
  `subagent.rs:331`; both called from the `#[uniffi::export]` `spawn_agent`/`resume_agent`). The
  fix (snapshot the tools before the `.await` so the guard drops first) would change
  lock-contention timing — a concurrency-semantics change the freeze forbids — so these are KEPT
  as-is + `#[allow(clippy::await_holding_lock)]` + comment, tracked in **issue #11** (Step 4b).

**Files:**
- Modify (delete items): `mux-ffi/src/engine/mcp.rs` (remove `execute_tool_with_captured_client`, ~line 691); `mux-ffi/src/engine/tool_wrappers.rs` (remove the `server_name` field ~line 29 and `server_name()` getter ~line 61 — **keep** the `new()` param, it builds `qualified_name`).
- Modify (annotate, keep): `mux-ffi/src/engine/messaging.rs` (`execute_task_tool` ~501), `mux-ffi/src/engine/mcp.rs` (`get_workspace_tools` ~601, `parse_tool_name` ~685), `mux-ffi/src/engine/helpers.rs` (`parse_qualified_tool_name` ~6), `mux-ffi/src/engine/subagent.rs` (`TaskToolEventProxy` ~18), `mux-ffi/src/engine/mod.rs` (`transcript_store` field ~73), `src/agent/async_handle.rs` (`set_running`/`set_completed`/`set_failed` ~191/197/208).
- Modify (annotate live `await_holding_lock`, keep): `mux-ffi/src/engine/subagent.rs` — `do_spawn_agent` (~157, await at ~214) and `do_resume_agent` (~281, await at ~331). Add `#[allow(clippy::await_holding_lock)]` + comment → #11. These are **live** (FFI-reachable), NOT dead — do **not** add `#[allow(dead_code)]`.

- [ ] **Step 1: Rename-safety sweep for the two DELETE targets**

```bash
rg -n 'execute_tool_with_captured_client'   # expect: 1 hit (the def)
rg -n '\bserver_name\b' mux-ffi/src/engine/tool_wrappers.rs   # expect: param use (line ~49), field decl/init (~29/52), getter (~61) — NO external readers
rg -n '\.server_name\(\)' --type rust       # expect: 0 hits (getter uncalled)
```
If any unexpected hit appears, STOP — it is not dead.

- [ ] **Step 2: Delete `execute_tool_with_captured_client`**

Read the file around `mux-ffi/src/engine/mcp.rs:691` first; delete the entire
`pub(super) async fn execute_tool_with_captured_client(...) { ... }` item.

- [ ] **Step 3: Delete the `server_name` field + getter (keep the ctor param)**

In `mux-ffi/src/engine/tool_wrappers.rs`: remove the `server_name: String` struct field (~29),
remove its assignment in `new()` (the `server_name,` field-init line ~52), and remove the
`pub fn server_name(&self) -> &str { &self.server_name }` getter (~61). **Keep** the `new()`
parameter `server_name: String` and the `format!("{}:{}", server_name, tool_name)` at ~49 — the
qualified name still depends on it. `McpToolWrapper` is not `#[uniffi::export]`, so this is not
an FFI-surface change (binding diff proves it).

- [ ] **Step 4: Annotate the FFI task/subagent-tool cluster (keep, tracked in #9)**

On each item below, add `#[allow(dead_code)]` plus a one-line comment. Use this exact comment
text (evergreen; states the fact, not the history):
```rust
// Reachable only via tests today: the task/subagent tool is implemented and tested but not
// yet wired into the production chat loop (build_tool_registry / do_send_message). See #9.
#[allow(dead_code)]
```
Items: `execute_task_tool` (`messaging.rs` ~501) — ALSO add `#[allow(clippy::await_holding_lock)]`
on the same item; `get_workspace_tools` (`mcp.rs` ~601); `parse_tool_name` (`mcp.rs` ~685);
`parse_qualified_tool_name` (`helpers.rs` ~6); `TaskToolEventProxy` (`subagent.rs` ~18);
`transcript_store` field (`mod.rs` ~73).

- [ ] **Step 4b: Annotate the 2 LIVE `await_holding_lock` sites (keep, tracked in #11)**

These are production code reachable from the FFI surface, so they are NOT dead code — add ONLY
`#[allow(clippy::await_holding_lock)]` (no `#[allow(dead_code)]`). Put the attribute on the
method, with this comment:
```rust
// Holds a custom_tools read guard across `.await` while registering tools into a local
// registry. The awaited work touches a different lock, so this path cannot self-deadlock;
// the snapshot-before-await fix changes lock-contention timing (a behavior change) and is
// deferred under the no-behavior-change freeze. See #11.
#[allow(clippy::await_holding_lock)]
```
Items: `do_spawn_agent` (`subagent.rs` ~157) and `do_resume_agent` (`subagent.rs` ~281).

- [ ] **Step 5: Annotate the core `RunHandle` setters (keep, tracked in #10)**

In `src/agent/async_handle.rs`, on `set_running` (~191), `set_completed` (~197), and
`set_failed` (~208), add `#[allow(dead_code)]` and this comment on the first of the three:
```rust
// Status-transition helpers exercised by tests but not yet called by the production run
// lifecycle. Kept until the lifecycle wiring is decided. See #10.
```
This alias/field stays private — no public-API change.

- [ ] **Step 6: Verify clippy is clean and behavior unchanged**

```bash
cargo clippy --workspace --all-targets -- -D warnings   # expect: only the `unused_variables: before` (Task 7) may remain
cargo test --workspace 2>&1 | rg 'test result'          # counts >= floor
```
Then the binding diff (Task 4 Step 3) — expect `bindings OK` (all changes are internal/`pub(super)`/private).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove orphaned dead code; retain+document unwired tested code (#9, #10, #11)"
```

### Task 7: Strengthen the hollow truncation test

**Files:**
- Modify: `mux-ffi/src/lib.rs` (the test at ~line 442-461, `test_auto_compaction_small_context_uses_truncation`)

The test computes `let before = engine.get_message_count(&conv.id);` then never uses it (the
source of the `unused_variable` warning).

**Why the obvious fix is hollow:** the test injects five ~40-byte messages (~10 tokens each,
~50 total) under a `context_limit` of 4096. `compact_context` computes
`target = effective_limit(4096) = 4096 * SAFETY_MARGIN(0.8) = 3276` tokens, and
`truncate_oldest` keeps every message that fits, scanning from newest backward. 50 tokens is
far under 3276, so `keep_from` stays at 0, nothing is drained, and `after == before == 5`.
`assert!(after <= before)` is therefore `5 <= 5` — trivially true. It clears the warning but
leaves the test as hollow as before. To genuinely test truncation, the conversation must
overflow the budget so the oldest messages are actually dropped.

- [ ] **Step 1: Read the test and confirm the shape**

Run: `sed -n '440,461p' mux-ffi/src/lib.rs`

- [ ] **Step 2: Size the messages to overflow the budget**

Replace the message-injection loop so the five messages together exceed the effective limit.
Each message of ~4096 bytes is ~1024 tokens (`4096 / APPROX_BYTES_PER_TOKEN(4)`); five of them
(~5120 tokens) exceed the ~3276-token effective limit, while each single message (1024 tokens)
stays well under it so the newest message always survives. Replace:
```rust
        // Add messages
        for i in 0..5 {
            engine.inject_test_message(
                &conv.id,
                Role::User,
                &format!("Message {} with content to consume tokens", i),
            );
        }
```
with:
```rust
        // Each message is ~1024 tokens (4096 bytes / APPROX_BYTES_PER_TOKEN=4); the five
        // together (~5120 tokens) exceed the effective limit
        // (4096 * SAFETY_MARGIN=0.8 ≈ 3276 tokens), so truncation must drop the oldest
        // messages while keeping the most recent.
        for i in 0..5 {
            engine.inject_test_message(
                &conv.id,
                Role::User,
                &format!("Message {} {}", i, "x".repeat(4096)),
            );
        }
```

- [ ] **Step 3: Assert truncation actually happened**

After the existing `let result = engine.compact_context(conv.id.clone());` / `assert!(result.is_ok(), ...)`
lines, add:
```rust
        let after = engine.get_message_count(&conv.id);
        assert!(
            after < before,
            "truncation must drop the oldest over-budget messages (before={before}, after={after})"
        );
        assert!(
            after >= 1,
            "truncation must retain the most recent message (after={after})"
        );
```
(`before` is now used, clearing the warning; `is_ok()` still proves the truncation strategy was
selected — summarization would need an API key — and the new asserts prove truncation actually
trimmed the history.)

- [ ] **Step 4: Verify**

Run: `cargo test -p mux-ffi test_auto_compaction_small_context_uses_truncation -- --nocapture`
Expected: PASS, no `unused_variable` warning.

- [ ] **Step 5: Commit**

```bash
git add mux-ffi/src/lib.rs
git commit -m "test: exercise real truncation in small-context compaction"
```

### Task 8: Give CI teeth

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add a lint job**

Add this job alongside the existing `test` job (same `env`):
```yaml
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - name: Cache cargo registry & build
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-lint-${{ hashFiles('**/Cargo.lock') }}
          restore-keys: ${{ runner.os }}-cargo-lint-
      - name: Format check
        run: cargo fmt --all --check
      - name: Clippy
        run: cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 2: Validate YAML locally**

Run: `cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings`
Expected: both exit 0 (this mirrors what CI will run).

- [ ] **Step 3: Re-baseline the spine and confirm fully green**

Now that fmt+clippy are clean, the harness should pass end-to-end:
Run: `./scripts/verify-refactor.sh`
Expected: fmt OK, clippy OK, tests green, `FFI bindings byte-identical OK`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: enforce cargo fmt --check and clippy -D warnings"
```

---

## Phase 2 — Test extraction (structural, near-zero risk)

For each big file: move its trailing `#[cfg(test)] mod <name> { ... }` block **verbatim** into a
sibling `*_test.rs`, then replace the inline block with a `#[path]` declaration **kept inside the
source file** (e.g. `#[cfg(test)] #[path = "messaging_test.rs"] mod tests;`). Keeping the
declaration in the source file makes the test module a child of that module, so `use super::*;`
still resolves to its scope — including the source file's private `use` imports and
`pub(super)`/private items — and the move is truly verbatim. Test count is invariant; `mod.rs`
is **not** modified.

Why not the `mod.rs`-declaration convention (`src/llm/mod.rs`, `src/tool/mod.rs`)? That style
only works when the parent re-exports the source module (`pub use openai::*`), so `super::*` in a
sibling test file resolves through the re-export. `engine/mod.rs` does **not** re-export
`messaging`/`mcp` (they are private impl-split modules), and even `openai`'s tests lean on
`openai`-private imports that `pub use openai::*` would not carry. The in-source `#[path]`
placement sidesteps both, so it is applied uniformly to all three files.

### Task 9: Extract `messaging.rs` tests

**Files:**
- Create: `mux-ffi/src/engine/messaging_test.rs`
- Modify: `mux-ffi/src/engine/messaging.rs` (no `engine/mod.rs` change)

- [ ] **Step 1: Create the sibling test file (verbatim body)**

Move the **body** of the inline test module — `messaging.rs` lines 640–1670, i.e. everything
between `mod tests {` and its closing `}` — verbatim into a new `messaging_test.rs`. Prefix the
ABOUTME header; the first executable line stays `use super::*;` and nothing else changes:
```rust
// ABOUTME: Tests for the engine messaging path (chat send, streaming, subagent task tool).
// ABOUTME: Extracted verbatim from messaging.rs; behavior unchanged.
use super::*;
// … rest of the module body, unchanged …
```

- [ ] **Step 2: Replace the inline module with an in-source `#[path]` declaration**

In `messaging.rs`, replace the entire inline module (lines 638–1671, from `#[cfg(test)]` through
its closing `}`) with:
```rust
#[cfg(test)]
#[path = "messaging_test.rs"]
mod tests;
```
Keep the declaration **inside `messaging.rs`** (not in `engine/mod.rs`): that makes `tests` a
child of `messaging`, so `use super::*;` still resolves to `messaging`'s scope — its items plus
its private `use` imports (`MuxEngine`, `StoredMessage`, `TaskToolEventProxy`, …). `#[path]`
keeps the file a flat sibling instead of forcing a `messaging/` subdirectory. The module name
stays `tests` (verbatim). Do **not** modify `engine/mod.rs`.

- [ ] **Step 3: Verify count unchanged**

Run: `cargo test -p mux-ffi 2>&1 | rg 'test result'`
Expected: same number of `mux-ffi` tests as the floor; all `ok`.

- [ ] **Step 4: Verify spine**

Run: `./scripts/verify-refactor.sh`
Expected: fully green; `bindings byte-identical`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(mux-ffi): extract messaging tests to messaging_test.rs"
```

### Task 10: Extract `mcp.rs` tests

**Files:**
- Create: `mux-ffi/src/engine/mcp_test.rs`
- Modify: `mux-ffi/src/engine/mcp.rs` (no `engine/mod.rs` change)

- [ ] **Step 1–5:** Repeat Task 9's procedure for `mcp.rs` (inline module is lines 696–1198; `mod tests`). ABOUTME header:
```rust
// ABOUTME: Tests for the engine MCP integration (connect, list, call tools).
// ABOUTME: Extracted verbatim from mcp.rs; behavior unchanged.
use super::*;
```
Verify with `cargo test -p mux-ffi 2>&1 | rg 'test result'` (count unchanged), then `./scripts/verify-refactor.sh`.

- [ ] **Commit:**
```bash
git add -A
git commit -m "refactor(mux-ffi): extract mcp tests to mcp_test.rs"
```

### Task 11: Extract `openai.rs` tests

**Files:**
- Create: `src/llm/openai_test.rs`
- Modify: `src/llm/openai.rs` (no `llm/mod.rs` change)

- [ ] **Step 1: Move tests** (begin ≈line 809) into `openai_test.rs` with header:
```rust
// ABOUTME: Tests for the OpenAI client (request building, response parsing, media).
// ABOUTME: Extracted verbatim from openai.rs; behavior unchanged.
use super::*;
```
- [ ] **Step 2: Replace the inline module with an in-source `#[path]` declaration**

In `openai.rs`, replace the entire inline module (lines 809–1008, `mod openai_test { … }`) with:
```rust
#[cfg(test)]
#[path = "openai_test.rs"]
mod openai_test;
```
Same in-source `#[path]` placement as Task 9: `openai_test` stays a child of `openai`, so
`use super::*;` keeps resolving `openai`'s scope (including `openai`-private items the tests use,
e.g. `try_into_openai_request`). Do **not** modify `src/llm/mod.rs`.
- [ ] **Step 3: Verify** `cargo test -p mux 2>&1 | rg 'test result'` (count unchanged), then `./scripts/verify-refactor.sh`.
- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "refactor(llm): extract openai tests to openai_test.rs"
```

---

## Phase 3 — Production splits (paths preserved)

### Task 12: Split `openai.rs` by responsibility

**Files:**
- Create: `src/llm/openai/mod.rs`, `src/llm/openai/types.rs`, `src/llm/openai/convert.rs`, `src/llm/openai/response.rs`
- Delete: `src/llm/openai.rs`
- Move: `src/llm/openai_test.rs` → `src/llm/openai/openai_test.rs`

**HIGH-RISK:** 32 public items reach `mux::llm::*` via `pub use openai::*`. The public-item
name set MUST be identical before and after.

- [ ] **Step 1: Snapshot the public-item name set (before)**

Run:
```bash
rg -oN '^\s*pub (fn|struct|enum|trait|const|type) (\w+)' src/llm/openai.rs -r '$2' | sort -u > /tmp/openai_pub_before.txt
cat /tmp/openai_pub_before.txt | wc -l   # expect 32
```

- [ ] **Step 2: Create the module directory and move code by concern**

Convert `src/llm/openai.rs` into `src/llm/openai/` with:
- `types.rs` — all wire `pub struct`/`pub enum` (the `OpenAIRequest`…`OpenAIFunctionDelta` block, lines ~20-209) + their imports.
- `convert.rs` — `impl From<&ToolDefinition> for OpenAITool`, `try_openai_messages`, `try_media_part`, `audio_format_from_mime`, `uses_max_completion_tokens`, `try_into_openai_request` (lines ~265-539).
- `response.rs` — `impl From<OpenAIResponse> for Response`, `parse_stop_reason`, `parse_sse_line` (lines ~540-613).
- `mod.rs` — `OpenAIClient` struct + `impl OpenAIClient` + `impl super::client::LlmClient for OpenAIClient` (lines ~209-264, 614-808), PLUS the re-exports below.

Each new file starts with an ABOUTME header and its own `use` lines (pull what it needs from
`crate::llm::{...}`, `super::types::*`, `serde`, etc.).

- [ ] **Step 3: Re-export everything from `mod.rs` so `openai::*` is unchanged**

At the top of `src/llm/openai/mod.rs`:
```rust
// ABOUTME: OpenAI provider client and LlmClient implementation.
// ABOUTME: Wire types live in types.rs, request building in convert.rs, parsing in response.rs.
mod convert;
mod response;
mod types;

pub use convert::*;
pub use response::*;
pub use types::*;
```
This preserves `pub use openai::*` in `src/llm/mod.rs` verbatim — `src/llm/mod.rs` needs **no
change at all** (the test module is declared inside `openai`, not in `llm/mod.rs`).

- [ ] **Step 4: Snapshot the public-item name set (after) and diff**

Run:
```bash
rg -oN '^\s*pub (fn|struct|enum|trait|const|type) (\w+)' src/llm/openai/types.rs src/llm/openai/convert.rs src/llm/openai/response.rs src/llm/openai/mod.rs -r '$2' | sort -u > /tmp/openai_pub_after.txt
diff /tmp/openai_pub_before.txt /tmp/openai_pub_after.txt && echo "PUBLIC ITEM SET UNCHANGED"
```
Expected: `PUBLIC ITEM SET UNCHANGED`. Any diff = a moved/renamed/forgotten public item — fix before continuing.

- [ ] **Step 5: Relocate the test module**

Task 11 left the test declared **inside `openai.rs`** as
`#[cfg(test)] #[path = "openai_test.rs"] mod openai_test;` (NOT in `llm/mod.rs`). When `openai.rs`
becomes `openai/mod.rs`, move the file `src/llm/openai_test.rs` → `src/llm/openai/openai_test.rs`
and replace that trailing declaration in `openai/mod.rs` with the plain form (no `#[path]` needed —
a `mod.rs` resolves sibling files directly):
```rust
#[cfg(test)]
mod openai_test;
```
Leave `src/llm/mod.rs` untouched.

**Import gotcha (the real risk):** the test keeps `use super::*;`, but `super` is now
`openai/mod.rs`. Today the tests resolve `Request`, `Message`, `ContentBlock`, `Role`,
`MediaKind`, `ToolDefinition` through `openai.rs`'s own `use super::{...}` import. After the split,
`use super::*;` only sees what `openai/mod.rs` itself imports plus the
`pub use {types,convert,response}::*` re-exports — which do NOT include those parent-crate types.
Fix by adding the missing names to the **test file's** imports (e.g.
`use crate::llm::{ContentBlock, MediaKind, Message, Request, Role, ToolDefinition};`); let the
compiler name the unresolved ones. Do NOT add test-only `use`s to `openai/mod.rs` — an import the
non-test code doesn't use trips `clippy -D warnings` (`unused_imports`). Test logic and the test
count (13) stay identical.

- [ ] **Step 6: Verify**

Run:
```bash
cargo test -p mux 2>&1 | rg 'test result'      # count unchanged
cargo test --test public_api_surface            # guard green
./scripts/verify-refactor.sh                     # full spine
```
Expected: all green; `bindings byte-identical`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(llm): split openai into types/convert/response modules"
```

### Task 13: Tidy `messaging.rs` internals

**Files:**
- Create: `mux-ffi/src/engine/messaging/mod.rs`, `mux-ffi/src/engine/messaging/callback_hook.rs`
- Move: `messaging.rs` body and `messaging_test.rs` accordingly
- Modify: `mux-ffi/src/engine/mod.rs`

These are private `engine` submodules (not glob-exported); the external contract is the
`#[uniffi::export]` surface, guarded by the binding diff.

- [ ] **Step 1: Read `messaging.rs` end-to-end** to confirm the boundary.

Run: `sed -n '1,140p' mux-ffi/src/engine/messaging.rs`

- [ ] **Step 2: Create `messaging/` and move `ChatCallbackHook`**

- `messaging/callback_hook.rs` — `ChatCallbackHook` struct + `impl ChatCallbackHook` + `impl Hook for ChatCallbackHook` (lines ~24-105) and the `use` lines they need.
- `messaging/mod.rs` — `ffi_media_into_blocks` + the `impl MuxEngine` messaging block (lines ~106-635), plus:
  ```rust
  // ABOUTME: Engine messaging path: chat send, streaming, media-bearing turns.
  // ABOUTME: ChatCallbackHook lives in callback_hook.rs.
  mod callback_hook;
  use callback_hook::ChatCallbackHook;
  ```
- Move `messaging_test.rs` → `messaging/messaging_test.rs`; keep its `#[cfg(test)] mod messaging_test;` declaration in `messaging/mod.rs`.

- [ ] **Step 3: Update `engine/mod.rs`** — `mod messaging;` already resolves to `messaging/mod.rs`; no change unless the test declaration was there (move it into `messaging/mod.rs`).

- [ ] **Step 4: Verify** `cargo test -p mux-ffi 2>&1 | rg 'test result'` (count unchanged) and `./scripts/verify-refactor.sh` (`bindings byte-identical`).

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "refactor(mux-ffi): extract ChatCallbackHook from messaging"
```

### Task 14: Split `mcp.rs` by concern group

**Files:**
- Create: `mux-ffi/src/engine/mcp/mod.rs`, and one sibling per `impl MuxEngine` concern group (e.g. `mcp/lifecycle.rs`, `mcp/tools.rs`)
- Move: `mcp_test.rs` → `mcp/mcp_test.rs`
- Modify: `mux-ffi/src/engine/mod.rs`

- [ ] **Step 1: Read both `impl MuxEngine` blocks** (≈38-405 and 406-727) to name the two groups by responsibility (connection/lifecycle vs tool listing/execution).

Run: `rg -n 'pub(\(super\))? (async )?fn ' mux-ffi/src/engine/mcp.rs | sed -n '1,40p'`

- [ ] **Step 2: Create `mcp/` directory** with `mod.rs` holding shared items + module decls:
```rust
// ABOUTME: Engine MCP integration: server lifecycle and tool execution.
// ABOUTME: Split into lifecycle.rs and tools.rs; both extend impl MuxEngine.
mod lifecycle;
mod tools;
```
Rust allows multiple `impl MuxEngine` blocks across files in the same crate, so each sibling
file carries `impl MuxEngine { ... }` for its group plus the `use` it needs. Keep any
`McpClientHandle`/shared types in `mod.rs` (recall `engine/mod.rs` does `use mcp::McpClientHandle;`).

- [ ] **Step 3: Move `mcp_test.rs`** → `mcp/mcp_test.rs`; declare `#[cfg(test)] mod mcp_test;` in `mcp/mod.rs`.

- [ ] **Step 4: Confirm `McpClientHandle` still reachable** as `mcp::McpClientHandle` for `engine/mod.rs`:
```bash
rg -n 'McpClientHandle' mux-ffi/src/engine/mod.rs mux-ffi/src/engine/mcp/mod.rs
```

- [ ] **Step 5: Verify** `cargo test -p mux-ffi 2>&1 | rg 'test result'` and `./scripts/verify-refactor.sh`.

- [ ] **Step 6: Commit**
```bash
git add -A
git commit -m "refactor(mux-ffi): split mcp into lifecycle and tools modules"
```

---

## Phase 4 — Panic restructure (behavior-preserving) + audit docs

### Task 15: Remove the provably-safe streaming unwraps

**Files:**
- Modify: `src/llm/openai/mod.rs` (the `LlmClient::stream` impl; was `openai.rs:721`), `src/llm/ollama.rs:228`, `src/llm/openrouter.rs:217`, `src/llm/gemini.rs:590`

Each site sets the index to `Some(idx)` immediately before `.unwrap()`, so the unwrap cannot
fire today. Restructure to thread `idx` through both branches — provably identical output,
no unwrap. **No new tests** (this repo forbids mocks; the streaming path can't be unit-tested
without a fake server, and the change is equivalence-by-construction — verified by the spine).

- [ ] **Step 1: openai — replace the text-delta block**

Find (in the OpenAI `stream` impl):
```rust
                            if let Some(text) = choice.delta.content {
                                // Emit ContentBlockStart for text on first text delta
                                if text_block_index.is_none() {
                                    let idx = next_block_index;
                                    next_block_index += 1;
                                    yield StreamEvent::ContentBlockStart {
                                        index: idx,
                                        block: ContentBlock::Text { text: String::new() },
                                    };
                                    text_block_index = Some(idx);
                                }
                                yield StreamEvent::ContentBlockDelta {
                                    index: text_block_index.unwrap(),
                                    text,
                                };
                            }
```
Replace with:
```rust
                            if let Some(text) = choice.delta.content {
                                // Emit ContentBlockStart for text on first text delta.
                                let idx = match text_block_index {
                                    Some(idx) => idx,
                                    None => {
                                        let idx = next_block_index;
                                        next_block_index += 1;
                                        yield StreamEvent::ContentBlockStart {
                                            index: idx,
                                            block: ContentBlock::Text { text: String::new() },
                                        };
                                        text_block_index = Some(idx);
                                        idx
                                    }
                                };
                                yield StreamEvent::ContentBlockDelta { index: idx, text };
                            }
```

- [ ] **Step 2: ollama — same block, same replacement** (`src/llm/ollama.rs`, ~lines 215-231; the code is identical to OpenAI's). Apply the identical rewrite.

- [ ] **Step 3: openrouter — same block, same replacement** (`src/llm/openrouter.rs`, ~lines 204-220). Apply the identical rewrite.

- [ ] **Step 4: gemini — analogous block** (`src/llm/gemini.rs`, ~lines 579-593).

Find:
```rust
                                if let Some(text) = part.text {
                                    // Start text block if needed
                                    if current_text_index.is_none() {
                                        yield StreamEvent::ContentBlockStart {
                                            index: block_index,
                                            block: ContentBlock::Text { text: String::new() },
                                        };
                                        current_text_index = Some(block_index);
                                        block_index += 1;
                                    }
                                    yield StreamEvent::ContentBlockDelta {
                                        index: current_text_index.unwrap(),
                                        text,
                                    };
                                }
```
Replace with:
```rust
                                if let Some(text) = part.text {
                                    // Start text block if needed.
                                    let idx = match current_text_index {
                                        Some(idx) => idx,
                                        None => {
                                            let idx = block_index;
                                            yield StreamEvent::ContentBlockStart {
                                                index: idx,
                                                block: ContentBlock::Text { text: String::new() },
                                            };
                                            current_text_index = Some(idx);
                                            block_index += 1;
                                            idx
                                        }
                                    };
                                    yield StreamEvent::ContentBlockDelta { index: idx, text };
                                }
```

- [ ] **Step 5: Confirm no production streaming unwraps remain**

Run:
```bash
rg -n 'text_block_index.unwrap\(\)|current_text_index.unwrap\(\)' src/llm/
```
Expected: no matches.

- [ ] **Step 6: Verify spine**

Run: `./scripts/verify-refactor.sh`
Expected: clippy/fmt clean, all tests green (counts ≥ floor), `bindings byte-identical`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(llm): remove provably-safe streaming index unwraps"
```

### Task 16: Document the KEEP panic sites

**Files:**
- Modify: `src/tools/web_fetch.rs:27`, `src/tools/web_search.rs:35`, `mux-ffi/src/engine/messaging/mod.rs` (the `.expect("Custom provider…")`)

Add a brief comment above each `expect`/`unwrap` we deliberately keep, explaining why it is
safe and why converting it is out of scope (would change a public signature). Do NOT change
the code, only add comments.

- [ ] **Step 1: web_fetch.rs / web_search.rs** — above `.expect("Failed to create HTTP client")`:
```rust
            // Safe: reqwest client construction only fails on catastrophic TLS-backend
            // init. Returning Result here would change the public `new()` signature.
```

- [ ] **Step 2: messaging — above the `.expect("Custom provider was captured…")`:**
```rust
                // Safe: captured_custom_client is Some whenever provider is Custom (set
                // just above). The factory closure type is infallible by design, so
                // propagating an error would change FfiTaskTool's contract.
```

- [ ] **Step 3: Verify** `./scripts/verify-refactor.sh` (comments only; everything stays green).

- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "docs: explain the deliberately-retained panic sites"
```

---

## Phase 5 — Docs & hygiene

> **Spec deviation (resolved):** the spec proposed `missing_docs = "warn"`, but CI runs
> `clippy -D warnings`, which would promote it to a hard error and force documenting every
> public item (scope explosion). We therefore do NOT enable `missing_docs`; we add targeted
> doc-comments to the prelude-exported types instead. Flag for Doctor Biz if full
> `missing_docs` coverage is desired as a follow-up.

### Task 17: Fix the README install version

**Files:**
- Modify: `README.md:9`

- [ ] **Step 1:** Replace `mux = "0.1"` with `mux = "0.10"`.
- [ ] **Step 2: Verify** `rg -n 'mux = ' README.md` shows `0.10`.
- [ ] **Step 3: Commit**
```bash
git add README.md
git commit -m "docs: correct README install version to 0.10"
```

### Task 18: Doc-comment the prelude public types

**Files:**
- Modify: the defining files of prelude exports lacking a doc comment (e.g. `src/tool/traits.rs`, `src/tool/result.rs`, `src/permission/policy.rs`, `src/llm/types.rs`, `src/agent/*.rs`).

- [ ] **Step 1: Find prelude exports missing a doc comment**

Run (for each public type from `src/prelude.rs`, check it has a preceding `///`):
```bash
rg -n -B1 '^pub (struct|enum|trait) (Tool|ToolResult|Registry|Decision|Policy|PolicyBuilder|ApprovalHandler|ApprovalContext|AgentDefinition|SubAgent)\b' src/
```

- [ ] **Step 2: Add a one-line `///` to each undocumented prelude type.** Keep it factual and
evergreen (describe what it is, not history). Example:
```rust
/// A registry of executable tools, keyed by name.
pub struct Registry { /* ... */ }
```
Do not touch items that already have docs. This is additive; signatures unchanged.

- [ ] **Step 3: Verify** `cargo doc --no-deps -p mux 2>&1 | rg -i 'warning|error' || echo "docs clean"`, then `./scripts/verify-refactor.sh`.

- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "docs: document prelude-exported public types"
```

### Task 19: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1:** Under `## [Unreleased]`, add:
```markdown
### Changed

- Internal robustness refactor: enforced `cargo fmt` + `clippy -D warnings` in CI via a
  shared `[workspace.lints]` posture; extracted inline test modules to sibling `*_test.rs`
  files; split `openai`, `messaging`, and `mcp` into focused submodules; removed dead code;
  documented retained panic sites. **No public API change and no behavioral change**
  (the Rust public surface and the UniFFI Swift/Kotlin bindings are byte-identical, verified
  by snapshot diff).
```
- [ ] **Step 2: Verify** the file renders (no broken markdown) and `./scripts/verify-refactor.sh`.
- [ ] **Step 3: Commit**
```bash
git add CHANGELOG.md
git commit -m "docs: changelog entry for the robustness refactor"
```

---

## Phase 6 — Final verification & finish

### Task 20: Full proof + branch finish

- [ ] **Step 1: Full spine, clean room**

Run:
```bash
cargo clean -p mux -p mux-ffi
./scripts/verify-refactor.sh
```
Expected: fmt OK, clippy `-D warnings` OK, all tests green (count ≥ Task 2 floor),
`FFI bindings byte-identical OK`.

- [ ] **Step 2: Re-affirm the public-item set for openai**

Run:
```bash
rg -oN '^\s*pub (fn|struct|enum|trait|const|type) (\w+)' src/llm/openai/*.rs -r '$2' | sort -u | wc -l
```
Expected: `32` (matches the Task 12 before-snapshot).

- [ ] **Step 3: Confirm no giant files remain**

Run: `find src mux-ffi/src -name '*.rs' -not -name '*_test.rs' | xargs wc -l | sort -rn | head -5`
Expected: every production file comfortably screen-sized (target < ~600 LOC; flag any outliers to Doctor Biz).

- [ ] **Step 4: Review the whole diff**

Run: `git log --oneline main..HEAD && git diff --stat main...HEAD`
Confirm every commit traces to a task and no stray behavior change slipped in.

- [ ] **Step 5: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to open a PR titled
`refactor: robustness & cleanup pass (no API/behavior change)` with a summary linking the
spec and listing the three proofs (test floor, public-item diff, empty binding diff).

---

## Self-Review

**Spec coverage:** §1 CI/lint → Tasks 3, 8. §2 mechanical cleanup → Tasks 4, 5, 6, 7. §3 test
extraction → Tasks 9–11. §4 production splits → Tasks 12–14. §5 panic audit → Tasks 15
(restructure), 16 (document KEEP sites). §6 docs/hygiene → Tasks 17–19. Verification spine →
Tasks 1, 2, used every phase. Finish → Task 20. **All spec sections covered.**

**Resolved deviations:** (1) `missing_docs` lint dropped to avoid the `-D warnings`
contradiction (Phase 5 note); doc value retained via Task 18. (2) §5 "new test per client"
superseded — no-mock rule + un-unit-testable streaming path; replaced by provable-equivalence
+ spine (Task 15). Both flagged for Doctor Biz.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Split tasks (12–14) give exact item-to-file
mapping, the critical re-export code, and a name-set diff check rather than reproducing
thousands of moved lines verbatim — appropriate for a mechanical move; the moved code is
unchanged by definition.

**Type consistency:** `LlmClientFactory` (Task 5) used consistently. `verify-refactor.sh`
interface (`check`/`baseline`) consistent across all references. Test-module convention
(`#[cfg(test)] mod <name>_test;` + `use super::*;`) consistent across Tasks 9–14.
