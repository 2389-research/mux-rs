# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- **Opt-in tool confinement (`mux::confine`).** Two off-by-default guardrails for the built-in tools, with no behavior change for existing callers.
  - **Filesystem jail.** `RootedFs::new(root)` plus `ReadFileTool::rooted`, `WriteFileTool::rooted`, `EditTool::rooted`, `SearchTool::rooted`, and `ListFilesTool::rooted` confine the five filesystem tools to a canonicalized root, rejecting `..` traversal and symlink escapes and dropping glob hits that resolve outside the root. Violations return a tool error result rather than aborting the run.
  - **`web_fetch` SSRF guard.** `WebFetchTool::guarded()` (and `with_url_policy`) deny unspecified/loopback/RFC1918/link-local/CGNAT/ULA addresses via `UrlPolicy::public_only()` / `is_globally_routable`, re-validating every redirect hop with manual redirect following.
  - **FFI.** Additive `MuxEngine::new_confined(data_dir, root)` builds a rooted filesystem toolset for the Swift/Kotlin consumer; the existing `new` is unchanged.
  - New public symbols `RootedFs`, `UrlPolicy`, `ConfinementError`, and `is_globally_routable` are exported from `mux::confine` and the prelude. See `docs/confining-mux.md`. This is in-process defense against a confused/injected model — not a sandbox; the filesystem jail is moot unless `bash` is also dropped or OS-sandboxed.

## [0.14.0] - 2026-06-09

### Added

- **Anthropic prompt caching.** `CacheControl::ephemeral()`, `SystemBlock::new(text)`, and `SystemBlock::cached(text)` are new public types. `Request::system_block(block)` / `Request::system_blocks(blocks)` set structured system content with optional cache markers, and `Request::effective_system()` provides a flat-string view for non-Anthropic providers. `ToolDefinition.cache_control` is a new optional field that propagates to the Anthropic wire format. Non-Anthropic providers (OpenAI, Gemini, Ollama, OpenRouter) silently ignore cache markers — no caller change needed.
- **`AgentDefinition` cache opt-ins.** `system_blocks: Vec<SystemBlock>` (builders `system_blocks(...)` / `system_block(...)`) takes precedence over the plain `system_prompt` string at request-build time when non-empty. `cache_tools: bool` (builder `cache_tools(bool)`) marks the last tool definition with an ephemeral `cache_control`, caching the whole tool block as one Anthropic breakpoint. Both default to empty/false; existing `AgentDefinition::new` callers are unchanged.
- **Task tool wired into the mux-ffi chat loop.** When a subagent event handler is set, the chat-loop tool registry now registers the `task` tool on every turn. Previously the subagent infrastructure existed but the tool was never registered, so the model had no `task` tool to dispatch.
- **MIT `LICENSE`** added at the repository root; `license = "MIT"` backfilled to the `agent-test-tui` and `code-agent` crates for workspace consistency.

### Changed

- Missing environment-variable keys now return `LlmError::Configuration` instead of a spurious `LlmError::Api { status: 0 }`, across the `from_env` / `from_env_var` constructors (OpenAI, OpenRouter, Anthropic, Gemini).
- `from_env_var` now rejects empty or whitespace-only API keys at construction time with `LlmError::Configuration`, rather than forwarding a blank key that fails later with a confusing 401.
- Malformed tool-argument JSON no longer leaks payload bytes in error messages: OpenAI/OpenRouter/Ollama parsing returns `LlmError::Configuration` naming the tool and the serde error position, without any fragment of the raw arguments string.
- Streaming responses now propagate `cache_read_input_tokens` and `cache_creation_input_tokens`; both `MessageStart` and `MessageDelta` usage events are merged so streaming callers see the same `Usage` fields as non-streaming callers.
- Tool definitions are sorted by name before the cache breakpoint is applied, making the cached tool block byte-identical across calls (previously non-deterministic registry order caused Anthropic to treat every call as a cache miss).
- Internal robustness refactor: enforced `cargo fmt` + `clippy -D warnings` in CI via a shared `[workspace.lints]` posture; extracted inline test modules to sibling `*_test.rs` files; split `openai`, `messaging`, and `mcp` into focused submodules; removed dead code; documented retained panic sites. **No public API change and no behavioral change** (the Rust public surface and the UniFFI Swift/Kotlin bindings are byte-identical, verified by snapshot diff).

### Fixed

- OpenAI/Ollama/OpenRouter tool-argument serialization errors are now propagated rather than silently substituted with empty strings or `Value::Null`. `From<OpenAIResponse> for Response` is now `TryFrom`, returning `LlmError::Configuration` on malformed `arguments` JSON; outbound serialization failures also propagate instead of silently dropping tool calls.
- MCP connect/disconnect lifecycle is serialized per workspace on a `tokio::sync::Mutex`, eliminating a race where a disconnect could run against an empty map while a concurrent connect's handles arrived afterward.
- MCP reconnect no longer leaks prior handles: `connect_workspace_servers` shuts down existing `McpClientHandle`s before overwriting the workspace entry, preventing stdio child-process and SSE-session leaks on repeated connects.
- MCP server names are validated at the FFI boundary: `add_mcp_server` / `update_mcp_server` reject empty names and names containing `:`, preventing silent mis-routing in `parse_qualified_tool_name`.
- mux-ffi: eliminated a lock-held-across-`.await` in subagent spawn/resume — `do_spawn_agent` and `do_resume_agent` previously held a `parking_lot::RwLock` read guard across `.await` during tool registration; replaced with a snapshot-before-await pattern.
- mux-ffi: closed a TOCTOU between Custom provider validation and capture in `try_build_ffi_task_tool`, where a concurrent `unregister_llm_provider` between validation and factory-closure population could panic now that the task tool is wired live; the two reads are collapsed into one atomic match.

## [0.13.0] - 2026-05-05

### Added

- Multimodal input across all four media kinds (image / audio / video / pdf), with provider support for Anthropic, OpenAI/OpenRouter, Gemini, and Ollama.
- mux-ffi multimodal types: `FfiMedia`, `FfiMediaKind`, and `FfiMediaSource`, with token estimation.
- `LlmError::UnsupportedSource { provider, kind, source_kind }` for (provider, source) mismatches.
- `LlmError::MediaTooLarge { limit, actual }` for oversize local files.
- `MediaSourceKind` enum (`Base64` | `Url` | `Path`) exposed alongside `MediaSource`, used in the new error variant.
- `MAX_MEDIA_BYTES` constant exposed from `src/llm/media.rs`.

### Changed

- **Breaking (Swift/FFI):** The library no longer fetches URLs for media attachments. Previously, `MediaSource::Url` was fetched and inlined as base64 for providers that don't accept URL sources natively (Gemini). Now, Gemini rejects `MediaSource::Url` pre-flight with `LlmError::UnsupportedSource`. Callers must fetch the bytes themselves (e.g., via `URLSession` on Apple platforms) and pass `MediaSource::Base64`.
- Anthropic, OpenAI, and OpenRouter continue to pass URL sources through natively — no caller change needed for those providers.
- Local file reads via `MediaSource::Path` are now bounded by `MAX_MEDIA_BYTES` (20MB); oversize files return `LlmError::MediaTooLarge`.

### Removed

- `resolve_request_media_fully` (no longer needed — URL never fetched).
- `validate_fetchable_url` and `is_public_ip` SSRF helpers (moot — URL fetch gone).
