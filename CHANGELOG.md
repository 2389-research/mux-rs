# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- **Breaking (Swift/FFI):** The library no longer fetches URLs for media attachments. Previously, `MediaSource::Url` was fetched and inlined as base64 for providers that don't accept URL sources natively (Gemini). Now, Gemini rejects `MediaSource::Url` pre-flight with `LlmError::UnsupportedSource`. Callers must fetch the bytes themselves (e.g., via `URLSession` on Apple platforms) and pass `MediaSource::Base64`.
- Anthropic, OpenAI, and OpenRouter continue to pass URL sources through natively — no caller change needed for those providers.
- Local file reads via `MediaSource::Path` are now bounded by `MAX_MEDIA_BYTES` (20MB); oversize files return `LlmError::MediaTooLarge`.

### Added

- `LlmError::UnsupportedSource { provider, kind, source_kind }` for (provider, source) mismatches.
- `LlmError::MediaTooLarge { limit, actual }` for oversize local files.
- `MediaSourceKind` enum (`Base64` | `Url` | `Path`) exposed alongside `MediaSource`, used in the new error variant.
- `MAX_MEDIA_BYTES` constant exposed from `src/llm/media.rs`.

### Removed

- `resolve_request_media_fully` (no longer needed — URL never fetched).
- `validate_fetchable_url` and `is_public_ip` SSRF helpers (moot — URL fetch gone).
