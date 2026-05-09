.PHONY: demo code-agent test build clean clippy fmt check

# Run the agent-test-tui demo (MCP tools)
demo:
	cargo run -p agent-test-tui

# Run the code-agent (built-in coding tools)
code-agent:
	cargo run -p code-agent

# Run demos in release mode
demo-release:
	cargo run -p agent-test-tui --release

code-agent-release:
	cargo run -p code-agent --release

# Run all tests
test:
	cargo test

# Build everything
build:
	cargo build --all-targets

# Build release
release:
	cargo build --release --all-targets

# Run clippy
clippy:
	cargo clippy --all-targets

# Format code
fmt:
	cargo fmt

# Check formatting and clippy
check: fmt clippy test
	@echo "All checks passed"

# Clean build artifacts
clean:
	cargo clean

# ----- release -----
# Cut a release. Updates both Cargo.toml versions in lockstep, runs tests,
# commits, and creates a vX.Y.Z tag. Does NOT push — review first, then:
#   git push origin main --tags
#
# Requires: cargo install cargo-release

.PHONY: release-patch release-minor release-major release-dry-run

release-patch:
	cargo release patch --execute --no-confirm

release-minor:
	cargo release minor --execute --no-confirm

release-major:
	cargo release major --execute --no-confirm

release-dry-run:
	cargo release patch
