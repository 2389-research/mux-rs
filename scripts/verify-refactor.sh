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
if [ ! -d "$OUT/baseline" ]; then
  echo "No baseline at $OUT/baseline — run: $0 baseline"; exit 1
fi
gen_bindings "$OUT/current"
if diff -ru "$OUT/baseline" "$OUT/current"; then
  echo "FFI bindings byte-identical OK"
else
  echo "FFI BINDINGS CHANGED — refactor altered the Swift/Kotlin contract"; exit 1
fi
