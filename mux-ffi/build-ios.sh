#!/bin/bash
# ABOUTME: Builds mux-ffi for iOS and creates XCFramework
# ABOUTME: Generates Swift bindings via UniFFI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==> Building for iOS device (aarch64-apple-ios)..."
cargo build --release --target aarch64-apple-ios -p mux-ffi

echo "==> Building for iOS simulator (aarch64-apple-ios-sim)..."
cargo build --release --target aarch64-apple-ios-sim -p mux-ffi

echo "==> Generating Swift bindings..."
uniffi-bindgen generate \
  --library target/aarch64-apple-ios/release/libmux_ffi.a \
  --language swift \
  --out-dir mux-ffi/bindings

OUTPUT_DIR="${1:-../poka/MuxFFI}"
mkdir -p "$OUTPUT_DIR"

echo "==> Creating XCFramework at $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR/MuxFFI.xcframework"

xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/release/libmux_ffi.a \
  -headers mux-ffi/bindings \
  -library target/aarch64-apple-ios-sim/release/libmux_ffi.a \
  -headers mux-ffi/bindings \
  -output "$OUTPUT_DIR/MuxFFI.xcframework"

echo "==> Copying Swift bindings..."
cp mux-ffi/bindings/*.swift "$OUTPUT_DIR/"

echo "==> Done! XCFramework and bindings at: $OUTPUT_DIR"
