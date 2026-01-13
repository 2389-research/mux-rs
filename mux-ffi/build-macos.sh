#!/bin/bash
# ABOUTME: Builds mux-ffi for macOS and creates XCFramework
# ABOUTME: Generates Swift bindings via UniFFI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "==> Building for macOS Apple Silicon (aarch64-apple-darwin)..."
cargo build --release --target aarch64-apple-darwin -p mux-ffi

MACOS_LIB="target/aarch64-apple-darwin/release/libmux_ffi.a"
if [ ! -f "$MACOS_LIB" ]; then
  echo "ERROR: macOS library not found at $MACOS_LIB"
  exit 1
fi

echo "==> Generating Swift bindings..."
uniffi-bindgen generate \
  --library "$MACOS_LIB" \
  --language swift \
  --out-dir mux-ffi/bindings-macos

echo "==> Renaming modulemap for SwiftPM compatibility..."
if [ -f "mux-ffi/bindings-macos/MuxFFIFFI.modulemap" ]; then
  mv mux-ffi/bindings-macos/MuxFFIFFI.modulemap mux-ffi/bindings-macos/module.modulemap
fi

OUTPUT_DIR="${1:-../homeoffice/MuxFFI}"
mkdir -p "$OUTPUT_DIR"

echo "==> Creating XCFramework at $OUTPUT_DIR..."
rm -rf "$OUTPUT_DIR/MuxFFI.xcframework"

xcodebuild -create-xcframework \
  -library "$MACOS_LIB" \
  -headers mux-ffi/bindings-macos \
  -output "$OUTPUT_DIR/MuxFFI.xcframework"

echo "==> Copying Swift bindings..."
if [ ! -d "mux-ffi/bindings-macos" ]; then
  echo "ERROR: Bindings directory not found at mux-ffi/bindings-macos"
  exit 1
fi
cp mux-ffi/bindings-macos/*.swift "$OUTPUT_DIR/"

echo "==> Done! XCFramework and bindings at: $OUTPUT_DIR"
