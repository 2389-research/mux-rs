#!/bin/bash
# ABOUTME: Builds mux-ffi for Android and generates Kotlin bindings
# ABOUTME: Creates .so libraries for arm64, arm32, and x86_64 architectures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Check for Android NDK
if [ -z "$ANDROID_NDK_HOME" ]; then
  # Try common locations
  if [ -d "$HOME/Library/Android/sdk/ndk" ]; then
    ANDROID_NDK_HOME=$(ls -d "$HOME/Library/Android/sdk/ndk"/*/ 2>/dev/null | head -1 | sed 's:/$::')
  elif [ -d "$ANDROID_HOME/ndk" ]; then
    ANDROID_NDK_HOME=$(ls -d "$ANDROID_HOME/ndk"/*/ 2>/dev/null | head -1 | sed 's:/$::')
  fi

  if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "ERROR: ANDROID_NDK_HOME not set and could not find NDK"
    echo "Please set ANDROID_NDK_HOME to your Android NDK installation path"
    exit 1
  fi
  echo "==> Auto-detected NDK at: $ANDROID_NDK_HOME"
fi

# Add NDK toolchain to PATH
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin:$PATH"
export PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"

# Install targets if needed
echo "==> Ensuring Android targets are installed..."
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android i686-linux-android 2>/dev/null || true

# Build for each target
echo "==> Building for Android arm64 (aarch64-linux-android)..."
cargo build --release --target aarch64-linux-android -p mux-ffi

ARM64_LIB="target/aarch64-linux-android/release/libmux_ffi.so"
if [ ! -f "$ARM64_LIB" ]; then
  echo "ERROR: arm64 library not found at $ARM64_LIB"
  exit 1
fi

echo "==> Building for Android arm32 (armv7-linux-androideabi)..."
cargo build --release --target armv7-linux-androideabi -p mux-ffi

ARM32_LIB="target/armv7-linux-androideabi/release/libmux_ffi.so"
if [ ! -f "$ARM32_LIB" ]; then
  echo "ERROR: arm32 library not found at $ARM32_LIB"
  exit 1
fi

echo "==> Building for Android x86_64 (x86_64-linux-android)..."
cargo build --release --target x86_64-linux-android -p mux-ffi

X86_64_LIB="target/x86_64-linux-android/release/libmux_ffi.so"
if [ ! -f "$X86_64_LIB" ]; then
  echo "ERROR: x86_64 library not found at $X86_64_LIB"
  exit 1
fi

echo "==> Building for Android x86 (i686-linux-android)..."
cargo build --release --target i686-linux-android -p mux-ffi

X86_LIB="target/i686-linux-android/release/libmux_ffi.so"
if [ ! -f "$X86_LIB" ]; then
  echo "ERROR: x86 library not found at $X86_LIB"
  exit 1
fi

echo "==> Generating Kotlin bindings..."
uniffi-bindgen generate \
  --library target/aarch64-linux-android/release/libmux_ffi.so \
  --language kotlin \
  --out-dir mux-ffi/bindings-kotlin

# Output directory for hibi-android
OUTPUT_DIR="${1:-../hibi-android/app/src/main}"
mkdir -p "$OUTPUT_DIR/jniLibs/arm64-v8a"
mkdir -p "$OUTPUT_DIR/jniLibs/armeabi-v7a"
mkdir -p "$OUTPUT_DIR/jniLibs/x86_64"
mkdir -p "$OUTPUT_DIR/jniLibs/x86"
mkdir -p "$OUTPUT_DIR/kotlin"

echo "==> Copying .so libraries to $OUTPUT_DIR/jniLibs..."
cp "$ARM64_LIB" "$OUTPUT_DIR/jniLibs/arm64-v8a/"
cp "$ARM32_LIB" "$OUTPUT_DIR/jniLibs/armeabi-v7a/"
cp "$X86_64_LIB" "$OUTPUT_DIR/jniLibs/x86_64/"
cp "$X86_LIB" "$OUTPUT_DIR/jniLibs/x86/"

echo "==> Copying Kotlin bindings..."
if [ ! -d "mux-ffi/bindings-kotlin" ]; then
  echo "ERROR: Kotlin bindings directory not found at mux-ffi/bindings-kotlin"
  exit 1
fi
cp -r mux-ffi/bindings-kotlin/* "$OUTPUT_DIR/kotlin/"

echo "==> Done! Libraries and bindings at: $OUTPUT_DIR"
echo ""
echo "jniLibs structure:"
find "$OUTPUT_DIR/jniLibs" -name "*.so" | head -10
echo ""
echo "Kotlin bindings:"
find "$OUTPUT_DIR/kotlin" -name "*.kt" | head -10
