#!/usr/bin/env bash
# Fetch the ONNX Runtime 1.31.0 shared library for the face-recognition
# engine into src-tauri/resources/, where the platform Tauri configs
# (tauri.windows.conf.json / tauri.linux.conf.json / tauri.macos.conf.json)
# pick it up as a bundle resource. The app locates it at runtime through
# the Tauri resource dir (see face_infer::onnxruntime_dylib).
#
# Usage: bash src-tauri/fetch-ort.sh [windows|macos|linux]
# With no argument the platform is detected from uname. The downloaded
# binaries are git-ignored; release builds run this in CI
# (.github/workflows/tauri-release.yml).
set -euo pipefail

ORT_VERSION="1.31.0"
BASE="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}"
cd "$(dirname "$0")"
mkdir -p resources
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

platform="${1:-}"
if [ -z "$platform" ]; then
  case "$(uname -s)" in
    Linux) platform=linux ;;
    Darwin) platform=macos ;;
    MINGW*|MSYS*|CYGWIN*) platform=windows ;;
    *) echo "cannot detect platform; pass windows|macos|linux" >&2; exit 1 ;;
  esac
fi

case "$platform" in
  windows)
    curl -sL -o "$TMP/ort.zip" "$BASE/onnxruntime-win-x64-${ORT_VERSION}.zip"
    unzip -o -q "$TMP/ort.zip" -d "$TMP/ort"
    cp "$TMP/ort/onnxruntime-win-x64-${ORT_VERSION}/lib/onnxruntime.dll" resources/
    cp "$TMP/ort/onnxruntime-win-x64-${ORT_VERSION}/lib/onnxruntime_providers_shared.dll" resources/
    # Note: this is the CPU package. DirectML/CUDA provider libraries are
    # distributed separately (NuGet / CUDA toolkit); placing a different
    # onnxruntime.dll build next to the app (or via ORT_DYLIB_PATH)
    # enables them - the provider ladder resolves what the library has.
    ;;
  macos)
    # Upstream ships only an arm64 package for 1.31.0; on Intel Macs the
    # library will not load and face features stay off (the app itself
    # is a universal binary and runs normally).
    curl -sL -o "$TMP/ort.tgz" "$BASE/onnxruntime-osx-arm64-${ORT_VERSION}.tgz"
    tar -xzf "$TMP/ort.tgz" -C "$TMP"
    cp -L "$TMP/onnxruntime-osx-arm64-${ORT_VERSION}/lib/libonnxruntime.${ORT_VERSION}.dylib" \
      resources/libonnxruntime.dylib
    ;;
  linux)
    curl -sL -o "$TMP/ort.tgz" "$BASE/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    tar -xzf "$TMP/ort.tgz" -C "$TMP"
    cp -L "$TMP/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" \
      resources/libonnxruntime.so
    ;;
  *)
    echo "unknown platform: $platform (expected windows|macos|linux)" >&2
    exit 1
    ;;
esac

echo "ONNX Runtime $ORT_VERSION ($platform) placed in src-tauri/resources/:"
ls -la resources/
