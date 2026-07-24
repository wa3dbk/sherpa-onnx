#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Swift OmniVoice demo. Builds sherpa-onnx as a static xcframework via
# ../build-swift-macos.sh if needed, then compiles tts-omnivoice.swift
# against it and runs the binary against a bundle under
# ./sherpa-onnx-omnivoice-en (or $BUNDLE_DIR).

set -ex

BUNDLE_DIR=${BUNDLE_DIR:-./sherpa-onnx-omnivoice-en}
BUNDLE_URL=${OMNIVOICE_BUNDLE_URL:-}

if [ ! -f "${BUNDLE_DIR}/omnivoice.onnx" ]; then
  if [ -n "${BUNDLE_URL}" ]; then
    tarball=$(basename "${BUNDLE_URL}")
    curl -SL -O "${BUNDLE_URL}"
    tar xvf "${tarball}"
    rm "${tarball}"
  else
    echo "No bundle at ${BUNDLE_DIR}. Build one with:"
    echo "  (cd ../scripts/omnivoice && ./build_bundle.sh)"
    exit 1
  fi
fi

# Follow the pattern established by run-tts-zipvoice.sh.
if [ ! -d ../build-swift-macos ]; then
  (cd .. && ./build-swift-macos.sh)
fi

swiftc \
  -lc++ \
  -I ../build-swift-macos/install/include \
  -L ../build-swift-macos/install/lib \
  -l sherpa-onnx \
  -l onnxruntime \
  -import-objc-header ./SherpaOnnx-Bridging-Header.h \
  ./tts-omnivoice.swift ./SherpaOnnx.swift \
  -o tts-omnivoice

BUNDLE_DIR="${BUNDLE_DIR}" \
  DYLD_LIBRARY_PATH=../build-swift-macos/install/lib \
    ./tts-omnivoice
