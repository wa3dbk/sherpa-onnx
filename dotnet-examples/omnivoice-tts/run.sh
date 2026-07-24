#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Runs the .NET OmniVoice demo. Expects a bundle produced by
# scripts/omnivoice/build_bundle.sh under ./sherpa-onnx-omnivoice-en
# (or set OMNIVOICE_BUNDLE_URL to a public tarball to auto-download).

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
    echo "No bundle found at ${BUNDLE_DIR}."
    echo "Build one with: (cd ../../scripts/omnivoice && ./build_bundle.sh)"
    echo "Then symlink or move the result to ${BUNDLE_DIR}."
    exit 1
  fi
fi

BUNDLE_DIR="${BUNDLE_DIR}" dotnet run
