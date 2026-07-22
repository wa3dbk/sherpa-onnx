#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# End-to-end sanity test for the OmniVoice TTS backend.
#
# Usage:
#   SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR=./sherpa-onnx-omnivoice-XXXX \
#   SHERPA_ONNX_OFFLINE_TTS_BIN=./build/bin/sherpa-onnx-offline-tts \
#     scripts/omnivoice/test_e2e.sh
#
# Skips (exit 0) if the bundle dir is not set, so CI can wire it in as a
# conditional job without failing on machines that don't have the model.
#
# Checks:
#   1. CLI produces a non-empty WAV whose duration is in a sane range.
#   2. Same --seed produces byte-identical output twice (determinism).
#   3. --split-strategy=sentence produces audio at least as long as a single
#      chunk on the same paragraph (chunking doesn't drop content).
#   4. --output-stream=1 writes a non-empty byte stream to stdout.

set -euo pipefail

BUNDLE=${SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR:-}
BIN=${SHERPA_ONNX_OFFLINE_TTS_BIN:-./build/bin/sherpa-onnx-offline-tts}
OUT_DIR=${SHERPA_ONNX_OMNIVOICE_TEST_OUT:-./omnivoice_e2e_out}

if [ -z "${BUNDLE}" ]; then
  echo "[e2e] SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR is not set; skipping."
  exit 0
fi
if [ ! -f "${BIN}" ]; then
  echo "[e2e] binary not found: ${BIN}"; exit 1
fi
for f in omnivoice.onnx higgs_codec_encoder.onnx higgs_codec_decoder.onnx \
         tokenizer test_wavs/ref.wav test_wavs/ref.txt; do
  if [ ! -e "${BUNDLE}/${f}" ]; then
    echo "[e2e] missing ${BUNDLE}/${f}"; exit 1
  fi
done

mkdir -p "${OUT_DIR}"
REF_WAV="${BUNDLE}/test_wavs/ref.wav"
REF_TXT_CONTENT="$(cat "${BUNDLE}/test_wavs/ref.txt")"
TEXT_SHORT="The quick brown fox jumps over the lazy dog."
TEXT_LONG="First sentence for the chunking test. Here is a second one, a bit \
longer, to give the splitter something to work with. And a third, which should \
also cross the sentence boundary cleanly. Finally, a fourth."

common_args=(
  --omnivoice-model="${BUNDLE}/omnivoice.onnx"
  --omnivoice-codec-encoder="${BUNDLE}/higgs_codec_encoder.onnx"
  --omnivoice-codec-decoder="${BUNDLE}/higgs_codec_decoder.onnx"
  --omnivoice-tokenizer-dir="${BUNDLE}/tokenizer"
  --reference-audio="${REF_WAV}"
  --reference-text="${REF_TXT_CONTENT}"
  --num-threads=2
)

wav_bytes() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }

echo "[e2e] 1/4 basic synthesis"
"${BIN}" "${common_args[@]}" \
  --seed=42 \
  --output-filename="${OUT_DIR}/basic.wav" \
  "${TEXT_SHORT}" >/dev/null
size=$(wav_bytes "${OUT_DIR}/basic.wav")
if [ "${size}" -lt 20000 ]; then
  echo "[e2e] FAIL: basic.wav is only ${size} bytes"; exit 1
fi
echo "[e2e]   OK (${size} bytes)"

echo "[e2e] 2/4 determinism with seed=42"
"${BIN}" "${common_args[@]}" \
  --seed=42 \
  --output-filename="${OUT_DIR}/det_a.wav" \
  "${TEXT_SHORT}" >/dev/null
"${BIN}" "${common_args[@]}" \
  --seed=42 \
  --output-filename="${OUT_DIR}/det_b.wav" \
  "${TEXT_SHORT}" >/dev/null
if ! cmp -s "${OUT_DIR}/det_a.wav" "${OUT_DIR}/det_b.wav"; then
  echo "[e2e] FAIL: two runs with --seed=42 differ (determinism broken)"
  ls -l "${OUT_DIR}/det_a.wav" "${OUT_DIR}/det_b.wav"
  exit 1
fi
echo "[e2e]   OK (identical bytes)"

echo "[e2e] 3/4 sentence chunking preserves total duration"
"${BIN}" "${common_args[@]}" \
  --seed=42 \
  --output-filename="${OUT_DIR}/nosplit.wav" \
  "${TEXT_LONG}" >/dev/null
"${BIN}" "${common_args[@]}" \
  --seed=42 \
  --split-strategy=sentence \
  --split-max-chars=200 \
  --output-filename="${OUT_DIR}/split.wav" \
  "${TEXT_LONG}" >/dev/null
n1=$(wav_bytes "${OUT_DIR}/nosplit.wav")
n2=$(wav_bytes "${OUT_DIR}/split.wav")
# Chunked output includes silence gaps, so it should be >= 90% of unsplit.
# Being strictly larger is fine (gaps). Much shorter means we dropped text.
if [ "${n2}" -lt $((n1 * 9 / 10)) ]; then
  echo "[e2e] FAIL: split.wav (${n2}) is much shorter than nosplit.wav (${n1})"
  exit 1
fi
echo "[e2e]   OK (nosplit=${n1}, split=${n2})"

echo "[e2e] 4/4 --output-stream writes a non-empty byte stream"
stream_bytes=$(
  "${BIN}" "${common_args[@]}" \
    --seed=42 \
    --output-stream=1 \
    "${TEXT_SHORT}" 2>/dev/null | wc -c | tr -d ' '
)
if [ "${stream_bytes}" -lt 10000 ]; then
  echo "[e2e] FAIL: stream produced only ${stream_bytes} bytes"; exit 1
fi
echo "[e2e]   OK (${stream_bytes} bytes on stdout)"

echo "[e2e] all checks passed"
