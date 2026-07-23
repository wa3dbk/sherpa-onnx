#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Benchmark harness for OmniVoice. Sweeps:
#   * provider   : cpu, cuda (if available in your ORT build)
#   * cached LM  : off, on (needs omnivoice_prefix.onnx + omnivoice_target.onnx)
#   * split      : none, sentence
#   * text len   : short (~10 words), medium (~50 words), long (~200 words)
#
# For each cell it runs the CLI, parses "Elapsed seconds" / "Audio duration"
# / "Real-time factor (RTF)" from stderr, and prints a markdown row you can
# paste under the <!-- BEGIN PERF TABLE --> marker in README.md.
#
# Usage:
#   SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR=./sherpa-onnx-omnivoice-XXXX \
#   SHERPA_ONNX_OFFLINE_TTS_BIN=./build/bin/sherpa-onnx-offline-tts \
#     scripts/omnivoice/bench.sh
#
# Optional:
#   BENCH_PROVIDERS="cpu cuda"    # default "cpu"
#   BENCH_STEPS=32
#   BENCH_THREADS=4

set -euo pipefail

BUNDLE=${SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR:-}
BIN=${SHERPA_ONNX_OFFLINE_TTS_BIN:-./build/bin/sherpa-onnx-offline-tts}
PROVIDERS=${BENCH_PROVIDERS:-cpu}
STEPS=${BENCH_STEPS:-32}
THREADS=${BENCH_THREADS:-4}
SEED=${BENCH_SEED:-42}

if [ -z "${BUNDLE}" ]; then
  echo "set SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR" >&2; exit 1
fi

REF_WAV="${BUNDLE}/test_wavs/ref.wav"
REF_TXT="$(cat "${BUNDLE}/test_wavs/ref.txt")"

TEXT_SHORT="Hello world, this is a short benchmark line."
TEXT_MEDIUM="This is a medium-length benchmark passage that runs across a \
couple of sentences. It exercises the language model on enough tokens to \
give a stable timing signal, without pushing into paragraph territory."
TEXT_LONG="$(printf '%s ' "${TEXT_MEDIUM}" "${TEXT_MEDIUM}" "${TEXT_MEDIUM}" \
                        "${TEXT_MEDIUM}")"

echo "| Provider | Cached LM | Split strategy | Text length (chars) | Steps | Threads | Latency (s) | Audio (s) | RTF |"
echo "|---|---|---|---|---|---|---|---|---|"

run_one() {
  local provider="$1" cached_flag="$2" split="$3" text="$4" label="$5"
  local args=(
    --omnivoice-model="${BUNDLE}/omnivoice.onnx"
    --omnivoice-codec-encoder="${BUNDLE}/higgs_codec_encoder.onnx"
    --omnivoice-codec-decoder="${BUNDLE}/higgs_codec_decoder.onnx"
    --omnivoice-tokenizer-dir="${BUNDLE}/tokenizer"
    --omnivoice-num-steps="${STEPS}"
    --reference-audio="${REF_WAV}"
    --reference-text="${REF_TXT}"
    --num-threads="${THREADS}"
    --provider="${provider}"
    --seed="${SEED}"
    --output-filename="/tmp/omnivoice_bench_${provider}_${cached_flag}_${split}_${label}.wav"
  )
  if [ "${cached_flag}" = "cached" ]; then
    if [ ! -f "${BUNDLE}/omnivoice_prefix.onnx" ] || \
       [ ! -f "${BUNDLE}/omnivoice_target.onnx" ]; then
      # No cached ONNX in this bundle; skip silently.
      return 0
    fi
    args+=(--omnivoice-prefix-model="${BUNDLE}/omnivoice_prefix.onnx"
           --omnivoice-target-model="${BUNDLE}/omnivoice_target.onnx")
  fi
  if [ "${split}" != "none" ]; then
    args+=(--split-strategy="${split}" --split-max-chars=200)
  fi

  local log
  log=$("${BIN}" "${args[@]}" "${text}" 2>&1 1>/dev/null || true)
  local elapsed audio rtf
  elapsed=$(printf '%s\n' "${log}" | awk '/Elapsed seconds/{print $3}' | tail -1)
  audio=$(printf '%s\n' "${log}"   | awk '/Audio duration/{print $3}'  | tail -1)
  rtf=$(printf '%s\n' "${log}"     | awk '/Real-time factor/{print $NF}' | tail -1)
  if [ -z "${elapsed}" ]; then
    elapsed="?"; audio="?"; rtf="?"
  fi
  printf "| %s | %s | %s | %d | %d | %d | %s | %s | %s |\n" \
    "${provider}" "${cached_flag}" "${split}" "${#text}" \
    "${STEPS}" "${THREADS}" "${elapsed}" "${audio}" "${rtf}"
}

for provider in ${PROVIDERS}; do
  for cached in uncached cached; do
    for split in none sentence; do
      # Text length dimension is only meaningful when combined with split
      # strategy; short text won't be split.
      run_one "${provider}" "${cached}" "${split}" "${TEXT_SHORT}"  short
      run_one "${provider}" "${cached}" "${split}" "${TEXT_MEDIUM}" medium
      run_one "${provider}" "${cached}" "${split}" "${TEXT_LONG}"   long
    done
  done
done
