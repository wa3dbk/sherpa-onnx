#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Stream OmniVoice output live to your speakers via ffplay.
#
# The trick: --split-strategy=sentence chunks the input, --output-stream=1
# writes raw float32 PCM to stdout as each chunk finishes decoding, and
# ffplay plays it as it arrives. First-audio-out latency is proportional
# to the first sentence, not the whole paragraph.
#
# Usage:
#   SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR=./sherpa-onnx-omnivoice-XXXX \
#   SHERPA_ONNX_OFFLINE_TTS_BIN=./build/bin/sherpa-onnx-offline-tts \
#     scripts/omnivoice/stream_to_ffplay.sh "Text to synthesize."
#
# Or pipe the text on stdin:
#   echo "Long paragraph..." | scripts/omnivoice/stream_to_ffplay.sh

set -euo pipefail

BUNDLE=${SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR:-}
BIN=${SHERPA_ONNX_OFFLINE_TTS_BIN:-./build/bin/sherpa-onnx-offline-tts}

if [ -z "${BUNDLE}" ]; then
  echo "set SHERPA_ONNX_OMNIVOICE_BUNDLE_DIR" >&2; exit 1
fi
command -v ffplay >/dev/null || {
  echo "ffplay not found. Install ffmpeg (brew install ffmpeg / apt install ffmpeg)." >&2
  exit 1
}

if [ "$#" -ge 1 ]; then
  TEXT="$*"
else
  TEXT="$(cat)"
fi

REF_WAV="${BUNDLE}/test_wavs/ref.wav"
REF_TXT="$(cat "${BUNDLE}/test_wavs/ref.txt")"

# stdout: raw f32le PCM at 24 kHz -> ffplay
# stderr: progress log (kept visible so you can see chunk-by-chunk progress)
"${BIN}" \
  --omnivoice-model="${BUNDLE}/omnivoice.onnx" \
  --omnivoice-codec-encoder="${BUNDLE}/higgs_codec_encoder.onnx" \
  --omnivoice-codec-decoder="${BUNDLE}/higgs_codec_decoder.onnx" \
  --omnivoice-tokenizer-dir="${BUNDLE}/tokenizer" \
  --reference-audio="${REF_WAV}" \
  --reference-text="${REF_TXT}" \
  --split-strategy=sentence \
  --split-max-chars=200 \
  --output-stream=1 \
  --num-threads=4 \
  "${TEXT}" \
  | ffplay -f f32le -ar 24000 -ch_layout mono -i - \
           -nodisp -autoexit -loglevel error
