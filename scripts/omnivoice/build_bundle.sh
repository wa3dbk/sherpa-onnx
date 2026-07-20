#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Build a ready-to-use OmniVoice bundle for sherpa-onnx.
#
# Requires: a Python env with the OmniVoice training repo installed
# (https://github.com/k2-fsa/OmniVoice), plus `onnx` and `onnxruntime`.
#
# Produces: sherpa-onnx-omnivoice-<DATE>/  and  sherpa-onnx-omnivoice-<DATE>.tar.bz2

set -euo pipefail

HF_REPO=${HF_REPO:-k2/OmniVoice}
HIGGS_REPO=${HIGGS_REPO:-eustlb/higgs-audio-v2-tokenizer}
DATE=${DATE:-$(date +%Y-%m-%d)}
OUT=${OUT:-sherpa-onnx-omnivoice-${DATE}}
DEVICE=${DEVICE:-cpu}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building bundle in ${OUT}/"
mkdir -p "${OUT}" "${OUT}/tokenizer" "${OUT}/test_wavs"

echo "==> Exporting OmniVoice LM to ONNX"
python3 "${SCRIPT_DIR}/export_omnivoice_onnx.py" \
    --hf-repo "${HF_REPO}" \
    --output "${OUT}/omnivoice.onnx" \
    --device "${DEVICE}"

echo "==> Exporting Higgs-Audio-V2 codec (encoder + decoder) to ONNX"
python3 "${SCRIPT_DIR}/export_higgs_codec_onnx.py" \
    --tokenizer-path "${HIGGS_REPO}" \
    --output-dir "${OUT}" \
    --device "${DEVICE}"

echo "==> Exporting Qwen3 tokenizer (slow format)"
python3 "${SCRIPT_DIR}/export_tokenizer.py" \
    --hf-repo "${HF_REPO}" \
    --out-dir "${OUT}/tokenizer"

cat > "${OUT}/README.md" <<EOF
# sherpa-onnx OmniVoice bundle (${DATE})

Voice-cloning TTS. Qwen3-0.6B LM + Higgs-Audio-V2 codec, MaskGIT-style
non-autoregressive decoding. Output: 24 kHz mono.

## Files

- \`omnivoice.onnx\` + \`omnivoice.onnx_data\` — language model
- \`higgs_codec_encoder.onnx\` — waveform -> 8-codebook audio tokens
- \`higgs_codec_decoder.onnx\` — audio tokens -> 24 kHz waveform
- \`tokenizer/\` — Qwen3 BPE + OmniVoice special tokens
- \`test_wavs/\` — sample reference clip + transcript

## Usage

\`\`\`bash
./bin/sherpa-onnx-offline-tts \\
  --omnivoice-model=./omnivoice.onnx \\
  --omnivoice-codec-encoder=./higgs_codec_encoder.onnx \\
  --omnivoice-codec-decoder=./higgs_codec_decoder.onnx \\
  --omnivoice-tokenizer-dir=./tokenizer \\
  --reference-audio=./test_wavs/ref.wav \\
  --reference-text="\$(cat ./test_wavs/ref.txt)" \\
  --output-filename=./out.wav \\
  "Text to synthesize in the reference speaker's voice."
\`\`\`
EOF

echo "==> Bundle contents"
find "${OUT}" -maxdepth 2 -type f -printf "  %p (%s bytes)\n"

echo "==> Packing tarball"
tar cjf "${OUT}.tar.bz2" "${OUT}"

echo "==> Done: ${OUT}.tar.bz2"
