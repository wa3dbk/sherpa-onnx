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
# Set WITH_CACHED=1 to also export the KV-cache-aware prefix/target ONNX pair
# used by the ~1.5-2x-faster --omnivoice-prefix-model/--omnivoice-target-model
# path. Off by default because the extra graphs roughly double bundle size.
WITH_CACHED=${WITH_CACHED:-0}
# Sample reference clip shipped inside the bundle under test_wavs/. Users can
# override with their own recording (3-10 s, mono, 16 kHz+). If REFERENCE_WAV
# is unset, we try to fetch a demo clip from the OmniVoice HF repo.
REFERENCE_WAV=${REFERENCE_WAV:-}
REFERENCE_TEXT=${REFERENCE_TEXT:-}
REFERENCE_HF_PATH=${REFERENCE_HF_PATH:-examples/reference.wav}
REFERENCE_HF_TEXT_PATH=${REFERENCE_HF_TEXT_PATH:-examples/reference.txt}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building bundle in ${OUT}/"
mkdir -p "${OUT}" "${OUT}/tokenizer" "${OUT}/test_wavs"

echo "==> Exporting OmniVoice LM to ONNX"
python3 "${SCRIPT_DIR}/export_omnivoice_onnx.py" \
    --hf-repo "${HF_REPO}" \
    --output "${OUT}/omnivoice.onnx" \
    --device "${DEVICE}"

if [ "${WITH_CACHED}" != "0" ]; then
  echo "==> Exporting cached OmniVoice LM (prefix + target)"
  python3 "${SCRIPT_DIR}/export_omnivoice_cached_onnx.py" \
      --model-path "${HF_REPO}" \
      --output-dir "${OUT}"
fi

echo "==> Exporting Higgs-Audio-V2 codec (encoder + decoder) to ONNX"
python3 "${SCRIPT_DIR}/export_higgs_codec_onnx.py" \
    --tokenizer-path "${HIGGS_REPO}" \
    --output-dir "${OUT}" \
    --device "${DEVICE}"

echo "==> Exporting Qwen3 tokenizer (slow format)"
python3 "${SCRIPT_DIR}/export_tokenizer.py" \
    --hf-repo "${HF_REPO}" \
    --out-dir "${OUT}/tokenizer"

echo "==> Populating test_wavs/"
if [ -n "${REFERENCE_WAV}" ]; then
  cp "${REFERENCE_WAV}" "${OUT}/test_wavs/ref.wav"
  if [ -n "${REFERENCE_TEXT}" ]; then
    printf '%s\n' "${REFERENCE_TEXT}" > "${OUT}/test_wavs/ref.txt"
  else
    echo "  (REFERENCE_TEXT not set; leaving ref.txt for the user to fill in)"
    : > "${OUT}/test_wavs/ref.txt"
  fi
else
  echo "  Trying to fetch a demo clip from HF repo ${HF_REPO}"
  if python3 -c "
import sys
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(repo_id='${HF_REPO}', filename='${REFERENCE_HF_PATH}')
    import shutil
    shutil.copyfile(p, '${OUT}/test_wavs/ref.wav')
    try:
        t = hf_hub_download(repo_id='${HF_REPO}', filename='${REFERENCE_HF_TEXT_PATH}')
        shutil.copyfile(t, '${OUT}/test_wavs/ref.txt')
    except Exception as e:
        print(f'  (no ref transcript on HF: {e})', file=sys.stderr)
        open('${OUT}/test_wavs/ref.txt', 'w').close()
except Exception as e:
    print(f'  HF fetch failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    echo "  Fetched ref.wav from HF"
  else
    echo "  WARNING: no reference clip in bundle. Drop your own into"
    echo "    ${OUT}/test_wavs/ref.wav   (3-10 s mono, 16 kHz+)"
    echo "  and its verbatim transcript into"
    echo "    ${OUT}/test_wavs/ref.txt"
    : > "${OUT}/test_wavs/ref.txt"
  fi
fi

cat > "${OUT}/README.md" <<EOF
# sherpa-onnx OmniVoice bundle (${DATE})

Voice-cloning TTS. Qwen3-0.6B LM + Higgs-Audio-V2 codec, MaskGIT-style
non-autoregressive decoding. Output: 24 kHz mono.

## Files

- \`omnivoice.onnx\` + \`omnivoice.onnx_data\` — language model
- \`higgs_codec_encoder.onnx\` — waveform -> 8-codebook audio tokens
- \`higgs_codec_decoder.onnx\` — audio tokens -> 24 kHz waveform
- \`tokenizer/\` — Qwen3 BPE + OmniVoice special tokens
- \`test_wavs/ref.wav\` + \`test_wavs/ref.txt\` — sample reference clip
  (3-10 s of clean speech, mono) and its verbatim transcript. Replace with
  your own recording to clone a different voice.
- (optional, when built with \`WITH_CACHED=1\`) \`omnivoice_prefix.onnx\` +
  \`omnivoice_target.onnx\` — KV-cache-aware LM pair for ~1.5-2x faster
  inference; wire in via \`--omnivoice-prefix-model\` and
  \`--omnivoice-target-model\`

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
