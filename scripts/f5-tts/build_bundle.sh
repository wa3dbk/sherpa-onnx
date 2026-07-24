#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Build a sherpa-onnx-consumable F5-TTS bundle:
#   1) download the F5-TTS PyTorch checkpoint from HF
#   2) export it to ONNX via export_f5_onnx.py
#   3) download + export the Vocos vocoder
#   4) drop tokens.txt (from the f5_tts tokenizer)
#   5) reuse espeak-ng-data from a nearby zipvoice bundle if present,
#      otherwise print instructions to fetch it
#   6) optionally fetch a reference clip
#   7) optionally quantize (DTYPE=fp16 or int8)
#
# See README.md in this directory for the env var contract.

set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

F5_TTS_CKPT=${F5_TTS_CKPT:-SWivid/F5-TTS}
F5_TTS_VARIANT=${F5_TTS_VARIANT:-F5TTS_Base}
VOCOS_CKPT=${VOCOS_CKPT:-charactr/vocos-mel-24khz}
OUT_DIR=${OUT_DIR:-./sherpa-onnx-f5-tts-multi}
REFERENCE_HF_PATH=${REFERENCE_HF_PATH:-}
REFERENCE_HF_TEXT_PATH=${REFERENCE_HF_TEXT_PATH:-}
DTYPE=${DTYPE:-fp32}

python3 -c "import f5_tts, vocos, torch, onnx" 2>/dev/null || {
  echo "Missing deps. Install with:" >&2
  echo "  pip install torch onnx onnxruntime f5-tts vocos huggingface_hub" >&2
  exit 1
}

mkdir -p "${OUT_DIR}/test_wavs" "${OUT_DIR}/espeak-ng-data"

# ---- 1. fetch F5-TTS checkpoint -------------------------------------
if [ -f "${F5_TTS_CKPT}" ]; then
  CKPT_LOCAL="${F5_TTS_CKPT}"
else
  echo ">>> Downloading F5-TTS checkpoint from HF (${F5_TTS_CKPT})"
  CKPT_LOCAL=$(python3 - <<PY
from huggingface_hub import hf_hub_download
fname = "${F5_TTS_VARIANT}/model_1250000.safetensors"
print(hf_hub_download(repo_id="${F5_TTS_CKPT}", filename=fname))
PY
)
fi

# ---- 2. export F5 DiT -----------------------------------------------
echo ">>> Exporting F5 DiT to ONNX"
python3 "${HERE}/export_f5_onnx.py" \
  --ckpt "${CKPT_LOCAL}" \
  --variant "${F5_TTS_VARIANT}" \
  --out "${OUT_DIR}/f5_transformer.onnx"

# ---- 3. export Vocos vocoder ----------------------------------------
echo ">>> Exporting Vocos vocoder to ONNX"
python3 "${HERE}/export_vocos_onnx.py" \
  --repo "${VOCOS_CKPT}" \
  --out "${OUT_DIR}/vocos.onnx"

# ---- 4. tokens.txt ---------------------------------------------------
echo ">>> Writing tokens.txt from the F5-TTS text tokenizer"
python3 - <<PY
import json, os
try:
    from f5_tts.model.utils import get_tokenizer
except Exception:
    # Fallback: F5-TTS ships tokens as a plain vocab.txt in its data dir.
    from importlib.resources import files
    text = (files("f5_tts") / "infer/examples/vocab.txt").read_text()
    open("${OUT_DIR}/tokens.txt", "w").write(text)
    raise SystemExit(0)

vocab, _ = get_tokenizer("Emilia_ZH_EN", "pinyin")
with open("${OUT_DIR}/tokens.txt", "w", encoding="utf-8") as f:
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        f.write(f"{tok}\t{idx}\n")
PY

# ---- 5. espeak-ng-data -----------------------------------------------
if [ -z "$(ls -A "${OUT_DIR}/espeak-ng-data" 2>/dev/null)" ]; then
  # Look for a nearby zipvoice bundle to copy from.
  CANDIDATE=$(find "${HERE}/../.." -maxdepth 3 -type d -name espeak-ng-data 2>/dev/null | head -n1 || true)
  if [ -n "${CANDIDATE}" ] && [ -f "${CANDIDATE}/phondata" ]; then
    echo ">>> Reusing espeak-ng-data from ${CANDIDATE}"
    cp -R "${CANDIDATE}/." "${OUT_DIR}/espeak-ng-data/"
  else
    cat <<'EOF' >&2

WARN: no espeak-ng-data found nearby. Fetch it once with:

  git clone --depth 1 https://github.com/rhasspy/piper-phonemize.git
  cp -R piper-phonemize/etc/espeak-ng-data OUT_DIR_HERE/espeak-ng-data/

Or copy from any zipvoice/piper/matcha bundle that ships one.

EOF
  fi
fi

# ---- 6. reference clip -----------------------------------------------
if [ ! -f "${OUT_DIR}/test_wavs/ref.wav" ]; then
  if [ -n "${REFERENCE_HF_PATH}" ]; then
    echo ">>> Fetching reference clip from HF"
    python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="${F5_TTS_CKPT}", filename="${REFERENCE_HF_PATH}")
shutil.copy(p, "${OUT_DIR}/test_wavs/ref.wav")
if "${REFERENCE_HF_TEXT_PATH}":
    p = hf_hub_download(repo_id="${F5_TTS_CKPT}",
                        filename="${REFERENCE_HF_TEXT_PATH}")
    shutil.copy(p, "${OUT_DIR}/test_wavs/ref.txt")
PY
  else
    cat <<'EOF' >&2

NOTE: no reference clip fetched. Drop a 3-10 s 24 kHz mono wav at
      ${OUT_DIR}/test_wavs/ref.wav and its transcript at
      ${OUT_DIR}/test_wavs/ref.txt before running the synthesizer.

EOF
  fi
fi

# ---- 7. optional quantization ---------------------------------------
if [ "${DTYPE}" != "fp32" ]; then
  echo ">>> Quantizing DiT to ${DTYPE}"
  # Reuses the omnivoice quantization script — it's model-agnostic
  # (walks a directory of .onnx files, fp16-converts or int8-quantizes
  # each). See scripts/omnivoice/export_omnivoice_quantized.py.
  python3 "${HERE}/../omnivoice/export_omnivoice_quantized.py" \
    --in "${OUT_DIR}" \
    --out "${OUT_DIR}-${DTYPE}" \
    --dtype "${DTYPE}"
  echo ">>> Quantized bundle at ${OUT_DIR}-${DTYPE}"
fi

echo
echo "F5-TTS bundle ready at: ${OUT_DIR}"
ls -la "${OUT_DIR}"
