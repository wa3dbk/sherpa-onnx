#!/usr/bin/env bash
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Build a sherpa-onnx-consumable IndexTTS-2 bundle.
# See README.md in this directory for the env-var contract.

set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

INDEXTTS2_CKPT=${INDEXTTS2_CKPT:-IndexTeam/IndexTTS-2}
INDEXTTS2_VARIANT=${INDEXTTS2_VARIANT:-base}
OUT_DIR=${OUT_DIR:-./sherpa-onnx-indextts2-base}
REFERENCE_HF_PATH=${REFERENCE_HF_PATH:-}
REFERENCE_HF_TEXT_PATH=${REFERENCE_HF_TEXT_PATH:-}
EMO_REF_HF_PATH=${EMO_REF_HF_PATH:-}
DTYPE=${DTYPE:-fp32}

python3 -c "import torch, onnx, pypinyin, sentencepiece" 2>/dev/null || {
  echo "Missing deps. Install with:" >&2
  echo "  pip install torch onnx onnxruntime pypinyin sentencepiece \\" >&2
  echo "              safetensors transformers huggingface_hub" >&2
  exit 1
}

mkdir -p "${OUT_DIR}/test_wavs"

echo ">>> Exporting LM (prefix + step)"
python3 "${HERE}/export_lm_onnx.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/lm.onnx"

echo ">>> Exporting voice encoder"
python3 "${HERE}/export_voice_encoder_onnx.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/voice_encoder.onnx"

echo ">>> Exporting emotion encoder"
python3 "${HERE}/export_emotion_encoder_onnx.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/emotion_encoder.onnx"

echo ">>> Exporting emotion text encoder"
python3 "${HERE}/export_emotion_text_encoder_onnx.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/emotion_text_encoder.onnx"

echo ">>> Exporting vocoder"
python3 "${HERE}/export_vocoder_onnx.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/vocoder.onnx"

echo ">>> Dumping tokens.txt + merges.txt"
python3 "${HERE}/dump_tokenizer.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out-tokens "${OUT_DIR}/tokens.txt" \
  --out-merges "${OUT_DIR}/merges.txt"

echo ">>> Dumping pinyin table (CJK char -> pinyin, from pypinyin)"
python3 "${HERE}/dump_pinyin_table.py" \
  --out "${OUT_DIR}/pinyin_table.txt"

echo ">>> Writing meta.json"
python3 "${HERE}/write_meta_json.py" \
  --ckpt "${INDEXTTS2_CKPT}" \
  --variant "${INDEXTTS2_VARIANT}" \
  --out "${OUT_DIR}/meta.json"

# Optional reference clips
if [ -n "${REFERENCE_HF_PATH}" ] && [ ! -f "${OUT_DIR}/test_wavs/voice.wav" ]; then
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="${INDEXTTS2_CKPT}", filename="${REFERENCE_HF_PATH}")
shutil.copy(p, "${OUT_DIR}/test_wavs/voice.wav")
if "${REFERENCE_HF_TEXT_PATH}":
    p = hf_hub_download(repo_id="${INDEXTTS2_CKPT}",
                        filename="${REFERENCE_HF_TEXT_PATH}")
    shutil.copy(p, "${OUT_DIR}/test_wavs/voice.txt")
PY
fi

if [ -n "${EMO_REF_HF_PATH}" ] && [ ! -f "${OUT_DIR}/test_wavs/emo_happy.wav" ]; then
  python3 - <<PY
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="${INDEXTTS2_CKPT}", filename="${EMO_REF_HF_PATH}")
shutil.copy(p, "${OUT_DIR}/test_wavs/emo_happy.wav")
PY
fi

if [ "${DTYPE}" != "fp32" ]; then
  echo ">>> Quantizing to ${DTYPE}"
  python3 "${HERE}/../omnivoice/export_omnivoice_quantized.py" \
    --in "${OUT_DIR}" --out "${OUT_DIR}-${DTYPE}" --dtype "${DTYPE}"
fi

echo
echo "IndexTTS-2 bundle ready at: ${OUT_DIR}"
ls -la "${OUT_DIR}"
