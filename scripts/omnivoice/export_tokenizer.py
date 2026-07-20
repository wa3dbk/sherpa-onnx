#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
"""Export the OmniVoice Qwen3 tokenizer in slow (vocab.json + merges.txt) format.

The sherpa-onnx C++ tokenizer (`QwenAsrTokenizer`) reads:
    vocab.json
    merges.txt
    tokenizer_config.json
    added_tokens.json          (contains OmniVoice special tokens)
    special_tokens_map.json

Qwen3 HF checkpoints ship only the fast `tokenizer.json`. `save_pretrained(...,
legacy_format=True)` regenerates the slow files.

Usage:
    python3 export_tokenizer.py \\
        --hf-repo k2/OmniVoice \\
        --out-dir ./tokenizer
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf-repo",
        default="k2/OmniVoice",
        help="HF repo id or local path with the original OmniVoice checkpoint.",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write vocab.json + merges.txt + configs.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.hf_repo)
    tok.save_pretrained(str(out), legacy_format=True)

    required = ["vocab.json", "merges.txt", "tokenizer_config.json"]
    missing = [f for f in required if not (out / f).exists()]
    if missing:
        raise SystemExit(f"missing after export: {missing}")

    print(f"wrote tokenizer to {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
