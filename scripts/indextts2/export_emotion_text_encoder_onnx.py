# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export IndexTTS-2's emotion-text encoder
(natural-language description -> emotion embedding).

I/O:
    inputs  = { "input_ids": int64 (1, T),
                "attention_mask": int64 (1, T) }
    outputs = { "embedding": float32 (1, emotion_embed_dim) }
"""

import argparse

import torch

from _indextts2_load import load_emotion_text_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = load_emotion_text_encoder(args.ckpt, args.variant).eval()

    dummy_ids = torch.zeros(1, 16, dtype=torch.int64)
    dummy_mask = torch.ones(1, 16, dtype=torch.int64)

    torch.onnx.export(
        enc,
        (dummy_ids, dummy_mask),
        args.out,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {1: "T"},
            "attention_mask": {1: "T"},
        },
        opset_version=17,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
