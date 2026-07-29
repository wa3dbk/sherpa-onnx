# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export IndexTTS-2's emotion encoder (emo ref wav -> emo embedding).

I/O:
    inputs  = { "audio": float32 (1, T) at 16kHz mono }
    outputs = { "embedding": float32 (1, emotion_embed_dim) }
"""

import argparse

import torch

from _indextts2_load import load_emotion_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = load_emotion_encoder(args.ckpt, args.variant).eval()

    dummy = torch.zeros(1, 16000, dtype=torch.float32)
    torch.onnx.export(
        enc,
        dummy,
        args.out,
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={"audio": {1: "T"}},
        opset_version=17,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
