# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export IndexTTS-2's voice encoder (reference wav -> speaker embedding).

I/O:
    inputs  = { "audio": float32 (1, T) at 16kHz mono }
    outputs = { "embedding": float32 (1, speaker_embed_dim) }

The 16 kHz sample rate is a property of the encoder; the C++ side
resamples via LinearResample before calling.
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from _indextts2_load import load_voice_encoder  # sibling loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = load_voice_encoder(args.ckpt, args.variant).eval()

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
