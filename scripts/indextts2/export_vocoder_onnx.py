# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export IndexTTS-2's neural vocoder (audio tokens -> waveform).

I/O:
    inputs  = { "audio_tokens": int64 (1, T) }
    outputs = { "audio": float32 (1, T * upsample_factor) }
"""

import argparse

import torch

from _indextts2_load import load_vocoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    voc = load_vocoder(args.ckpt, args.variant).eval()

    dummy = torch.zeros(1, 64, dtype=torch.int64)
    torch.onnx.export(
        voc,
        dummy,
        args.out,
        input_names=["audio_tokens"],
        output_names=["audio"],
        dynamic_axes={"audio_tokens": {1: "T"}, "audio": {1: "S"}},
        opset_version=17,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
