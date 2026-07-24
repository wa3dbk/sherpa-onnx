#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Export the Charactr Vocos mel→waveform vocoder to ONNX for
# sherpa-onnx. F5-TTS pairs with vocos-mel-24khz.
#
# Usage:
#   pip install vocos torch onnx huggingface_hub
#   python3 export_vocos_onnx.py --out ./vocos.onnx

import argparse
import os
import sys

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="charactr/vocos-mel-24khz",
                   help="HF repo of the Vocos checkpoint.")
    p.add_argument("--out", required=True)
    p.add_argument("--mel-bins", type=int, default=100)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dummy-frames", type=int, default=256,
                   help="Placeholder T for tracing; real T is dynamic.")
    return p.parse_args()


class VocosExportWrapper(nn.Module):
    """Wraps Vocos.decode with a fixed I/O signature.

    Upstream Vocos exposes `decode(features)` where `features` is a
    mel spectrogram of shape (B, mel_bins, T). It returns a waveform
    of shape (B, T * hop). We keep that signature and expose only
    the batch=1 path (sherpa-onnx does per-utterance inference).
    """

    def __init__(self, vocos):
        super().__init__()
        self.vocos = vocos

    def forward(self, mel):
        # mel: (1, 100, T)
        # audio: (1, T * hop)  — vocos-mel-24khz has hop=256
        return self.vocos.decode(mel)


def main():
    args = parse_args()

    try:
        from vocos import Vocos
    except ImportError:
        print("Install vocos first: pip install vocos", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Vocos from {args.repo}", file=sys.stderr)
    vocos = Vocos.from_pretrained(args.repo)
    vocos.eval()

    wrapper = VocosExportWrapper(vocos)
    mel = torch.randn(1, args.mel_bins, args.dummy_frames)

    print(f"Exporting to {args.out} (opset {args.opset})", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        (mel,),
        args.out,
        input_names=["mel"],
        output_names=["audio"],
        opset_version=args.opset,
        dynamic_axes={
            "mel":   {2: "T"},
            "audio": {1: "audio_len"},
        },
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
