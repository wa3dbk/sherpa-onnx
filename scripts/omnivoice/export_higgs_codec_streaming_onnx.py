#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# EXPERIMENTAL: split the Higgs-Audio-V2 codec decoder into a
# frame-streaming ONNX graph. When present in the bundle as
# `higgs_codec_decoder_streaming.onnx`, the C++ runtime uses it to
# emit PCM as soon as each MaskGIT step finishes, instead of only
# after the full utterance is decoded.
#
# The default codec decoder consumes the full [num_codebooks, T]
# code tensor and returns the full waveform. Streaming requires:
#
#   1) removing any causally-blocking layers (e.g. non-causal
#      convolutions, layer norms over the full T axis) or replacing
#      them with left-context-only variants;
#   2) exporting with a dynamic frame-count axis and an initial-state
#      input/output so the runtime can call the graph in a loop, one
#      MaskGIT chunk at a time;
#   3) verifying frame-boundary continuity by comparing streamed vs
#      offline output samples (should be bit-identical or within
#      -60 dB PSNR).
#
# This script is a *scaffold*. It documents the required graph
# surgery and provides an entry point to plug the actual conversion
# once you have model source. It does NOT run without your changes.

import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codec-model-dir", required=True,
                   help="Path to the PyTorch Higgs-Audio-V2 codec checkpoint.")
    p.add_argument("--out", required=True,
                   help="Output ONNX path (write as higgs_codec_decoder_streaming.onnx).")
    p.add_argument("--frames-per-chunk", type=int, default=8,
                   help="Number of codec frames emitted per streaming call.")
    args = p.parse_args()

    print("This is a scaffold. The graph surgery required is:", file=sys.stderr)
    print("  1) Wrap the decoder in a nn.Module with:", file=sys.stderr)
    print("     - inputs:  codes_chunk[B, K, frames_per_chunk], state_in", file=sys.stderr)
    print("     - outputs: pcm_chunk[B, frames_per_chunk*hop], state_out", file=sys.stderr)
    print("  2) Replace non-causal convs with causal + left-pad state.", file=sys.stderr)
    print("  3) torch.onnx.export with dynamic axes on T dim.", file=sys.stderr)
    print("  4) onnxruntime.InferenceSession sanity check: concatenated", file=sys.stderr)
    print("     stream output must match offline output within -60 dB.", file=sys.stderr)
    print(f"  5) Save to: {args.out}", file=sys.stderr)
    print(f"     Frames per chunk: {args.frames_per_chunk}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Once implemented, the C++ runtime will detect the file via", file=sys.stderr)
    print("OfflineTtsOmnivoiceModelConfig.streaming_codec_decoder (see", file=sys.stderr)
    print("offline-tts-omnivoice-model.cc TODO markers).", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
