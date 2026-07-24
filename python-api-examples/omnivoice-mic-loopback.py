#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Microphone-in -> OmniVoice TTS-out loopback demo.
#
# Records ~5 s from the default input device, uses that clip as the
# zero-shot reference, then synthesizes a target sentence in the
# captured voice and plays it back through the default output device.
#
# Requires:
#   pip install sherpa-onnx sounddevice numpy
#
# Usage:
#   BUNDLE_DIR=./sherpa-onnx-omnivoice-en \
#     python3 omnivoice-mic-loopback.py "Hello, this is my cloned voice."
#
# The reference *transcript* is what you actually said during the 5 s
# window. Pass it via --ref-text; otherwise the script uses a generic
# placeholder, which usually still works but degrades quality.

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd

import sherpa_onnx


REF_SECONDS = 5.0
REF_SR = 16000  # OmniVoice resamples internally to 24 kHz
OUT_SR = 24000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?",
                   default="Hello world, this is a test of voice cloning.")
    p.add_argument("--ref-text", default="This is my voice sample for cloning.",
                   help="Transcript of what you say during the mic capture.")
    p.add_argument("--bundle-dir",
                   default=os.environ.get("BUNDLE_DIR",
                                          "./sherpa-onnx-omnivoice-en"))
    p.add_argument("--split-strategy", default="none",
                   choices=["none", "sentence", "chars"])
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def record_reference():
    print(f"Recording {REF_SECONDS:.0f} s of reference audio at {REF_SR} Hz.")
    print("Speak now...")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    print("  GO")
    frames = int(REF_SECONDS * REF_SR)
    buf = sd.rec(frames, samplerate=REF_SR, channels=1, dtype="float32",
                 blocking=True)
    print("Done recording.")
    return buf.reshape(-1).astype(np.float32)


def build_tts(bundle_dir):
    def p(name):
        return os.path.join(bundle_dir, name)

    for required in ("omnivoice.onnx", "higgs_codec_encoder.onnx",
                     "higgs_codec_decoder.onnx", "tokenizer"):
        if not os.path.exists(p(required)):
            print(f"Missing {p(required)}", file=sys.stderr)
            sys.exit(1)

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            omnivoice=sherpa_onnx.OfflineTtsOmnivoiceModelConfig(
                model=p("omnivoice.onnx"),
                codec_encoder=p("higgs_codec_encoder.onnx"),
                codec_decoder=p("higgs_codec_decoder.onnx"),
                tokenizer_dir=p("tokenizer"),
            ),
            num_threads=2,
            provider="cpu",
        ),
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(config)


def main():
    args = parse_args()
    tts = build_tts(args.bundle_dir)
    ref = record_reference()

    extras = {"split_strategy": args.split_strategy, "split_max_chars": "200"}
    if args.seed is not None:
        extras["seed"] = str(args.seed)

    gen_cfg = sherpa_onnx.GenerationConfig(
        reference_audio=ref,
        reference_sample_rate=REF_SR,
        reference_text=args.ref_text,
        extra=extras,
    )

    t0 = time.time()
    audio = tts.generate(args.text, generation_config=gen_cfg)
    dt = time.time() - t0
    dur = len(audio.samples) / audio.sample_rate
    print(f"Synthesized {dur:.2f} s in {dt:.2f} s (RTF={dt/dur:.2f}).")

    print("Playing back...")
    sd.play(np.asarray(audio.samples, dtype=np.float32),
            samplerate=audio.sample_rate, blocking=True)


if __name__ == "__main__":
    main()
