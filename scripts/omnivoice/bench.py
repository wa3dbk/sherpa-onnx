#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Python companion to bench.sh: benchmark OmniVoice via the Python
# bindings instead of the CLI. Useful when you want to measure the
# in-process reference-cache hit specifically (the CLI's second run
# picks it up via the disk cache; this one exercises the RAM path).
#
# Usage:
#   BUNDLE_DIR=./sherpa-onnx-omnivoice-en \
#     python3 scripts/omnivoice/bench.py --providers cpu --num-threads 4
#
# Emits a markdown table on stdout suitable for pasting into README.

import argparse
import os
import statistics
import sys
import time

import sherpa_onnx


SHORT = "Hello world."
MEDIUM = ("Zero-shot text to speech clones any voice from a five second "
          "reference clip.")
LONG = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir",
                   default=os.environ.get("BUNDLE_DIR",
                                          "./sherpa-onnx-omnivoice-en"))
    p.add_argument("--providers", default="cpu",
                   help="Comma-separated list, e.g. 'cpu,cuda'.")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    return p.parse_args()


def build_tts(bundle_dir, provider, num_threads):
    def p(name):
        return os.path.join(bundle_dir, name)

    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            omnivoice=sherpa_onnx.OfflineTtsOmnivoiceModelConfig(
                model=p("omnivoice.onnx"),
                codec_encoder=p("higgs_codec_encoder.onnx"),
                codec_decoder=p("higgs_codec_decoder.onnx"),
                tokenizer_dir=p("tokenizer"),
            ),
            num_threads=num_threads,
            provider=provider,
        ),
        max_num_sentences=1,
    ))


def rtf(tts, text, gen_cfg):
    t0 = time.time()
    out = tts.generate(text, generation_config=gen_cfg)
    dt = time.time() - t0
    dur = len(out.samples) / out.sample_rate
    return dt / dur


def main():
    args = parse_args()
    bundle = args.bundle_dir
    ref_wav = sherpa_onnx.read_wave(os.path.join(bundle, "test_wavs/ref.wav"))
    with open(os.path.join(bundle, "test_wavs/ref.txt")) as f:
        ref_txt = f.read().strip()

    print("| provider | text | cached | strategy | rtf (median of "
          f"{args.repeats}) |")
    print("|---|---|---|---|---|")

    for provider in args.providers.split(","):
        provider = provider.strip()
        tts = build_tts(bundle, provider, args.num_threads)

        for label, text in (("short", SHORT), ("medium", MEDIUM),
                            ("long", LONG)):
            for strategy in ("none", "sentence"):
                gen = sherpa_onnx.GenerationConfig(
                    reference_audio=ref_wav.samples,
                    reference_sample_rate=ref_wav.sample_rate,
                    reference_text=ref_txt,
                    extra={"split_strategy": strategy,
                           "split_max_chars": "200", "seed": "42"},
                )
                # First run: cold (encodes reference).
                cold = rtf(tts, text, gen)
                # Subsequent runs: warm (in-process cache hit).
                warm = statistics.median(
                    [rtf(tts, text, gen) for _ in range(args.repeats)])
                print(f"| {provider} | {label} | no  | {strategy} | "
                      f"{cold:.3f} |")
                print(f"| {provider} | {label} | yes | {strategy} | "
                      f"{warm:.3f} |")
                sys.stdout.flush()


if __name__ == "__main__":
    main()
