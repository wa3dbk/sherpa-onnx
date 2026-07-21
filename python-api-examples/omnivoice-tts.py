#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder

"""
Zero-shot voice cloning with OmniVoice via the sherpa-onnx Python API.

Bundle: build with scripts/omnivoice/build_bundle.sh (add WITH_CACHED=1 to
also produce the KV-cache fast-path graphs).

Usage:

    python3 ./python-api-examples/omnivoice-tts.py \\
        --bundle-dir /path/to/sherpa-onnx-omnivoice-YYYY-MM-DD \\
        --reference-audio /path/to/reference.wav \\
        --reference-text "Verbatim transcript of the reference clip." \\
        --text "Text to speak in the reference speaker's voice." \\
        [--cached]

Add --cached (requires WITH_CACHED=1 in build_bundle.sh) to enable the
prefix/target KV-cache fast path for a ~1.5-2x speedup.
"""

import argparse
import time
from pathlib import Path

import librosa
import sherpa_onnx
import soundfile as sf


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--reference-audio", required=True)
    ap.add_argument("--reference-text", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--cached", action="store_true",
                    help="Enable the KV-cache fast path (needs "
                         "omnivoice_prefix.onnx and omnivoice_target.onnx "
                         "in the bundle).")
    ap.add_argument("--num-threads", type=int, default=2)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--output", default="./generated-omnivoice-python.wav")
    return ap.parse_args()


def create_tts(args: argparse.Namespace) -> sherpa_onnx.OfflineTts:
    bundle = Path(args.bundle_dir)
    omni = sherpa_onnx.OfflineTtsOmnivoiceModelConfig(
        model=str(bundle / "omnivoice.onnx"),
        codec_encoder=str(bundle / "higgs_codec_encoder.onnx"),
        codec_decoder=str(bundle / "higgs_codec_decoder.onnx"),
        tokenizer_dir=str(bundle / "tokenizer"),
    )
    if args.cached:
        omni.prefix_model = str(bundle / "omnivoice_prefix.onnx")
        omni.target_model = str(bundle / "omnivoice_target.onnx")

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            omnivoice=omni,
            num_threads=args.num_threads,
            debug=False,
            provider=args.provider,
        )
    )
    if not tts_config.validate():
        raise ValueError(
            "Please read the previous error messages and re-check your config"
        )
    return sherpa_onnx.OfflineTts(tts_config)


def main() -> None:
    args = parse_args()
    if not Path(args.reference_audio).is_file():
        raise ValueError(f"Reference audio {args.reference_audio} does not exist")

    tts = create_tts(args)

    reference_audio, sample_rate = librosa.load(args.reference_audio, sr=None)

    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.reference_audio = reference_audio
    gen_config.reference_sample_rate = sample_rate
    gen_config.reference_text = args.reference_text
    # Per-request overrides go in .extra (all values must be strings).
    # gen_config.extra["language"] = "en"
    # gen_config.extra["denoise"] = "1"
    # gen_config.extra["seed"] = "42"

    start = time.time()
    audio = tts.generate(args.text, gen_config)
    elapsed = time.time() - start

    if len(audio.samples) == 0:
        print("Error in generating audio. Please read previous error messages.")
        return

    duration = len(audio.samples) / audio.sample_rate
    rtf = elapsed / duration
    sf.write(args.output, audio.samples, samplerate=audio.sample_rate,
             subtype="PCM_16")
    print(f"Saved to {args.output}")
    print(f"The text is '{args.text}'")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Audio duration in seconds: {duration:.3f}")
    print(f"RTF: {elapsed:.3f}/{duration:.3f} = {rtf:.3f}")


if __name__ == "__main__":
    main()
