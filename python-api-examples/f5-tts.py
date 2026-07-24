#!/usr/bin/env python3
#
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder

"""
Zero-shot voice cloning with F5-TTS via sherpa-onnx.

Usage:

  # Build a bundle first (see scripts/f5-tts/README.md), e.g.:
  #   OUT_DIR=./sherpa-onnx-f5-tts-base-24khz bash scripts/f5-tts/build_bundle.sh
  #
  # Then run:
  python3 ./python-api-examples/f5-tts.py \\
      --bundle ./sherpa-onnx-f5-tts-base-24khz \\
      --reference ./sherpa-onnx-f5-tts-base-24khz/test_wavs/ref.wav \\
      --reference-text "This is the reference text spoken in the clip." \\
      --text "Hello world, this is F5-TTS running under sherpa-onnx."
"""

import argparse
import time
from pathlib import Path

import librosa
import sherpa_onnx
import soundfile as sf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True,
                   help="Path to the F5-TTS bundle directory")
    p.add_argument("--reference", required=True,
                   help="Path to the reference audio clip (wav)")
    p.add_argument("--reference-text", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--sway-coef", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--output", default="./generated-f5-tts.wav")
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def create_tts(args):
    b = Path(args.bundle)
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            f5=sherpa_onnx.OfflineTtsF5ModelConfig(
                transformer=str(b / "transformer.onnx"),
                vocoder=str(b / "vocos.onnx"),
                tokens=str(b / "tokens.txt"),
                data_dir=str(b / "espeak-ng-data"),
                lexicon="",
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                sway_coef=args.sway_coef,
                target_rms=0.1,
                seed=args.seed,
            ),
            debug=args.debug,
            num_threads=args.num_threads,
            provider="cpu",
        )
    )
    if not tts_config.validate():
        raise ValueError("Config invalid — see error messages above")
    return sherpa_onnx.OfflineTts(tts_config)


def main():
    args = parse_args()
    ref_path = Path(args.reference)
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)

    tts = create_tts(args)
    ref_audio, sr = librosa.load(str(ref_path), sr=None)

    gen = sherpa_onnx.GenerationConfig()
    gen.reference_audio = ref_audio
    gen.reference_sample_rate = sr
    gen.reference_text = args.reference_text
    gen.num_steps = args.num_steps

    t0 = time.time()
    audio = tts.generate(args.text, gen)
    t1 = time.time()

    if not len(audio.samples):
        print("No audio generated. Check errors above.")
        return

    sf.write(args.output, audio.samples, samplerate=audio.sample_rate,
             subtype="PCM_16")
    dur = len(audio.samples) / audio.sample_rate
    print(f"Wrote {args.output}  ({dur:.2f}s audio)")
    print(f"Elapsed {t1 - t0:.2f}s  RTF {(t1 - t0) / dur:.3f}")


if __name__ == "__main__":
    main()
