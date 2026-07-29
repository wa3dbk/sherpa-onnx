#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Minimal IndexTTS-2 example — voice cloning with optional emo audio/text.

Usage:
    python3 indextts2.py \
        --bundle-dir ./sherpa-onnx-indextts2-base \
        --emo-text "sad and tired" \
        "Text to synthesize."
"""

import argparse
import wave
from pathlib import Path

import sherpa_onnx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--emo-audio", default="")
    ap.add_argument("--emo-text", default="")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("text")
    return ap.parse_args()


def main():
    args = parse_args()
    b = Path(args.bundle_dir)

    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            indextts2=sherpa_onnx.OfflineTtsIndexTts2ModelConfig(
                lm=str(b / "lm.onnx"),
                voice_encoder=str(b / "voice_encoder.onnx"),
                emotion_encoder=str(b / "emotion_encoder.onnx"),
                emotion_text_encoder=str(b / "emotion_text_encoder.onnx"),
                vocoder=str(b / "vocoder.onnx"),
                tokens=str(b / "tokens.txt"),
                merges=str(b / "merges.txt"),
                pinyin_table=str(b / "pinyin_table.txt"),
            ),
        )
    )
    tts = sherpa_onnx.OfflineTts(cfg)

    with wave.open(str(b / "test_wavs" / "voice.wav"), "rb") as w:
        assert w.getnchannels() == 1
        ref_sr = w.getframerate()
        ref = list(
            (int.from_bytes(w.readframes(w.getnframes()), "little", signed=True))
            for _ in range(0)
        )
    # Prefer sherpa-onnx's own wav reader:
    ref, ref_sr = sherpa_onnx.read_wave(str(b / "test_wavs" / "voice.wav"))

    emo, emo_sr = [], 0
    if args.emo_audio:
        emo, emo_sr = sherpa_onnx.read_wave(args.emo_audio)

    gcfg = sherpa_onnx.GenerationConfig(
        reference_audio=ref,
        reference_sample_rate=ref_sr,
        emotion_audio=emo,
        emotion_audio_sample_rate=emo_sr,
        emotion_text=args.emo_text,
    )

    audio = tts.generate(args.text, gcfg)
    sherpa_onnx.write_wave(args.out, audio.samples, audio.sample_rate)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
