"""Export the Higgs-Audio-V2 audio codec to ONNX (encode + decode).

Why this is needed
------------------
OmniVoice's LM predicts audio *tokens* (8 codebooks, 1025 vocab). Turning
those tokens back into a waveform - and turning a reference clip into tokens
for voice cloning - both go through ``eustlb/higgs-audio-v2-tokenizer``, a
separate model. The published ``omnivoice-onnx/`` bundle only ships the LM,
so a sherpa-onnx C++ backend cannot produce audio without this codec.

This script exports two ONNX graphs:

    higgs_codec_encoder.onnx
        input:  input_values  float [batch, 1, wav_samples]  (raw 24 kHz mono
                              waveform; the mel front-end lives inside the
                              codec graph, not in AutoFeatureExtractor)
        output: audio_codes   int64 [batch, num_codebook, num_frames]

    higgs_codec_decoder.onnx
        input:  audio_codes   int64 [batch, num_codebook, num_frames]
        output: audio_values  float [batch, 1, wav_samples]

Notes
-----
* Empirical: for higgs-audio-v2-tokenizer the AutoFeatureExtractor returns
  raw audio [batch, 1, T] at 24 kHz. The mel front-end is inside the codec
  itself, so the ONNX encoder input is just the raw waveform - no external
  mel reimplementation needed in C++.
* The decoder is pure and safely exportable.
* If your transformers version is old and the codec module lacks
  ``forward``-callable encode/decode, upgrade transformers or patch here.

Usage
-----
    python -m omnivoice.scripts.export_higgs_codec_onnx \\
        --tokenizer-path eustlb/higgs-audio-v2-tokenizer \\
        --output-dir omnivoice-onnx/onnx \\
        [--opset 17] [--parts encode,decode]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel


class HiggsEncoderWrapper(nn.Module):
    """Wrap ``HiggsAudioV2TokenizerModel.encode`` for ONNX export.

    Accepts the same ``input_values`` tensor that ``AutoFeatureExtractor``
    produces (mel features), i.e. the tensor OmniVoice's Python code passes
    to ``audio_tokenizer.encode(...)``.
    """

    def __init__(self, codec: HiggsAudioV2TokenizerModel):
        super().__init__()
        self.codec = codec

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.codec.encode(input_values)
        # HiggsAudioV2TokenizerModel.encode returns an object with
        # `.audio_codes` of shape [B, C, T]
        return out.audio_codes


class HiggsDecoderWrapper(nn.Module):
    """Wrap ``HiggsAudioV2TokenizerModel.decode`` for ONNX export."""

    def __init__(self, codec: HiggsAudioV2TokenizerModel):
        super().__init__()
        self.codec = codec

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        out = self.codec.decode(audio_codes)
        # Depending on transformers version this may be a tensor or a
        # ModelOutput with `.audio_values`.
        if hasattr(out, "audio_values"):
            return out.audio_values
        return out


def _write_meta(out_dir: Path, feat_extractor, codec: HiggsAudioV2TokenizerModel) -> None:
    cfg = codec.config
    meta = {
        "model_type": "higgs-audio-v2-tokenizer",
        "sampling_rate": getattr(feat_extractor, "sampling_rate", None),
        "frame_rate": getattr(cfg, "frame_rate", None),
        "hop_length": getattr(cfg, "hop_length", None),
        "num_codebook": getattr(cfg, "num_codebook", None) or getattr(cfg, "n_q", None),
        "codebook_size": getattr(cfg, "codebook_size", None),
        # Mel front-end params (needed to reimplement in C++)
        "feature_extractor": {
            k: getattr(feat_extractor, k, None)
            for k in (
                "feature_size",
                "sampling_rate",
                "hop_length",
                "chunk_length",
                "n_fft",
                "n_samples",
                "num_mel_bins",
                "padding_value",
                "return_attention_mask",
            )
        },
    }
    (out_dir / "higgs_codec.meta.json").write_text(json.dumps(meta, indent=2, default=str))


def export_encoder(codec, feat_extractor, out_dir: Path, opset: int) -> None:
    device = next(codec.parameters()).device
    wrapper = HiggsEncoderWrapper(codec).eval()

    # Build a plausible dummy mel-like tensor. The exact channel dimension is
    # tokenizer-specific; we probe the feature extractor with a 1-second
    # silent waveform to get the real shape.
    sr = feat_extractor.sampling_rate
    import numpy as np

    dummy_wav = np.zeros(sr, dtype=np.float32)  # 1 s silence
    feats = feat_extractor(
        raw_audio=dummy_wav, sampling_rate=sr, return_tensors="pt"
    )["input_values"].to(device)
    print(f"encoder dummy input_values shape: {tuple(feats.shape)}")

    path = out_dir / "higgs_codec_encoder.onnx"
    # Force legacy TorchScript tracer. The dynamo exporter chokes on
    # data-dependent shape math inside the Higgs mel/encoder pipeline
    # (int_truediv on symbolic shapes -> TraceError).
    torch.onnx.export(
        wrapper,
        (feats,),
        str(path),
        input_names=["input_values"],
        output_names=["audio_codes"],
        dynamic_axes={
            "input_values": {0: "batch", 2: "frames"},
            "audio_codes": {0: "batch", 2: "tokens"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {path}")


def export_decoder(codec, out_dir: Path, opset: int) -> None:
    device = next(codec.parameters()).device
    wrapper = HiggsDecoderWrapper(codec).eval()

    C = getattr(codec.config, "num_codebook", None) or getattr(codec.config, "n_q", 8)
    vocab = getattr(codec.config, "codebook_size", 1024)
    T = 50  # arbitrary; dynamic at runtime
    dummy = torch.randint(0, vocab, (1, C, T), dtype=torch.long, device=device)

    path = out_dir / "higgs_codec_decoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(path),
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes": {0: "batch", 2: "tokens"},
            "audio_values": {0: "batch", 2: "samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-path", default="eustlb/higgs-audio-v2-tokenizer")
    ap.add_argument("--output-dir", default="omnivoice-onnx/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--parts", default="encode,decode",
                    help="Comma-separated: encode,decode")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Higgs codec from {args.tokenizer_path} on {device} ...")
    codec = HiggsAudioV2TokenizerModel.from_pretrained(
        args.tokenizer_path, device_map=device
    ).eval()
    feat = AutoFeatureExtractor.from_pretrained(args.tokenizer_path)

    parts = {p.strip() for p in args.parts.split(",")}
    if "encode" in parts:
        export_encoder(codec, feat, out_dir, args.opset)
    if "decode" in parts:
        export_decoder(codec, out_dir, args.opset)

    _write_meta(out_dir, feat, codec)
    print("Done.")


if __name__ == "__main__":
    main()
