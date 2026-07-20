"""Export the OmniVoice LM to ONNX.

The exported graph matches the interface of the shipped ``omnivoice.onnx``
(HuggingFace-style 2-D padding mask + explicit position_ids). Internally
the wrapper converts the 2-D mask into the 4-D block mask that
``OmniVoice.forward`` expects.

    inputs
      input_ids       int64  [batch, num_codebook, seq_len]
      audio_mask      bool   [batch, seq_len]
      attention_mask  int64  [batch, seq_len]      # 1 = real, 0 = pad
      position_ids    int64  [batch, seq_len]
    output
      logits          float  [batch, num_codebook, seq_len, audio_vocab_size]

The model is *non-autoregressive* (MaskGIT-style iterative unmasking). One full
forward pass per denoising step; no KV cache needed. That makes it a clean
ONNX export - no control-flow, no state.

Batch during CFG generation is ``2 * B`` (first half conditional, second
half unconditional). The exported graph is agnostic to which half is which -
the caller just concatenates and slices.

Usage:
    python -m omnivoice.scripts.export_omnivoice_onnx \\
        --model-path /path/to/omnivoice_hf_checkpoint \\
        --output-dir omnivoice-onnx/onnx \\
        [--opset 17] [--dtype float32] [--dynamo]
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from omnivoice.models.omnivoice import OmniVoice


class OmniVoiceExportWrapper(nn.Module):
    """Thin wrapper so torch.onnx.export sees a positional-arg forward.

    The real ``OmniVoice.forward`` accepts extra kwargs used only during
    training (labels, document_ids, position_ids). We hide them so the
    exported graph has exactly the three tensors the C++ runner will feed.
    """

    def __init__(self, model: OmniVoice):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,       # int64 [B, C, S]
        audio_mask: torch.Tensor,      # bool  [B, S]
        attention_mask: torch.Tensor,  # int64 [B, S]   (1 = keep, 0 = pad)
        position_ids: torch.Tensor,    # int64 [B, S]
    ) -> torch.Tensor:
        # Convert HF-style 2-D padding mask [B, S] to the 4-D block bool
        # mask [B, 1, S, S] that OmniVoice.forward wants. A token j is
        # visible from query i iff both i and j are unpadded. Because
        # OmniVoice is non-causal (MaskGIT-style bidirectional), no
        # triangular masking is applied.
        keep = attention_mask.to(torch.bool)              # [B, S]
        block_mask = keep.unsqueeze(1) & keep.unsqueeze(2)  # [B, S, S]
        block_mask = block_mask.unsqueeze(1)              # [B, 1, S, S]

        out = self.model(
            input_ids=input_ids,
            audio_mask=audio_mask,
            attention_mask=block_mask,
            position_ids=position_ids,
        )
        return out.logits  # [B, C, S, V]


def _write_meta(out_dir: Path, model: OmniVoice, sample_rate: int | None) -> None:
    cfg = model.config
    meta = {
        "model_type": "omnivoice",
        "num_audio_codebook": cfg.num_audio_codebook,
        "audio_vocab_size": cfg.audio_vocab_size,
        "audio_mask_id": cfg.audio_mask_id,
        "audio_codebook_weights": list(cfg.audio_codebook_weights),
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
        "llm_hidden_size": cfg.llm_config.hidden_size,
        "llm_num_hidden_layers": cfg.llm_config.num_hidden_layers,
        "sample_rate": sample_rate,
    }
    (out_dir / "omnivoice.meta.json").write_text(json.dumps(meta, indent=2))


def export(args: argparse.Namespace) -> None:
    torch.manual_seed(0)

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading OmniVoice from {args.model_path} (dtype={args.dtype}) ...")
    model = OmniVoice.from_pretrained(
        args.model_path,
        train=False,
        torch_dtype=dtype,
        attn_implementation="sdpa",  # flex_attention is not exportable
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    wrapper = OmniVoiceExportWrapper(model)

    B = args.example_batch  # e.g. 2 (one cond + one uncond)
    C = model.config.num_audio_codebook
    S = args.example_seq_len

    dummy_input_ids = torch.randint(
        0, model.config.audio_vocab_size, (B, C, S), dtype=torch.long, device=device
    )
    dummy_audio_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    dummy_audio_mask[:, S // 2:] = True
    dummy_attn = torch.ones(B, S, dtype=torch.long, device=device)
    dummy_pos = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1).contiguous()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "omnivoice.onnx"

    print(f"Tracing/exporting to {onnx_path} (opset={args.opset}, dynamo={args.dynamo}) ...")
    if args.dynamo:
        # Newer dynamo exporter - preferred for HF transformer graphs on
        # recent torch versions. Falls back to legacy tracer via --no-dynamo.
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_audio_mask, dummy_attn, dummy_pos),
            str(onnx_path),
            input_names=["input_ids", "audio_mask", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_shapes={
                "input_ids": {0: "batch", 2: "seq"},
                "audio_mask": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {0: "batch", 1: "seq"},
            },
            dynamo=True,
            opset_version=args.opset,
            external_data=True,
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_audio_mask, dummy_attn, dummy_pos),
            str(onnx_path),
            input_names=["input_ids", "audio_mask", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 2: "seq"},
                "audio_mask": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch", 2: "seq"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
        )

    _write_meta(out_dir, model, sample_rate=getattr(model, "sampling_rate", None))
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF checkpoint dir or hub id")
    ap.add_argument("--output-dir", default="omnivoice-onnx/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    ap.add_argument("--example-batch", type=int, default=2,
                    help="Example batch size used only for tracing; runtime is dynamic.")
    ap.add_argument("--example-seq-len", type=int, default=64,
                    help="Example seq_len used only for tracing; runtime is dynamic.")
    ap.add_argument("--dynamo", action="store_true",
                    help="Use torch.onnx.dynamo_export (recommended, torch>=2.4).")
    export(ap.parse_args())


if __name__ == "__main__":
    main()
