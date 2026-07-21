#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export OmniVoice LM as a KV-cache-aware pair of ONNX graphs.

Produces two files instead of the single ``omnivoice.onnx``:

    omnivoice_prefix.onnx
        Runs once per Generate() call on the CFG-conditional prefix
        (style + text + reference-audio codes). Emits per-layer K/V
        tensors that the target graph reuses across every MaskGIT step.

        inputs
          input_ids       int64  [1, C, S_pref]
          audio_mask      bool   [1, S_pref]
          attention_mask  int64  [1, S_pref]           # 1 = real, 0 = pad
          position_ids    int64  [1, S_pref]
        outputs
          past_k_{i}      float  [1, num_kv_heads, S_pref, head_dim]
          past_v_{i}      float  [1, num_kv_heads, S_pref, head_dim]
          for i in [0, num_layers)

    omnivoice_target.onnx
        Runs once per MaskGIT step on the target-region tokens only,
        using the cached prefix K/V. Returns logits for the target
        positions.

        inputs
          input_ids       int64  [1, C, S_tgt]
          audio_mask      bool   [1, S_tgt]
          attention_mask  int64  [1, S_pref + S_tgt]   # 1 = real, 0 = pad
          position_ids    int64  [1, S_tgt]            # absolute (>= S_pref)
          past_k_{i}      float  [1, num_kv_heads, S_pref, head_dim]
          past_v_{i}      float  [1, num_kv_heads, S_pref, head_dim]
        output
          logits          float  [1, C, S_tgt, audio_vocab_size]

Both graphs are batch-1. The C++ runner:
  - runs the prefix graph once for the CFG-conditional branch,
  - per MaskGIT step, runs the target graph for the conditional branch
    (with cached past K/V) and continues to use the original
    single-batch ``omnivoice.onnx`` for the CFG-unconditional branch
    (which has no prefix by construction).

Compute per MaskGIT step drops from
    2 * (S_pref + S_tgt) transformer positions
to
    S_tgt (cond, cached) + S_tgt (uncond, non-cached) = 2 * S_tgt
i.e. a factor of (S_pref + S_tgt) / S_tgt speedup - typically ~1.5-2x.

Usage:
    python -m scripts.omnivoice.export_omnivoice_cached_onnx \\
        --model-path /path/to/omnivoice_hf_checkpoint \\
        --output-dir omnivoice-onnx/onnx \\
        [--opset 17] [--dtype float32]
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from omnivoice.models.omnivoice import OmniVoice


def _to_legacy(pkv):
    """Normalize HF Cache -> tuple[tuple[K, V]] for ONNX export."""
    if pkv is None:
        return None
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    return pkv


class OmniVoicePrefixExportWrapper(nn.Module):
    """First-pass wrapper: encode prefix, expose per-layer K/V."""

    def __init__(self, model: OmniVoice):
        super().__init__()
        self.model = model
        self.num_layers = model.llm.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,       # int64 [1, C, S_pref]
        audio_mask: torch.Tensor,      # bool  [1, S_pref]
        attention_mask: torch.Tensor,  # int64 [1, S_pref]
        position_ids: torch.Tensor,    # int64 [1, S_pref]
    ):
        keep = attention_mask.to(torch.bool)
        block_mask = keep.unsqueeze(1) & keep.unsqueeze(2)
        block_mask = block_mask.unsqueeze(1)  # [1, 1, S_pref, S_pref]

        embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)
        out = self.model.llm(
            inputs_embeds=embeds,
            attention_mask=block_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        pkv = _to_legacy(out.past_key_values)
        # pkv is tuple of (K, V) per layer; flatten for ONNX I/O
        flat: List[torch.Tensor] = []
        for layer_kv in pkv:
            flat.append(layer_kv[0])
            flat.append(layer_kv[1])
        return tuple(flat)


class OmniVoiceTargetExportWrapper(nn.Module):
    """Per-step wrapper: run target region with cached prefix K/V."""

    def __init__(self, model: OmniVoice):
        super().__init__()
        self.model = model
        self.num_layers = model.llm.config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,       # int64 [1, C, S_tgt]
        audio_mask: torch.Tensor,      # bool  [1, S_tgt]
        attention_mask: torch.Tensor,  # int64 [1, S_pref + S_tgt]
        position_ids: torch.Tensor,    # int64 [1, S_tgt]  (absolute)
        *past: torch.Tensor,           # flat: k0, v0, k1, v1, ...
    ):
        # Reassemble past as tuple[tuple[K, V]].
        assert len(past) == 2 * self.num_layers, (
            f"expected {2 * self.num_layers} past tensors, got {len(past)}"
        )
        pkv = tuple(
            (past[2 * i], past[2 * i + 1]) for i in range(self.num_layers)
        )

        # Build 4-D non-causal block mask. Queries are the current target
        # positions; keys span past prefix + current target.
        keep = attention_mask.to(torch.bool)  # [1, S_pref + S_tgt]
        s_tgt = input_ids.size(-1)
        q_keep = keep[:, -s_tgt:]  # [1, S_tgt]
        block_mask = q_keep.unsqueeze(2) & keep.unsqueeze(1)  # [1, S_tgt, T]
        block_mask = block_mask.unsqueeze(1)  # [1, 1, S_tgt, T]

        embeds = self.model._prepare_embed_inputs(input_ids, audio_mask)
        out = self.model.llm(
            inputs_embeds=embeds,
            attention_mask=block_mask,
            position_ids=position_ids,
            past_key_values=pkv,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # [1, S_tgt, hidden]

        b, s, _ = hidden.shape
        logits_flat = self.model.audio_heads(hidden)
        logits = logits_flat.view(
            b, s, self.model.config.num_audio_codebook,
            self.model.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)  # [1, C, S_tgt, V]
        return logits


def _write_meta(out_dir: Path, model: OmniVoice, sample_rate) -> None:
    cfg = model.config
    lc = cfg.llm_config
    # num_kv_heads may live under different attribute names depending on
    # transformers version. Try the most common ones.
    num_kv_heads = getattr(lc, "num_key_value_heads",
                           getattr(lc, "num_attention_heads", None))
    head_dim = getattr(lc, "head_dim", None)
    if head_dim is None and num_kv_heads is not None:
        head_dim = lc.hidden_size // lc.num_attention_heads

    meta = {
        "model_type": "omnivoice_cached",
        "num_audio_codebook": cfg.num_audio_codebook,
        "audio_vocab_size": cfg.audio_vocab_size,
        "audio_mask_id": cfg.audio_mask_id,
        "audio_codebook_weights": list(cfg.audio_codebook_weights),
        "pad_token_id": cfg.pad_token_id,
        "eos_token_id": cfg.eos_token_id,
        "llm_hidden_size": lc.hidden_size,
        "llm_num_hidden_layers": lc.num_hidden_layers,
        "llm_num_kv_heads": num_kv_heads,
        "llm_head_dim": head_dim,
        "sample_rate": sample_rate,
    }
    (out_dir / "omnivoice_cached.meta.json").write_text(
        json.dumps(meta, indent=2))


def _kv_names(num_layers: int, prefix: str) -> List[str]:
    names: List[str] = []
    for i in range(num_layers):
        names.append(f"{prefix}_k_{i}")
        names.append(f"{prefix}_v_{i}")
    return names


def export(args: argparse.Namespace) -> None:
    torch.manual_seed(0)

    dtype_map = {"float32": torch.float32,
                 "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"Loading OmniVoice from {args.model_path} (dtype={args.dtype}) ...")
    model = OmniVoice.from_pretrained(
        args.model_path,
        train=False,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    C = model.config.num_audio_codebook
    num_layers = model.llm.config.num_hidden_layers
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- prefix graph -------------------------------------------------
    prefix_wrapper = OmniVoicePrefixExportWrapper(model)
    S_pref = args.example_prefix_len
    p_input_ids = torch.randint(
        0, model.config.audio_vocab_size, (1, C, S_pref),
        dtype=torch.long, device=device,
    )
    p_audio_mask = torch.zeros(1, S_pref, dtype=torch.bool, device=device)
    p_audio_mask[:, S_pref // 2:] = True
    p_attn = torch.ones(1, S_pref, dtype=torch.long, device=device)
    p_pos = torch.arange(
        S_pref, dtype=torch.long, device=device).unsqueeze(0)

    prefix_out_names = _kv_names(num_layers, "past")
    prefix_path = out_dir / "omnivoice_prefix.onnx"
    print(f"Exporting prefix graph -> {prefix_path}")

    prefix_dynamic = {
        "input_ids": {2: "s_pref"},
        "audio_mask": {1: "s_pref"},
        "attention_mask": {1: "s_pref"},
        "position_ids": {1: "s_pref"},
    }
    for n in prefix_out_names:
        prefix_dynamic[n] = {2: "s_pref"}

    torch.onnx.export(
        prefix_wrapper,
        (p_input_ids, p_audio_mask, p_attn, p_pos),
        str(prefix_path),
        input_names=["input_ids", "audio_mask",
                     "attention_mask", "position_ids"],
        output_names=prefix_out_names,
        dynamic_axes=prefix_dynamic,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    # -------- target graph -------------------------------------------------
    target_wrapper = OmniVoiceTargetExportWrapper(model)
    S_tgt = args.example_target_len

    # Rerun the prefix wrapper on the dummy input to get plausibly-shaped
    # past tensors for tracing the target graph.
    with torch.no_grad():
        flat_past = prefix_wrapper(p_input_ids, p_audio_mask, p_attn, p_pos)
    assert len(flat_past) == 2 * num_layers

    t_input_ids = torch.randint(
        0, model.config.audio_vocab_size, (1, C, S_tgt),
        dtype=torch.long, device=device,
    )
    t_audio_mask = torch.ones(1, S_tgt, dtype=torch.bool, device=device)
    t_attn = torch.ones(1, S_pref + S_tgt, dtype=torch.long, device=device)
    t_pos = torch.arange(
        S_pref, S_pref + S_tgt, dtype=torch.long, device=device
    ).unsqueeze(0)

    target_in_names = ["input_ids", "audio_mask",
                       "attention_mask", "position_ids"] + \
        _kv_names(num_layers, "past")
    target_path = out_dir / "omnivoice_target.onnx"
    print(f"Exporting target graph -> {target_path}")

    target_dynamic = {
        "input_ids": {2: "s_tgt"},
        "audio_mask": {1: "s_tgt"},
        "attention_mask": {1: "total"},
        "position_ids": {1: "s_tgt"},
        "logits": {2: "s_tgt"},
    }
    for i in range(num_layers):
        target_dynamic[f"past_k_{i}"] = {2: "s_pref"}
        target_dynamic[f"past_v_{i}"] = {2: "s_pref"}

    torch.onnx.export(
        target_wrapper,
        (t_input_ids, t_audio_mask, t_attn, t_pos, *flat_past),
        str(target_path),
        input_names=target_in_names,
        output_names=["logits"],
        dynamic_axes=target_dynamic,
        opset_version=args.opset,
        do_constant_folding=True,
    )

    _write_meta(out_dir, model,
                sample_rate=getattr(model, "sampling_rate", None))
    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", required=True,
                    help="HF checkpoint dir or hub id")
    ap.add_argument("--output-dir", default="omnivoice-onnx/onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dtype",
                    choices=["float32", "float16", "bfloat16"],
                    default="float32")
    ap.add_argument("--example-prefix-len", type=int, default=192,
                    help="Example S_pref for tracing; runtime is dynamic.")
    ap.add_argument("--example-target-len", type=int, default=128,
                    help="Example S_tgt for tracing; runtime is dynamic.")
    export(ap.parse_args())


if __name__ == "__main__":
    main()
