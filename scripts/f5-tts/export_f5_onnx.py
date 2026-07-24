#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
#
# Export the F5-TTS DiT (Diffusion Transformer) to ONNX for sherpa-onnx.
#
# The exported graph performs ONE flow-matching Euler step: given a
# noisy mel x_t at time t, the reference mel `cond`, and concatenated
# text ids, it returns the predicted velocity. The C++ runtime runs
# the ODE loop (typically ~32 steps) around this graph.
#
# The F5-TTS repo (SWivid/F5-TTS) is the source of truth for the
# PyTorch model. This script imports from `f5_tts` if installed,
# otherwise from a local checkout via --repo-path.
#
# Usage:
#   pip install f5-tts torch onnx
#   python3 export_f5_onnx.py \
#     --ckpt /path/to/model_1250000.safetensors \
#     --out ./f5_transformer.onnx \
#     --variant F5TTS_Base

import argparse
import os
import sys

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to F5-TTS .safetensors or .pt checkpoint. "
                        "Use `huggingface-cli download SWivid/F5-TTS "
                        "F5TTS_Base/model_1250000.safetensors` to fetch.")
    p.add_argument("--out", required=True, help="Output .onnx path.")
    p.add_argument("--variant", default="F5TTS_Base",
                   choices=["F5TTS_Base", "F5TTS_v1_Base"],
                   help="Which F5-TTS architecture variant.")
    p.add_argument("--mel-dim", type=int, default=100)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--dummy-seq-len", type=int, default=512,
                   help="Placeholder T for tracing; real T is dynamic.")
    return p.parse_args()


def load_f5_model(ckpt_path, variant):
    """Load the F5-TTS DiT from a checkpoint.

    We import from the `f5_tts` package. If the layout of the upstream
    package changes, this is the one place to update.
    """
    try:
        from f5_tts.model import DiT
    except ImportError:
        print("Install f5-tts first: pip install f5-tts", file=sys.stderr)
        sys.exit(1)

    if variant == "F5TTS_Base":
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2,
                         text_dim=512, conv_layers=4)
    elif variant == "F5TTS_v1_Base":
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2,
                         text_dim=512, conv_layers=4, pe_attn_head=None)
    else:
        raise ValueError(variant)

    model = DiT(**model_cfg, mel_dim=100, text_num_embeds=2545)

    # Load weights. F5-TTS ships checkpoints as EMA-averaged state
    # dicts under key "ema_model_state_dict" — the DiT weights are
    # keyed under "ema_model.transformer.*".
    if ckpt_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu")

    if "ema_model_state_dict" in state:
        state = state["ema_model_state_dict"]

    dit_state = {}
    prefix = "ema_model.transformer."
    for k, v in state.items():
        if k.startswith(prefix):
            dit_state[k[len(prefix):]] = v
        elif k.startswith("transformer."):
            dit_state[k[len("transformer."):]] = v

    missing, unexpected = model.load_state_dict(dit_state, strict=False)
    if unexpected:
        print(f"WARN: unexpected keys: {unexpected[:5]} ...", file=sys.stderr)
    if missing:
        print(f"WARN: missing keys: {missing[:5]} ...", file=sys.stderr)

    model.eval()
    return model


class F5ExportWrapper(nn.Module):
    """Wrap DiT.forward with the exact signature sherpa-onnx expects.

    The upstream DiT.forward takes many kwargs and dispatches to CFM
    at training time. For inference we only need the noisy-mel-to-
    velocity path. This wrapper also folds classifier-free guidance
    into a single graph call by running the batched cond/uncond
    forward and combining internally.
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, cond, text_ids, t, guidance_scale):
        # x:     (1, T, 100)  noisy mel
        # cond:  (1, T, 100)  reference mel (zero-padded to T)
        # text_ids: (1, T)    text token ids (reference+target concatenated,
        #                     -1 for positions with no text)
        # t:     (1,)         flow-matching timestep in [0, 1]
        # guidance_scale: (1,)

        # Batch of 2 for classifier-free guidance: [cond, uncond]
        x_in = torch.cat([x, x], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        text_in = torch.cat([text_ids, torch.full_like(text_ids, -1)], dim=0)
        t_in = t.expand(2)

        # DiT.forward signature (SWivid/F5-TTS main branch as of 2026):
        #   forward(x, cond, text, time, drop_audio_cond=False,
        #           drop_text=False, mask=None)
        pred = self.dit(x=x_in, cond=cond_in, text=text_in, time=t_in,
                        drop_audio_cond=False, drop_text=False, mask=None)

        pred_cond, pred_uncond = pred[0:1], pred[1:2]
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


def main():
    args = parse_args()

    print(f"Loading F5-TTS {args.variant} from {args.ckpt}", file=sys.stderr)
    dit = load_f5_model(args.ckpt, args.variant)
    wrapper = F5ExportWrapper(dit)

    T = args.dummy_seq_len
    x = torch.randn(1, T, args.mel_dim)
    cond = torch.randn(1, T, args.mel_dim)
    text_ids = torch.zeros(1, T, dtype=torch.long)
    t = torch.tensor([0.5], dtype=torch.float32)
    guidance_scale = torch.tensor([2.0], dtype=torch.float32)

    print(f"Exporting to {args.out} (opset {args.opset})", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    torch.onnx.export(
        wrapper,
        (x, cond, text_ids, t, guidance_scale),
        args.out,
        input_names=["x", "cond", "text_ids", "t", "guidance_scale"],
        output_names=["pred"],
        opset_version=args.opset,
        dynamic_axes={
            "x":        {1: "T"},
            "cond":     {1: "T"},
            "text_ids": {1: "T"},
            "pred":     {1: "T"},
        },
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
