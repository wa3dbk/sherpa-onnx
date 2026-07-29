# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Export IndexTTS-2's LM as a single ONNX file with two modes.

The exported graph accepts a "mode" tensor plus a superset of the
inputs the two modes need; unused tensors are placeholders. This keeps
the C++ side to a single Ort::Session.

Mode 0 (prefix):
    inputs  = mode(int32 scalar = 0),
              text_tokens(int64 (1, Tt)),
              speaker_embed(float32 (1, D_s)),
              emotion_embed(float32 (1, D_e)),
              prev_token(int64 (1, 1) — ignored),
              past_kv (float32, zero-filled placeholder shape)
    outputs = logits(float32 (1, vocab_size)),
              present_kv (float32 (num_layers, 2, 1, num_heads, T_prefix, head_dim))

Mode 1 (step):
    inputs  = mode(int32 scalar = 1),
              text_tokens (ignored — empty (1,0)),
              speaker_embed (ignored),
              emotion_embed (ignored),
              prev_token(int64 (1, 1)),
              past_kv (float32 (num_layers, 2, 1, num_heads, T_past, head_dim))
    outputs = logits(float32 (1, vocab_size)),
              present_kv (concatenated to T_past + 1)

This dual-mode wrapper is a small torch.nn.Module that dispatches on
`mode`; see the loader for the exact wiring.
"""

import argparse

import torch
import torch.nn as nn

from _indextts2_load import load_lm


class DualModeLm(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(
        self,
        mode,
        text_tokens,
        speaker_embed,
        emotion_embed,
        prev_token,
        past_kv,
    ):
        if int(mode.item()) == 0:
            return self.lm.prefix_forward(
                text_tokens, speaker_embed, emotion_embed
            )
        return self.lm.step_forward(prev_token, past_kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lm = load_lm(args.ckpt, args.variant).eval()
    wrapper = DualModeLm(lm).eval()

    L = lm.config.num_hidden_layers
    H = lm.config.num_attention_heads
    D = lm.config.hidden_size // H
    Ds = lm.config.speaker_embed_dim
    De = lm.config.emotion_embed_dim

    dummy = (
        torch.tensor(0, dtype=torch.int32),
        torch.zeros(1, 16, dtype=torch.int64),
        torch.zeros(1, Ds, dtype=torch.float32),
        torch.zeros(1, De, dtype=torch.float32),
        torch.zeros(1, 1, dtype=torch.int64),
        torch.zeros(L, 2, 1, H, 1, D, dtype=torch.float32),
    )

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=[
            "mode",
            "text_tokens",
            "speaker_embed",
            "emotion_embed",
            "prev_token",
            "past_kv",
        ],
        output_names=["logits", "present_kv"],
        dynamic_axes={
            "text_tokens": {1: "Tt"},
            "past_kv": {4: "T_past"},
            "present_kv": {4: "T_present"},
        },
        opset_version=17,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
