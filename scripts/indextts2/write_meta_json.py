# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Write meta.json — runtime knobs the C++ side embeds into every ONNX
model's `custom_metadata_map`. We also write a standalone meta.json for
humans / tooling. The values here are pulled from the model config.

Fields written (keys mirror `OfflineTtsIndexTts2ModelMetaData`):
    version             int
    sample_rate         int   (vocoder output rate)
    vocab_size          int
    bos_token_id        int
    eos_token_id        int
    pad_token_id        int
    speaker_embed_dim   int
    emotion_embed_dim   int
    kv_num_layers       int
    kv_num_heads        int
    kv_head_dim         int
    max_audio_tokens    int   (default cap; overridable at request time)
"""

import argparse
import json

from huggingface_hub import hf_hub_download


def load_config(ckpt: str, variant: str) -> dict:
    p = hf_hub_download(repo_id=ckpt, filename=f"{variant}/config.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.ckpt, args.variant)

    meta = {
        "version": 1,
        "sample_rate": int(cfg.get("sample_rate", 24000)),
        "vocab_size": int(cfg["text_vocab_size"]),
        "bos_token_id": int(cfg.get("bos_token_id", 1)),
        "eos_token_id": int(cfg.get("eos_token_id", 2)),
        "pad_token_id": int(cfg.get("pad_token_id", 0)),
        "speaker_embed_dim": int(cfg["speaker_embed_dim"]),
        "emotion_embed_dim": int(cfg["emotion_embed_dim"]),
        "kv_num_layers": int(cfg["num_hidden_layers"]),
        "kv_num_heads": int(cfg["num_attention_heads"]),
        "kv_head_dim": int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]),
        "max_audio_tokens": int(cfg.get("max_audio_tokens", 2000)),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
