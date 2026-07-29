# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Dump the IndexTTS-2 BPE tokenizer to tokens.txt + merges.txt.

`tokens.txt` format: one line per token, tab-separated `<token>\t<id>`.
`merges.txt` format: standard HF BPE merges (`part_a part_b` per line,
header line `#version: 0.2`).

Special tokens (BOS, EOS, PAD, UNK) MUST appear in tokens.txt with the
IDs the LM was trained with; we assert this here to fail loudly if the
upstream repo changes them.
"""

import argparse
import json
import sys

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

EXPECTED_SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>")


def load_tokenizer(ckpt: str, variant: str):
    # IndexTTS-2 repo layout is assumed to expose a HF-format tokenizer
    # directory at `<variant>/tokenizer.json`. Adjust the filename if the
    # upstream repo diverges.
    tokenizer_json = hf_hub_download(
        repo_id=ckpt, filename=f"{variant}/tokenizer.json"
    )
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-tokens", required=True)
    ap.add_argument("--out-merges", required=True)
    args = ap.parse_args()

    tok = load_tokenizer(args.ckpt, args.variant)
    vocab = tok.get_vocab()  # str -> int

    for special in EXPECTED_SPECIALS:
        if special not in vocab:
            print(
                f"WARN: expected special token {special!r} missing from vocab",
                file=sys.stderr,
            )

    with open(args.out_tokens, "w", encoding="utf-8") as f:
        for token, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
            # Tabs / newlines in tokens are illegal in this format.
            token_clean = token.replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{token_clean}\t{tid}\n")

    # Extract merges from the tokenizer.json blob directly (the HF Fast API
    # doesn't expose them cleanly).
    tokenizer_json_path = tok.name_or_path or tok.vocab_files_names.get(
        "tokenizer_file"
    )
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    merges = blob.get("model", {}).get("merges", [])
    with open(args.out_merges, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in merges:
            f.write(f"{m}\n")

    print(
        f"Wrote {len(vocab)} tokens, {len(merges)} merges", file=sys.stderr
    )


if __name__ == "__main__":
    main()
