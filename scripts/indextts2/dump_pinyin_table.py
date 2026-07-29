# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Dump CJK char -> pinyin lookup table from pypinyin.

The table is written line-by-line as:
    <U+HEX>\t<pinyin1>[,<pinyin2>,...]

where multiple pinyin are separated by commas (heteronym support), tone
marks are STRIPPED (F5-TTS / IndexTTS-2 style), and the CJK char is
represented as a hex codepoint for robustness against editor mangling.

Runtime cost: reading ~20 k lines at startup, one hash-map insert each.
"""

import argparse
import sys

from pypinyin import Style, pinyin
from pypinyin.constants import RE_HANS

CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
]


def iter_cjk_codepoints():
    for lo, hi in CJK_RANGES:
        for cp in range(lo, hi + 1):
            yield cp


def to_pinyin_for(cp):
    ch = chr(cp)
    if not RE_HANS.match(ch):
        return None
    result = pinyin(ch, style=Style.NORMAL, heteronym=True, errors="ignore")
    if not result or not result[0]:
        return None
    return result[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for cp in iter_cjk_codepoints():
            pys = to_pinyin_for(cp)
            if not pys:
                continue
            f.write(f"U+{cp:04X}\t{','.join(pys)}\n")
            n_written += 1

    print(f"Wrote {n_written} entries to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
