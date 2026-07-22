#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#                2026  Waad Ben Kheder
"""Quantize an exported OmniVoice LM ONNX to int8 or fp16.

fp16 halves the file size with no measurable quality drop on English.
int8 (dynamic quantization of MatMul/Gemm ops) roughly quarters it at a
small cost in prosody. Both are drop-in replacements for the fp32 file
that build_bundle.sh produces -- point --omnivoice-model at the quantized
file, keep every other flag identical, and the CLI will pick it up.

Usage:

    # fp16 (recommended)
    python export_omnivoice_quantized.py \\
        --input-onnx  ./sherpa-onnx-omnivoice-<DATE>/omnivoice.onnx \\
        --dtype       fp16 \\
        --output-onnx ./sherpa-onnx-omnivoice-<DATE>/omnivoice.fp16.onnx

    # int8
    python export_omnivoice_quantized.py \\
        --input-onnx  ./sherpa-onnx-omnivoice-<DATE>/omnivoice.onnx \\
        --dtype       int8 \\
        --output-onnx ./sherpa-onnx-omnivoice-<DATE>/omnivoice.int8.onnx

Also quantizes the paired cached-LM ONNX pair (prefix + target) when
present under the same directory, using the same --dtype.

Requires: onnx, onnxruntime, onnxconverter-common (for fp16).
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantize an OmniVoice LM ONNX to fp16 or int8.")
    ap.add_argument("--input-onnx", required=True,
                    help="Path to omnivoice.onnx (fp32).")
    ap.add_argument("--dtype", required=True, choices=["fp16", "int8"],
                    help="Target precision.")
    ap.add_argument("--output-onnx", default="",
                    help="Output path. Default: same directory as input, "
                    "with the dtype suffix inserted before .onnx.")
    ap.add_argument("--also-quantize-cached", action="store_true",
                    default=True,
                    help="If omnivoice_prefix.onnx / omnivoice_target.onnx "
                    "exist alongside the input, quantize them too "
                    "(default: true).")
    ap.add_argument("--no-quantize-cached", dest="also_quantize_cached",
                    action="store_false")
    return ap.parse_args()


def default_out_path(inp: Path, dtype: str) -> Path:
    stem = inp.stem
    return inp.with_name(f"{stem}.{dtype}.onnx")


def quantize_fp16(src: Path, dst: Path) -> None:
    """Cast weights and activations to fp16 in-place on an ONNX proto."""
    import onnx
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("error: install onnxconverter-common: pip install "
              "onnxconverter-common", file=sys.stderr)
        raise

    model = onnx.load(str(src))
    # keep_io_types=True so downstream C++ still feeds fp32 PCM / int64
    # tokens without needing per-call casts.
    fp16_model = float16.convert_float_to_float16(
        model, keep_io_types=True, disable_shape_infer=False)
    onnx.save(fp16_model, str(dst),
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=f"{dst.name}_data",
              size_threshold=1024,
              convert_attribute=False)


def quantize_int8(src: Path, dst: Path) -> None:
    """Dynamic int8 quantization of MatMul/Gemm ops."""
    try:
        from onnxruntime.quantization import (quantize_dynamic, QuantType)
    except ImportError:
        print("error: install onnxruntime: pip install onnxruntime",
              file=sys.stderr)
        raise

    # QUInt8 is broadly compatible across CPU EPs. QInt8 sometimes gives
    # a small quality edge on ARM but is not always supported everywhere.
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
    )


def do_one(src: Path, dst: Path, dtype: str) -> None:
    print(f"==> {src.name} -> {dst.name}  ({dtype})", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dtype == "fp16":
        quantize_fp16(src, dst)
    else:
        quantize_int8(src, dst)
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    # Include any external-data files that share the stem.
    for extra in dst.parent.glob(dst.name + "_data*"):
        dst_size += extra.stat().st_size
    for extra in src.parent.glob(src.name + "_data*"):
        src_size += extra.stat().st_size
    ratio = dst_size / max(1, src_size)
    print(f"    {src_size/1e6:.1f} MB -> {dst_size/1e6:.1f} MB "
          f"({100*ratio:.1f}%)", flush=True)


def main() -> None:
    args = parse_args()
    src = Path(args.input_onnx).resolve()
    if not src.exists():
        sys.exit(f"input not found: {src}")

    dst = Path(args.output_onnx).resolve() if args.output_onnx else \
        default_out_path(src, args.dtype)
    do_one(src, dst, args.dtype)

    if args.also_quantize_cached:
        for name in ("omnivoice_prefix.onnx", "omnivoice_target.onnx"):
            candidate = src.parent / name
            if candidate.exists():
                out = default_out_path(candidate, args.dtype)
                do_one(candidate, out, args.dtype)

    print("done.")


if __name__ == "__main__":
    main()
