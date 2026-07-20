"""Dump the I/O signature and metadata of an ONNX model.

Usage:
    python -m omnivoice.scripts.inspect_onnx path/to/model.onnx
"""

import argparse
import sys

import onnx


_ELEM_TYPE = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 9: "bool", 10: "float16", 11: "float64",
    12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128",
    16: "bfloat16",
}


def _shape(t):
    dims = []
    for d in t.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else (d.dim_param or "?"))
    return dims


def dump(path: str) -> None:
    m = onnx.load(path, load_external_data=False)
    print(f"file: {path}")
    print(f"ir_version: {m.ir_version}")
    print(f"producer: {m.producer_name} {m.producer_version}")
    print(f"opsets: {[(op.domain or 'ai.onnx', op.version) for op in m.opset_import]}")

    print("\n=== inputs ===")
    for i in m.graph.input:
        et = _ELEM_TYPE.get(i.type.tensor_type.elem_type, i.type.tensor_type.elem_type)
        print(f"  {i.name}: {et} shape={_shape(i)}")

    print("\n=== outputs ===")
    for o in m.graph.output:
        et = _ELEM_TYPE.get(o.type.tensor_type.elem_type, o.type.tensor_type.elem_type)
        print(f"  {o.name}: {et} shape={_shape(o)}")

    if m.metadata_props:
        print("\n=== metadata ===")
        for p in m.metadata_props:
            v = p.value if len(p.value) < 400 else p.value[:400] + "..."
            print(f"  {p.key}: {v}")

    ext = [
        init.name
        for init in m.graph.initializer
        if init.data_location == onnx.TensorProto.EXTERNAL
    ]
    if ext:
        print(f"\n{len(ext)} initializers stored externally (weights in .onnx_data)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    dump(args.path)
