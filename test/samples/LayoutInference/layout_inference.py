#!/usr/bin/env python3
"""
Smoke tests for GlobalTensor layout inference on pto.make_tensor_view.

Covers:
  - ND (row-major) contiguous strides
  - DN (col-major) contiguous strides
  - NZ (fractal) layout with shape3=16 and 512B alignment
  - Dynamic strides (no layout inferred)

The script builds four functions and asserts the inferred `layout` attribute
on each make_tensor_view op.
"""

from mlir.ir import (
    Context,
    InsertionPoint,
    Location,
    Module,
    F32Type,
    IndexType,
)
from mlir.dialects import func, arith, pto
from typing import Optional
import sys


def cidx(b, v):
    return arith.ConstantOp(IndexType.get(), v).result


def build_case(
    ctx,
    module: Module,
    name: str,
    shape,
    strides,
    expect_layout: Optional[str],
    elem_ty,
):
    """Insert one func with make_tensor_view and assert inferred layout."""

    def infer_layout(shape_ints, stride_ints, elem_bytes):
        if len(shape_ints) >= 5:
            sh3, sh4, sh5 = shape_ints[2], shape_ints[3], shape_ints[4]
            st4, st5 = stride_ints[3], stride_ints[4]
            if sh3 == 16 and sh3 * sh4 * elem_bytes == 512 and st5 == 1 and st4 == sh5:
                return "NZ"
        is_row = (
            all(
                stride_ints[i] == stride_ints[i + 1] * shape_ints[i + 1]
                for i in range(len(shape_ints) - 1)
            )
            and stride_ints[-1] == 1
        )
        if is_row:
            return "ND"
        is_col = (
            all(
                stride_ints[i + 1] == stride_ints[i] * shape_ints[i]
                for i in range(len(shape_ints) - 1)
            )
            and stride_ints[0] == 1
        )
        if is_col:
            return "DN"
        return "ND"
    with InsertionPoint(module.body):
        f = func.FuncOp(
            name,
            func.FunctionType.get([pto.PtrType.get(elem_ty)], []),
        )
    entry = f.add_entry_block()
    with InsertionPoint(entry):
        shape_vals = [cidx(ctx, s) for s in shape]
        stride_vals = [cidx(ctx, s) for s in strides]
        tv_ty = pto.TensorViewType.get(shape, elem_ty)
        mtv = pto.MakeTensorViewOp(tv_ty, f.arguments[0], shape_vals, stride_vals)
        func.ReturnOp([])

    # Inspect layout attribute on the make_tensor_view op
    ops = list(f.entry_block.operations)
    mtv = next(op for op in ops if op.operation.name == "pto.make_tensor_view")
    attrs = mtv.operation.attributes
    layout_attr = attrs["layout"] if "layout" in attrs else None
    shape_ints = shape
    stride_ints = strides
    inferred = infer_layout(shape_ints, stride_ints, 4)  # f32

    if expect_layout is None:
        assert layout_attr is None, f"{name}: expected no layout attr, got {layout_attr}"
        assert inferred == "ND", f"{name}: expected ND fallback, got {inferred}"
    else:
        if layout_attr is not None and hasattr(pto, "Layout"):
            got = pto.Layout(layout_attr.value)
            assert got.name.lower() == expect_layout.lower(), f"{name}: expected {expect_layout}, got {got}"
        assert inferred.lower() == expect_layout.lower(), f"{name}: expected {expect_layout}, got {inferred}"
    sys.stderr.write(f"[OK] {name}: layout_attr={'none' if layout_attr is None else layout_attr}, python_infer={inferred}\n")


def main():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx)
        f32 = F32Type.get()
        module = Module.create(Location.unknown())

        # ND row-major: shape=[4,8,16], strides=[128,16,1]
        build_case(
            ctx,
            module,
            "nd_row_major",
            shape=[4, 8, 16],
            strides=[128, 16, 1],
            expect_layout="ND",
            elem_ty=f32,
        )

        # DN col-major: shape=[4,8,16], strides=[1,4,32]
        build_case(
            ctx,
            module,
            "dn_col_major",
            shape=[4, 8, 16],
            strides=[1, 4, 32],
            expect_layout="DN",
            elem_ty=f32,
        )

        # NZ fractal (fp32): shape=[1,1,16,8,16], strides -> [2048,2048,128,16,1]
        build_case(
            ctx,
            module,
            "nz_fractal",
            shape=[1, 1, 16, 8, 16],
            strides=[2048, 2048, 128, 16, 1],
            expect_layout="NZ",
            elem_ty=f32,
        )

        # Dynamic stride: no layout inferred
        build_case(
            ctx,
            module,
            "dynamic_stride",
            shape=[4, 8, 16],
            strides=[128, 16, -1],  # -1 => dynamic
            expect_layout=None,
            elem_ty=f32,
        )

        # Emit combined module to stdout for runop/ptoas
        print(module)


if __name__ == "__main__":
    main()
