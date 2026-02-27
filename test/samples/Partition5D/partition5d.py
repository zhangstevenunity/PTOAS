"""
5D partition_view -> tile load/store sample (static shape).

Scenario from design doc:
1) make_tensor_view defines 5D global tensor
2) partition_view slices a 5D window
3) tload collapses to 2D tile_buf
4) tstore writes back to the destination tensor_view
"""

from mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    Module,
    IndexType,
)
from mlir.dialects import arith, func, pto, builtin


def idx(val: int):
    return arith.ConstantOp(IndexType.get(), val).result


def build_module():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        f32 = builtin.F32Type.get()
        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

        tensor_view_ty = pto.TensorViewType.get([1, 1, 16, 64, 64], f32)
        part_view_ty = pto.PartitionTensorViewType.get([1, 1, 16, 16, 16], f32)
        tile_buf_ty = pto.TileBufType.get(
            [256, 16], f32, vec, [256, 16], pto.TileBufConfigAttr.get_default(ctx)
        )

        ptr_f32 = pto.PtrType.get(f32)

        m = Module.create()
        with InsertionPoint(m.body):
            @func.FuncOp.from_py_func(ptr_f32, ptr_f32)
            def run_partition(src, dst):
                c0 = idx(0)
                # Shapes/strides for make_tensor_view
                shape = [idx(1), idx(1), idx(16), idx(64), idx(64)]
                strides = [idx(65536), idx(65536), idx(4096), idx(64), idx(1)]

                base_view = pto.MakeTensorViewOp(tensor_view_ty, src, shape, strides).result

                part = pto.PartitionViewOp(
                    part_view_ty,
                    base_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result

                tile = pto.AllocTileOp(tile_buf_ty).result
                pto.TLoadOp(None, part, tile)

                dst_view = pto.MakeTensorViewOp(tensor_view_ty, dst, shape, strides).result
                dst_part = pto.PartitionViewOp(
                    part_view_ty,
                    dst_view,
                    offsets=[c0, c0, c0, c0, c0],
                    sizes=[idx(1), idx(1), idx(16), idx(16), idx(16)],
                ).result
                pto.TStoreOp(None, tile, dst_part)

                return

        return m


if __name__ == "__main__":
    module = build_module()
    print(module)
