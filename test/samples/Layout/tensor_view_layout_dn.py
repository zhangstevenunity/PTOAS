"""
Explicit tensor_view layout sample (DN).

This checks that:
  - User-specified `layout` on pto.make_tensor_view is preserved through
    view lowering (make_tensor_view -> reinterpret_cast -> subview)
  - The generated C++ GlobalTensor uses pto::Layout::DN.
"""

from mlir.ir import Context, Location, InsertionPoint, Module, IndexType
from mlir.dialects import arith, func, pto, builtin


def idx(val: int):
    return arith.ConstantOp(IndexType.get(), val).result


def build_module():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx, load=True)

        f32 = builtin.F32Type.get()
        vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
        bl = pto.BLayoutAttr.get(pto.BLayout.ColMajor, ctx)
        sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
        pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
        cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)

        tensor_view_ty = pto.TensorViewType.get(2, f32, ctx)
        part_view_ty = pto.PartitionTensorViewType.get([16, 1], f32, ctx)
        tile_buf_ty = pto.TileBufType.get([16, 1], f32, vec, [16, 1], cfg, ctx)

        ptr_f32 = pto.PtrType.get(f32)
        layout_dn = pto.LayoutAttr.get(pto.Layout.DN, ctx)

        m = Module.create()
        with InsertionPoint(m.body):

            @func.FuncOp.from_py_func(ptr_f32, ptr_f32)
            def run(src, dst):
                c0 = idx(0)

                c1 = idx(1)
                c16 = idx(16)
                shape = [c16, c1]
                # DN for (16 x 1): addr(r, c) = base + r*1 + c*rows.
                strides = [c1, c1]

                src_view = pto.MakeTensorViewOp(
                    tensor_view_ty, src, shape, strides, layout=layout_dn
                ).result
                src_part = pto.PartitionViewOp(
                    part_view_ty,
                    src_view,
                    offsets=[c0, c0],
                    sizes=[c16, c1],
                ).result

                tile = pto.AllocTileOp(tile_buf_ty).result
                pto.TLoadOp(None, src_part, tile)

                dst_view = pto.MakeTensorViewOp(
                    tensor_view_ty, dst, shape, strides, layout=layout_dn
                ).result
                dst_part = pto.PartitionViewOp(
                    part_view_ty,
                    dst_view,
                    offsets=[c0, c0],
                    sizes=[c16, c1],
                ).result

                pto.TStoreOp(None, tile, dst_part)
                return

        return m


if __name__ == "__main__":
    print(build_module())
