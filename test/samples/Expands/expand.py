from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_expand_scalar_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(f32, 3.14).result

                arg0 = entry.arguments[0]
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result

                tb1 = pto.AllocTileOp(tile_buf_32).result
                pto.TExpandsOp(scale, tb1)

                # Using constant c32 for subview size
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())