from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cmp = pto.CmpModeAttr.get(ctx, pto.CmpMode.LT)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            u8 = IntegerType.get_unsigned(8, ctx)
            ptr_u8 = pto.PtrType.get(u8, ctx)

            tv2_u8 = pto.TensorViewType.get(2, u8, ctx)
            tile_view_u8 = pto.PartitionTensorViewType.get([32, 32], u8, ctx)
            tile_buf_u8 = pto.TileBufType.get([32, 32], u8, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_u8], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_cmp_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                # Add constants for subview offsets and sizes
                subview_offset = arith.ConstantOp(IndexType.get(ctx), 0).result  # constant for offset
                subview_size = arith.ConstantOp(IndexType.get(ctx), 32).result  # constant for size

                arg0, arg1, arg2 = entry.arguments

                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_u8, arg2, [c32, c32], [c32, c1]).result

                # Use constants for subview offsets and sizes
                sv0 = pto.PartitionViewOp(tile_view_f32, tv0, offsets=[subview_offset, subview_offset], sizes=[subview_size, subview_size]).result
                sv1 = pto.PartitionViewOp(tile_view_f32, tv1, offsets=[subview_offset, subview_offset], sizes=[subview_size, subview_size]).result

                tb0 = pto.AllocTileOp(tile_buf_f32).result
                tb1 = pto.AllocTileOp(tile_buf_f32).result
                tb2 = pto.AllocTileOp(tile_buf_u8).result

                pto.TLoadOp(None, sv0, tb0)  # result=None
                pto.TLoadOp(None, sv1, tb1)  # result=None

                pto.TCmpOp(tb0, tb1, tb2, cmpMode=cmp)

                sv2 = pto.PartitionViewOp(tile_view_u8, tv2, offsets=[subview_offset, subview_offset], sizes=[subview_size, subview_size]).result

                pto.TStoreOp(None, tb2, sv2)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
