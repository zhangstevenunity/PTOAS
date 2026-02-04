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

            tv2_f32 = pto.TensorViewType.get([32, 32], f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32, ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_ternary_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # Create constants instead of immediate numbers
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1, arg2, arg3 = entry.arguments

                # Use constants in tensor views instead of literals
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_f32, arg2, [c32, c32], [c32, c1]).result
                tv3 = pto.MakeTensorViewOp(tv2_f32, arg3, [c32, c32], [c32, c1]).result

                # Use constants for offsets and sizes in SubviewOp
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                sv2 = pto.PartitionViewOp(tile_view_32, tv2, offsets=[c0, c0], sizes=[c32, c32]).result

                # Allocate tiles
                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result
                tb2 = pto.AllocTileOp(tile_buf_32).result
                tb3 = pto.AllocTileOp(tile_buf_32).result

                # Load data into tiles
                pto.TLoadOp(None, sv0, tb0)  # result=None
                pto.TLoadOp(None, sv1, tb1)  # result=None
                pto.TLoadOp(None, sv2, tb2)  # result=None

                # Perform operation (Add in this case, it can be other operations)
                pto.TAddCOp(tb0, tb1, tb2, tb3)

                # Subview on output tensor view
                sv3 = pto.PartitionViewOp(tile_view_32, tv3, offsets=[c0, c0], sizes=[c32, c32]).result

                # Store the result
                pto.TStoreOp(None, tb3, sv3)

                # Return from function
                func.ReturnOp([])

            # Verify the operation
            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
