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
            tile_view_32 = pto.TileViewType.get([32, 32], f32, ctx)
            ub = pto.AddressSpaceAttr.get(pto.AddressSpace.UB, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, ub, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("print_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # Create constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0 = entry.arguments[0]

                # Create tensor view
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result

                # Create tile_view using subview
                sv0 = pto.SubviewOp(tile_view_32, tv0, [c0, c0], [c32, c32]).result

                # Allocate tile buffer
                tb0 = pto.AllocTileOp(tile_buf_32).result

                # Load data into tile buffer
                pto.TLoadOp(None, sv0, tb0)

                # Test TPrintOp with TileBufType
                pto.TPrintOp(tb0)

                # Test TPrintOp with TileViewType
                pto.TPrintOp(sv0)

                # Return from function
                func.ReturnOp([])

            # Verify the operation
            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
