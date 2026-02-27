from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import IndexType, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            i32 = IntegerType.get_signless(32, ctx)
            ptr_i32 = pto.PtrType.get(i32, ctx)

            tv2_i32 = pto.TensorViewType.get(2, i32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], i32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_i32 = pto.TileBufType.get([32, 32], i32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32, ptr_i32, ptr_i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("mscatter_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1, arg2 = entry.arguments

                # %0/%1/%2 = pto.make_tensor_view %arg?, shape=[%c32,%c32] strides=[%c32,%c1]
                tv0 = pto.MakeTensorViewOp(tv2_i32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_i32, arg1, [c32, c32], [c32, c1]).result
                tv2 = pto.MakeTensorViewOp(tv2_i32, arg2, [c32, c32], [c32, c1]).result

                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result
                sv2 = pto.PartitionViewOp(tile_view_32, tv2, offsets=[c0, c0], sizes=[c32, c32]).result

                tb0 = pto.AllocTileOp(tile_buf_i32).result
                tb1 = pto.AllocTileOp(tile_buf_i32).result

                # pto.load_dps_tb ins(%sv) outs(%tb)
                pto.TLoadOp(None, sv0, tb0)
                pto.TLoadOp(None, sv1, tb1)  # result=None

                pto.MScatterOp(tb0, tb1, sv2)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
