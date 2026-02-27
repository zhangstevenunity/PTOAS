from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import IntegerType, IndexType, IntegerAttr


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            i16 = IntegerType.get_signless(16, ctx)
            ptr_i16 = pto.PtrType.get(i16, ctx)

            tv2_i16 = pto.TensorViewType.get(2, i16, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], i16, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], i16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i16, ptr_i16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("ors_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(i16, IntegerAttr.get(i16, 88)).result

                arg0, arg1 = entry.arguments

                # %0/%1/%2 = pto.make_tensor_view %arg?, shape=[%c32,%c32] strides=[%c32,%c1]
                # 这里用原生 builder：通常签名会是 (result_type, ptr, shape, strides)
                tv_src = pto.MakeTensorViewOp(tv2_i16, arg0, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_i16, arg1, [c32, c32], [c32, c1]).result

                # %3/%4/%8 = pto.subview %tv, offsets=[%c0,%c0], sizes=[%c32,%c32]
                sv_src = pto.PartitionViewOp(tile_view_32, tv_src, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_dst = pto.PartitionViewOp(tile_view_32, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result

                # %5/%6/%7 = pto.alloc_tile : <32x32xi32>
                tb_src = pto.AllocTileOp(tile_buf_32).result
                tb_dst = pto.AllocTileOp(tile_buf_32).result

                pto.TLoadOp(None, sv_src, tb_src)  # result=None

                pto.TOrSOp(tb_src, scale, tb_dst)

                # pto.store_dps_tb ins(%tb1) outs(%sv1)
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())
