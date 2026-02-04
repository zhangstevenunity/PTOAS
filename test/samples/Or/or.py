from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import IntegerType, IndexType


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

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], i32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_i32, ptr_i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_scalar_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1 = entry.arguments

                # %0/%1/%2 = pto.make_tensor_view %arg?, shape=[%c32,%c32] strides=[%c32,%c1]
                # 这里用原生 builder：通常签名会是 (result_type, ptr, shape, strides)
                tv_src0 = pto.MakeTensorViewOp(tv2_i32, arg0, [c32, c32], [c32, c1]).result
                tv_src1 = pto.MakeTensorViewOp(tv2_i32, arg0, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_i32, arg1, [c32, c32], [c32, c1]).result

                # %3/%4/%8 = pto.subview %tv, offsets=[%c0,%c0], sizes=[%c32,%c32]
                sv_src0 = pto.PartitionViewOp(tile_view_32, tv_src0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_src1 = pto.PartitionViewOp(tile_view_32, tv_src1, offsets=[c0, c0], sizes=[c32, c32]).result

                # %5/%6/%7 = pto.alloc_tile : <32x32xi32>
                tb_src0 = pto.AllocTileOp(tile_buf_32).result
                tb_src1 = pto.AllocTileOp(tile_buf_32).result
                tb_dst = pto.AllocTileOp(tile_buf_32).result

                pto.TLoadOp(None, sv_src0, tb_src0)  # result=None
                pto.TLoadOp(None, sv_src1, tb_src1)  # result=None

                pto.TOrOp(tb_src0, tb_src1, tb_dst)

                # %8 = subview on output tensor_view
                sv_dst = pto.PartitionViewOp(tile_view_32, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result

                # pto.store_dps_tb ins(%tb1) outs(%sv1)
                pto.TStoreOp(None, tb_dst, sv_dst)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())