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

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("vec_add_scalar_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                scale = arith.ConstantOp(f32, 3.14).result

                arg0, arg1 = entry.arguments

                # %0/%1/%2 = pto.make_tensor_view %arg?, shape=[%c32,%c32] strides=[%c32,%c1]
                # 这里用原生 builder：通常签名会是 (result_type, ptr, shape, strides)
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

                # %3/%4/%8 = pto.subview %tv, offsets=[%c0,%c0], sizes=[32,32]
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result

                # %5/%6/%7 = pto.alloc_tile : <32x32xf32>
                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result

                pto.TLoadOp(None, sv0, tb0)  # result=None

                pto.TAddSOp(tb0, scale, tb1)

                # %8 = subview on output tensor_view
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                # pto.store_dps_tb ins(%tb1) outs(%sv1)
                pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])
            
            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())