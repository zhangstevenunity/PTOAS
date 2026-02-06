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

            # tile = 32 x 128
            tile_view = pto.PartitionTensorViewType.get([32, 128], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            tile_buf = pto.TileBufType.get([32, 128], f32, vec, [32, 128], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("relu_kernel_blocked", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # constants
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result
                c128 = arith.ConstantOp(IndexType.get(ctx), 128).result
                c1024 = arith.ConstantOp(IndexType.get(ctx), 1024).result

                scale = arith.ConstantOp(f32, 0.0).result  # relu threshold

                arg0, arg1 = entry.arguments

                cid = pto.GetBlockIdxOp().result
                sub_bid = pto.GetSubBlockIdxOp().result
                sub_bnum = pto.GetSubBlockNumOp().result

                cidmul = arith.MulIOp(cid, sub_bnum).result
                vid = arith.AddIOp(cidmul, sub_bid).result

                # offset in N dimension: vid * 128
                offset_n = arith.MulIOp( arith.IndexCastOp(IndexType.get(), vid).result, c128).result

                # tensor views for full matrix 32 x 1024
                tv0 = pto.MakeTensorViewOp( tv2_f32, arg0, [c32, c1024], [c1024, c1]).result
                tv1 = pto.MakeTensorViewOp( tv2_f32, arg1, [c32, c1024], [c1024, c1]).result

                # subview: tile = [32,128], offset = [0, offset_n]
                sv0 = pto.PartitionViewOp( tile_view, tv0, offsets=[c0, offset_n], sizes=[c32, c128]).result

                tb0 = pto.AllocTileOp(tile_buf).result
                tb1 = pto.AllocTileOp(tile_buf).result

                pto.TLoadOp(None, sv0, tb0)  # result=None

                pto.TLReluOp(tb0, scale, tb1)

                sv1 = pto.PartitionViewOp( tile_view, tv1, offsets=[c0, offset_n], sizes=[c32, c128]).result

                pto.TStoreOp(None, tb1, sv1)

                func.ReturnOp([])

            m.operation.verify()

            return m


if __name__ == "__main__":
    print(build())