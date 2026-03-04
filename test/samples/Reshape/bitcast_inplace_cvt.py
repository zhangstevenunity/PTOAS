from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            f32 = F32Type.get(ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32_f16 = pto.PartitionTensorViewType.get([32, 32], f16, ctx)
            tile_view_32_f32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tile_buf_32_f16 = pto.TileBufType.get([32, 32], f16, vec, [32, 32], cfg, ctx)
            tile_buf_32_f32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            rmode_attr = pto.RoundModeAttr.get(pto.RoundMode.ROUND, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("bitcast_inplace_cvt", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg_in, arg_out = entry.arguments

                tv_in = pto.MakeTensorViewOp(tv2_f16, arg_in, [c32, c32], [c32, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f32, arg_out, [c32, c32], [c32, c1]).result

                sv_in = pto.PartitionViewOp(tile_view_32_f16, tv_in, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_out = pto.PartitionViewOp(tile_view_32_f32, tv_out, offsets=[c0, c0], sizes=[c32, c32]).result

                # Allocate a f32 tile, then bitcast it as a f16 view to load input.
                # Use tcvt ins/outs to express an in-place conversion back to f32.
                tb_f32 = pto.AllocTileOp(tile_buf_32_f32).result
                tb_f16 = pto.BitcastOp(tile_buf_32_f16, tb_f32).result

                pto.TLoadOp(None, sv_in, tb_f16)  # load f16 into the aliased view
                pto.TCvtOp(tb_f16, tb_f32, rmode=rmode_attr)
                pto.TStoreOp(None, tb_f32, sv_out)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

