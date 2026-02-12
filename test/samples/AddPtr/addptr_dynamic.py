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
            idx_t = IndexType.get(ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, idx_t, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("addptr_dynamic_offset", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx_t, 0).result
                c1 = arith.ConstantOp(idx_t, 1).result
                c32 = arith.ConstantOp(idx_t, 32).result

                src, row_idx, dst = entry.arguments
                row_off = arith.MulIOp(row_idx, c32).result
                src_off = pto.AddPtrOp(src, row_off).result

                tv0 = pto.MakeTensorViewOp(tv2_f32, src_off, [c32, c32], [c32, c1]).result
                sv0 = pto.PartitionViewOp(
                    tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                tb0 = pto.AllocTileOp(tile_buf_32).result
                pto.TLoadOp(None, sv0, tb0)

                tv1 = pto.MakeTensorViewOp(tv2_f32, dst, [c32, c32], [c32, c1]).result
                sv1 = pto.PartitionViewOp(
                    tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]
                ).result
                pto.TStoreOp(None, tb0, sv1)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
