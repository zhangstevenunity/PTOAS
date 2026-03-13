from mlir.ir import Context, Location, Module, InsertionPoint, F32Type, IndexType
from mlir.dialects import func, arith, pto


def _idx_const(v: int):
    return arith.ConstantOp(IndexType.get(), v).result


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2 = pto.TensorViewType.get(2, f32, ctx)
            tile_view = pto.PartitionTensorViewType.get([16, 16], f32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tile_buf = pto.TileBufType.get([16, 16], f32, vec, [16, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_scalar_cross_pipe", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments
                c0 = _idx_const(0)
                c1 = _idx_const(1)
                c4 = _idx_const(4)
                c16 = _idx_const(16)

                src_tv = pto.MakeTensorViewOp(tv2, src, [c16, c16], [c16, c1]).result
                dst_tv = pto.MakeTensorViewOp(tv2, dst, [c16, c16], [c16, c1]).result
                src_part = pto.PartitionViewOp(
                    tile_view, src_tv, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                dst_part = pto.PartitionViewOp(
                    tile_view, dst_tv, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                ub = pto.AllocTileOp(tile_buf).result

                src_off = pto.addptr(src, c4)
                dst_off = pto.addptr(dst, c4)
                one = arith.ConstantOp(f32, 1.0).result
                pto.TLoadOp(None, src_part, ub)
                pto.store_scalar(src_off, c0, one)
                loaded = pto.load_scalar(f32, dst_off, c0)
                pto.store_scalar(dst_off, c1, loaded)
                pto.TStoreOp(None, ub, dst_part)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
