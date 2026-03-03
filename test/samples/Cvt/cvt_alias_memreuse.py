from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            num_tiles = 120
            tile = 32

            f32 = F32Type.get(ctx)
            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            ptr_f16 = pto.PtrType.get(f16, ctx)

            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tile_view_f32 = pto.PartitionTensorViewType.get([tile, tile], f32, ctx)
            tile_view_f16 = pto.PartitionTensorViewType.get([tile, tile], f16, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, pto.TileConfig.fractalABSize, pd, ctx)
            tb_f32 = pto.TileBufType.get([tile, tile], f32, vec, [tile, tile], cfg, ctx)
            tb_f16 = pto.TileBufType.get([tile, tile], f16, vec, [tile, tile], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("cvt_alias_memreuse_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                arg_in, arg_out = entry.arguments
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c32 = arith.ConstantOp(idx, tile).result
                c_rows = arith.ConstantOp(idx, num_tiles * tile).result

                in_tv = pto.MakeTensorViewOp(tv2_f32, arg_in, [c32, c32], [c32, c1]).result
                out_tv = pto.MakeTensorViewOp(tv2_f16, arg_out, [c_rows, c32], [c32, c1]).result
                in_sv = pto.PartitionViewOp(
                    tile_view_f32, in_tv, offsets=[c0, c0], sizes=[c32, c32]
                ).result

                src = pto.AllocTileOp(tb_f32).result
                pto.TLoadOp(None, in_sv, src)

                converted = []
                for _ in range(num_tiles):
                    dst = pto.AllocTileOp(tb_f16).result
                    pto.TCvtOp(src, dst)
                    converted.append(dst)

                for i, dst in enumerate(converted):
                    row_off = arith.ConstantOp(idx, i * tile).result
                    out_sv = pto.PartitionViewOp(
                        tile_view_f16, out_tv, offsets=[row_off, c0], sizes=[c32, c32]
                    ).result
                    pto.TStoreOp(None, dst, out_sv)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
