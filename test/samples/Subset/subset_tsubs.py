from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            tile_8x128 = pto.TileBufType.get([8, 128], f32, vec, [8, 128], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("subset_tsubs_demo", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                scale = arith.ConstantOp(f32, 1.0).result

                workspace = pto.AllocTileOp(tile_8x128).result
                sub0 = pto.SubsetOp(workspace, [c0, c0], sizes=[8, 64]).result

                # Use subset as both src and dst to ensure tile lowering is preserved.
                pto.TSubSOp(sub0, scale, sub0)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
