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

            tile_dynamic = pto.TileBufType.get([32, 32], f32, vec, [-1, -1], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("set_validshape_demo", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c16 = arith.ConstantOp(idx, 16).result
                c24 = arith.ConstantOp(idx, 24).result
                c32 = arith.ConstantOp(idx, 32).result

                tb = pto.AllocTileOp(tile_dynamic, valid_row=c32, valid_col=c32).result
                pto.SetValidShapeOp(tb, c16, c24)

                pto.TAddOp(tb, tb, tb)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
