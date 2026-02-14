from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F16Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.RowMajor, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            # Boxed layout: innerRows=16, innerCols=16 (f16).
            # Dynamic row offset aligned to innerRows; col offset must be 0.
            tile_ty = pto.TileBufType.get([32, 32], f16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([idx], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("subset_boxed_dynamic", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                i0 = entry.arguments[0]
                c0 = arith.ConstantOp(idx, 0).result
                c16 = arith.ConstantOp(idx, 16).result
                row_off = arith.MulIOp(i0, c16).result

                t0 = pto.AllocTileOp(tile_ty).result
                _sub = pto.SubsetOp(t0, [row_off, c0], sizes=[16, 32]).result

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
