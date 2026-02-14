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

            # Boxed layout: innerRows=16, innerCols=32/2=16 (f16).
            # Invalid subset: column offset not aligned (offC=8).
            tile_ty = pto.TileBufType.get([32, 32], f16, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("subset_invalid_boxed", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c8 = arith.ConstantOp(idx, 8).result

                t0 = pto.AllocTileOp(tile_ty).result
                # Expect verifier failure: offC=8 not multiple of innerCols=16.
                _bad = pto.SubsetOp(t0, [c0, c8], sizes=[16, 16]).result

                func.ReturnOp([])

            ok = m.operation.verify()
            if ok:
                return m
            # Expected failure for invalid subset; make python exit non-zero.
            raise SystemExit(1)


if __name__ == "__main__":
    print(build())
