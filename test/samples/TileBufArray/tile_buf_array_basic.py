from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto, scf
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

            tile_ty = pto.TileBufType.get([16, 16], f32, vec, [16, 16], cfg, ctx)
            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("tile_buf_array_basic", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result

                tile0 = pto.AllocTileOp(tile_ty).result
                tile1 = pto.AllocTileOp(tile_ty).result

                # Array-style API:
                #   arr[0]      -> constant-index get
                #   arr[iv]     -> dynamic-index get
                arr = pto.TileBufArray.from_elements([tile0, tile1])
                slot0 = arr[0]
                slot1 = arr[c1]
                pto.TAddOp(slot0, slot1, slot0)

                loop = scf.ForOp(c0, c2, c1, [])
                with InsertionPoint(loop.body):
                    dyn_slot = arr[loop.induction_variable]
                    pto.TAddOp(dyn_slot, dyn_slot, dyn_slot)
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
