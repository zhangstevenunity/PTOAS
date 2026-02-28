from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IntegerType, IntegerAttr, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)
            i64 = IntegerType.get_signless(64, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            # valid_shape=[-1, -1] means v_row/v_col are dynamic and must be
            # provided via valid_row/valid_col operands.
            tile_buf_dynamic = pto.TileBufType.get([32, 32], f32, vec, [-1, -1], cfg, ctx)

            # Demo signature: (base_addr:i64, vrow:i32, vcol:i32) -> ()
            fn_ty = func.FunctionType.get([i64, i32, i32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("alloc_tile_with_addr_demo", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                base_addr_i64, vrow_i32, vcol_i32 = entry.arguments
                vrow = arith.IndexCastOp(IndexType.get(ctx), vrow_i32).result
                vcol = arith.IndexCastOp(IndexType.get(ctx), vcol_i32).result

                # addr comes from function argument (i64 value).
                _ = pto.AllocTileOp(tile_buf_dynamic, addr=base_addr_i64,
                                    valid_row=vrow, valid_col=vcol).result

                # addr as a constant (i64 value).
                addr_const = arith.ConstantOp(i64, IntegerAttr.get(i64, 0x1000)).result
                _ = pto.AllocTileOp(tile_buf_dynamic, addr=addr_const,
                                    valid_row=vrow, valid_col=vcol).result

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

