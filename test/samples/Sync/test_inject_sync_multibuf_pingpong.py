from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import arith, func, scf, pto
from mlir.ir import F16Type, IndexType, IntegerAttr, IntegerType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)

            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

            ptr_f16 = pto.PtrType.get(f16, ctx)
            tv2_f16 = pto.TensorViewType.get(2, f16, ctx)
            tile_view_16 = pto.PartitionTensorViewType.get([16, 16], f16, ctx)

            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)
            tile_buf_16 = pto.TileBufType.get([16, 16], f16, vec, [16, 16], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f16, ptr_f16], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_inject_sync_multibuf_pingpong_py", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c4 = arith.ConstantOp(idx, 4).result
                c16 = arith.ConstantOp(idx, 16).result

                tv_in = pto.MakeTensorViewOp(tv2_f16, src, [c16, c16], [c16, c1]).result
                tv_out = pto.MakeTensorViewOp(tv2_f16, dst, [c16, c16], [c16, c1]).result
                sv_in = pto.PartitionViewOp(
                    tile_view_16, tv_in, offsets=[c0, c0], sizes=[c16, c16]
                ).result
                sv_out = pto.PartitionViewOp(
                    tile_view_16, tv_out, offsets=[c0, c0], sizes=[c16, c16]
                ).result

                alloc = pto.AllocTileOp(tile_buf_16)
                alloc.operation.attributes["pto.multi_buffer"] = IntegerAttr.get(i32, 2)
                ub = alloc.result

                # Loop-carried hazard:
                # - TLOAD writes to UB on PIPE_MTE2.
                # - TSTORE reads from UB on PIPE_MTE3.
                # With multi-buffer enabled, the compiler should materialize a
                # ping/pong selector and use dynamic event-id sync on the
                # back-edge dependency.
                loop = scf.ForOp(c0, c4, c1, [])
                with InsertionPoint(loop.body):
                    pto.TLoadOp(None, sv_in, ub)
                    pto.TStoreOp(None, ub, sv_out)
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
