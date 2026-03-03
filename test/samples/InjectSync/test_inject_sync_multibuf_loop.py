#!/usr/bin/env python3
from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    F32Type,
    IndexType,
)
from mlir.dialects import func, arith, scf, pto
from mlir.dialects.arith import CmpIPredicate


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()
            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)

            # A minimal ping-pong (double-buffer) loop.
            #
            # This triggers multi-buffer synchronization insertion on the loop
            # back-edge (MTE3 -> MTE2) when `--enable-insert-sync` is enabled:
            # the inserted sync needs 2 event IDs (ping-pong) because the
            # dependency touches local memory (VEC/UB), so the event id is
            # selected by `iv % 2`.
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)

            fractal_ab_size = pto.TileConfig.fractalABSize
            cfg = pto.TileBufConfigAttr.get(bl, sl, fractal_ab_size, pd, ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            ws_type = pto.TileBufType.get([32, 64], f32, vec, [32, 32], cfg, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_inject_sync_multibuf_loop", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c2 = arith.ConstantOp(idx, 2).result
                c4 = arith.ConstantOp(idx, 4).result
                c32 = arith.ConstantOp(idx, 32).result

                src_ptr, dst_ptr = entry.arguments

                tv_src = pto.MakeTensorViewOp(tv2_f32, src_ptr, [c32, c32], [c32, c1]).result
                tv_dst = pto.MakeTensorViewOp(tv2_f32, dst_ptr, [c32, c32], [c32, c1]).result
                sv_src = pto.PartitionViewOp(tile_view_32, tv_src, offsets=[c0, c0], sizes=[c32, c32]).result
                sv_dst = pto.PartitionViewOp(tile_view_32, tv_dst, offsets=[c0, c0], sizes=[c32, c32]).result

                # Allocate a single workspace and create two non-overlapping
                # 32x32 subsets (ping/pong) to model double buffering.
                workspace = pto.AllocTileOp(ws_type).result
                ping = pto.SubsetOp(workspace, [c0, c0], sizes=[32, 32]).result
                pong = pto.SubsetOp(workspace, [c0, c32], sizes=[32, 32]).result

                loop = scf.ForOp(c0, c4, c1, [])
                with InsertionPoint(loop.body):
                    iv = loop.induction_variable

                    parity = arith.RemUIOp(iv, c2).result
                    is_even = arith.CmpIOp(CmpIPredicate.eq, parity, c0).result
                    ifop = scf.IfOp(is_even, [], hasElse=True)
                    with InsertionPoint(ifop.then_block):
                        pto.TLoadOp(None, sv_src, ping)
                        pto.TStoreOp(None, ping, sv_dst)
                        scf.YieldOp([])
                    with InsertionPoint(ifop.else_block):
                        pto.TLoadOp(None, sv_src, pong)
                        pto.TStoreOp(None, pong, sv_dst)
                        scf.YieldOp([])

                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
