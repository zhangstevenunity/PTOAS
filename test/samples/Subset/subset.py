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
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)

            # Workspace in UB: 32x64, split into two 32x32 tiles (ping/pong).
            ws_type = pto.TileBufType.get([32, 64], f32, vec, [32, 64], cfg, ctx)

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("subset_pingpong_demo", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                c32 = arith.ConstantOp(idx, 32).result

                workspace = pto.AllocTileOp(ws_type).result
                ping = pto.SubsetOp(workspace, [c0, c0], sizes=[32, 32]).result
                pong = pto.SubsetOp(workspace, [c0, c32], sizes=[32, 32]).result

                pto.TAddOp(ping, ping, ping)
                pto.TAddOp(pong, pong, pong)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

