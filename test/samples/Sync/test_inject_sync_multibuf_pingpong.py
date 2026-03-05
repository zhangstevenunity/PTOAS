from mlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    F16Type,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
)
from mlir.dialects import arith, func, memref, scf, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f16 = F16Type.get(ctx)
            idx = IndexType.get(ctx)
            i32 = IntegerType.get_signless(32, ctx)

            gm = pto.AddressSpaceAttr.get(pto.AddressSpace.GM, ctx)
            vec = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)

            gm_ty = MemRefType.get([16, 16, 16], f16, memory_space=gm)
            ub_ty = MemRefType.get([16, 16, 16], f16, memory_space=vec)

            fn_ty = func.FunctionType.get([gm_ty, gm_ty], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_inject_sync_multibuf_pingpong_py", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                src, dst = entry.arguments

                c0 = arith.ConstantOp(idx, 0).result
                c1 = arith.ConstantOp(idx, 1).result
                c4 = arith.ConstantOp(idx, 4).result

                alloc = memref.AllocOp(ub_ty, [], [])
                ub = alloc.result
                alloc.operation.attributes["pto.multi_buffer"] = IntegerAttr.get(
                    i32, 2
                )

                # Loop-carried hazard:
                # - TLOAD writes to UB on PIPE_MTE2.
                # - TSTORE reads from UB on PIPE_MTE3.
                # With multi-buffer enabled, the compiler should materialize a
                # ping/pong selector and use dynamic event-id sync on the
                # back-edge dependency.
                loop = scf.ForOp(c0, c4, c1, [])
                with InsertionPoint(loop.body):
                    pto.TLoadOp(None, src, ub)
                    pto.TStoreOp(None, ub, dst)
                    scf.YieldOp([])

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())

