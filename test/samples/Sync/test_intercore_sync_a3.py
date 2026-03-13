#!/usr/bin/env python3
from mlir.ir import Context, InsertionPoint, IntegerType, Location, MemRefType, Module
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            module = Module.create()

            i64 = IntegerType.get_signless(64, ctx)
            # Minimal valid memref operand for pto.set_ffts verifier (i64 element).
            ffts_ty = MemRefType.get([1], i64)
            fn_ty = func.FunctionType.get([ffts_ty], [])

            with InsertionPoint(module.body):
                fn = func.FuncOp("test_intercore_sync_a3", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pipe_fix = pto.PipeAttr.get(pto.PIPE.PIPE_FIX, ctx)
                pipe_v = pto.PipeAttr.get(pto.PIPE.PIPE_V, ctx)
                pto.set_ffts(entry.arguments[0])
                pto.sync_set(pipe_fix, 3)
                pto.sync_wait(pipe_v, 3)
                func.ReturnOp([])

            module.operation.verify()
            return module


if __name__ == "__main__":
    print(build())
