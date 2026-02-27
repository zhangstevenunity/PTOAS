#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, pto, arith
from mlir.dialects.pto import (
    TLOAD, TSTORE_VEC,
    TVEC,
    EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
    EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7,
)

def cidx(v):
    return arith.ConstantOp(IndexType.get(), v).result

def main():
    with Context() as ctx, Location.unknown():
        pto.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f = func.FuncOp("run_sync_high", func.FunctionType.get([], []))
        entry = f.add_entry_block()
        with InsertionPoint(entry):
            # NOTE(A5): SyncHigh is a regression testcase that stress-tests sync
            # primitives with high event IDs.
            #
            # On Ascend910B(A5), `set_flag/wait_flag` and `pipe_barrier` have
            # stricter PIPE constraints. In particular:
            # - `set_flag(PIPE_V, PIPE_V, ...)` / `wait_flag(PIPE_V, PIPE_V, ...)`
            #   is rejected by bisheng.
            # - `pipe_barrier(PIPE_V)` is rejected; pto-isa A5 testcases use
            #   `pipe_barrier(PIPE_ALL)` instead.
            #
            # Therefore this testcase only uses cross-pipe dependencies that
            # match pto-isa A5 patterns:
            #   MTE2 -> V -> MTE3
            pto.record_event(TLOAD, TVEC, EVENT_ID6)
            pto.wait_event(TLOAD, TVEC, EVENT_ID6)

            pto.record_event(TVEC, TSTORE_VEC, EVENT_ID7)
            pto.wait_event(TVEC, TSTORE_VEC, EVENT_ID7)

            pto.record_event(TLOAD, TVEC, EVENT_ID0)
            pto.wait_event(TLOAD, TVEC, EVENT_ID0)

            pto.record_event(TVEC, TSTORE_VEC, EVENT_ID1)
            pto.wait_event(TVEC, TSTORE_VEC, EVENT_ID1)

            pipe_all = pto.PipeAttr.get(pto.PIPE.PIPE_ALL, ctx)
            pto.barrier(pipe_all)
            func.ReturnOp([])
        print(module)

if __name__ == "__main__":
    main()
