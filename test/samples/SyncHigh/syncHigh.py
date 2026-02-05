#!/usr/bin/env python3
from mlir.ir import Context, Location, Module, InsertionPoint, IndexType
from mlir.dialects import func, pto, arith
from mlir.dialects.pto import (
    TLOAD, TSTORE_ACC, TSTORE_VEC,
    TMOV_M2L, TMOV_M2S, TMOV_M2B, TMOV_M2V, TMOV_V2M,
    TMATMUL, TVEC, TVECWAIT_EVENT,
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
            # Unrolled coverage for each SyncOpType (record + wait)
            # Use string names to exercise helper auto-conversion.
            pto.record_event(TLOAD,       TLOAD,       EVENT_ID0)
            pto.wait_event  (TLOAD,       TLOAD,       EVENT_ID0)

            pto.record_event(TSTORE_ACC,  TSTORE_ACC,  EVENT_ID1)
            pto.wait_event  (TSTORE_ACC,  TSTORE_ACC,  EVENT_ID1)

            pto.record_event(TSTORE_VEC,  TSTORE_VEC,  EVENT_ID2)
            pto.wait_event  (TSTORE_VEC,  TSTORE_VEC,  EVENT_ID2)

            pto.record_event(TMOV_M2L,    TMOV_M2L,    EVENT_ID3)
            pto.wait_event  (TMOV_M2L,    TMOV_M2L,    EVENT_ID3)

            pto.record_event(TMOV_M2S,    TMOV_M2S,    EVENT_ID4)
            pto.wait_event  (TMOV_M2S,    TMOV_M2S,    EVENT_ID4)

            pto.record_event(TMOV_M2B,    TMOV_M2B,    EVENT_ID5)
            pto.wait_event  (TMOV_M2B,    TMOV_M2B,    EVENT_ID5)

            pto.record_event(TMOV_M2V,    TMOV_M2V,    EVENT_ID6)
            pto.wait_event  (TMOV_M2V,    TMOV_M2V,    EVENT_ID6)

            pto.record_event(TMOV_V2M,    TMOV_V2M,    EVENT_ID7)
            pto.wait_event  (TMOV_V2M,    TMOV_V2M,    EVENT_ID7)

            pto.record_event(TMATMUL,     TMATMUL,     EVENT_ID0)
            pto.wait_event  (TMATMUL,     TMATMUL,     EVENT_ID0)

            pto.record_event(TVEC,        TVEC,        EVENT_ID1)
            pto.wait_event  (TVEC,        TVEC,        EVENT_ID1)

            pto.record_event(TVECWAIT_EVENT, TVECWAIT_EVENT, EVENT_ID2)
            pto.wait_event  (TVECWAIT_EVENT, TVECWAIT_EVENT, EVENT_ID2)

            # Barrier coverage for TMATMUL and TVEC
            pto.barrier(TMATMUL)
            pto.barrier(TVEC)
            func.ReturnOp([])
        print(module)

if __name__ == "__main__":
    main()
