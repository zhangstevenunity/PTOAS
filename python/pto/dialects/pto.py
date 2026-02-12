#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from . import _pto_ops_gen as _pto_ops_gen
from ._pto_ops_gen import *
from mlir import ir as _ods_ir

from .._mlir_libs._pto import (
    register_dialect,
    PtrType,
    TensorViewType,
    PartitionTensorViewType,
    TileType,
    TileBufType,
    AddressSpace,
    AddressSpaceAttr,
    TileBufConfigAttr,
    BLayout,
    BLayoutAttr,
    SLayout,
    SLayoutAttr,
    PadValue,
    PadValueAttr,
    RoundMode,
    RoundModeAttr,
    CmpMode,
    CmpModeAttr,
    PIPE,
    PipeAttr,
    Layout,
    LayoutAttr,
    SyncOpType,
    SyncOpTypeAttr,
    EVENT,
    EventAttr,
    MaskPattern,
    MaskPatternAttr,
)

__all__ = [
    # Dialect utilities
    "register_dialect",

    # Types
    "PtrType",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "AddressSpace", "AddressSpaceAttr",
    "BLayout","BLayoutAttr",
    "SLayout","SLayoutAttr",
    "PadValue","PadValueAttr",
    "RoundMode", "RoundModeAttr",
    "CmpMode", "CmpModeAttr",
    "PIPE", "PipeAttr",
    "Layout", "LayoutAttr",
    "SyncOpType", "SyncOpTypeAttr",
    "EVENT", "EventAttr",
    "MaskPattern", "MaskPatternAttr",
    "TileBufConfigAttr",
    "TileConfig",
    # High-level sync helpers
    "record_event", "wait_event", "barrier"

    # Aliases for SyncOpType enums (for terse calls)
    ,"TLOAD","TSTORE_ACC","TSTORE_VEC","TMOV_M2L","TMOV_M2S",
    "TMOV_M2B","TMOV_M2V","TMOV_V2M","TMATMUL","TVEC","TVECWAIT_EVENT"
    # Aliases for EVENT enums
    ,"EVENT_ID0","EVENT_ID1","EVENT_ID2","EVENT_ID3",
    "EVENT_ID4","EVENT_ID5","EVENT_ID6","EVENT_ID7"
]

# -----------------------------------------------------------------------------
# Convenience wrappers for high-level sync to allow passing enums directly
# -----------------------------------------------------------------------------

def _ensure_sync_attr(val, ctx):
    # Accept SyncOpType enum, SyncOpTypeAttr, or string name ("TMATMUL"/"tmatmul").
    if isinstance(val, SyncOpType):
        return SyncOpTypeAttr.get(val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(SyncOpType, name)
        except AttributeError:
            raise ValueError(f"Unknown SyncOpType name: {val}")
        return SyncOpTypeAttr.get(enum_val, ctx)
    return val

def _ensure_event_attr(val, ctx):
    if isinstance(val, EVENT):
        return EventAttr.get(val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(EVENT, name)
        except AttributeError:
            raise ValueError(f"Unknown EVENT name: {val}")
        return EventAttr.get(enum_val, ctx)
    return val

def record_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.record_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def wait_event(src_op, dst_op, event_id, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    return _pto_ops_gen.wait_event(
        _ensure_sync_attr(src_op, ctx),
        _ensure_sync_attr(dst_op, ctx),
        _ensure_event_attr(event_id, ctx),
        loc=loc, ip=ip)

def barrier(op, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    # If user passes SyncOpType/Attr, route to barrier_sync (maps to PIPE)
    if isinstance(op, (SyncOpType, SyncOpTypeAttr, str)):
        op_attr = _ensure_sync_attr(op, ctx)
        return _pto_ops_gen.barrier_sync(op_attr, loc=loc, ip=ip)
    # Otherwise fall back to low-level barrier expecting PipeAttr
    return _pto_ops_gen.barrier(op, loc=loc, ip=ip)

# -----------------------------------------------------------------------------
# Export enum aliases for terse calls: pto.record_event(TLOAD, TLOAD, EVENT_ID0)
# -----------------------------------------------------------------------------
TLOAD = SyncOpType.TLOAD
TSTORE_ACC = SyncOpType.TSTORE_ACC
TSTORE_VEC = SyncOpType.TSTORE_VEC
TMOV_M2L = SyncOpType.TMOV_M2L
TMOV_M2S = SyncOpType.TMOV_M2S
TMOV_M2B = SyncOpType.TMOV_M2B
TMOV_M2V = SyncOpType.TMOV_M2V
TMOV_V2M = SyncOpType.TMOV_V2M
TMATMUL = SyncOpType.TMATMUL
TVEC = SyncOpType.TVEC
TVECWAIT_EVENT = SyncOpType.TVECWAIT_EVENT

EVENT_ID0 = EVENT.EVENT_ID0
EVENT_ID1 = EVENT.EVENT_ID1
EVENT_ID2 = EVENT.EVENT_ID2
EVENT_ID3 = EVENT.EVENT_ID3
EVENT_ID4 = EVENT.EVENT_ID4
EVENT_ID5 = EVENT.EVENT_ID5
EVENT_ID6 = EVENT.EVENT_ID6
EVENT_ID7 = EVENT.EVENT_ID7

class TileConfig:
    alignedSize = 32
    fixedRowSize = 16
    fixedColSize = 16
    fixedMxRowSize = 16
    fixedMxColSize = 2
    fractalABSize = 512
    fractalCSize = 1024
    fractalMxSize = 32
