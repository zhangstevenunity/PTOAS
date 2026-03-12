#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from . import _pto_ops_gen as _pto_ops_gen
from ._pto_ops_gen import *
from mlir import ir as _ods_ir

def _load_local_pto_ext():
    import importlib.util
    from pathlib import Path
    lib_dir = Path(__file__).resolve().parent.parent / "_mlir_libs"
    for suffix in ("*.so", "*.pyd", "*.dll", "*.dylib"):
        for so_path in lib_dir.glob(f"_pto{suffix}"):
            spec = importlib.util.spec_from_file_location("mlir._mlir_libs._pto", so_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
    raise ImportError("cannot locate local _pto extension in _mlir_libs")


try:
    _pto_mod = _load_local_pto_ext()
except Exception:
    from .._mlir_libs import _pto as _pto_mod

register_dialect = _pto_mod.register_dialect
PtrType = _pto_mod.PtrType
TensorViewType = _pto_mod.TensorViewType
PartitionTensorViewType = _pto_mod.PartitionTensorViewType
TileType = _pto_mod.TileType
TileBufType = _pto_mod.TileBufType
TileBufArrayType = _pto_mod.TileBufArrayType
AddressSpace = _pto_mod.AddressSpace
AddressSpaceAttr = _pto_mod.AddressSpaceAttr
TileBufConfigAttr = _pto_mod.TileBufConfigAttr
BLayout = _pto_mod.BLayout
BLayoutAttr = _pto_mod.BLayoutAttr
SLayout = _pto_mod.SLayout
SLayoutAttr = _pto_mod.SLayoutAttr
PadValue = _pto_mod.PadValue
PadValueAttr = _pto_mod.PadValueAttr
RoundMode = _pto_mod.RoundMode
RoundModeAttr = _pto_mod.RoundModeAttr
CmpMode = _pto_mod.CmpMode
CmpModeAttr = _pto_mod.CmpModeAttr
PIPE = _pto_mod.PIPE
PipeAttr = _pto_mod.PipeAttr
Layout = _pto_mod.Layout
LayoutAttr = _pto_mod.LayoutAttr
SyncOpType = _pto_mod.SyncOpType
SyncOpTypeAttr = _pto_mod.SyncOpTypeAttr
EVENT = _pto_mod.EVENT
EventAttr = _pto_mod.EventAttr
MaskPattern = _pto_mod.MaskPattern
MaskPatternAttr = _pto_mod.MaskPatternAttr

__all__ = [
    # Dialect utilities
    "register_dialect",

    # Types
    "PtrType",
    "TensorViewType",
    "PartitionTensorViewType",
    "TileType",
    "TileBufType",
    "TileBufArrayType",
    "TileBufArray",
    "make_tile_buf_array",
    "tile_buf_array_get",
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
    "record_event", "wait_event", "barrier",
    # A5 buffer-id sync helpers
    "get_buf", "rls_buf",
    # Scalar pointer helpers
    "load_scalar", "store_scalar"

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

def _ensure_pipe_attr(val, ctx):
    if isinstance(val, PipeAttr):
        return val
    if isinstance(val, PIPE):
        return PipeAttr.get(val, ctx)
    if isinstance(val, str):
        name = val.upper()
        try:
            enum_val = getattr(PIPE, name)
        except AttributeError:
            raise ValueError(f"Unknown PIPE name: {val}")
        return PipeAttr.get(enum_val, ctx)
    return val

def _ensure_i32_attr(val, name, ctx):
    if isinstance(val, _ods_ir.IntegerAttr):
        return val
    if isinstance(val, int):
        i32 = _ods_ir.IntegerType.get_signless(32, ctx)
        return _ods_ir.IntegerAttr.get(i32, val)
    raise TypeError(f"{name} must be int or IntegerAttr, got {type(val).__name__}")

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
# A5 buffer-id sync helpers
# -----------------------------------------------------------------------------
def get_buf(op_type, buf_id, mode=0, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    if isinstance(op_type, (PipeAttr, PIPE)):
        raise TypeError("get_buf expects SyncOpType (or SyncOpTypeAttr), not PIPE/PipeAttr")
    attrs = {
        "op_type": _ensure_sync_attr(op_type, ctx),
        "buf_id": _ensure_i32_attr(buf_id, "buf_id", ctx),
        "mode": _ensure_i32_attr(mode, "mode", ctx),
    }
    return _ods_ir.Operation.create(
        "pto.get_buf",
        attributes=attrs,
        loc=loc,
        ip=ip,
    )


def rls_buf(op_type, buf_id, mode=0, *, loc=None, ip=None):
    ctx = loc.context if loc else _ods_ir.Context.current
    if isinstance(op_type, (PipeAttr, PIPE)):
        raise TypeError("rls_buf expects SyncOpType (or SyncOpTypeAttr), not PIPE/PipeAttr")
    attrs = {
        "op_type": _ensure_sync_attr(op_type, ctx),
        "buf_id": _ensure_i32_attr(buf_id, "buf_id", ctx),
        "mode": _ensure_i32_attr(mode, "mode", ctx),
    }
    return _ods_ir.Operation.create(
        "pto.rls_buf",
        attributes=attrs,
        loc=loc,
        ip=ip,
    )

# -----------------------------------------------------------------------------
# Scalar pointer helpers (manual wrappers until python ops are regenerated)
# -----------------------------------------------------------------------------
def load_scalar(result_type, ptr, offset, *, loc=None, ip=None):
    operands = [
        _pto_ops_gen._get_op_result_or_value(ptr),
        _pto_ops_gen._get_op_result_or_value(offset),
    ]
    op = _ods_ir.Operation.create(
        "pto.load_scalar",
        results=[result_type],
        operands=operands,
        loc=loc,
        ip=ip,
    )
    return op.results[0]


def store_scalar(ptr, offset, value, *, loc=None, ip=None):
    operands = [
        _pto_ops_gen._get_op_result_or_value(ptr),
        _pto_ops_gen._get_op_result_or_value(offset),
        _pto_ops_gen._get_op_result_or_value(value),
    ]
    return _ods_ir.Operation.create(
        "pto.store_scalar",
        operands=operands,
        loc=loc,
        ip=ip,
    )


# -----------------------------------------------------------------------------
# TileBufArray helpers (array-like ergonomics)
# -----------------------------------------------------------------------------
def _to_index_value(index, *, loc=None, ip=None):
    if isinstance(index, int):
        idx_ty = _ods_ir.IndexType.get()
        idx_attr = _ods_ir.IntegerAttr.get(idx_ty, index)
        cst = _ods_ir.Operation.create(
            "arith.constant",
            results=[idx_ty],
            attributes={"value": idx_attr},
            loc=loc,
            ip=ip,
        )
        return cst.results[0]
    return _pto_ops_gen._get_op_result_or_value(index)


def _to_tile_buf_array_value(array):
    if isinstance(array, TileBufArray):
        return array.value
    return _pto_ops_gen._get_op_result_or_value(array)


def make_tile_buf_array(elements, *, loc=None, ip=None):
    if not elements:
        raise ValueError("make_tile_buf_array expects at least one element")

    ops = [_pto_ops_gen._get_op_result_or_value(v) for v in elements]
    elem_ty = ops[0].type
    for i, v in enumerate(ops[1:], start=1):
        if v.type != elem_ty:
            raise ValueError(
                f"all elements must have the same type, got index 0={elem_ty} and index {i}={v.type}"
            )

    arr_ty = TileBufArrayType.get(len(ops), elem_ty, elem_ty.context)
    op = _ods_ir.Operation.create(
        "pto.make_tile_buf_array",
        results=[arr_ty],
        operands=ops,
        loc=loc,
        ip=ip,
    )
    return op.results[0]


def tile_buf_array_get(array, index, *, loc=None, ip=None):
    arr_val = _to_tile_buf_array_value(array)
    arr_ty = TileBufArrayType(arr_val.type)

    idx = index
    if isinstance(index, int) and index < 0:
        idx = int(arr_ty.size) + index
    idx_val = _to_index_value(idx, loc=loc, ip=ip)

    op = _ods_ir.Operation.create(
        "pto.tile_buf_array_get",
        results=[arr_ty.element_type],
        operands=[arr_val, idx_val],
        loc=loc,
        ip=ip,
    )
    return op.results[0]


class TileBufArray:
    def __init__(self, value):
        v = _pto_ops_gen._get_op_result_or_value(value)
        self._value = v
        self._type = TileBufArrayType(v.type)

    @classmethod
    def from_elements(cls, elements, *, loc=None, ip=None):
        return cls(make_tile_buf_array(elements, loc=loc, ip=ip))

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type

    @property
    def size(self):
        return int(self._type.size)

    @property
    def element_type(self):
        return self._type.element_type

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return tile_buf_array_get(self._value, index)

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

# -----------------------------------------------------------------------------
# Op aliases without "Op" suffix (user-facing)
# -----------------------------------------------------------------------------
def _install_op_aliases():
    added = []
    for name, obj in _pto_ops_gen.__dict__.items():
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, _ods_ir.OpView):
            continue
        alias = None
        if name.endswith("Op_DPS"):
            alias = f"{name[:-6]}_DPS"
        elif name.endswith("Op"):
            alias = name[:-2]
        if not alias or alias in globals():
            continue
        globals()[alias] = obj
        added.append(alias)
    return added

__all__.extend(_install_op_aliases())
