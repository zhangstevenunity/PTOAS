# Pythonic frontend for the PTO dialect. Use: import pto_frontend as pto
# The underlying dialect is kept as _pto to avoid namespace clash in user code.

from mlir.ir import (
    Context,
    Location,
    InsertionPoint,
    Attribute,
    IndexType,
    IntegerType,
    F16Type,
    F32Type,
    StringAttr,
)
from mlir.dialects import pto as _pto
from mlir.dialects import func, arith, scf, builtin
from mlir.dialects.arith import CmpIPredicate


# -----------------------------------------------------------------------------
# Type helpers (call inside Context)
# -----------------------------------------------------------------------------

def float32():
    """Element type for float32. Call inside MLIR Context."""
    return F32Type.get()


def float16():
    """Element type for float16. Call inside MLIR Context."""
    return F16Type.get()


def tensor(dtype):
    """Pointer type to elements of dtype. E.g. ptr_out = pto.tensor(dtype=pto.float32())."""
    return _pto.PtrType.get(dtype)


def bool_():
    """1-bit signless integer type (i1). Used for predicate/bool in function args."""
    return IntegerType.get_signless(1)


# Alias so user can write pto.bool() without shadowing builtin bool
bool = bool_


# -----------------------------------------------------------------------------
# Index constants and arithmetic (call inside InsertionPoint)
# -----------------------------------------------------------------------------

def const(v: int):
    """Index constant. E.g. c0 = pto.const(0)."""
    return arith.ConstantOp(IndexType.get(), v).result


# TODO: overload __mul__ on index values so that i * cBASEK can be written as mul(i, cBASEK)
def mul(a, b):
    """Index multiplication. E.g. kOff = pto.mul(i, cBASEK)."""
    return arith.MulIOp(a, b).result


# TODO: overload __eq__ on index values so that i == c0 can be written as eq(i, c0)
def eq(a, b):
    """Index equality comparison. E.g. is_i0 = pto.eq(i, c0)."""
    return arith.CmpIOp(CmpIPredicate.eq, a, b).result


# -----------------------------------------------------------------------------
# Tensor view (call inside InsertionPoint; ptr is an SSA value)
# -----------------------------------------------------------------------------

def tensor_view(ptr, dims_and_strides, dtype=None):
    """
    Create a rank-2 tensor view from a pointer.
    dims_and_strides: ((dim0, dim1), (stride0, stride1))
    dtype: optional element type (e.g. pto.float32()); required if ptr.type doesn't expose element_type.
    E.g. tvA = pto.tensor_view(a_ptr, ((cM, cK), (cK, c1)), dtype=pto.float32())
    """
    dims, strides = dims_and_strides[0], dims_and_strides[1]
    rank = len(dims)
    elem_type = dtype if dtype is not None else getattr(ptr.type, "element_type", None)
    if elem_type is None:
        raise ValueError(
            "tensor_view: could not get element type from ptr (type has no element_type); pass dtype=..."
        )
    tv_type = _pto.TensorViewType.get(rank, elem_type)
    return _pto.MakeTensorViewOp(tv_type, ptr, list(dims), list(strides)).result


def subview(tv, tile_shape, offsets, sizes, dtype=None):
    """
    Subview of a tensor view to a tile view.
    tile_shape: (M, N) or [M, N]; offsets and sizes: tuples/lists of index values.
    dtype: optional element type; required if tv.type doesn't expose element_type.
    E.g. svA = pto.subview(tvA, (M, BASEK), (c0, kOff), (cTileM, cBASEK))
    """
    tile_shape = list(tile_shape)
    offsets = list(offsets)
    sizes = list(sizes)
    elem_type = dtype if dtype is not None else getattr(tv.type, "element_type", None)
    if elem_type is None:
        raise ValueError(
            "subview: could not get element type from tensor view; pass dtype=..."
        )
    tile_view_type = _pto.TileViewType.get(tile_shape, elem_type)
    return _pto.SubviewOp(tile_view_type, tv, offsets, sizes).result


# -----------------------------------------------------------------------------
# Tile buffer config and allocation (call inside InsertionPoint)
# -----------------------------------------------------------------------------

def _tile_config(buf_type, s_fractal_ab, s_fractal_c):
    """Build TileBufConfigAttr for the given buf_type string."""
    pad = _pto.PadValueAttr.get(_pto.PadValue.Null)
    if buf_type == "mat":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.ColMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            s_fractal_ab,
            pad,
        )
    if buf_type == "mat_bias":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.NoneBox),
            s_fractal_ab,
            pad,
        )
    if buf_type == "left":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            s_fractal_ab,
            pad,
        )
    if buf_type == "right":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.ColMajor),
            s_fractal_ab,
            pad,
        )
    if buf_type == "acc":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.ColMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.RowMajor),
            s_fractal_c,
            pad,
        )
    if buf_type == "bias":
        return _pto.TileBufConfigAttr.get(
            _pto.BLayoutAttr.get(_pto.BLayout.RowMajor),
            _pto.SLayoutAttr.get(_pto.SLayout.NoneBox),
            512,
            pad,
        )
    raise ValueError(f"Unknown buf_type: {buf_type}")


def _address_space(buf_type):
    """Map buf_type string to AddressSpaceAttr."""
    space_map = {
        "mat": _pto.AddressSpace.MAT,
        "mat_bias": _pto.AddressSpace.MAT,
        "left": _pto.AddressSpace.LEFT,
        "right": _pto.AddressSpace.RIGHT,
        "acc": _pto.AddressSpace.ACC,
        "bias": _pto.AddressSpace.BIAS,
    }
    if buf_type not in space_map:
        raise ValueError(f"Unknown buf_type: {buf_type}")
    return _pto.AddressSpaceAttr.get(space_map[buf_type])


def tile(
    shape,
    dtype,
    buf_type,
    *,
    s_fractal_ab=512,
    s_fractal_c=1024,
):
    """
    Allocate a tile buffer. shape is [M, N] or (M, N); buf_type one of
    "mat", "mat_bias", "left", "right", "acc", "bias".
    E.g. aMatTile = pto.tile([M, BASEK], dtype=pto.float32(), buf_type="mat")
    """
    shape = list(shape)
    cfg = _tile_config(buf_type, s_fractal_ab, s_fractal_c)
    addr = _address_space(buf_type)
    tile_buf_type = _pto.TileBufType.get(shape, dtype, addr, shape, cfg)
    return _pto.AllocTileOp(tile_buf_type).result


def tile_buf(shape, dtype, buf_type, *, s_fractal_ab=512, s_fractal_c=1024):
    """Alias for tile(). E.g. cTile = pto.tile_buf([M, N], dtype=pto.float32(), buf_type="acc")."""
    return tile(shape, dtype, buf_type, s_fractal_ab=s_fractal_ab, s_fractal_c=s_fractal_c)


# -----------------------------------------------------------------------------
# Pipeline/event attributes (call inside Context; Attribute.parse needs context)
# -----------------------------------------------------------------------------

def _parse_attr(s: str, what: str):
    try:
        return Attribute.parse(s)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {what} attr from: {s}\nError: {e}")


def EVENT_ID0():
    """Pipeline event attribute. Call inside Context."""
    return _parse_attr("#pto.event<EVENT_ID0>", "EVENT_ID0")


def PIPE_MTE2():
    return _parse_attr("#pto.pipe<PIPE_MTE2>", "PIPE_MTE2")


def PIPE_MTE1():
    return _parse_attr("#pto.pipe<PIPE_MTE1>", "PIPE_MTE1")


def PIPE_M():
    return _parse_attr("#pto.pipe<PIPE_M>", "PIPE_M")


def PIPE_FIX():
    return _parse_attr("#pto.pipe<PIPE_FIX>", "PIPE_FIX")


# -----------------------------------------------------------------------------
# Ops (call inside InsertionPoint)
# -----------------------------------------------------------------------------

def load(subview_val, tile_val):
    """TLoad: load from subview into tile. E.g. pto.load(svA, aMatTile)."""
    _pto.TLoadOp(None, subview_val, tile_val)


def store(tile_val, subview_val):
    """TStore: store tile to subview. E.g. pto.store(cTile, svOut)."""
    _pto.TStoreOp(None, tile_val, subview_val)


def set_flag(pipe_from, pipe_to, event):
    """SetFlag. E.g. pto.set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)."""
    _pto.SetFlagOp(pipe_from, pipe_to, event)


def wait_flag(pipe_from, pipe_to, event):
    """WaitFlag. E.g. pto.wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0)."""
    _pto.WaitFlagOp(pipe_from, pipe_to, event)


def move(src_tile, dst_tile):
    """TMov: copy tile to tile. E.g. pto.move(aMatTile, aTile)."""
    _pto.TMovOp(None, src_tile, dst_tile)


def matmul_bias(a_tile, b_tile, bias_tile, c_tile):
    """TMatmulBias: C = A*B + bias (clears acc first)."""
    _pto.TMatmulBiasOp(None, a_tile, b_tile, bias_tile, c_tile)


def matmul(a_tile, b_tile, c_tile):
    """TMatmul: C = A*B (clears acc first)."""
    _pto.TMatmulOp(None, a_tile, b_tile, c_tile)


def matmul_acc(c_tile, a_tile, b_tile, out_tile):
    """TMatmulAcc: out += A*B (accumulate)."""
    _pto.TMatmulAccOp(None, c_tile, a_tile, b_tile, out_tile)


# -----------------------------------------------------------------------------
# Dialect registration (pass-through)
# -----------------------------------------------------------------------------

def register_dialect(ctx, load=True):
    """Register PTO dialect with context. Call at start of build()."""
    return _pto.register_dialect(ctx, load=load)
