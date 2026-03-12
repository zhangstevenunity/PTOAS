#!/usr/bin/python3
# coding=utf-8

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


SEED = 19
ROWS = 32
COLS = 32

_HOST_TYPE_TO_NP = {
    "aclFloat16": np.float16,
    "bfloat16_t": np.uint16,
    "bool": np.bool_,
    "double": np.float64,
    "float": np.float32,
    "half": np.float16,
    "int": np.int32,
    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "size_t": np.uint64,
    "uint8_t": np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,
    "unsigned": np.uint32,
}


@dataclass
class CaseMeta:
    elem_counts: Dict[str, int]
    np_types: Dict[str, np.dtype]
    read_order: List[str]
    outputs: List[str]

    @property
    def inputs(self) -> List[str]:
        return [name for name in self.read_order if name not in self.outputs]


def _host_type_to_np(host_type: str) -> np.dtype:
    host_type = host_type.strip()
    if host_type not in _HOST_TYPE_TO_NP:
        raise KeyError(f"unsupported host type: {host_type}")
    return np.dtype(_HOST_TYPE_TO_NP[host_type])


def load_case_meta(main_cpp: str = "main.cpp", outputs_txt: str = "outputs.txt") -> CaseMeta:
    text = Path(main_cpp).read_text(encoding="utf-8")
    elem_counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r"size_t\s+elemCount_(\w+)\s*=\s*(\d+);", text)
    }
    np_types = {
        match.group(1): _host_type_to_np(match.group(2))
        for match in re.finditer(
            r"size_t\s+fileSize_(\w+)\s*=\s*elemCount_\1\s*\*\s*sizeof\(([^)]+)\);",
            text,
        )
    }
    read_order = re.findall(r'ReadFile\("\./([^"]+)\.bin"', text)
    if Path(outputs_txt).is_file():
        outputs = [line.strip() for line in Path(outputs_txt).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        outputs = []
    return CaseMeta(elem_counts=elem_counts, np_types=np_types, read_order=read_order, outputs=outputs)


def _rng():
    return np.random.default_rng(SEED)


def _as_matrix(arr: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    flat = np.asarray(arr).reshape(-1)
    expected = rows * cols
    if flat.size != expected:
        raise ValueError(f"expected {expected} elements, got {flat.size}")
    return flat.reshape(rows, cols)


def _float_values(rng, count: int, *, style: str) -> np.ndarray:
    if style == "signed":
        arr = rng.uniform(-3.0, 3.0, size=count).astype(np.float32)
    elif style == "signed_small":
        arr = rng.uniform(-1.5, 1.5, size=count).astype(np.float32)
    elif style == "nonzero_signed":
        arr = rng.uniform(-3.0, 3.0, size=count).astype(np.float32)
        mask = np.abs(arr) < np.float32(0.25)
        arr[mask] = np.where(arr[mask] >= 0.0, np.float32(0.25), np.float32(-0.25))
    elif style == "positive":
        arr = rng.uniform(0.25, 4.0, size=count).astype(np.float32)
    elif style == "exp":
        arr = rng.uniform(-2.0, 2.0, size=count).astype(np.float32)
    elif style == "cmp":
        arr = rng.uniform(-2.0, 2.0, size=count).astype(np.float32)
    else:
        raise ValueError(f"unsupported float style: {style}")
    return arr


def _int_values(rng, count: int, dtype: np.dtype, *, style: str) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int16):
        if style == "bitwise":
            vals = rng.integers(-256, 256, size=count, dtype=np.int32)
        else:
            raise ValueError(f"unsupported int16 style: {style}")
    elif dtype == np.dtype(np.int32):
        if style == "bitwise":
            vals = rng.integers(-256, 256, size=count, dtype=np.int32)
        elif style == "shift_small":
            vals = rng.integers(0, 4, size=count, dtype=np.int32)
        else:
            raise ValueError(f"unsupported int32 style: {style}")
    else:
        raise ValueError(f"unsupported dtype/style pair: {dtype}/{style}")
    return vals.astype(dtype, copy=False)


def _packed_row_bytes(cols: int) -> int:
    return ((cols + 63) // 64) * 8


def pack_predicate_mask(bits: np.ndarray, *, storage_cols: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.bool_)
    if bits.ndim != 2:
        raise ValueError("mask bits must be a 2D array")
    rows, cols = bits.shape
    row_bytes = _packed_row_bytes(cols)
    if storage_cols < row_bytes:
        raise ValueError(f"storage_cols={storage_cols} is too small for cols={cols}")
    out = np.zeros((rows, storage_cols), dtype=np.uint8)
    for row in range(rows):
        for word_idx, base_col in enumerate(range(0, cols, 64)):
            width = min(64, cols - base_col)
            word = 0
            for bit_idx in range(width):
                if bits[row, base_col + bit_idx]:
                    word |= 1 << bit_idx
            out[row, word_idx * 8:(word_idx + 1) * 8] = np.frombuffer(word.to_bytes(8, "little"), dtype=np.uint8)
    return out.reshape(-1)


def unpack_predicate_mask(buf: np.ndarray, *, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    buf = np.asarray(buf, dtype=np.uint8).reshape(-1)
    if rows <= 0 or cols <= 0:
        raise ValueError("rows/cols must be positive")
    if buf.size % rows != 0:
        raise ValueError(f"mask buffer size {buf.size} is not divisible by rows={rows}")
    storage_cols = buf.size // rows
    row_bytes = _packed_row_bytes(cols)
    if storage_cols < row_bytes:
        raise ValueError(f"storage_cols={storage_cols} is too small for cols={cols}")
    packed = buf.reshape(rows, storage_cols)
    bits = np.zeros((rows, cols), dtype=np.bool_)
    for row in range(rows):
        for word_idx, base_col in enumerate(range(0, cols, 64)):
            word = int.from_bytes(bytes(packed[row, word_idx * 8:(word_idx + 1) * 8]), "little")
            width = min(64, cols - base_col)
            for bit_idx in range(width):
                bits[row, base_col + bit_idx] = ((word >> bit_idx) & 1) != 0
    return bits


def _zero_buffer(meta: CaseMeta, name: str) -> np.ndarray:
    return np.zeros(meta.elem_counts[name], dtype=meta.np_types[name])


def _default_buffers(meta: CaseMeta) -> Dict[str, np.ndarray]:
    return {name: _zero_buffer(meta, name) for name in meta.read_order}


def _write_buffers(meta: CaseMeta, buffers: Dict[str, np.ndarray]):
    for name in meta.read_order:
        if name not in buffers:
            raise KeyError(f"missing buffer for {name}")
        arr = np.asarray(buffers[name], dtype=meta.np_types[name]).reshape(-1)
        expected = meta.elem_counts[name]
        if arr.size != expected:
            raise ValueError(f"{name}: expected {expected} elements, got {arr.size}")
        arr.tofile(f"{name}.bin")


def _write_golden(meta: CaseMeta, outputs: Dict[str, np.ndarray]):
    for name in meta.outputs:
        if name not in outputs:
            raise KeyError(f"missing golden for {name}")
        arr = np.asarray(outputs[name], dtype=meta.np_types[name]).reshape(-1)
        expected = meta.elem_counts[name]
        if arr.size != expected:
            raise ValueError(f"{name}: expected {expected} golden elements, got {arr.size}")
        arr.tofile(f"golden_{name}.bin")


def _single_output(meta: CaseMeta) -> str:
    if len(meta.outputs) != 1:
        raise ValueError(f"expected exactly one output, got {meta.outputs}")
    return meta.outputs[0]


def generate_binary_float_case(op: str):
    meta = load_case_meta()
    if len(meta.inputs) != 2:
        raise ValueError(f"{op}: expected 2 inputs, got {meta.inputs}")
    rng = _rng()
    lhs_name, rhs_name = meta.inputs
    lhs = _float_values(rng, meta.elem_counts[lhs_name], style="signed")
    rhs_style = "nonzero_signed" if op in {"div", "rem"} else "signed"
    rhs = _float_values(rng, meta.elem_counts[rhs_name], style=rhs_style)
    buffers = _default_buffers(meta)
    buffers[lhs_name] = lhs
    buffers[rhs_name] = rhs
    _write_buffers(meta, buffers)

    if op == "add":
        out = lhs + rhs
    elif op == "sub":
        out = lhs - rhs
    elif op == "mul":
        out = lhs * rhs
    elif op == "div":
        out = lhs / rhs
    elif op == "max":
        out = np.maximum(lhs, rhs)
    elif op == "min":
        out = np.minimum(lhs, rhs)
    elif op == "rem":
        out = np.fmod(lhs, rhs)
    else:
        raise ValueError(f"unsupported binary float op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_scalar_float_case(op: str, scalar: float, *, scalar_left: bool = False):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    style = "positive" if op in {"log", "sqrt", "rsqrt", "recip"} else "signed"
    if op == "exp":
        style = "exp"
    if op == "cmps":
        style = "cmp"
    if op in {"divs", "rems"}:
        style = "signed"
    src = _float_values(rng, meta.elem_counts[src_name], style=style if op != "divs2" else "nonzero_signed")
    if op in {"divs", "rems"}:
        src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    if op in {"log", "sqrt", "rsqrt", "recip"}:
        src = _float_values(rng, meta.elem_counts[src_name], style="positive")
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)

    if op == "adds":
        out = src + np.float32(scalar)
    elif op == "subs":
        out = src - np.float32(scalar)
    elif op == "muls":
        out = src * np.float32(scalar)
    elif op == "divs":
        out = np.float32(scalar) / src if scalar_left else src / np.float32(scalar)
    elif op == "maxs":
        out = np.maximum(src, np.float32(scalar))
    elif op == "mins":
        out = np.minimum(src, np.float32(scalar))
    elif op == "rems":
        out = np.fmod(src, np.float32(scalar))
    elif op == "lrelu":
        out = np.where(src > 0.0, src, src * np.float32(scalar))
    elif op == "exp":
        out = np.exp(src)
    elif op == "log":
        out = np.log(src)
    elif op == "sqrt":
        out = np.sqrt(src)
    elif op == "rsqrt":
        out = 1.0 / np.sqrt(src)
    elif op == "recip":
        out = 1.0 / src
    else:
        raise ValueError(f"unsupported scalar/unary float op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_unary_float_case(op: str):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    style = "signed"
    if op in {"exp"}:
        style = "exp"
    elif op in {"log", "sqrt", "rsqrt", "recip"}:
        style = "positive"
    src = _float_values(rng, meta.elem_counts[src_name], style=style)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)

    if op == "abs":
        out = np.abs(src)
    elif op == "neg":
        out = -src
    elif op == "exp":
        out = np.exp(src)
    elif op == "log":
        out = np.log(src)
    elif op == "sqrt":
        out = np.sqrt(src)
    elif op == "rsqrt":
        out = 1.0 / np.sqrt(src)
    elif op == "recip":
        out = 1.0 / src
    elif op == "relu":
        out = np.maximum(src, np.float32(0.0))
    else:
        raise ValueError(f"unsupported unary float op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_prelu_case():
    meta = load_case_meta()
    if len(meta.inputs) != 2:
        raise ValueError(f"prelu: expected 2 inputs, got {meta.inputs}")
    rng = _rng()
    src_name, slope_name = meta.inputs
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    slope = _float_values(rng, meta.elem_counts[slope_name], style="signed_small")
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    buffers[slope_name] = slope
    _write_buffers(meta, buffers)
    out = np.where(src > 0.0, src, src * slope)
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_addc_case(op: str):
    meta = load_case_meta()
    if len(meta.inputs) != 3:
        raise ValueError(f"{op}: expected 3 inputs, got {meta.inputs}")
    rng = _rng()
    a_name, b_name, c_name = meta.inputs
    a = _float_values(rng, meta.elem_counts[a_name], style="signed")
    b = _float_values(rng, meta.elem_counts[b_name], style="signed")
    c = _float_values(rng, meta.elem_counts[c_name], style="signed_small")
    buffers = _default_buffers(meta)
    buffers[a_name] = a
    buffers[b_name] = b
    buffers[c_name] = c
    _write_buffers(meta, buffers)
    if op == "addc":
        out = a + b + c
    elif op == "subc":
        out = a - b + c
    else:
        raise ValueError(f"unsupported carry op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_scalar_carry_case(op: str, scalar: float):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    if op == "addsc":
        out = src + np.float32(scalar) + src
    elif op == "subsc":
        out = src - np.float32(scalar) + src
    else:
        raise ValueError(f"unsupported scalar carry op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_row_reduce_case(op: str):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    src_m = _as_matrix(src)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    if op == "rowsum":
        out = src_m.sum(axis=1, dtype=np.float32)
    elif op == "rowmax":
        out = src_m.max(axis=1)
    elif op == "rowmin":
        out = src_m.min(axis=1)
    else:
        raise ValueError(f"unsupported row reduction op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_col_reduce_case(op: str, *, accumulate: bool = False):
    meta = load_case_meta()
    if op == "colsum":
        if len(meta.inputs) != 2:
            raise ValueError(f"{op}: expected 2 non-output inputs, got {meta.inputs}")
        src_name, tmp_name = meta.inputs
    else:
        if len(meta.inputs) != 1:
            raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
        src_name = meta.inputs[0]
        tmp_name = None
    rng = _rng()
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    src_m = _as_matrix(src)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    if tmp_name is not None:
        buffers[tmp_name] = np.zeros(meta.elem_counts[tmp_name], dtype=meta.np_types[tmp_name])
    out_name = _single_output(meta)
    out_init = np.zeros(meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    if accumulate:
        out_init = _float_values(rng, meta.elem_counts[out_name], style="signed_small")
        buffers[out_name] = out_init
    _write_buffers(meta, buffers)
    if op == "colsum":
        out = src_m.sum(axis=0, dtype=np.float32)
        if accumulate:
            out = out + out_init
    elif op == "colmax":
        out = src_m.max(axis=0)
    elif op == "colmin":
        out = src_m.min(axis=0)
    else:
        raise ValueError(f"unsupported col reduction op: {op}")
    _write_golden(meta, {out_name: out.astype(np.float32)})


def generate_rowexpand_case():
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"rowexpand: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    src_m = _as_matrix(src)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    out = np.repeat(src_m[:, :1], COLS, axis=1)
    _write_golden(meta, {_single_output(meta): out.astype(np.float32).reshape(-1)})


def generate_colexpand_case():
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"colexpand: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    src = _float_values(rng, meta.elem_counts[src_name], style="signed")
    src_m = _as_matrix(src)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    out = np.repeat(src_m[:1, :], ROWS, axis=0)
    _write_golden(meta, {_single_output(meta): out.astype(np.float32).reshape(-1)})


def generate_rowexpand_bin_case(op: str):
    meta = load_case_meta()
    if len(meta.inputs) != 2:
        raise ValueError(f"{op}: expected 2 inputs, got {meta.inputs}")
    rng = _rng()
    src0_name, src1_name = meta.inputs
    src0 = _float_values(rng, meta.elem_counts[src0_name], style="signed")
    src1 = _float_values(rng, meta.elem_counts[src1_name], style="nonzero_signed" if op == "rowexpanddiv" else "signed")
    src0_m = _as_matrix(src0)
    src1_m = _as_matrix(src1)
    row_scalars = src1_m.reshape(-1)[:ROWS].astype(np.float32)
    buffers = _default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    _write_buffers(meta, buffers)
    if op == "rowexpandmul":
        out = src0_m * row_scalars[:, None]
    elif op == "rowexpanddiv":
        out = src0_m / row_scalars[:, None]
    elif op == "rowexpandsub":
        out = src0_m - row_scalars[:, None]
    else:
        raise ValueError(f"unsupported rowexpand binary op: {op}")
    _write_golden(meta, {_single_output(meta): out.astype(np.float32).reshape(-1)})


def generate_expands_case(scalar: float):
    meta = load_case_meta()
    buffers = _default_buffers(meta)
    _write_buffers(meta, buffers)
    out_name = _single_output(meta)
    out = np.full(meta.elem_counts[out_name], np.float32(scalar), dtype=np.float32)
    _write_golden(meta, {out_name: out})


def generate_cmp_case(op: str, *, scalar: float = 0.0):
    meta = load_case_meta()
    rng = _rng()
    if op == "cmp":
        if len(meta.inputs) != 2:
            raise ValueError(f"{op}: expected 2 inputs, got {meta.inputs}")
        src0_name, src1_name = meta.inputs
        src0 = _float_values(rng, meta.elem_counts[src0_name], style="cmp")
        src1 = _float_values(rng, meta.elem_counts[src1_name], style="cmp")
        pred = _as_matrix(src0) < _as_matrix(src1)
    elif op == "cmps":
        if len(meta.inputs) != 1:
            raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
        src0_name = meta.inputs[0]
        src0 = _float_values(rng, meta.elem_counts[src0_name], style="cmp")
        src1_name = None
        src1 = None
        pred = _as_matrix(src0) > np.float32(scalar)
    else:
        raise ValueError(f"unsupported compare op: {op}")
    buffers = _default_buffers(meta)
    buffers[src0_name] = src0
    if src1 is not None and src1_name is not None:
        buffers[src1_name] = src1
    _write_buffers(meta, buffers)
    out_name = _single_output(meta)
    if meta.elem_counts[out_name] % ROWS != 0:
        raise ValueError(f"{out_name}: cannot derive mask storage stride from count={meta.elem_counts[out_name]}")
    storage_cols = meta.elem_counts[out_name] // ROWS
    packed = pack_predicate_mask(pred, storage_cols=storage_cols)
    _write_golden(meta, {out_name: packed})


def generate_sel_case():
    meta = load_case_meta()
    if len(meta.inputs) != 3:
        raise ValueError(f"sel: expected 3 inputs, got {meta.inputs}")
    rng = _rng()
    mask_name, src0_name, src1_name = meta.inputs
    storage_cols = meta.elem_counts[mask_name] // ROWS
    mask_bits = rng.integers(0, 2, size=(ROWS, COLS), dtype=np.uint8).astype(np.bool_)
    mask = pack_predicate_mask(mask_bits, storage_cols=storage_cols)
    src0 = _float_values(rng, meta.elem_counts[src0_name], style="signed")
    src1 = _float_values(rng, meta.elem_counts[src1_name], style="signed")
    buffers = _default_buffers(meta)
    buffers[mask_name] = mask
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    _write_buffers(meta, buffers)
    out = np.where(mask_bits, _as_matrix(src0), _as_matrix(src1))
    _write_golden(meta, {_single_output(meta): out.astype(np.float32).reshape(-1)})


def generate_sels_case(select_mode: int):
    meta = load_case_meta()
    if len(meta.inputs) != 2:
        raise ValueError(f"sels: expected 2 inputs, got {meta.inputs}")
    rng = _rng()
    src0_name, src1_name = meta.inputs
    src0 = _float_values(rng, meta.elem_counts[src0_name], style="signed")
    src1 = _float_values(rng, meta.elem_counts[src1_name], style="signed")
    buffers = _default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    _write_buffers(meta, buffers)
    out = src0 if int(select_mode) == 1 else src1
    _write_golden(meta, {_single_output(meta): out.astype(np.float32)})


def generate_bitwise_self_case(op: str, dtype: np.dtype):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    style = "shift_small" if op in {"shl", "shr"} else "bitwise"
    src = _int_values(rng, meta.elem_counts[src_name], dtype, style=style)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    if op == "and":
        out = np.bitwise_and(src, src)
    elif op == "or":
        out = np.bitwise_or(src, src)
    elif op == "xor":
        out = np.bitwise_xor(src, src)
    elif op == "shl":
        out = np.left_shift(src, src)
    elif op == "shr":
        out = np.right_shift(src, src)
    elif op == "not":
        out = np.bitwise_not(src)
    else:
        raise ValueError(f"unsupported bitwise op: {op}")
    _write_golden(meta, {_single_output(meta): np.asarray(out, dtype=dtype)})


def generate_bitwise_scalar_case(op: str, scalar: int, dtype: np.dtype):
    meta = load_case_meta()
    if len(meta.inputs) != 1:
        raise ValueError(f"{op}: expected 1 input, got {meta.inputs}")
    rng = _rng()
    src_name = meta.inputs[0]
    style = "shift_small" if op in {"shls", "shrs"} else "bitwise"
    src = _int_values(rng, meta.elem_counts[src_name], dtype, style=style)
    buffers = _default_buffers(meta)
    buffers[src_name] = src
    _write_buffers(meta, buffers)
    scalar = np.asarray(scalar, dtype=dtype).item()
    if op == "ands":
        out = np.bitwise_and(src, scalar)
    elif op == "ors":
        out = np.bitwise_or(src, scalar)
    elif op == "xors":
        out = np.bitwise_xor(src, scalar)
    elif op == "shls":
        out = np.left_shift(src, scalar)
    elif op == "shrs":
        out = np.right_shift(src, scalar)
    else:
        raise ValueError(f"unsupported scalar bitwise op: {op}")
    _write_golden(meta, {_single_output(meta): np.asarray(out, dtype=dtype)})


def compare_bin(golden_path, output_path, dtype, eps):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    dtype_np = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype_np)
    output = np.fromfile(output_path, dtype=dtype_np)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps, equal_nan=True):
        if golden.size:
            if np.issubdtype(dtype_np, np.floating):
                golden_cmp = golden.astype(np.float64, copy=False)
                output_cmp = output.astype(np.float64, copy=False)
            elif np.issubdtype(dtype_np, np.integer) or np.issubdtype(dtype_np, np.unsignedinteger):
                golden_cmp = golden.astype(np.int64, copy=False)
                output_cmp = output.astype(np.int64, copy=False)
            else:
                golden_cmp = golden.astype(np.float64, copy=False)
                output_cmp = output.astype(np.float64, copy=False)
            abs_diff = np.abs(golden_cmp - output_cmp)
            idx = int(np.argmax(abs_diff))
            diff = float(abs_diff[idx])
            print(
                f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff} at idx={idx} "
                f"(golden={golden_cmp[idx]}, out={output_cmp[idx]}, dtype={dtype_np})"
            )
        else:
            print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers, dtype={dtype_np}")
        return False
    return True


def compare_packed_pred_mask(golden_path, output_path, rows, cols):
    if not os.path.exists(output_path):
        print(f"[ERROR] Output missing: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden missing: {golden_path}")
        return False
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    need = int(rows) * int(cols)
    if golden.size < need or output.size < need:
        print(
            f"[ERROR] Packed mask buffer too small: need={need} bytes, "
            f"golden={golden.size}, out={output.size}"
        )
        return False
    golden = golden[:need].reshape(rows, cols)
    output = output[:need].reshape(rows, cols)
    row_bytes = min(_packed_row_bytes(cols), cols)
    golden_sel = golden[:, :row_bytes].reshape(-1)
    output_sel = output[:, :row_bytes].reshape(-1)
    if not np.array_equal(golden_sel, output_sel):
        diff = np.nonzero(golden_sel != output_sel)[0]
        idx = int(diff[0]) if diff.size else 0
        print(
            f"[ERROR] Mismatch (packed mask): {golden_path} vs {output_path}, first diff at idx={idx} "
            f"(golden={int(golden_sel[idx])}, out={int(output_sel[idx])}, rows={rows}, cols={cols}, row_bytes={row_bytes})"
        )
        return False
    return True


def compare_all_outputs(dtype, eps):
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        ok = compare_bin(f"golden_{name}.bin", f"{name}.bin", dtype, eps) and ok
    return finalize_compare(ok)


def compare_all_packed_mask_outputs(rows: int = ROWS, cols: int = COLS):
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        ok = compare_packed_pred_mask(f"golden_{name}.bin", f"{name}.bin", rows, cols) and ok
    return finalize_compare(ok)


def finalize_compare(ok: bool):
    strict = os.getenv("COMPARE_STRICT", "1") != "0"
    if not ok:
        if strict:
            print("[ERROR] compare failed")
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
        return False
    print("[INFO] compare passed")
    return True

if __name__ == "__main__":
    generate_unary_float_case("relu")
