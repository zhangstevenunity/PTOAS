#!/usr/bin/python3
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


def load_case_meta(main_cpp: str = 'main.cpp', outputs_txt: str = 'outputs.txt') -> CaseMeta:
    text = Path(main_cpp).read_text(encoding='utf-8')
    elem_counts = {
        match.group(1): int(match.group(2))
        for match in re.finditer(r'size_t\s+elemCount_(\w+)\s*=\s*(\d+);', text)
    }
    np_types = {
        match.group(1): np.dtype(_HOST_TYPE_TO_NP[match.group(2).strip()])
        for match in re.finditer(
            r'size_t\s+fileSize_(\w+)\s*=\s*elemCount_\1\s*\*\s*sizeof\(([^)]+)\);',
            text,
        )
    }
    read_order = re.findall(r'ReadFile\("\./([^"]+)\.bin"', text)
    outputs_path = Path(outputs_txt)
    outputs = []
    if outputs_path.is_file():
        outputs = [line.strip() for line in outputs_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    return CaseMeta(elem_counts=elem_counts, np_types=np_types, read_order=read_order, outputs=outputs)


def load_scalar_assignments(ctype: str, main_cpp: str = 'main.cpp') -> List[int]:
    text = Path(main_cpp).read_text(encoding='utf-8')
    pattern = rf'{re.escape(ctype)}\s+\w+\s*=\s*(-?\d+);'
    return [int(value) for value in re.findall(pattern, text)]


def load_int32_assignments(main_cpp: str = 'main.cpp') -> List[int]:
    return load_scalar_assignments('int32_t', main_cpp=main_cpp)


def rng():
    return np.random.default_rng(SEED)


def float_values(generator, count: int, *, style: str) -> np.ndarray:
    if style == 'signed':
        values = generator.uniform(-3.0, 3.0, size=count).astype(np.float32)
    elif style == 'signed_small':
        values = generator.uniform(-1.5, 1.5, size=count).astype(np.float32)
    elif style == 'nonzero_signed':
        values = generator.uniform(-3.0, 3.0, size=count).astype(np.float32)
        mask = np.abs(values) < np.float32(0.25)
        values[mask] = np.where(values[mask] >= 0.0, np.float32(0.25), np.float32(-0.25))
    elif style == 'positive':
        values = generator.uniform(0.25, 4.0, size=count).astype(np.float32)
    elif style in {'exp', 'cmp'}:
        values = generator.uniform(-2.0, 2.0, size=count).astype(np.float32)
    else:
        raise ValueError(f'unsupported float style: {style}')
    return values


def int_values(generator, count: int, dtype: np.dtype, *, style: str) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.int16):
        if style != 'bitwise':
            raise ValueError(f'unsupported int16 style: {style}')
        values = generator.integers(-256, 256, size=count, dtype=np.int32)
    elif dtype == np.dtype(np.int32):
        if style == 'bitwise':
            values = generator.integers(-256, 256, size=count, dtype=np.int32)
        elif style == 'shift_small':
            values = generator.integers(0, 4, size=count, dtype=np.int32)
        else:
            raise ValueError(f'unsupported int32 style: {style}')
    else:
        raise ValueError(f'unsupported dtype/style pair: {dtype}/{style}')
    return values.astype(dtype, copy=False)


def matrix32(values: np.ndarray, rows: int = ROWS, cols: int = COLS) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    expected = rows * cols
    if flat.size != expected:
        raise ValueError(f'expected {expected} elements, got {flat.size}')
    return flat.reshape(rows, cols)


def default_buffers(meta: CaseMeta):
    return {name: np.zeros(meta.elem_counts[name], dtype=meta.np_types[name]) for name in meta.read_order}


def write_buffers(meta: CaseMeta, buffers):
    for name in meta.read_order:
        if name not in buffers:
            raise KeyError(f'missing buffer for {name}')
        array = np.asarray(buffers[name], dtype=meta.np_types[name]).reshape(-1)
        expected = meta.elem_counts[name]
        if array.size != expected:
            raise ValueError(f'{name}: expected {expected} elements, got {array.size}')
        array.tofile(f'{name}.bin')


def write_golden(meta: CaseMeta, outputs):
    for name in meta.outputs:
        if name not in outputs:
            raise KeyError(f'missing golden for {name}')
        array = np.asarray(outputs[name], dtype=meta.np_types[name]).reshape(-1)
        expected = meta.elem_counts[name]
        if array.size != expected:
            raise ValueError(f'{name}: expected {expected} golden elements, got {array.size}')
        array.tofile(f'golden_{name}.bin')


def single_output(meta: CaseMeta) -> str:
    if len(meta.outputs) != 1:
        raise ValueError(f'expected exactly one output, got {meta.outputs}')
    return meta.outputs[0]


def packed_row_bytes(cols: int) -> int:
    return (cols + 7) // 8


def packed_mask_storage_bytes(elem_count: int, dtype) -> int:
    return int(elem_count) * np.dtype(dtype).itemsize


def packed_mask_storage_cols(*, elem_count: int, dtype, rows: int = ROWS) -> int:
    storage_bytes = packed_mask_storage_bytes(elem_count, dtype)
    if storage_bytes % rows != 0:
        raise ValueError(f'packed mask storage {storage_bytes} bytes is not divisible by rows={rows}')
    return storage_bytes // rows


def pack_predicate_mask(bits: np.ndarray, *, storage_cols: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.bool_)
    if bits.ndim != 2:
        raise ValueError('mask bits must be a 2D array')
    rows, cols = bits.shape
    used_cols = packed_row_bytes(cols)
    if storage_cols < used_cols:
        raise ValueError(f'storage_cols={storage_cols} is too small for cols={cols}')
    packed = np.zeros((rows, storage_cols), dtype=np.uint8)
    for row in range(rows):
        for byte_index, base_col in enumerate(range(0, cols, 8)):
            width = min(8, cols - base_col)
            packed_byte = 0
            for bit_index in range(width):
                if bits[row, base_col + bit_index]:
                    packed_byte |= 1 << bit_index
            packed[row, byte_index] = packed_byte
    return packed.reshape(-1)


def pack_predicate_mask_for_buffer(bits: np.ndarray, *, elem_count: int, dtype, rows: int = ROWS) -> np.ndarray:
    dtype = np.dtype(dtype)
    storage_cols = packed_mask_storage_cols(elem_count=elem_count, dtype=dtype, rows=rows)
    packed_bytes = pack_predicate_mask(bits, storage_cols=storage_cols)
    expected_bytes = packed_mask_storage_bytes(elem_count, dtype)
    if packed_bytes.nbytes != expected_bytes:
        raise ValueError(
            f'packed mask byte size mismatch: expected {expected_bytes}, got {packed_bytes.nbytes}'
        )
    return np.frombuffer(packed_bytes.tobytes(), dtype=dtype).copy()


def _report_compare_failure(golden: np.ndarray, output: np.ndarray, golden_path: str, output_path: str):
    if golden.size == 0:
        print(f'[ERROR] Mismatch: {golden_path} vs {output_path}, empty buffers')
        return
    if np.issubdtype(golden.dtype, np.integer) or np.issubdtype(golden.dtype, np.unsignedinteger):
        golden_cmp = golden.astype(np.int64, copy=False)
        output_cmp = output.astype(np.int64, copy=False)
    else:
        golden_cmp = golden.astype(np.float64, copy=False)
        output_cmp = output.astype(np.float64, copy=False)
    diff = np.abs(golden_cmp - output_cmp)
    index = int(np.argmax(diff))
    print(
        f'[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={float(diff[index])} at idx={index} '
        f'(golden={golden_cmp[index]}, out={output_cmp[index]}, dtype={golden.dtype})'
    )


def compare_file(golden_path: str, output_path: str, dtype, atol: float) -> bool:
    if not os.path.exists(output_path):
        print(f'[ERROR] Output missing: {output_path}')
        return False
    if not os.path.exists(golden_path):
        print(f'[ERROR] Golden missing: {golden_path}')
        return False
    dtype = np.dtype(dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f'[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}')
        return False
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.unsignedinteger):
        if atol == 0.0:
            ok = np.array_equal(golden, output)
        else:
            ok = np.allclose(golden, output, atol=atol, rtol=atol)
    else:
        ok = np.allclose(golden, output, atol=atol, rtol=atol, equal_nan=True)
    if not ok:
        _report_compare_failure(golden, output, golden_path, output_path)
        return False
    return True


def compare_packed_mask_file(golden_path: str, output_path: str, *, rows: int = ROWS, cols: int = COLS) -> bool:
    if not os.path.exists(output_path):
        print(f'[ERROR] Output missing: {output_path}')
        return False
    if not os.path.exists(golden_path):
        print(f'[ERROR] Golden missing: {golden_path}')
        return False
    golden = np.fromfile(golden_path, dtype=np.uint8)
    output = np.fromfile(output_path, dtype=np.uint8)
    if golden.size % rows != 0 or output.size % rows != 0:
        print(f'[ERROR] Packed mask buffer size is not divisible by rows={rows}')
        return False
    golden_cols = golden.size // rows
    output_cols = output.size // rows
    used_cols = packed_row_bytes(cols)
    if golden_cols < used_cols or output_cols < used_cols:
        print(f'[ERROR] Packed mask storage is too small: need {used_cols} bytes per row')
        return False
    golden_view = golden.reshape(rows, golden_cols)[:, :used_cols].reshape(-1)
    output_view = output.reshape(rows, output_cols)[:, :used_cols].reshape(-1)
    if not np.array_equal(golden_view, output_view):
        diff = np.nonzero(golden_view != output_view)[0]
        index = int(diff[0]) if diff.size else 0
        print(
            f'[ERROR] Packed mask mismatch: {golden_path} vs {output_path}, idx={index} '
            f'(golden={int(golden_view[index])}, out={int(output_view[index])})'
        )
        return False
    return True


def finalize_compare(ok: bool):
    strict = os.getenv('COMPARE_STRICT', '1') != '0'
    if not ok:
        if strict:
            print('[ERROR] compare failed')
            sys.exit(2)
        print('[WARN] compare failed (non-gating)')
        return False
    print('[INFO] compare passed')
    return True


def compare_outputs(dtype, atol: float):
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        ok = compare_file(f'golden_{name}.bin', f'{name}.bin', dtype, atol) and ok
    return finalize_compare(ok)


def compare_packed_mask_outputs(*, rows: int = ROWS, cols: int = COLS):
    meta = load_case_meta()
    ok = True
    for name in meta.outputs:
        ok = compare_packed_mask_file(f'golden_{name}.bin', f'{name}.bin', rows=rows, cols=cols) and ok
    return finalize_compare(ok)
