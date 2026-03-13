#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    ROWS,
    COLS,
    default_buffers,
    float_values,
    load_case_meta,
    matrix32,
    pack_predicate_mask_for_buffer,
    rng,
    single_output,
    write_buffers,
    write_golden,
)


def main():
    meta = load_case_meta()
    mask_name, src0_name, src1_name = meta.inputs
    generator = rng()
    mask_bits = generator.integers(0, 2, size=(ROWS, COLS), dtype=np.uint8).astype(np.bool_)
    mask = pack_predicate_mask_for_buffer(
        mask_bits, elem_count=meta.elem_counts[mask_name], dtype=meta.np_types[mask_name]
    )
    src0 = float_values(generator, meta.elem_counts[src0_name], style='signed')
    src1 = float_values(generator, meta.elem_counts[src1_name], style='signed')
    buffers = default_buffers(meta)
    buffers[mask_name] = mask
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    write_buffers(meta, buffers)
    out = np.where(mask_bits, matrix32(src0), matrix32(src1))
    write_golden(meta, {single_output(meta): out.astype(np.float32).reshape(-1)})


if __name__ == '__main__':
    main()
