#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
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
    src0_name, src1_name = meta.inputs
    generator = rng()
    src0 = float_values(generator, meta.elem_counts[src0_name], style='cmp')
    src1 = float_values(generator, meta.elem_counts[src1_name], style='cmp')
    pred = matrix32(src0) < matrix32(src1)
    buffers = default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    write_buffers(meta, buffers)
    out_name = single_output(meta)
    packed = pack_predicate_mask_for_buffer(pred, elem_count=meta.elem_counts[out_name], dtype=meta.np_types[out_name])
    write_golden(meta, {out_name: packed})


if __name__ == '__main__':
    main()
