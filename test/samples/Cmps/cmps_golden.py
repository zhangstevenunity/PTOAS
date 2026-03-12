#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import ROWS, default_buffers, float_values, load_case_meta, matrix32, pack_predicate_mask, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='cmp')
    pred = matrix32(src) > np.float32(1.0)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    out_name = single_output(meta)
    storage_cols = meta.elem_counts[out_name] // 32
    packed = pack_predicate_mask(pred, storage_cols=storage_cols)
    write_golden(meta, {out_name: packed})


if __name__ == '__main__':
    main()
