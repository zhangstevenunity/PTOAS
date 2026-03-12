#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import ROWS, COLS, default_buffers, float_values, load_case_meta, matrix32, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    out_name = single_output(meta)
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    src_m = matrix32(src)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    reduced = np.asarray(src_m.sum(axis=1, dtype=np.float32), dtype=np.float32)
    out = np.zeros(meta.elem_counts[out_name], dtype=np.float32)
    if out.size == ROWS * COLS:
        out[:ROWS] = reduced
    elif out.size == ROWS:
        out = reduced
    else:
        raise ValueError(f'unsupported row-reduce output size: {out.size}')
    write_golden(meta, {out_name: out})


if __name__ == '__main__':
    main()
