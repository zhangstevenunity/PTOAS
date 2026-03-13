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
    reduced = np.asarray(src_m.min(axis=0), dtype=np.float32)
    out = np.asarray(buffers.get(out_name, np.zeros(meta.elem_counts[out_name], dtype=np.float32)), dtype=np.float32).reshape(-1).copy()
    if out.size == ROWS * COLS:
        out_m = matrix32(out)
        out_m[0, :] = reduced
        out = out_m.reshape(-1)
    elif out.size == COLS:
        out = reduced
    else:
        raise ValueError(f'unsupported col-reduce output size: {out.size}')
    write_golden(meta, {out_name: out})


if __name__ == '__main__':
    main()
