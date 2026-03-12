#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, matrix32, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    [src_name] = meta.inputs
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    src_m = matrix32(src)
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    out = np.repeat(src_m[:, :1], 32, axis=1)
    write_golden(meta, {single_output(meta): out.astype(np.float32).reshape(-1)})


if __name__ == '__main__':
    main()
