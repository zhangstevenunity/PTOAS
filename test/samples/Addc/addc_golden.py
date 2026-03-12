#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, float_values, load_case_meta, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    a_name, b_name, c_name = meta.inputs
    generator = rng()
    a = float_values(generator, meta.elem_counts[a_name], style='signed')
    b = float_values(generator, meta.elem_counts[b_name], style='signed')
    c = float_values(generator, meta.elem_counts[c_name], style='signed_small')
    buffers = default_buffers(meta)
    buffers[a_name] = a
    buffers[b_name] = b
    buffers[c_name] = c
    write_buffers(meta, buffers)
    out = a + b + c
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
