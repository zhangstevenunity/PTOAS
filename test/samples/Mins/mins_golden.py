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
    [src_name] = meta.inputs
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)
    out = np.minimum(src, np.float32(3.14))
    write_golden(meta, {single_output(meta): np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
