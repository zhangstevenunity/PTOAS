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
    src_name = meta.inputs[0]
    out_name = single_output(meta)
    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style='signed')
    dst_init = float_values(generator, meta.elem_counts[out_name], style='signed_small')
    buffers = default_buffers(meta)
    buffers[src_name] = src
    buffers[out_name] = dst_init
    write_buffers(meta, buffers)

    src_array = np.asarray(src, dtype=np.float32)
    out_array = np.asarray(dst_init, dtype=np.float32).copy()
    touched = []
    for depth in range(16):
        for row in range(16):
            base = depth * 1048576 + row * 1024
            touched.extend(base + col for col in range(16))
    touched = np.asarray(touched, dtype=np.int64)
    out_array[touched] = src_array[touched]
    write_golden(meta, {out_name: out_array})


if __name__ == '__main__':
    main()
