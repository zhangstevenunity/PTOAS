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
    offset = 24
    out = src[offset:offset + meta.elem_counts[out_name]]
    write_golden(meta, {out_name: np.asarray(out, dtype=np.float32)})


if __name__ == '__main__':
    main()
