#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import ROWS, default_buffers, float_values, load_case_meta, matrix32, rng, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    src0_name, src1_name = meta.inputs
    generator = rng()
    src0 = float_values(generator, meta.elem_counts[src0_name], style='signed')
    src1 = float_values(generator, meta.elem_counts[src1_name], style='nonzero_signed' if 'Rowexpandsub/rowexpandsub_golden.py' == 'Rowexpanddiv/rowexpanddiv_golden.py' else 'signed')
    src0_m = matrix32(src0)
    src1_m = matrix32(src1)
    row_scalars = src1_m.reshape(-1)[:ROWS].astype(np.float32)
    buffers = default_buffers(meta)
    buffers[src0_name] = src0
    buffers[src1_name] = src1
    write_buffers(meta, buffers)
    out = src0_m - row_scalars[:, None]
    write_golden(meta, {single_output(meta): out.astype(np.float32).reshape(-1)})


if __name__ == '__main__':
    main()
