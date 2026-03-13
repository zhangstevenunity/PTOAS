#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import default_buffers, load_case_meta, single_output, write_buffers, write_golden


def main():
    meta = load_case_meta()
    write_buffers(meta, default_buffers(meta))
    out_name = single_output(meta)
    out = np.full(meta.elem_counts[out_name], np.float32(3.14), dtype=np.float32)
    write_golden(meta, {out_name: out})


if __name__ == '__main__':
    main()
