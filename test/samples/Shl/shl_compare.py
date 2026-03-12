#!/usr/bin/python3
from pathlib import Path
import sys

for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (search_root / 'validation_runtime.py').is_file():
        sys.path.insert(0, str(search_root))
        break

import numpy as np
from validation_runtime import compare_outputs

if __name__ == '__main__':
    compare_outputs(np.int32, atol=0.0)
