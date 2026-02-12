#!/usr/bin/python3
# coding=utf-8

import numpy as np


def compare_bin(golden_path, output_path, dtype, eps):
    golden = np.fromfile(golden_path, dtype=dtype)
    output = np.fromfile(output_path, dtype=dtype)
    if golden.shape != output.shape:
        print(f"[ERROR] Shape mismatch: {golden_path} {golden.shape} vs {output_path} {output.shape}")
        return False
    if not np.allclose(golden, output, atol=eps, rtol=eps):
        diff = np.max(np.abs(golden - output)) if golden.size else 1e-3
        print(f"[ERROR] Mismatch: {golden_path} vs {output_path}, max diff={diff}")
        return False
    return True


def main():
    ok = True
    ok = compare_bin("golden_v7.bin", "v7.bin", np.float32, 1e-3) and ok
    if not ok:
        print("[ERROR] compare failed")
        raise SystemExit(1)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
