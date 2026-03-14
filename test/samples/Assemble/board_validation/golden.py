#!/usr/bin/python3
# coding=utf-8

import numpy as np


def main():
    np.random.seed(19)

    src_a = np.random.random(size=(32, 32)).astype(np.float32)
    src_b = np.random.random(size=(32, 32)).astype(np.float32)
    # Identity matrix used by the post-assemble matmul path.
    rhs_identity = np.eye(32, dtype=np.float16)
    out_init = np.zeros((32, 32), dtype=np.float32)

    src_a.tofile("v1.bin")
    src_b.tofile("v2.bin")
    rhs_identity.tofile("v3.bin")
    out_init.tofile("v4.bin")


if __name__ == "__main__":
    main()
