#!/usr/bin/python3
# coding=utf-8

import numpy as np


def generate_golden(v1):
    outputs = []
    outputs.append(np.full(1024, -1, dtype=np.float32))
    return outputs


def main():
    np.random.seed(19)
    v1 = np.random.random(size=(1024,)).astype(np.float32)
    v1.tofile("v1.bin")
    outputs = generate_golden(v1)
    outputs[0].tofile("golden_v2.bin")


if __name__ == "__main__":
    main()
