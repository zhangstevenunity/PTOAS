#!/usr/bin/python3
# coding=utf-8

import numpy as np


def _exp_poly(x):
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    return (
        np.float32(1.0001)
        + x
        + np.float32(0.5) * x2
        + np.float32(0.1666667) * x3
        + np.float32(0.04166667) * x4
    )


def generate_golden(v1, v2, v3, v4, v5):
    q = v1.reshape(32, 32).astype(np.float32)
    kt = v2.reshape(32, 32).astype(np.float32)
    v = v3.reshape(32, 32).astype(np.float32)
    _ = v4
    _ = v5

    scale = np.float32(0.1767767)
    scores = (q @ kt).astype(np.float32)
    scores = scores * scale
    scores = np.clip(scores, np.float32(-4.0001), np.float32(4.0001))

    exp = _exp_poly(scores).astype(np.float32)
    softmax = exp / np.float32(32.0001)
    out = (softmax @ v).astype(np.float32)
    return [out.reshape(-1)]


def main():
    np.random.seed(19)
    v1 = np.random.random(size=(1024,)).astype(np.float32)
    v1.tofile("v1.bin")
    v2 = np.random.random(size=(1024,)).astype(np.float32)
    v2.tofile("v2.bin")
    v3 = np.random.random(size=(1024,)).astype(np.float32)
    v3.tofile("v3.bin")
    v4 = np.random.random(size=(1024,)).astype(np.float32)
    v4.tofile("v4.bin")
    v5 = np.random.random(size=(1024,)).astype(np.float32)
    v5.tofile("v5.bin")
    outputs = generate_golden(v1, v2, v3, v4, v5)
    outputs[0].tofile("golden_v6.bin")


if __name__ == "__main__":
    main()
