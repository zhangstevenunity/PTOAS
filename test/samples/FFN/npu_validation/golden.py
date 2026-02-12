#!/usr/bin/python3
# coding=utf-8

import numpy as np


def _sigmoid_poly(x):
    x2 = x * x
    x3 = x2 * x
    return np.float32(0.5) + np.float32(0.25) * x + np.float32(-0.020833333) * x3


def generate_golden(v1, v2, v3, v4, v5, v6, v7, v9, v10):
    x = v1.reshape(32, 32).astype(np.float32)
    w1 = v2.reshape(32, 32).astype(np.float32)
    w3 = v3.reshape(32, 32).astype(np.float32)
    w2 = v4.reshape(32, 32).astype(np.float32)
    _ = v5
    _ = v6
    _ = v7
    _ = v9
    _ = v10

    h1 = (x @ w1).astype(np.float32)
    h2 = (x @ w3).astype(np.float32)

    h1 = np.clip(h1, np.float32(-4.0001), np.float32(4.0001))
    gate = _sigmoid_poly(h1).astype(np.float32)
    gated = (gate * h2).astype(np.float32)
    out = (gated @ w2).astype(np.float32)
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
    v6 = np.random.random(size=(1024,)).astype(np.float32)
    v6.tofile("v6.bin")
    v7 = np.random.random(size=(1024,)).astype(np.float32)
    v7.tofile("v7.bin")
    v9 = 32
    v10 = 32
    outputs = generate_golden(v1, v2, v3, v4, v5, v6, v7, v9, v10)
    outputs[0].tofile("golden_v8.bin")


if __name__ == "__main__":
    main()
