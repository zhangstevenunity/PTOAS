#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch_npu

from jit_util_add import jit_compile


def test_add():
    device = "npu:6"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    shape = [20, 128]
    BLOCK_DIM = 4
    x = torch.rand(shape, device=device, dtype=dtype) - 0.5
    y = torch.full(shape, -10, device=device, dtype=dtype)
    print('jitting!')
    relu_kernel = jit_compile("gen_cpp1d.cpp", block_dim=BLOCK_DIM)
    relu_kernel(x, y)
    torch.npu.synchronize()

    y_ref = torch.max(x, torch.zeros_like(x))
    correct = y == y_ref
    print('inp:')
    print(x)

    print('res:')
    print(y)

    step = 4
    for i in range(0, shape[0]):
        for j in range(0, shape[1], step):
            if correct[i, j:j+step].all():
                print('X', end='')
            else:
                print('.', end='')
        print('|')

    torch.testing.assert_close(y, y_ref)
    print("RELU test pass!")


if __name__ == "__main__":
    test_add()
