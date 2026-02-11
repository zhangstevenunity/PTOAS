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

import os
import subprocess
import ctypes

import torch


def compile_cpp(kernel_cpp: str, verbose: bool = True, timeout: int = 120) -> str:
    lib_path = os.path.join(os.path.dirname(kernel_cpp), "temp_jitted_kernel.so")

    PTO_LIB_PATH = os.environ["PTO_LIB_PATH"]

    flags = [
        "-fPIC",
        "-shared",
        "-xcce",
        "--npu-arch=dav-2201",
        "-DMEMORY_BASE",  # here hardcoded for A2A3; TODO: expose this option to jit interface
        "-O2",
        "-std=c++17",
        f"-I{PTO_LIB_PATH}/include",
    ]

    command = ["bisheng", *flags, kernel_cpp, "-o", lib_path]
    if verbose:
        print(f"compile {kernel_cpp} with command: \n", command)

    try:
        subprocess.run(command, timeout=timeout, check=True)
    except Exception as e:
        raise RuntimeError(f"Compile failed: {e}") from e

    if verbose:
        print(f"generated {lib_path}")
    return lib_path


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path, block_dim, check_type=True):
    lib_path = os.path.abspath(lib_path)
    lib = ctypes.CDLL(lib_path)

    if check_type:  # otherwise will get segfault for mismatched type
        # TODO: generate checker according to cpp `void call_kernel` signature
        lib.call_kernel.argtypes = [
            ctypes.c_uint32,  # blockDim
            ctypes.c_void_p,  # stream
            ctypes.c_void_p,  # x
            ctypes.c_void_p,  # y
            ctypes.c_uint32,  # num of elements
        ]
        lib.call_kernel.restype = None

    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def add_func(x, y, n, block_dim=block_dim, stream_ptr=default_stream_ptr):
        # TODO: customize call args according to cpp `void call_kernel` signature
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y),
            n,
        )

    return add_func


def jit_compile(src_path, block_dim, clean_up=True):
    lib_path = compile_cpp(src_path)
    func = load_lib(lib_path, block_dim)
    if clean_up:
        os.remove(lib_path)
    return func
