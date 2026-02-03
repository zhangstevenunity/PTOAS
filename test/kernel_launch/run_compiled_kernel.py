import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # NOTE: kernel is single-core for now
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def mul_func(
        c, a, b,
        block_dim=default_block_dim,
        stream_ptr=default_stream_ptr
    ):
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(c),
            torch_to_ctypes(a),
            torch_to_ctypes(b),
        )

    return mul_func


def test_add():
    device = "npu"
    dtype = torch.float32

    shape = [32, 32]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    a = torch.nrand(shape, device=device, dtype=dtype)
    b = torch.nrand(shape, device=device, dtype=dtype)
    c = torch.zeros(shape, device=device, dtype=dtype)

    relu_func = load_lib("./mul_kernel.so")
    relu_func(c=c, a=a, b=b)
    torch.npu.synchronize()

    print('result:')
    print(c)

    print('ref:')
    print(a@b)

    print('max diff:')
    print((c-a@b).abs().max())

if __name__ == "__main__":
    test_add()