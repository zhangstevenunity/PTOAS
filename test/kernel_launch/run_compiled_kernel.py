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
    device = "npu:1"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    torch.set_printoptions(
        threshold=10000,
        linewidth=300,
    )


    # NOTE: fails when k=64/128/256
    m, k, n = 32, 32, 32
    torch.manual_seed(0)
    a = torch.rand((m,k), device=device, dtype=dtype)
    b = torch.rand((k,n), device=device, dtype=dtype)
    c = torch.zeros((m, n), device=device, dtype=dtype)

    relu_func = load_lib("./mul_kernel.so")
    relu_func(c=c, a=a, b=b)
    torch.npu.synchronize()

    print('result:')
    print(c)

    print('ref:')
    print(a@b)

    print('max diff:')
    print((c-a@b).abs().max())

    print('diff')
    print(c-a@b)
    
if __name__ == "__main__":
    test_add()
