import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 1  # NOTE: kernel is single-core for now
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def relu_func(
        x,
        y,
        block_dim=default_block_dim,
        stream_ptr=default_stream_ptr
        ):
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y)
        )

    return relu_func


def test_add():
    device = "npu"
    dtype = torch.float32

    shape = [32, 32]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    x = torch.rand(shape, device=device, dtype=dtype) - 0.5
    y = torch.empty(shape, device=device, dtype=dtype)

    relu_func = load_lib("./relu_kernel.so")
    relu_func(x, y)
    torch.npu.synchronize()

    y_ref = torch.nn.functional.relu(x)
    print("first row of y", y[0,:])
    print("first row of y_ref", y_ref[0,:])
    torch.testing.assert_close(y[0,:], y_ref[0,:])
    print("first row equal!")

    print("second row of y", y[1,:])
    print("second row of y_ref", y_ref[1,:])

if __name__ == "__main__":
    test_add()
