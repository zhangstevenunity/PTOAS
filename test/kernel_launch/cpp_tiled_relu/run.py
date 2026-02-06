import ctypes
import torch
import torch_npu


def torch_to_ctypes(tensor):
    return ctypes.c_void_p(tensor.data_ptr())


def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)

    default_block_dim = 16  # NOTE: kernel is single-core for now
    default_stream_ptr = torch.npu.current_stream()._as_parameter_

    def relu_func(
        x,
        y,
        block_dim=default_block_dim,
        stream_ptr=default_stream_ptr
        ):
        print(f"streamptr: {stream_ptr}")
        lib.call_kernel(
            block_dim,
            stream_ptr,
            torch_to_ctypes(x),
            torch_to_ctypes(y)
        )

    return relu_func


def test_add():
    import os
    device = "npu:1"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    shape = [128, 128]  # shape hard-coded as the kernel
    torch.manual_seed(0)
    x = torch.rand(shape, device=device, dtype=dtype) - 0.5
    y = torch.zeros(shape, device=device, dtype=dtype)

    relu_func = load_lib("./tiled_relu.so")
    relu_func(x, y)
    os.system("npu-smi info")
    print('running!')
    torch.npu.synchronize()
    y_ref = torch.nn.functional.relu(x)
    torch.testing.assert_close(y, y_ref)
    print("result equal!")

def visualize_relu_tile_bug():
    import torch
    import numpy as np

    device = "npu:1"
    torch.set_default_device(device)
    torch.npu.set_device(device)
    dtype = torch.float32

    H, W = 128, 128
    TILE_H, TILE_W = 32, 32
    TILE_Y0, TILE_X0 = 0, 0

    SENTINEL = -7777.0

    torch.manual_seed(0)

    #x = torch.full((H, W), -5.0, device=device, dtype=dtype)
    x = torch.rand((H, W), device=device, dtype=dtype) - 0.5
    x2 = x.clone()
    y = torch.full((H, W), SENTINEL, device=device, dtype=dtype)

    import os
    relu_func = load_lib("./tiled_relu.so")
    for i in range(100):
        relu_func(x, y)
        torch.npu.synchronize()
        c = x+1
        d = y + 1


#    touched = (y != SENTINEL)
#
#    # mask of intended tile
#    tile_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
#    tile_mask[touched] = True
#    #tile_mask[TILE_Y0:TILE_Y0+TILE_H, TILE_X0:TILE_X0+TILE_W] = True
#    
#    
#    y_ref = torch.max(x2, torch.zeros_like(x2))
#
#    print('number of zeros:')
#    print((y==0.0).sum())
#    print((y>0.0).sum())
#    print((y<0.0).sum())
#    print(y)
#    print((y_ref == y).sum())
#
#    save_similarity_image(y_ref, y)



#    step = 4
#    for i in range(0, H, step):
#        row = ""
#        for j in range(0, W, step):
#            block = tile_mask[i:i+step, j:j+step]
#            if (block).any():
#                row += "X"
#            else:
#                row += "."
#        print(row)


def save_similarity_image(t1, t2, filename="similarity.png", tol=1e-7):
    import cv2
    import numpy as np
    assert t1.shape == t2.shape, "Tensors must have same shape"

    # compute similarity mask (True = similar)
    mask = torch.abs(t1 - t2) < tol

    # convert to uint8 image: white=255, black=0
    img = mask.cpu().numpy().astype(np.uint8) * 255

    cv2.imwrite(filename, img)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    visualize_relu_tile_bug()
#    test_add()