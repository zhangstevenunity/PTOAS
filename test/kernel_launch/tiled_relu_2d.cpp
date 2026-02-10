
#include <pto/pto-inst.hpp>
using namespace pto;

__global__ AICORE void sync_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  // let's say the input is 128x128 with stride (128, 1) (Row-major)
  // but we only load and work with 32x32 tiles in this func
  // this kernel takes a row major tensor
  // tiles it in row-major order
  // the tiles are also row-major 


  #if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)

  using shape3232 = pto::Shape<1, 1, 1, 32, 32>;
  // This is the stride used for the 32x32 tiles. Note that it's
  // the same stride as the 128x128 input matrix.
  using stride32_1 = pto::Stride<16384, 16384, 16384, 128, 1>;

  shape3232 v16 = shape3232(); stride32_1 v17 = stride32_1();
  shape3232 v26 = shape3232(); stride32_1 v27 = stride32_1();
  // for 1d tiling. contiguous chunks in 1d space:
//  unsigned int gm_tile_offset = block_idx * 1024; // b_idx * 32 * 32 # of elements
  // 2d tiling: we need to map the single block_idx to tile i,j
  unsigned int num_tiles_h = 4; // big tensor is 128x128, each tile is 32x32, so 4 tiles per dim
  unsigned int num_tiles_w = 4;

  // This mapping here from block_index to which tile T_ij
  // is also row major. So you could have swizzling here...
  unsigned int tile_idx_i = block_idx / num_tiles_h; 
  unsigned int tile_idx_j = block_idx % num_tiles_w; 
  // Calculate the offset to T_ij. 32 * (i, j) * stride_inp_tensor
  unsigned int gm_tile_offset = 32 * (tile_idx_i * 128 + tile_idx_j);
  __gm__ float* gm_tile_in = v1 + gm_tile_offset;
  __gm__ float* gm_tile_out = v2 + gm_tile_offset;
  // Now for every core, this will have loaded it's own tile from gmem.
  GlobalTensor<float, shape3232, stride32_1> gten_in = GlobalTensor<float, shape3232, stride32_1>(gm_tile_in, v16, v17);
  GlobalTensor<float, shape3232, stride32_1> gten_out = GlobalTensor<float, shape3232, stride32_1>(gm_tile_out, v26, v27);

  // Why is fractal size specified here 512?
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> tile_in;
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> tile_out;
  int64_t v7 = 0;
  int64_t v8 = 4096; // using tiles 32x32 and each element is 4 bytes so offset is 32*32*4=4096 bytes
  TASSIGN(tile_in, v7);
  TASSIGN(tile_out, v8);

  TLOAD(tile_in, gten_in);

  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TRELU(tile_out, tile_in);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

  TSTORE(gten_out, tile_out);
  #endif
  return;
}

extern "C" void call_kernel( uint32_t blockDim, void* stream, uint8_t* v1, uint8_t* v2) {
    sync_kernel_2d<<<blockDim, nullptr, stream>>>((float *)v1, (float *)v2);
}
