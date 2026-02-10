
#include <pto/pto-inst.hpp>
using namespace pto;

__global__ AICORE void sync_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  // let's say the input is Hx128 with stride (128, 1) (Row-major)
  // but we only load and work with tile_hx128 tiles in this func

  #if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  // this should be the same as num_blocks 
  constexpr unsigned NUM_CORES = 20;
  static_assert((20 % NUM_CORES) == 0, "block_num must divide number of input rows (128)");
  constexpr int tile_w = 128;
  // each core will work with a tile with this many rows
  constexpr int tile_h = 20 / NUM_CORES;

  // what are these? and do they relate to setting -1 in Tile?
  //set_mask_norm();
  //set_vector_mask(-1, -1);

  // In this kernel since we only derive 1 tile for each Globaltensor
  // the GlobalTensor just contains the memory region of 1 tile.
  // Rather than having a big GlobalTensor that we then derive many
  // tiles from using TASSIGN to offset inside the GM region.
  using ShapeGlobal = pto::Shape<1, 1, 1, tile_h, tile_w>;
  using StrideGlobal = pto::Stride<16384, 16384, 16384, 128, 1>;

  ShapeGlobal v16 = ShapeGlobal(); StrideGlobal v17 = StrideGlobal();
  ShapeGlobal v26 = ShapeGlobal(); StrideGlobal v27 = StrideGlobal();
  // for 1d tiling. contiguous chunks in 1d space:
  unsigned int gm_tile_offset = block_idx * tile_h * tile_w; // each tile has tile_h*tile_w contiguous elements
  __gm__ float* gm_tile_in = v1 + gm_tile_offset;
  __gm__ float* gm_tile_out = v2 + gm_tile_offset;
  // Now for every core, this will have loaded it's own tile from gmem.
  GlobalTensor<float, ShapeGlobal, StrideGlobal> gten_in = GlobalTensor<float, ShapeGlobal, StrideGlobal>(gm_tile_in, v16, v17);
  GlobalTensor<float, ShapeGlobal, StrideGlobal> gten_out = GlobalTensor<float, ShapeGlobal, StrideGlobal>(gm_tile_out, v26, v27);
  Tile<TileType::Vec, float, tile_h, tile_w, BLayout::RowMajor, tile_h, tile_w> tile_in;
  Tile<TileType::Vec, float, tile_h, tile_w, BLayout::RowMajor, tile_h, tile_w> tile_out;

  int64_t v7 = 0;
  // we just have to make sure that the second tile is somewhere in the 196 kb of UB?
  // and it must not overlap with the first tile
  int64_t v8 = tile_h * tile_w * 4 + 0x100; // using tiles and each element is 4 bytes so offset #elements * bytes/element
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

extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2) {
    sync_kernel_2d<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2);
}
