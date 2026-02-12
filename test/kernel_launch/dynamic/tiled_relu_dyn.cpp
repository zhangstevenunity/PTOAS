
#include <pto/pto-inst.hpp>
using namespace pto;

/*
  We want 1 kernel that allows arbitrary length inputs not known at compile time.
  We still have a compile-time known tile size, let's say (2, 128)
  Then if we get a input tensor of length 512 each core will run
  
  GlobalTensor -> 2 x 512 memory region in GM

  for (i = 0,1,2,3)
    load chunk [:, i*128:(i+1)*128] from GM into tile
    .... do calcs and store back into GM output.

  QUESTION: how do we want to treat > 1d inputs for vectors ops?
  If the input tensor is row-major we can just treat it as a 1d vector
  without any second thoughts.
  What if it's not contiguous?
    - option 1: make it contiguous and then pass in to kernel
    - option 2: dispatch special kernel for that layout
*/

__global__ AICORE void sync_kernel_dyn(__gm__ float* v1, __gm__ float* v2, const uint32_t N) {
  // let's say the input has n elements with stride (1) (Row-major)
  // each core then gets assigned a range
  // core 0: [0, n/num_blocks]
  // core 1: [n/num_blocks + 1 : 2*n/num_blocks]
  // So let's say n/num_blocks = num_el_per_core
  // num_el_per_core should be divisible by tile_size for simplicity.
  // and num_blocks should be divisible by n for simplicity.


  // Example: input is 20x128 so (N=2560)
  // with 10 blocks each core should do 256 elements
  // and for each core to do 256 elements it's 8 tiles
  // if we assume 32 element tiles.

  #if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  // each tile is tile_w consequtive elements
  const unsigned tile_w = 32;
  const unsigned NUM_BLOCKS = 10;
  int num_el_per_core = N / NUM_BLOCKS;
  int num_tiles = num_el_per_core / tile_w;

  // In this kernel since we only derive 1 tile for each Globaltensor
  // the GlobalTensor just contains the memory region of 1 tile.
  // Rather than having a big GlobalTensor that we then derive many
  // tiles from using TASSIGN to offset inside the GM region.
  using ShapeGlobal = pto::Shape<1, 1, 1, 1, tile_w>;
  using StrideGlobal = pto::Stride<1, 1, 1, 1, 1>;

  ShapeGlobal v16 = ShapeGlobal(); StrideGlobal v17 = StrideGlobal();
  ShapeGlobal v26 = ShapeGlobal(); StrideGlobal v27 = StrideGlobal();
    // for 1d tiling. contiguous chunks in 1d space:
    unsigned int gm_tile_offset = block_idx * tile_w * num_tiles; // each tile has tile_h*tile_w contiguous elements
    __gm__ float* gm_tile_in = v1 + gm_tile_offset;
    __gm__ float* gm_tile_out = v2 + gm_tile_offset;
  // Now for every core, this will have loaded it's own tile from gmem.
  GlobalTensor<float, ShapeGlobal, StrideGlobal> gten_in = GlobalTensor<float, ShapeGlobal, StrideGlobal>(gm_tile_in, v16, v17);
  GlobalTensor<float, ShapeGlobal, StrideGlobal> gten_out = GlobalTensor<float, ShapeGlobal, StrideGlobal>(gm_tile_out, v26, v27);
  Tile<TileType::Vec, float, 1, tile_w, BLayout::RowMajor, 1, tile_w> tile_in;
  Tile<TileType::Vec, float, 1, tile_w, BLayout::RowMajor, 1, tile_w> tile_out;

  int64_t v7 = 0;
  // we just have to make sure that the second tile is somewhere in the 196 kb of UB?
  // and it must not overlap with the first tile
  int64_t v8 = tile_w * 4 + 0x100; // using tiles and each element is 4 bytes so offset #elements * bytes/element
  TASSIGN(tile_in, v7);
  TASSIGN(tile_out, v8);


  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (uint32_t i = 0; i < num_tiles; i++){
    unsigned offset_tile = i * tile_w;
    TASSIGN(gten_in, gm_tile_in + offset_tile);
    TASSIGN(gten_out, gm_tile_out + offset_tile);

    // in the MTE2 (load memory pipeline) we must wait for relu comp
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(tile_in, gten_in);

    // signal to PIPE_V pipeline that mte2 is done
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    // For the PIPE_V pipeline we must make sure
    // that tile load complete (MTE2) and previous tile store complete (MTE3)
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TRELU(tile_out, tile_in);
    // signal to load (MTE3) and store (MTE2) that Vector core is done
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    // For the store pipeline (MTE3) we make sure that relu(Vector pipeline) is completed
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gten_out, tile_out);
    // signal to PIPE_V that store is done
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  //TASSIGN(gten_out, gm_tile_out);
  #endif
  return;
}
    // MTE2: transport memory GB->UB (load)
    // V: a vector pipeline
    // MTE3: UB->GM (store)

/*
set_flag(pipe, tpipe, pipeID);

It means:

In pipeline pipe, set an event flag with ID pipeID that will be used to trigger pipeline tpipe.

When you call:

wait_flag(pipe, tpipe, pipeID);

It means:

In pipeline tpipe, wait until pipeline pipe has set the event flag with ID pipeID.


*/

extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2, uint32_t n) {
    sync_kernel_dyn<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2, n);
}
