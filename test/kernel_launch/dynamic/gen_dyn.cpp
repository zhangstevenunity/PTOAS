#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void sync_kernel_dyn(__gm__ float* v1, __gm__ float* v2, int32_t v3) {
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 0;
  int32_t v7 = 1;
  int32_t v8 = 32;
  int32_t v9 = 256;
  int32_t v10 = 8;
  int64_t v11 = 0;
  int64_t v12 = 128;
  using T = float;

  #if defined(__DAV_VEC__)
  int64_t v13 = get_block_idx();
  int32_t v14 = (int32_t) v13;
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v15;
  TASSIGN(v15, v11);
  Tile<TileType::Vec, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v16;
  TASSIGN(v16, v12);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  for (int32_t v17 = v6; v17 < v10; v17 += v7) {
    int32_t v18 = v14 * v9;
    int32_t v19 = v17 * v8;
    int32_t v20 = v18 + v19;
    unsigned v21 = (unsigned) v20;
    unsigned v22 = (unsigned) v7;
    unsigned v23 = v21 * v22;
    unsigned v24 = v5 + v23;
    __gm__ float* v25 = v1 + v24;
    using GTShape_563597648 = pto::Shape<1, 1, 1, 1, 32>;;
    using GTStride_563597648 = pto::Stride<32, 32, 32, 32, 1>;;
    constexpr pto::Layout GT_563597648_layout = pto::Layout::ND;;
    GTShape_563597648 v26 = GTShape_563597648();
    GTStride_563597648 v27 = GTStride_563597648();
    using GT_563597648 = GlobalTensor<float, GTShape_563597648, GTStride_563597648, GT_563597648_layout>;;
    GT_563597648 v28 = GT_563597648(v25, v26, v27);
    unsigned v29 = (unsigned) v20;
    unsigned v30 = (unsigned) v7;
    unsigned v31 = v29 * v30;
    unsigned v32 = v5 + v31;
    __gm__ float* v33 = v2 + v32;
    using GTShape_563414896 = pto::Shape<1, 1, 1, 1, 32>;;
    using GTStride_563414896 = pto::Stride<32, 32, 32, 32, 1>;;
    constexpr pto::Layout GT_563414896_layout = pto::Layout::ND;;
    GTShape_563414896 v34 = GTShape_563414896();
    GTStride_563414896 v35 = GTStride_563414896();
    using GT_563414896 = GlobalTensor<float, GTShape_563414896, GTStride_563414896, GT_563414896_layout>;;
    GT_563414896 v36 = GT_563414896(v33, v34, v35);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v15, v28);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TRELU(v16, v15);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v36, v16);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  pipe_barrier(PIPE_ALL);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  #endif // __DAV_VEC__

  return;
}

extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2, uint32_t n) {
    sync_kernel_dyn<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2, n);
}