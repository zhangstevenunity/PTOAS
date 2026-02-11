#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void relu_kernel_1d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 1;
  int32_t v6 = 2;
  int32_t v7 = 20;
  int32_t v8 = 128;
  int64_t v9 = 0;
  int64_t v10 = 1024;
  using T = float;

  #if defined(__DAV_VEC__)
  int64_t v11 = get_block_idx();
  int32_t v12 = (int32_t) v11;
  int32_t v13 = v12 * v6;
  unsigned v14 = (unsigned) v13;
  unsigned v15 = (unsigned) v8;
  unsigned v16 = v14 * v15;
  unsigned v17 = v4 + v16;
  unsigned v18 = (unsigned) v5;
  unsigned v19 = v4 * v18;
  unsigned v20 = v17 + v19;
  __gm__ float* v21 = v1 + v20;
  using GTShape_609456192 = pto::Shape<1, 1, 1, 2, 128>;
  using GTStride_609456192 = pto::Stride<256, 256, 256, 128, 1>;
  constexpr pto::Layout GT_609456192_layout = pto::Layout::ND;
  GTShape_609456192 v22 = GTShape_609456192();
  GTStride_609456192 v23 = GTStride_609456192();
  using GT_609456192 = GlobalTensor<float, GTShape_609456192, GTStride_609456192, GT_609456192_layout>;
  GT_609456192 v24 = GT_609456192(v21, v22, v23);
  Tile<TileType::Vec, float, 2, 128, BLayout::RowMajor, 2, 128, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v9);
  Tile<TileType::Vec, float, 2, 128, BLayout::RowMajor, 2, 128, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v10);
  TLOAD(v25, v24);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TRELU(v26, v25);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  unsigned v27 = (unsigned) v13;
  unsigned v28 = (unsigned) v8;
  unsigned v29 = v27 * v28;
  unsigned v30 = v4 + v29;
  unsigned v31 = (unsigned) v5;
  unsigned v32 = v4 * v31;
  unsigned v33 = v30 + v32;
  __gm__ float* v34 = v2 + v33;
  using GTShape_609684048 = pto::Shape<1, 1, 1, 2, 128>;
  using GTStride_609684048 = pto::Stride<256, 256, 256, 128, 1>;
  constexpr pto::Layout GT_609684048_layout = pto::Layout::ND;
  GTShape_609684048 v35 = GTShape_609684048();
  GTStride_609684048 v36 = GTStride_609684048();
  using GT_609684048 = GlobalTensor<float, GTShape_609684048, GTStride_609684048, GT_609684048_layout>;
  GT_609684048 v37 = GT_609684048(v34, v35, v36);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v37, v26);
  pipe_barrier(PIPE_ALL);
  #endif // __DAV_VEC__

  return;
}


extern "C" void call_kernel( uint32_t blockDim, void* stream, void* v1, void* v2) {
    relu_kernel_1d<<<blockDim, nullptr, stream>>>(( __gm__ float *)v1, (__gm__ float *)v2);
}