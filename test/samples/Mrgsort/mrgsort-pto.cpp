#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void vec_add_scalar_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 64;
  int32_t v6 = 256;
  int32_t v7 = 32;
  int32_t v8 = 1;
  int32_t v9 = 1024;
  int64_t v10 = 0;
  int64_t v11 = 4096;
  using T = float;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v4 * v12;
  unsigned v14 = v4 + v13;
  unsigned v15 = (unsigned) v8;
  unsigned v16 = v4 * v15;
  unsigned v17 = v14 + v16;
  __gm__ float* v18 = v1 + v17;
  using GTShape_5207643728 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_5207643728 = pto::Stride<8192, 8192, 8192, 256, 1>;
  constexpr pto::Layout GT_5207643728_layout = pto::Layout::ND;
  GTShape_5207643728 v19 = GTShape_5207643728();
  GTStride_5207643728 v20 = GTStride_5207643728();
  using GT_5207643728 = GlobalTensor<float, GTShape_5207643728, GTStride_5207643728, GT_5207643728_layout>;
  GT_5207643728 v21 = GT_5207643728(v18, v19, v20);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v10);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v11);
  TLOAD(v22, v21);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMRGSORT(v23, v22, v5);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  unsigned v24 = (unsigned) v6;
  unsigned v25 = v4 * v24;
  unsigned v26 = v4 + v25;
  unsigned v27 = (unsigned) v8;
  unsigned v28 = v4 * v27;
  unsigned v29 = v26 + v28;
  __gm__ float* v30 = v2 + v29;
  using GTShape_5207614448 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_5207614448 = pto::Stride<8192, 8192, 8192, 256, 1>;
  constexpr pto::Layout GT_5207614448_layout = pto::Layout::ND;
  GTShape_5207614448 v31 = GTShape_5207614448();
  GTStride_5207614448 v32 = GTStride_5207614448();
  using GT_5207614448 = GlobalTensor<float, GTShape_5207614448, GTStride_5207614448, GT_5207614448_layout>;
  GT_5207614448 v33 = GT_5207614448(v30, v31, v32);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v33, v23);
  pipe_barrier(PIPE_ALL);
  return;
}


