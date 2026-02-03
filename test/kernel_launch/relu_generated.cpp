#include "common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void sync_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 32;
  int32_t v6 = 1;
  int64_t v7 = 0;
  int64_t v8 = 4096;
  using T = float;
  unsigned v9 = (unsigned) v5;
  unsigned v10 = v4 * v9;
  unsigned v11 = v4 + v10;
  unsigned v12 = (unsigned) v6;
  unsigned v13 = v4 * v12;
  unsigned v14 = v11 + v13;
  __gm__ float* v15 = v1 + v14;
  using GTShape_72927360 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_72927360 = pto::Stride<1024, 1024, 1024, 32, 1>;
  GTShape_72927360 v16 = GTShape_72927360();
  GTStride_72927360 v17 = GTStride_72927360();
  using GT_72927360 = GlobalTensor<float, GTShape_72927360, GTStride_72927360>;
  GT_72927360 v18 = GT_72927360(v15, v16, v17);
  unsigned v19 = (unsigned) v5;
  unsigned v20 = v4 * v19;
  unsigned v21 = v4 + v20;
  unsigned v22 = (unsigned) v6;
  unsigned v23 = v4 * v22;
  unsigned v24 = v21 + v23;
  __gm__ float* v25 = v2 + v24;
  using GTShape_72927536 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_72927536 = pto::Stride<1024, 1024, 1024, 32, 1>;
  GTShape_72927536 v26 = GTShape_72927536();
  GTStride_72927536 v27 = GTStride_72927536();
  using GT_72927536 = GlobalTensor<float, GTShape_72927536, GTStride_72927536>;
  GT_72927536 v28 = GT_72927536(v25, v26, v27);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v7);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v8);
  TLOAD(v29, v18);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TRELU(v30, v29);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v28, v30);
  return;
}


