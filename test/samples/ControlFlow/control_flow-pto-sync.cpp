#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void control_flow_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  float v5 = -0.25f;
  float v6 = 0.5f;
  float v7 = -0.75f;
  float v8 = 1.25f;
  int32_t v9 = 32;
  int32_t v10 = 3;
  int32_t v11 = 2;
  int32_t v12 = 1;
  int32_t v13 = 0;
  int64_t v14 = 0;
  using T = float;
  unsigned v15 = (unsigned) v9;
  unsigned v16 = v4 * v15;
  unsigned v17 = v4 + v16;
  unsigned v18 = (unsigned) v12;
  unsigned v19 = v4 * v18;
  unsigned v20 = v17 + v19;
  __gm__ float* v21 = v1 + v20;
  using GTShape_14525696 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_14525696 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_14525696_layout = pto::Layout::ND;
  GTShape_14525696 v22 = GTShape_14525696();
  GTStride_14525696 v23 = GTStride_14525696();
  using GT_14525696 = GlobalTensor<float, GTShape_14525696, GTStride_14525696, GT_14525696_layout>;
  GT_14525696 v24 = GT_14525696(v21, v22, v23);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  TLOAD(v25, v24);
  set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  for (int32_t v26 = v13; v26 < v11; v26 += v12) {
    bool v27 = v26 == v13;
    if (v27) {
      TADDS(v25, v25, v8);
    } else {
      TADDS(v25, v25, v7);
    };
    for (int32_t v28 = v13; v28 < v10; v28 += v12) {
      bool v29 = v28 == v12;
      if (v29) {
        TADDS(v25, v25, v6);
      } else {
        TADDS(v25, v25, v5);
      };
    };
  }
  unsigned v30 = (unsigned) v9;
  unsigned v31 = v4 * v30;
  unsigned v32 = v4 + v31;
  unsigned v33 = (unsigned) v12;
  unsigned v34 = v4 * v33;
  unsigned v35 = v32 + v34;
  __gm__ float* v36 = v2 + v35;
  using GTShape_14525872 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_14525872 = pto::Stride<1024, 1024, 1024, 32, 1>;
  constexpr pto::Layout GT_14525872_layout = pto::Layout::ND;
  GTShape_14525872 v37 = GTShape_14525872();
  GTStride_14525872 v38 = GTStride_14525872();
  using GT_14525872 = GlobalTensor<float, GTShape_14525872, GTStride_14525872, GT_14525872_layout>;
  GT_14525872 v39 = GT_14525872(v36, v37, v38);
  wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
  TSTORE(v39, v25);
  pipe_barrier(PIPE_ALL);
  return;
}


