#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void relu_kernel_blocked(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  float v5 = 0f;
  int32_t v6 = 1024;
  int32_t v7 = 128;
  int32_t v8 = 32;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 16384;
  using T = float;
  int64_t v12 = get_block_idx();
  int64_t v13 = get_subblockid();
  int64_t v14 = get_subblockdim();
  int64_t v15 = v12 * v14;
  int64_t v16 = v15 + v13;
  size_t v17 = (size_t) v16;
  int32_t v18 = v17 * v7;
  unsigned v19 = (unsigned) v6;
  unsigned v20 = v4 * v19;
  unsigned v21 = v4 + v20;
  unsigned v22 = (unsigned) v18;
  unsigned v23 = (unsigned) v9;
  unsigned v24 = v22 * v23;
  unsigned v25 = v21 + v24;
  __gm__ float* v26 = v1 + v25;
  using GTShape_63529392 = pto::Shape<1, 1, 1, 32, 128>;
  using GTStride_63529392 = pto::Stride<32768, 32768, 32768, 1024, 1>;
  GTShape_63529392 v27 = GTShape_63529392();
  GTStride_63529392 v28 = GTStride_63529392();
  using GT_63529392 = GlobalTensor<float, GTShape_63529392, GTStride_63529392>;
  GT_63529392 v29 = GT_63529392(v26, v27, v28);
  Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v10);
  Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v11);
  TLOAD(v30, v29);
  TLRELU(v31, v30, v5);
  unsigned v32 = (unsigned) v6;
  unsigned v33 = v4 * v32;
  unsigned v34 = v4 + v33;
  unsigned v35 = (unsigned) v18;
  unsigned v36 = (unsigned) v9;
  unsigned v37 = v35 * v36;
  unsigned v38 = v34 + v37;
  __gm__ float* v39 = v2 + v38;
  using GTShape_63641104 = pto::Shape<1, 1, 1, 32, 128>;
  using GTStride_63641104 = pto::Stride<32768, 32768, 32768, 1024, 1>;
  GTShape_63641104 v40 = GTShape_63641104();
  GTStride_63641104 v41 = GTStride_63641104();
  using GT_63641104 = GlobalTensor<float, GTShape_63641104, GTStride_63641104>;
  GT_63641104 v42 = GT_63641104(v39, v40, v41);
  TSTORE(v42, v31);
  return;
}


PTO Driver Success!!!
