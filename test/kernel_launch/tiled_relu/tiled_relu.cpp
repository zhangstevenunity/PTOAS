#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void relu_kernel_blocked(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 0;
  int32_t v5 = 1;
  int32_t v6 = 32;
  int32_t v7 = 128;
  int32_t v8 = 1024;
  float v9 = 0f;
  int64_t v10 = 0;
  int64_t v11 = 16384;
  using T = float;

  #if defined(__DAV_VEC__)
  int64_t v12 = get_block_idx();
  int64_t v13 = get_subblockid();
  int64_t v14 = get_subblockdim();
  int64_t v15 = v12 * v14;
  int64_t v16 = v15 + v13;
  size_t v17 = (size_t) v16;
  int32_t v18 = v17 * v7;
  unsigned v19 = (unsigned) v8;
  unsigned v20 = v4 * v19;
  unsigned v21 = v4 + v20;
  unsigned v22 = (unsigned) v18;
  unsigned v23 = (unsigned) v5;
  unsigned v24 = v22 * v23;
  unsigned v25 = v21 + v24;
  __gm__ float* v26 = v1 + v25;
  using GTShape_994916336 = pto::Shape<1, 1, 1, 32, 128>;
  using GTStride_994916336 = pto::Stride<32768, 32768, 32768, 1024, 1>;
  constexpr pto::Layout GT_994916336_layout = pto::Layout::ND;
  GTShape_994916336 v27 = GTShape_994916336();
  GTStride_994916336 v28 = GTStride_994916336();
  using GT_994916336 = GlobalTensor<float, GTShape_994916336, GTStride_994916336, GT_994916336_layout>;
  GT_994916336 v29 = GT_994916336(v26, v27, v28);
  Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v10);
  Tile<TileType::Vec, float, 32, 128, BLayout::RowMajor, 32, 128, SLayout::NoneBox, 512, PadValue::Null> v31;
  TASSIGN(v31, v11);
  TLOAD(v30, v29);
  TLRELU(v31, v30, v9);
  unsigned v32 = (unsigned) v8;
  unsigned v33 = v4 * v32;
  unsigned v34 = v4 + v33;
  unsigned v35 = (unsigned) v18;
  unsigned v36 = (unsigned) v5;
  unsigned v37 = v35 * v36;
  unsigned v38 = v34 + v37;
  __gm__ float* v39 = v2 + v38;
  using GTShape_995146464 = pto::Shape<1, 1, 1, 32, 128>;
  using GTStride_995146464 = pto::Stride<32768, 32768, 32768, 1024, 1>;
  constexpr pto::Layout GT_995146464_layout = pto::Layout::ND;
  GTShape_995146464 v40 = GTShape_995146464();
  GTStride_995146464 v41 = GTStride_995146464();
  using GT_995146464 = GlobalTensor<float, GTShape_995146464, GTStride_995146464, GT_995146464_layout>;
  GT_995146464 v42 = GT_995146464(v39, v40, v41);
  TSTORE(v42, v31);
  #endif // __DAV_VEC__

  return;
}


PTO Driver Success!!!
