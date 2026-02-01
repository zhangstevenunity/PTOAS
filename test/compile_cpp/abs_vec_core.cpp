// TODO: let ptoas auto-insert `__DAV_C220_VEC__` guard without manual editing

#if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
#include "pto/common/pto_instr.hpp"
using namespace pto;
__global__ AICORE void abs_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
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
  using GTShape_933612592 = pto::Shape<32, 32>;
  using GTStride_933612592 = pto::Stride<32, 1>;
  GTShape_933612592 v16 = GTShape_933612592();
  GTStride_933612592 v17 = GTStride_933612592();
  using GT_933612592 = GlobalTensor<float, GTShape_933612592, GTStride_933612592>;
  GT_933612592 v18 = GT_933612592(v15, v16, v17);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v7);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v8);
  TLOAD(v19, v18);
  TABS(v20, v19);
  unsigned v21 = (unsigned) v5;
  unsigned v22 = v4 * v21;
  unsigned v23 = v4 + v22;
  unsigned v24 = (unsigned) v6;
  unsigned v25 = v4 * v24;
  unsigned v26 = v23 + v25;
  __gm__ float* v27 = v2 + v26;
  using GTShape_933612768 = pto::Shape<32, 32>;
  using GTStride_933612768 = pto::Stride<32, 1>;
  GTShape_933612768 v28 = GTShape_933612768();
  GTStride_933612768 v29 = GTStride_933612768();
  using GT_933612768 = GlobalTensor<float, GTShape_933612768, GTStride_933612768>;
  GT_933612768 v30 = GT_933612768(v27, v28, v29);
  TSTORE(v30, v20);
  return;
}
#else
// here the cube core code should be ....
#endif
