#include "pto/common/pto_instr.hpp"
#include "pto/pto-inst.hpp"


using namespace pto;


// This is a matmul for out, src0, src1, src2, is_bias
__global__ AICORE void RunTMATMULSplitK(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, bool v5) {

  #if __CCE_AICORE__ == 220 && defined(__DAV_C220_VEC__)
  #else  // else branch for `#if defined(__DAV_C220_VEC__)`
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 8;
  int32_t v9 = 256;
  int32_t v10 = 32;
  int32_t v11 = 1;
  int32_t v12 = 0;
  int64_t v13 = 0;
  int64_t v14 = 4096;
  int64_t v15 = 8192;
  using T = float;
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v16;
  TASSIGN(v16, v13);
  Tile<TileType::Mat, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v17;
  TASSIGN(v17, v14);
  Tile<TileType::Mat, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v15);
  Tile<TileType::Left, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::RowMajor, 512, PadValue::Null> v19;
  TASSIGN(v19, v13);
  Tile<TileType::Right, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::ColMajor, 512, PadValue::Null> v20;
  TASSIGN(v20, v13);
  Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, 1024, PadValue::Null> v21;
  TASSIGN(v21, v13);
  Tile<TileType::Bias, float, 1, 32, BLayout::RowMajor, 1, 32, SLayout::NoneBox, 512, PadValue::Null> v22;
  TASSIGN(v22, v13);
  for (int32_t v23 = v12; v23 < v8; v23 += v11) {
    int32_t v24 = v23 * v10;
    unsigned v25 = (unsigned) v9;
    unsigned v26 = v7 * v25;
    unsigned v27 = v7 + v26;
    unsigned v28 = (unsigned) v24;
    unsigned v29 = (unsigned) v11;
    unsigned v30 = v28 * v29;
    unsigned v31 = v27 + v30;
    __gm__ float* v32 = v2 + v31;
    using GTShape_489524496 = pto::Shape<1, 1, 1, 32, 32>;;
    using GTStride_489524496 = pto::Stride<1024, 1024, 1024, 32, 1>;;
    GTShape_489524496 v33 = GTShape_489524496();
    GTStride_489524496 v34 = GTStride_489524496();
    using GT_489524496 = GlobalTensor<float, GTShape_489524496, GTStride_489524496>;;
    GT_489524496 v35 = GT_489524496(v32, v33, v34);
    unsigned v36 = (unsigned) v24;
    unsigned v37 = (unsigned) v10;
    unsigned v38 = v36 * v37;
    unsigned v39 = v7 + v38;
    unsigned v40 = (unsigned) v11;
    unsigned v41 = v7 * v40;
    unsigned v42 = v39 + v41;
    __gm__ float* v43 = v3 + v42;
    using GTShape_489524704 = pto::Shape<1, 1, 1, 32, 32>;;
    using GTStride_489524704 = pto::Stride<1024, 1024, 1024, 32, 1>;;
    GTShape_489524704 v44 = GTShape_489524704();
    GTStride_489524704 v45 = GTStride_489524704();
    using GT_489524704 = GlobalTensor<float, GTShape_489524704, GTStride_489524704>;;
    GT_489524704 v46 = GT_489524704(v43, v44, v45);
    unsigned v47 = (unsigned) v10;
    unsigned v48 = v7 * v47;
    unsigned v49 = v7 + v48;
    unsigned v50 = (unsigned) v11;
    unsigned v51 = v7 * v50;
    unsigned v52 = v49 + v51;
    __gm__ float* v53 = v4 + v52;
    using GTShape_489524912 = pto::Shape<1, 1, 1, 1, 32>;;
    using GTStride_489524912 = pto::Stride<32, 32, 32, 32, 1>;;
    GTShape_489524912 v54 = GTShape_489524912();
    GTStride_489524912 v55 = GTStride_489524912();
    using GT_489524912 = GlobalTensor<float, GTShape_489524912, GTStride_489524912>;;
    GT_489524912 v56 = GT_489524912(v53, v54, v55);
    TLOAD(v16, v35);
    TLOAD(v17, v46);
    if (v5) {
      TLOAD(v18, v56);
    } else {
    };
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(v19, v16);
    TMOV(v20, v17);
    if (v5) {
      TMOV(v22, v18);
    } else {
    };
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    bool v57 = v23 == v12;
    if (v57) {
      if (v5) {
        TMATMUL_BIAS(v21, v19, v20, v22);
      } else {
        TMATMUL(v21, v19, v20);
      };
    } else {
      TMATMUL_ACC(v21, v21, v19, v20);
    };
    set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  unsigned v58 = (unsigned) v10;
  unsigned v59 = v7 * v58;
  unsigned v60 = v7 + v59;
  unsigned v61 = (unsigned) v11;
  unsigned v62 = v7 * v61;
  unsigned v63 = v60 + v62;
  __gm__ float* v64 = v1 + v63;
  using GTShape_489525088 = pto::Shape<1, 1, 1, 32, 32>;
  using GTStride_489525088 = pto::Stride<1024, 1024, 1024, 32, 1>;
  GTShape_489525088 v65 = GTShape_489525088();
  GTStride_489525088 v66 = GTStride_489525088();
  using GT_489525088 = GlobalTensor<float, GTShape_489525088, GTStride_489525088>;
  GT_489525088 v67 = GT_489525088(v64, v65, v66);
  TSTORE(v67, v21);
  #endif
  return;
}


extern "C" void call_kernel(
    uint32_t blockDim, void* stream,
    uint8_t* c, uint8_t* a, uint8_t* b)
{
    RunTMATMULSplitK<<<blockDim, nullptr, stream>>>(
      reinterpret_cast<float*>(c),
      reinterpret_cast<float*>(a),
      reinterpret_cast<float*>(b),
      nullptr, false
    );
}