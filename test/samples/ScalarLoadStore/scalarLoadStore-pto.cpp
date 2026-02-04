#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void scalar_load_store_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1;
  unsigned v4 = 32;
  unsigned v5 = 0;
  int32_t v6 = 0;
  using T = float;
  unsigned v7 = (unsigned) v6;
  unsigned v8 = v7 * v4;
  unsigned v9 = v5 + v8;
  unsigned v10 = (unsigned) v6;
  unsigned v11 = v10 * v3;
  unsigned v12 = v9 + v11;
  float v13 = (float) v1[v12];
  unsigned v14 = (unsigned) v6;
  unsigned v15 = v14 * v4;
  unsigned v16 = v5 + v15;
  unsigned v17 = (unsigned) v6;
  unsigned v18 = v17 * v3;
  unsigned v19 = v16 + v18;
  __gm__ float v20 = (__gm__ float) v13;
  v2[v19] = v20;
  return;
}


