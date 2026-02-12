#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void ffn_swiglu_block(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, __gm__ float* v4, __gm__ float* v5, __gm__ float* v6, __gm__ float* v7, __gm__ float* v8, int32_t v9, int32_t v10);

void LaunchFfn_swiglu_block(float *v1, float *v2, float *v3, float *v4, float *v5, float *v6, float *v7, float *v8, int32_t v9, int32_t v10, void *stream) {
    ffn_swiglu_block<<<1, nullptr, stream>>>(v1, v2, v3, v4, v5, v6, v7, v8, v9, v10);
}
