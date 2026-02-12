#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void gqa_qk_block(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3);

void LaunchGqa_qk_block(float *v1, float *v2, float *v3, void *stream) {
    gqa_qk_block<<<1, nullptr, stream>>>(v1, v2, v3);
}
