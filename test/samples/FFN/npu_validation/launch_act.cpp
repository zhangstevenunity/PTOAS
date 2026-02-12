#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void ffn_act_block(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5);

void LaunchFfn_act_block(float *v1, float *v2, float *v3, int32_t v4, int32_t v5, void *stream) {
    ffn_act_block<<<1, nullptr, stream>>>(v1, v2, v3, v4, v5);
}
