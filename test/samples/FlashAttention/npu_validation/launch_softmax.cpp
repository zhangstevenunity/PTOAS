#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void flash_attention_softmax_block(__gm__ float* v1, __gm__ float* v2);

void LaunchFlash_attention_softmax_block(float *v1, float *v2, void *stream) {
    flash_attention_softmax_block<<<1, nullptr, stream>>>(v1, v2);
}
