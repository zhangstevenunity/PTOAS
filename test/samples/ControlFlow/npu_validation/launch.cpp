#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

__global__ AICORE void control_flow_kernel_2d(__gm__ float* v1, __gm__ float* v2);

void LaunchControl_flow_kernel_2d(float *v1, float *v2, void *stream) {
    control_flow_kernel_2d<<<1, nullptr, stream>>>(v1, v2);
}
