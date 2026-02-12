/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <cstdlib>

using namespace PtoTestCommon;

void LaunchControl_flow_kernel_2d(float *v1, float *v2, void *stream);

int main() {
        size_t fileSize = 32 * 32 * sizeof(float);
    float *dstHost0 = nullptr;
    float *dstDevice0 = nullptr;
    float *srcHost0 = nullptr;
    float *srcDevice0 = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

        aclrtMallocHost((void **)(&dstHost0), fileSize);
    aclrtMallocHost((void **)(&srcHost0), fileSize);
        aclrtMalloc((void **)&dstDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile("./v1.bin", fileSize, srcHost0, fileSize);
        aclrtMemcpy(srcDevice0, fileSize, srcHost0, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        LaunchControl_flow_kernel_2d(srcDevice0, dstDevice0, stream);

    aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost0, fileSize, dstDevice0, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        WriteFile("./v2.bin", dstHost0, fileSize);

        aclrtFree(srcDevice0);
    aclrtFree(dstDevice0);
        aclrtFreeHost(srcHost0);
    aclrtFreeHost(dstHost0);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
