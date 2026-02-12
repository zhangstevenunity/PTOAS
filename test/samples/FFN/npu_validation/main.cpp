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

void LaunchFfn_fc1_block(float *v1, float *v2, float *v3, float *v4, float *v5, void *stream);
void LaunchFfn_act_block(float *v1, float *v2, float *v3, int32_t v4, int32_t v5, void *stream);
void LaunchFfn_fc2_block(float *v1, float *v2, float *v3, void *stream);

int main() {
        size_t fileSize = 32 * 32 * sizeof(float);
    int32_t v9 = 32;
    int32_t v10 = 32;
    float *dstHost0 = nullptr;
    float *dstDevice0 = nullptr;
    float *srcHost0 = nullptr;
    float *srcHost1 = nullptr;
    float *srcHost2 = nullptr;
    float *srcHost3 = nullptr;
    float *srcHost4 = nullptr;
    float *srcHost5 = nullptr;
    float *srcHost6 = nullptr;
    float *srcDevice0 = nullptr;
    float *srcDevice1 = nullptr;
    float *srcDevice2 = nullptr;
    float *srcDevice3 = nullptr;
    float *srcDevice4 = nullptr;
    float *srcDevice5 = nullptr;
    float *srcDevice6 = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

        aclrtMallocHost((void **)(&dstHost0), fileSize);
    aclrtMallocHost((void **)(&srcHost0), fileSize);
    aclrtMallocHost((void **)(&srcHost1), fileSize);
    aclrtMallocHost((void **)(&srcHost2), fileSize);
    aclrtMallocHost((void **)(&srcHost3), fileSize);
    aclrtMallocHost((void **)(&srcHost4), fileSize);
    aclrtMallocHost((void **)(&srcHost5), fileSize);
    aclrtMallocHost((void **)(&srcHost6), fileSize);
        aclrtMalloc((void **)&dstDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice1, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice2, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice3, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice4, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice5, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice6, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile("./v1.bin", fileSize, srcHost0, fileSize);
    ReadFile("./v2.bin", fileSize, srcHost1, fileSize);
    ReadFile("./v3.bin", fileSize, srcHost2, fileSize);
    ReadFile("./v4.bin", fileSize, srcHost3, fileSize);
    ReadFile("./v5.bin", fileSize, srcHost4, fileSize);
    ReadFile("./v6.bin", fileSize, srcHost5, fileSize);
    ReadFile("./v7.bin", fileSize, srcHost6, fileSize);
        aclrtMemcpy(srcDevice0, fileSize, srcHost0, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice1, fileSize, srcHost1, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice2, fileSize, srcHost2, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice3, fileSize, srcHost3, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice4, fileSize, srcHost4, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice5, fileSize, srcHost5, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice6, fileSize, srcHost6, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        LaunchFfn_fc1_block(srcDevice0, srcDevice1, srcDevice2, srcDevice4, srcDevice5, stream);
    LaunchFfn_act_block(srcDevice4, srcDevice5, srcDevice6, v9, v10, stream);
    LaunchFfn_fc2_block(srcDevice6, srcDevice3, dstDevice0, stream);

    aclrtSynchronizeStream(stream);
        aclrtMemcpy(dstHost0, fileSize, dstDevice0, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

        WriteFile("./v8.bin", dstHost0, fileSize);

        aclrtFree(srcDevice0);
    aclrtFree(srcDevice1);
    aclrtFree(srcDevice2);
    aclrtFree(srcDevice3);
    aclrtFree(srcDevice4);
    aclrtFree(srcDevice5);
    aclrtFree(srcDevice6);
    aclrtFree(dstDevice0);
        aclrtFreeHost(srcHost0);
    aclrtFreeHost(srcHost1);
    aclrtFreeHost(srcHost2);
    aclrtFreeHost(srcHost3);
    aclrtFreeHost(srcHost4);
    aclrtFreeHost(srcHost5);
    aclrtFreeHost(srcHost6);
    aclrtFreeHost(dstHost0);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
