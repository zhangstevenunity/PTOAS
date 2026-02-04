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

@LAUNCH_DECL@

int main() {
    @PARAM_DECLS@

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    @ALLOC_HOST@
    @ALLOC_DEVICE@

    @READ_INPUTS@
    @COPY_TO_DEVICE@
    @LAUNCH_CALL@

    aclrtSynchronizeStream(stream);
    @COPY_BACK@

    @WRITE_OUTPUT@

    @FREE_DEVICE@
    @FREE_HOST@
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
