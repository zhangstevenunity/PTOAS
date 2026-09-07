// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_PTOLAYOUTUTILS_H
#define PTO_IR_PTOLAYOUTUTILS_H

#include "PTO/IR/PTO.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace mlir::pto {

inline constexpr unsigned kPTOLayoutRank = 5;

std::optional<std::string>
getNZViewCompatibilityError(ArrayRef<int64_t> shape5D,
                            ArrayRef<int64_t> stride5D,
                            unsigned storageElemBytes);

std::optional<std::string>
getNZSubviewCompatibilityError(ArrayRef<int64_t> sourceShape5D,
                               ArrayRef<int64_t> offsets5D,
                               ArrayRef<int64_t> sizes5D);

bool isLayoutCompatible5D(Layout layout, ArrayRef<int64_t> shape,
                          ArrayRef<int64_t> stride, unsigned storageElemBytes);

std::optional<Layout>
inferLayout5D(ArrayRef<int64_t> shape, ArrayRef<int64_t> stride,
              unsigned storageElemBytes,
              std::optional<Layout> preferredMinor2D = std::nullopt,
              bool *isMinor2DAmbiguous = nullptr);

} // namespace mlir::pto

#endif // PTO_IR_PTOLAYOUTUTILS_H
