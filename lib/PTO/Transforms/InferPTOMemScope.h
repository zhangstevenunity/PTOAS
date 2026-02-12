//===- InferPTOMemScope.h --Infer Memory Scope for PTO Ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef PTO_INFERPTOMEMSCOPE_H
#define PTO_INFERPTOMEMSCOPE_H

#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "Utils.h"

namespace mlir {
namespace pto {

class MemScopeInferAndPropagateHelper {
public:
  LogicalResult Run(Value operand, const AddressSpaceAttr &targetMemScope);

private:
  /// Propagate the memory scope change to users of the value.
  LogicalResult propagateMemScopeToUsers(Value val);

  /// Set memory scope for the root alloc op.
  void setMemRefAllocScope(memref::AllocOp op,
                           const AddressSpaceAttr &newScope);
  /// Set memory scope for the block argument.
  void setBlockArgumentScope(BlockArgument operand,
                             const AddressSpaceAttr &targetMemScope);
};

/// Infer, propagate, and set memory scope information to MmadL1Op.
/// \note MmadL1Op should be bufferized beforehand.
//LogicalResult inferAndPropagateMemScopeForMmadL1(MmadL1Op op);

LogicalResult inferAndPropagateMemScopeForMatmulDps(TMatmulOp op);
LogicalResult inferAndPropagateMemScopeForMatmulAccDps(TMatmulAccOp op);
LogicalResult inferAndPropagateMemScopeForMatmulBiasDps(TMatmulBiasOp op);
LogicalResult inferAndPropagateMemScopeForMovDps(TMovOp op);
/// Infer, propagate, and set memory scope information to FuncOp.
/// \note FuncOp should be bufferized beforehand.
LogicalResult inferAndPropagateMemScopeForFunc(func::FuncOp op);

/// Infer, propagate, and set memory scope information to AllocOp.
/// \note Set alloc memory scope to ub.
LogicalResult inferAndPropagateUbufMemScope(memref::AllocOp allocOp);

/// Infer, propagate, and set memory scope information to GPUFuncOp.
LogicalResult inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op);

} // namespace pto
} // namespace mlir

#endif // PTO_INFERPTOMEMSCOPE_H
