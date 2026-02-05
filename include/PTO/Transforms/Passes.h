//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H

#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"
#include "PTO/IR/PTODialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL
#include "PTO/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createPTOHighDimLoweringPass();
std::unique_ptr<Pass> createPTOVFloopGatherPass();
std::unique_ptr<Pass> createLoweringSyncToPipePass();

// Creates a pass for ...
std::unique_ptr<Pass> createPTOInsertSyncPass();
std::unique_ptr<Pass> createEmitPTOManualPass();


/// Create a pass to convert ops from other dialects to PTO Ops.
std::unique_ptr<Pass> createConvertToPTOOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// PTO Ops.
std::unique_ptr<Pass> createInferPTOMemScopePass();

/// Create a pass to plan memory.
std::unique_ptr<Pass>
createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption = {});

std::unique_ptr<Pass> createPTOInsertCVMovPass();
std::unique_ptr<Pass> createPTOConvertToDPSPass();
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass();
std::unique_ptr<Pass> createPTOViewToMemrefPass();
std::unique_ptr<Pass> createEmitPTOManualPass();
std::unique_ptr<mlir::Pass> createPTOInsertLoadStoreForMixCVPass();
std::unique_ptr<Pass> createInferPTOLayoutPass();
// Declare register function
void registerPTOPasses();

} // namespace pto
} // namespace mlir

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#undef GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "PTO/Transforms/Passes.h.inc"

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
