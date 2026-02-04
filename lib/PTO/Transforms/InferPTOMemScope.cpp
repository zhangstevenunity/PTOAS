//===- InferPTOMemScope.cpp - Infer Memory Scope for pto Ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InferPTOMemScope.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"


#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "PTO-infer-mem-scope"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << X << "\n")

namespace mlir {
#define GEN_PASS_DEF_INFERPTOMEMSCOPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace pto;

LogicalResult
MemScopeInferAndPropagateHelper::propagateMemScopeToUsers(Value val) {
  // Get new memory scope from result.
  auto memrefScope = getPTOAddressSpaceAttr(val.getType());
  // This function propagates the type change of an SSA result to the operation
  // that uses it. The result type of the updated operation might be affected,
  // so we need to cascade the change.
  auto propagateFn = [&](OpOperand &user) -> LogicalResult {
    Operation *userDefiningOp = user.getOwner();
    return TypeSwitch<Operation *, LogicalResult>(userDefiningOp)
        .Case<scf::YieldOp>([&](scf::YieldOp op) {
          Operation *parentOp = op->getParentOp();
          auto yieldResult = op.getOperand(user.getOperandNumber());
          auto parentResult = parentOp->getResult(user.getOperandNumber());

          Type yieldType = yieldResult.getType();
          Type valType = val.getType();
          if (!isa<BaseMemRefType>(yieldType))
            return success();
          if (!isa<BaseMemRefType>(valType))
            return success();
          auto mtype = dyn_cast<BaseMemRefType>(yieldType);
          auto vtype = dyn_cast<BaseMemRefType>(valType);
          if (mtype.getElementType() != vtype.getElementType())
            return success();
          setBaseMemRefTypeScope(parentResult, memrefScope);
          if (failed(propagateMemScopeToUsers(parentResult))) {
            return failure();
          }
          return success();
        })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          auto result = op.getTiedLoopResult(&user);
          setBaseMemRefTypeScope(result, memrefScope);
          auto bbArg = op.getTiedLoopRegionIterArg(&user);
          setBaseMemRefTypeScope(bbArg, memrefScope);
          return success(propagateMemScopeToUsers(bbArg).succeeded() &&
                         propagateMemScopeToUsers(result).succeeded());
        })
        .Case<memref::SubViewOp, memref::ViewOp, memref::ReinterpretCastOp,
              memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::ReshapeOp, memref::TransposeOp,
              memref::ExtractStridedMetadataOp>([&](auto op) {
          auto result = op->getResult(0);
          setBaseMemRefTypeScope(result, memrefScope);
          return propagateMemScopeToUsers(result);
        })
        // .Case<pto::BitcastOp>([&](auto op) {
        //   auto result = op->getResult(0);
        //   setBaseMemRefTypeScope(result, memrefScope);
        //   return propagateMemScopeToUsers(result);
        // })
        .Case<func::CallOp>([&](auto op) {
          // For function calls, we cannot propagate the memory scope because
          // we don't know the relationship between the inputs and results.
          // But we don't need to report failure because we can run propagation
          // for the results.
          return success();
        })
        .Case<gpu::LaunchFuncOp>([&](auto op) {
          // Same as above
          return success();
        })
        .Default([&](Operation *op) {
          // Don't need to update Ops that don't have results.
          if (op->getNumResults() == 0) {
            return success();
          }
          // Or results that are not memrefs.
          if (llvm::none_of(op->getResults(), [&](OpResult result) {
                return isa<MemRefType>(result.getType());
              })) {
            return success();
          }
          op->emitOpError("Unsupported user for root alloc op.");
          return failure();
        });
  };
  // Iterate over the users of the val.
  for (OpOperand &user : val.getUses()) {
    // Update the type of the result that corresponds to the operand.
    if (failed(propagateFn(user))) {
      return failure();
    }
  }
  return success();
}

LogicalResult
MemScopeInferAndPropagateHelper::Run(Value operand,
                                     const AddressSpaceAttr &targetMemScope) {
  auto memRefType = dyn_cast<BaseMemRefType>(operand.getType());
  if (!memRefType) {
    return failure();
  }

  auto memSpace = memRefType.getMemorySpace();
  if (memSpace) {
    if (memSpace != targetMemScope) {
      return failure();
    }
    return success();
  }

  // Update its scope.
  setBaseMemRefTypeScope(operand, targetMemScope);

  // Propagate the new memref type to its users.
  return propagateMemScopeToUsers(operand);
}

namespace {
struct InferPTOMemScopePass
    : public impl::InferPTOMemScopeBase<InferPTOMemScopePass> {
  void runOnOperation() override;

private:
  LogicalResult fixDeviceCallSite(func::FuncOp op);
  LogicalResult fixHostFuncSignature(func::FuncOp op);
};
} // namespace

LogicalResult pto::inferAndPropagateMemScopeForMovDps(pto::MovDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  Value mA = op.getSrc();
  Value mB = op.getDst();

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  auto memRefType = dyn_cast<BaseMemRefType>(allocB.value().getType());
  if (!memRefType) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }

  auto memSpace = memRefType.getMemorySpace();
  if (!memSpace) {
    return success();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);
  auto l1SpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::MAT);
  auto ubSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::VEC);
  auto biasSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::BIAS);

  MemScopeInferAndPropagateHelper helper;

  if (memSpace == ubSpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, ubSpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  if (memSpace == l1SpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, l0cSpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  if (memSpace == l0aSpaceAttr ||
      memSpace == l0bSpaceAttr ||
      memSpace == biasSpaceAttr) {
    // For MmadL1Op, operand mA should be in L1.
    if (failed(helper.Run(*allocA, l1SpaceAttr))) {
      return op->emitOpError("Failed to infer/propagate memory scope for mA");
    }
    return success();
  }

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulAccDps(pto::MatmulAccDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mAcc = op.getAccIn();
  Value mA = op.getLhs();
  Value mB = op.getRhs();
  Value mC = op.getDst();

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocAcc = tracebackMemRefToAlloc(mAcc);
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);
  

  if (!allocAcc.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mAcc of this op.";
    return failure();
  }
  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);

  MemScopeInferAndPropagateHelper helper;

   // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocAcc, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mAcc");
  }

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}


LogicalResult pto::inferAndPropagateMemScopeForMatmulBiasDps(pto::MatmulBiasDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mA = op.getA();
  Value mB = op.getB();
  Value mC = op.getDst(); 
  Value mD = op.getBias(); 

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);
  auto allocD = tracebackMemRefToAlloc(mD);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }
  if (!allocD.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mD of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);
  auto l0dSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::BIAS);

  MemScopeInferAndPropagateHelper helper;

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mD should be in BIAS.
  if (failed(helper.Run(*allocD, l0dSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForMatmulDps(pto::MatmulDpsOp op) {
  // 替换 hasPureBufferSemantics()
  // 在 PTO 的语义中，如果 Op 没有返回值 (Result)，就意味着它是 Buffer 语义（操作的是 TileBuf 或 MemRef）
  if (op.getNumResults() != 0) {
    return op->emitOpError("Run infer memory scope after bufferization (Op must have 0 results).");
  }

  // 替换 getDpsInputOperand/getDpsInitOperand
  // 直接使用 ODS 生成的命名函数，更直观且安全
  // 原逻辑: Input(0)->LHS, Input(1)->RHS, Init(0)->DST
  Value mA = op.getLhs();
  Value mB = op.getRhs();
  Value mC = op.getDst(); 

  // 直接使用 Value，不需要再调 ->get()
  // mA, mB, mC 现在已经是 Value 类型了
  auto allocA = tracebackMemRefToAlloc(mA);
  auto allocB = tracebackMemRefToAlloc(mB);
  auto allocC = tracebackMemRefToAlloc(mC);

  if (!allocA.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mA of this op.";
    return failure();
  }
  if (!allocB.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mB of this op.";
    return failure();
  }
  if (!allocC.has_value()) {
    emitError(op.getLoc()) << "Cannot find root memref.alloc for mC of this op.";
    return failure();
  }

  auto l0aSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::LEFT);
  auto l0bSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::RIGHT);
  auto l0cSpaceAttr = AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::ACC);

  MemScopeInferAndPropagateHelper helper;

  // For MmadL1Op, operand mA should be in L1.
  if (failed(helper.Run(*allocA, l0aSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mA");
  }
  LDBG("IR after setting mem scope for mA:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mB should be in L1.
  if (failed(helper.Run(*allocB, l0bSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mB");
  }
  LDBG("IR after setting mem scope for mB:\n" << *(op->getParentOfType<ModuleOp>()));

  // For MmadL1Op, operand mC should be in L0C.
  if (failed(helper.Run(*allocC, l0cSpaceAttr))) {
    return op->emitOpError("Failed to infer/propagate memory scope for mC");
  }
  LDBG("IR after setting mem scope for mC:\n" << *(op->getParentOfType<ModuleOp>()));

  return success();
}

LogicalResult InferPTOMemScopePass::fixDeviceCallSite(func::FuncOp op) {
  LDBG("Begin fixing call site for " << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  SymbolTable::UseRange uses = *op.getSymbolUses(getOperation());
  for (SymbolTable::SymbolUse use : uses) {
    func::CallOp call = cast<func::CallOp>(use.getUser());
    // propagate call operand's memory scope
    for (auto [idx, callOperand] : llvm::enumerate(call.getArgOperands())) {
      if (!isa<BaseMemRefType>(callOperand.getType()))
        continue;

      auto funcOperandType = op.getFunctionType().getInput(idx);
      if (!isa<BaseMemRefType>(funcOperandType))
        continue;

      LDBG("call operand: " << callOperand);
      if (failed(helper.Run(tracebackMemRef(callOperand),
                            getPTOAddressSpaceAttr(funcOperandType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for operand "
               << callOperand;
      }
      LDBG("call operand after: " << callOperand);
    }
    // propagate call return value memory scope
    for (auto [idx, returnValue] : llvm::enumerate(call->getResults())) {
      if (!isa<BaseMemRefType>(returnValue.getType()))
        continue;

      auto funcReturnType = op.getFunctionType().getResult(idx);
      if (!isa<BaseMemRefType>(funcReturnType))
        continue;

      if (failed(helper.Run(returnValue,
                            getPTOAddressSpaceAttr(funcReturnType)))) {
        return op->emitOpError()
               << "Failed to propagate memory scope for result " << returnValue;
      }
    }
  }
  return success();
}

/// Update the function type for the host function.
///
/// Because we propagate information from the call site to the caller, we only
/// updated the memref type of the BlockArgument of or the return operation
/// within the function (if they are updated at all). So we need to use those
/// information to update the function's type.
LogicalResult InferPTOMemScopePass::fixHostFuncSignature(func::FuncOp op) {
  // Skip external host functions because we know nothing about it.
  if (op.isExternal())
    return success();

  func::ReturnOp returnOp = getAssumedUniqueReturnOp(op);
  if (!returnOp)
    return failure();

  SmallVector<Type> newArgsType(llvm::map_to_vector(
      op.getArguments(), [](const BlockArgument &ba) { return ba.getType(); }));
  SmallVector<Type> newReturnType(llvm::map_to_vector(
      returnOp.getOperandTypes(), [](const Type &type) { return type; }));
  auto newFt = op.getFunctionType().clone(newArgsType, newReturnType);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult inferAndPropagateMemScopeForExternFunc(func::FuncOp op) {
  if (!op.isExternal())
    return failure();

  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  LDBG("Begin infer and propagate memory scope for extern func"
       << op.getSymName());
  auto newArgTypes = SmallVector<Type>(op.getArgumentTypes());
  for (auto &argType : newArgTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(argType)) {
      if (memrefType.getMemorySpace())
        continue;
      argType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  // For extern functions that have results, we assume that the memory scope
  // is Global Memory.
  auto newReturnTypes = SmallVector<Type>(op.getResultTypes());
  for (auto &resultType : newReturnTypes) {
    // If not base memref and already has memspace then skip
    if (auto memrefType = dyn_cast<BaseMemRefType>(resultType)) {
      if (memrefType.getMemorySpace())
        continue;
      resultType = getBaseMemRefTypeWithNewScope(memrefType, gmSpaceAttr);
    }
  }
  auto newFt = op.getFunctionType().clone(newArgTypes, newReturnTypes);
  op.setFunctionType(newFt);
  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForFunc(func::FuncOp op) {
  if (op.isExternal())
    return inferAndPropagateMemScopeForExternFunc(op);

  LDBG("Begin infer and propagate memory scope for func" << op.getSymName());
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::VEC);
  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    if (op->hasAttr(pto::VectorFunctionAttr::name)) {
      if (failed(helper.Run(arg, ubSpaceAttr)))
        return op->emitOpError()
               << "Failed to propagate UB memory scope for argument # in VF"
               << arg.getArgNumber();
    } else if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }
  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }
  if (op->getNumResults() > 0)
    op.emitWarning()
        << "non-externl function has return value after bufferization!";

  return success();
}

LogicalResult pto::inferAndPropagateMemScopeForGpuFunc(gpu::GPUFuncOp op) {
  MemScopeInferAndPropagateHelper helper;
  auto gmSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::GM);

  auto args = op.getArguments();
  for (auto arg : args) {
    if (!isa<BaseMemRefType>(arg.getType())) {
      continue;
    }

    // TODO: handle case when ub arguments are passed in the GPUFuncOp
    if (failed(helper.Run(arg, gmSpaceAttr))) {
      return op->emitOpError()
             << "Failed to propagate memory scope for argument #"
             << arg.getArgNumber();
    }
  }

  if (!args.empty()) {
    auto newFt = op.getFunctionType().clone(
        op.getBody().front().getArgumentTypes(), op.getResultTypes());
    op.setFunctionType(newFt);
  }

  return success();
}

LogicalResult pto::inferAndPropagateUbufMemScope(memref::AllocOp op) {
  LDBG("Begin infer and propagate memory scope for: " << *op);
  auto memorySpace = op.getType().getMemorySpace();
  if (memorySpace)
    return success();

  MemScopeInferAndPropagateHelper helper;
  auto ubSpaceAttr =
      AddressSpaceAttr::get(op->getContext(), pto::AddressSpace::VEC);
  if (failed(helper.Run(op, ubSpaceAttr))) {
    return op->emitOpError("Failed to propagate memory scope ub for allocOp");
  }
  return success();
}

void InferPTOMemScopePass::runOnOperation() {
  llvm::errs() << "Hello PTO Infer Mem Scope!\n";
  auto op = getOperation();
  op->dump();

  SmallVector<func::FuncOp> deviceFuncList;
  SetVector<StringRef> deviceFuncNames;
  SmallVector<func::FuncOp> hostFuncList;
  getOperation()->walk([&](func::FuncOp func) {
    // if (!hacc::utils::isHost(func)) {
    //   deviceFuncList.push_back(func);
    //   deviceFuncNames.insert(func.getSymName());
    //   return;
    // }
    //hostFuncList.push_back(func);
    deviceFuncList.push_back(func);
    deviceFuncNames.insert(func.getSymName());
    return;
  });

  SmallVector<gpu::GPUFuncOp> gpuFuncList;
  getOperation()->walk([&](gpu::GPUModuleOp gpuModule) {
    gpuModule->walk([&](gpu::GPUFuncOp gpuFunc) -> void {
      gpuFuncList.push_back(gpuFunc);
    });
  });

  for (auto func : gpuFuncList) {
    if (failed(inferAndPropagateMemScopeForGpuFunc(func)))
      signalPassFailure();
  }

  // Infer and propagate memory scope for device functions.
  for (auto func : deviceFuncList) {
    // Set the memory scope of values related to `pto::MmadL1Op` to L1 or L0C.
    func->walk([&](mlir::pto::MatmulDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MatmulAccDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulAccDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MatmulBiasDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMatmulBiasDps(op)))
        signalPassFailure();
    });

    func->walk([&](mlir::pto::MovDpsOp op) {
      if (failed(pto::inferAndPropagateMemScopeForMovDps(op)))
        signalPassFailure();
    });

    // Set device function arguments' memory scope to GM.
    if (failed(pto::inferAndPropagateMemScopeForFunc(func)))
      signalPassFailure();

    // Finally, set the remaining memory scope in the device kernel to UB.
    func->walk([&](memref::AllocOp op) {
      if (failed(pto::inferAndPropagateUbufMemScope(op)))
        signalPassFailure();
    });
  }

  for (auto func : deviceFuncList) {
    if (failed(fixDeviceCallSite(func)))
      signalPassFailure();
  }

  for (auto func : hostFuncList) {
    if (failed(fixHostFuncSignature(func)))
      signalPassFailure();
  }

  llvm::errs() << "end PTO Infer Mem Scope!\n";
  op = getOperation();
  op->dump();
}

std::unique_ptr<Pass> mlir::pto::createInferPTOMemScopePass() {
  return std::make_unique<InferPTOMemScopePass>();
}
