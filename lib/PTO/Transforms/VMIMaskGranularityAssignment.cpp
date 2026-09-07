// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMIMaskGranularityAssignment.cpp - Assign VMI mask granularity -----===//
//===----------------------------------------------------------------------===//
//
// This pass assigns concrete b8/b16/b32 granularity to VMI mask values before
// layout assignment.  It deliberately does not choose layouts: mask layout is
// assigned later by vmi-layout-assignment.  When a mask value has conflicting
// granularity uses, this pass keeps the value's primary granularity and either
// rematerializes cheap mask producers at the use site or inserts
// pto.vmi.ensure_mask_granularity.

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMIMASKGRANULARITYASSIGNMENT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr unsigned kMaskGranularity8Bit = 8;
constexpr unsigned kMaskGranularity16Bit = 16;
constexpr unsigned kMaskGranularity32Bit = 32;
constexpr unsigned kInvalidMaskId = ~0U;

struct MaskNode {
  Value value;
  VMIMaskType type;
  unsigned parent = 0;
  std::string granularity;
};

struct MaskUseRequest {
  OpOperand *operand;
  std::string granularity;
};

static unsigned getElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return mlir::pto::kValue64;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

static StringRef getMaskGranularityForElement(Type elementType) {
  switch (getElementBitWidth(elementType)) {
  case kMaskGranularity8Bit:
    return "b8";
  case kMaskGranularity16Bit:
    return "b16";
  case kMaskGranularity32Bit:
    return "b32";
  default:
    return "";
  }
}

static bool containsVMIType(Type type) {
  if (isa<VMIVRegType, VMIMaskType>(type)) {
    return true;
  }
  if (auto functionType = dyn_cast<FunctionType>(type)) {
    return llvm::any_of(functionType.getInputs(), containsVMIType) ||
           llvm::any_of(functionType.getResults(), containsVMIType);
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return containsVMIType(shapedType.getElementType());
  }
  return false;
}

struct MaskGranularitySolver {
  explicit MaskGranularitySolver(ModuleOp module)
      : module(module), ctx(module.getContext()) {}

  unsigned addMaskValue(Value value) {
    auto type = dyn_cast<VMIMaskType>(value.getType());
    if (!type) {
      return kInvalidMaskId;
    }
    auto [it, inserted] = maskIds.try_emplace(value, maskNodes.size());
    if (inserted) {
      std::string granularity;
      if (VMIMaskType::isConcreteGranularity(type.getGranularity())) {
        granularity = type.getGranularity().str();
      }
      maskNodes.push_back(MaskNode{value, type, it->second, granularity});
    }
    return it->second;
  }

  unsigned findMask(unsigned id) {
    if (maskNodes[id].parent == id) {
      return id;
    }
    maskNodes[id].parent = findMask(maskNodes[id].parent);
    return maskNodes[id].parent;
  }

  LogicalResult uniteMask(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addMaskValue(lhs);
    unsigned rhsId = addMaskValue(rhs);
    if (lhsId == kInvalidMaskId || rhsId == kInvalidMaskId) {
      return success();
    }
    unsigned lhsRoot = findMask(lhsId);
    unsigned rhsRoot = findMask(rhsId);
    if (lhsRoot == rhsRoot) {
      return success();
    }

    MaskNode &lhsNode = maskNodes[lhsRoot];
    MaskNode &rhsNode = maskNodes[rhsRoot];
    if (!lhsNode.granularity.empty() && !rhsNode.granularity.empty() &&
        lhsNode.granularity != rhsNode.granularity) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "conflicting mask granularities "
                             << lhsNode.granularity << " and "
                             << rhsNode.granularity;
    }

    rhsNode.parent = lhsRoot;
    if (lhsNode.granularity.empty()) {
      lhsNode.granularity = rhsNode.granularity;
    }
    return success();
  }

  LogicalResult requestMask(Value mask, StringRef granularity, Operation *op) {
    unsigned id = addMaskValue(mask);
    if (id == ~0U) {
      return success();
    }
    if (granularity.empty()) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "cannot infer concrete mask granularity";
    }
    MaskNode &node = maskNodes[findMask(id)];
    if (!node.granularity.empty() && node.granularity != granularity) {
      return op->emitError()
             << kVMIDiagLayoutContractPrefix
             << "conflicting mask granularities " << node.granularity << " and "
             << granularity;
    }
    node.granularity = granularity.str();
    return success();
  }

  LogicalResult requestMaskUse(OpOperand &operand, StringRef granularity,
                               Operation *op) {
    if (!isa<VMIMaskType>(operand.get().getType())) {
      return success();
    }
    if (granularity.empty()) {
      return op->emitError() << kVMIDiagLayoutContractPrefix
                             << "cannot infer concrete mask use granularity";
    }
    maskUseRequests.push_back(MaskUseRequest{&operand, granularity.str()});
    return success();
  }

  LogicalResult collect() {
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        addMaskValue(result);
      }
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            addMaskValue(arg);
          }
        }
      }
    });
    return success();
  }

  static std::optional<WalkResult> constraintResult(LogicalResult result) {
    return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
  }

  std::optional<WalkResult> addMaskValueConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIMaskAndOp, VMIMaskOrOp, VMIMaskXOrOp>([this, op](auto maskOp) {
          bool failedToUnite =
              failed(uniteMask(maskOp.getLhs(), maskOp.getRhs(), op)) ||
              failed(uniteMask(maskOp.getLhs(), maskOp.getResult(), op));
          return constraintResult(failure(failedToUnite));
        })
        .Case<VMIMaskNotOp, VMIEnsureMaskLayoutOp>([this, op](auto maskOp) {
          return constraintResult(
              uniteMask(maskOp.getSource(), maskOp.getResult(), op));
        })
        .Case<VMICmpFOp, VMICmpIOp>([this, op](auto compareOp) {
          auto lhsType = cast<VMIVRegType>(compareOp.getLhs().getType());
          return constraintResult(requestMask(
              compareOp.getResult(),
              getMaskGranularityForElement(lhsType.getElementType()), op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addResultMaskUseConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMISelectOp, VMIActivePrefixIndexOp, VMICompressOp,
              VMIStrideLoadOp, VMIMaskedLoadOp, VMIGatherOp, VMIExpandLoadOp>(
            [this, op](auto maskOp) {
              auto resultType = cast<VMIVRegType>(maskOp.getResult().getType());
              return constraintResult(requestMaskUse(
                  maskOp.getMaskMutable(),
                  getMaskGranularityForElement(resultType.getElementType()),
                  op));
            })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addSourceMaskUseConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp,
              VMIShrSOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getSrc(), op));
        })
        .Case<VMIReduceAddIOp, VMIReduceAddFOp, VMIReduceMaxFOp,
              VMIReduceMinFOp, VMIReduceMaxIOp, VMIReduceMinIOp,
              VMIGroupReduceAddFOp, VMIGroupReduceMaxFOp,
              VMIGroupReduceMinFOp, VMIGroupReduceAddIOp,
              VMIGroupReduceMaxIOp, VMIGroupReduceMinIOp>(
            [this, op](auto maskOp) {
              return constraintResult(requestMaskUseForSource(
                  maskOp.getMaskMutable(), maskOp.getSource(), op));
            })
        .Case<VMIVexpdifOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getX(), op));
        })
        .Case<VMIVintlvOp, VMIVdintlvOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getLhs(), op));
        })
        .Case<VMIMaskedStoreOp, VMIStrideStoreOp, VMIScatterOp,
              VMICompressStoreOp>([this, op](auto maskOp) {
          return constraintResult(requestMaskUseForSource(
              maskOp.getMaskMutable(), maskOp.getValue(), op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  std::optional<WalkResult> addSpecialMaskConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<VMIVmullOp>([this, op](VMIVmullOp vmull) {
          return constraintResult(requestMaskUse(vmull.getMaskMutable(),
                                                 "b32", op));
        })
        .Case<VMIVaddcOp>([this, op](VMIVaddcOp addc) {
          bool failedToConstrain =
              failed(requestMaskUse(addc.getMaskMutable(), "b32", op)) ||
              failed(requestMask(addc.getCarry(), "b32", op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<VMIVaddcsOp>([this, op](VMIVaddcsOp addcs) {
          bool failedToConstrain =
              failed(requestMaskUse(addcs.getCarryInMutable(), "b32", op)) ||
              failed(requestMaskUse(addcs.getMaskMutable(), "b32", op)) ||
              failed(requestMask(addcs.getCarry(), "b32", op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<VMIVdhistOp, VMIVchistOp>([this, op](auto histogram) {
          return constraintResult(
              requestMaskUse(histogram.getMaskMutable(), "b8", op));
        })
        .Case<VMIVmulaOp>([this, op](VMIVmulaOp vmula) {
          if (vmula.getMask().empty()) {
            return constraintResult(success());
          }
          auto accType = cast<VMIVRegType>(vmula.getAcc().getType());
          return constraintResult(requestMaskUse(
              *vmula.getMaskMutable().begin(),
              getMaskGranularityForElement(accType.getElementType()), op));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  LogicalResult addSwitchConstraints(cf::SwitchOp switchOp) {
    if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                    switchOp.getDefaultOperands(), switchOp))) {
      return failure();
    }
    for (auto [destination, operands] :
         llvm::zip(switchOp.getCaseDestinations(), switchOp.getCaseOperands())) {
      if (failed(addBranchConstraints(destination, operands, switchOp))) {
        return failure();
      }
    }
    return success();
  }

  std::optional<WalkResult> addControlFlowConstraint(Operation *op) {
    return llvm::TypeSwitch<Operation *, std::optional<WalkResult>>(op)
        .Case<scf::IfOp>([this](auto controlOp) {
          return constraintResult(addIfConstraints(controlOp));
        })
        .Case<scf::ExecuteRegionOp>([this](auto controlOp) {
          return constraintResult(addExecuteRegionConstraints(controlOp));
        })
        .Case<scf::IndexSwitchOp>([this](auto controlOp) {
          return constraintResult(addIndexSwitchConstraints(controlOp));
        })
        .Case<scf::WhileOp>([this](auto controlOp) {
          return constraintResult(addWhileConstraints(controlOp));
        })
        .Case<scf::ForOp>([this](auto controlOp) {
          return constraintResult(addForConstraints(controlOp));
        })
        .Case<cf::BranchOp>([this, op](cf::BranchOp branch) {
          return constraintResult(addBranchConstraints(
              branch.getDest(), branch.getDestOperands(), op));
        })
        .Case<cf::CondBranchOp>([this, op](cf::CondBranchOp branch) {
          bool failedToConstrain =
              failed(addBranchConstraints(branch.getTrueDest(),
                                          branch.getTrueDestOperands(), op)) ||
              failed(addBranchConstraints(branch.getFalseDest(),
                                          branch.getFalseOperands(), op));
          return constraintResult(failure(failedToConstrain));
        })
        .Case<cf::SwitchOp>([this](auto switchOp) {
          return constraintResult(addSwitchConstraints(switchOp));
        })
        .Case<func::ReturnOp>([this](auto returnOp) {
          return constraintResult(addReturnConstraints(returnOp));
        })
        .Case<func::CallOp>([this](auto callOp) {
          return constraintResult(addCallConstraints(callOp));
        })
        .Default([](Operation *) { return std::nullopt; });
  }

  WalkResult addConstraint(Operation *op) {
    for (auto constraint : {addMaskValueConstraint(op),
                            addResultMaskUseConstraint(op),
                            addSourceMaskUseConstraint(op),
                            addSpecialMaskConstraint(op),
                            addControlFlowConstraint(op)}) {
      if (constraint) {
        return *constraint;
      }
    }
    bool invalidIndirectCall =
        op->getName().getStringRef() == "func.call_indirect" &&
        hasVMIValueTypes(op);
    if (invalidIndirectCall) {
      op->emitError() << kVMIDiagLayoutContractPrefix
                      << "VMI typed call requires a direct internal callee "
                         "with a body";
      return WalkResult::interrupt();
    }
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      bool invalidDeclaration = funcOp.empty() && hasVMIFunctionType(funcOp);
      if (invalidDeclaration) {
        funcOp.emitError()
            << kVMIDiagLayoutContractPrefix
            << "VMI typed function declaration requires an explicit "
               "external ABI materialization plan";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  }

  LogicalResult addConstraints() {
    WalkResult result =
        module.walk([this](Operation *op) { return addConstraint(op); });
    return failure(result.wasInterrupted());
  }

  LogicalResult requestMaskUseForSource(OpOperand &mask, Value source,
                                        Operation *op) {
    auto sourceType = dyn_cast<VMIVRegType>(source.getType());
    if (!sourceType) {
      return success();
    }
    return requestMaskUse(mask,
                          getMaskGranularityForElement(
                              sourceType.getElementType()),
                          op);
  }

  LogicalResult uniteEquivalentValues(Value lhs, Value rhs, Operation *op) {
    return uniteMask(lhs, rhs, op);
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned resultNo = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yieldOp = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        if (!yieldOp || resultNo >= yieldOp.getNumOperands()) {
          continue;
        }
        if (failed(uniteEquivalentValues(result, yieldOp.getOperand(resultNo),
                                         ifOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yieldOp,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yieldOp.getNumOperands()) {
        break;
      }
      if (failed(uniteEquivalentValues(result, yieldOp.getOperand(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp executeOp) {
    WalkResult result = executeOp.getRegion().walk([&](scf::YieldOp yieldOp) {
      if (yieldOp->getParentOp() != executeOp.getOperation()) {
        return WalkResult::advance();
      }
      if (failed(
              addYieldConstraints(executeOp->getResults(), yieldOp, executeOp))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitchOp) {
    auto addBlockTerminator = [&](Block &block) -> LogicalResult {
      auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
      if (!yieldOp) {
        return success();
      }
      return addYieldConstraints(indexSwitchOp->getResults(), yieldOp,
                                 indexSwitchOp);
    };
    if (failed(addBlockTerminator(indexSwitchOp.getDefaultBlock()))) {
      return failure();
    }
    for (unsigned idx = 0, e = indexSwitchOp.getNumCases(); idx < e; ++idx) {
      if (failed(addBlockTerminator(indexSwitchOp.getCaseBlock(idx)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    return VMIControlFlowSupport::addWhileConstraints(
        whileOp, [&](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    return VMIControlFlowSupport::addForConstraints(
        forOp, [&](Value lhs, Value rhs, Operation *op) {
          return uniteEquivalentValues(lhs, rhs, op);
        });
  }

  LogicalResult addBranchConstraints(Block *dest, OperandRange operands,
                                     Operation *op) {
    if (!dest) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= dest->getNumArguments()) {
        break;
      }
      if (failed(uniteEquivalentValues(operand, dest->getArgument(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    auto func = returnOp->getParentOfType<func::FuncOp>();
    if (!func) {
      return success();
    }

    auto it = firstReturnOperandsByFunc.find(func);
    if (it == firstReturnOperandsByFunc.end()) {
      SmallVector<Value> operands(returnOp.getOperands());
      firstReturnOperandsByFunc.try_emplace(func, std::move(operands));
      return success();
    }

    ArrayRef<Value> firstOperands = it->second;
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= firstOperands.size()) {
        break;
      }
      if (failed(
              uniteEquivalentValues(firstOperands[index], operand, returnOp))) {
        return failure();
      }
    }
    return success();
  }

  bool hasVMIValueTypes(Operation *op) {
    return llvm::any_of(op->getOperandTypes(), containsVMIType) ||
           llvm::any_of(op->getResultTypes(), containsVMIType);
  }

  bool hasVMIFunctionType(func::FuncOp func) {
    FunctionType type = func.getFunctionType();
    return llvm::any_of(type.getInputs(), containsVMIType) ||
           llvm::any_of(type.getResults(), containsVMIType);
  }

  LogicalResult addCallConstraints(func::CallOp callOp) {
    if (!hasVMIValueTypes(callOp)) {
      return success();
    }

    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || callee.empty()) {
      return callOp.emitError()
             << kVMIDiagLayoutContractPrefix
             << "VMI typed call requires a direct internal callee with a body";
    }

    for (auto [operand, argument] :
         llvm::zip(callOp.getOperands(), callee.getArguments())) {
      if (failed(uniteEquivalentValues(operand, argument, callOp))) {
        return failure();
      }
    }

    SmallVector<func::ReturnOp> returns;
    callee.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
    for (func::ReturnOp returnOp : returns) {
      for (auto [index, result] : llvm::enumerate(callOp.getResults())) {
        if (index >= returnOp.getNumOperands()) {
          break;
        }
        if (failed(uniteEquivalentValues(result, returnOp.getOperand(index),
                                         callOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  void rewriteMaskTypes() {
    for (MaskNode &node : maskNodes) {
      MaskNode &root = maskNodes[findMask(maskIds.lookup(node.value))];
      StringRef granularity =
          root.granularity.empty() ? StringRef("b32") : StringRef(root.granularity);
      node.value.setType(VMIMaskType::get(ctx, node.type.getElementCount(),
                                          granularity,
                                          node.type.getLayoutAttr()));
    }
  }

  SmallVector<Type> getCallResultTypes(func::FuncOp func) {
    SmallVector<Type> resultTypes;
    bool found = false;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() != func.getSymName()) {
        return;
      }
      if (!found) {
        resultTypes.assign(call.getResultTypes().begin(),
                           call.getResultTypes().end());
        found = true;
        return;
      }
      if (resultTypes.size() != call.getNumResults()) {
        return;
      }
      for (auto [index, type] : llvm::enumerate(call.getResultTypes())) {
        if (index < resultTypes.size() && resultTypes[index] != type) {
          resultTypes[index] = {};
        }
      }
    });
    return found ? resultTypes : SmallVector<Type>{};
  }

  void rewriteFunctionType() {
    module.walk([&](func::FuncOp func) {
      if (func.empty()) {
        return;
      }

      SmallVector<Type> inputs;
      inputs.reserve(func.getNumArguments());
      for (BlockArgument arg : func.getArguments()) {
        inputs.push_back(arg.getType());
      }

      SmallVector<Type> results;
      SmallVector<Type> callResultTypes = getCallResultTypes(func);
      auto it = firstReturnOperandsByFunc.find(func);
      if (!callResultTypes.empty()) {
        for (Type type : callResultTypes) {
          results.push_back(type ? type : Type{});
        }
      } else if (it != firstReturnOperandsByFunc.end()) {
        for (Value operand : it->second) {
          results.push_back(operand.getType());
        }
      } else {
        FunctionType functionType = func.getFunctionType();
        for (Type type : functionType.getResults()) {
          if (auto maskType = dyn_cast<VMIMaskType>(type)) {
            StringRef granularity =
                VMIMaskType::isConcreteGranularity(maskType.getGranularity())
                    ? maskType.getGranularity()
                    : StringRef("b32");
            results.push_back(VMIMaskType::get(
                ctx, maskType.getElementCount(), granularity,
                maskType.getLayoutAttr()));
          } else {
            results.push_back(type);
          }
        }
      }

      for (auto [index, type] : llvm::enumerate(results)) {
        if (!type) {
          results[index] = func.getFunctionType().getResult(index);
        }
      }

      func.setFunctionType(FunctionType::get(ctx, inputs, results));
    });
  }

  LogicalResult insertMaskUseMaterializations() {
    OpBuilder builder(ctx);
    for (MaskUseRequest request : maskUseRequests) {
      Value value = request.operand->get();
      auto sourceType = dyn_cast<VMIMaskType>(value.getType());
      if (!sourceType) {
        continue;
      }
      if (sourceType.getGranularity() == request.granularity) {
        continue;
      }

      builder.setInsertionPoint(request.operand->getOwner());
      auto resultType = VMIMaskType::get(ctx, sourceType.getElementCount(),
                                         request.granularity,
                                         sourceType.getLayoutAttr());
      Value current = rematerializeMaskProducer(
          value, resultType, request.operand->getOwner()->getLoc(), builder);
      if (!current) {
        current = builder.create<VMIEnsureMaskGranularityOp>(
            request.operand->getOwner()->getLoc(), resultType, value);
      }
      request.operand->set(current);
    }
    return success();
  }

  Value rematerializeMaskProducer(Value value, VMIMaskType resultType,
                                  Location loc, OpBuilder &builder) {
    if (auto createMask = value.getDefiningOp<VMICreateMaskOp>()) {
      return builder
          .create<VMICreateMaskOp>(loc, resultType, createMask.getActiveLanes())
          .getResult();
    }

    if (auto createGroupMask = value.getDefiningOp<VMICreateGroupMaskOp>()) {
      return builder
          .create<VMICreateGroupMaskOp>(
              loc, resultType, createGroupMask.getActiveElemsPerGroup(),
              createGroupMask.getNumGroupsAttr(),
              createGroupMask.getGroupSizeAttr())
          .getResult();
    }

    if (auto constantMask = value.getDefiningOp<VMIConstantMaskOp>()) {
      return builder
          .create<VMIConstantMaskOp>(loc, resultType,
                                     constantMask.getValueAttr())
          .getResult();
    }

    return {};
  }

  LogicalResult run() {
    if (failed(collect())) {
      return failure();
    }
    if (failed(addConstraints())) {
      return failure();
    }
    rewriteMaskTypes();
    rewriteFunctionType();
    return insertMaskUseMaterializations();
  }

  ModuleOp module;
  MLIRContext *ctx;
  DenseMap<Value, unsigned> maskIds;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;
  SmallVector<MaskNode> maskNodes;
  SmallVector<MaskUseRequest> maskUseRequests;
};

struct VMIMaskGranularityAssignmentPass
    : public mlir::pto::impl::VMIMaskGranularityAssignmentBase<
          VMIMaskGranularityAssignmentPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIMaskGranularityAssignmentPass)

  void runOnOperation() override {
    if (failed(MaskGranularitySolver(getOperation()).run())) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createVMIMaskGranularityAssignmentPass() {
  return std::make_unique<VMIMaskGranularityAssignmentPass>();
}
