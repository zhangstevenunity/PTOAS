//===- InferPTOTileConfig.cpp - Infer arch-aware tile config -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_INFERPTOTILECONFIG
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static PTOArch getTargetArch(Operation *op) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    module = op->getParentOfType<ModuleOp>();
  if (!module)
    return PTOArch::A3;
  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  if (arch && arch.getValue().equals_insensitive("a5"))
    return PTOArch::A5;
  return PTOArch::A3;
}

static TileBufConfigAttr inferTileConfigForSpace(MLIRContext *ctx,
                                                 AddressSpace space,
                                                 PTOArch arch,
                                                 PadValueAttr padAttr) {
  BLayout blayout = BLayout::RowMajor;
  SLayout slayout = SLayout::NoneBox;
  int32_t fractal = 512;

  switch (space) {
  case AddressSpace::LEFT:
    blayout = arch == PTOArch::A5 ? BLayout::ColMajor : BLayout::RowMajor;
    slayout = SLayout::RowMajor;
    fractal = 512;
    break;
  case AddressSpace::RIGHT:
    blayout = BLayout::RowMajor;
    slayout = SLayout::ColMajor;
    fractal = 512;
    break;
  case AddressSpace::ACC:
    blayout = BLayout::ColMajor;
    slayout = SLayout::RowMajor;
    fractal = 1024;
    break;
  default:
    return {};
  }

  Builder builder(ctx);
  if (!padAttr)
    padAttr = PadValueAttr::get(ctx, PadValue::Null);
  return TileBufConfigAttr::get(
      ctx, BLayoutAttr::get(ctx, blayout), SLayoutAttr::get(ctx, slayout),
      builder.getI32IntegerAttr(fractal), padAttr);
}

static TileBufType normalizeTileBufType(TileBufType tileTy, PTOArch arch) {
  if (tileTy.hasExplicitConfig())
    return tileTy;

  auto spaceAttr =
      dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!spaceAttr)
    return {};

  auto currentConfig = tileTy.getConfigAttr();
  auto desiredConfig = inferTileConfigForSpace(
      tileTy.getContext(), spaceAttr.getAddressSpace(), arch,
      dyn_cast_or_null<PadValueAttr>(currentConfig.getPad()));
  if (!desiredConfig)
    return {};

  if (desiredConfig == currentConfig)
    return tileTy;

  return TileBufType::get(tileTy.getContext(), tileTy.getShape(),
                          tileTy.getElementType(), tileTy.getMemorySpace(),
                          tileTy.getValidShape(), desiredConfig);
}

static TileBufConfigAttr inferMemRefTileConfig(Type memrefLikeType, PTOArch arch,
                                               MLIRContext *ctx,
                                               TileBufConfigAttr currentConfig) {
  auto memrefTy = dyn_cast<BaseMemRefType>(memrefLikeType);
  if (!memrefTy)
    return {};
  auto spaceAttr = dyn_cast_or_null<AddressSpaceAttr>(memrefTy.getMemorySpace());
  if (!spaceAttr)
    return {};
  return inferTileConfigForSpace(
      ctx, spaceAttr.getAddressSpace(), arch,
      currentConfig ? dyn_cast_or_null<PadValueAttr>(currentConfig.getPad())
                    : PadValueAttr());
}

static Type normalizeType(Type type, PTOArch arch) {
  auto tileTy = dyn_cast<TileBufType>(type);
  if (!tileTy)
    return type;
  auto normalizedTy = normalizeTileBufType(tileTy, arch);
  return normalizedTy ? Type(normalizedTy) : type;
}

static bool normalizeValue(Value value, PTOArch arch) {
  Type currentType = value.getType();
  Type normalizedType = normalizeType(currentType, arch);
  if (normalizedType == currentType)
    return false;
  value.setType(normalizedType);
  return true;
}

static LogicalResult syncFunctionSignature(func::FuncOp func, PTOArch arch) {
  SmallVector<Type> newInputs;
  SmallVector<Type> newResults;

  if (func.isExternal()) {
    llvm::transform(func.getArgumentTypes(), std::back_inserter(newInputs),
                    [&](Type type) { return normalizeType(type, arch); });
    llvm::transform(func.getResultTypes(), std::back_inserter(newResults),
                    [&](Type type) { return normalizeType(type, arch); });
  } else {
    Block &entry = func.front();
    newInputs.assign(entry.getArgumentTypes().begin(), entry.getArgumentTypes().end());

    if (func.getNumResults() != 0) {
      bool sawReturn = false;
      func.walk([&](func::ReturnOp ret) {
        SmallVector<Type> operandTypes(ret.getOperandTypes().begin(),
                                       ret.getOperandTypes().end());
        if (!sawReturn) {
          newResults = operandTypes;
          sawReturn = true;
          return;
        }
        if (newResults != operandTypes) {
          ret.emitOpError("all return ops must agree on result types after "
                          "tile config inference");
          func.emitError("inconsistent function result types after tile config "
                         "inference");
        }
      });
      if (!sawReturn)
        return func.emitOpError("non-external function with results must have "
                                "a return op after tile config inference");
    }
  }

  auto newFunctionType = FunctionType::get(func.getContext(), newInputs, newResults);
  if (newFunctionType != func.getFunctionType())
    func.setFunctionType(newFunctionType);
  return success();
}

static LogicalResult syncCallSites(ModuleOp module, func::FuncOp callee) {
  auto uses = callee.getSymbolUses(module);
  if (!uses)
    return success();

  for (SymbolTable::SymbolUse use : *uses) {
    auto call = dyn_cast<func::CallOp>(use.getUser());
    if (!call)
      continue;

    auto expectedInputs = callee.getFunctionType().getInputs();
    if (call.getNumOperands() != expectedInputs.size())
      return call.emitOpError("operand count does not match updated callee "
                              "signature for ")
             << callee.getSymName();

    for (auto [idx, operand] : llvm::enumerate(call.getArgOperands())) {
      if (operand.getType() != expectedInputs[idx]) {
        return call.emitOpError("operand type does not match updated callee "
                                "signature at index ")
               << idx << " for " << callee.getSymName();
      }
    }

    if (llvm::equal(call.getResultTypes(), callee.getResultTypes()))
      continue;

    OpBuilder builder(call);
    auto newCall =
        builder.create<func::CallOp>(call.getLoc(), callee, call.getArgOperands());
    newCall->setAttrs(call->getAttrs());
    call.replaceAllUsesWith(newCall.getResults());
    call.erase();
  }

  return success();
}

struct InferPTOTileConfigPass
    : public impl::InferPTOTileConfigBase<InferPTOTileConfigPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    PTOArch arch = getTargetArch(module);

    auto normalizeRegion = [&](Region &region, auto &self) -> void {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments())
          (void)normalizeValue(arg, arch);

        for (Operation &op : block) {
          for (Value result : op.getResults())
            (void)normalizeValue(result, arch);

          if (auto pointerCast = dyn_cast<pto::PointerCastOp>(op)) {
            auto currentConfig = pointerCast.getConfig();
            if (!currentConfig) {
              auto desiredConfig = inferMemRefTileConfig(
                  pointerCast.getResult().getType(), arch, &getContext(),
                  TileBufConfigAttr());
              if (desiredConfig)
                pointerCast->setAttr("config", desiredConfig);
            }
          }

          for (Region &nested : op.getRegions())
            self(nested, self);
        }
      }
    };

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func.isExternal())
        normalizeRegion(func.getBody(), normalizeRegion);
      if (failed(syncFunctionSignature(func, arch))) {
        signalPassFailure();
        return;
      }
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (failed(syncCallSites(module, func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOTileConfigPass() {
  return std::make_unique<InferPTOTileConfigPass>();
}
