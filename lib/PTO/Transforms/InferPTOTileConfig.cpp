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
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_INFERPTOTILECONFIG
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static PTOArch getTargetArch(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
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

struct InferPTOTileConfigPass
    : public impl::InferPTOTileConfigBase<InferPTOTileConfigPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    PTOArch arch = getTargetArch(func);

    auto normalizeValue = [&](Value value) {
      auto tileTy = dyn_cast<TileBufType>(value.getType());
      if (!tileTy)
        return;
      auto normalizedTy = normalizeTileBufType(tileTy, arch);
      if (normalizedTy && normalizedTy != tileTy)
        value.setType(normalizedTy);
    };

    auto normalizeRegion = [&](Region &region, auto &self) -> void {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments())
          normalizeValue(arg);

        for (Operation &op : block) {
          for (Value result : op.getResults())
            normalizeValue(result);

          if (auto bind = dyn_cast<pto::BindTileOp>(op)) {
            auto currentConfig = bind.getConfigAttr();
            if (!currentConfig)
              currentConfig = TileBufConfigAttr::getDefault(&getContext());
            auto desiredConfig = inferMemRefTileConfig(
                bind.getResult().getType(), arch, &getContext(), currentConfig);
            if (desiredConfig && desiredConfig != currentConfig)
              bind->setAttr("config", desiredConfig);
          } else if (auto pointerCast = dyn_cast<pto::PointerCastOp>(op)) {
            auto currentConfig =
                dyn_cast_or_null<TileBufConfigAttr>(pointerCast->getAttr("config"));
            if (!currentConfig)
              currentConfig = TileBufConfigAttr::getDefault(&getContext());
            auto desiredConfig = inferMemRefTileConfig(
                pointerCast.getResult().getType(), arch, &getContext(),
                currentConfig);
            if (desiredConfig && desiredConfig != currentConfig)
              pointerCast->setAttr("config", desiredConfig);
          }

          for (Region &nested : op.getRegions())
            self(nested, self);
        }
      }
    };

    normalizeRegion(func.getBody(), normalizeRegion);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOTileConfigPass() {
  return std::make_unique<InferPTOTileConfigPass>();
}
