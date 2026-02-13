//===- ConvertToPTOOp.cpp - Convert ops to PTO Ops ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTOPTOOP
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace pto;

namespace {
//===---------------------------------------------------------------------===//
// Patterns that convert ops from other dialects to PTO ops.
//===---------------------------------------------------------------------===//

std::optional<Value> getPadValue(std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;
  // Compatible with older versions
//   for (auto *user : maybeAlloc.value()->getUsers()) {
//     if (llvm::isa_and_nonnull<pto::VBrcOp>(user) &&
//         user->getOperand(0).getType().isIntOrFloat()) {
//       return user->getOperand(0);
//     }
//   }

//   auto allocValue = maybeAlloc.value();
//   auto padMarkOp = utils::getAnnotateOpWithAttr(allocValue, "pad_const");
//   if (!padMarkOp.has_value())
//     return std::nullopt;
//   auto padValue = dyn_cast<annotation::MarkOp>(padMarkOp.value())
//                       .getDynamicAttrValue("pad_const");
//   return std::optional<Value>(padValue);
  return std::nullopt;
}

std::optional<Value> getLeftPadNum(PatternRewriter &rewriter,
                                   std::optional<memref::AllocOp> maybeAlloc) {
  if (!maybeAlloc.has_value())
    return std::nullopt;

  for (auto *user : maybeAlloc.value()->getUsers()) {
    if (auto subviewOp = llvm::dyn_cast<memref::SubViewOp>(user)) {
      auto offsets = subviewOp.getMixedOffsets();
      auto offset = offsets[offsets.size() - 1];
      Value offsetValue =
          offset.is<Value>()
              ? dyn_cast<Value>(offset)
              : rewriter.create<arith::ConstantIndexOp>(
                    subviewOp->getLoc(), getConstantIntValue(offset).value());
      return offsetValue;
    }
  }
  return std::nullopt;
}

std::pair<std::optional<Operation *>, std::optional<Value>>
getInitInfo(Operation *op, pto::TLoadOp loadOp) {
//   if (!llvm::isa<pto::VBrcOp>(op))
//     return {std::nullopt, std::nullopt};
//   if (!op->getOperand(0).getType().isIntOrFloat())
//     return {std::nullopt, std::nullopt};

//   if (op->getBlock() == loadOp->getBlock())
//     return {op, std::nullopt};
//   if (op->getParentOp()->getBlock() == loadOp->getBlock() &&
//       isa<scf::IfOp>(op->getParentOp())) {
//     auto ifOp = cast<scf::IfOp>(op->getParentOp());
//     return {op, ifOp.getCondition()};
//   }

  return {std::nullopt, std::nullopt};
}

std::pair<std::optional<Operation *>, std::optional<Value>>
getUniqueInitInfo(PatternRewriter &rewriter,
                  std::optional<memref::AllocOp> maybeAlloc,
                  pto::TLoadOp loadOp) {
  if (!maybeAlloc.has_value())
    return {std::nullopt, std::nullopt};

  std::optional<Operation *> initOp = std::nullopt;
  std::optional<Value> initCondition = std::nullopt;
  for (auto *user : (*maybeAlloc)->getUsers()) {
    if (llvm::isa<pto::TLoadOp>(user))
      continue;
    auto maybeInitOp = getInitInfo(user, loadOp).first;
    if (maybeInitOp.has_value() && !initOp.has_value()) {
      std::tie(initOp, initCondition) = getInitInfo(user, loadOp);
    } else if (user->getDialect()->getNamespace() ==
               PTODialect::getDialectNamespace()) {
      // there are other write access op among alloc and load op, cannot
      // inline load with init
      return {std::nullopt, std::nullopt};
    }
  }

  return {initOp, initCondition};
}

LogicalResult replaceMemCopyByPTOLoadOp(memref::CopyOp copyOp,
                                         PatternRewriter &rewriter) {
  Value dst = copyOp.getTarget();
  auto maybeAlloc = tracebackMemRefToAlloc(dst);
  auto maybePadValue = getPadValue(maybeAlloc);
  auto maybeLeftPadNum = getLeftPadNum(rewriter, maybeAlloc);

  auto loadOp = rewriter.create<pto::TLoadOp>(copyOp->getLoc(), TypeRange(),
                                              copyOp.getSource(), dst, nullptr, nullptr, nullptr, nullptr, false, nullptr);
  if (maybeLeftPadNum.has_value()) {
    loadOp.getLeftPaddingNumMutable().assign(maybeLeftPadNum.value());
  }
  if (maybePadValue.has_value()) {
    auto padModeAttr =
        rewriter.getAttr<pto::PadModeAttr>(pto::PadMode::PadValue);
    loadOp.setPadModeAttr(padModeAttr);
    loadOp.getPadValueMutable().assign(maybePadValue.value());
    auto [inlineInitOp, inlineInitCond] =
        getUniqueInitInfo(rewriter, maybeAlloc, loadOp);
    if (inlineInitOp.has_value()) {
      loadOp.setInitOutBuffer(true);
      rewriter.eraseOp(inlineInitOp.value());
    }
    if (inlineInitCond.has_value()) {
      loadOp.getInitConditionMutable().assign(inlineInitCond.value());
    }
  }
  rewriter.replaceOp(copyOp, loadOp);
  return success();
}


struct MemrefCopyOpLowering : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    // do not convert mem copy inside simt_vf
    if (copyOp->getParentOfType<scf::ForallOp>() != nullptr) {
      return failure();
    }

    Value src = copyOp.getSource();
    bool convertToLoad = isFromFunctionArg(src);
    if (convertToLoad) {
      return replaceMemCopyByPTOLoadOp(copyOp, rewriter);
    }

    Value dst = copyOp.getTarget();
    bool convertToStore = isFromFunctionArg(dst);
    if (convertToStore) {
      rewriter.replaceOpWithNewOp<pto::TStoreOp>(copyOp, TypeRange(), src, dst);
      return success();
    }

    rewriter.replaceOpWithNewOp<pto::TMovOp>(copyOp, TypeRange(), src, dst);
    return success();
  }
};

struct BufferizeMaterializeOpLowering
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern<
      bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(bufferization::MaterializeInDestinationOp bufMIDOp,
                  PatternRewriter &rewriter) const override {
    Value dst = bufMIDOp.getDest();
    bool convertToStore = isFromFunctionArg(dst);
    if (convertToStore) {
      rewriter.replaceOpWithNewOp<pto::TStoreOp>(bufMIDOp, TypeRange(),
                                                 bufMIDOp.getSource(), dst);
      return success();
    }
    return failure();
  }
};

void populatePTOOpRewritingRule(RewritePatternSet &patterns) {
  patterns.add<MemrefCopyOpLowering, BufferizeMaterializeOpLowering>(
      patterns.getContext());
}


struct ConvertToPTOOpPass
    : public impl::ConvertToPTOOpBase<ConvertToPTOOpPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertToPTOOpPass::runOnOperation() {
  auto *ctx = &getContext();
  Operation *moduleOp = getOperation();
  moduleOp->walk([&](func::FuncOp funcOp) {
    // if (hacc::utils::isHost(funcOp))
    //   // avoid convert host op to pto op
    //   return;

    // rewrite op within cur funcOp
    RewritePatternSet patterns(ctx);
    populatePTOOpRewritingRule(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    // rewrite cur funcOp
    // RewritePatternSet funcOpPatterns(ctx);
    // (void)applyOpPatternsAndFold({funcOp}, std::move(funcOpPatterns));
  });
}

std::unique_ptr<Pass> mlir::pto::createConvertToPTOOpPass() {
  return std::make_unique<ConvertToPTOOpPass>();
}
