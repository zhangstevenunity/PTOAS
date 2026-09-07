// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <tuple>
#include <utility>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOCANONICALIZEIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// ---------------------------------------------------------------------------
// Design note: which ops need structural rewriting vs. type-only walk
// ---------------------------------------------------------------------------
//
// This pass canonicalizes low-rank TensorViewType / PartitionTensorViewType
// into the right-aligned rank-5 form used by all backends
// (A3, A5, and VPTO EmitC codegen).
//
// Ops that carry **rank-dependent operands** must be structurally rewritten
// (their operand count or operand values change when rank changes):
//   - MakeTensorViewOp  : shape/strides expanded from rank N → 5
//   - PartitionViewOp   : offsets/sizes expanded from rank N → 5
//   - GetTensorViewDimOp / GetTensorViewStrideOp : dim index offset by 5 - N
//
// Ops that only **carry view-typed operands/results** (no rank-dependent
// operand structure) are handled by the type walk (canonicalizeValueTypes)
// which in-place mutates TensorViewType and PartitionTensorViewType from
// rank N to rank-5:
//   - TAllocToAivOp, TAllocToAicOp, DeclareGlobalOp (producers)
//   - TAllocOp, TPushOp, TPopOp, TFreeOp, AicInitializePipeOp,
//     AivInitializePipeOp, TensorViewAddrOp (consumers)
//   - All PTODpsType consumers (TLoadOp, TStoreOp, TMatmulOp, etc.)
//   - All PTOPipeEntryType consumers (TPushToAivOp, TPopFromAicOp, etc.)
//
// A post-canonicalization verification (verifyNoLowRankViewSurvivors) detects
// any surviving low-rank view types to prevent silent failures when new
// view-consuming ops with rank-dependent operands are added.
//
// NZ layout cannot appear on low-rank views (it requires rank >= 5 with
// shape[2] == 16), so only ND and DN strides need expansion logic.
// ---------------------------------------------------------------------------

constexpr unsigned kCanonicalRank5 = 5;
constexpr int64_t kUnitExtent = 1;

static SmallVector<int64_t, kCanonicalRank5>
rightAlignShapeToRank5(ArrayRef<int64_t> shape) {
  SmallVector<int64_t, kCanonicalRank5> result(kCanonicalRank5, kUnitExtent);
  unsigned shift = kCanonicalRank5 - shape.size();
  for (auto [idx, dim] : llvm::enumerate(shape)) {
    result[shift + idx] = dim;
  }
  return result;
}

static Value getOrCreateIndexConstant(OpBuilder &builder, Location loc,
                                      int64_t value) {
  return builder.create<arith::ConstantIndexOp>(loc, value);
}

static SmallVector<Value, kCanonicalRank5>
rightAlignValuesToRank5(ValueRange values, Value fill) {
  SmallVector<Value, kCanonicalRank5> result(kCanonicalRank5, fill);
  unsigned shift = kCanonicalRank5 - values.size();
  for (auto [idx, value] : llvm::enumerate(values)) {
    result[shift + idx] = value;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Stride expansion: uses the same cumulative-product rule as
// rightAlignTo5D (InferPTOLayout.cpp) and buildGlobalTensorShapeAndStride
// (PTOToEmitC.cpp): stride[i] = shape[i+1] * stride[i+1].
// For a rank-2 view [R, C] right-aligned into [1, 1, 1, R, C]:
//   - ND (row-major): original strides = [C, 1]
//     padded strides: stride[2] = shape[3]*stride[3] = R*C,
//                    stride[1] = shape[2]*stride[2] = 1*R*C = R*C,
//                    stride[0] = shape[1]*stride[1] = 1*R*C = R*C
//     → [R*C, R*C, R*C, C, 1]
//   - DN (col-major): original strides = [1, R]
//     padded strides: stride[2] = shape[3]*stride[3] = R*1 = R,
//                    stride[1] = shape[2]*stride[2] = 1*R = R,
//                    stride[0] = shape[1]*stride[1] = 1*R = R
//     → [R, R, R, 1, R]
// Note: the ND branch was previously incorrectly using rowStride (=C) for
// all three leading dims, producing [C, C, C, C, 1] instead of the correct
// cumulative product [R*C, R*C, R*C, C, 1]. The DN branch was correct by
// coincidence because colStride == R and the cumulative product of unit-extent
// leading dims also collapses to R.
// For rank-1 [N], the same rule produces [N*S, N*S, N*S, N*S, S], where S is
// the original stride.
// ---------------------------------------------------------------------------
static SmallVector<Value, kCanonicalRank5>
buildCanonicalStrides(MakeTensorViewOp op, IRRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();
  unsigned rank = op.getShape().size();
  unsigned shift = kCanonicalRank5 - rank;
  SmallVector<Value, kCanonicalRank5> result =
      rightAlignValuesToRank5(op.getStrides(), Value());

  // Compute the first padded leading stride using the cumulative-product rule.
  // Unit-extent dimensions before `shift` then reuse the same value.
  Value leadingStride = rewriter.create<arith::MulIOp>(
      loc, op.getShape().front(), op.getStrides().front());

  for (unsigned i = 0; i < shift; ++i) {
    result[i] = leadingStride;
  }
  return result;
}

static bool isLowRankViewLike(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type)) {
    return viewType.getRank() > 0 && viewType.getRank() < kCanonicalRank5;
  }
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type)) {
    return viewType.getRank() > 0 && viewType.getRank() < kCanonicalRank5;
  }
  return false;
}

static std::optional<unsigned> getLowRankViewRank(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type)) {
    unsigned rank = viewType.getRank();
    if (rank > 0 && rank < kCanonicalRank5) {
      return rank;
    }
  }
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type)) {
    unsigned rank = viewType.getRank();
    if (rank > 0 && rank < kCanonicalRank5) {
      return rank;
    }
  }
  return std::nullopt;
}

static Type canonicalViewType(Type type) {
  if (auto viewType = dyn_cast<TensorViewType>(type)) {
    if (viewType.getRank() > 0 && viewType.getRank() < kCanonicalRank5) {
      return TensorViewType::get(type.getContext(),
                                 rightAlignShapeToRank5(viewType.getShape()),
                                 viewType.getElementType(),
                                 viewType.getLayout());
    }
    return type;
  }
  if (auto viewType = dyn_cast<PartitionTensorViewType>(type)) {
    if (viewType.getRank() > 0 && viewType.getRank() < kCanonicalRank5) {
      return PartitionTensorViewType::get(
          type.getContext(), rightAlignShapeToRank5(viewType.getShape()),
          viewType.getElementType(), viewType.getLayout());
    }
    return type;
  }
  return type;
}

static bool canonicalizeValueType(Value value) {
  Type oldType = value.getType();
  Type newType = canonicalViewType(oldType);
  if (newType == oldType) {
    return false;
  }
  value.setType(newType);
  return true;
}

static LogicalResult rewriteMakeTensorView(MakeTensorViewOp op,
                                           IRRewriter &rewriter) {
  auto oldType = dyn_cast<TensorViewType>(op.getResult().getType());
  if (!oldType || oldType.getRank() == 0 ||
      oldType.getRank() >= kCanonicalRank5) {
    return success();
  }

  unsigned rank = oldType.getRank();
  if (op.getShape().size() != rank || op.getStrides().size() != rank) {
    return op.emitOpError(
        "low-rank tensor_view must have matching shape and stride operands");
  }

  rewriter.setInsertionPoint(op);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newShape =
      rightAlignValuesToRank5(op.getShape(), one);
  SmallVector<Value, kCanonicalRank5> newStrides =
      buildCanonicalStrides(op, rewriter);
  auto newType = cast<TensorViewType>(canonicalViewType(oldType));

  auto newOp = rewriter.create<MakeTensorViewOp>(
      op.getLoc(), newType, op.getPtr(), newShape, newStrides,
      op.getLayoutAttr());
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static LogicalResult rewritePartitionView(PartitionViewOp op,
                                          IRRewriter &rewriter) {
  auto resultType = dyn_cast<PartitionTensorViewType>(op.getResult().getType());
  if (!resultType) {
    return success();
  }
  int64_t sourceRank = 0;
  if (auto sourceType =
          dyn_cast<TensorViewType>(op.getSource().getType())) {
    sourceRank = sourceType.getRank();
  } else if (auto sourceType =
                 dyn_cast<PartitionTensorViewType>(
                     op.getSource().getType())) {
    sourceRank = sourceType.getRank();
  } else {
    return success();
  }

  unsigned operandRank = op.getOffsets().size();
  if (operandRank == 0 || operandRank >= kCanonicalRank5 ||
      op.getSizes().size() != operandRank) {
    return success();
  }

  if (sourceRank != kCanonicalRank5) {
    return op.emitOpError(
        "low-rank partition_tensor_view normalization expects canonical rank-5 "
        "source view");
  }

  rewriter.setInsertionPoint(op);
  Value zero = getOrCreateIndexConstant(rewriter, op.getLoc(), 0);
  Value one = getOrCreateIndexConstant(rewriter, op.getLoc(), kUnitExtent);
  SmallVector<Value, kCanonicalRank5> newOffsets =
      rightAlignValuesToRank5(op.getOffsets(), zero);
  SmallVector<Value, kCanonicalRank5> newSizes =
      rightAlignValuesToRank5(op.getSizes(), one);
  auto newType = cast<PartitionTensorViewType>(canonicalViewType(resultType));

  auto newOp = rewriter.create<PartitionViewOp>(
      op.getLoc(), newType, op.getSource(), newOffsets, newSizes);
  rewriter.replaceOp(op, newOp.getResult());
  return success();
}

static Value buildCanonicalDimIndex(Value dimIndex, unsigned rank,
                                    IRRewriter &rewriter, Location loc) {
  rewriter.setInsertionPointAfterValue(dimIndex);
  Value offset =
      getOrCreateIndexConstant(rewriter, loc, kCanonicalRank5 - rank);
  return rewriter.create<arith::AddIOp>(loc, dimIndex, offset);
}

static void rewriteTensorViewDimOperand(Operation *op, Value dimIndex,
                                        unsigned rank,
                                        IRRewriter &rewriter) {
  Value newDim = buildCanonicalDimIndex(dimIndex, rank, rewriter, op->getLoc());
  op->setOperand(1, newDim);
}

static void canonicalizeFunctionType(func::FuncOp func) {
  auto oldType = func.getFunctionType();
  SmallVector<Type> inputs;
  SmallVector<Type> results;
  bool changed = false;

  inputs.reserve(oldType.getNumInputs());
  for (Type type : oldType.getInputs()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    inputs.push_back(newType);
  }

  results.reserve(oldType.getNumResults());
  for (Type type : oldType.getResults()) {
    Type newType = canonicalViewType(type);
    changed |= newType != type;
    results.push_back(newType);
  }

  if (changed) {
    func.setFunctionType(FunctionType::get(func.getContext(), inputs, results));
  }
}

static void canonicalizeValueTypes(func::FuncOp func) {
  canonicalizeFunctionType(func);

  func->walk([](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          canonicalizeValueType(arg);
        }
      }
    }

    for (OpResult result : op->getResults()) {
      canonicalizeValueType(result);
    }
  });
}

/// Verify that no low-rank view types survived canonicalization.
/// This catches cases where a new op with rank-dependent operands
/// was added but not given a structural rewrite in this pass.
static LogicalResult verifyNoLowRankViewSurvivors(func::FuncOp func) {
  bool anyFailed = false;
  func.walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          if (isLowRankViewLike(arg.getType())) {
            emitError(arg.getLoc())
                << "low-rank view type survived canonicalization: "
                << arg.getType() << " as block argument";
            anyFailed = true;
          }
        }
      }
    }
    for (OpResult result : op->getResults()) {
      if (isLowRankViewLike(result.getType())) {
        emitError(op->getLoc())
            << "low-rank view type survived canonicalization: "
            << result.getType() << " in op " << op->getName();
        anyFailed = true;
      }
    }
  });
  return anyFailed ? failure() : success();
}

struct PTOCanonicalizeIRPass
    : public mlir::pto::impl::PTOCanonicalizeIRBase<PTOCanonicalizeIRPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<MakeTensorViewOp> makeViews;
    SmallVector<PartitionViewOp> partitionViews;
    SmallVector<std::tuple<Operation *, Value, unsigned>> dimIndexOps;

    func.walk([&](MakeTensorViewOp op) {
      if (isLowRankViewLike(op.getResult().getType())) {
        makeViews.push_back(op);
      }
    });
    func.walk([&](PartitionViewOp op) {
      unsigned rank = op.getOffsets().size();
      if (rank > 0 && rank < kCanonicalRank5 && op.getSizes().size() == rank) {
        partitionViews.push_back(op);
      }
    });
    func.walk([&](GetTensorViewDimOp op) {
      if (std::optional<unsigned> rank =
              getLowRankViewRank(op.getTensorView().getType())) {
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex(), *rank);
      }
    });
    func.walk([&](GetTensorViewStrideOp op) {
      if (std::optional<unsigned> rank =
              getLowRankViewRank(op.getTensorView().getType())) {
        dimIndexOps.emplace_back(op.getOperation(), op.getDimIndex(), *rank);
      }
    });

    IRRewriter rewriter(func.getContext());
    for (MakeTensorViewOp op : makeViews) {
      if (failed(rewriteMakeTensorView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }
    for (auto [op, dimIndex, rank] : dimIndexOps) {
      rewriteTensorViewDimOperand(op, dimIndex, rank, rewriter);
    }
    canonicalizeValueTypes(func);
    for (PartitionViewOp op : partitionViews) {
      if (failed(rewritePartitionView(op, rewriter))) {
        signalPassFailure();
        return;
      }
    }

    // VPTO consumes the shared FFTS sync representation.  Normalize named
    // cross-block ops to mode 0 and A2/A3 intra-block ops to mode 2; A5 keeps
    // named intra ops for dedicated lowering.
    SmallVector<SetCrossBlockOp> crossSets;
    SmallVector<WaitCrossBlockOp> crossWaits;
    SmallVector<SetIntraBlockOp> intraSets;
    SmallVector<WaitIntraBlockOp> intraWaits;
    func.walk([&](SetCrossBlockOp op) { crossSets.push_back(op); });
    func.walk([&](WaitCrossBlockOp op) { crossWaits.push_back(op); });
    PTOArch targetArch = getTargetArch(func);
    if (targetArch != PTOArch::A5) {
      func.walk([&](SetIntraBlockOp op) { intraSets.push_back(op); });
      func.walk([&](WaitIntraBlockOp op) { intraWaits.push_back(op); });
    }
    auto mode0 = IntegerAttr::get(IntegerType::get(func.getContext(), 32), 0);
    auto mode2 = IntegerAttr::get(IntegerType::get(func.getContext(), 32), 2);
    for (SetCrossBlockOp op : crossSets) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<SyncSetOp>(
          op, op.getPipe(), op.getEventIdAttr(), mode0, op.getEventIdDyn());
    }
    for (WaitCrossBlockOp op : crossWaits) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<SyncWaitOp>(
          op, op.getPipe(), op.getEventIdAttr(), mode0, op.getEventIdDyn());
    }
    for (SetIntraBlockOp op : intraSets) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<SyncSetOp>(
          op, op.getPipe(), op.getEventIdAttr(), mode2, op.getEventIdDyn());
    }
    for (WaitIntraBlockOp op : intraWaits) {
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<SyncWaitOp>(
          op, op.getPipe(), op.getEventIdAttr(), mode2, op.getEventIdDyn());
    }

    // Post-canonicalization verification: ensure no low-rank view types
    // survived. If any do, it means an op with rank-dependent operands
    // was not given a structural rewrite.
    if (failed(verifyNoLowRankViewSurvivors(func))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOCanonicalizeIRPass() {
  return std::make_unique<PTOCanonicalizeIRPass>();
}
