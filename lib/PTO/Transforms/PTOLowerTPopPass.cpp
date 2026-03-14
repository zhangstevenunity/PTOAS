#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERTPOP
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static LogicalResult validateSlotUsers(func::FuncOp funcOp) {
  WalkResult walkResult = funcOp.walk([&](TPopOp op) {
    GetFifoTileOp getFifoTileOp;
    TFreeOp tfreeOp;

    for (OpOperand &use : op.getSlotId().getUses()) {
      Operation *user = use.getOwner();
      if (auto getTile = dyn_cast<GetFifoTileOp>(user)) {
        if (getFifoTileOp) {
          op.emitError("slot_id must have exactly one pto.get_fifo_tile user");
          return WalkResult::interrupt();
        }
        if (getTile.getPipeHandle() != op.getPipeHandle()) {
          op.emitError(
              "slot_id users must use the same pipe_handle as the producing pto.tpop");
          return WalkResult::interrupt();
        }
        getFifoTileOp = getTile;
        continue;
      }
      if (auto tfree = dyn_cast<TFreeOp>(user)) {
        if (tfreeOp) {
          op.emitError("slot_id must have exactly one pto.tfree user");
          return WalkResult::interrupt();
        }
        if (tfree.getPipeHandle() != op.getPipeHandle()) {
          op.emitError(
              "slot_id users must use the same pipe_handle as the producing pto.tpop");
          return WalkResult::interrupt();
        }
        tfreeOp = tfree;
        continue;
      }

      op.emitError(
          "slot_id can only be used by pto.get_fifo_tile or pto.tfree");
      return WalkResult::interrupt();
    }

    if (!getFifoTileOp) {
      op.emitError("slot_id must have exactly one pto.get_fifo_tile user");
      return WalkResult::interrupt();
    }
    if (!tfreeOp) {
      op.emitError("slot_id must have exactly one pto.tfree user");
      return WalkResult::interrupt();
    }

    if (getFifoTileOp->getBlock() != op->getBlock()) {
      op.emitError(
          "pto.get_fifo_tile must be in the same block as the producing pto.tpop");
      return WalkResult::interrupt();
    }
    if (tfreeOp->getBlock() != op->getBlock()) {
      op.emitError(
          "pto.tfree must be in the same block as the producing pto.tpop");
      return WalkResult::interrupt();
    }
    if (!getFifoTileOp->isBeforeInBlock(tfreeOp)) {
      op.emitError(
          "pto.tfree must appear after the corresponding pto.get_fifo_tile");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

struct LowerGetFifoTilePattern : public OpRewritePattern<GetFifoTileOp> {
  using OpRewritePattern<GetFifoTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetFifoTileOp op,
                                PatternRewriter &rewriter) const override {
    auto tpopOp = op.getSlotId().getDefiningOp<TPopOp>();
    if (!tpopOp)
      return rewriter.notifyMatchFailure(op, "slot_id must come from pto.tpop");

    auto declaredTile =
        rewriter.create<DeclareTileOp>(op.getLoc(), op.getTile().getType());
    rewriter.create<TPopInternalOp>(op.getLoc(), declaredTile.getTile(),
                                    op.getPipeHandle());
    rewriter.replaceOp(op, declaredTile.getTile());
    return success();
  }
};

struct LowerTFreePattern : public OpRewritePattern<TFreeOp> {
  using OpRewritePattern<TFreeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFreeOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFreeInternalOp>(op, op.getPipeHandle());
    return success();
  }
};

struct EraseLoweredTPopPattern : public OpRewritePattern<TPopOp> {
  using OpRewritePattern<TPopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TPopOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getSlotId().use_empty())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOLowerTPopPass
    : public mlir::pto::impl::PTOLowerTPopBase<PTOLowerTPopPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    if (failed(validateSlotUsers(funcOp))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);
    patterns.add<LowerGetFifoTilePattern, LowerTFreePattern,
                 EraseLoweredTPopPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowerTPopPass() {
  return std::make_unique<PTOLowerTPopPass>();
}
