#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

static Value getCountFromVLoadTail(Value v) {
  if (!v)
    return Value();
  if (auto lt = v.getDefiningOp<mlir::pto::VLoadTailOp>())
    return lt.getCount();
  return Value();
}

static Value getOrCreatePredAll(Location loc, PatternRewriter &rewriter) {
  for (auto &op : *rewriter.getInsertionBlock()) {
    if (auto all = dyn_cast<mlir::pto::VPredAllOp>(&op))
      return all.getPred();
  }
  return rewriter.create<mlir::pto::VPredAllOp>(loc).getPred();
}

// vload(tile,row,col, pred=tail(count)) -> vload_tail(tile,row,col,count)
struct VLoadTailFromPred final : OpRewritePattern<mlir::pto::VLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pto::VLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto tail = op.getPred().getDefiningOp<mlir::pto::VPredTailOp>();
    if (!tail)
      return failure();

    auto vt = rewriter.create<mlir::pto::VLoadTailOp>(op.getLoc(), op.getValue().getType(),
                                                      op.getTile(), op.getRow(),
                                                      op.getCol(), tail.getCount());
    rewriter.replaceOp(op, vt.getValue());
    return success();
  }
};

// vstore(tile,row,col,val, pred=tail(count)) -> vstore_tail(tile,row,col,count,val)
struct VStoreTailFromPred final : OpRewritePattern<mlir::pto::VStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pto::VStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto tail = op.getPred().getDefiningOp<mlir::pto::VPredTailOp>();
    if (!tail)
      return failure();

    rewriter.create<mlir::pto::VStoreTailOp>(op.getLoc(), op.getTile(), op.getRow(),
                                             op.getCol(), tail.getCount(),
                                             op.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

// Binop pred propagation: all -> tail(count) if any operand comes from vload_tail(count).
template <typename BinOp>
struct BinOpUseTailPred final : OpRewritePattern<BinOp> {
  using OpRewritePattern<BinOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(BinOp op,
                                PatternRewriter &rewriter) const override {
    auto all = op.getPred().template getDefiningOp<mlir::pto::VPredAllOp>();
    if (!all)
      return failure();

    Value c0 = getCountFromVLoadTail(op.getLhs());
    Value c1 = getCountFromVLoadTail(op.getRhs());
    if (!c0 && !c1)
      return failure();

    Value count = c0 ? c0 : c1;
    if (c0 && c1 && c0 != c1)
      return failure();

    auto tail = rewriter.create<mlir::pto::VPredTailOp>(op.getLoc(), count);
    auto repl = rewriter.create<BinOp>(op.getLoc(), op.getDst().getType(), op.getLhs(),
                                       op.getRhs(), tail.getPred());
    rewriter.replaceOp(op, repl.getDst());
    return success();
  }
};

// Store tail-ization: vstore(all) -> vstore_tail if value depends on vload_tail.
struct VStoreAllToTail final : OpRewritePattern<mlir::pto::VStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pto::VStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto all = op.getPred().getDefiningOp<mlir::pto::VPredAllOp>();
    if (!all)
      return failure();

    Value count = getCountFromVLoadTail(op.getValue());
    if (!count)
      return failure();

    rewriter.create<mlir::pto::VStoreTailOp>(op.getLoc(), op.getTile(), op.getRow(),
                                             op.getCol(), count, op.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

// count==lanes : vload_tail -> vload(pred.all)
struct VLoadTailToVLoad final : OpRewritePattern<mlir::pto::VLoadTailOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pto::VLoadTailOp op,
                                PatternRewriter &rewriter) const override {
    auto cst = op.getCount().getDefiningOp<arith::ConstantIndexOp>();
    if (!cst)
      return failure();
    auto lanes = cast<mlir::pto::VTileType>(op.getValue().getType()).getLanes();
    if ((int64_t)cst.value() != lanes)
      return failure();

    Value pAll = getOrCreatePredAll(op.getLoc(), rewriter);
    auto nl = rewriter.create<mlir::pto::VLoadOp>(op.getLoc(), op.getValue().getType(),
                                                  op.getTile(), op.getRow(),
                                                  op.getCol(), pAll);
    rewriter.replaceOp(op, nl.getValue());
    return success();
  }
};

// count==lanes : vstore_tail -> vstore(pred.all)
struct VStoreTailToVStore final : OpRewritePattern<mlir::pto::VStoreTailOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::pto::VStoreTailOp op,
                                PatternRewriter &rewriter) const override {
    auto cst = op.getCount().getDefiningOp<arith::ConstantIndexOp>();
    if (!cst)
      return failure();
    auto lanes = cast<mlir::pto::VTileType>(op.getValue().getType()).getLanes();
    if ((int64_t)cst.value() != lanes)
      return failure();

    Value pAll = getOrCreatePredAll(op.getLoc(), rewriter);
    rewriter.create<mlir::pto::VStoreOp>(op.getLoc(), op.getTile(), op.getRow(),
                                         op.getCol(), op.getValue(), pAll);
    rewriter.eraseOp(op);
    return success();
  }
};

// Conservative loop-invariant hoisting: hoist a single pure pto op producing vtile/uscalar/preg.
struct HoistPureVtileOpFromFor final : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (forOp.getNumIterOperands() != 0)
      return failure();

    Value iv = forOp.getInductionVar();

    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!op.getDialect() || op.getDialect()->getNamespace() != "pto")
        continue;
      if (!MemoryEffectOpInterface::hasNoEffect(&op))
        continue;
      if (op.getNumResults() != 1)
        continue;

      Type ty = op.getResult(0).getType();
      if (!isa<mlir::pto::VTileType, mlir::pto::UScalarType, mlir::pto::PregType>(ty))
        continue;

      bool usesIV = llvm::any_of(op.getOperands(), [&](Value v) { return v == iv; });
      if (usesIV)
        continue;

      bool dependsOnLoop = false;
      for (Value v : op.getOperands()) {
        if (auto *def = v.getDefiningOp()) {
          if (def->getParentOp() == forOp) {
            dependsOnLoop = true;
            break;
          }
        }
      }
      if (dependsOnLoop)
        continue;

      rewriter.setInsertionPoint(forOp);
      Operation *cloned = rewriter.clone(op);
      op.replaceAllUsesWith(cloned->getResults());
      rewriter.eraseOp(&op);
      return success();
    }

    return failure();
  }
};

struct PTOCanonicalizeVopsPass
    : public mlir::pto::impl::PTOCanonicalizeVopsBase<PTOCanonicalizeVopsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<HoistPureVtileOpFromFor>(ctx);
    patterns.add<VLoadTailToVLoad, VStoreTailToVStore>(ctx);

    patterns.add<VLoadTailFromPred, VStoreTailFromPred, VStoreAllToTail>(ctx);

    patterns.add<BinOpUseTailPred<mlir::pto::VAddOp>, BinOpUseTailPred<mlir::pto::VSubOp>,
                 BinOpUseTailPred<mlir::pto::VMulOp>, BinOpUseTailPred<mlir::pto::VMinOp>,
                 BinOpUseTailPred<mlir::pto::VMaxOp>, BinOpUseTailPred<mlir::pto::VAndOp>,
                 BinOpUseTailPred<mlir::pto::VOrOp>, BinOpUseTailPred<mlir::pto::VXorOp>>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::pto {
std::unique_ptr<Pass> createPTOCanonicalizeVopsPass() {
  return std::make_unique<PTOCanonicalizeVopsPass>();
}
} // namespace mlir::pto
