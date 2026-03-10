#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOLOWERINGSYNCTOPIPE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// Map high-level SyncOpType to hardware PIPE enum.
static PIPE getPipeFromOpType(SyncOpType opType) {
  switch (opType) {
  case SyncOpType::TLOAD:
    return PIPE::PIPE_MTE2; // GM -> L1/UB
  case SyncOpType::TSTORE_VEC:
    return PIPE::PIPE_MTE3; // UB -> GM
  case SyncOpType::TSTORE_ACC:
    return PIPE::PIPE_FIX;

  case SyncOpType::TMOV_M2L:
  case SyncOpType::TMOV_M2B:
    return PIPE::PIPE_MTE1; // L1 -> L0
  case SyncOpType::TMOV_M2S:
    return PIPE::PIPE_FIX;
  case SyncOpType::TMOV_M2V:
    return PIPE::PIPE_V;
  case SyncOpType::TMOV_V2M:
    return PIPE::PIPE_FIX;

  case SyncOpType::TMATMUL:
    return PIPE::PIPE_M; // Cube
  case SyncOpType::TVEC:
  case SyncOpType::TVECWAIT_EVENT:
    return PIPE::PIPE_V; // Vector

  default:
    return PIPE::PIPE_UNASSIGNED;
  }
}

static bool isTargetArchA5(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
  return arch && arch.getValue().equals_insensitive("a5");
}

static FailureOr<SyncOpType> getSyncOpTypeFromAttr(Attribute attr, Operation *op,
                                                   StringRef name) {
  if (auto a = attr.dyn_cast<PipeEventTypeAttr>())
    return a.getOpType();
  if (auto a = attr.dyn_cast<SyncOpTypeAttr>())
    return a.getOpType();
  auto diag =
      op->emitError("expected PipeEventTypeAttr or SyncOpTypeAttr for ");
  diag << name;
  return failure();
}

struct RecordEventLowering : public OpRewritePattern<RecordEventOp> {
  using OpRewritePattern<RecordEventOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecordEventOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTypeOr = getSyncOpTypeFromAttr(op.getSrcOpAttr(), op, "src_op");
    if (failed(srcTypeOr))
      return failure();
    auto dstTypeOr = getSyncOpTypeFromAttr(op.getDstOpAttr(), op, "dst_op");
    if (failed(dstTypeOr))
      return failure();
    SyncOpType srcType = *srcTypeOr;
    SyncOpType dstType = *dstTypeOr;

    PIPE srcPipe = getPipeFromOpType(srcType);
    PIPE dstPipe = getPipeFromOpType(dstType);
    if (srcPipe == PIPE::PIPE_UNASSIGNED || dstPipe == PIPE::PIPE_UNASSIGNED)
      return op.emitError("Failed to map SyncOpType to hardware pipe during lowering.");

    rewriter.replaceOpWithNewOp<SetFlagOp>(
        op, PipeAttr::get(op.getContext(), srcPipe),
        PipeAttr::get(op.getContext(), dstPipe), op.getEventIdAttr());
    return success();
  }
};

struct WaitEventLowering : public OpRewritePattern<WaitEventOp> {
  using OpRewritePattern<WaitEventOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitEventOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTypeOr = getSyncOpTypeFromAttr(op.getSrcOpAttr(), op, "src_op");
    if (failed(srcTypeOr))
      return failure();
    auto dstTypeOr = getSyncOpTypeFromAttr(op.getDstOpAttr(), op, "dst_op");
    if (failed(dstTypeOr))
      return failure();
    SyncOpType srcType = *srcTypeOr;
    SyncOpType dstType = *dstTypeOr;

    PIPE srcPipe = getPipeFromOpType(srcType);
    PIPE dstPipe = getPipeFromOpType(dstType);
    if (srcPipe == PIPE::PIPE_UNASSIGNED || dstPipe == PIPE::PIPE_UNASSIGNED)
      return op.emitError("Failed to map SyncOpType to hardware pipe during lowering.");

    rewriter.replaceOpWithNewOp<WaitFlagOp>(
        op, PipeAttr::get(op.getContext(), srcPipe),
        PipeAttr::get(op.getContext(), dstPipe), op.getEventIdAttr());
    return success();
  }
};

// High-level barrier -> barrier with mapped pipe
struct BarrierSyncLowering : public OpRewritePattern<BarrierSyncOp> {
  using OpRewritePattern<BarrierSyncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierSyncOp op,
                                PatternRewriter &rewriter) const override {
    SyncOpType ty = op.getOpType().getOpType();
    PIPE pipe = getPipeFromOpType(ty);
    if (pipe == PIPE::PIPE_UNASSIGNED) {
      auto diag = op.emitError(
          "barrier_sync failed to map SyncOpType to hardware pipe during lowering: ");
      diag << op.getOpType();
      return failure();
    }

    // A5: TVEC single-pipe barrier is unnecessary/unsupported.
    if (pipe == PIPE::PIPE_V && isTargetArchA5(op.getOperation())) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.replaceOpWithNewOp<BarrierOp>(
        op, PipeAttr::get(op.getContext(), pipe));
    return success();
  }
};

// Legalize explicit low-level barriers for arch constraints.
struct BarrierLegalizeForArch : public OpRewritePattern<BarrierOp> {
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (isTargetArchA5(op.getOperation()) &&
        op.getPipe().getPipe() == PIPE::PIPE_V) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

struct LoweringSyncToPipe
    : public mlir::pto::impl::PTOLoweringSyncToPipeBase<LoweringSyncToPipe> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<RecordEventLowering, WaitEventLowering, BarrierSyncLowering,
                 BarrierLegalizeForArch>(
        context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createLoweringSyncToPipePass() {
  return std::make_unique<LoweringSyncToPipe>();
}
