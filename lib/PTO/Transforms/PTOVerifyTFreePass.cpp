#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOVERIFYTFREE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

/// Find a matching lowered tfree in the same block.
static TFreeInternalOp findMatchingTFree(TPopInternalOp tpopOp) {
  Value pipeHandle = tpopOp.getPipeHandle();
  Block *block = tpopOp->getBlock();
  for (auto it = std::next(tpopOp->getIterator()), end = block->end();
       it != end; ++it) {
    if (auto tfreeOp = dyn_cast<TFreeInternalOp>(&*it)) {
      if (tfreeOp.getPipeHandle() == pipeHandle)
        return tfreeOp;
    }
  }
  return {};
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  Operation *current = op;
  while (current && current->getBlock() != block) {
    Region *parentRegion = current->getParentRegion();
    if (!parentRegion)
      return nullptr;
    current = parentRegion->getParentOp();
  }
  return current;
}

static bool hasSamePipeTPopInRegion(Operation *op, Value pipeHandle,
                                    TPopInternalOp current) {
  bool found = false;
  op->walk([&](TPopInternalOp nestedTpop) {
    if (nestedTpop == current)
      return WalkResult::advance();
    if (nestedTpop.getPipeHandle() == pipeHandle) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult verifySingleOutstandingUntil(TPopInternalOp tpopOp,
                                                  Operation *freeBoundary) {
  if (!freeBoundary)
    return success();
  if (freeBoundary == tpopOp.getOperation())
    return success();

  Value pipeHandle = tpopOp.getPipeHandle();
  Block *block = tpopOp->getBlock();
  for (auto it = std::next(tpopOp->getIterator()), end = block->end();
       it != end; ++it) {
    Operation *op = &*it;
    if (hasSamePipeTPopInRegion(op, pipeHandle, tpopOp)) {
      return tpopOp.emitOpError(
          "multiple outstanding pops on the same pipe are not supported");
    }
    if (op == freeBoundary)
      break;
  }

  return success();
}

static LogicalResult verifyNoTileUsesAfterTFree(TPopInternalOp tpopOp,
                                                TFreeInternalOp tfreeOp) {
  Value tile = tpopOp.getTile();
  Block *block = tpopOp->getBlock();

  for (OpOperand &use : tile.getUses()) {
    Operation *topLevelOwner =
        getTopLevelAncestorInBlock(use.getOwner(), block);
    if (!topLevelOwner) {
      return tpopOp.emitOpError(
          "borrowed tile uses must stay in the same parent block as the producing tpop");
    }
    if (tfreeOp->isBeforeInBlock(topLevelOwner)) {
      return tpopOp.emitOpError(
          "tfree_internal must appear after the last use of the borrowed tile");
    }
  }

  return success();
}

struct PTOVerifyTFreePass
    : public mlir::pto::impl::PTOVerifyTFreeBase<PTOVerifyTFreePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect tpop ops first to avoid iterator invalidation.
    SmallVector<TPopInternalOp> tpops;
    funcOp.walk([&](TPopInternalOp op) { tpops.push_back(op); });

    for (TPopInternalOp tpopOp : tpops) {
      // Must be inside a section.
      if (!tpopOp->getParentOfType<SectionCubeOp>() &&
          !tpopOp->getParentOfType<SectionVectorOp>())
        continue;

      TFreeInternalOp existingTFree = findMatchingTFree(tpopOp);
      if (!existingTFree) {
        tpopOp.emitOpError("requires an explicit matching tfree_internal");
        signalPassFailure();
        return;
      }

      // Reject cases where the same pipe is popped again before this borrowed
      // tile reaches its explicit free boundary.
      if (failed(
              verifySingleOutstandingUntil(tpopOp, existingTFree.getOperation()))) {
        signalPassFailure();
        return;
      }

      if (failed(verifyNoTileUsesAfterTFree(tpopOp, existingTFree))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVerifyTFreePass() {
  return std::make_unique<PTOVerifyTFreePass>();
}
