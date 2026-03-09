#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;
#define GEN_PASS_DEF_PTOINSERTTFREE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

/// Check whether `tpopOp` already has a matching tfree in the same block.
static bool alreadyHasTFree(TPopOp tpopOp) {
  Value pipeHandle = tpopOp.getPipeHandle();
  Block *block = tpopOp->getBlock();
  for (auto it = std::next(tpopOp->getIterator()), end = block->end();
       it != end; ++it) {
    if (auto tfreeOp = dyn_cast<TFreeOp>(&*it)) {
      if (tfreeOp.getPipeHandle() == pipeHandle)
        return true;
    }
  }
  return false;
}

/// Find the last operation in the same block (after startOp) that reads `tile`
/// via MemoryEffectsOpInterface.
static Operation *findLastReadOf(Value tile, Operation *startOp) {
  Block *block = startOp->getBlock();
  Operation *lastRead = nullptr;

  for (auto it = std::next(startOp->getIterator()), end = block->end();
       it != end; ++it) {
    Operation *op = &*it;

    // Check via MemoryEffectsOpInterface.
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
          effects;
      memEffect.getEffects(effects);
      for (auto &effect : effects) {
        if (effect.getValue() == tile &&
            isa<MemoryEffects::Read>(effect.getEffect())) {
          lastRead = op;
          break;
        }
      }
    }
  }

  return lastRead;
}

struct PTOInsertTFreePass
    : public mlir::pto::impl::PTOInsertTFreeBase<PTOInsertTFreePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect tpop ops first to avoid iterator invalidation.
    SmallVector<TPopOp> tpops;
    funcOp.walk([&](TPopOp op) { tpops.push_back(op); });

    for (TPopOp tpopOp : tpops) {
      // Skip if already has a matching tfree.
      if (alreadyHasTFree(tpopOp))
        continue;

      // Must be inside a section.
      if (!tpopOp->getParentOfType<SectionCubeOp>() &&
          !tpopOp->getParentOfType<SectionVectorOp>())
        continue;

      Value tile = tpopOp.getTile();
      Value pipeHandle = tpopOp.getPipeHandle();

      // Find the last read of the tile after the tpop.
      Operation *lastRead = findLastReadOf(tile, tpopOp.getOperation());

      // Insert tfree after the last read, or right after tpop if no reads.
      Operation *insertAfter = lastRead ? lastRead : tpopOp.getOperation();
      OpBuilder builder(insertAfter->getContext());
      builder.setInsertionPointAfter(insertAfter);
      builder.create<TFreeOp>(tpopOp.getLoc(), pipeHandle);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertTFreePass() {
  return std::make_unique<PTOInsertTFreePass>();
}
