//===- PTOMarkMultiBuffer.cpp - Mark tile buffers for multi-buffering -----===//
//
// Inserts pto.mark_multibuffer on tile buffers used in looped copy-like ops.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pto {
  namespace func = ::mlir::func;
  #define GEN_PASS_DEF_PTOMARKMULTIBUFFER
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static Value traceTileRoot(Value v) {
  while (auto mark = v.getDefiningOp<pto::MarkMultiBufferOp>()) {
    v = mark.getSource();
  }
  while (auto subset = v.getDefiningOp<pto::SubsetOp>()) {
    v = subset.getSource();
  }
  return v;
}

static bool isLocalTileBuf(Value v) {
  auto tbTy = dyn_cast<pto::TileBufType>(v.getType());
  if (!tbTy)
    return false;
  auto addrAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(tbTy.getMemorySpace());
  if (!addrAttr)
    return true;
  return addrAttr.getAddressSpace() != pto::AddressSpace::GM;
}

static pto::MarkMultiBufferOp getExistingMark(Value v) {
  for (auto *user : v.getUsers()) {
    if (auto mark = dyn_cast<pto::MarkMultiBufferOp>(user)) {
      if (mark.getSource() == v)
        return mark;
    }
  }
  return nullptr;
}

struct PTOMarkMultiBufferPass
    : public mlir::pto::impl::PTOMarkMultiBufferBase<PTOMarkMultiBufferPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DenseMap<Value, Value> marked;
    IRRewriter rewriter(func.getContext());

    auto markIfNeeded = [&](Value v) {
      v = traceTileRoot(v);
      auto alloc = v.getDefiningOp<pto::AllocTileOp>();
      if (!alloc)
        return;
      if (!isLocalTileBuf(alloc.getResult()))
        return;
      if (!alloc->getParentOfType<scf::ForOp>())
        return;

      if (auto existing = getExistingMark(alloc.getResult())) {
        marked.try_emplace(alloc.getResult(), existing.getResult());
        return;
      }
      if (marked.count(alloc.getResult()))
        return;

      rewriter.setInsertionPointAfter(alloc);
      auto numAttr = rewriter.getI32IntegerAttr(2);
      auto mark = rewriter.create<pto::MarkMultiBufferOp>(
          alloc.getLoc(), alloc.getResult().getType(), alloc.getResult(),
          numAttr);

      alloc.getResult().replaceAllUsesExcept(mark.getResult(), mark);
      marked.try_emplace(alloc.getResult(), mark.getResult());
    };

    func.walk([&](Operation *op) {
      if (!op->getParentOfType<scf::ForOp>())
        return;

      if (auto tload = dyn_cast<pto::TLoadOp>(op)) {
        markIfNeeded(tload.getDst());
        return;
      }
      if (auto tstore = dyn_cast<pto::TStoreOp>(op)) {
        markIfNeeded(tstore.getSrc());
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMarkMultiBufferPass() {
  return std::make_unique<PTOMarkMultiBufferPass>();
}

