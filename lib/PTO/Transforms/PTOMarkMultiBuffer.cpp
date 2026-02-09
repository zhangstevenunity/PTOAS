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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <optional>

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

static Value traceMemrefRoot(Value v) {
  while (true) {
    if (auto mark = v.getDefiningOp<pto::MarkMultiBufferOp>()) {
      v = mark.getSource();
      continue;
    }
    if (auto bind = v.getDefiningOp<pto::BindTileOp>()) {
      v = bind.getSource();
      continue;
    }
    if (auto sub = v.getDefiningOp<memref::SubViewOp>()) {
      v = sub.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto rc = v.getDefiningOp<memref::ReinterpretCastOp>()) {
      v = rc.getSource();
      continue;
    }
    break;
  }
  return v;
}

static bool isLocalMemRef(Value v) {
  auto mrTy = dyn_cast<MemRefType>(v.getType());
  if (!mrTy)
    return false;
  auto addrAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
  if (!addrAttr)
    return false;
  return addrAttr.getAddressSpace() != pto::AddressSpace::GM;
}

static std::optional<pto::AddressSpace> getMemSpace(Value v) {
  auto mrTy = dyn_cast<MemRefType>(v.getType());
  if (!mrTy)
    return std::nullopt;
  auto addrAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(mrTy.getMemorySpace());
  if (!addrAttr)
    return std::nullopt;
  return addrAttr.getAddressSpace();
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
      v = traceMemrefRoot(v);
      auto alloc = v.getDefiningOp<memref::AllocOp>();
      if (!alloc)
        return;
      if (!isLocalMemRef(alloc.getResult()))
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

      if (auto load = dyn_cast<pto::LoadDpsOp>(op)) {
        markIfNeeded(load.getDst());
        return;
      }
      if (auto store = dyn_cast<pto::StoreDpsOp>(op)) {
        markIfNeeded(store.getSrc());
        return;
      }
      if (auto mov = dyn_cast<pto::MovDpsOp>(op)) {
        auto srcSpace = getMemSpace(mov.getSrc());
        auto dstSpace = getMemSpace(mov.getDst());
        if (srcSpace && dstSpace) {
          if (*srcSpace == pto::AddressSpace::GM && *dstSpace != pto::AddressSpace::GM)
            markIfNeeded(mov.getDst());
          else if (*dstSpace == pto::AddressSpace::GM && *srcSpace != pto::AddressSpace::GM)
            markIfNeeded(mov.getSrc());
        }
        return;
      }
      if (auto movfp = dyn_cast<pto::MovFPOp_DPS>(op)) {
        auto srcSpace = getMemSpace(movfp.getSrc());
        auto dstSpace = getMemSpace(movfp.getDst());
        if (srcSpace && dstSpace) {
          if (*srcSpace == pto::AddressSpace::GM && *dstSpace != pto::AddressSpace::GM)
            markIfNeeded(movfp.getDst());
          else if (*dstSpace == pto::AddressSpace::GM && *srcSpace != pto::AddressSpace::GM)
            markIfNeeded(movfp.getSrc());
        }
        return;
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOMarkMultiBufferPass() {
  return std::make_unique<PTOMarkMultiBufferPass>();
}
