//===- EnableMultiBuffer.cpp - Materialize ping/pong buffer selection ------===//
//
// This pass rewrites `pto.pointer_cast(addrs=[ping, pong])` into:
//   - two single-address pointer_cast ops hoisted outside a selected loop, and
//   - a loop-local `arith.select` that chooses the active buffer based on the
//     flattened loop iteration parity.
//
// The goal is to make multi-buffer planning observable in the emitted C++
// (the default PointerCast lowering uses only the first address operand).
//
// Currently only double-buffer (2) is supported.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/MultiBufferSelector.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace pto {
namespace {

#define GEN_PASS_DEF_PTOENABLEMULTIBUFFER
#include "PTO/Transforms/Passes.h.inc"

static bool isGlobalMemRef(MemRefType ty) {
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    return (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  // Treat missing memory_space as GM.
  return true;
}

static bool getConstI64(Value v, int64_t &out) {
  llvm::APInt ap;
  if (!matchPattern(v, m_ConstantInt(&ap)))
    return false;
  out = ap.getSExtValue();
  return true;
}

static bool isAncestorLoop(scf::ForOp ancestor, scf::ForOp loop) {
  if (!ancestor || !loop)
    return false;
  Operation *cur = loop.getOperation();
  while (cur) {
    if (cur == ancestor.getOperation())
      return true;
    cur = cur->getParentOp();
  }
  return false;
}

static scf::ForOp lowestCommonAncestorLoop(ArrayRef<scf::ForOp> loops) {
  if (loops.empty())
    return {};
  scf::ForOp lca = loops.front();
  for (scf::ForOp loop : loops.drop_front()) {
    while (lca && !isAncestorLoop(lca, loop))
      lca = lca->getParentOfType<scf::ForOp>();
    if (!lca)
      return {};
  }
  return lca;
}

static bool isInLoopBody(Operation *op, scf::ForOp loop) {
  if (!op || !loop)
    return false;
  Operation *cur = op;
  while (cur) {
    if (cur == loop.getOperation())
      return op != cur;
    cur = cur->getParentOp();
  }
  return false;
}

struct PTOEnableMultiBufferPass
    : public impl::PTOEnableMultiBufferBase<PTOEnableMultiBufferPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());
    DominanceInfo dom(func);

    DenseMap<Operation *, Value> loop2Cond;

    SmallVector<pto::PointerCastOp> casts;
    func.walk([&](pto::PointerCastOp op) {
      if (op.getAddrs().size() > 1)
        casts.push_back(op);
    });

    for (pto::PointerCastOp op : casts) {
      auto mrTy = dyn_cast<MemRefType>(op.getType());
      if (!mrTy)
        continue;
      if (isGlobalMemRef(mrTy))
        continue;

      auto addrs = op.getAddrs();
      if (addrs.size() != 2) {
        op.emitError("only double-buffer pointer_cast (2 addresses) is supported");
        return signalPassFailure();
      }

      int64_t addr0 = 0, addr1 = 0;
      if (!getConstI64(addrs[0], addr0) || !getConstI64(addrs[1], addr1) ||
          addr0 < 0 || addr1 < 0) {
        op.emitError("expected constant non-negative i64 addrs for double-buffer pointer_cast");
        return signalPassFailure();
      }

      // Collect the enclosing loop for each use site. The resulting LCA is the
      // loop in which we materialize the ping/pong selector.
      SmallVector<scf::ForOp> useLoops;
      llvm::DenseSet<Operation *> seen;
      for (OpOperand &use : op.getResult().getUses()) {
        Operation *owner = use.getOwner();
        if (!owner)
          continue;
        scf::ForOp enclosing = owner->getParentOfType<scf::ForOp>();
        if (!enclosing)
          continue;
        if (seen.insert(enclosing.getOperation()).second)
          useLoops.push_back(enclosing);
      }

      scf::ForOp baseLoop = lowestCommonAncestorLoop(useLoops);
      if (!baseLoop) {
        // No loop uses: keep behavior deterministic by dropping the extra addr.
        rewriter.setInsertionPoint(op);
        Attribute config = op.getConfig() ? Attribute(*op.getConfig()) : Attribute();
        Value vRow = op.getValidRow();
        Value vCol = op.getValidCol();
        auto collapsed = rewriter.create<pto::PointerCastOp>(
            op.getLoc(), op.getType(), ValueRange{addrs[0]},
            vRow ? vRow : Value(), vCol ? vCol : Value(), config);
        rewriter.replaceOp(op, collapsed.getResult());
        continue;
      }

      // If the original pointer_cast is used as an operand of the selected base
      // loop op, we cannot replace that use with a value defined inside the
      // loop. Treat this as unsupported to avoid miscompilation.
      for (OpOperand &use : op.getResult().getUses()) {
        if (use.getOwner() == baseLoop.getOperation()) {
          op.emitError("unsupported: multi-buffer pointer_cast used as an operand of the base scf.for");
          return signalPassFailure();
        }
      }

      // Hoist two single-address pointer_cast ops just before the base loop.
      rewriter.setInsertionPoint(baseLoop);
      Value c0 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), addr0, 64);
      Value c1 = rewriter.create<arith::ConstantIntOp>(op.getLoc(), addr1, 64);
      Attribute config = op.getConfig() ? Attribute(*op.getConfig()) : Attribute();
      Value vRow = op.getValidRow();
      Value vCol = op.getValidCol();

      if ((vRow && !dom.dominates(vRow, baseLoop.getOperation())) ||
          (vCol && !dom.dominates(vCol, baseLoop.getOperation()))) {
        op.emitError("unsupported: valid_row/valid_col must dominate the selected loop for hoisting");
        return signalPassFailure();
      }

      auto ptr0 = rewriter.create<pto::PointerCastOp>(
          op.getLoc(), op.getType(), ValueRange{c0}, vRow ? vRow : Value(),
          vCol ? vCol : Value(), config);
      auto ptr1 = rewriter.create<pto::PointerCastOp>(
          op.getLoc(), op.getType(), ValueRange{c1}, vRow ? vRow : Value(),
          vCol ? vCol : Value(), config);

      // Build (or reuse) loop-parity condition and select the active buffer.
      Value cond;
      auto it = loop2Cond.find(baseLoop.getOperation());
      if (it != loop2Cond.end()) {
        cond = it->second;
      } else {
        cond = buildLoopNestParityCond(rewriter, baseLoop);
        if (!cond) {
          op.emitError("failed to build loop-nest parity condition for multi-buffer selection");
          return signalPassFailure();
        }
        loop2Cond[baseLoop.getOperation()] = cond;
      }

      rewriter.setInsertionPointAfter(cond.getDefiningOp());
      Value selected = rewriter.create<arith::SelectOp>(
          op.getLoc(), cond, ptr1.getResult(), ptr0.getResult());

      // Replace uses that are inside the base loop body (including nested ops).
      SmallVector<OpOperand *> toReplace;
      for (OpOperand &use : op.getResult().getUses()) {
        Operation *owner = use.getOwner();
        if (owner && isInLoopBody(owner, baseLoop))
          toReplace.push_back(&use);
      }
      for (OpOperand *use : toReplace)
        use->set(selected);

      if (op.getResult().use_empty())
        op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPTOEnableMultiBufferPass() {
  return std::make_unique<PTOEnableMultiBufferPass>();
}

} // namespace pto
} // namespace mlir
