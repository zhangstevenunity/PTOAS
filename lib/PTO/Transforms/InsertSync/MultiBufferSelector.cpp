#include "PTO/Transforms/InsertSync/MultiBufferSelector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace mlir {
namespace pto {

Value buildLoopNestParityCond(IRRewriter &rewriter, scf::ForOp baseLoop) {
  if (!baseLoop)
    return nullptr;

  Location loc = baseLoop.getLoc();

  // Insert at the beginning of the base loop body so it dominates all uses
  // within the loop nest.
  rewriter.setInsertionPointToStart(baseLoop.getBody());

  Value idx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value nElems = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Collect loop nest from inner to outer (baseLoop, parent, ...).
  SmallVector<scf::ForOp> loops;
  for (scf::ForOp cur = baseLoop; cur; cur = cur->getParentOfType<scf::ForOp>())
    loops.push_back(cur);

  for (scf::ForOp loop : loops) {
    Value iv = loop.getInductionVar();
    Value lb = loop.getLowerBound();
    Value ub = loop.getUpperBound();
    Value step = loop.getStep();

    // iter = (iv - lb) / step
    Value iter = rewriter.create<arith::DivUIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, iv, lb), step);
    idx = rewriter.create<arith::AddIOp>(
        loc, idx, rewriter.create<arith::MulIOp>(loc, iter, nElems));

    // tripCount = ceilDiv(ub - lb, step) = (ub - lb + step - 1) / step
    Value span = rewriter.create<arith::SubIOp>(loc, ub, lb);
    Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, step, one);
    Value num = rewriter.create<arith::AddIOp>(loc, span, stepMinusOne);
    Value tripCount = rewriter.create<arith::DivUIOp>(loc, num, step);
    nElems = rewriter.create<arith::MulIOp>(loc, nElems, tripCount);
  }

  Value two = rewriter.create<arith::ConstantIndexOp>(loc, 2);
  Value mod = rewriter.create<arith::RemUIOp>(loc, idx, two);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, mod,
                                        zero);
}

} // namespace pto
} // namespace mlir

