#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace pto {

/// Build a boolean `cond` that flips between even/odd iterations across a loop
/// nest.
///
/// - The condition is inserted at the beginning of `baseLoop`'s body.
/// - The computed parity is based on a flattened linear index across `baseLoop`
///   and all its parent `scf.for` loops, supporting non-unit steps.
/// - Returns a null Value if `baseLoop` is invalid.
Value buildLoopNestParityCond(IRRewriter &rewriter, scf::ForOp baseLoop);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERTSYNC_MULTIBUFFERSELECTOR_H
