#pragma once

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class MLIRContext;
}

namespace ptobc {

/// Decode a PTOBC v0 buffer (the full file contents, including header)
/// into an MLIR module.
///
/// Throws std::runtime_error on malformed input.
mlir::OwningOpRef<mlir::ModuleOp>
decodePTOBCToModule(llvm::ArrayRef<uint8_t> fileBytes, mlir::MLIRContext &ctx);

} // namespace ptobc
