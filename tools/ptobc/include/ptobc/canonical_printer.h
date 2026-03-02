#pragma once

#include <string>

#include <mlir/IR/BuiltinOps.h>

namespace ptobc {

struct CanonicalPrintOptions {
  /// If true, print operations in MLIR generic form (quoted op names).
  bool generic = false;

  /// If true, keep MLIR's default float printing. If false, force scalar
  /// FloatAttr constants to be printed as hex bitpatterns (`0x... : f32`).
  bool keepMLIRFloatPrinting = false;

  /// If true, print `loc(...)` debug locations (parseable form).
  bool printDebugInfo = false;
};

/// Print a ModuleOp in a canonical, parseable `.pto` form.
///
/// Today this is implemented as: MLIR pretty printer + targeted canonicalization
/// of scalar float constants.
std::string printModuleCanonical(mlir::ModuleOp module,
                                 const CanonicalPrintOptions &opt = {});

} // namespace ptobc
