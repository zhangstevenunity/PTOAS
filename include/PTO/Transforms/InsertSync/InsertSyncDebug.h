#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INSERT_SYNC_DEBUG_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INSERT_SYNC_DEBUG_H

#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {

enum class InsertSyncDebugLevel : unsigned {
  Off = 0,
  Phase = 1,
  SyncIR = 2,
  Trace = 3,
};

/// Runtime-configurable debug verbosity for the InsertSync pipeline.
unsigned getInsertSyncDebugLevel();

/// Returns true when InsertSync debug is enabled at or above \p minLevel.
bool isInsertSyncDebugEnabled(
    InsertSyncDebugLevel minLevel = InsertSyncDebugLevel::Phase);

struct InsertSyncDumpOptions {
  bool showMemInfo{false};
  bool showUselessSync{false};
};

/// Print per-phase state for the InsertSync pipeline.
///
/// - Level >= Phase: prints a compact summary header and counts.
/// - Level >= SyncIR: prints a structured SyncIR dump.
/// - Level >= Trace: may include additional details (e.g. def/use mem infos).
void dumpInsertSyncPhase(llvm::StringRef phase, const SyncIRs &syncIR,
                         const SyncOperations &syncOperations,
                         Operation *opForPrinting = nullptr,
                         llvm::raw_ostream &os = llvm::errs());

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_INSERT_SYNC_DEBUG_H
