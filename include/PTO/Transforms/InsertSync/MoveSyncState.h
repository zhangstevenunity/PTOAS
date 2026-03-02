#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_MOVESYNCSTATE_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_MOVESYNCSTATE_H
 
#include "PTO/Transforms/InsertSync/SyncCommon.h"
 
namespace mlir {
namespace pto {
 
class MoveSyncState {
public:
  MoveSyncState(SyncIRs &syncIR, SyncOperations &syncOperations)
      : syncIR_(syncIR), syncOperations_(syncOperations){};
 
  ~MoveSyncState() = default;
 
  /// 入口函数：执行同步指令移动优化 (Hoist/Sink)
  void Run();
 
private:
  SyncIRs &syncIR_;
  SyncOperations &syncOperations_;
 
private:
  // --- Branch (If/Else) Optimization ---
  void MoveOutBranchSync();
 
  void PlanMoveOutBranchSync(InstanceElement *e,
                             std::pair<unsigned int, unsigned int> pair,
                             std::pair<unsigned int, unsigned int> bound);
 
  void PlanMoveOutIfWaitSync(SyncOps &newPipeBefore, SyncOperation *s,
                             std::pair<unsigned int, unsigned int> pair,
                             std::pair<unsigned int, unsigned int> bound);
 
  void PlanMoveOutIfSetSync(SyncOps &newPipeAfter, SyncOperation *s,
                            std::pair<unsigned int, unsigned int> pair,
                            std::pair<unsigned int, unsigned int> bound);
 
  // --- Loop Optimization ---
  void MoveForSync();
 
  void MoveOutSync(InstanceElement *e,
                   std::pair<unsigned int, unsigned int> pair);
 
  void PlanMoveOutWaitSync(SyncOps &newPipeBefore, SyncOperation *s,
                           std::pair<unsigned int, unsigned int> pair);
 
  void PlanMoveOutSetSync(SyncOps &newPipeAfter, SyncOperation *s,
                          const std::pair<unsigned int, unsigned int> pair);
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_MOVESYNCSTATE_H
