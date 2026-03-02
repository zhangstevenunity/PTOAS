#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_REMOVEREDUNDANTSYNC_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_REMOVEREDUNDANTSYNC_H
 
#include "PTO/Transforms/InsertSync/SyncCommon.h"
 
namespace mlir {
namespace pto {
 
class RemoveRedundantSync {
public:
  RemoveRedundantSync(
      SyncIRs &syncIR, SyncOperations &syncOperations,
      SyncAnalysisMode syncAnalysisMode = SyncAnalysisMode::NORMALSYNC)
      : syncIR_(syncIR), syncOperations_(syncOperations),
        syncAnalysisMode_(syncAnalysisMode){};
 
  ~RemoveRedundantSync() = default;
 
  /// 优化入口：执行冗余同步消除
  void Run();
 
private:
  SyncIRs &syncIR_;
  SyncOperations &syncOperations_;
  SyncAnalysisMode syncAnalysisMode_;
 
private:
  /// 检查某对同步 (set/wait) 是否被其他同步覆盖（从而变得多余）
  bool CheckAllSync(SyncOperation *setFlag, SyncOperation *waitFlag);
 
  /// 在指定的生命周期范围内 [begin, end] 检查是否存在重复/更紧的同步
  bool CheckRepeatSync(unsigned int begin, unsigned int end,
                       SmallVector<bool> &syncFinder, SyncOperation *setFlag);
 
  /// 处理 Branch (If/Else) 结构中的冗余检查
  /// 如果 If 和 Else 分支内都存在覆盖，则整体覆盖
  bool CheckBranchBetween(BranchInstanceElement *branchElement,
                          SmallVector<bool> syncFinder, SyncOperation *setFlag,
                          unsigned endId, unsigned &i);
 
  /// 处理 Loop 结构中的冗余检查 (当前实现较为保守)
  bool CheckLoopBetween(LoopInstanceElement *loopElement,
                        SyncOperation *setFlag, unsigned &i);
 
  /// 核心判断逻辑：检查遇到的 relatedSync 是否能构成对 setFlag 的覆盖
  bool CanMatchedSync(SmallVector<bool> &syncFinder, SyncOperation *relatedSync,
                      SyncOperation *setFlag);
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_REMOVEREDUNDANTSYNC_H
