#ifndef MLIR_DIALECT_PTO_TRANSFORMS_BLOCKSYNCANALYSIS_H
#define MLIR_DIALECT_PTO_TRANSFORMS_BLOCKSYNCANALYSIS_H
 
#include "PTO/Transforms/SyncCommon.h"
#include "PTO/Transforms/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <array>
 
namespace mlir {
namespace pto {
 
struct SyncRecord {
  // Record pipes that have already been waited for (per multi-buffer slot).
  //
  // Use PIPE_LAST + 1 to keep indices stable even if PIPE_UNASSIGNED (99)
  // appears transiently in intermediate states.
  std::array<bool, static_cast<unsigned>(PipelineType::PIPE_LAST) + 1U>
      alreadySync{false};

  // Record the pairing status of set/wait within a syncIndex.
  llvm::DenseMap<int, bool> syncFinder;
};
 
using SyncRecordList = std::array<SyncRecord, MAX_MULTI_BUFFER_NUM>;
 
class BlockSyncAnalysis {
public:
  BlockSyncAnalysis(SyncIRs &syncIR, 
                    MemoryDependentAnalyzer &memDepAnalyzer,
                    SyncOperations &syncOperations, 
                    func::FuncOp func,
                    SyncAnalysisMode syncAnalysisMode = SyncAnalysisMode::NORMALSYNC)
      : syncIR_(syncIR), 
        memAnalyzer_(memDepAnalyzer),
        syncOperations_(syncOperations), 
        func_(func),
        syncAnalysisMode_(syncAnalysisMode) {}
 
  ~BlockSyncAnalysis() = default;
 
  /// 入口函数：执行分析并注入同步
  /// insertBarAllAtLast: 是否在最后插入一个全局 Barrier (通常需要)
  void Run(bool insertBarAllAtLast = true);
 
private:
  // --- Data Members ---
  SyncIRs &syncIR_;
  MemoryDependentAnalyzer &memAnalyzer_;
  SyncOperations &syncOperations_;
  func::FuncOp func_;
  SyncAnalysisMode syncAnalysisMode_;
  
  // 全局同步索引计数
  unsigned syncIndex_{0};
 
private:
  // --- Core Logic Methods ---
 
  /// 主调度器：处理 Compound (Op) 节点的同步
  void DealWithCompoundSync(CompoundInstanceElement *nowCompound);
 
  /// 递归调度器：处理 Loop 节点
  void DealWithLoopSync(LoopInstanceElement *nowElement);
 
  /// 核心算法：在前向扫描中，尝试插入同步
  /// 遍历 range [begin, end) 之间的节点
  void InsertSeqSync(CompoundInstanceElement *nowCompound, 
                     SyncIRs &syncElement,
                     int begin, int end, 
                     SyncRecordList &syncRecordList,
                     const std::optional<unsigned> &forEndIndex);
                     
  /// 处理 Loop 内部的递归调用
  unsigned InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex);

  unsigned InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex);

  /// 合并两个分支的同步状态 (Intersection)
  void MergeAlreadySync(SyncRecordList &syncRecordList,
                        const SyncRecordList &syncRecordIfList,
                        const SyncRecordList &syncRecordElseList);
 
  // --- Dependency & Sync Insertion ---

  /// Inset backward sync with LoopInstanceElement's end by copying a loop body
  /// slice and running the sequential inserter on the copied structure.
  void InsertBackForSync(CompoundInstanceElement *nowCompound,
                         SyncIRs &backSyncIr,
                         const LoopInstanceElement *loopElement);
 
  /// 检查 nowCompound 和 frontCompound 是否需要同步
  void InsertSync(CompoundInstanceElement *nowCompound,
                  CompoundInstanceElement *frontCompound,
                  SyncRecordList &syncRecordList,
                  const std::optional<unsigned> &forEndIndex);
 
  /// 调用 memAnalyzer 判断内存依赖
  void MemAnalyze(CompoundInstanceElement *nowCompound,
                  CompoundInstanceElement *frontCompound,
                  SyncRecordList &syncRecordList,
                  const std::optional<unsigned> &forEndIndex);
 
  /// 判断两个节点是否存在 RAW/WAR/WAW 依赖
  bool IsMemInfoHasDependency(CompoundInstanceElement *nowCompound,
                              CompoundInstanceElement *frontCompound,
                              DepBaseMemInfoPairVec &depBaseMemInfosVec);
 
  /// 实际创建 SyncOperation 对象并插入列表
  void InsertSyncOperation(CompoundInstanceElement *nowCompound,
                           CompoundInstanceElement *frontCompound,
                           DepBaseMemInfoPairVec &depBaseMemInfosVec,
                           const std::optional<unsigned> &forEndIndex);
 
  // --- Utility Methods ---
 
  /// 检查是否已经同步过 (Transitive Dependency Elimination)
  bool isAlreadySync(CompoundInstanceElement *nowCompound,
                     CompoundInstanceElement *frontCompound,
                     SyncRecordList &syncRecordList, 
                     unsigned recordListIndex);
 
  /// 更新 SyncRecord (当插入新同步后)
  void UpdateAlreadySync(const SyncOps &syncVector,
                         SyncRecordList &syncRecordList,
                         const PipelineType nowPipeValue);
                            
  void UpdateSyncRecordInfo(CompoundInstanceElement *frontCompound,
                            SyncRecordList &syncRecordList);
 
  void UpdateSyncRecord(const SyncOperation *sync, SyncRecord &syncRecord,
                        PipelineType nowPipeValue);
                        
  void InsertLastPipeAll();
  
  /// 快速判断是否不需要插入同步 (Fast Path Pruning)
  /// isBackwardDep: 是否是跨循环迭代的依赖 (Back-edge)
  bool IsNoNeedToInsertSync(const CompoundInstanceElement *nowCompound,
                            const CompoundInstanceElement *frontCompound,
                            bool isBackwardDep) const;
 
  /// 获取依赖对涉及的 Event ID 数量 (用于 Multi-Buffer 分析)
  int GetEventIdNum(const DepBaseMemInfoPairVec &depBaseMemInfosVec);
 
  /// 辅助函数：获取所有涉及的 Buffer (用于 LCA 计算，虽然现在简化了，保留接口)
  SmallVector<Value> GetMemInfoBuffers(const DepBaseMemInfoPairVec &depBaseMemInfosVec);
 
  /// 判断 buffer 是否是 Alloc 类操作 (用于溯源)
  bool IsMemAllocOp(Operation *op) const;
 
  /// 判断两个操作是否构成GM的读写冲突
  bool IsGMHazard(const CompoundInstanceElement *nowCompound,
                  const CompoundInstanceElement *frontCompound) const;
 
  // 暂时不需要处理 unlikely scope, block sync, virtual pipe 等复杂逻辑
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_BLOCKSYNCANALYSIS_H
