#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCCODEGEN_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCCODEGEN_H
 
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
 
namespace mlir {
namespace pto {
 
/// 记录每个 Op 前后需要插入的同步指令
struct SyncPipeBuild {
  SyncOps pipeBefore;
  SyncOps pipeAfter;
};
 
class SyncCodegen {
public:
  SyncCodegen(SyncIRs &syncIR, func::FuncOp func,
              SyncAnalysisMode syncAnalysisMode)
      : syncIR_(syncIR), func_(func), syncAnalysisMode_(syncAnalysisMode) {};
 
  ~SyncCodegen() = default;
 
  /// 入口函数：执行代码生成
  void Run();
 
private:
  // --- 核心插入逻辑 ---
  void SyncInsert(IRRewriter &rewriter, Operation *op, SyncOperation *sync,
                  bool beforeInsert);
 
  // --- 预处理：构建 op2InsertSync 映射 ---
  void UpdateOpInsertSync(IRRewriter &rewriter);
  void UpdateCompoundOpInsertSync(CompoundInstanceElement *nowCompound);
  void updatePlaceHolderOpInsertSync(PlaceHolderInstanceElement *placeHolder);
  void UpdateLoopOpInsertSync(LoopInstanceElement *nowElement);
  void UpdateBranchOpInsertSync(BranchInstanceElement *nowElement);
 
  // --- 指令生成 ---
  void CreateBarrierOp(IRRewriter &rewriter, Operation *op, SyncOperation *sync,
                       bool beforeInsert);
 
  void CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter, Operation *op,
                                      SyncOperation *sync, bool beforeInsert);
 
  void CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter, Operation *op,
                                     SyncOperation *sync, bool beforeInsert);
 
  void CreateBlockSyncBarrierOp(IRRewriter &rewriter, Operation *op,
                                const SyncOperation *sync, bool beforeInsert);
 
  // --- 辅助函数 ---
  
  // 生成用于多缓冲 ID 选择的 Value (e.g., select(cond, id0, id1))
  Value GetBufferSelected(IRRewriter &rewriter, Operation *op,
                          SyncOperation *sync);
 
  // 生成嵌套循环的计数器 (用于多缓冲切换)
  Value createNestedIndexModular(IRRewriter &rewriter, Operation *defineOp);
 
private:
  SyncIRs &syncIR_;
  func::FuncOp func_;
  SyncAnalysisMode syncAnalysisMode_;
 
  // 记录 Op -> Sync 的映射
  DenseMap<const Operation *, SyncPipeBuild> op2InsertSync;
 
  // 记录 Loop -> Counter 的映射 (缓存)
  DenseMap<Operation *, Value> loop2BufferCounter;
 
  // 记录 SyncIndex -> EventID Value 的映射 (缓存)
  DenseMap<unsigned, Value> SyncIndex2SelectBuffer;
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCCODEGEN_HN_H
