#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
 
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
 
namespace mlir {
namespace pto {
 
class PTOIRTranslator {
public:
  PTOIRTranslator(SyncIRs &syncIR,
                  MemoryDependentAnalyzer &memDepAnalyzer,
                  Buffer2MemInfoMap &buffer2MemInfoMap,
                  func::FuncOp func,
                  SyncAnalysisMode syncAnalysisMode)
    : func_(func), 
      index(0),
      syncIR_(syncIR), 
      buffer2MemInfoMap_(buffer2MemInfoMap),
      memAnalyzer_(memDepAnalyzer),
      mode_(syncAnalysisMode) { };
 
  // 核心入口：执行 IR 分析和转换
  void Build();
 
  // 获取生成的 SyncIR (指令序列)
  SyncIRs &getSyncIR() { return syncIR_; }
 
  // 获取 Buffer 分析结果 (别名映射)
  Buffer2MemInfoMap &getBuffer2MemInfoMap() { return buffer2MemInfoMap_; }
 
  // 打印调试信息 (Buffer Map 和 SyncIR)
  void print();
 
private:
  func::FuncOp func_;
  unsigned index; // 当前 SyncIR 节点的索引计数器
  
  // 核心数据结构 (定义在 SyncCommon.h 中)
  SyncIRs &syncIR_;
  Buffer2MemInfoMap &buffer2MemInfoMap_;
  MemoryDependentAnalyzer &memAnalyzer_;
  SyncAnalysisMode mode_;
 
  // --- 递归遍历逻辑 ---
  void RecursionIR(Region *region);
 
  // --- 内存/Alias 分析 ---
  void UpdateKernelArgMemInfo();
  LogicalResult UpdateAllocTileOpMemInfo(pto::AllocTileOp op);
  LogicalResult UpdatePointerCastOpMemInfo(pto::PointerCastOp op);
  LogicalResult UpdateMemrefAllocOpMemInfo(memref::AllocOp op);
  
  // 处理 View/Alias (MakeTensorView, Subview, Mov)
  void UpdateAliasBufferInfo(Value result, Value source);
 
  // --- 控制流处理 (SCF) ---
  void UpdateForOpInfo(scf::ForOp forOp);
  void UpdateWhileOpInfo(scf::WhileOp whileOp);
  void UpdateIfOpInfo(scf::IfOp ifOp);
  void UpdateYieldOpInfo(scf::YieldOp yieldOp);
 
  // --- 核心：处理计算/搬运指令 (生成 Compound 节点) ---
  void UpdatePTOOpInfo(Operation *op);
 
  // --- 辅助函数 ---
  
  // 获取 PTO Op 对应的硬件流水线类型
  PipelineType getOpPipeline(Operation *op);
 
  // 根据 Values 填充 Def/Use 列表
  void UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec);
 
  // 调试辅助
  std::string getPipelineName(PipelineType pipe);
  void printMemInfoList(llvm::raw_ostream &os, 
                        const SmallVector<const BaseMemInfo *> &list, 
                        AsmState &state);
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
