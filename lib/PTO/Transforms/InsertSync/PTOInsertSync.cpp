//===- PTOInsertSync.cpp - PTO Insert Synchronization for PTO Pipeline ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "PTO/Transforms/InsertSync/MoveSyncState.h"
#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // [FIX] 确保 FuncOp 定义可见

// [CRITICAL FIX] 必须在包含 .inc 之前设置好命名空间环境
// 将 Passes.h.inc 生成的声明包裹在 namespace mlir 中
// 此外，为了确保 Passes.h.inc 中生成的 func::FuncOp 能被解析，
// 我们需要在 pto 命名空间内给 func 做一个别名。

namespace mlir {
namespace pto {
  // [FIX] 给 mlir::func 起别名为 func，这样 .inc 文件里的 func::FuncOp 就能找到了
  namespace func = ::mlir::func; 
 
  #define GEN_PASS_DEF_PTOINSERTSYNC 
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir
 
using namespace mlir;
using namespace mlir::pto;
 
namespace {
 
// ==============================================================================
// Main Pass Implementation
// ==============================================================================
 
struct PTOInsertSyncPass : public mlir::pto::impl::PTOInsertSyncBase<PTOInsertSyncPass> {
  
  void runOnOperation() override {
    llvm::errs() << "\n// === [PTOInsertSync] Start === //\n";
    func::FuncOp func = getOperation();

    // If the function already contains explicit synchronization ops (either
    // low-level pipe flags or the higher-level record/wait events), do not run
    // the automatic insertion pass again. Re-inserting on top of manual sync
    // can introduce duplicated/mismatched event dependencies that may lead to
    // runtime failures on NPU.
    //
    bool hasExplicitSync = false;
    func.walk([&](Operation *op) {
      if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::RecordEventOp,
              pto::WaitEventOp>(op)) {
        hasExplicitSync = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasExplicitSync) {
      return;
    }
    
    // 0. 数据结构准备
    MemoryDependentAnalyzer memAnalyzer;
    SyncIRs syncIR;
    SyncOperations syncOpsStorage;
    Buffer2MemInfoMap buffer2MemInfoMap;
 
    // 1. Translator: 构建 SyncIR
    PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func, SyncAnalysisMode::NORMALSYNC);
    translator.Build();
    
    // 如果 IR 太简单，直接跳过
    if (syncIR.size() <= 1) return;
    
    dumpInsertSyncPhase("After Translator", syncIR, syncOpsStorage,
                        func.getOperation());
 
    // 2. Analyzer: 依赖分析与插入逻辑 Sync
    InsertSyncAnalysis analyzer(syncIR, memAnalyzer, syncOpsStorage, func,
                                SyncAnalysisMode::NORMALSYNC);
    analyzer.Run(/*insertBarAllAtLast=*/true);
 
    dumpInsertSyncPhase("After Analysis", syncIR, syncOpsStorage,
                        func.getOperation());
 
    // [NEW] 3. Optimization: Sync Motion
    // 将不必要的 Wait 提至 Loop 外，将不必要的 Set 沉降到 Loop 后
    MoveSyncState syncMove(syncIR, syncOpsStorage);
    syncMove.Run(); // 执行优化
 
    dumpInsertSyncPhase("After Sync Motion", syncIR, syncOpsStorage,
                        func.getOperation());
 
    // 4. [NEW] Optimization 2: Remove Redundant Sync
    // 消除由于 Motion 或 Analysis 产生的冗余同步对
    RemoveRedundantSync removeRedundant(syncIR, syncOpsStorage, SyncAnalysisMode::NORMALSYNC);
    removeRedundant.Run();
 
    dumpInsertSyncPhase("After Remove Redundant Sync", syncIR, syncOpsStorage,
                        func.getOperation());
    
    SyncEventIdAllocation eventIdAllocation(syncIR, syncOpsStorage);
    eventIdAllocation.Allocate();
 
    dumpInsertSyncPhase("After EventId Allocation", syncIR, syncOpsStorage,
                        func.getOperation());
 
    SyncCodegen codegen(syncIR, func, SyncAnalysisMode::NORMALSYNC);
    codegen.Run();
 
    // 6. 最终结果打印
    llvm::errs() << "\n// === [PTOInsertSync] Final Result === //\n";
    func.print(llvm::errs());
    llvm::errs() << "\n\n";
    llvm::errs() << "// === [PTOInsertSync] End === //\n";
  }
};
 
} // namespace
 
std::unique_ptr<Pass> mlir::pto::createPTOInsertSyncPass() {
  return std::make_unique<PTOInsertSyncPass>();
}
