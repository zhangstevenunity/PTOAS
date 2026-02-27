#include "PTO/Transforms/MoveSyncState.h"
#include "llvm/ADT/STLExtras.h" // For llvm::reverse
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
void MoveSyncState::Run() {
  MoveOutBranchSync();
  MoveForSync();
}
 
// ============================================================================
// Branch (If/Else) Logic
// ============================================================================
 
void MoveSyncState::MoveOutBranchSync() {
  for (auto &e : syncIR_) {
    if (auto *branchElement = dyn_cast<BranchInstanceElement>(e.get())) {
      // 只处理 IF_BEGIN，它是整个 Block 的入口
      if (branchElement->getBranchKind() == KindOfBranch::IF_BEGIN) {
        std::pair<unsigned, unsigned> bound = {branchElement->beginId,
                                               branchElement->endId};
        
        // 1. 遍历 THEN 分支内的指令
        for (unsigned i = branchElement->beginId + 1;
             i < branchElement->branchId; i++) {
          PlanMoveOutBranchSync(
              syncIR_[i].get(),
              {branchElement->beginId, branchElement->branchId}, bound);
        }
 
        // 如果没有 ELSE 分支，跳过
        if (branchElement->endId == branchElement->branchId) {
          continue;
        }
 
        // 2. 遍历 ELSE 分支内的指令
        for (unsigned i = branchElement->branchId + 1; i < branchElement->endId;
             i++) {
          PlanMoveOutBranchSync(syncIR_[i].get(),
                                {branchElement->branchId, branchElement->endId},
                                bound);
        }
      }
    }
  }
}
 
void MoveSyncState::PlanMoveOutBranchSync(
    InstanceElement *e, std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  // 处理 PipeBefore (Wait/Barrier) - 保持优化 (Hoist Wait)
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutIfWaitSync(newPipeBefore, s, pair, bound);
  }
  e->pipeBefore = newPipeBefore;
 
  // 处理 PipeAfter (Set) - Sink Set out of If/Else when the matched Wait is
  // outside the branch region (AscendNPU-IR behavior).
  //
  // This avoids patterns like:
  //   if (...) { set_flag(A) } else { set_flag(B) }
  //   wait_flag(A); wait_flag(B);   // may deadlock if wait consumes the flag
  //
  // By sinking the conditional sets to IF_END, we effectively materialize a
  // "merge" signal: regardless of which branch executed, the required events
  // are set after the join point.
  SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutIfSetSync(newPipeAfter, s, pair, bound);
  }
  e->pipeAfter = newPipeAfter;
}
 
void MoveSyncState::PlanMoveOutIfWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  // 只处理 WaitEvent
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  
  // 找到配对的 Set 操作
  auto *setSync = syncPair[0].get();
 
  // 如果 Set 操作在 If 块的外部 (index < pair.first 或 index > pair.second)
  // 那么这个 Wait 可以被提至 If 之前 (bound.first)
  if ((setSync->GetSyncIRIndex() >= pair.second) ||
      (setSync->GetSyncIRIndex() <= pair.first)) {
    
    // [Optimization]: Hoist Wait out of If
    checkSyncIRIndex(syncIR_, bound.first);
    syncIR_[bound.first]->pipeBefore.push_back(s); // 移到 IfBegin 之前
    s->SetSyncIRIndex(bound.first); // 更新索引
  } else {
    // 无法移动，保留在原地
    newPipeBefore.push_back(s);
  }
}
 
void MoveSyncState::PlanMoveOutIfSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair,
    std::pair<unsigned int, unsigned int> bound) {
  
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  
  // 找到配对的 Wait 操作
  auto *waitSync = syncPair[1].get();
 
  // 如果 Wait 操作在 If 块的外部
  // 那么这个 Set 可以沉降到 If 之后 (bound.second)
  if ((waitSync->GetSyncIRIndex() >= pair.second) ||
      (waitSync->GetSyncIRIndex() <= pair.first)) {
    
    // [Optimization]: Sink Set out of If
    checkSyncIRIndex(syncIR_, bound.second);
    syncIR_[bound.second]->pipeAfter.push_front(s); // 移到 IfEnd 之后
    s->SetSyncIRIndex(bound.second);
  } else {
    newPipeAfter.push_back(s);
  }
}
 
// ============================================================================
// Loop Optimization Logic
// ============================================================================
 
void MoveSyncState::MoveForSync() {
  for (auto &e : syncIR_) {
    if (auto *forCompound = dyn_cast<LoopInstanceElement>(e.get())) {
      // 找到 Loop End 节点（代表循环体的结束）
      if (forCompound->getLoopKind() == KindOfLoop::LOOP_END) {
        if (forCompound->ignore_block_sync_move_out) {
          continue;
        }
        // 遍历循环体内的所有指令
        for (unsigned i = forCompound->beginId + 1; i < forCompound->endId; i++)
          MoveOutSync(syncIR_[i].get(),
                      {forCompound->beginId, forCompound->endId});
      }
    }
  }
}
 
void MoveSyncState::MoveOutSync(InstanceElement *e,
                                std::pair<unsigned int, unsigned int> pair) {
  checkCondition(pair.first < e->GetIndex() && e->GetIndex() < pair.second,
                 "MoveOutSync expected element to be within pair bounds");
  
  // 处理 PipeBefore (Wait/Barrier)
  SyncOps newPipeBefore;
  for (auto &s : e->pipeBefore) {
    PlanMoveOutWaitSync(newPipeBefore, s, pair);
  }
  e->pipeBefore = newPipeBefore;
 
  // 处理 PipeAfter (Set)
  SyncOps newPipeAfter;
  for (auto &s : llvm::reverse(e->pipeAfter)) {
    PlanMoveOutSetSync(newPipeAfter, s, pair);
  }
  e->pipeAfter = newPipeAfter;
}
 
void MoveSyncState::PlanMoveOutWaitSync(
    SyncOps &newPipeBefore, SyncOperation *s,
    std::pair<unsigned int, unsigned int> pair) {
  
  if (s->GetType() != SyncOperation::TYPE::WAIT_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    newPipeBefore.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(!syncPair.empty(), "expected syncPair not to be empty");
  auto *setSync = syncPair[0].get();
 
  // 如果 Set 操作在 Loop 外部 (index > loop_end 或 index < loop_begin)
  // 说明依赖不来自循环内部（非 Loop-Carried Dependency）
  // 可以将 Wait 提至 Loop Begin 之前
  if ((setSync->GetSyncIRIndex() > pair.second) ||
      (setSync->GetSyncIRIndex() < pair.first)) {
    
    // [Optimization]: Hoist Wait out of Loop
    checkSyncIRIndex(syncIR_, pair.first);
    // pair.first 是 LoopBegin 节点
    syncIR_[pair.first]->pipeBefore.push_back(s); 
    s->SetSyncIRIndex(pair.first);
    return;
  }
  
  // 否则依赖来自循环内部，必须在循环内等待
  newPipeBefore.push_back(s);
}
 
void MoveSyncState::PlanMoveOutSetSync(
    SyncOps &newPipeAfter, SyncOperation *s,
    const std::pair<unsigned int, unsigned int> pair) {
  
  if (s->GetType() != SyncOperation::TYPE::SET_EVENT &&
      s->GetType() != SyncOperation::TYPE::SYNC_BLOCK_SET) {
    newPipeAfter.push_back(s);
    return;
  }
 
  auto &syncPair = syncOperations_[s->GetSyncIndex()];
  checkCondition(syncPair.size() > 1, "expected syncPair size > 1");
  auto *waitSync = syncPair[1].get();
 
  // 如果 Wait 操作在 Loop 外部
  // 说明循环内产生的信号，只在循环外被消费
  // 可以将 Set 沉降到 Loop End 之后
  if ((waitSync->GetSyncIRIndex() > pair.second) ||
      (waitSync->GetSyncIRIndex() < pair.first)) {
    
    // [Optimization]: Sink Set out of Loop
    checkSyncIRIndex(syncIR_, pair.second);
    // pair.second 是 LoopEnd 节点
    syncIR_[pair.second]->pipeAfter.push_front(s); 
    s->SetSyncIRIndex(pair.second);
    return;
  }
  
  newPipeAfter.push_back(s);
}
