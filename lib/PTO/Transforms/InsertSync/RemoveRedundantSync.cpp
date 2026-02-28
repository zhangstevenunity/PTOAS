#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <vector>
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
void RemoveRedundantSync::Run() {
  // 1. 收集所有成对的同步指令 (Set/Wait)
  std::vector<std::pair<SyncOperation *, SyncOperation *>> syncOps;
  for (auto &syncPair : syncOperations_) {
    // 只有成对的 (Set, Wait) 才能进行此类消除，Barrier 不适用
    if (syncPair.size() == 2) {
      auto *setFlag = syncPair[0].get();
      auto *waitFlag = syncPair[1].get();
      syncOps.push_back(std::make_pair(setFlag, waitFlag));
    }
  }
 
  // 2. 排序：优先处理范围较小的或者是 Loop 内部的，
  // 这样如果它们被保留，可以用来消除外部更大的。
  // (这里采用简单且稳定的排序策略，确保处理顺序可预测)
  std::sort(syncOps.begin(), syncOps.end(),
       [](std::pair<SyncOperation *, SyncOperation *> syncPair1,
          std::pair<SyncOperation *, SyncOperation *> syncPair2) {
         auto *syncOp1 = syncPair1.first;
         auto *syncOp2 = syncPair2.first;
         
         bool hasLoop1 = syncOp1->GetForEndIndex().has_value();
         bool hasLoop2 = syncOp2->GetForEndIndex().has_value();
 
         if (hasLoop1 && hasLoop2) {
           if (syncOp1->GetForEndIndex().value() != syncOp2->GetForEndIndex().value()) {
             return syncOp1->GetForEndIndex().value() > syncOp2->GetForEndIndex().value();
           } else {
             return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
           }
         }
         if (hasLoop1 || hasLoop2) {
           return hasLoop1 > hasLoop2;
         }
         return syncOp1->GetSyncIndex() > syncOp2->GetSyncIndex();
       });
 
  // 3. 逐个检查并移除冗余
  for (auto [setFlag, waitFlag] : syncOps) {
    bool useless = CheckAllSync(setFlag, waitFlag);
    if (useless) {
      // 标记为冗余 (虽然这里是物理移除)
      
      // 从 SyncIR 中移除 Set
      auto &pipeAfter = syncIR_[setFlag->GetSyncIRIndex()]->pipeAfter;
      auto it0 = std::find(pipeAfter.begin(), pipeAfter.end(), setFlag);
      if (it0 != pipeAfter.end()) {
        pipeAfter.erase(it0);
      }
 
      // 从 SyncIR 中移除 Wait
      auto &pipeBefore = syncIR_[waitFlag->GetSyncIRIndex()]->pipeBefore;
      auto it1 = std::find(pipeBefore.begin(), pipeBefore.end(), waitFlag);
      if (it1 != pipeBefore.end()) {
        pipeBefore.erase(it1);
      }
      
      // 标记对象本身，避免 EventID 分配时分配给它
      setFlag->uselessSync = true;
      waitFlag->uselessSync = true;
    }
  }
}
 
bool RemoveRedundantSync::CheckAllSync(SyncOperation *setFlag,
                                       SyncOperation *waitFlag) {
  // syncFinder 用于跟踪在当前范围内，哪些 SyncIndex 的 Set 已经被看到了。
  // 如果随后看到了对应的 Wait，说明找到了一对完整的内部同步。
  SmallVector<bool> syncFinder(syncOperations_.size(), false);
  
  unsigned int begin = setFlag->GetSyncIRIndex();
  unsigned int end = waitFlag->GetSyncIRIndex();
  auto forEndIndex = setFlag->GetForEndIndex();
 
  if (begin < end) {
    // 普通的前向依赖
    return CheckRepeatSync(begin, end, syncFinder, setFlag);
  } else {
    // 循环回边 (Back-edge): Set 在后面，Wait 在前面 (Loop Head)
    // 这种情况下，范围是 [set, LoopEnd] + [LoopBegin, wait]
    checkCondition(forEndIndex.has_value(), "setFlag expected to have forEndIndex for back-edge sync");
    
    // 获取 Loop 节点
    auto *ptr = dyn_cast<LoopInstanceElement>(syncIR_[forEndIndex.value()].get());
    checkCondition(ptr != nullptr, "Invalid loop element for sync");
    
    // 分两段检查：只要任意一段路径上有覆盖，或者两段组合覆盖？
    // 注意：这里使用 OR（任意一段覆盖即认为冗余），属于更激进的策略。
    // 对回边来说，这通常意味着只要循环体内有更强的回边同步，或者...
    // 这是一个激进的策略。
    return CheckRepeatSync(begin, ptr->endId, syncFinder, setFlag) ||
           CheckRepeatSync(ptr->beginId, end, syncFinder, setFlag);
  }
}
 
bool RemoveRedundantSync::CheckRepeatSync(unsigned int begin, unsigned int end,
                                          SmallVector<bool> &syncFinder,
                                          SyncOperation *setFlag) {
  checkCondition(begin <= end, "expected begin <= end");
  checkSyncIRIndex(syncIR_, end);
  
  bool res = false;
 
  // 1. 检查起始节点的 Post (Set)
  for (auto &relatedSync : syncIR_[begin]->pipeAfter) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }
 
  // 2. 遍历中间节点
  for (unsigned i = begin + 1; i <= end - 1; i++) {
    checkSyncIRIndex(syncIR_, i);
    
    // Check Pre (Wait)
    for (auto &relatedSync : syncIR_[i]->pipeBefore) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
 
    // Recursion into Branch
    if (auto *branchElement = dyn_cast<BranchInstanceElement>(syncIR_[i].get())) {
      if (CheckBranchBetween(branchElement, syncFinder, setFlag, end, i)) {
        return true;
      }
    }
    
    // Recursion into Loop
    if (auto *forElement = dyn_cast<LoopInstanceElement>(syncIR_[i].get())) {
      if (CheckLoopBetween(forElement, setFlag, i)) {
        return true;
      }
    }
 
    // Check Post (Set)
    for (auto &relatedSync : syncIR_[i]->pipeAfter) {
      res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
    }
  }
 
  // 3. 检查结束节点的 Pre (Wait)
  for (auto &relatedSync : syncIR_[end]->pipeBefore) {
    res = res || CanMatchedSync(syncFinder, relatedSync, setFlag);
  }
  
  return res;
}
 
bool RemoveRedundantSync::CheckBranchBetween(
    BranchInstanceElement *branchElement, SmallVector<bool> syncFinder,
    SyncOperation *setFlag, unsigned endId, unsigned &i) {
  
  // 只处理 IF_BEGIN
  if (branchElement->getBranchKind() != KindOfBranch::IF_BEGIN) {
    i = branchElement->endId;
    return false;
  }
 
  bool hasElseBranch = branchElement->branchId < branchElement->endId;
  
  // 检查 waitFlag (endId) 是否在分支内部。如果是，我们不能简单跳过分支。
  // 这里逻辑是：如果当前的冗余检查范围跨越了整个分支（即 begin 在 if 前，end 在 if 后），
  // 那么我们需要检查是否在 THEN 和 ELSE 两个路径上都找到了内部同步。
  bool endIsInsideThenBranch =
      (!hasElseBranch && endId < branchElement->endId) ||
      (hasElseBranch && endId < branchElement->branchId);
  if (endIsInsideThenBranch) return false;
 
  bool endIsInsideElseBranch = hasElseBranch &&
                               endId >= branchElement->branchId &&
                               endId < branchElement->endId;
  if (endIsInsideElseBranch) {
    i = branchElement->branchId;
    return false;
  }
 
  // 核心：如果两个分支都存在内部覆盖，则整体覆盖
  if (hasElseBranch) {
    bool coveredInThen = CheckRepeatSync(branchElement->beginId, branchElement->branchId, syncFinder, setFlag);
    bool coveredInElse = CheckRepeatSync(branchElement->branchId, branchElement->endId, syncFinder, setFlag);
    
    if (coveredInThen && coveredInElse) {
      return true;
    }
  }
  // 如果只有 Then 分支 (Implicit Else)，除非我们在 Else (空路径) 上也能找到同步（不可能），
  // 否则无法断定冗余。所以单 If 分支通常无法帮助消除跨越它的外部同步。
 
  i = branchElement->endId; // 跳过整个分支块
  return false;
}
 
bool RemoveRedundantSync::CheckLoopBetween(LoopInstanceElement *loopElement,
                                           SyncOperation *setFlag,
                                           unsigned &i) {
  // 对于循环，保守起见暂时不深入检查内部是否覆盖外部。
  // 因为循环可能执行 0 次，如果循环内有同步，但循环不执行，外部依赖就没法满足。
  // 除非通过 Range Analysis 证明循环至少执行一次，否则这里返回 false 是安全的。
  i = loopElement->endId;
  return false;
}
 
bool RemoveRedundantSync::CanMatchedSync(SmallVector<bool> &syncFinder,
                                         SyncOperation *relatedSync,
                                         SyncOperation *setFlag) {
  // 1. 过滤不相关的同步
  // - 类型必须匹配 (Wait/Set)
  // - 不能是自己 (Index 相同)
  // - Pipe 必须完全一致 (Src->Dst)
  // - EventIdNum: 内部的同步能力必须强于外部 (related.eventIdNum >= set.eventIdNum ???) 
  //   这里暂时假设 Single Buffer (eventIdNum=1) 场景即可覆盖主流程
  
  bool isWait = (relatedSync->GetType() == SyncOperation::TYPE::WAIT_EVENT);
  bool isSet = (relatedSync->GetType() == SyncOperation::TYPE::SET_EVENT);
  
  // 支持 BlockSync 模式的检查
  if (syncAnalysisMode_ == SyncAnalysisMode::BLOCKSYNC) {
      isWait |= (relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT);
      isSet |= (relatedSync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET);
  }
 
  if (!isWait && !isSet) return false;
  if (relatedSync->GetSyncIndex() == setFlag->GetSyncIndex()) return false;
  
  // Pipe 检查：内部同步必须也是解决同样的 Src -> Dst 依赖
  if (relatedSync->GetSrcPipe() != setFlag->GetSrcPipe()) return false;
  if (relatedSync->GetDstPipe() != setFlag->GetDstPipe()) return false;
 
  // 2. 状态机逻辑
  // 如果遇到了 Set，记录下来
  if (isSet) {
    checkCondition(relatedSync->GetSyncIndex() < syncFinder.size(), "Index OOB");
    syncFinder[relatedSync->GetSyncIndex()] = true;
  }
  
  // 如果遇到了 Wait，且对应的 Set 之前已经遇到过 (syncFinder=true)
  // 说明我们在 [Begin, End] 范围内找到了一对完整的、同 Pipe 的同步 (Inner Pair)。
  // 这对 Inner Pair 保证了顺序，因此外部的 Pair 是冗余的。
  if (isWait) {
    checkCondition(relatedSync->GetSyncIndex() < syncFinder.size(), "Index OOB");
    if (syncFinder[relatedSync->GetSyncIndex()]) {
      return true; // Found redundant!
    }
  }
 
  return false;
}
