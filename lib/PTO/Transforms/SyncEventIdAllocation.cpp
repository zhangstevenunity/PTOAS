#include "PTO/Transforms/SyncEventIdAllocation.h"
#include "PTO/Transforms/SyncCommon.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;

static size_t getEventIdPoolSize(const SyncOperation *sync,
                                 uint64_t reservedBlockSyncEventIdNum) {
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    return kBlockSyncSetWaitEventIdNum - reservedBlockSyncEventIdNum;
  }
  return kTotalEventIdNum;
}

void SyncEventIdAllocation::Allocate(uint32_t runNum) {
  // 1. 正常分配
  for (auto &element : syncIR_) {
    AllocateEventId(element.get());
  }
  // 2. 尝试 Widen (复用)
  for (auto &e : syncIR_) {
    WidenEventId(e->pipeAfter);
  }
  
  IgnoreBackHeadAndTailSync();
  
  // 3. 处理资源不足需要重分配的情况
  if (!reallocatedPipePair.empty()) {
    ReallocatedEventId();
    for (auto &e : syncIR_) {
      WidenEventId(e->pipeAfter);
    }
  }
 
  // 4. 降级策略：如果还是没有 ID，降级为 PipeAll 全局同步
  auto status = ChangeNoEventIdSyncToPipeAll();
  if (status.failed() && runNum < kMaxWidenTryNum) {
    if (tryWidenOnFirstFound()) {
      // Clear and Retry
      reallocatedPipePair.clear();
      eventCyclePool.clear();
      clearAllocatedEventId();
      Allocate(runNum + 1);
    }
  }
}
 
bool SyncEventIdAllocation::tryWidenOnFirstFound() {
  for (auto pipePair : reallocatedPipePair) {
    for (auto &e : syncIR_) {
      for (auto &sync : e->pipeAfter) {
        if (sync->isSyncSetType() && !sync->uselessSync &&
            (ScopePair(sync) == pipePair)) {
          if (TryWidenByOtherSync(sync)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}
 
void SyncEventIdAllocation::reserveBlockAllEventIds() {
  bool blockSyncAllExists = false;
  for (auto &element : syncIR_) {
    for (auto &syncOp : element->pipeBefore) {
      if (syncOp->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
        blockSyncAllExists = true;
        break;
      }
    }
    // ... check pipeAfter ...
    if (blockSyncAllExists) break;
  }
  if (blockSyncAllExists) {
    reservedBlockSyncEventIdNum = 2;
  }
}
 
void SyncEventIdAllocation::SetBlockSyncAllEventID(SyncOperation *sync) {
  if (sync->syncCoreType == TCoreType::CUBE) {
    sync->eventIds.push_back(kBlockSyncAllCubeEventId);
  } else if (sync->syncCoreType == TCoreType::VECTOR) {
    sync->eventIds.push_back(kBlockSyncAllVectorEventId);
  } else {
    llvm_unreachable("auto-inserted sync all operation must be all cube or all vector");
  }
}
 
void SyncEventIdAllocation::AllocateEventId(InstanceElement *e) {
  for (auto &sync : e->pipeBefore) {
    if (sync->uselessSync) continue;
    if (!sync->eventIds.empty()) continue; // Already allocated
    if (sync->isBarrierType()) continue;   // Barrier needs no ID
 
    if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_ALL) {
      SetBlockSyncAllEventID(sync);
    } else if (sync->isSyncSetType() || sync->isSyncWaitType()) {
      SetEventId(sync);
    }
  }
}
 
size_t SyncEventIdAllocation::GetCompilerAvailableEventIdNum(const SyncOperation *sync) {
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    return kBlockSyncSetWaitEventIdNum - reservedBlockSyncEventIdNum;
  }
  auto it = reservedEventIdNum.find({sync->GetSrcPipe(), sync->GetDstPipe()});
  if (it != reservedEventIdNum.end()) {
    return kTotalEventIdNum - it->second;
  }
  return kTotalEventIdNum;
}
 
void SyncEventIdAllocation::SetEventId(SyncOperation *sync) {
  const size_t poolSize = getEventIdPoolSize(sync, reservedBlockSyncEventIdNum);
  const size_t availableEventIdNum = GetCompilerAvailableEventIdNum(sync);

  SmallVector<bool> eventIdLifetimeAvailableStatus = GetEventPool(sync, poolSize);
  SmallVector<bool> eventIdIdleStatus = GetEventIdIdleStatus(sync, poolSize);
  
  assert(eventIdLifetimeAvailableStatus.size() == poolSize);
  assert(eventIdIdleStatus.size() == poolSize);

  // Apply per-(src,dst) reservations by marking the "reserved tail" as
  // unavailable. Historically this pass treated reserved IDs as being at the
  // end of the [0..kTotalEventIdNum) range.
  if (availableEventIdNum < poolSize) {
    for (size_t id = availableEventIdNum; id < poolSize; ++id)
      eventIdLifetimeAvailableStatus[id] = false;
  }
  
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
  SmallVector<int> canAllocaEventId = GetAvailableEventId(
      sync, eventIdLifetimeAvailableStatus, eventIdIdleStatus, poolSize);
 
  if (canAllocaEventId.empty()) {
    return;
  } else if (canAllocaEventId.size() >= idSize) {
    for (auto &id : canAllocaEventId) {
      SetEventPool(sync, id);
    }
  } else if (reallocatedPipePair.count(ScopePair(sync)) &&
             (canAllocaEventId.size() < idSize)) {
    // Reallocate strategy: reduce usage to 1
    assert(canAllocaEventId.size() > 0);
    SetEventPool(sync, canAllocaEventId[0]);
    sync->eventIdNum = 1;
  }
}
 
SmallVector<int> SyncEventIdAllocation::UpdateBlockAvailableEventId(
    SyncOperation *sync, SmallVector<bool> eventIdLifetimeAvailableStatus,
    size_t eventIdNum) {
  SmallVector<int> canAllocaEventId;
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) break;
    if (!canAllocaEventId.empty() && !eventIdLifetimeAvailableStatus[id]) {
      canAllocaEventId.clear();
      continue;
    }
    if (eventIdLifetimeAvailableStatus[id]) {
      canAllocaEventId.push_back(id);
    }
  }
  return canAllocaEventId;
}
 
SmallVector<int> SyncEventIdAllocation::GetAvailableEventId(
    SyncOperation *sync, SmallVector<bool> eventIdLifetimeAvailableStatus,
    SmallVector<bool> eventIdIdleStatus, size_t eventIdNum) {
  
  SmallVector<int> canAllocaEventId;
  size_t idSize = static_cast<size_t>(sync->eventIdNum);
 
  if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    return UpdateBlockAvailableEventId(sync, eventIdLifetimeAvailableStatus, eventIdNum);
  }
 
  // Strategy 1: Prioritize idle IDs
  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) break;
    if (eventIdLifetimeAvailableStatus[id] && eventIdIdleStatus[id]) {
      eventIdLifetimeAvailableStatus[id] = false;
      canAllocaEventId.push_back(id);
    }
  }
 
  // Strategy 2: Use any available
  for (unsigned id = 0; id < eventIdNum; id++) {
    if (canAllocaEventId.size() == idSize) break;
    if (eventIdLifetimeAvailableStatus[id]) {
      eventIdLifetimeAvailableStatus[id] = false;
      canAllocaEventId.push_back(id);
    }
  }
  return canAllocaEventId;
}
 
SmallVector<bool> SyncEventIdAllocation::GetEventIdIdleStatus(SyncOperation *sync,
                                                              size_t eventIdNum) {
  SmallVector<bool> eventIdIdleStatus;
  int scopePair = ScopePair(sync);
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  for (size_t i = 0; i < eventIdNum; i++) {
    auto &syncLifeCycle = seqPool.slot[i];
    eventIdIdleStatus.push_back(syncLifeCycle.empty());
  }
  return eventIdIdleStatus;
}
 
SmallVector<bool> SyncEventIdAllocation::GetEventPool(const SyncOperation *sync,
                                                      size_t eventIdNum) {
  SmallVector<bool> eventIdPool(eventIdNum, true);
  assert(sync->GetSyncIndex() < syncOperations_.size());
  auto &syncPair = syncOperations_[sync->GetSyncIndex()];
  auto *setFlag = syncPair[0].get();
  auto *waitFlag = syncPair[1].get();
 
  if (setFlag->GetForEndIndex().has_value()) {
    if (reallocatedPipePair.count(ScopePair(sync))) {
      auto *ptr = dyn_cast<LoopInstanceElement>(
          syncIR_[setFlag->GetForEndIndex().value()].get());
      assert(ptr != nullptr);
      FindUseEventID(ptr->beginId, ptr->endId, setFlag, eventIdPool);
    } else {
      FindUseEventID(0, syncIR_.size() - 1, setFlag, eventIdPool);
    }
  } else {
    FindUseEventID(setFlag->GetSyncIRIndex(), waitFlag->GetSyncIRIndex(),
                   setFlag, eventIdPool);
  }
  return eventIdPool;
}
 
int SyncEventIdAllocation::ScopePair(const SyncOperation *s) {
  if (s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET ||
      s->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    return 0;
  }
  // Event IDs are a limited shared resource and must not be reused across
  // overlapping lifetimes within the same (src,dst) pipe pair.
  //
  // Key by (src,dst) pipe pair to keep independent hardware pipe directions from
  // fighting over the same small event-id pool. This avoids unnecessary event-id
  // pressure where a single source pipe syncs to multiple destinations (e.g.
  // PIPE_M -> PIPE_MTE1 and PIPE_M -> PIPE_FIX), which can otherwise push some
  // pairs into high event IDs and trigger device-side failures.
  auto srcT = static_cast<unsigned int>(s->GetActualSrcPipe());
  auto dstT = static_cast<unsigned int>(s->GetActualDstPipe());
  // Offset by 1 so non-block scopes never collide with block-sync scope 0.
  return static_cast<int>(((dstT << 8U) | srcT) + 1U);
}
 
void SyncEventIdAllocation::FindUseEventID(unsigned int begin, unsigned int end,
                                           const SyncOperation *s,
                                           SmallVector<bool> &eventId) {
  const auto eventIdSize = eventId.size();
  if (begin < end) llvm::errs() << "begin: " << begin << " end: " << end << "\n";
  assert(begin < end);
  int scopePair = ScopePair(s);
  eventCyclePool.try_emplace(scopePair, EventCyclePool(eventIdSize));
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  // The pool is keyed by scopePair and should have a stable size.
  if (seqPool.slot.size() < eventIdSize)
    seqPool.slot.resize(eventIdSize);
  
  for (size_t i = 0; i < eventIdSize; i++) {
    auto &syncLifeCycle = seqPool.slot[i];
    if (syncLifeCycle.empty()) continue;
    
    if (CheckSyncLifeCycleConflict(syncLifeCycle, begin, end, eventId, i)) {
      continue;
    }
  }
}
 
bool SyncEventIdAllocation::CheckSyncLifeCycleConflict(
    SmallVector<unsigned int> &syncLifeCycle, unsigned int begin,
    unsigned int end, SmallVector<bool> &eventId, unsigned i) const {
  assert((syncLifeCycle.size() & 0x1) == 0 && "sync_life_cycle error.");
  if (syncLifeCycle[0] <= begin) {
    return true; // Conflict!
  }
  UpdateEventId(syncLifeCycle, begin, end, eventId, i);
  return false;
}
 
void SyncEventIdAllocation::UpdateEventId(
    SmallVector<unsigned int> &syncLifeCycle, const unsigned int begin,
    const unsigned int end, SmallVector<bool> &eventId,
    const unsigned index) const {
  for (size_t j = 0; j < syncLifeCycle.size(); j++) {
    if (syncLifeCycle[j] <= begin) {
      if (syncLifeCycle[j - 1] >= end && (j & 0x1) == 0) {
        break; // Safe interval
      } else {
        eventId[index] = false; // Conflict
      }
    } else if (j == syncLifeCycle.size() - 1) {
      assert((j & 0x1) == 1);
      if (syncLifeCycle[j] >= end) {
        break; // Safe
      } else {
        eventId[index] = false; // Conflict
      }
    }
  }
}
 
void SyncEventIdAllocation::SetEventPool(const SyncOperation *sync,
                                         unsigned eventId) {
  assert(sync->GetSyncIndex() < syncOperations_.size());
  auto &syncPair = syncOperations_[sync->GetSyncIndex()];
  
  // [Fix] 遍历组内所有 SyncOperation，为它们统一分配 Event ID
  // 这样无论是 Then-Set, Else-Set 还是 Wait，都会得到相同的 ID
  for (auto &op : syncPair) {
      op->eventIds.push_back(eventId);
  }
 
  // 下面的生命周期计算逻辑 (SetUseEventID) 可以保持不变，
  // 继续使用 syncPair[0] (Then-Set) 和 syncPair[1] (Wait) 来代表整个组的生命周期。
  // 因为 Then-Set 到 Wait 的区间通常覆盖了 Else-Set 到 Wait 的区间。
  auto &setFlag = syncPair[0];
  auto &waitFlag = syncPair[1];
 
  if (setFlag->GetForEndIndex().has_value()) {
    if (reallocatedPipePair.count(ScopePair(sync))) {
      auto *ptr = dyn_cast<LoopInstanceElement>(
          syncIR_[setFlag->GetForEndIndex().value()].get());
      assert(ptr != nullptr);
      SetUseEventID(ptr->beginId, ptr->endId, setFlag.get(), eventId);
    } else {
      SetUseEventID(0, syncIR_.size(), setFlag.get(), eventId);
    }
  } else {
    SetUseEventID(setFlag->GetSyncIRIndex(), waitFlag->GetSyncIRIndex(),
                  setFlag.get(), eventId);
  }
 
  // UpdateBackwardMatchSync 只处理回边同步，通常不需要 Phantom Set 参与
  if (setFlag->GetForEndIndex().has_value()) {
    UpdateBackwardMatchSync(setFlag.get(), waitFlag.get(), eventId);
  }
}
 
void SyncEventIdAllocation::UpdateBackwardMatchSync(
    const SyncOperation *setFlag, const SyncOperation *waitFlag,
    unsigned eventId) {
  // Creating new sync pair for backward match
  auto syncFront = std::make_unique<SyncOperation>(
      setFlag->GetType(), setFlag->GetSrcPipe(), setFlag->GetDstPipe(),
      static_cast<unsigned>(syncOperations_.size()), setFlag->GetSyncIRIndex(),
      setFlag->GetForEndIndex());
      
  auto syncEnd = syncFront->GetMatchSync(waitFlag->GetSyncIRIndex());
  
  syncFront->syncCoreType = setFlag->syncCoreType;
  syncEnd->syncCoreType = waitFlag->syncCoreType;
  syncFront->eventIds.push_back(eventId);
  syncEnd->eventIds.push_back(eventId);
 
  if (reallocatedPipePair.count(ScopePair(setFlag))) {
    auto *ptr = dyn_cast<LoopInstanceElement>(
        syncIR_[setFlag->GetForEndIndex().value()].get());
    assert(ptr != nullptr);
    syncFront->SetSyncIRIndex(ptr->beginId);
    syncEnd->SetSyncIRIndex(ptr->endId);
    syncFront->reallocatedLoopHeadTailSync = true;
    syncEnd->reallocatedLoopHeadTailSync = true;
    syncIR_[ptr->beginId]->pipeBefore.push_back(syncFront.get());
    syncIR_[ptr->endId]->pipeAfter.push_back(syncEnd.get());
  } else {
    syncFront->SetSyncIRIndex(0);
    syncEnd->SetSyncIRIndex(syncIR_.size() - 1);
    syncIR_[0]->pipeBefore.push_back(syncFront.get());
    syncIR_[syncIR_.size() - 1]->pipeAfter.push_back(syncEnd.get());
  }
 
  insertedBackwardSync.insert(syncFront.get());
  insertedBackwardSync.insert(syncEnd.get());
  
  SmallVector<std::unique_ptr<SyncOperation>> newSync;
  newSync.emplace_back(std::move(syncFront));
  newSync.emplace_back(std::move(syncEnd));
  syncOperations_.emplace_back(std::move(newSync));
}
 
void SyncEventIdAllocation::SetUseEventID(unsigned int begin, unsigned int end,
                                          const SyncOperation *setFlag,
                                          unsigned int eventId) {
  assert(begin < end);
  int scopePair = ScopePair(setFlag);
  const size_t poolSize =
      getEventIdPoolSize(setFlag, reservedBlockSyncEventIdNum);
  eventCyclePool.try_emplace(scopePair, EventCyclePool(poolSize));
  
  EventCyclePool &seqPool = eventCyclePool[scopePair];
  if (seqPool.slot.size() < poolSize)
    seqPool.slot.resize(poolSize);
  auto &syncLifeCycle = seqPool.slot[eventId];
  bool isInsert = false;
 
  if (syncLifeCycle.empty()) {
    syncLifeCycle.push_back(end);
    syncLifeCycle.push_back(begin);
    isInsert = true;
  } else {
    if (syncLifeCycle[0] <= begin) {
      syncLifeCycle.insert(syncLifeCycle.begin(), begin);
      syncLifeCycle.insert(syncLifeCycle.begin(), end);
      return;
    } else if (syncLifeCycle.back() >= end) {
      syncLifeCycle.insert(syncLifeCycle.end(), end);
      syncLifeCycle.insert(syncLifeCycle.end(), begin);
      return;
    } else if (ExtendLifecycle(syncLifeCycle, begin, end)) {
      return;
    }
  }
  if (!isInsert) llvm_unreachable("Can't insert this sync cycle!");
}
 
bool SyncEventIdAllocation::ExtendLifecycle(
    SmallVector<unsigned int> &syncLifeCycle, unsigned int beginNew,
    unsigned int endNew) const {
  for (size_t j = 0; j < syncLifeCycle.size() / 2U; j++) {
    uint &endOld = syncLifeCycle[j * 2U];
    uint &beginOld = syncLifeCycle[j * 2U + 1];
 
    bool widenLifeCycleBegin = endOld >= endNew && endNew > beginOld;
    bool widenLifeCycleEnd = endOld > beginNew && beginNew >= beginOld;
    bool insertMiddleLifecycle = j < ((syncLifeCycle.size() / 2U) - 1) &&
                                 beginOld >= endNew &&
                                 beginNew >= syncLifeCycle[(j + 1) * 2U];
    if (widenLifeCycleBegin) {
      beginOld = std::min(beginOld, beginNew);
      return true;
    } else if (widenLifeCycleEnd) {
      endOld = std::max(endOld, endNew);
      return true;
    } else if (insertMiddleLifecycle) {
      syncLifeCycle.insert(syncLifeCycle.begin() + (j + 1) * 2U, beginNew);
      syncLifeCycle.insert(syncLifeCycle.begin() + (j + 1) * 2U, endNew);
      return true;
    }
  }
  return false;
}
 
void SyncEventIdAllocation::WidenEventId(SyncOps syncVector) {
  for (auto &sync : syncVector) {
    if (sync->isSyncSetType() && sync->eventIds.empty() && !sync->uselessSync) {
      bool canWiden = TryWidenByOtherSync(sync);
      if (!canWiden) {
        int scopePair = ScopePair(sync);
        reallocatedPipePair.insert(scopePair);
      }
    }
  }
}
 
void SyncEventIdAllocation::clearAllocatedEventId() {
  // Remove generated BackwardSync
  for (auto &e : syncIR_) {
    SyncOps newPipeBefore;
    for (auto *sync : e->pipeBefore) {
      if (!insertedBackwardSync.contains(sync)) {
        newPipeBefore.push_back(sync);
      }
    }
    e->pipeBefore = newPipeBefore;
 
    SyncOps newPipeAfter;
    for (auto *sync : e->pipeAfter) {
      if (!insertedBackwardSync.contains(sync)) {
        newPipeAfter.push_back(sync);
      }
    }
    e->pipeAfter = newPipeAfter;
  }
  // Clear IDs
  for (auto &e : syncIR_) {
    for (auto &sync : e->pipeBefore) ClearEventId(sync);
    for (auto &sync : e->pipeAfter) ClearEventId(sync);
  }
}
 
void SyncEventIdAllocation::ReallocatedEventId() {
  for (auto pipePair : reallocatedPipePair) {
    eventCyclePool.erase(pipePair);
  }
  ClearReallocatedBackwardMatchSync();
  for (auto &e : syncIR_) {
    for (auto &sync : e->pipeBefore) {
      if (!sync->isBarrierType() && reallocatedPipePair.count(ScopePair(sync))) {
        ClearEventId(sync);
        SetEventId(sync);
      }
    }
  }
}
 
void SyncEventIdAllocation::ClearEventId(const SyncOperation *sync) {
  if (sync->isBarrierType()) return;
  auto &syncPair = syncOperations_[sync->GetSyncIndex()];
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();
  setSync->eventIds.clear();
  waitSync->eventIds.clear();
}
 
void SyncEventIdAllocation::ClearReallocatedBackwardMatchSync() {
  SyncOps newPipeBefore;
  for (auto &sync : syncIR_[0]->pipeBefore) {
    if (!(sync->isSyncSetType() && reallocatedPipePair.count(ScopePair(sync)))) {
      newPipeBefore.push_back(sync);
    }
  }
  syncIR_[0]->pipeBefore = newPipeBefore;
 
  SyncOps newPipeAfter;
  for (auto &sync : syncIR_[syncIR_.size() - 1]->pipeAfter) {
    if (!(sync->isSyncWaitType() && reallocatedPipePair.count(ScopePair(sync)))) {
      newPipeAfter.push_back(sync);
    }
  }
  syncIR_[syncIR_.size() - 1]->pipeAfter = newPipeAfter;
}
 
llvm::LogicalResult SyncEventIdAllocation::ChangeNoEventIdSyncToPipeAll() {
  for (auto &e : syncIR_) {
    for (auto &sync : e->pipeAfter) {
      if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT &&
          sync->reallocatedLoopHeadTailSync) {
        MoveOutBackwardMatchSync(sync);
      }
      if (sync->GetType() == SyncOperation::TYPE::SET_EVENT &&
          sync->eventIds.empty() && !sync->uselessSync) {
        // Fallback to PipeAll
        auto &syncPair = syncOperations_[sync->GetSyncIndex()];
        syncPair[0]->uselessSync = true;
        syncPair[1]->SetPipeAll();
      }
      if (sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET &&
          sync->eventIds.empty() && !sync->uselessSync) {
        return failure();
      }
    }
  }
  return success();
}
 
void SyncEventIdAllocation::MoveOutBackwardMatchSync(
    const SyncOperation *reallocatedSync) {
  auto &syncPair = syncOperations_[reallocatedSync->GetSyncIndex()];
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();
  bool isConflictEventId = false;
 
  // Conflict detection logic (simplified for PTO port)
  for (unsigned int i = 0; i <= syncIR_.size() - 1; i++) {
    if (isConflictEventId) break;
    if ((i > setSync->GetSyncIRIndex()) && (i < waitSync->GetSyncIRIndex())) continue;
    
    for (auto &sync : syncIR_[i]->pipeBefore) {
      if (!sync->uselessSync &&
          reallocatedSync->GetSyncIndex() != sync->GetSyncIndex() &&
          sync->GetActualSrcPipe() == reallocatedSync->GetActualSrcPipe() &&
          sync->GetActualDstPipe() == reallocatedSync->GetActualDstPipe() &&
          sync->eventIds == reallocatedSync->eventIds) {
        isConflictEventId = true;
        break;
      }
    }
  }
 
  if (!isConflictEventId) {
    setSync->uselessSync = true;
    waitSync->uselessSync = true;
    UpdateBackwardMatchSync(setSync, waitSync, setSync->eventIds[0]);
  }
}
 
void SyncEventIdAllocation::IgnoreBackHeadAndTailSync() {
  // Implementation specific logic for MTE1->M pipe pair optimization
  for (auto &sync : syncIR_[0]->pipeBefore) {
    bool isPipeMTE1ToPipeMSync = sync->GetSrcPipe() == PipelineType::PIPE_M &&
                                 sync->GetDstPipe() == PipelineType::PIPE_MTE1;
    if (!isPipeMTE1ToPipeMSync) continue;
    
    auto &syncPair = syncOperations_[sync->GetSyncIndex()];
    if (sync->eventIds.empty()) {
      syncPair[0]->uselessSync = true;
      syncPair[1]->uselessSync = true;
    }
  }
}
 
bool SyncEventIdAllocation::TryWidenByOtherSync(const SyncOperation *sync) {
  assert(!sync->isBarrierType());
  auto &syncPair = syncOperations_[sync->GetSyncIndex()];
  SyncOperation *setSync = syncPair[0].get();
  SyncOperation *waitSync = syncPair[1].get();
 
  SyncOperation *widenSync = FindWidenSync(setSync, waitSync);
  if (widenSync == nullptr) return false;
  
  setSync->uselessSync = true;
  waitSync->uselessSync = true;
  
  auto &widenSyncPair = syncOperations_[widenSync->GetSyncIndex()];
  SyncOperation *widenSet = widenSyncPair[0].get();
 
  // If sync ranges are disjoint, we might need to merge the sync nodes in IR
  if (setSync->GetSyncIRIndex() != widenSet->GetSyncIRIndex()) {
    auto *widenSetSyncIR = syncIR_[widenSet->GetSyncIRIndex()].get();
    SyncOps newPipeAfter;
    bool removeSync = false;
    for (auto &s : widenSetSyncIR->pipeAfter) {
      if (s == widenSet) {
        syncIR_[setSync->GetSyncIRIndex()]->pipeAfter.push_back(widenSet);
        widenSet->SetSyncIRIndex(setSync->GetSyncIRIndex());
        widenSet->reuseCntForWiden++;
        removeSync = true;
      } else {
        newPipeAfter.push_back(s);
      }
    }
    widenSetSyncIR->pipeAfter = newPipeAfter;
    if (!removeSync) llvm_unreachable("in widen fun, remove sync failed");
  }
  return true;
}
 
SyncOperation *
SyncEventIdAllocation::FindWidenSync(const SyncOperation *setSync,
                                     const SyncOperation *waitSync) {
  // Complex logic to find a compatible sync to widen (reuse ID)
  // Iterating backwards from setSync position
  int endIndex = 0;
  if (setSync->GetForEndIndex().has_value()) {
     auto *forCompound = dyn_cast<LoopInstanceElement>(syncIR_[setSync->GetForEndIndex().value()].get());
     endIndex = static_cast<int>(forCompound->beginId);
  }
 
  for (int loopId = static_cast<int>(setSync->GetSyncIRIndex()); loopId >= endIndex; loopId--) {
    auto *tmpIr = syncIR_[loopId].get();
    
    // Stop at control flow boundaries logic...
    if (auto *loopInst = dyn_cast<LoopInstanceElement>(tmpIr)) {
       if (loopInst->getLoopKind() == KindOfLoop::LOOP_BEGIN) break;
       if (loopInst->getLoopKind() == KindOfLoop::LOOP_END) loopId = static_cast<int>(loopInst->beginId);
    }
    // ... Branch checks ...
 
    for (auto &setSame : tmpIr->pipeAfter) {
        // ... Logic to check compatibility (Type, Pipe, Direction) ...
        bool isSameTypeSync = (setSame != setSync) &&
                              (setSame->GetDstPipe() == setSync->GetDstPipe()) &&
                              (setSame->GetSrcPipe() == setSync->GetSrcPipe());
        
        if (!isSameTypeSync || setSame->uselessSync || setSame->eventIds.empty()) continue;
 
        auto &syncPair = syncOperations_[setSame->GetSyncIndex()];
        SyncOperation *waitSame = syncPair[1].get();
 
        // Check coverage/overlap
        bool canForwardReuse = !setSame->GetForEndIndex().has_value() &&
             (setSync->GetSyncIRIndex() > setSame->GetSyncIRIndex() && 
              setSync->GetSyncIRIndex() <= waitSame->GetSyncIRIndex());
        
        // ... Backward reuse logic ...
 
        if (canForwardReuse /* || canBackwardReuse */) {
            return setSame; // Simplification: return first valid match
        }
    }
  }
  return nullptr;
}
 
// PTO Reserved IDs map
const llvm::DenseMap<std::pair<PipelineType, PipelineType>, uint64_t>
    SyncEventIdAllocation::reservedEventIdNum = {
        {{PipelineType::PIPE_V, PipelineType::PIPE_S}, 1},
        {{PipelineType::PIPE_S, PipelineType::PIPE_V}, 1},
        {{PipelineType::PIPE_MTE2, PipelineType::PIPE_V}, 1},
};
