#include "PTO/Transforms/BlockSyncAnalysis.h"
#include "PTO/Transforms/SyncCommon.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "pto-block-sync-analysis"

using namespace mlir;
using namespace mlir::pto;

namespace mlir::pto {

static constexpr unsigned kPipeStateSize =
    static_cast<unsigned>(PipelineType::PIPE_LAST) + 1U;

static bool isValidPipeIndex(PipelineType pipe) {
  return static_cast<unsigned>(pipe) < kPipeStateSize;
}

// ==============================================================================
// 1. Entry Point
// ==============================================================================

void BlockSyncAnalysis::Run(bool insertBarAllAtLast) {
  syncIndex_ = syncOperations_.size();

  for (auto &nowElement : syncIR_) {
    if (auto *nowCompound =
            dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      DealWithCompoundSync(nowCompound);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(nowElement.get())) {
      DealWithLoopSync(loopElement);
    } else if (isa<BranchInstanceElement>(nowElement.get())) {
      continue;
    } else if (isa<PlaceHolderInstanceElement>(nowElement.get())) {
      continue;
    }
  }

  if (insertBarAllAtLast) {
    InsertLastPipeAll();
  }
}

// ==============================================================================
// 2. High-Level Traversal
// ==============================================================================

void BlockSyncAnalysis::DealWithCompoundSync(CompoundInstanceElement *nowCompound) {
  SyncRecordList syncRecordList;
  InsertSeqSync(nowCompound, syncIR_, 0, nowCompound->GetIndex(), syncRecordList,
                std::nullopt);
}

void BlockSyncAnalysis::DealWithLoopSync(LoopInstanceElement *nowElement) {
  // Insert backward sync by copying the loop body slice and running the same
  // sequential insertion on the copied structure (AscendNPU-IR style).
  if (nowElement->getLoopKind() != KindOfLoop::LOOP_END) {
    return;
  }

  SyncIRs backSyncIr;
  assert(syncIR_.size() >= nowElement->endId);
  for (unsigned i = nowElement->beginId; i < nowElement->endId; i++) {
    if (auto *compound = dyn_cast<CompoundInstanceElement>(syncIR_[i].get())) {
      InsertBackForSync(compound, backSyncIr, nowElement);
    } else if (auto *loopElement =
                   dyn_cast<LoopInstanceElement>(syncIR_[i].get())) {
      auto loopKind = loopElement->getLoopKind();
      backSyncIr.emplace_back(loopElement->CloneFor(loopKind));
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(
          branchElement->CloneBranch(branchElement->getBranchKind()));
    } else if (auto *placeHolderElement =
                   dyn_cast<PlaceHolderInstanceElement>(syncIR_[i].get())) {
      backSyncIr.emplace_back(placeHolderElement->Clone());
    }
  }
}

void BlockSyncAnalysis::InsertBackForSync(CompoundInstanceElement *nowCompound,
                                         SyncIRs &backSyncIr,
                                         const LoopInstanceElement *loopElement) {
  SyncRecordList syncRecordList;

  auto backCompound = std::make_unique<CompoundInstanceElement>(
      nowCompound->GetIndex(), nowCompound->defVec, nowCompound->useVec,
      nowCompound->kPipeValue, nowCompound->opName);
  backCompound->compoundCoreType = nowCompound->compoundCoreType;
  backCompound->elementOp = nowCompound->elementOp;

  auto *backCompoundPtr = backCompound.get();
  backSyncIr.emplace_back(std::move(backCompound));

  // Insert sync between the copied commands (j+1 slice).
  InsertSeqSync(backCompoundPtr, backSyncIr, 0,
                static_cast<int>(backSyncIr.size()) - 1, syncRecordList,
                loopElement->endId);

  // Insert sync between original and copied commands to model loop-carried deps.
  InsertSeqSync(nowCompound, syncIR_, nowCompound->GetIndex(), loopElement->endId,
                syncRecordList, loopElement->endId);
}

// ==============================================================================
// 3. Sequential Sync Insertion (Core Logic)
// ==============================================================================

bool BlockSyncAnalysis::IsNoNeedToInsertSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound, bool isBackwardDep) const {
  const PipelineType frontPipe = frontCompound->kPipeValue;
  const PipelineType nowPipe = nowCompound->kPipeValue;

  if (frontPipe == nowPipe && frontPipe == PipelineType::PIPE_S) {
    return true;
  }

  if (nowCompound->elementOp == frontCompound->elementOp && !isBackwardDep) {
    return true;
  }

  if (frontPipe == nowPipe && !isBackwardDep) {
    // Only insert an intra-pipe barrier for real GM hazards.
    return !IsGMHazard(nowCompound, frontCompound);
  }

  return false;
}

void BlockSyncAnalysis::InsertSeqSync(CompoundInstanceElement *nowCompound,
                                      SyncIRs &syncElement, int begin, int end,
                                      SyncRecordList &syncRecordList,
                                      const std::optional<unsigned> &forEndIndex) {
  const PipelineType nowPipeValue = nowCompound->kPipeValue;

  checkSyncIRIndex(syncElement, begin);
  checkSyncIRIndex(syncElement, end);

  unsigned syncIRIndex = syncElement[end]->GetIndex();
  UpdateAlreadySync(syncIR_[syncIRIndex]->pipeBefore, syncRecordList, nowPipeValue);

  for (int i = end - 1; i >= begin; i--) {
    auto &frontPtr = syncElement[i];
    unsigned frontIndex = frontPtr->GetIndex();
    assert(frontIndex < syncIR_.size());
    assert(syncIR_[frontIndex] != nullptr);

    if (auto *frontCompound =
            dyn_cast<CompoundInstanceElement>(frontPtr.get())) {
      UpdateAlreadySync(syncIR_[frontIndex]->pipeAfter, syncRecordList,
                        nowPipeValue);
      InsertSync(nowCompound, frontCompound, syncRecordList, forEndIndex);
      UpdateAlreadySync(syncIR_[frontIndex]->pipeBefore, syncRecordList,
                        nowPipeValue);
    } else if (auto *loopInstance =
                   dyn_cast<LoopInstanceElement>(frontPtr.get())) {
      int skipLoop = static_cast<int>(InsertLoopSync(
          i, nowCompound, begin, loopInstance, syncElement, syncRecordList,
          forEndIndex));
      i -= skipLoop;
    } else if (auto *branchElement =
                   dyn_cast<BranchInstanceElement>(frontPtr.get())) {
      int skipBranch = static_cast<int>(InsertBranchSync(
          i, nowCompound, begin, branchElement, syncElement, syncRecordList,
          forEndIndex));
      i -= skipBranch;
    }
  }
}

unsigned BlockSyncAnalysis::InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (loopElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncRecordList syncRecordForList = syncRecordList;
    unsigned newBegin =
        std::max(begin, index - (loopElement->endId - loopElement->beginId));
    unsigned newEnd = index;
    InsertSeqSync(nowCompound, syncElement, static_cast<int>(newBegin),
                  static_cast<int>(newEnd), syncRecordForList, forEndIndex);
    // Conservatively assume loop bodies are executed and keep the updated state.
    syncRecordList = std::move(syncRecordForList);
    return (loopElement->endId - loopElement->beginId);
  }
  return 0;
}

unsigned BlockSyncAnalysis::InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList,
    const std::optional<unsigned> &forEndIndex) {
  if (branchElement->getBranchKind() == KindOfBranch::IF_END) {
    SyncRecordList syncRecordIfList = syncRecordList;

    // The indices here are positions in `syncElement` (which may be a slice
    // like backSyncIr), so compute ranges relative to `index`.
    unsigned branchIf =
        index - (branchElement->endId - branchElement->beginId);
    unsigned branchElse =
        index - (branchElement->endId - branchElement->branchId);
    unsigned branchEnd = index;

    InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchIf),
                  static_cast<int>(branchElse), syncRecordIfList, forEndIndex);

    if (branchElement->branchId != branchElement->endId) {
      SyncRecordList syncRecordElseList = syncRecordList;
      InsertSeqSync(nowCompound, syncElement, static_cast<int>(branchElse),
                    static_cast<int>(branchEnd), syncRecordElseList, forEndIndex);
      MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordElseList);
    } else {
      // No else-branch: do not promote `alreadySync`, but keep syncFinder
      // updates from the then-branch.
      for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++)
        syncRecordList[bufferIdx].syncFinder = syncRecordIfList[bufferIdx].syncFinder;
    }
    return (branchElement->endId - branchElement->beginId);
  } else if (branchElement->getBranchKind() == KindOfBranch::ELSE_BEGIN &&
             index != begin) {
    assert(nowCompound->GetIndex() > branchElement->branchId);
    return (branchElement->branchId - branchElement->beginId);
  }
  return 0;
}

void BlockSyncAnalysis::MergeAlreadySync(SyncRecordList &syncRecordList,
                                         const SyncRecordList &syncRecordIfList,
                                         const SyncRecordList &syncRecordElseList) {
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    for (size_t pipeIdx = 0; pipeIdx < kPipeStateSize; pipeIdx++) {
      if (syncRecordIfList[bufferIdx].alreadySync[pipeIdx] &&
          syncRecordElseList[bufferIdx].alreadySync[pipeIdx]) {
        syncRecordList[bufferIdx].alreadySync[pipeIdx] = true;
      }
    }
  }
}

// ==============================================================================
// 4. Dependency Analysis & Operation Insertion
// ==============================================================================

void BlockSyncAnalysis::InsertSync(CompoundInstanceElement *nowCompound,
                                   CompoundInstanceElement *frontCompound,
                                   SyncRecordList &syncRecordList,
                                   const std::optional<unsigned> &forEndIndex) {
  if (IsNoNeedToInsertSync(nowCompound, frontCompound, forEndIndex.has_value())) {
    return;
  }
  MemAnalyze(nowCompound, frontCompound, syncRecordList, forEndIndex);
}

void BlockSyncAnalysis::MemAnalyze(CompoundInstanceElement *nowCompound,
                                   CompoundInstanceElement *frontCompound,
                                   SyncRecordList &syncRecordList,
                                   const std::optional<unsigned> &forEndIndex) {
  if (isAlreadySync(nowCompound, frontCompound, syncRecordList, 0)) {
    return;
  }

  DepBaseMemInfoPairVec depVec;
  if (!IsMemInfoHasDependency(nowCompound, frontCompound, depVec)) {
    return;
  }

  // Intra-pipe dependencies that do not touch GM can be ignored to avoid
  // over-synchronization on local memories.
  if (nowCompound->kPipeValue == frontCompound->kPipeValue) {
    bool touchesGM = false;
    for (auto &pair : depVec) {
      if ((pair.first && pair.first->scope == pto::AddressSpace::GM) ||
          (pair.second && pair.second->scope == pto::AddressSpace::GM)) {
        touchesGM = true;
        break;
      }
    }
    if (!touchesGM) {
      return;
    }
  }

  if (forEndIndex.has_value()) {
    int eventIdNum = GetEventIdNum(depVec);
    for (int i = 1; i < eventIdNum; i++) {
      if (isAlreadySync(nowCompound, frontCompound, syncRecordList,
                        static_cast<unsigned>(i))) {
        return;
      }
    }
  }

  InsertSyncOperation(nowCompound, frontCompound, depVec, forEndIndex);
  UpdateSyncRecordInfo(frontCompound, syncRecordList);
}

bool BlockSyncAnalysis::IsMemInfoHasDependency(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  bool hasDependency = false;
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->defVec,
                                          depBaseMemInfosVec);
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->useVec,
                                          depBaseMemInfosVec);
  hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->defVec,
                                          depBaseMemInfosVec);

  // Special hazard: ACC (L0C) read/read cross-pipe ordering.
  //
  // Some PTO-ISA sequences have semantically "read/read" patterns on ACC, but
  // executing them concurrently across pipelines can trigger device-side issues.
  if (nowCompound->kPipeValue != frontCompound->kPipeValue) {
    DepBaseMemInfoPairVec rrDepVec;
    if (memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->useVec,
                               rrDepVec)) {
      for (auto &pair : rrDepVec) {
        if (!pair.first) continue;
        if (pair.first->scope != pto::AddressSpace::ACC) continue;
        depBaseMemInfosVec.push_back(pair);
        hasDependency = true;
      }
    }
  }

  return hasDependency;
}

void BlockSyncAnalysis::InsertSyncOperation(
    CompoundInstanceElement *nowCompound, CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex) {
  PipelineType nowPipe = nowCompound->kPipeValue;
  PipelineType frontPipe = frontCompound->kPipeValue;

  if (nowPipe == frontPipe) {
    unsigned insertBarrierId = nowCompound->GetIndex();
    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe, syncIndex_,
        insertBarrierId, forEndIndex);
    barrierOp->SetDepSyncIRIndex(frontCompound->GetIndex());
    syncIR_[insertBarrierId]->pipeBefore.push_back(barrierOp.get());
    barrierOp->SetSyncIRIndex(insertBarrierId);

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(newSync));
  } else {
    unsigned insertWaitId = nowCompound->GetIndex();
    unsigned insertSetId = frontCompound->GetIndex();
    auto setOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe, syncIndex_,
        insertSetId, forEndIndex);
    auto waitOp = setOp->GetMatchSync(insertWaitId);

    // Back-edge dependencies may require multi-buffer event IDs.
    if (forEndIndex.has_value()) {
      int eventIdNum = GetEventIdNum(depBaseMemInfosVec);
      setOp->eventIdNum = eventIdNum;
      waitOp->eventIdNum = eventIdNum;
    }

    syncIR_[insertSetId]->pipeAfter.push_back(setOp.get());
    syncIR_[insertWaitId]->pipeBefore.push_back(waitOp.get());

    SmallVector<std::unique_ptr<SyncOperation>> newSync;
    newSync.emplace_back(std::move(setOp));
    newSync.emplace_back(std::move(waitOp));
    syncOperations_.emplace_back(std::move(newSync));
  }

  syncIndex_++;
  assert(syncOperations_.size() == syncIndex_);
}

// ==============================================================================
// 5. Sync Record Maintenance
// ==============================================================================

bool BlockSyncAnalysis::isAlreadySync(CompoundInstanceElement *nowCompound,
                                      CompoundInstanceElement *frontCompound,
                                      SyncRecordList &syncRecordList,
                                      unsigned recordListIndex) {
  (void)nowCompound;
  const PipelineType frontPipe = frontCompound->kPipeValue;
  if (recordListIndex >= syncRecordList.size()) return false;
  if (!isValidPipeIndex(frontPipe)) return false;
  return syncRecordList[recordListIndex]
      .alreadySync[static_cast<unsigned>(frontPipe)];
}

void BlockSyncAnalysis::UpdateAlreadySync(const SyncOps &syncVector,
                                         SyncRecordList &syncRecordList,
                                         const PipelineType nowPipeValue) {
  for (auto *sync : syncVector) {
    for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
      if (bufferIdx == 0 && sync->eventIdNum > 1 &&
          sync->GetForEndIndex().has_value()) {
        continue;
      }
      UpdateSyncRecord(sync, syncRecordList[bufferIdx], nowPipeValue);
    }
  }
}

void BlockSyncAnalysis::UpdateSyncRecord(const SyncOperation *sync,
                                        SyncRecord &syncRecord,
                                        PipelineType nowPipeValue) {
  PipelineType setPipeValue = sync->GetSrcPipe();
  PipelineType waitPipeValue = sync->GetDstPipe();

  // Block-sync mode behaves like a global blocking pipe-s wait.
  if (syncAnalysisMode_ == SyncAnalysisMode::BLOCKSYNC) {
    nowPipeValue = PipelineType::PIPE_S;
    waitPipeValue = PipelineType::PIPE_S;
  }

  if (!isValidPipeIndex(nowPipeValue) || !isValidPipeIndex(waitPipeValue) ||
      !isValidPipeIndex(setPipeValue)) {
    return;
  }

  auto &recordAlready = syncRecord.alreadySync;
  auto &recordFinder = syncRecord.syncFinder;

  bool barrierFinder =
      (nowPipeValue == waitPipeValue) &&
      (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER);
  if (barrierFinder) {
    recordAlready[static_cast<unsigned>(nowPipeValue)] = true;
    return;
  }

  bool canTransitivelyEliminate =
      recordAlready[static_cast<unsigned>(waitPipeValue)] ||
      (nowPipeValue == waitPipeValue);
  if (!canTransitivelyEliminate) return;

  if (recordFinder[sync->GetSyncIndex()] &&
      (sync->GetType() == SyncOperation::TYPE::SET_EVENT ||
       sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_SET)) {
    recordAlready[static_cast<unsigned>(setPipeValue)] = true;
  }

  if (sync->GetType() == SyncOperation::TYPE::WAIT_EVENT ||
      sync->GetType() == SyncOperation::TYPE::SYNC_BLOCK_WAIT) {
    recordFinder[sync->GetSyncIndex()] = true;
  }
}

void BlockSyncAnalysis::UpdateSyncRecordInfo(CompoundInstanceElement *frontCompound,
                                             SyncRecordList &syncRecordList) {
  (void)frontCompound;
  assert(!syncOperations_.empty());
  auto &syncPair = syncOperations_.back();
  assert(!syncPair.empty());

  auto *newSync = syncPair[0].get();
  for (size_t bufferIdx = 0; bufferIdx < syncRecordList.size(); bufferIdx++) {
    if (bufferIdx == 0 && newSync->eventIdNum > 1) {
      continue;
    }
    if (!isValidPipeIndex(newSync->GetSrcPipe())) continue;
    syncRecordList[bufferIdx]
        .alreadySync[static_cast<unsigned>(newSync->GetSrcPipe())] = true;
  }
}

// ==============================================================================
// 6. Final Barrier
// ==============================================================================

void BlockSyncAnalysis::InsertLastPipeAll() {
  for (auto it = syncIR_.rbegin(); it != syncIR_.rend(); ++it) {
    auto *element = it->get();
    if (isa<PlaceHolderInstanceElement>(element)) continue;

    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, PipelineType::PIPE_ALL,
        PipelineType::PIPE_ALL, syncIndex_, element->GetIndex(), std::nullopt);

    SyncOperation *barrierRawPtr = barrierOp.get();
    SmallVector<std::unique_ptr<SyncOperation>> syncGroup;
    syncGroup.emplace_back(std::move(barrierOp));
    syncOperations_.emplace_back(std::move(syncGroup));
    syncIndex_++;

    element->pipeAfter.push_back(barrierRawPtr);
    return;
  }
}

// ==============================================================================
// 7. Helpers
// ==============================================================================

bool BlockSyncAnalysis::IsMemAllocOp(Operation *op) const {
  return isa<memref::AllocOp>(op) || isa<pto::PointerCastOp>(op);
}

SmallVector<Value>
BlockSyncAnalysis::GetMemInfoBuffers(const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  llvm::DenseSet<Value> touchedBuffer;
  SmallVector<Value> result;
  for (auto &pair : depBaseMemInfosVec) {
    if (pair.first && pair.first->rootBuffer)
      touchedBuffer.insert(pair.first->rootBuffer);
    if (pair.second && pair.second->rootBuffer)
      touchedBuffer.insert(pair.second->rootBuffer);
  }
  for (auto v : touchedBuffer)
    result.push_back(v);
  return result;
}

int BlockSyncAnalysis::GetEventIdNum(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  for (const auto &pair : depBaseMemInfosVec) {
    bool isLocalA =
        pair.first && (pair.first->scope == pto::AddressSpace::MAT ||
                       pair.first->scope == pto::AddressSpace::VEC);
    bool isLocalB =
        pair.second && (pair.second->scope == pto::AddressSpace::MAT ||
                        pair.second->scope == pto::AddressSpace::VEC);
    if (isLocalA || isLocalB) return 2;
  }
  return 1;
}

bool BlockSyncAnalysis::IsGMHazard(const CompoundInstanceElement *nowCompound,
                                  const CompoundInstanceElement *frontCompound) const {
  auto hasGM = [](const SmallVector<const BaseMemInfo *> &vec) {
    for (const auto *info : vec) {
      if (info->scope == pto::AddressSpace::GM) return true;
    }
    return false;
  };

  bool frontWritesGM = hasGM(frontCompound->defVec);
  bool frontReadsGM = hasGM(frontCompound->useVec);

  bool nowWritesGM = hasGM(nowCompound->defVec);
  bool nowReadsGM = hasGM(nowCompound->useVec);

  if (frontWritesGM && nowReadsGM) return true;  // RAW
  if (frontReadsGM && nowWritesGM) return true;  // WAR
  if (frontWritesGM && nowWritesGM) return true; // WAW

  // RAR is considered safe for GM in this simplified model.
  return false;
}

} // namespace mlir::pto
