#include "PTO/Transforms/BlockSyncAnalysis.h"
#include "PTO/Transforms/SyncCommon.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <utility>
#include <memory>
#include <optional>
 
#define DEBUG_TYPE "pto-block-sync-analysis"
 
using namespace mlir;
using namespace mlir::pto;
 
namespace mlir {
namespace pto {
 
static void printMemInfoVec(const char* tag, const SmallVector<const BaseMemInfo *> &vec) {
  llvm::errs() << tag << " (Size " << vec.size() << "):\n";
  for (const auto *info : vec) {
    llvm::errs() << " - Root: ";
    if (info->rootBuffer) {
      if (auto *op = info->rootBuffer.getDefiningOp())
        llvm::errs() << op->getName();
      else
        llvm::errs() << "BlockArg";
      llvm::errs() << " | Scope: " << (int)info->scope << "\n";
    } else {
      llvm::errs() << "NULL\n";
    }
  }
}
 
// ==============================================================================
// 1. Plan / Entry Point
// ==============================================================================
void BlockSyncAnalysis::Run(bool insertBarAllAtLast) {
  syncIndex_ = syncOperations_.size();
  
  for (auto &nowElement : syncIR_) {
    if (auto *nowCompound = dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      DealWithCompoundSync(nowCompound);
    } else if (auto *loopElement = dyn_cast<LoopInstanceElement>(nowElement.get())) {
      DealWithLoopSync(loopElement);
    } else if (isa<BranchInstanceElement>(nowElement.get())) {
      continue;
    }
  }
 
  if (insertBarAllAtLast) {
    InsertLastPipeAll();
  }
}
 
// ==============================================================================
// 2. High-Level Recursion & Traversal
// ==============================================================================
 
void BlockSyncAnalysis::DealWithCompoundSync(CompoundInstanceElement *nowCompound) {
  // [Fix] Initialize vector size
  SyncRecordList syncRecordList(syncIR_.size());
  
  InsertSeqSync(nowCompound, syncIR_, 0, nowCompound->GetIndex(), syncRecordList,
                std::nullopt, nullptr);
}
 
void BlockSyncAnalysis::DealWithLoopSync(LoopInstanceElement *nowElement) {
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncIRs backSyncIr;
    assert(syncIR_.size() >= nowElement->endId);
    
    for (unsigned i = nowElement->beginId; i < nowElement->endId; i++) {
      if (auto *compound = dyn_cast<CompoundInstanceElement>(syncIR_[i].get())) {
         auto backCompound = std::make_unique<CompoundInstanceElement>(
             compound->GetIndex(), compound->defVec, compound->useVec,
             compound->kPipeValue, compound->opName);
         backCompound->elementOp = compound->elementOp;
         backSyncIr.emplace_back(std::move(backCompound));
      } 
    }
 
    SyncRecordList syncRecordList(syncIR_.size());
    unsigned loopEndIndex = nowElement->endId;
    
    for (auto &backElem : backSyncIr) {
      if (auto *backCompound = dyn_cast<CompoundInstanceElement>(backElem.get())) {
        // [Fix] 将最后一个参数 waitAnchor 设为 nullptr
        // 这样 Wait 指令就会自然地附着在循环内部的消费者 (nowCompound) 上
        InsertSeqSync(backCompound, syncIR_, nowElement->beginId, 
                    loopEndIndex, syncRecordList, loopEndIndex, nullptr);
      }
    }
  }
}
 
// ==============================================================================
// 3. Sequential Sync Insertion (Core Logic)
// ==============================================================================
 
bool BlockSyncAnalysis::IsNoNeedToInsertSync(
    const CompoundInstanceElement *nowCompound,
    const CompoundInstanceElement *frontCompound, 
    bool isBackwardDep) const {
 
  const PipelineType frontPipe = frontCompound->kPipeValue;
  const PipelineType nowPipe = nowCompound->kPipeValue;
 
  if (frontPipe == nowPipe && frontPipe == PipelineType::PIPE_S) {
    return true;
  }
 
  if (nowCompound->elementOp == frontCompound->elementOp && !isBackwardDep) {
    return true;
  }
 
  if (frontPipe == nowPipe) {
      if (!isBackwardDep) {
          if (IsGMHazard(nowCompound, frontCompound)) {
              return false;
          }
          return true;
      }
  }
 
  return false;
}
 
void BlockSyncAnalysis::InsertSeqSync(CompoundInstanceElement *nowCompound,
                                      SyncIRs &syncElement, int begin, int end,
                                      SyncRecordList &syncRecordList,
                                      const std::optional<unsigned> &forEndIndex,
                                      InstanceElement *waitAnchor) {
  const PipelineType nowPipeValue = nowCompound->kPipeValue;
  
  if (end >= 0 && static_cast<size_t>(end) < syncElement.size()) {
      unsigned syncIRIndex = syncElement[end]->GetIndex();
      if (syncIRIndex < syncIR_.size()) {
          UpdateAlreadySync(syncIR_[syncIRIndex]->pipeBefore, syncRecordList, nowPipeValue, syncIRIndex);
      }
  }
 
  for (int i = end - 1; i >= begin; i--) {
    auto &frontPtr = syncElement[i];
    unsigned frontIndex = frontPtr->GetIndex();
    
    if (auto *frontCompound = dyn_cast<CompoundInstanceElement>(frontPtr.get())) {
        UpdateAlreadySync(syncIR_[frontIndex]->pipeAfter, syncRecordList, nowPipeValue, frontIndex);
        
        InsertSync(nowCompound, frontCompound, syncRecordList, forEndIndex, waitAnchor);
        
        UpdateAlreadySync(syncIR_[frontIndex]->pipeBefore, syncRecordList, nowPipeValue, frontIndex);
        
    } else if (auto *loopInstance = dyn_cast<LoopInstanceElement>(frontPtr.get())) {
        int skipLoop = static_cast<int>(InsertLoopSync(i, nowCompound, begin, loopInstance,
                                        syncElement, syncRecordList, forEndIndex, waitAnchor));
        i -= skipLoop;
    } else if (auto *branchElement = dyn_cast<BranchInstanceElement>(frontPtr.get())) {
        int skipBranch = static_cast<int>(InsertBranchSync(i, nowCompound, begin, 
                                          branchElement, syncElement,
                                          syncRecordList, forEndIndex, waitAnchor));
        i -= skipBranch;
    }
  }
}
 
unsigned BlockSyncAnalysis::InsertLoopSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    LoopInstanceElement *loopElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList, const std::optional<unsigned> &forEndIndex,
    InstanceElement *waitAnchor) { 
  if (loopElement->getLoopKind() == KindOfLoop::LOOP_END) {
    SyncRecordList syncRecordForList = syncRecordList;
    
    unsigned loopLen = loopElement->endId - loopElement->beginId;
    unsigned newBegin = std::max((int)begin, (int)(index - loopLen));
    unsigned newEnd = index; 
    
    InsertSeqSync(nowCompound, syncElement, newBegin, newEnd, syncRecordForList, forEndIndex, waitAnchor);
    
    syncRecordList = std::move(syncRecordForList);
    
    return loopLen; 
  }
  return 0;
}
 
// ==============================================================================
// 4. Dependency Analysis & Operation Insertion
// ==============================================================================
 
void BlockSyncAnalysis::InsertSync(CompoundInstanceElement *nowCompound,
                                   CompoundInstanceElement *frontCompound,
                                   SyncRecordList &syncRecordList,
                                   const std::optional<unsigned> &forEndIndex,
                                   InstanceElement *waitAnchor) {
    if (IsNoNeedToInsertSync(nowCompound, frontCompound, forEndIndex.has_value())) {
        return;
    }
    MemAnalyze(nowCompound, frontCompound, syncRecordList, forEndIndex, waitAnchor);
}
 
void BlockSyncAnalysis::MemAnalyze(CompoundInstanceElement *nowCompound,
                                   CompoundInstanceElement *frontCompound,
                                   SyncRecordList &syncRecordList,
                                   const std::optional<unsigned> &forEndIndex,
                                   InstanceElement *waitAnchor) {
    // ... [Debug Prints Keep Same] ...
    llvm::errs() << "\n[MemAnalyze] Analyzing Dependency:\n";
    llvm::errs() << "  Now Node: " << nowCompound->elementOp->getName() << "\n";
    printMemInfoVec("    DefVec", nowCompound->defVec);
    printMemInfoVec("    UseVec", nowCompound->useVec);
    llvm::errs() << "  Front Node: " << frontCompound->elementOp->getName() << "\n";
    printMemInfoVec("    DefVec", frontCompound->defVec);
    printMemInfoVec("    UseVec", frontCompound->useVec);
 
    if (isAlreadySync(nowCompound, frontCompound, syncRecordList, frontCompound->GetIndex())) {
        llvm::errs() << "  -> Already Synced at Pipe Level. Skipping.\n";
        return;
    }
 
    DepBaseMemInfoPairVec depVec;
    bool hasDep = IsMemInfoHasDependency(nowCompound, frontCompound, depVec);
    
    if (hasDep) {
        llvm::errs() << "  -> Dependency FOUND!\n";
    } else {
        llvm::errs() << "  -> No Dependency.\n";
        return; // No dep -> return
    }
    
    // [Fix] Intra-Pipe Barrier 过滤逻辑
    if (nowCompound->kPipeValue == frontCompound->kPipeValue) {
        bool isRealGMHazard = false;
        for (auto &pair : depVec) {
            // 只要有一对依赖涉及 GM，就认为是真正的 Hazard
            if ((pair.first && pair.first->scope == pto::AddressSpace::GM) ||
                (pair.second && pair.second->scope == pto::AddressSpace::GM)) {
                isRealGMHazard = true;
                break;
            }
        }
        if (!isRealGMHazard) {
             llvm::errs() << "  -> Intra-Pipe dep on Local Mem (No GM). Skipping Barrier.\n";
             return; 
        }
    }
    
    if (forEndIndex.has_value()) {
        int eventIdNum = GetEventIdNum(depVec);
        for (int i = 1; i < eventIdNum; i++) {
             if (isAlreadySync(nowCompound, frontCompound, syncRecordList, frontCompound->GetIndex())) {
                 return;
             }
        }
    }
    
    InsertSyncOperation(nowCompound, frontCompound, depVec, forEndIndex, waitAnchor);
    UpdateSyncRecordInfo(frontCompound, syncRecordList);
}
 
bool BlockSyncAnalysis::IsMemInfoHasDependency(
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
    
    bool hasDependency = false;
    hasDependency |= memAnalyzer_.DepBetween(nowCompound->useVec, frontCompound->defVec, depBaseMemInfosVec);
    hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->useVec, depBaseMemInfosVec);
    hasDependency |= memAnalyzer_.DepBetween(nowCompound->defVec, frontCompound->defVec, depBaseMemInfosVec);

    // ---------------------------------------------------------------------
    // Special hazard: ACC (L0C) read/read cross-pipe ordering.
    //
    // Some PTO-ISA sequences (e.g. TMOV_FP reading ACC on PIPE_MTE1 and TSTORE
    // reading ACC on PIPE_MTE3) are semantically "read/read", but executing
    // them concurrently can trigger nondeterminism or device exceptions on NPU.
    //
    // Model this as a dependency to force an explicit event ordering between
    // different pipelines when they alias on ACC.
    // ---------------------------------------------------------------------
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
    CompoundInstanceElement *nowCompound,
    CompoundInstanceElement *frontCompound,
    DepBaseMemInfoPairVec &depBaseMemInfosVec,
    const std::optional<unsigned> &forEndIndex,
    InstanceElement *waitAnchor) {
    
    PipelineType nowPipe = nowCompound->kPipeValue;
    PipelineType frontPipe = frontCompound->kPipeValue;
    
    if (nowPipe == frontPipe) {
        unsigned insertId = nowCompound->GetIndex();
        
        auto barrier = std::make_unique<SyncOperation>(
            SyncOperation::TYPE::PIPE_BARRIER, frontPipe, nowPipe,
            syncIndex_++, nowCompound->GetIndex(), forEndIndex);
        
        llvm::errs() << " [Trace Insert] Intra-Pipe Barrier at Node " << insertId << " (PRE)\n";
        syncIR_[insertId]->pipeBefore.push_back(barrier.get());
        
        SmallVector<std::unique_ptr<SyncOperation>> newSync;
        newSync.emplace_back(std::move(barrier));
        syncOperations_.emplace_back(std::move(newSync));
        
    } else {
        unsigned insertSetId = frontCompound->GetIndex();
        unsigned insertWaitId = nowCompound->GetIndex();
        
        bool useAnchor = (waitAnchor != nullptr);
 
        auto setOp = std::make_unique<SyncOperation>(
            SyncOperation::TYPE::SET_EVENT, frontPipe, nowPipe,
            syncIndex_, insertSetId, forEndIndex);
            
        auto waitOp = setOp->GetMatchSync(insertWaitId); 
        
        llvm::errs() << " [Trace Insert] SET_EVENT at Node " << insertSetId << " (POST)\n";
        syncIR_[insertSetId]->pipeAfter.push_back(setOp.get());
        
        if (useAnchor) {
            llvm::errs() << " [Trace Insert] WAIT_EVENT at Anchor Node " << waitAnchor->GetIndex() << " (PRE)\n";
            waitAnchor->pipeBefore.push_back(waitOp.get());
        } else {
            llvm::errs() << " [Trace Insert] WAIT_EVENT at Node " << insertWaitId << " (PRE)\n";
            syncIR_[insertWaitId]->pipeBefore.push_back(waitOp.get());
        }
        
        SmallVector<std::unique_ptr<SyncOperation>> newSync;
        newSync.emplace_back(std::move(setOp));
        newSync.emplace_back(std::move(waitOp));
        syncOperations_.emplace_back(std::move(newSync));
        
        syncIndex_++;
    }
}
 
// ==============================================================================
// 5. Utility & Record Management
// ==============================================================================
 
bool BlockSyncAnalysis::isAlreadySync(CompoundInstanceElement *nowCompound,
                                      CompoundInstanceElement *frontCompound,
                                      SyncRecordList &syncRecordList,
                                      unsigned recordListIndex) {
    PipelineType frontPipe = frontCompound->kPipeValue;
    if (recordListIndex >= syncRecordList.size()) return false; 
    return syncRecordList[recordListIndex].alreadySync[static_cast<unsigned>(frontPipe)] != nullptr;
}
 
void BlockSyncAnalysis::UpdateAlreadySync(const SyncOps &syncVector,
                                          SyncRecordList &syncRecordList,
                                          const PipelineType nowPipeValue,
                                          unsigned index) {
    if (index >= syncRecordList.size()) return;
 
    for (auto &sync : syncVector) {
        UpdateSyncRecord(sync, syncRecordList[index], nowPipeValue);
    }
}
 
void BlockSyncAnalysis::UpdateSyncRecord(const SyncOperation *sync,
                                         SyncRecord &syncRecord,
                                         PipelineType nowPipeValue) {
    PipelineType setPipe = sync->GetSrcPipe();
    PipelineType waitPipe = sync->GetDstPipe();
    
    bool isBarrier = (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER);
    
    if (isBarrier) {
        syncRecord.alreadySync[static_cast<unsigned>(nowPipeValue)] = sync;
    } else if (syncRecord.alreadySync[static_cast<unsigned>(waitPipe)] || 
               nowPipeValue == waitPipe) {
        if (sync->GetType() == SyncOperation::TYPE::SET_EVENT) {
             syncRecord.alreadySync[static_cast<unsigned>(setPipe)] = sync;
        }
    }
}
 
void BlockSyncAnalysis::UpdateSyncRecordInfo(CompoundInstanceElement *frontCompound,
                                             SyncRecordList &syncRecordList) {
    assert(!syncOperations_.empty());
    auto &lastSyncPair = syncOperations_.back(); 
    auto *setOp = lastSyncPair[0].get();
    
    unsigned idx = frontCompound->GetIndex();
    // [Trace]
    llvm::errs() << " [Trace Record] Updating Record for Node " << idx 
                 << ", SetPipe=" << (int)setOp->GetSrcPipe() << "\n";
 
    if (idx < syncRecordList.size()) {
        syncRecordList[idx].alreadySync[static_cast<unsigned>(setOp->GetSrcPipe())] = setOp;
    }
}
 
void BlockSyncAnalysis::InsertLastPipeAll() {
  llvm::errs() << "\n[InsertLastPipeAll] Scan backwards for injection point...\n";
  for (auto it = syncIR_.rbegin(); it != syncIR_.rend(); ++it) {
    auto *element = it->get();
    
    // [Debug] Print Element Type
    llvm::errs() << "  Scanning Node " << element->GetIndex() << " Kind: ";
    switch(element->GetKind()) {
        case InstanceElement::KindTy::COMPOUND: llvm::errs() << "COMPOUND\n"; break;
        case InstanceElement::KindTy::LOOP: llvm::errs() << "LOOP\n"; break;
        case InstanceElement::KindTy::BRANCH: llvm::errs() << "BRANCH\n"; break;
        case InstanceElement::KindTy::PLACE_HOLDER: llvm::errs() << "PLACE_HOLDER\n"; break;
        default: llvm::errs() << "UNKNOWN\n"; break;
    }
 
    if (isa<PlaceHolderInstanceElement>(element)) continue;
 
    auto barrierOp = std::make_unique<SyncOperation>(
        SyncOperation::TYPE::PIPE_BARRIER, 
        PipelineType::PIPE_ALL, 
        PipelineType::PIPE_ALL, 
        syncOperations_.size(), 0, std::nullopt);
 
    SyncOperation* barrierRawPtr = barrierOp.get();
    SmallVector<std::unique_ptr<SyncOperation>> syncGroup;
    syncGroup.push_back(std::move(barrierOp));
    syncOperations_.push_back(std::move(syncGroup));
 
    if (auto *compound = dyn_cast<CompoundInstanceElement>(element)) {
      llvm::errs() << "  [Trace Insert] Final Barrier at Node " << compound->GetIndex() << " (POST)\n";
      compound->pipeAfter.push_back(barrierRawPtr);
      return; 
    } 
    else if (auto *loop = dyn_cast<LoopInstanceElement>(element)) {
      llvm::errs() << "  [Trace Insert] Final Barrier at Node " << loop->GetIndex() << " (POST)\n";
      loop->pipeAfter.push_back(barrierRawPtr);
      return; 
    }
    else if (auto *branch = dyn_cast<BranchInstanceElement>(element)) {
      // 只有 IF_END 才适合挂载 Barrier
      if (branch->getBranchKind() == KindOfBranch::IF_END) {
        llvm::errs() << "  [Trace Insert] Final Barrier at Node " << branch->GetIndex() << " (POST)\n";
        branch->pipeAfter.push_back(barrierRawPtr);
        return; 
      }
    }
  }
  llvm::errs() << "  [Warning] No valid insertion point found for Final Barrier.\n";
}
 
// ==============================================================================
// 6. Branch Handling Logic
// ==============================================================================
 
unsigned BlockSyncAnalysis::InsertBranchSync(
    unsigned index, CompoundInstanceElement *nowCompound, unsigned begin,
    BranchInstanceElement *branchElement, SyncIRs &syncElement,
    SyncRecordList &syncRecordList, const std::optional<unsigned> &forEndIndex,
    InstanceElement *waitAnchor) { 
  if (branchElement->getBranchKind() == KindOfBranch::IF_END) {
    SyncRecordList syncRecordIfList = syncRecordList;   
    SyncRecordList syncRecordElseList = syncRecordList; 
 
    unsigned ifStart = branchElement->beginId;
    unsigned ifEnd   = branchElement->branchId; 
    unsigned elseStart = branchElement->branchId;
    unsigned elseEnd   = branchElement->endId;   
 
    InstanceElement* ifAnchor = nullptr;
    if (branchElement->branchId > 0) {
        ifAnchor = syncElement[branchElement->branchId - 1].get();
    }
 
    InstanceElement* elseAnchor = nullptr;
    if (branchElement->branchId < syncElement.size()) {
        if (auto *elseBeginElem = dyn_cast<BranchInstanceElement>(syncElement[branchElement->branchId].get())) {
            if (elseBeginElem->endId > 0) {
                elseAnchor = syncElement[elseBeginElem->endId - 1].get();
            }
        }
    }
    
    // [Trace]
    llvm::errs() << "\n [Trace Branch] IF Analysis. IfStart=" << ifStart << ", IfEnd=" << ifEnd 
                 << ", ElseStart=" << elseStart << ", ElseEnd=" << elseEnd << "\n";
    llvm::errs() << "   IfAnchor=" << (ifAnchor ? ifAnchor->GetIndex() : -1) 
                 << ", ElseAnchor=" << (elseAnchor ? elseAnchor->GetIndex() : -1) << "\n";
 
    InsertSeqSync(nowCompound, syncElement, ifStart + 1, ifEnd, 
                  syncRecordIfList, forEndIndex, nullptr);

    // [Fix] 传递 waitAnchor 给 If 分支
    InsertSeqSync(nowCompound, syncElement, ifStart + 1, ifEnd, 
                  syncRecordIfList, forEndIndex, waitAnchor);

    if (branchElement->branchId != branchElement->endId) {
      InsertSeqSync(nowCompound, syncElement, elseStart + 1, elseEnd, 
                    syncRecordElseList, forEndIndex, waitAnchor);
      
      MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordElseList, 
                       ifAnchor, elseAnchor, forEndIndex);
      
    } else {
        MergeAlreadySync(syncRecordList, syncRecordIfList, syncRecordList, 
                         ifAnchor, elseAnchor, forEndIndex);
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
                                         const SyncRecordList &syncRecordElseList,
                                         InstanceElement* ifAnchor,
                                         InstanceElement* elseAnchor,
                                         const std::optional<unsigned> &forEndIndex) {
  
  size_t maxPipe = getPipeNum();
  
  // [Trace]
  llvm::errs() << " [Trace Merge] Merging branch results...\n";
  
  for (size_t i = 0; i < syncRecordList.size(); i++) {
    if (syncRecordList[i].alreadySync.size() <= maxPipe) {
        syncRecordList[i].alreadySync.resize(maxPipe + 1, nullptr);
    }
    
    for (size_t j = 0; j < maxPipe; j++) {
      if (j >= syncRecordIfList[i].alreadySync.size() || 
          j >= syncRecordElseList[i].alreadySync.size()) {
          continue; 
      }
 
      const SyncOperation* syncIf = syncRecordIfList[i].alreadySync[j];
      const SyncOperation* syncElse = syncRecordElseList[i].alreadySync[j];
      
      if (syncIf && syncElse) {
        // Both synced
        syncRecordList[i].alreadySync[j] = syncIf;
      } 
      else if (syncIf && !syncElse) {
        llvm::errs() << "   Compensating Else: Node " << i << " Pipe " << j << " -> Set at ElseAnchor\n";
        if (elseAnchor) {
            unsigned targetSyncIndex = syncIf->GetSyncIndex();
            auto phantomSet = std::make_unique<SyncOperation>(
                SyncOperation::TYPE::SET_EVENT, 
                syncIf->GetSrcPipe(), syncIf->GetDstPipe(),
                targetSyncIndex, 
                elseAnchor->GetIndex(), forEndIndex, true);
            
            SyncOperation* rawPtr = phantomSet.get();
            if (targetSyncIndex < syncOperations_.size()) {
                syncOperations_[targetSyncIndex].push_back(std::move(phantomSet));
            } else {
                SmallVector<std::unique_ptr<SyncOperation>> newSync;
                newSync.emplace_back(std::move(phantomSet));
                syncOperations_.emplace_back(std::move(newSync));
            }
            // [Trace]
            llvm::errs() << "   [Trace Insert] Phantom SET at Node " << elseAnchor->GetIndex() << " (PRE)\n";
            elseAnchor->pipeBefore.push_back(rawPtr);
            syncRecordList[i].alreadySync[j] = rawPtr;
        } else {
            syncRecordList[i].alreadySync[j] = nullptr;
        }
      } 
      else if (!syncIf && syncElse) {
        llvm::errs() << "   Compensating Then: Node " << i << " Pipe " << j << " -> Set at IfAnchor\n";
        if (ifAnchor) {
            unsigned targetSyncIndex = syncElse->GetSyncIndex();
            auto phantomSet = std::make_unique<SyncOperation>(
                SyncOperation::TYPE::SET_EVENT, 
                syncElse->GetSrcPipe(), syncElse->GetDstPipe(),
                targetSyncIndex, 
                ifAnchor->GetIndex(), forEndIndex, true);
            
            SyncOperation* rawPtr = phantomSet.get();
            if (targetSyncIndex < syncOperations_.size()) {
                syncOperations_[targetSyncIndex].push_back(std::move(phantomSet));
            } else {
                SmallVector<std::unique_ptr<SyncOperation>> newSync;
                newSync.emplace_back(std::move(phantomSet));
                syncOperations_.emplace_back(std::move(newSync));
            }
            // [Trace]
            llvm::errs() << "   [Trace Insert] Phantom SET at Node " << ifAnchor->GetIndex() << " (PRE)\n";
            ifAnchor->pipeBefore.push_back(rawPtr);
            syncRecordList[i].alreadySync[j] = rawPtr;
        } else {
            syncRecordList[i].alreadySync[j] = nullptr;
        }
      } 
      else {
        syncRecordList[i].alreadySync[j] = nullptr;
      }
    }
  }
}
 
// ... Helpers (IsMemAllocOp, GetMemInfoBuffers, GetEventIdNum, IsGMHazard) 保持不变 ...
bool BlockSyncAnalysis::IsMemAllocOp(Operation *op) const {
    return isa<memref::AllocOp>(op) || isa<pto::PointerCastOp>(op); 
}
 
SmallVector<Value> BlockSyncAnalysis::GetMemInfoBuffers(
    const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
    llvm::DenseSet<Value> touchedBuffer;
    SmallVector<Value> result;
    for (auto &pair : depBaseMemInfosVec) {
        if (pair.first && pair.first->rootBuffer) touchedBuffer.insert(pair.first->rootBuffer);
        if (pair.second && pair.second->rootBuffer) touchedBuffer.insert(pair.second->rootBuffer);
    }
    for (auto v : touchedBuffer) result.push_back(v);
    return result;
}
 
int BlockSyncAnalysis::GetEventIdNum(const DepBaseMemInfoPairVec &depBaseMemInfosVec) {
    for (const auto &pair : depBaseMemInfosVec) {
        // 逻辑含义：只要涉及 Matrix Buffer (MAT) 或者 Vector Buffer (UB)，都视为片上依赖，可能需要 Double Buffer。
        bool isLocalA = pair.first && (pair.first->scope == pto::AddressSpace::MAT || pair.first->scope == pto::AddressSpace::VEC);
        bool isLocalB = pair.second && (pair.second->scope == pto::AddressSpace::MAT || pair.second->scope == pto::AddressSpace::VEC);
        if (isLocalA || isLocalB) return 2;
    }
    return 1; 
}
 
bool BlockSyncAnalysis::IsGMHazard(const CompoundInstanceElement *nowCompound,
                                   const CompoundInstanceElement *frontCompound) const {
  // Helper: 检查集合中是否包含 GM 资源
  auto hasGM = [](const SmallVector<const BaseMemInfo *> &vec) {
    for (const auto *info : vec) {
        if (info->scope == pto::AddressSpace::GM) return true;
    }
    return false;
  };

  // 1. 获取前驱节点 (Front) 的 GM 读写状态
  bool frontWritesGM = hasGM(frontCompound->defVec); // Def = Write
  bool frontReadsGM  = hasGM(frontCompound->useVec); // Use = Read

  // 2. 获取当前节点 (Now) 的 GM 读写状态
  bool nowWritesGM = hasGM(nowCompound->defVec);
  bool nowReadsGM  = hasGM(nowCompound->useVec);

  // 3. 判定 Hazard (只要命中以下任意一种，就需要 Barrier)
  
  // Case A: RAW (Read After Write) - 前面写，后面读
  if (frontWritesGM && nowReadsGM) return true;

  // Case B: WAR (Write After Read) - 前面读，后面写
  if (frontReadsGM && nowWritesGM) return true;

  // Case C: WAW (Write After Write) - 前面写，后面也写
  if (frontWritesGM && nowWritesGM) return true;

  // Case D: RAR (Read After Read) - 前面读，后面也读
  // 这是唯一的安全情况！也是本次优化的目标。
  // explicitly: if (frontReadsGM && nowReadsGM) return false;

  return false;
}
 
} // namespace pto
} // namespace mlir
