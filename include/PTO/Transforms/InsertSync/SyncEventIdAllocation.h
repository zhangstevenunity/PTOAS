#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCEVENTIDALLOCATION_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCEVENTIDALLOCATION_H
 
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include <cstdint>
 
namespace mlir {
namespace pto {
 
constexpr const uint kTotalEventIdNum = 8;
constexpr const uint kBlockSyncAllCubeEventId = 14;
constexpr const uint kBlockSyncAllVectorEventId = 15;
constexpr const uint kBlockSyncSetWaitEventIdNum = 16;
constexpr const uint kMaxWidenTryNum = 99;
 
/// Event ID 生命周期池
struct EventCyclePool {
  SmallVector<SmallVector<unsigned>> slot;
  explicit EventCyclePool(size_t size = 0) : slot(size) {}
};
 
using SyncCycle = DenseMap<int, EventCyclePool>;
 
class SyncEventIdAllocation {
public:
  SyncEventIdAllocation(SyncIRs &syncIR, SyncOperations &syncOperations)
      : syncIR_(syncIR), syncOperations_(syncOperations) {
    reserveBlockAllEventIds();
  };
 
  ~SyncEventIdAllocation() = default;
 
  /// 分配入口
  void Allocate(uint32_t runNum = 0);
 
private:
  void AllocateEventId(InstanceElement *e);
  size_t GetCompilerAvailableEventIdNum(const SyncOperation *sync);
  void SetEventId(SyncOperation *sync);
 
  SmallVector<bool> GetEventPool(const SyncOperation *sync, size_t eventIdNum);
  int ScopePair(const SyncOperation *s);
  void FindUseEventID(unsigned int begin, unsigned int end,
                      const SyncOperation *s, SmallVector<bool> &eventId);
 
  bool CheckSyncLifeCycleConflict(SmallVector<unsigned int> &syncLifeCycle,
                                  unsigned int begin, unsigned int end,
                                  SmallVector<bool> &eventId, unsigned i) const;
 
  void UpdateEventId(SmallVector<unsigned int> &syncLifeCycle,
                     const unsigned int begin, const unsigned int end,
                     SmallVector<bool> &eventId, const unsigned index) const;
 
  void SetEventPool(const SyncOperation *sync, unsigned eventId);
 
  void UpdateBackwardMatchSync(const SyncOperation *setFlag,
                               const SyncOperation *waitFlag, unsigned eventId);
 
  void SetUseEventID(unsigned int begin, unsigned int end,
                     const SyncOperation *setFlag, unsigned int eventId);
 
  bool ExtendLifecycle(SmallVector<unsigned int> &syncLifeCycle,
                       unsigned int beginNew, unsigned int endNew) const;
 
  // --- Optimization & Reallocation ---
  void WidenEventId(SyncOps syncVector);
  void ReallocatedEventId();
  void ClearReallocatedBackwardMatchSync();
  void clearAllocatedEventId();
  SmallVector<bool> GetEventIdIdleStatus(SyncOperation *sync,
                                         size_t eventIdNum);
  llvm::LogicalResult ChangeNoEventIdSyncToPipeAll();
  void MoveOutBackwardMatchSync(const SyncOperation *reallocatedSync);
  bool TryWidenByOtherSync(const SyncOperation *sync);
  bool tryWidenOnFirstFound();
  SyncOperation *FindWidenSync(const SyncOperation *setSync,
                               const SyncOperation *waitSync);
  void ClearEventId(const SyncOperation *sync);
 
  SmallVector<int>
  GetAvailableEventId(SyncOperation *sync,
                      SmallVector<bool> eventIdLifetimeAvailableStatus,
                      SmallVector<bool> eventIdIdleStatus, size_t eventIdNum);
 
  SmallVector<int>
  UpdateBlockAvailableEventId(SyncOperation *sync,
                              SmallVector<bool> eventIdLifetimeAvailableStatus,
                              size_t eventIdNum);
 
  void SetBlockSyncAllEventID(SyncOperation *sync);
  void IgnoreBackHeadAndTailSync();
  void reserveBlockAllEventIds();
 
private:
  SyncIRs &syncIR_;
  SyncOperations &syncOperations_;
  SyncCycle eventCyclePool;
  llvm::SmallSet<int, 16> reallocatedPipePair;
  llvm::DenseSet<SyncOperation *> insertedBackwardSync;
 
  static const llvm::DenseMap<std::pair<PipelineType, PipelineType>, uint64_t>
      reservedEventIdNum;
  uint64_t reservedBlockSyncEventIdNum{0};
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_SYNCEVENTIDALLOCATION_H
