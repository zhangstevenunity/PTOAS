#ifndef MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H
#define MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H
 
#include <utility>
#include <deque>
#include <string>
#include <map>
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>
 
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
 
// 引入 PTO Dialect 定义 (获取 AddressSpace 等)
#include "PTO/IR/PTO.h"
 
#define MAX_MULTI_BUFFER_NUM 16
 
namespace mlir {
namespace pto {
 
enum class SyncAnalysisMode {
  NORMALSYNC, // 核内同步 (Intra-Core): 解决流水线冒险
  BLOCKSYNC   // 核间同步 (Inter-Core): 解决 CV 分离后的通讯
};
 
/// Pipeline 类型定义 (对应硬件流水线)
/// 必须与 PTOIRTranslator 中的 getOpPipeline 保持一致
enum class PipelineType : uint32_t {
  PIPE_S = 0,
  PIPE_V = 1,
  PIPE_M = 2,
  PIPE_MTE1 = 3,   // L1 -> L0
  PIPE_MTE2 = 4,   // GM -> L1 (Load)
  PIPE_MTE3 = 5,   // L1 -> GM (Store)
  PIPE_ALL = 6,    // Barrier
  
  // 预留/扩展
  PIPE_MTE4 = 7,
  PIPE_MTE5 = 8,
  PIPE_V2 = 9,
  PIPE_FIX = 10,
  
  // 虚拟 Pipe (如果有)
  VIRTUAL_PIPE_MTE2_L1A = 11,
  VIRTUAL_PIPE_MTE2_L1B = 12,
  
  PIPE_NUM = 13,
  PIPE_UNASSIGNED = 99,
  PIPE_LAST = PIPE_UNASSIGNED
};
 
// 辅助函数：获取 Pipeline 数量
static constexpr unsigned getPipeNum() {
  return static_cast<unsigned>(PipelineType::PIPE_LAST);
}
 
/// 核心类型定义：对应计算单元类型
enum class TCoreType {
  VECTOR,
  CUBE,
  CUBE_OR_VECTOR,
  CUBE_AND_VECTOR
};
 
/// Meminfo of the target buffer
/// 用于追踪 Buffer 的别名和根节点
struct BaseMemInfo {
  BaseMemInfo(
      Value baseBuffer, Value rootBuffer, pto::AddressSpace scope,
      SmallVector<uint64_t> baseAddresses, uint64_t allocateSize)
      : baseBuffer(baseBuffer), rootBuffer(rootBuffer), scope(scope),
        baseAddresses(std::move(baseAddresses)), allocateSize(allocateSize) {}
 
  /// baseBuffer: 当前操作直接使用的 Buffer (可能是 View 或 Alias)
  Value baseBuffer;
  /// rootBuffer: 最底层的分配源头 (alloc_tile 或 Kernel Argument)
  Value rootBuffer;
  
  pto::AddressSpace scope;
  SmallVector<uint64_t> baseAddresses; // 用于 Offset 分析
  uint64_t allocateSize;
 
  bool areVectorEqual(const SmallVector<uint64_t>& vec1,
                      const SmallVector<uint64_t>& vec2) const {
    if (vec1.size() != vec2.size()) return false;
    for (size_t i = 0; i < vec1.size(); ++i) {
      if (vec1[i] != vec2[i]) return false;
    }
    return true;
  }
 
  bool operator==(const BaseMemInfo &other) const {
    if (!areVectorEqual(baseAddresses, other.baseAddresses)) return false;
    if (rootBuffer != other.rootBuffer) return false;
    if (scope != other.scope) return false;
    // allocateSize 和 baseBuffer 的严格相等性在某些别名分析中可能太强了，
    // 但为了保持原有逻辑，先保留。重点是 rootBuffer 必须一致。
    if (allocateSize != other.allocateSize) return false;
    if (baseBuffer != other.baseBuffer) return false;
    return true;
  }
 
  std::unique_ptr<BaseMemInfo> clone() const {
    return std::make_unique<BaseMemInfo>(
        baseBuffer, rootBuffer, scope, baseAddresses, allocateSize);
  }
 
  std::unique_ptr<BaseMemInfo> clone(Value cloneBaseBuffer) const {
    return std::make_unique<BaseMemInfo>(
        cloneBaseBuffer, rootBuffer, scope, baseAddresses, allocateSize);
  }
};
 
using DepBaseMemInfoPairVec =
    SmallVector<std::pair<const BaseMemInfo *, const BaseMemInfo *>>;
 
// 表示一个具体的同步指令 (Set, Wait, Barrier)
class SyncOperation {
public:
  enum class TYPE {
    SET_EVENT,
    WAIT_EVENT,
    PIPE_BARRIER,
    PIPE_BARRIER_CUBE,
    PIPE_BARRIER_VECTOR,
    SYNC_BLOCK_SET,
    SYNC_BLOCK_WAIT,
    SYNC_BLOCK_ALL,
  };
  
  bool isCompensation = false;

  static const int kNullEventId{-1};
 
public:
  SmallVector<int> eventIds;
  bool uselessSync{false};
  int eventIdNum{1};
  Value lowestCommonAncestorBuffer{nullptr};
  int reuseCntForWiden{0};
  bool reallocatedLoopHeadTailSync{false};
  TCoreType syncCoreType{TCoreType::CUBE_OR_VECTOR};
  Value block_sync_event_value{nullptr};
 
public:
  SyncOperation(TYPE type, pto::PipelineType srcPipe, pto::PipelineType dstPipe,
                unsigned kSyncIndex, unsigned syncIRIndex,
                std::optional<int> forEndIndex, bool isComp = false)
      : eventIds({}), type_(type), srcPipe_(srcPipe), dstPipe_(dstPipe),
        kSyncIndex_(kSyncIndex), syncIRIndex_(syncIRIndex),
        forEndIndex_(forEndIndex), isCompensation(isComp) {};
 
  ~SyncOperation() = default;
 
  std::unique_ptr<SyncOperation> GetMatchSync(unsigned index) const;
  
  TYPE GetType() const { return type_; }
  pto::PipelineType GetSrcPipe() const { return srcPipe_; }
  pto::PipelineType GetDstPipe() const { return dstPipe_; }
  
  // PTO 暂时不需要 Virtual Pipe 映射，直接返回原值
  pto::PipelineType GetActualSrcPipe() const { return srcPipe_; }
  pto::PipelineType GetActualDstPipe() const { return dstPipe_; }
 
  SmallVector<int> GetEventIDs() const { return eventIds; }
  unsigned GetSyncIndex() const { return kSyncIndex_; }
  unsigned GetSyncIRIndex() const { return syncIRIndex_; }
  void SetSyncIRIndex(unsigned index) { syncIRIndex_ = index; }
  std::optional<int> GetForEndIndex() const { return forEndIndex_; }
  void SetDepSyncIRIndex(unsigned index) { depSyncIRIndex_ = index; }
  unsigned GetDepSyncIRIndex() const { return depSyncIRIndex_; }
  void SetType(TYPE syncType) { type_ = syncType; }
  bool operator==(const SyncOperation &other) const;
 
  bool isSyncSetType() const;
  bool isSyncWaitType() const;
  bool isBarrierType() const;
 
  static std::string TypeName(TYPE t);
  std::string GetCoreTypeName(TCoreType t) const;
 
  using SyncOperations =
      SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;
 
  // 设置为 PipeAll (用于资源耗尽时的降级)
  void SetPipeAll();
 
private:
  TYPE type_;
  pto::PipelineType srcPipe_;
  pto::PipelineType dstPipe_;
  const unsigned kSyncIndex_;
  unsigned syncIRIndex_;
  std::optional<int> forEndIndex_{};
  unsigned depSyncIRIndex_{0};
};
 
using SyncOps = std::deque<SyncOperation *>;
 
// SyncIR 的基类节点
class InstanceElement {
public:
  Operation *elementOp = nullptr;
  SyncOps pipeBefore;
  SyncOps pipeAfter;
  enum class KindTy { COMPOUND, LOOP, BRANCH, PLACE_HOLDER };
 
public:
  virtual ~InstanceElement() = default;
  unsigned GetIndex() const { return kIndex; }
  KindTy GetKind() const { return kKindTy; }
  static bool RemoveSync(SyncOps &syncVector, const SyncOperation *sync);
 
protected:
  const unsigned kIndex;
  InstanceElement(KindTy kind, unsigned index)
      : kIndex(index), kKindTy(kind) {};
 
private:
  const KindTy kKindTy;
};
 
class PlaceHolderInstanceElement : public InstanceElement {
public:
  unsigned parentScopeId;
  PlaceHolderInstanceElement(unsigned index, unsigned parentScopeId)
      : InstanceElement(KindTy::PLACE_HOLDER, index),
        parentScopeId(parentScopeId) {};
 
  std::unique_ptr<PlaceHolderInstanceElement> Clone() const;
  static bool classof(const InstanceElement *e);
 
  // [新增] 标记这是一个为了同步而虚拟出来的 Else 块
  bool isVirtualElse = false; 
  // [新增] 保存父 Op (scf.if)，以便后续操作
  Operation* parentIfOp = nullptr;
};
 
enum class KindOfLoop { LOOP_BEGIN, LOOP_END };
 
class LoopInstanceElement : public InstanceElement {
public:
  unsigned beginId;
  unsigned endId;
 
public:
  LoopInstanceElement(unsigned index, unsigned beginId, unsigned endId,
                      KindOfLoop loopKind = KindOfLoop::LOOP_BEGIN)
      : InstanceElement(KindTy::LOOP, index), beginId(beginId), endId(endId),
        kLoopKind(loopKind) {}
 
  ~LoopInstanceElement() override = default;
  std::unique_ptr<InstanceElement> CloneFor(KindOfLoop loopKind) const;
  KindOfLoop getLoopKind() const { return kLoopKind; }
  static bool classof(const InstanceElement *e);
  bool ignore_block_sync_move_out{false};
 
private:
  const KindOfLoop kLoopKind;
};
 
enum class KindOfBranch { IF_BEGIN, ELSE_BEGIN, IF_END };
 
class BranchInstanceElement : public InstanceElement {
public:
  unsigned beginId;
  unsigned branchId;
  unsigned endId{0};
 
public:
  BranchInstanceElement(unsigned index, unsigned beginId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(beginId), kBranchKind(branchKind) {}
 
  ~BranchInstanceElement() override = default;
  std::unique_ptr<BranchInstanceElement>
  CloneBranch(KindOfBranch branchKind) const;
 
  KindOfBranch getBranchKind() const { return kBranchKind; }
  static bool classof(const InstanceElement *e);
 
  BranchInstanceElement(unsigned index, unsigned beginId, unsigned branchId,
                        unsigned endId,
                        KindOfBranch branchKind = KindOfBranch::IF_BEGIN)
      : InstanceElement(KindTy::BRANCH, index), beginId(beginId),
        branchId(branchId), endId(endId), kBranchKind(branchKind) {}
 
private:
  const KindOfBranch kBranchKind;
};
 
// Unit Flag 状态 (用于指令级同步优化)
enum class UNIT_FLAG {
  DISABLED,
  ENABLED_WITH_UPDATE,
  ENABLED_ONLY_FIRST_ITER,
  ENABLED_ONLY_LAST_ITER,
  ENABLED_ONLY_FIRST_AND_LAST_ITERS
};
 
// 核心节点：代表一个计算或搬运指令
class CompoundInstanceElement : public InstanceElement {
public:
  // Def/Use 列表 (指向 BaseMemInfo)
  SmallVector<const BaseMemInfo *> defVec;
  SmallVector<const BaseMemInfo *> useVec;
 
  // 该指令归属的 Pipeline
  pto::PipelineType kPipeValue;
 
  OperationName opName;
  TCoreType compoundCoreType{TCoreType::CUBE_OR_VECTOR};
  SyncOperation *BwdPipeMPipeMTE1SyncPtr{nullptr};
 
  UNIT_FLAG unitFlagModeAsSet{UNIT_FLAG::DISABLED};
  UNIT_FLAG unitFlagModeAsWait{UNIT_FLAG::DISABLED};
  CompoundInstanceElement *linkedUnitFlagCompAsSet{nullptr};
  CompoundInstanceElement *linkedUnitFlagCompAsWait{nullptr};
  int macroOpInstanceId{-1};
 
public:
  CompoundInstanceElement(unsigned index,
                          SmallVector<const BaseMemInfo *> defVec,
                          SmallVector<const BaseMemInfo *> useVec,
                          const pto::PipelineType PipeValue, OperationName opName)
      : InstanceElement(KindTy::COMPOUND, index), defVec(std::move(defVec)),
        useVec(std::move(useVec)), kPipeValue(PipeValue), opName(opName) {}
 
  ~CompoundInstanceElement() override = default;
 
  static bool classof(const InstanceElement *e);
  UNIT_FLAG getUnitFlagMode() const;
  
  // PTO 暂时简化，去掉复杂的 UnitFlag 条件生成逻辑，或者稍后在 CPP 中适配
  std::optional<mlir::Value> getUnitFlagCond(Location loc, OpBuilder &rewriter);
};
 
using SyncIRs = SmallVector<std::unique_ptr<InstanceElement>>;
using SyncOperations = SmallVector<SmallVector<std::unique_ptr<SyncOperation>>>;
using Buffer2MemInfoMap =
    llvm::DenseMap<Value, llvm::SmallVector<std::unique_ptr<BaseMemInfo>>>;
 
// Utilities
bool checkAllParentLoopsAreForLoops(Operation *op);
void checkSyncIRIndex(const SyncIRs &syncIR, int index);
void checkCondition(bool condition, const std::string &message);
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_SYNC_COMMON_H