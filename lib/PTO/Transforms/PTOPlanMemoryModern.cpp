// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPlanMemoryModern.cpp - modern static local memory planner -------===//

#include <limits>
#include <tuple>
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"


using namespace mlir;
using namespace mlir::pto;

namespace {

struct MemSpec {
  uint64_t capacityBytes = 0;
  uint64_t alignmentBytes = 1;
};

struct RootInfo {
  Value root;
  Operation *defOp = nullptr;
  AddressSpace space = AddressSpace::Zero;
  uint64_t slotBytes = 0;
  uint64_t totalBytes = 0;
  uint64_t alignmentBytes = 1;
  uint64_t slotCount = 1;
  unsigned allocIndex = 0;
  unsigned freeIndex = 0;
  unsigned stableOrder = 0;
  bool hasWriter = false;
  bool hasUseBeforeFirstWrite = false;
  unsigned accessCount = 0;
  unsigned writeAccessCount = 0;
  unsigned loopAccessCount = 0;
  unsigned pipeVAccessCount = 0;
  unsigned mteAccessCount = 0;
  SmallVector<uint64_t> offsets;
};

using RootList = SmallVector<Value, mlir::pto::kValue4>;

struct OpAccess {
  Operation *op = nullptr;
  PIPE pipe = PIPE::PIPE_ALL;
  unsigned opIndex = 0;
  bool inLoop = false;
  RootList reads;
  RootList writes;
};

struct ConflictFacts {
  DenseMap<Value, RootList> forbidAlias;
  DenseSet<Value> loadDerivedRoots;
  DenseSet<Value> tpopConsumerRoots;
  DenseMap<Value, SmallVector<unsigned>> tpopConsumerWriteIndices;
  DenseMap<Value, RootList> branchExclusiveRoots;
  DenseSet<Value> loopCarriedRoots;
  SmallVector<OpAccess> opAccesses;
  bool targetHazardEnabled = false;
};

struct InplacePolicy {
  bool notInplaceSafe = false;
  SmallVector<unsigned> forbidOutputAliasOperands;
};

struct ReuseGroup {
  AddressSpace space = AddressSpace::Zero;
  uint64_t sizeBytes = 0;
  uint64_t alignmentBytes = 1;
  uint64_t offsetBytes = 0;
  SmallVector<RootInfo *> members;
};

static uint64_t alignUp(uint64_t value, uint64_t align) {
  if (align == 0) {
    return value;
  }
  return ((value + align - 1) / align) * align;
}

static std::optional<AddressSpace> getBufferAddressSpace(Type type) {
  if (auto tileType = dyn_cast<TileBufType>(type)) {
    if (auto attr =
            dyn_cast_or_null<AddressSpaceAttr>(tileType.getMemorySpace())) {
      return attr.getAddressSpace();
    }
    return std::nullopt;
  }

  if (auto multiType = dyn_cast<MultiTileBufType>(type)) {
    return getBufferAddressSpace(multiType.getSlotType());
  }

  return std::nullopt;
}

static bool isPlannableLocalSpace(std::optional<AddressSpace> space) {
  return space && *space != AddressSpace::GM && *space != AddressSpace::Zero;
}

static bool isCubeLocalSpace(AddressSpace space) {
  return space == AddressSpace::MAT || space == AddressSpace::LEFT ||
         space == AddressSpace::RIGHT || space == AddressSpace::ACC;
}

static bool isNameIn(StringRef name, ArrayRef<StringRef> names) {
  return llvm::is_contained(names, name);
}

static bool isIgnoredA5TmpOperandUse(OpOperand &use) {
  Operation *owner = use.getOwner();
  unsigned operandNo = use.getOperandNumber();
  StringRef name = owner->getName().getStringRef();

  if (auto dpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(owner)) {
    if (llvm::is_contained(dpsOp.getDpsInits(), use.get())) {
      return false;
    }
  } else if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(owner)) {
    if (llvm::is_contained(dpsOp.getDpsInits(), use.get())) {
      return false;
    }
  }

  if (isNameIn(name, {"pto.trowargmax", "pto.trowargmin", "pto.trowmax",
                     "pto.trowmin", "pto.trowsum", "pto.trowprod"})) {
    return operandNo == 1;
  }

  if (name == "pto.txors") {
    return operandNo == mlir::pto::kValue2;
  }

  if (isNameIn(name, {"pto.tprelu", "pto.txor", "pto.tsels",
                      "pto.trowexpand", "pto.tcolexpand",
                      "pto.trowexpandadd", "pto.trowexpanddiv",
                      "pto.trowexpandexpdif", "pto.trowexpandmax",
                      "pto.trowexpandmin", "pto.trowexpandmul",
                      "pto.trowexpandsub", "pto.tcolexpandadd",
                      "pto.tcolexpanddiv", "pto.tcolexpandexpdif",
                      "pto.tcolexpandmax", "pto.tcolexpandmin",
                      "pto.tcolexpandmul", "pto.tcolexpandsub"})) {
    return operandNo == mlir::pto::kValue2;
  }

  if (name == "pto.tsel") {
    return operandNo == mlir::pto::kValue3;
  }

  return false;
}

static bool isA5IgnoredTmpAlloc(pto::AllocTileOp allocTile) {
  if (getTargetArch(allocTile.getOperation()) != PTOArch::A5) {
    return false;
  }

  Value value = allocTile.getResult();
  if (value.use_empty()) {
    return false;
  }

  for (OpOperand &use : value.getUses()) {
    if (!isIgnoredA5TmpOperandUse(use)) {
      return false;
    }
  }
  return true;
}

static MemSpec getMemSpec(PTOArch arch, AddressSpace space) {
  switch (space) {
  case AddressSpace::VEC:
    return {arch == PTOArch::A5 ? 253952ULL : 196608ULL, 256};
  case AddressSpace::MAT:
    return {524288ULL, 256};
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
    return {65536ULL, 4096};
  case AddressSpace::ACC:
    return {arch == PTOArch::A5 ? 262144ULL : 131072ULL, 4096};
  case AddressSpace::BIAS:
    return {65536ULL, 256};
  case AddressSpace::SCALING:
    return {arch == PTOArch::A5 ? 253952ULL : 196608ULL, 256};
  case AddressSpace::GM:
  case AddressSpace::Zero:
    break;
  }
  return {};
}

static FailureOr<uint64_t> computeStaticBufferBytes(Value value) {
  auto computeTileBytes = [](TileBufType type) -> FailureOr<uint64_t> {
    ArrayRef<int64_t> shape = type.getShape();
    uint64_t elemBytes = getPTOStorageElemByteSize(type.getElementType());
    if (elemBytes == 0) {
      return failure();
    }

    if (type.getCompactModeI32() ==
        static_cast<int32_t>(CompactMode::RowPlusOne)) {
      if (shape.size() != 2 ||
          llvm::is_contained(shape, ShapedType::kDynamic)) {
        return failure();
      }

      bool rowMajor = type.getBLayoutValueI32() ==
                      static_cast<int32_t>(BLayout::RowMajor);
      uint64_t major = static_cast<uint64_t>(rowMajor ? shape[0] : shape[1]);
      uint64_t minor = static_cast<uint64_t>(rowMajor ? shape[1] : shape[0]);
      if (major == 0 || minor == 0) {
        return uint64_t{0};
      }
      return ((major - 1) * (minor + 1) + minor) * elemBytes;
    }

    uint64_t numel = 1;
    for (int64_t dim : shape) {
      if (dim == ShapedType::kDynamic) {
        return failure();
      }
      numel *= static_cast<uint64_t>(dim);
    }
    return numel * elemBytes;
  };

  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    return computeTileBytes(tileType);
  }
  if (auto multiType = dyn_cast<MultiTileBufType>(value.getType())) {
    return computeTileBytes(multiType.getSlotType());
  }
  return failure();
}

static void appendUniqueRoot(RootList &roots, Value root) {
  if (llvm::is_contained(roots, root)) {
    return;
  }
  roots.push_back(root);
}

static RootList unionRoots(const RootList &lhs, const RootList &rhs) {
  RootList result = lhs;
  for (Value root : rhs) {
    appendUniqueRoot(result, root);
  }
  return result;
}

static bool isOneOf(StringRef name, ArrayRef<StringRef> names) {
  return llvm::is_contained(names, name);
}

static InplacePolicy getInplacePolicy(Operation *op) {
  StringRef name = op->getName().getStringRef();
  InplacePolicy policy;

  policy.notInplaceSafe = isOneOf(
      name, {
                "pto.tands",      "pto.tfmod",
                "pto.tfmods",     "pto.tgather",         "pto.tmrgsort",
                "pto.tors",       "pto.trecip",          "pto.trsqrt",
                "pto.tsort32",    "pto.ttrans",          "pto.trowargmax",
                "pto.trowargmin", "pto.trowmax",         "pto.trowmin",
                "pto.trowprod",   "pto.trowsum",         "pto.tcolargmax",
                "pto.tcolargmin", "pto.tcvt",            "pto.txors",
            });

  if (auto fillPad = dyn_cast<TFillPadOp>(op)) {
    auto expanded = hasTFillPadExpandedPhysicalShape(fillPad);
    policy.notInplaceSafe |= failed(expanded) || *expanded;
  }

  if (name == "pto.tsel") {
    policy.forbidOutputAliasOperands.push_back(0); // mask
    policy.forbidOutputAliasOperands.push_back(mlir::pto::kValue3); // tmp
  }

  if (isOneOf(name, {
                        "pto.trowexpand",
                        "pto.tcolexpand",
                    })) {
    policy.forbidOutputAliasOperands.push_back(0);
  }

  if (isOneOf(name, {
                        "pto.trowexpandadd",
                        "pto.trowexpanddiv",
                        "pto.trowexpandexpdif",
                        "pto.trowexpandmax",
                        "pto.trowexpandmin",
                        "pto.trowexpandmul",
                        "pto.trowexpandsub",
                        "pto.tcolexpandadd",
                        "pto.tcolexpanddiv",
                        "pto.tcolexpandexpdif",
                        "pto.tcolexpandmax",
                        "pto.tcolexpandmin",
                        "pto.tcolexpandmul",
                        "pto.tcolexpandsub",
                    })) {
    policy.forbidOutputAliasOperands.push_back(1);
  }

  return policy;
}

static SmallVector<Value> getWrittenNonDpsOperands(Operation *op,
                                                   ValueRange dpsInits) {
  SmallVector<Value> scratchOperands;
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    return scratchOperands;
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, mlir::pto::kValue8> effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }
    Value value = effect.getValue();
    if (!value) {
      continue;
    }
    if (!llvm::is_contained(op->getOperands(), value)) {
      continue;
    }
    if (llvm::is_contained(dpsInits, value)) {
      continue;
    }
    if (!llvm::is_contained(scratchOperands, value)) {
      scratchOperands.push_back(value);
    }
  }
  return scratchOperands;
}

static bool hasReadEffectOnValue(Operation *op, Value value) {
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect) {
    return false;
  }

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, mlir::pto::kValue8> effects;
  memEffect.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect())) {
      continue;
    }
    if (effect.getValue() == value) {
      return true;
    }
  }
  return false;
}

static bool isInsideLoop(Operation *op) {
  return op->getParentOfType<scf::ForOp>() != nullptr ||
         op->getParentOfType<scf::WhileOp>() != nullptr;
}

static ValueRange getDpsInits(Operation *op) {
  if (auto dpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    return dpsOp.getDpsInits();
  }
  if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op)) {
    return dpsOp.getDpsInits();
  }
  return ValueRange();
}

struct PlannerAnalysis {
  func::FuncOp func;
  DenseMap<Value, RootList> valueToRoots;
  DenseSet<Value> splitTpopDerivedValues;
  SmallVector<Operation *> linearOps;
  DenseMap<Operation *, unsigned> opToIndex;
  SmallVector<RootInfo> roots;
  DenseMap<Value, unsigned> rootIndexByValue;
  ConflictFacts facts;
  bool failed = false;

  explicit PlannerAnalysis(func::FuncOp func) : func(func) {
    facts.targetHazardEnabled =
        getTargetArch(func.getOperation()) == PTOArch::A3;
  }

  RootList getRoots(Value value) const {
    auto it = valueToRoots.find(value);
    if (it == valueToRoots.end()) {
      return {};
    }
    return it->second;
  }

  void setRoots(Value value, const RootList &roots) {
    if (roots.empty()) {
      return;
    }
    valueToRoots[value] = roots;
  }

  void mergeRoots(Value value, const RootList &roots) {
    if (roots.empty()) {
      return;
    }
    setRoots(value, unionRoots(getRoots(value), roots));
  }

  void addRoot(Value value, Operation *defOp) {
    if (rootIndexByValue.count(value)) {
      return;
    }

    auto space = getBufferAddressSpace(value.getType());
    if (!isPlannableLocalSpace(space)) {
      return;
    }

    auto bytesOr = computeStaticBufferBytes(value);
    if (mlir::failed(bytesOr)) {
      defOp->emitError("requires a static local buffer shape and known element "
                       "byte size for memory planning");
      failed = true;
      return;
    }

    uint64_t slotCount = 1;
    if (auto multiType = dyn_cast<MultiTileBufType>(value.getType())) {
      slotCount = multiType.getCount();
    } else if (auto attr =
                   defOp->getAttrOfType<IntegerAttr>(pto::kPtoMultiBufferAttrName)) {
      slotCount = attr.getValue().getZExtValue();
    }
    MemSpec spec = getMemSpec(getTargetArch(func), *space);
    uint64_t slotBytes = alignUp(*bytesOr, spec.alignmentBytes);

    RootInfo info;
    info.root = value;
    info.defOp = defOp;
    info.space = *space;
    info.slotBytes = slotBytes;
    info.totalBytes = slotBytes * slotCount;
    info.alignmentBytes = spec.alignmentBytes;
    info.slotCount = slotCount;
    info.stableOrder = roots.size();
    roots.push_back(info);
    rootIndexByValue[value] = roots.size() - 1;
    valueToRoots[value] = RootList{value};
  }

  void markUse(Value value, unsigned index) {
    for (Value root : getRoots(value)) {
      auto found = rootIndexByValue.find(root);
      if (found == rootIndexByValue.end()) {
        continue;
      }
      RootInfo &info = roots[found->second];
      if (!info.hasWriter) {
        info.hasUseBeforeFirstWrite = true;
      }
      if (index < info.allocIndex) {
        info.allocIndex = index;
      }
      info.freeIndex = std::max(info.freeIndex, index);
    }
  }

  void markWrite(Value value, unsigned index, bool pureOverwrite) {
    for (Value root : getRoots(value)) {
      auto found = rootIndexByValue.find(root);
      if (found == rootIndexByValue.end()) {
        continue;
      }
      RootInfo &info = roots[found->second];
      if (!info.hasWriter) {
        if (pureOverwrite && !info.hasUseBeforeFirstWrite) {
          info.allocIndex = index;
        }
        info.hasWriter = true;
      }
      info.freeIndex = std::max(info.freeIndex, index);
    }
  }

  void addForbidAlias(Value a, Value b) {
    if (a == b) {
      return;
    }
    if (!rootIndexByValue.count(a) || !rootIndexByValue.count(b)) {
      return;
    }
    appendUniqueRoot(facts.forbidAlias[a], b);
    appendUniqueRoot(facts.forbidAlias[b], a);
  }

  bool hasForbidAlias(Value a, Value b) const {
    if (a == b) {
      return false;
    }
    auto it = facts.forbidAlias.find(a);
    if (it == facts.forbidAlias.end()) {
      return false;
    }
    return llvm::is_contained(it->second, b);
  }

  void addForbidAliasBetweenRoots(const RootList &lhsRoots,
                                  const RootList &rhsRoots) {
    for (Value lhs : lhsRoots) {
      for (Value rhs : rhsRoots) {
        addForbidAlias(lhs, rhs);
      }
    }
  }

  void markRoots(DenseSet<Value> &set, const RootList &roots) {
    for (Value root : roots) {
      set.insert(root);
    }
  }

  bool rootsContain(const DenseSet<Value> &set, const RootList &roots) const {
    return llvm::any_of(roots, [&](Value root) { return set.contains(root); });
  }

  bool isSplitTpopDerived(Value value) const {
    return splitTpopDerivedValues.contains(value);
  }

  bool operandsContainSplitTpopDerived(Operation *op) const {
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      return isSplitTpopDerived(operand);
    });
  }

  bool operandsContainLoadDerivedRoot(Operation *op) const {
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      return rootsContain(facts.loadDerivedRoots, getRoots(operand));
    });
  }

  void propagateSplitTpopDerived(Value result, ValueRange sources) {
    if (!result) {
      return;
    }
    if (llvm::any_of(sources, [&](Value source) {
          return isSplitTpopDerived(source);
        })) {
      splitTpopDerivedValues.insert(result);
    }
  }

  void propagateSplitTpopDerivedFromRoots(Value result, Value source) {
    if (!result || !source) {
      return;
    }
    if (isSplitTpopDerived(source)) {
      splitTpopDerivedValues.insert(result);
    }
  }

  void propagateSplitTpopDerivedFromOperands(Operation *op) {
    if (!operandsContainSplitTpopDerived(op)) {
      return;
    }
    for (Value result : op->getResults()) {
      splitTpopDerivedValues.insert(result);
    }
  }

  bool isRootDefinedInRegion(Value root, Region &region) const {
    auto it = rootIndexByValue.find(root);
    if (it == rootIndexByValue.end()) {
      return false;
    }

    Operation *defOp = roots[it->second].defOp;
    for (Operation *cur = defOp; cur; cur = cur->getParentOp()) {
      if (cur->getParentRegion() == &region) {
        return true;
      }
    }
    return false;
  }

  void addBranchExclusivePair(Value lhs, Value rhs) {
    if (lhs == rhs) {
      return;
    }
    if (!rootIndexByValue.count(lhs) || !rootIndexByValue.count(rhs)) {
      return;
    }
    appendUniqueRoot(facts.branchExclusiveRoots[lhs], rhs);
    appendUniqueRoot(facts.branchExclusiveRoots[rhs], lhs);
  }

  void recordIfBranchExclusivity(scf::IfOp ifOp) {
    if (ifOp.getNumResults() == 0) {
      return;
    }

    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    for (auto [thenVal, elseVal] :
         llvm::zip(thenYield.getResults(), elseYield.getResults())) {
      RootList thenLocalRoots;
      RootList elseLocalRoots;
      for (Value root : getRoots(thenVal)) {
        if (isRootDefinedInRegion(root, ifOp.getThenRegion())) {
          appendUniqueRoot(thenLocalRoots, root);
        }
      }
      for (Value root : getRoots(elseVal)) {
        if (isRootDefinedInRegion(root, ifOp.getElseRegion())) {
          appendUniqueRoot(elseLocalRoots, root);
        }
      }

      for (Value thenRoot : thenLocalRoots) {
        for (Value elseRoot : elseLocalRoots) {
          addBranchExclusivePair(thenRoot, elseRoot);
        }
      }
    }
  }

  void recordDpsScratchConflicts(Operation *op, ValueRange dpsInits) {
    RootList outputRoots;
    for (Value init : dpsInits) {
      outputRoots = unionRoots(outputRoots, getRoots(init));
    }
    if (outputRoots.empty()) {
      return;
    }

    for (Value scratch : getWrittenNonDpsOperands(op, dpsInits)) {
      addForbidAliasBetweenRoots(getRoots(scratch), outputRoots);
      for (Value operand : op->getOperands()) {
        if (operand == scratch || llvm::is_contained(dpsInits, operand)) {
          continue;
        }
        addForbidAliasBetweenRoots(getRoots(scratch), getRoots(operand));
      }
    }
  }

  void recordInplacePolicyConflicts(Operation *op, ValueRange dpsInits) {
    RootList outputRoots;
    for (Value init : dpsInits) {
      outputRoots = unionRoots(outputRoots, getRoots(init));
    }
    if (outputRoots.empty()) {
      return;
    }

    InplacePolicy policy = getInplacePolicy(op);
    if (policy.notInplaceSafe) {
      for (Value operand : op->getOperands()) {
        if (llvm::is_contained(dpsInits, operand)) {
          continue;
        }
        addForbidAliasBetweenRoots(getRoots(operand), outputRoots);
      }
    }

    for (unsigned operandIndex : policy.forbidOutputAliasOperands) {
      if (operandIndex >= op->getNumOperands()) {
        continue;
      }
      Value operand = op->getOperand(operandIndex);
      if (llvm::is_contained(dpsInits, operand)) {
        continue;
      }
      addForbidAliasBetweenRoots(getRoots(operand), outputRoots);
    }
  }

  void recordDpsInplaceConflicts(Operation *op) {
    auto dpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
    if (!dpsOp) {
      return;
    }

    ValueRange dpsInits = dpsOp.getDpsInits();
    recordDpsScratchConflicts(op, dpsInits);
    recordInplacePolicyConflicts(op, dpsInits);
  }

  void recordSemanticNoAliasConflicts(Operation *op) {
    for (const auto &[lhs, rhs] : getSemanticNoAliasPairs(op))
      addForbidAliasBetweenRoots(getRoots(lhs), getRoots(rhs));
  }

  void recordLoadDerivedRoots(Operation *op, ValueRange dpsInits) {
    if (!isa<pto::TLoadOp, pto::TPrefetchOp>(op)) {
      return;
    }

    for (Value init : dpsInits) {
      markRoots(facts.loadDerivedRoots, getRoots(init));
    }
  }

  void recordTpopConsumerRoots(Operation *op, ValueRange dpsInits,
                               unsigned opIndex) {
    if (!facts.targetHazardEnabled) {
      return;
    }
    if (!operandsContainSplitTpopDerived(op) || !operandsContainLoadDerivedRoot(op)) {
      return;
    }

    RootList outputRoots;
    for (Value init : dpsInits) {
      outputRoots = unionRoots(outputRoots, getRoots(init));
    }
    markRoots(facts.tpopConsumerRoots, outputRoots);
    for (Value root : outputRoots) {
      SmallVector<unsigned> &indices = facts.tpopConsumerWriteIndices[root];
      if (!llvm::is_contained(indices, opIndex)) {
        indices.push_back(opIndex);
      }
    }
  }

  void recordDpsTargetHazardFacts(Operation *op, unsigned opIndex) {
    auto dpsOp = dyn_cast<pto::PTO_DpsInitOpInterface>(op);
    if (!dpsOp) {
      return;
    }

    ValueRange dpsInits = dpsOp.getDpsInits();
    recordLoadDerivedRoots(op, dpsInits);
    recordTpopConsumerRoots(op, dpsInits, opIndex);
  }

  void appendAccessRoots(RootList &dst, Value value) {
    for (Value root : getRoots(value)) {
      appendUniqueRoot(dst, root);
    }
  }

  void recordRootAccessStats(ArrayRef<Value> accessRoots, PIPE pipe,
                             bool isWrite, bool inLoop) {
    for (Value root : accessRoots) {
      auto found = rootIndexByValue.find(root);
      if (found == rootIndexByValue.end()) {
        continue;
      }
      RootInfo &info = roots[found->second];
      ++info.accessCount;
      if (isWrite) {
        ++info.writeAccessCount;
      }
      if (inLoop) {
        ++info.loopAccessCount;
      }
      if (pipe == PIPE::PIPE_V) {
        ++info.pipeVAccessCount;
      }
      if (pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3) {
        ++info.mteAccessCount;
      }
    }
  }

  void recordOpAccess(Operation *op, ValueRange dpsInits) {
    auto pipeOp = dyn_cast<pto::OpPipeInterface>(op);
    auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
    if (!pipeOp || !memEffect) {
      return;
    }

    OpAccess access;
    access.op = op;
    access.pipe = pipeOp.getPipe();
    access.opIndex = facts.opAccesses.size();
    access.inLoop = isInsideLoop(op);

    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, mlir::pto::kValue8> effects;
    memEffect.getEffects(effects);
    for (const auto &effect : effects) {
      Value value = effect.getValue();
      if (!value) {
        continue;
      }
      if (isa<MemoryEffects::Read>(effect.getEffect())) {
        appendAccessRoots(access.reads, value);
      }
      if (isa<MemoryEffects::Write>(effect.getEffect())) {
        appendAccessRoots(access.writes, value);
      }
    }

    // Some PTO tile ops expose DPS destinations through the project-specific
    // interface even when a future op forgets to model MemoryEffects. Keep the
    // performance side table conservative by treating DPS inits as writes.
    for (Value init : dpsInits) {
      appendAccessRoots(access.writes, init);
    }

    if (!access.reads.empty() || !access.writes.empty()) {
      recordRootAccessStats(access.reads, access.pipe, /*isWrite=*/false,
                            access.inLoop);
      recordRootAccessStats(access.writes, access.pipe, /*isWrite=*/true,
                            access.inLoop);
      facts.opAccesses.push_back(std::move(access));
    }
  }

  void recordSplitTpopDerivedValue(Operation *op) {
    if (!facts.targetHazardEnabled) {
      return;
    }

    if (auto pop = dyn_cast<pto::TPopFromAicOp>(op)) {
      if (pop.getSplit() != 0) {
        splitTpopDerivedValues.insert(pop.getTile());
      }
      return;
    }

    if (auto pop = dyn_cast<pto::TPopOp>(op)) {
      if (pop.getSplit() != 0) {
        splitTpopDerivedValues.insert(pop.getTile());
      }
      return;
    }

    propagateSplitTpopDerivedFromOperands(op);
  }

  void seedForIterArgAliases(scf::ForOp forOp) {
    if (forOp.getRegion().empty()) {
      return;
    }
    Block &body = forOp.getRegion().front();
    for (auto [iterArg, initArg] :
         llvm::zip(body.getArguments().drop_front(1), forOp.getInitArgs())) {
      setRoots(iterArg, getRoots(initArg));
    }
  }

  void mapRegionBranchValues(ValueRange sources, ValueRange destinations) {
    for (auto [source, destination] : llvm::zip(sources, destinations)) {
      mergeRoots(destination, getRoots(source));
      propagateSplitTpopDerivedFromRoots(destination, source);
    }
  }

  void seedRegionBranchEntryAliases(RegionBranchOpInterface branchOp) {
    SmallVector<RegionSuccessor> successors;
    branchOp.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (RegionSuccessor successor : successors) {
      if (successor.isParent()) {
        continue;
      }
      mapRegionBranchValues(branchOp.getEntrySuccessorOperands(successor),
                            successor.getSuccessorInputs());
    }
  }

  void finalizeRegionBranchAliasesFromRegion(RegionBranchOpInterface branchOp,
                                             Region &region) {
    if (region.empty()) {
      return;
    }

    SmallVector<RegionSuccessor> successors;
    branchOp.getSuccessorRegions(region, successors);
    for (RegionSuccessor successor : successors) {
      ValueRange destinations = successor.getSuccessorInputs();
      if (destinations.empty()) {
        continue;
      }

      for (Block &block : region) {
        auto terminator =
            dyn_cast<RegionBranchTerminatorOpInterface>(block.getTerminator());
        if (!terminator) {
          continue;
        }
        mapRegionBranchValues(terminator.getSuccessorOperands(successor),
                              destinations);
      }
    }
  }

  void finalizeRegionBranchAliases(RegionBranchOpInterface branchOp) {
    for (Region &region : branchOp->getRegions()) {
      finalizeRegionBranchAliasesFromRegion(branchOp, region);
    }
  }

  void finalizeForLoopLiveness(scf::ForOp forOp, unsigned loopStartIndex,
                               unsigned loopEndIndex) {
    // A root defined outside the loop and touched in the body remains live
    // across the back-edge, even if its last use appears before a loop-local
    // allocation in the linear walk.
    for (RootInfo &info : roots) {
      if (isRootDefinedInRegion(info.root, forOp.getRegion())) {
        continue;
      }
      if (info.freeIndex >= loopStartIndex) {
        info.freeIndex = std::max(info.freeIndex, loopEndIndex);
      }
    }

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (auto [iterArg, result, initArg, yielded] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getResults(),
                   forOp.getInitArgs(), yieldOp.getResults())) {
      RootList loopCarriedRoots =
          unionRoots(getRoots(initArg), getRoots(yielded));
      setRoots(iterArg, loopCarriedRoots);
      setRoots(result, loopCarriedRoots);

      // Every possible root of an iter arg may be consumed on a later
      // iteration. Conservatively keep the family live for the whole loop and
      // do not apply branch-exclusivity from one iteration across the back-edge.
      for (Value root : loopCarriedRoots) {
        auto found = rootIndexByValue.find(root);
        if (found == rootIndexByValue.end()) {
          continue;
        }
        facts.loopCarriedRoots.insert(root);
        RootInfo &info = roots[found->second];
        info.allocIndex = std::min(info.allocIndex, loopStartIndex);
        info.freeIndex = std::max(info.freeIndex, loopEndIndex);
      }
    }
  }

  void finalizeWhileLoopLiveness(scf::WhileOp whileOp, unsigned loopStartIndex,
                                 unsigned loopEndIndex) {
    // scf.while may carry tile roots through both before/after regions:
    // init -> before args -> condition args -> after args -> yield -> before
    // args. Keep every participating root live across the whole region branch
    // cycle so a later linear operation in one region cannot reuse storage that
    // a future iteration still reads.
    RootList loopCarriedRoots;
    for (Value init : whileOp.getInits()) {
      loopCarriedRoots = unionRoots(loopCarriedRoots, getRoots(init));
    }
    for (Value result : whileOp.getResults()) {
      loopCarriedRoots = unionRoots(loopCarriedRoots, getRoots(result));
    }
    for (Region &region : whileOp->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument arg : block.getArguments()) {
          loopCarriedRoots = unionRoots(loopCarriedRoots, getRoots(arg));
        }
        if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(
                block.getTerminator())) {
          for (Value operand : terminator->getOperands()) {
            loopCarriedRoots = unionRoots(loopCarriedRoots, getRoots(operand));
          }
        }
      }
    }

    for (Value root : loopCarriedRoots) {
      auto found = rootIndexByValue.find(root);
      if (found == rootIndexByValue.end()) {
        continue;
      }
      facts.loopCarriedRoots.insert(root);
      RootInfo &info = roots[found->second];
      info.allocIndex = std::min(info.allocIndex, loopStartIndex);
      info.freeIndex = std::max(info.freeIndex, loopEndIndex);
    }
  }

  void walkRegion(Region &region) {
    for (Block &block : region) {
      for (Operation &opRef : block) {
        Operation *op = &opRef;
        unsigned index = linearOps.size();
        linearOps.push_back(op);
        opToIndex[op] = index;

        if (auto allocTile = dyn_cast<pto::AllocTileOp>(op)) {
          if (!allocTile.getAddr()) {
            if (isA5IgnoredTmpAlloc(allocTile)) {
              continue;
            }
            addRoot(allocTile.getResult(), op);
            if (failed) {
              return;
            }
            auto found = rootIndexByValue.find(allocTile.getResult());
            if (found != rootIndexByValue.end()) {
              roots[found->second].allocIndex = index;
              roots[found->second].freeIndex = index;
            }
          }
        } else if (auto allocMulti = dyn_cast<pto::AllocMultiTileOp>(op)) {
          if (!allocMulti.getAddr()) {
            addRoot(allocMulti.getResult(), op);
            if (failed) {
              return;
            }
            auto found = rootIndexByValue.find(allocMulti.getResult());
            if (found != rootIndexByValue.end()) {
              roots[found->second].allocIndex = index;
              roots[found->second].freeIndex = index;
            }
          }
        }

        if (auto multiGet = dyn_cast<pto::MultiTileGetOp>(op)) {
          setRoots(multiGet.getResult(), getRoots(multiGet.getSource()));
          propagateSplitTpopDerived(multiGet.getResult(),
                                    ValueRange{multiGet.getSource()});
        } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
          setRoots(select.getResult(),
                   unionRoots(getRoots(select.getTrueValue()),
                              getRoots(select.getFalseValue())));
          propagateSplitTpopDerived(select.getResult(),
                                    ValueRange{select.getTrueValue(),
                                               select.getFalseValue()});
        } else if (auto subview = dyn_cast<pto::SubViewOp>(op)) {
          setRoots(subview.getResult(), getRoots(subview.getSource()));
          propagateSplitTpopDerived(subview.getResult(),
                                    ValueRange{subview.getSource()});
        } else if (auto bitcast = dyn_cast<pto::BitcastOp>(op)) {
          setRoots(bitcast.getResult(), getRoots(bitcast.getSrc()));
          propagateSplitTpopDerived(bitcast.getResult(),
                                    ValueRange{bitcast.getSrc()});
        } else if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
          setRoots(reshape.getResult(), getRoots(reshape.getSrc()));
          propagateSplitTpopDerived(reshape.getResult(),
                                    ValueRange{reshape.getSrc()});
        } else if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
          seedRegionBranchEntryAliases(branchOp);
        }

        ValueRange dpsInits = getDpsInits(op);
        for (Value operand : op->getOperands()) {
          if (llvm::is_contained(dpsInits, operand)) {
            continue;
          }
          markUse(operand, index);
        }
        for (Value init : dpsInits) {
          bool readsOldValue = hasReadEffectOnValue(op, init);
          if (readsOldValue) {
            markUse(init, index);
          }
          markWrite(init, index, /*pureOverwrite=*/!readsOldValue);
        }

        if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
          for (Region &nested : op->getRegions()) {
            walkRegion(nested);
            if (failed) {
              return;
            }
            finalizeRegionBranchAliasesFromRegion(branchOp, nested);
          }
        } else {
          for (Region &nested : op->getRegions()) {
            walkRegion(nested);
            if (failed) {
              return;
            }
          }
        }

        if (auto branchOp = dyn_cast<RegionBranchOpInterface>(op)) {
          finalizeRegionBranchAliases(branchOp);
        }

        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          if (ifOp.getNumResults() != 0) {
            recordIfBranchExclusivity(ifOp);
          }
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          unsigned loopEndIndex =
              linearOps.empty() ? index : linearOps.size() - 1;
          finalizeForLoopLiveness(forOp, index, loopEndIndex);
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
          unsigned loopEndIndex =
              linearOps.empty() ? index : linearOps.size() - 1;
          finalizeWhileLoopLiveness(whileOp, index, loopEndIndex);
        } else if (auto fusionRegion = dyn_cast<pto::FusionRegionOp>(op)) {
          auto yieldOp = cast<pto::YieldOp>(
              fusionRegion.getBody().front().getTerminator());
          for (auto [result, yielded] :
               llvm::zip(fusionRegion.getResults(), yieldOp.getOperands())) {
            setRoots(result, getRoots(yielded));
          }
        }

        recordSplitTpopDerivedValue(op);
        recordDpsTargetHazardFacts(op, index);
        recordDpsInplaceConflicts(op);
        recordSemanticNoAliasConflicts(op);
        recordOpAccess(op, dpsInits);
      }
    }
  }
};

static bool lifetimesStrictlyOverlap(const RootInfo &lhs, const RootInfo &rhs) {
  return !(lhs.freeIndex <= rhs.allocIndex || rhs.freeIndex <= lhs.allocIndex);
}

static bool areBranchExclusive(Value lhs, Value rhs, const ConflictFacts &facts) {
  if (facts.loopCarriedRoots.contains(lhs) ||
      facts.loopCarriedRoots.contains(rhs)) {
    return false;
  }
  auto it = facts.branchExclusiveRoots.find(lhs);
  if (it == facts.branchExclusiveRoots.end()) {
    return false;
  }
  return llvm::is_contained(it->second, rhs);
}

static bool gateLifetimeAndPhi(const RootInfo &lhs, const RootInfo &rhs,
                               const ConflictFacts &facts) {
  if (!lifetimesStrictlyOverlap(lhs, rhs)) {
    return true;
  }

  // Gate 5 is intentionally embedded in Gate 1: branch-local roots yielded from
  // opposite scf.if branches are mutually exclusive at runtime, so their
  // post-phi liveness extension does not by itself forbid reuse.
  return areBranchExclusive(lhs.root, rhs.root, facts);
}

static bool hasTargetHazard(const RootInfo &input, const RootInfo &writer,
                            const ConflictFacts &facts) {
  if (!facts.targetHazardEnabled) {
    return false;
  }

  if (!facts.loadDerivedRoots.contains(input.root) ||
      !facts.tpopConsumerRoots.contains(writer.root)) {
    return false;
  }

  auto found = facts.tpopConsumerWriteIndices.find(writer.root);
  if (found == facts.tpopConsumerWriteIndices.end()) {
    return false;
  }

  return llvm::is_contained(found->second, input.freeIndex);
}

static bool gateTargetHazard(const RootInfo &lhs, const RootInfo &rhs,
                             const ConflictFacts &facts) {
  return !hasTargetHazard(lhs, rhs, facts) && !hasTargetHazard(rhs, lhs, facts);
}

static bool gateSemanticNoAlias(const RootInfo &lhs, const RootInfo &rhs,
                                const PlannerAnalysis &analysis) {
  return !analysis.hasForbidAlias(lhs.root, rhs.root);
}

static bool canShare(const RootInfo &lhs, const RootInfo &rhs,
                     const PlannerAnalysis &analysis) {
  const ConflictFacts &facts = analysis.facts;
  return gateLifetimeAndPhi(lhs, rhs, facts) &&
         gateTargetHazard(lhs, rhs, facts) &&
         gateSemanticNoAlias(lhs, rhs, analysis);
}

static bool canJoinReuseGroup(const RootInfo &info, const ReuseGroup &group,
                              const PlannerAnalysis &analysis) {
  if (info.space != group.space) {
    return false;
  }
  for (const RootInfo *member : group.members) {
    if (!canShare(info, *member, analysis)) {
      return false;
    }
  }
  return true;
}

static bool containsRoot(ArrayRef<Value> roots, Value root) {
  return llvm::is_contained(roots, root);
}

static bool accessesRoot(const OpAccess &access, Value root) {
  return containsRoot(access.reads, root) || containsRoot(access.writes, root);
}

static bool accessPairHasMemoryDependency(const OpAccess &lhs,
                                          const OpAccess &rhs, Value lhsRoot,
                                          Value rhsRoot) {
  auto depends = [&lhs, &rhs](Value writtenRoot, Value otherRoot) {
    return containsRoot(lhs.writes, writtenRoot) &&
           (containsRoot(rhs.reads, otherRoot) ||
            containsRoot(rhs.writes, otherRoot));
  };
  auto reverseDepends = [&lhs, &rhs](Value readRoot, Value writtenRoot) {
    return containsRoot(lhs.reads, readRoot) &&
           containsRoot(rhs.writes, writtenRoot);
  };

  return depends(lhsRoot, rhsRoot) || depends(rhsRoot, lhsRoot) ||
         reverseDepends(lhsRoot, rhsRoot) ||
         reverseDepends(rhsRoot, lhsRoot);
}

static bool isNearNeighbor(const OpAccess &lhs, const OpAccess &rhs,
                           unsigned lookahead) {
  unsigned lhsIndex = lhs.opIndex;
  unsigned rhsIndex = rhs.opIndex;
  unsigned distance = lhsIndex > rhsIndex ? lhsIndex - rhsIndex
                                          : rhsIndex - lhsIndex;
  return distance > 0 && distance <= lookahead;
}

static uint64_t loopWeightedPenalty(uint64_t penalty, const OpAccess &lhs,
                                    const OpAccess &rhs) {
  constexpr uint64_t kLoopWeight = 4;
  return (lhs.inLoop || rhs.inLoop) ? penalty * kLoopWeight : penalty;
}

static uint64_t getRootPairReuseCost(Value lhsRoot, Value rhsRoot,
                                     const ConflictFacts &facts) {
  if (lhsRoot == rhsRoot) {
    return 0;
  }

  constexpr unsigned kPipeVLookahead = 1;
  constexpr unsigned kMteLookahead = 1;
  constexpr uint64_t kPipeVOverlapPenalty = 10;
  constexpr uint64_t kMte3ToMte2Penalty = 20;

  uint64_t cost = 0;
  for (const OpAccess &lhs : facts.opAccesses) {
    if (!accessesRoot(lhs, lhsRoot) && !accessesRoot(lhs, rhsRoot)) {
      continue;
    }

    for (const OpAccess &rhs : facts.opAccesses) {
      if (lhs.opIndex >= rhs.opIndex) {
        continue;
      }
      if (!accessesRoot(rhs, lhsRoot) && !accessesRoot(rhs, rhsRoot)) {
        continue;
      }

      if (lhs.pipe == PIPE::PIPE_V && rhs.pipe == PIPE::PIPE_V &&
          isNearNeighbor(lhs, rhs, kPipeVLookahead) &&
          accessPairHasMemoryDependency(lhs, rhs, lhsRoot, rhsRoot)) {
        cost += loopWeightedPenalty(kPipeVOverlapPenalty, lhs, rhs);
      }

      if (lhs.pipe == PIPE::PIPE_MTE3 && rhs.pipe == PIPE::PIPE_MTE2 &&
          isNearNeighbor(lhs, rhs, kMteLookahead)) {
        bool storeSourceThenLoadDst =
            (containsRoot(lhs.reads, lhsRoot) &&
             containsRoot(rhs.writes, rhsRoot)) ||
            (containsRoot(lhs.reads, rhsRoot) &&
             containsRoot(rhs.writes, lhsRoot));
        if (storeSourceThenLoadDst) {
          cost += loopWeightedPenalty(kMte3ToMte2Penalty, lhs, rhs);
        }
      }
    }
  }
  return cost;
}

static uint64_t getGroupReuseCost(const RootInfo &info,
                                  const ReuseGroup &group,
                                  const ConflictFacts &facts) {
  uint64_t cost = 0;
  for (const RootInfo *member : group.members) {
    cost += getRootPairReuseCost(info.root, member->root, facts);
  }
  return cost;
}

static bool isHotRoot(const RootInfo &info) {
  if (info.loopAccessCount > 0 &&
      (info.accessCount >= mlir::pto::kValue2 || info.pipeVAccessCount > 0 ||
       info.mteAccessCount > 0)) {
    return true;
  }

  // Some PTODSL kernels express repeated work through task/block parallelism
  // rather than an explicit scf.for in PTO IR. Repeated local accesses on
  // PIPE_V/MTE are still hot enough that over-clustering can hurt scheduling.
  return info.accessCount >= mlir::pto::kValue2 &&
         (info.pipeVAccessCount > 0 || info.mteAccessCount > 0);
}

static uint64_t getRootHotness(const RootInfo &info) {
  uint64_t hotness = info.accessCount;
  hotness += static_cast<uint64_t>(info.loopAccessCount) * mlir::pto::kValue2;
  hotness += static_cast<uint64_t>(info.writeAccessCount);
  hotness += static_cast<uint64_t>(info.pipeVAccessCount) * mlir::pto::kValue2;
  hotness += static_cast<uint64_t>(info.mteAccessCount) * mlir::pto::kValue2;
  return hotness;
}

static uint64_t getHotClusterReuseCost(const RootInfo &info,
                                       const ReuseGroup &group) {
  if (!isHotRoot(info)) {
    return 0;
  }

  constexpr uint64_t kHotClusterPenalty = 6;
  constexpr uint64_t kLoopHotClusterPenalty = 12;
  constexpr uint64_t kBankConflictPenalty = 4;
  constexpr uint64_t kBankConflictModuloBytes = 8192;

  uint64_t cost = 0;
  uint64_t infoHotness = getRootHotness(info);
  for (const RootInfo *member : group.members) {
    if (!isHotRoot(*member)) {
      continue;
    }

    uint64_t pairHotness =
        std::min<uint64_t>(infoHotness, getRootHotness(*member));
    cost += kHotClusterPenalty + pairHotness;
    if (info.loopAccessCount > 0 && member->loopAccessCount > 0) {
      cost += kLoopHotClusterPenalty;
    }

    // Exact co-location is the strongest possible same-bank signal. The
    // planner does not materialize final byte intervals yet, so only model the
    // bank pattern that is known for a reuse candidate: the new root would
    // start at the same offset as every member in this group.
    if (info.alignmentBytes <= kBankConflictModuloBytes &&
        member->alignmentBytes <= kBankConflictModuloBytes) {
      cost += kBankConflictPenalty;
    }
  }
  return cost;
}

static uint64_t getProjectedPackedBytes(ArrayRef<ReuseGroup> groups,
                                        const RootInfo &info,
                                        std::optional<unsigned> joinGroup) {
  uint64_t cursor = 0;
  for (auto [index, group] : llvm::enumerate(groups)) {
    uint64_t sizeBytes = group.sizeBytes;
    uint64_t alignmentBytes = group.alignmentBytes;
    if (joinGroup && *joinGroup == index) {
      sizeBytes = std::max(sizeBytes, info.totalBytes);
      alignmentBytes = std::max(alignmentBytes, info.alignmentBytes);
    }
    cursor = alignUp(cursor, alignmentBytes);
    cursor += sizeBytes;
  }

  if (!joinGroup) {
    cursor = alignUp(cursor, info.alignmentBytes);
    cursor += info.totalBytes;
  }
  return cursor;
}

static ReuseGroup *chooseReuseGroupByCost(
    RootInfo &info, SmallVectorImpl<ReuseGroup> &groups,
    const PlannerAnalysis &analysis, const MemSpec &spec) {
  ReuseGroup *bestGroup = nullptr;
  uint64_t bestCost = std::numeric_limits<uint64_t>::max();
  uint64_t bestProjectedBytes = std::numeric_limits<uint64_t>::max();
  unsigned bestOrder = std::numeric_limits<unsigned>::max();
  bool bestFits = false;

  for (auto [index, group] : llvm::enumerate(groups)) {
    if (!canJoinReuseGroup(info, group, analysis)) {
      continue;
    }

    uint64_t projectedBytes =
        getProjectedPackedBytes(groups, info, static_cast<unsigned>(index));
    bool fits = projectedBytes <= spec.capacityBytes;
    uint64_t cost = getGroupReuseCost(info, group, analysis.facts) +
                    getHotClusterReuseCost(info, group);
    if ((!bestGroup && !bestFits) || (fits != bestFits && fits) ||
        (fits == bestFits &&
         std::tie(cost, projectedBytes, index) <
             std::tie(bestCost, bestProjectedBytes, bestOrder))) {
      bestGroup = &group;
      bestCost = cost;
      bestProjectedBytes = projectedBytes;
      bestOrder = static_cast<unsigned>(index);
      bestFits = fits;
    }
  }

  uint64_t freshProjectedBytes = getProjectedPackedBytes(groups, info,
                                                         std::nullopt);
  bool freshFits = freshProjectedBytes <= spec.capacityBytes;
  uint64_t freshSlack =
      freshFits ? spec.capacityBytes - freshProjectedBytes : 0;
  uint64_t pressureReserve =
      std::max(info.totalBytes, info.alignmentBytes);
  // Fresh groups are not free: cost 1 lets true zero-cost reuse keep the old
  // compact layout, while still avoiding positive-cost PIPE/MTE co-location
  // when local capacity is available.
  uint64_t freshCost = groups.empty() ? 0 : 1;
  unsigned freshOrder = groups.size();
  if (!bestGroup) {
    return nullptr;
  }

  // The cost model is a performance hint, not a correctness gate. When local
  // memory is already tight, do not let a fresh address outrank a legal reuse
  // group; future roots may still need the remaining tail bytes.
  if (bestFits && freshFits && freshSlack < pressureReserve) {
    return bestGroup;
  }

  auto isBetter = [](bool lhsFits, uint64_t lhsCost, uint64_t lhsBytes,
                     unsigned lhsOrder, bool rhsFits, uint64_t rhsCost,
                     uint64_t rhsBytes, unsigned rhsOrder) {
    if (lhsFits != rhsFits) {
      return lhsFits;
    }
    return std::tie(lhsCost, lhsBytes, lhsOrder) <
           std::tie(rhsCost, rhsBytes, rhsOrder);
  };
  if (isBetter(freshFits, freshCost, freshProjectedBytes, freshOrder, bestFits,
               bestCost, bestProjectedBytes, bestOrder)) {
    return nullptr;
  }
  return bestGroup;
}

static SmallVector<uint64_t> buildSlotOffsets(uint64_t base, uint64_t slotBytes,
                                              uint64_t slotCount) {
  SmallVector<uint64_t> offsets;
  offsets.reserve(slotCount);
  for (uint64_t slot = 0; slot < slotCount; ++slot) {
    offsets.push_back(base + slot * slotBytes);
  }
  return offsets;
}

static FailureOr<uint64_t> planReserveBufferBase(
    pto::ReserveBufferOp reserveOp, const MemSpec &spec,
    SmallVectorImpl<std::pair<uint64_t, uint64_t>> &occupied) {
  uint64_t sizeBytes =
      alignUp(static_cast<uint64_t>(reserveOp.getSize()), spec.alignmentBytes);
  llvm::sort(occupied, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  SmallVector<std::pair<uint64_t, uint64_t>> merged;
  for (const auto &interval : occupied) {
    if (merged.empty() || interval.first > merged.back().second) {
      merged.push_back(interval);
      continue;
    }
    merged.back().second = std::max(merged.back().second, interval.second);
  }

  uint64_t cursor = 0;
  for (const auto &interval : merged) {
    cursor = alignUp(cursor, spec.alignmentBytes);
    if (cursor + sizeBytes <= interval.first) {
      return cursor;
    }
    cursor = std::max(cursor, interval.second);
  }
  cursor = alignUp(cursor, spec.alignmentBytes);
  if (cursor + sizeBytes > spec.capacityBytes) {
    return failure();
  }
  occupied.push_back({cursor, cursor + sizeBytes});
  return cursor;
}

static LogicalResult
validateManualReserveBufferBase(pto::ReserveBufferOp reserveOp,
                                const MemSpec &spec) {
  auto baseAttr = reserveOp.getBaseAttr();
  if (!baseAttr) {
    return reserveOp.emitError("expects 'base' when 'auto' is false");
  }

  int64_t signedBase = baseAttr.getInt();
  if (signedBase < 0) {
    return reserveOp.emitError(
        "expects 'base' to be non-negative when present");
  }

  uint64_t base = static_cast<uint64_t>(signedBase);
  if (base % spec.alignmentBytes != 0) {
    return reserveOp.emitError("expects 'base' to be aligned to ")
           << spec.alignmentBytes << " bytes for "
           << stringifyEnum(reserveOp.getLocation().getAddressSpace());
  }

  uint64_t size = static_cast<uint64_t>(reserveOp.getSize());
  if (base > spec.capacityBytes || size > spec.capacityBytes - base) {
    return reserveOp.emitError("reserved range exceeds ")
           << stringifyEnum(reserveOp.getLocation().getAddressSpace())
           << " capacity: base " << base << " + size " << size << " > "
           << spec.capacityBytes << " bytes";
  }

  return success();
}

static LogicalResult
validateReserveBufferForPlanMemory(pto::ReserveBufferOp reserveOp,
                                   PTOArch arch) {
  MemSpec spec = getMemSpec(arch, reserveOp.getLocation().getAddressSpace());

  if (reserveOp.getAutoAlloc()) {
    if (reserveOp.getBaseAttr()) {
      return reserveOp.emitError(
          "unexpected pre-populated 'base' on auto reserve_buffer before "
          "pto-plan-memory; omit 'base' and let pto-plan-memory assign it");
    }
    return success();
  }

  if (mlir::failed(validateManualReserveBufferBase(reserveOp, spec))) {
    return failure();
  }

  return reserveOp.emitError(
      "pto.reserve_buffer with explicit 'base' (auto = false) is not "
      "supported in PlanMemory; use --pto-level=level3 or set auto = true");
}

class AllocTileOpAddPlannedAddressPattern
    : public OpRewritePattern<pto::AllocTileOp> {
public:
  explicit AllocTileOpAddPlannedAddressPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocTileOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {}

  LogicalResult matchAndRewrite(pto::AllocTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddr()) {
      return failure();
    }

    auto tileType = dyn_cast<TileBufType>(op.getResult().getType());
    if (!tileType) {
      return failure();
    }

    auto it = buffer2Offsets.find(op.getResult());
    if (it == buffer2Offsets.end() || it->second.empty()) {
      return failure();
    }

    if (it->second.size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "single alloc_tile root expects exactly one planned address");
    }

    Value addr = rewriter.create<arith::ConstantIntOp>(op.getLoc(),
                                                       it->second.front(), 64);
    auto planned = rewriter.create<pto::AllocTileOp>(
        op.getLoc(), tileType, addr,
        op.getValidRow() ? op.getValidRow() : Value(),
        op.getValidCol() ? op.getValidCol() : Value());
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName().getValue() == "operandSegmentSizes") {
        continue;
      }
      planned->setAttr(attr.getName(), attr.getValue());
    }

    rewriter.replaceOp(op, planned.getResult());
    return success();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
};

class AllocMultiTileOpAddPlannedAddressesPattern
    : public OpRewritePattern<pto::AllocMultiTileOp> {
public:
  explicit AllocMultiTileOpAddPlannedAddressesPattern(
      MLIRContext *context,
      DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets)
      : OpRewritePattern<pto::AllocMultiTileOp>(context),
        buffer2Offsets(std::move(buffer2Offsets)) {}

  LogicalResult matchAndRewrite(pto::AllocMultiTileOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getAddr() || op->hasAttr(pto::kPtoMultiBufferAddrsAttrName)) {
      return failure();
    }

    auto it = buffer2Offsets.find(op.getResult());
    if (it == buffer2Offsets.end() || it->second.empty()) {
      return failure();
    }
    if (it->second.size() != op.getResult().getType().getCount()) {
      return rewriter.notifyMatchFailure(
          op, "planned address count does not match multi_tile_buf count");
    }

    SmallVector<int64_t> addrs;
    addrs.reserve(it->second.size());
    for (uint64_t offset : it->second) {
      addrs.push_back(static_cast<int64_t>(offset));
    }
    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr(pto::kPtoMultiBufferAddrsAttrName,
                  rewriter.getDenseI64ArrayAttr(addrs));
    });
    return success();
  }

private:
  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
};

static LogicalResult materializePlannedOffsets(
    func::FuncOp func, DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets) {
  RewritePatternSet patterns(func.getContext());
  patterns.add<AllocTileOpAddPlannedAddressPattern>(patterns.getContext(),
                                                    buffer2Offsets);
  patterns.add<AllocMultiTileOpAddPlannedAddressesPattern>(
      patterns.getContext(), buffer2Offsets);
  if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
    return failure();
  return success();
}

} // namespace

static LogicalResult runModernPlanMemory(func::FuncOp func,
                                         llvm::StringRef memMode,
                                         bool orderBySize) {
  if (!memMode.equals_insensitive("local")) {
    func.emitError("unsupported mem-mode '")
        << memMode << "'; only 'local' is currently implemented";
    return failure();
  }

  PlannerAnalysis analysis(func);
  analysis.walkRegion(func.getBody());
  if (analysis.failed) {
    return failure();
  }

  DenseMap<Value, SmallVector<uint64_t>> buffer2Offsets;
  llvm::MapVector<AddressSpace, SmallVector<RootInfo *>> rootsBySpace;
  for (RootInfo &info : analysis.roots) {
    rootsBySpace[info.space].push_back(&info);
  }

  for (auto &entry : rootsBySpace) {
    AddressSpace space = entry.first;
    SmallVector<RootInfo *> &roots = entry.second;
    MemSpec spec = getMemSpec(getTargetArch(func), space);

    bool sizeFirstForSpace = orderBySize && !isCubeLocalSpace(space);
    llvm::stable_sort(roots, [&](const RootInfo *lhs, const RootInfo *rhs) {
      if (sizeFirstForSpace && lhs->totalBytes != rhs->totalBytes) {
        return lhs->totalBytes > rhs->totalBytes;
      }
      if (lhs->allocIndex != rhs->allocIndex) {
        return lhs->allocIndex < rhs->allocIndex;
      }
      return lhs->stableOrder < rhs->stableOrder;
    });

    SmallVector<ReuseGroup> groups;
    uint64_t scopeRequiredBytes = 0;
    for (RootInfo *info : roots) {
      ReuseGroup *chosen =
          chooseReuseGroupByCost(*info, groups, analysis, spec);
      if (!chosen) {
        ReuseGroup group;
        group.space = space;
        group.alignmentBytes = info->alignmentBytes;
        groups.push_back(std::move(group));
        chosen = &groups.back();
      }

      chosen->members.push_back(info);
      chosen->sizeBytes = std::max(chosen->sizeBytes, info->totalBytes);
      chosen->alignmentBytes =
          std::max(chosen->alignmentBytes, info->alignmentBytes);
    }

    uint64_t cursor = 0;
    for (ReuseGroup &group : groups) {
      cursor = alignUp(cursor, group.alignmentBytes);
      group.offsetBytes = cursor;
      if (group.offsetBytes + group.sizeBytes > spec.capacityBytes) {
        scopeRequiredBytes = group.offsetBytes + group.sizeBytes;
        break;
      }

      for (RootInfo *member : group.members) {
        member->offsets = buildSlotOffsets(group.offsetBytes, member->slotBytes,
                                           member->slotCount);
      }
      scopeRequiredBytes =
          std::max(scopeRequiredBytes, group.offsetBytes + group.sizeBytes);
      cursor = group.offsetBytes + group.sizeBytes;
    }

    if (scopeRequiredBytes > spec.capacityBytes) {
      func.emitError() << stringifyEnum(space) << " overflow, requires "
                       << (scopeRequiredBytes * mlir::pto::kValue8) << " bits while "
                       << (spec.capacityBytes * mlir::pto::kValue8) << " bits available";
      return failure();
    }

    for (RootInfo *info : roots) {
      buffer2Offsets[info->root] = info->offsets;
    }
  }

  DenseMap<AddressSpace, SmallVector<std::pair<uint64_t, uint64_t>>>
      occupiedBySpace;
  for (const RootInfo &info : analysis.roots) {
    if (info.offsets.empty()) {
      continue;
    }
    occupiedBySpace[info.space].push_back(
        {info.offsets.front(), info.offsets.front() + info.totalBytes});
  }

  MLIRContext *ctx = func.getContext();
  PTOArch arch = getTargetArch(func);
  func.walk([&](pto::ReserveBufferOp reserveOp) -> WalkResult {
    if (mlir::failed(validateReserveBufferForPlanMemory(reserveOp, arch))) {
      analysis.failed = true;
      return WalkResult::interrupt();
    }

    AddressSpace space = reserveOp.getLocation().getAddressSpace();
    MemSpec spec = getMemSpec(arch, space);
    auto baseOr =
        planReserveBufferBase(reserveOp, spec, occupiedBySpace[space]);
    if (mlir::failed(baseOr)) {
      reserveOp.emitError("failed to reserve a local memory hole for "
                          "reserve_buffer");
      analysis.failed = true;
      return WalkResult::interrupt();
    }

    reserveOp->setAttr("base", IntegerAttr::get(IntegerType::get(ctx, mlir::pto::kValue32),
                                                static_cast<int32_t>(*baseOr)));
    return WalkResult::advance();
  });

  if (analysis.failed ||
      mlir::failed(materializePlannedOffsets(func, buffer2Offsets))) {
    return failure();
  }
  if (failed(verifySemanticNoAliasRanges(func)))
    return failure();

  bool hasUnplannedAllocTile = false;
  func.walk([&](pto::AllocTileOp op) {
    if (op.getAddr()) {
      return;
    }
    if (op->use_empty()) {
      return;
    }
    if (isA5IgnoredTmpAlloc(op)) {
      return;
    }
    op.emitError("PTOPlanMemory failed to assign an address to pto.alloc_tile");
    hasUnplannedAllocTile = true;
  });
  if (hasUnplannedAllocTile) {
    return failure();
  }
  return success();
}

namespace {
struct PlanMemoryModernPass
    : public PassWrapper<PlanMemoryModernPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanMemoryModernPass)

  PlanMemoryModernPass() = default;
  explicit PlanMemoryModernPass(const PlanMemoryOptions &options)
      : memMode(options.memMode), orderBySize(options.orderBySize) {}

  StringRef getArgument() const final { return "pto-plan-memory"; }
  StringRef getDescription() const final {
    return "Plan local memory using the modern PTO planner";
  }
  StringRef getName() const final { return "PlanMemoryModern"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<func::FuncOp> funcs;
    moduleOp.walk([&](func::FuncOp funcOp) {
      // `pto.tileop.helper` identifies compute-only helpers, not whether a
      // child module needs memory planning.  Skip those helpers themselves,
      // while planning every ordinary function in nested backend modules.
      if (!funcOp->hasAttr("pto.tileop.helper")) {
        funcs.push_back(funcOp);
      }
    });

    for (func::FuncOp funcOp : funcs) {
      if (failed(
              runModernPlanMemory(funcOp, memMode, orderBySize))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  std::string memMode = "local";
  bool orderBySize = false;
};
} // namespace

std::unique_ptr<Pass>
mlir::pto::createPlanMemoryModernPass(const PlanMemoryOptions &options) {
  return std::make_unique<PlanMemoryModernPass>(options);
}

// Anchor the generated pass base registration used by pto-test-opt and the
// textual pass pipeline (`-pass-pipeline=...`), replacing the removed legacy
// planner under the same `pto-plan-memory` argument.
namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PLANMEMORY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir
