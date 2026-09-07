// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOLayoutUtils.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "PTO/IR/PTOTypeUtils.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

// Forward declarations for custom shape/type printers used by tensor_view and
// partition_tensor_view.
namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic = true);
static LogicalResult parseViewShapeElemAndLayout(
    AsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elementType,
    Attribute &layout, bool allowDynamic = true);
static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType);
static void printViewShapeElemAndLayout(AsmPrinter &printer,
                                        ArrayRef<int64_t> shape,
                                        Type elementType, Attribute layout);
} // namespace pto
} // namespace mlir

// =============================================================================
// TileBufType 的自定义 Shape 解析与打印函数
// =============================================================================

// 解析逻辑：解析形如 "32x32" 的维度列表
[[maybe_unused]] static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 会解析 "dim x dim x ...", 遇到无法解析为维度的字符停止
  // 参数 allowDynamic=true (允许 ?), withTrailingX=false (不吞掉末尾的 x)
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/false)) {
    return failure();
  }
  return success();
}

// 打印逻辑：打印形如 "32x32" 的维度列表
[[maybe_unused]] static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) {
      printer << "x"; // 维度间的分隔符
    }
    if (*it == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << *it;
}
  }
  // 注意：我们不在这里打印末尾的 'x'，因为 assemblyFormat 中已经写了 `x` $elementType
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty);
enum class VerifierTargetArch {
  A2A3,
  A5,
};
static VerifierTargetArch getVerifierTargetArch(Operation *op);
static std::optional<StringRef> getVerifierArchName(Operation *op);
static bool isSupportedVecElemType(Type ty, bool allowBf16 = true,
                                   bool allowInt8 = true);
static bool isSupportedLoadStoreElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA2A3(Type ty);
static bool isSupportedGatherElemTypeA5(Type ty);
static bool isA5TLoadStoreTransferElemType(Type ty);
static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem);
static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem);
static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem);
static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op);
static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName);
static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName);
static bool isTileLikeType(Type ty);
static SmallVector<int64_t, 4> getShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Type ty);
static SmallVector<int64_t, 4> getValidShapeVec(Value value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool isByteIntegerType(Type ty);
static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision = false);
static LogicalResult verifyTmpCapacityAtLeast(Operation *op, Type tmpTy,
                                              uint64_t requiredBytes,
                                              StringRef tmpName = "tmp");

namespace {
struct PTOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
};
} // namespace
static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName);
static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape);

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName);
static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyVecTileStorage(Operation *op, Type ty,
                                          StringRef name);
static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty,
                                          StringRef name,
                                          bool allowLowPrecision = false);
static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc);
static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name);
static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName = "src",
                                          StringRef dstName = "dst",
                                          bool allowBf16 = true,
                                          bool allowInt8 = true);
static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name);
static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name);
static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name);
static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision = false);
static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision = false);
static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision = false);
static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy);
static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name);
static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy);
static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy);
static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias = false);
static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias = false);
static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy);
static std::optional<pto::Layout> getLogicalViewLayout(Value value);
static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type);
static std::optional<int64_t> getConstantIntegerValue(Value value);
static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy);
static Type getElemTy(Type ty);
static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty,
                                         Type src1Ty, Type dstTy);
static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual);
static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                  Type dstTy);
static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error);
static bool isRowMajorTileBuf(Type ty);
static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr);
static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr);
static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr);

#define GET_ENUM_CLASSES
#include "PTO/IR/PTOEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

[[maybe_unused]] static LogicalResult parseShapeAndElemStable(mlir::AsmParser &parser,
                                             llvm::SmallVectorImpl<int64_t> &shape,
                                             mlir::Type &elementType) {
  if (failed(parser.parseLess())) {
    return failure();
  }

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true))) {
    return failure();
  }

  if (failed(parser.parseType(elementType))) {
    return failure();
  }

  if (failed(parser.parseGreater())) {
    return failure();
  }

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank()) {
      return shapedTy.getRank();
    }
    return -1; // Unranked type
  }
  
  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type)) {
    return tvTy.getRank();
  }

  if (auto tileTy = dyn_cast<pto::TileType>(type)) {
    return tileTy.getRank();
  }

  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type)) {
    return tileViewTy.getRank();
  }

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type)) {
    return tileBufTy.getRank();
  }

  // 3. 不支持的类型
  return -1;
}

func::FuncOp mlir::pto::lookupPeerFuncAcrossContainer(Operation *op,
                                                      FlatSymbolRefAttr peerAttr) {
  if (!op || !peerAttr) {
    return {};
  }

  auto currentFunc = op->getParentOfType<func::FuncOp>();
  if (!currentFunc) {
    return {};
  }

  auto currentChildModule = currentFunc->getParentOfType<ModuleOp>();
  if (!currentChildModule) {
    return {};
  }

  StringRef target = peerAttr.getValue();
  for (func::FuncOp funcOp : currentChildModule.getOps<func::FuncOp>()) {
    if (funcOp.getSymName() == target) {
      return funcOp;
    }
  }
  if (auto localPeer = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupSymbolIn(currentChildModule, target))) {
    return localPeer;
  }

  Operation *maybeOuter = currentChildModule->getParentOp();
  auto outerModule = dyn_cast_or_null<ModuleOp>(maybeOuter);
  if (!outerModule) {
    return {};
  }

  SmallVector<func::FuncOp> fallbackMatches;
  outerModule.walk([&](func::FuncOp funcOp) {
    auto visibility = funcOp->getAttrOfType<StringAttr>("sym_visibility");
    if (visibility && visibility.getValue() == "private") {
      return WalkResult::advance();
    }

    StringRef symbolName = funcOp.getSymName();
    if (symbolName == target ||
        (funcOp->hasAttr(kPTODSLLogicalNameAttrName) &&
         getPTODSLLogicalNameOrSymbolName(funcOp) == target)) {
      fallbackMatches.push_back(funcOp);
    }
    return WalkResult::advance();
  });

  if (fallbackMatches.size() == 1) {
    return fallbackMatches.front();
  }
  return {};
}

static bool isA5DeviceSpec(StringRef spec) {
  return spec.starts_with("Ascend950") || spec.starts_with("Ascend910_95");
}

static bool isA5ModuleTarget(ModuleOp module) {
  if (!module) {
    return false;
  }
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
    if (arch.getValue().equals_insensitive("a5")) {
      return true;
    }
  }
  if (auto spec = module->getAttrOfType<StringAttr>("pto.device-spec")) {
    return isA5DeviceSpec(spec.getValue());
  }
  return false;
}

PTOArch mlir::pto::getTargetArch(ModuleOp module) {
  if (isA5ModuleTarget(module)) {
    return PTOArch::A5;
  }

  switch (getPTOParserTargetArch(module ? module.getContext() : nullptr)) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
}

PTOArch mlir::pto::getTargetArch(Operation *op) {
  if (!op) {
    return PTOArch::A3;
  }
  if (auto module = op->getParentOfType<ModuleOp>()) {
    return getTargetArch(module);
  }
  switch (getPTOParserTargetArch(op->getContext())) {
  case PTOParserTargetArch::A5:
    return PTOArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    break;
  }
  return PTOArch::A3;
}

bool mlir::pto::isTargetArchA3(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(ModuleOp module) {
  return getTargetArch(module) == PTOArch::A5;
}

bool mlir::pto::isTargetArchA3(Operation *op) {
  return getTargetArch(op) == PTOArch::A3;
}

bool mlir::pto::isTargetArchA5(Operation *op) {
  return getTargetArch(op) == PTOArch::A5;
}

constexpr int64_t kA5VectorLengthBytes = 256;

enum class PredicateLoadDist {
  Norm,
  Us,
  Ds,
};

enum class PredicateStoreDist {
  Norm,
  Pk,
};

struct PredicateLoadAlignmentRule {
  PredicateLoadDist dist;
  int64_t alignmentBytes;
};

struct PredicateStoreAlignmentRule {
  PredicateStoreDist dist;
  int64_t alignmentBytes;
};

constexpr PredicateLoadAlignmentRule kA5PredicateLoadAlignmentRules[] = {
    {PredicateLoadDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateLoadDist::Us, kA5VectorLengthBytes / 16},
    {PredicateLoadDist::Ds, std::min<int64_t>(32, kA5VectorLengthBytes / 4)},
};

constexpr PredicateStoreAlignmentRule kA5PredicateStoreAlignmentRules[] = {
    {PredicateStoreDist::Norm, kA5VectorLengthBytes / 8},
    {PredicateStoreDist::Pk, kA5VectorLengthBytes / 16},
};

static std::optional<PredicateLoadDist>
parsePredicateLoadDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateLoadDist::Norm;
  }
  if (dist == "US") {
    return PredicateLoadDist::Us;
  }
  if (dist == "DS") {
    return PredicateLoadDist::Ds;
  }
  return std::nullopt;
}

static std::optional<PredicateStoreDist>
parsePredicateStoreDist(StringRef dist) {
  if (dist == "NORM") {
    return PredicateStoreDist::Norm;
  }
  if (dist == "PK") {
    return PredicateStoreDist::Pk;
  }
  return std::nullopt;
}

template <typename Rule, typename Dist, size_t N>
static std::optional<int64_t> findAlignmentSize(const Rule (&rules)[N],
                                                Dist dist) {
  auto rule = llvm::find_if(
      rules, [&](const Rule &entry) { return entry.dist == dist; });
  if (rule == std::end(rules)) {
    return std::nullopt;
  }
  return rule->alignmentBytes;
}

std::optional<int64_t>
mlir::pto::getLoadStoreVecAlignmentSize(Operation *op) {
  if (!op || getTargetArch(op) != PTOArch::A5) {
    return std::nullopt;
  }

  if (auto pldi = dyn_cast<PldiOp>(op)) {
    auto dist = parsePredicateLoadDist(pldi.getDist());
    return dist ? findAlignmentSize(kA5PredicateLoadAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto psti = dyn_cast<PstiOp>(op)) {
    auto dist = parsePredicateStoreDist(psti.getDist());
    return dist ? findAlignmentSize(kA5PredicateStoreAlignmentRules, *dist)
                : std::nullopt;
  }
  if (auto sprsti = dyn_cast<SprstiOp>(op)) {
    if (sprsti.getSpr() == "AR") {
      return 4;
    }
  }
  return std::nullopt;
}

static llvm::TypeSize getOneByteTypeSize() {
  return llvm::TypeSize::getFixed(8);
}

llvm::TypeSize mlir::pto::HiF8Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

llvm::TypeSize mlir::pto::F8E8M0Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::HiF8Type::getABIAlignment(const DataLayout &,
                                              DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F8E8M0Type::getABIAlignment(const DataLayout &,
                                                DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::HiF8Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F8E8M0Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

static llvm::TypeSize getTwoByteTypeSize() {
  return llvm::TypeSize::getFixed(16);
}

llvm::TypeSize mlir::pto::HiF8x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getTwoByteTypeSize();
}

uint64_t mlir::pto::HiF8x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 2;
}

uint64_t mlir::pto::HiF8x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 2;
}

llvm::TypeSize mlir::pto::F4E1M2x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E1M2x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F4E1M2x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

llvm::TypeSize mlir::pto::F4E2M1x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getOneByteTypeSize();
}

uint64_t mlir::pto::F4E2M1x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

uint64_t mlir::pto::F4E2M1x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 1;
}

static llvm::TypeSize getFourByteTypeSize() {
  return llvm::TypeSize::getFixed(32);
}

llvm::TypeSize mlir::pto::BF16x2Type::getTypeSizeInBits(
    const DataLayout &, DataLayoutEntryListRef) const {
  return getFourByteTypeSize();
}

uint64_t mlir::pto::BF16x2Type::getABIAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 4;
}

uint64_t mlir::pto::BF16x2Type::getPreferredAlignment(
    const DataLayout &, DataLayoutEntryListRef) const {
  return 4;
}

static VerifierTargetArch getVerifierTargetArch(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (isA5ModuleTarget(module)) {
    return VerifierTargetArch::A5;
  }

  if (auto archName = getVerifierArchName(op)) {
    return archName->equals_insensitive("a5") ? VerifierTargetArch::A5
                            : VerifierTargetArch::A2A3;
  }

  switch (getPTOParserTargetArch(op ? op->getContext() : nullptr)) {
  case PTOParserTargetArch::A5:
    return VerifierTargetArch::A5;
  case PTOParserTargetArch::A3:
  case PTOParserTargetArch::Unspecified:
    return VerifierTargetArch::A2A3;
  }

  return VerifierTargetArch::A2A3;
}

static std::optional<StringRef> getVerifierArchName(Operation *op) {
  auto module = op ? op->getParentOfType<ModuleOp>() : ModuleOp();
  if (!module) {
    return std::nullopt;
  }
  if (auto arch = module->getAttrOfType<StringAttr>(kPTOTargetArchAttrName)) {
    return arch.getValue();
  }
  return std::nullopt;
}

static SmallVector<int64_t, 4> canonicalizeTileBufValidShape(ArrayRef<int64_t> validShape) {
  SmallVector<int64_t, 4> canonical;
  canonical.reserve(validShape.size());
  for (int64_t dim : validShape) {
    canonical.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return canonical;
}

template <typename FnA2A3, typename FnA5>
static LogicalResult dispatchVerifierByArch(Operation *op, FnA2A3 &&verifyA2A3,
                                            FnA5 &&verifyA5) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}
static std::optional<pto::AddressSpace> parsePtrAddressSpaceKeyword(StringRef keyword) {
  return llvm::StringSwitch<std::optional<pto::AddressSpace>>(keyword)
      .Case("gm", pto::AddressSpace::GM)
      .Case("mat", pto::AddressSpace::MAT)
      .Case("l1", pto::AddressSpace::MAT)
      .Case("left", pto::AddressSpace::LEFT)
      .Case("l0a", pto::AddressSpace::LEFT)
      .Case("right", pto::AddressSpace::RIGHT)
      .Case("l0b", pto::AddressSpace::RIGHT)
      .Case("acc", pto::AddressSpace::ACC)
      .Case("l0c", pto::AddressSpace::ACC)
      .Case("vec", pto::AddressSpace::VEC)
      .Case("ub", pto::AddressSpace::VEC)
      .Case("bias", pto::AddressSpace::BIAS)
      .Case("bt", pto::AddressSpace::BIAS)
      .Case("scaling", pto::AddressSpace::SCALING)
      .Case("fb", pto::AddressSpace::SCALING)
      .Default(std::nullopt);
}

static StringRef printPtrAddressSpaceKeyword(pto::AddressSpace space) {
  switch (space) {
  case pto::AddressSpace::GM:
  case pto::AddressSpace::Zero:
    return "gm";
  case pto::AddressSpace::MAT:
    return "l1";
  case pto::AddressSpace::LEFT:
    return "l0a";
  case pto::AddressSpace::RIGHT:
    return "l0b";
  case pto::AddressSpace::ACC:
    return "l0c";
  case pto::AddressSpace::VEC:
    return "ub";
  case pto::AddressSpace::BIAS:
    return "bt";
  case pto::AddressSpace::SCALING:
    return "fb";
  }
  llvm_unreachable("unhandled pointer address space");
}

static ParseResult parseSyncEventOpCommon(OpAsmParser &parser,
                                          OperationState &result,
                                          StringAttr pipeAttrName,
                                          StringAttr eventIdAttrName) {
  PipeAttr pipeAttr;
  if (succeeded(parser.parseOptionalLess())) {
    StringRef pipeTok;
    if (parser.parseKeyword(&pipeTok) || parser.parseGreater()) {
      return failure();
    }
    auto pipeOr = symbolizePIPE(pipeTok);
    if (!pipeOr) {
      return parser.emitError(parser.getCurrentLocation())
             << "unknown pipe token: " << pipeTok;
    }
    pipeAttr = PipeAttr::get(parser.getContext(), *pipeOr);
    result.addAttribute(pipeAttrName, pipeAttr);
  } else if (parser.parseAttribute(pipeAttr, pipeAttrName,
                                   result.attributes)) {
    return failure();
  }
  if (parser.parseComma()) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand eventOperand;
  OptionalParseResult parseEventOperand =
      parser.parseOptionalOperand(eventOperand);
  if (parseEventOperand.has_value()) {
    if (failed(*parseEventOperand)) {
      return failure();
    }
    if (parser.resolveOperand(eventOperand, parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }
  } else {
    IntegerAttr eventAttr;
    if (parser.parseAttribute(eventAttr, parser.getBuilder().getI32Type(),
                              eventIdAttrName, result.attributes)) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

static void printSyncEventOpCommon(OpAsmPrinter &p, Operation *op,
                                   PipeAttr pipeAttr, IntegerAttr eventAttr,
                                   Value eventDyn, StringRef pipeAttrName,
                                   StringRef eventIdAttrName) {
  p << " <" << stringifyPIPE(pipeAttr.getPipe()) << ">, ";
  if (eventAttr) {
    p << eventAttr.getInt();
  } else {
    p << eventDyn;
}
  p.printOptionalAttrDict(op->getAttrs(), {pipeAttrName, eventIdAttrName});
}

[[maybe_unused]] static mlir::Type parsePTOTypeAllowNoBang(mlir::OpAsmParser &parser) {
  mlir::Type ty;

  mlir::OptionalParseResult opt = parser.parseOptionalType(ty);

  if (opt.has_value()) {
    if (failed(*opt)) {
      return mlir::Type();
    }
    return ty;                    
  }


  llvm::StringRef head;
  if (failed(parser.parseKeyword(&head))) {
    return mlir::Type();
  }

  mlir::MLIRContext *ctx = parser.getContext();

  auto parseShapeElemForOpParser =
      [&](llvm::SmallVectorImpl<int64_t> &shape, mlir::Type &elem) -> mlir::LogicalResult {
    if (failed(parser.parseLess())) {
      return failure();
    }
    if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true))) {
      return failure();
    }
    if (failed(parser.parseType(elem))) {
      return failure();
    }
    if (failed(parser.parseGreater())) {
      return failure();
    }
    return success();
  };

  if (head == "pto.tile_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem))) {
      return mlir::Type();
    }
    return mlir::pto::PartitionTensorViewType::get(ctx, shape, elem);
  }

  if (head == "pto.tile") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem))) {
      return mlir::Type();
    }
    return mlir::pto::TileType::get(ctx, shape, elem);
  }

  if (head == "pto.ptr") {
    if (failed(parser.parseLess())) {
      return mlir::Type();
    }
    mlir::Type elem;
    if (failed(parser.parseType(elem))) {
      return mlir::Type();
    }
    auto memorySpace = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::GM);
    if (succeeded(parser.parseOptionalComma())) {
      StringRef memorySpaceKeyword;
      if (failed(parser.parseKeyword(&memorySpaceKeyword))) {
        return mlir::Type();
      }
      auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
      if (!parsed) {
        parser.emitError(parser.getCurrentLocation(),
                         "!pto.ptr address space must be one of "
                         "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
        return mlir::Type();
      }
      memorySpace = pto::AddressSpaceAttr::get(ctx, *parsed);
    }
    if (failed(parser.parseGreater())) {
      return mlir::Type();
    }
    return mlir::pto::PtrType::get(ctx, elem, memorySpace);
  }

  if (head == "pto.tensor_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem))) {
      return mlir::Type();
    }
    return mlir::pto::TensorViewType::get(ctx, shape, elem);
  }

  return mlir::Type();
}

mlir::Type TensorViewType::parse(::mlir::AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  Attribute layout;
  if (failed(parseViewShapeElemAndLayout(
          parser, shape, elementType, layout, /*allowDynamic=*/true))) {
    return Type();
  }
  return TensorViewType::get(parser.getContext(), shape, elementType, layout);
}

void TensorViewType::print(::mlir::AsmPrinter &printer) const {
  printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                              getLayout());
}

mlir::Type PtrType::parse(::mlir::AsmParser &parser) {
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseType(elementType))) {
    return {};
  }

  auto memorySpace =
      pto::AddressSpaceAttr::get(parser.getContext(), pto::AddressSpace::GM);
  if (succeeded(parser.parseOptionalComma())) {
    StringRef memorySpaceKeyword;
    if (failed(parser.parseKeyword(&memorySpaceKeyword))) {
      return {};
    }
    auto parsed = parsePtrAddressSpaceKeyword(memorySpaceKeyword);
    if (!parsed) {
      parser.emitError(parser.getCurrentLocation(),
                       "!pto.ptr address space must be one of "
                       "`gm|ub|mat|l1|left|l0a|right|l0b|acc|l0c|vec|bias|bt|scaling|fb`");
      return {};
    }
    memorySpace = pto::AddressSpaceAttr::get(parser.getContext(), *parsed);
  }

  if (failed(parser.parseGreater())) {
    return {};
  }
  return PtrType::get(parser.getContext(), elementType, memorySpace);
}

void PtrType::print(::mlir::AsmPrinter &printer) const {
  printer << "<" << getElementType();
  StringRef memorySpaceKeyword =
      printPtrAddressSpaceKeyword(getMemorySpace().getAddressSpace());
  if (!memorySpaceKeyword.empty()) {
    printer << ", " << memorySpaceKeyword;
  }
  printer << ">";
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op follows textual input order.
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op0, op1, dst;
  Type ty0, ty1, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(op0) || parser.parseComma() ||
      parser.parseOperand(op1) || parser.parseColonType(ty0) ||
      parser.parseComma() || parser.parseType(ty1) || parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
  auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
  if ((tile0 && tile1) || (!tile0 && !tile1)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one tile_buf operand and one scalar operand");
  }

  if (!dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected outs type to be !pto.tile_buf<...>");
  }

  // Keep textual order so later lowering can distinguish the two APIs by the
  // first ins operand type.
  if (parser.resolveOperand(op0, ty0, result.operands) ||
      parser.resolveOperand(op1, ty1, result.operands)) {
    return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  result.addAttributes(attrs);
  return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter &p) {
  p << " ins(";
  p << getSrc() << ", " << getScalar() << " : "
    << getSrc().getType() << ", " << getScalar().getType();
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  p.printOptionalAttrDict((*this)->getAttrs());
}


//===----------------------------------------------------------------------===//
// pto.tgather custom asm supports three PTO-ISA forms:
//   1) index+tmp   : ins(%src, %indices, %tmp : srcTy, indicesTy, tmpTy) outs(%dst : dstTy)
//   2) compare+tmp : ins(%src, %kValue, %tmp : srcTy, scalarTy, tmpTy)
//                    outs(%dst, %cdst : dstTy, cdstTy) {cmpMode = #pto.cmp<gt>, offset = 7}
//   3) mask        : ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : srcTy) outs(%dst : dstTy)
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TGatherOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, dst, cdst;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOps;
  SmallVector<Type, 3> insTypes;
  Type srcTy, dstTy, cdstTy;
  bool hasCdst = false;
  bool hasMask = false;
  bool hasIndices = false;
  bool hasTmp = false;
  bool hasKValue = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }

  if (!succeeded(parser.parseOptionalComma())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");
  }

  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("maskPattern") || parser.parseEqual()) {
      return failure();
    }

    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr) || parser.parseRBrace()) {
      return failure();
    }

    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }

    result.addAttribute("maskPattern", mp);
    hasMask = true;

    if (parser.parseColonType(srcTy)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      StringAttr axisAttr;
      if (parser.parseAttribute(axisAttr)) {
        return failure();
      }
      if (axisAttr.getValue() != "row" && axisAttr.getValue() != "col") {
        return parser.emitError(parser.getCurrentLocation(),
                                "axis must be \"row\" or \"col\"");
      }
      result.addAttribute("axis", axisAttr);
    }
    if (parser.parseRParen()) {
      return failure();
    }
  } else {
    OpAsmParser::UnresolvedOperand extra;
    if (parser.parseOperand(extra)) {
      return failure();
    }
    insOps.push_back(extra);
    while (succeeded(parser.parseOptionalComma())) {
      if (insOps.size() == 3) {
        return parser.emitError(parser.getCurrentLocation(),
                                "expected at most 3 extra operands in tgather ins(...)");
      }
      if (parser.parseOperand(extra)) {
        return failure();
      }
      insOps.push_back(extra);
    }

    if (parser.parseColon() || parser.parseType(srcTy)) {
      return failure();
    }
    for (size_t i = 0; i < insOps.size(); ++i) {
      Type ty;
      if (parser.parseComma() || parser.parseType(ty)) {
        return failure();
      }
      insTypes.push_back(ty);
    }
    if (parser.parseRParen()) {
      return failure();
    }
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() || parser.parseOperand(dst)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(cdst)) {
      return failure();
    }
    hasCdst = true;
  }
  if (parser.parseColonType(dstTy)) {
    return failure();
  }
  if (hasCdst && (parser.parseComma() || parser.parseType(cdstTy))) {
    return failure();
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("maskPattern"))) {
    if (hasMask) {
      return parser.emitError(parser.getCurrentLocation(),
                              "maskPattern may only be specified once");
    }
    if (parser.parseEqual()) {
      return failure();
    }
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr)) {
      return failure();
    }
    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }
    result.addAttribute("maskPattern", mp);
    hasMask = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (hasMask) {
    if (!insOps.empty()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "mask-pattern tgather does not take extra ins operands");
    }
    if (hasCdst) {
      return parser.emitError(parser.getCurrentLocation(),
                              "mask-pattern tgather expects a single outs operand");
    }
  } else if (hasCdst) {
    if (insOps.empty() ||
        !(mlir::isa<IntegerType>(insTypes.front()) ||
          mlir::isa<FloatType>(insTypes.front()))) {
      return parser.emitError(parser.getCurrentLocation(),
                              "compare-form tgather expects a scalar kValue operand");
    }
    hasKValue = true;
    if (insOps.size() >= 2) {
      if (!isTileLikeType(insTypes[1])) {
        return parser.emitError(parser.getCurrentLocation(),
                                "compare-form tgather tmp must be tile-like");
      }
      hasTmp = true;
    }
    if (insOps.size() == 3) {
      return parser.emitError(parser.getCurrentLocation(),
                              "compare-form tgather expects at most src, kValue, tmp in ins(...)");
    }
  } else {
    if (!insOps.empty() && !isTileLikeType(insTypes.front())) {
      return parser.emitError(parser.getCurrentLocation(),
                              "index-form tgather expects tile-like indices; "
                              "compare-form must use outs(dst, cdst)");
    }
    if (!insOps.empty()) {
      hasIndices = true;
      if (insOps.size() >= 2) {
        if (!isTileLikeType(insTypes[1])) {
          return parser.emitError(parser.getCurrentLocation(),
                                  "index-form tgather tmp must be tile-like");
        }
        hasTmp = true;
      }
    }
    if (insOps.size() == 3) {
      return parser.emitError(parser.getCurrentLocation(),
                              "index-form tgather expects at most src, indices, tmp in ins(...)");
    }
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasCdst && parser.resolveOperand(cdst, cdstTy, result.operands)) {
    return failure();
  }
  if (hasIndices && parser.resolveOperand(insOps[0], insTypes[0], result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(insOps[hasIndices ? 1 : 1], insTypes[1], result.operands)) {
    return failure();
  }
  if (hasKValue && parser.resolveOperand(insOps[0], insTypes[0], result.operands)) {
    return failure();
  }

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, 1, hasCdst ? 1 : 0, hasIndices ? 1 : 0,
                           hasTmp ? 1 : 0, hasKValue ? 1 : 0}));
  return success();
}

void mlir::pto::TGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (auto mp = getMaskPatternAttr()) {
    p << "{maskPattern = " << mp << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
  } else if (getCdst()) {
    p << getKValue();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getKValue().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getKValue().getType();
    }
  } else {
    p << getIndices();
    if (getTmp()) {
      p << ", " << getTmp();
      p << " : " << getSrc().getType() << ", " << getIndices().getType()
        << ", " << getTmp().getType();
    } else {
      p << " : " << getSrc().getType() << ", " << getIndices().getType();
    }
  }
  p << ") outs(" << getDst();
  if (getCdst()) {
    p << ", " << getCdst();
  }
  p << " : " << getDst().getType();
  if (getCdst()) {
    p << ", " << getCdst().getType();
  }
  p << ")";

  if (getMaskPatternAttr()) {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"maskPattern", "axis", "operandSegmentSizes"});
  } else {
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"axis", "operandSegmentSizes"});
  }
}

ParseResult mlir::pto::TScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src, indexes, dst;
  Type srcTy, idxTy, dstTy;
  bool hasMask = false;
  bool hasIndexes = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src)) {
    return failure();
  }

  if (!succeeded(parser.parseOptionalComma())) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected ',' after src operand in ins(...)");
  }

  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("maskPattern") || parser.parseEqual()) {
      return failure();
    }
    Attribute rawMaskAttr;
    if (parser.parseAttribute(rawMaskAttr)) {
      return failure();
    }
    auto mp = llvm::dyn_cast<mlir::pto::MaskPatternAttr>(rawMaskAttr);
    if (!mp) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected #pto.mask_pattern<Pxxxx> for maskPattern");
    }
    result.addAttribute("maskPattern", mp);
    hasMask = true;
    if (parser.parseRBrace() || parser.parseColonType(srcTy)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      StringAttr axisAttr;
      if (parser.parseAttribute(axisAttr)) {
        return failure();
      }
      if (axisAttr.getValue() != "row" && axisAttr.getValue() != "col") {
        return parser.emitError(parser.getCurrentLocation(),
                                "axis must be \"row\" or \"col\"");
      }
      result.addAttribute("axis", axisAttr);
    }
    if (parser.parseRParen()) {
      return failure();
    }
  } else {
    if (parser.parseOperand(indexes)) {
      return failure();
    }
    hasIndexes = true;
    if (parser.parseColon() || parser.parseType(srcTy) || parser.parseComma() ||
        parser.parseType(idxTy) || parser.parseRParen()) {
      return failure();
    }
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (result.attributes.get("maskPattern")) {
    hasMask = true;
  }

  if (hasMask && hasIndexes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "mask-pattern tscatter does not take indexes");
  }
  if (!hasMask && !hasIndexes) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected indexes operand or maskPattern for tscatter");
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands) ||
      (hasIndexes && parser.resolveOperand(indexes, idxTy, result.operands))) {
    return failure();
  }

  return success();
}

void mlir::pto::TScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", ";
  if (getMaskPatternAttr()) {
    p << "{maskPattern = " << getMaskPatternAttr() << "} : " << getSrc().getType();
    if (auto axisAttr = getAxisAttr()) {
      p << ", " << axisAttr;
    }
  } else {
    p << getIndexes() << " : " << getSrc().getType() << ", "
      << getIndexes().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"maskPattern", "axis"});
}

namespace {
struct CommRecvClause {
  OpAsmParser::UnresolvedOperand ping;
  std::optional<OpAsmParser::UnresolvedOperand> pong;
  Type pingTy;
  Type pongTy;
};

static ParseResult parseCommRecvClause(OpAsmParser &parser,
                                       CommRecvClause &recvClause) {
  if (parser.parseKeyword("recv") || parser.parseLParen() ||
      parser.parseOperand(recvClause.ping)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::UnresolvedOperand pong;
    if (parser.parseOperand(pong)) {
      return failure();
    }
    recvClause.pong = pong;
  }
  return parser.parseRParen();
}

static ParseResult parseCommCollectiveTail(
    OpAsmParser &parser, OperationState &result,
    ArrayRef<OpAsmParser::UnresolvedOperand> fixedOperands,
    SmallVectorImpl<Type> &fixedTypes, CommRecvClause &recvClause,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &groupOps,
    SmallVectorImpl<Type> &groupTypes, ArrayRef<int32_t> operandSegmentsPrefix,
    ArrayRef<StringRef> requiredAttrs) {
  if (parser.parseComma() || parser.parseKeyword("group") || parser.parseLParen()) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand group;
  if (parser.parseOperand(group)) {
    return failure();
  }
  groupOps.push_back(group);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(group)) {
      return failure();
    }
    groupOps.push_back(group);
  }

  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseColon()) {
    return failure();
  }

  for (size_t i = 0; i < fixedTypes.size(); ++i) {
    if (i != 0 && parser.parseComma()) {
      return failure();
    }
    if (parser.parseType(fixedTypes[i])) {
      return failure();
    }
  }
  if (parser.parseComma() || parser.parseType(recvClause.pingTy)) {
    return failure();
  }
  if (recvClause.pong) {
    if (parser.parseComma() || parser.parseType(recvClause.pongTy)) {
      return failure();
    }
  }
  for (size_t i = 0; i < groupOps.size(); ++i) {
    Type groupTy;
    if (parser.parseComma() || parser.parseType(groupTy)) {
      return failure();
    }
    groupTypes.push_back(groupTy);
  }
  if (parser.parseRParen()) {
    return failure();
  }

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }
  for (StringRef attrName : requiredAttrs) {
    if (!attrs.get(attrName)) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected '" << attrName << "' attribute";
    }
  }
  result.addAttributes(attrs);

  for (auto [operand, type] : llvm::zip_equal(fixedOperands, fixedTypes)) {
    if (parser.resolveOperand(operand, type, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(recvClause.ping, recvClause.pingTy, result.operands)) {
    return failure();
  }
  if (recvClause.pong &&
      parser.resolveOperand(*recvClause.pong, recvClause.pongTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(groupOps, groupTypes, parser.getCurrentLocation(),
                             result.operands)) {
    return failure();
  }

  SmallVector<int32_t, 5> segmentSizes(operandSegmentsPrefix.begin(),
                                       operandSegmentsPrefix.end());
  segmentSizes.push_back(static_cast<int32_t>(groupOps.size()));
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

static void printCommRecvClause(OpAsmPrinter &p, Value ping, Value pong) {
  p << "recv(" << ping;
  if (pong) {
    p << ", " << pong;
  }
  p << ")";
}

static void printCommGroupTypes(OpAsmPrinter &p, ValueRange group) {
  for (Value groupValue : group) {
    p << ", " << groupValue.getType();
  }
}

static void printCommGroupClause(OpAsmPrinter &p, ValueRange group) {
  p << "group(";
  p.printOperands(group);
  p << ")";
}

} // namespace

ParseResult mlir::pto::TBroadcastOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(parser, result, fixedOperands, fixedTypes,
                                     recvClause, groupOps, groupTypes,
                                     {1, 1, recvClause.pong ? 1 : 0}, {"root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::TBroadcastOp::print(OpAsmPrinter &p) {
  p << "(" << getSrc() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getSrc().getType() << ", " << getPing().getType();
  if (getPong()) {
    p << ", " << getPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::CommTGatherOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
  OpAsmParser::UnresolvedOperand dst;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{dst};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, recvClause.pong ? 1 : 0},
          {"root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::CommTGatherOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getPing().getType();
  if (getPong()) {
    p << ", " << getPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::CommTScatterOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(src) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 1> fixedOperands{src};
  SmallVector<Type, 1> fixedTypes(1);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, recvClause.pong ? 1 : 0},
          {"root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::CommTScatterOp::print(OpAsmPrinter &p) {
  p << "(" << getSrc() << ", ";
  printCommRecvClause(p, getPing(), getPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getSrc().getType() << ", " << getPing().getType();
  if (getPong()) {
    p << ", " << getPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TReduceOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand dst, acc;
  CommRecvClause recvClause;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> groupOps;
  SmallVector<Type, 4> groupTypes;

  if (parser.parseLParen() || parser.parseOperand(dst) || parser.parseComma() ||
      parser.parseOperand(acc) || parser.parseComma()) {
    return failure();
  }
  if (failed(parseCommRecvClause(parser, recvClause))) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 2> fixedOperands{dst, acc};
  SmallVector<Type, 2> fixedTypes(2);
  if (failed(parseCommCollectiveTail(
          parser, result, fixedOperands, fixedTypes, recvClause, groupOps,
          groupTypes, {1, 1, 1, recvClause.pong ? 1 : 0},
          {"reduceOp", "root"}))) {
    return failure();
  }
  return success();
}

void mlir::pto::TReduceOp::print(OpAsmPrinter &p) {
  p << "(" << getDst() << ", " << getAcc() << ", ";
  printCommRecvClause(p, getRecvPing(), getRecvPong());
  p << ", ";
  printCommGroupClause(p, getGroup());
  p << " : " << getDst().getType() << ", " << getAcc().getType() << ", "
    << getRecvPing().getType();
  if (getRecvPong()) {
    p << ", " << getRecvPong().getType();
  }
  printCommGroupTypes(p, getGroup());
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr)) {
    return failure();
  }

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // strides = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare()) {
    return failure();
  }

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // : result-type
  if (parser.parseColonType(resultTy)) {
    return failure();
  }
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");
  }

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);

  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands)) {
    return failure();
  }

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperands(strideOps, indexTy, result.operands)) {
    return failure();
  }

  auto segAttr = parser.getBuilder().getDenseI32ArrayAttr(
      {1, (int32_t)shapeOps.size(), (int32_t)strideOps.size()});
  result.addAttribute("operandSegmentSizes", segAttr);

  return success();
}

void mlir::pto::MakeTensorViewOp::print(OpAsmPrinter &p) {
  p << " " << getPtr();

  p << ", shape = [";
  p.printOperands(getShape());
  p << "]";

  p << ", strides = [";
  p.printOperands(getStrides());
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                        /*elidedAttrs=*/{"operandSegmentSizes"});

  p << " : " << getResult().getType();
}

// Layout inference helpers for make_tensor_view
static std::optional<int64_t> getConstIndexValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static FailureOr<mlir::pto::PartitionTensorViewType>
inferPartitionViewResultTypeFromSizes(Type sourceType, ValueRange sizes) {
  int64_t sourceRank = 0;
  Type elementType;
  Attribute layout;
  if (auto tensorView = dyn_cast<mlir::pto::TensorViewType>(sourceType)) {
    sourceRank = tensorView.getRank();
    elementType = tensorView.getElementType();
    layout = tensorView.getLayout();
  } else if (auto partitionView =
                 dyn_cast<mlir::pto::PartitionTensorViewType>(sourceType)) {
    sourceRank = partitionView.getRank();
    elementType = partitionView.getElementType();
    layout = partitionView.getLayout();
  } else {
    return failure();
  }

  if ((int64_t)sizes.size() != sourceRank) {
    return failure();
  }

  SmallVector<int64_t, 4> shape;
  shape.reserve(sizes.size());
  for (Value size : sizes) {
    auto constSize = getConstIndexValue(size);
    if (constSize && *constSize >= 0) {
      shape.push_back(*constSize);
    } else {
      shape.push_back(ShapedType::kDynamic);
}
  }

  return mlir::pto::PartitionTensorViewType::get(sourceType.getContext(), shape,
                                                 elementType, layout);
}

ParseResult mlir::pto::PartitionViewOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> sizes;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseKeyword("offsets") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(offsets) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseKeyword("sizes") || parser.parseEqual() ||
      parser.parseLSquare() || parser.parseOperandList(sizes) ||
      parser.parseRSquare() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy)) {
      return failure();
    }
    hasExplicitResultTy = true;
  }

  if (parser.resolveOperand(source, sourceTy, result.operands)) {
    return failure();
  }

  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsets, indexTy, result.operands) ||
      parser.resolveOperands(sizes, indexTy, result.operands)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<PartitionViewOp::Properties>();
  llvm::copy(ArrayRef<int32_t>(
                 {1, static_cast<int32_t>(offsets.size()),
                  static_cast<int32_t>(sizes.size())}),
             properties.operandSegmentSizes.begin());

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  ValueRange allOperands(result.operands);
  ValueRange sizeOperands =
      allOperands.slice(1 + offsets.size(), sizes.size());
  auto inferredResultType =
      inferPartitionViewResultTypeFromSizes(sourceTy, sizeOperands);
  if (failed(inferredResultType)) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.partition_view result type");
  }

  result.addTypes(*inferredResultType);
  return success();
}

void mlir::pto::PartitionViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << ", offsets = [";
  printer.printOperands(getOffsets());
  printer << "], sizes = [";
  printer.printOperands(getSizes());
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getSource().getType();

  auto inferredResultType =
      inferPartitionViewResultTypeFromSizes(getSource().getType(), getSizes());
  if (succeeded(inferredResultType) && *inferredResultType == getResult().getType()) {
    return;
  }

  printer << " -> " << getResult().getType();
}

static std::optional<int64_t> getConstantIntegerValueEx(
    Value v, bool includeIndexAndIntOpsInConstFold) {
  if (includeIndexAndIntOpsInConstFold) {
    if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
      return c.value();
    }
    if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
      return c.value();
    }
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static LogicalResult verifyNonNegativeIndexRowCol(
    Operation &op, Value indexRow, Value indexCol,
    bool includeIndexAndIntOpsInConstFold) {
  if (!indexRow.getType().isIndex() || !indexCol.getType().isIndex()) {
    return op.emitOpError("expects indexRow and indexCol to be index type");
  }
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  if (row && *row < 0) {
    return op.emitOpError("expects indexRow to be non-negative");
  }
  if (col && *col < 0) {
    return op.emitOpError("expects indexCol to be non-negative");
  }
  return success();
}

static LogicalResult verifyExtractStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  }
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + dstShape[0] > srcShape[0]) {
    return op.emitOpError("expects indexRow + dst.rows <= src.rows");
  }
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + dstShape[1] > srcShape[1]) {
    return op.emitOpError("expects indexCol + dst.cols <= src.cols");
  }
  return success();
}

static LogicalResult verifyInsertStaticBoundsCommon(
    Operation &op, Value indexRow, Value indexCol, Type srcTy, Type dstTy,
    bool includeIndexAndIntOpsInConstFold) {
  auto row =
      getConstantIntegerValueEx(indexRow, includeIndexAndIntOpsInConstFold);
  auto col =
      getConstantIntegerValueEx(indexCol, includeIndexAndIntOpsInConstFold);
  auto srcShape = getValidShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op.emitOpError("expects src and dst to be rank-2 tile_buf");
  }
  if (row && srcShape[0] != ShapedType::kDynamic &&
      dstShape[0] != ShapedType::kDynamic &&
      *row + srcShape[0] > dstShape[0]) {
    return op.emitOpError("expects indexRow + src.rows <= dst.rows");
  }
  if (col && srcShape[1] != ShapedType::kDynamic &&
      dstShape[1] != ShapedType::kDynamic &&
      *col + srcShape[1] > dstShape[1]) {
    return op.emitOpError("expects indexCol + src.cols <= dst.cols");
  }
  return success();
}

static unsigned getElemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static LogicalResult verifyTileBufLayoutConstraints(Operation *op,
                                                    pto::TileBufType tb,
                                                    StringRef name) {
  auto shape = tb.getShape();
  if (shape.size() != 2) {
    return op->emitOpError() << "expects " << name << " to be rank-2";
  }

  int64_t rows = shape[0];
  int64_t cols = shape[1];
  if (rows != ShapedType::kDynamic && rows <= 0) {
    return op->emitOpError() << "expects " << name << " rows to be positive";
  }
  if (cols != ShapedType::kDynamic && cols <= 0) {
    return op->emitOpError() << "expects " << name << " cols to be positive";
  }

  unsigned elemBytes = getElemByteSize(tb.getElementType());
  if (elemBytes == 0) {
    return op->emitOpError() << "expects " << name
                             << " element type to have a byte size";
  }

  auto cfg = tb.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(tb.getContext());
  }
  auto readBLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<BLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  auto readSLayout = [](Attribute attr, int32_t &out) -> bool {
    if (auto layout = dyn_cast_or_null<SLayoutAttr>(attr)) {
      out = static_cast<int32_t>(layout.getValue());
      return true;
    }
    if (auto value = dyn_cast_or_null<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(value.getInt());
      return true;
    }
    return false;
  };
  int32_t blayout = 0;
  int32_t slayout = 0;
  if (!readBLayout(cfg.getBLayout(), blayout) ||
      !readSLayout(cfg.getSLayout(), slayout)) {
    return op->emitOpError() << "expects " << name
                             << " to have concrete tile layout attributes";
  }
  constexpr int64_t kAlignedBytes = 32;
  constexpr int64_t kPackedFp4AlignedBytes = 16;
  const int64_t requiredAlignment = isPTOFloat4PackedType(tb.getElementType())
                                        ? kPackedFp4AlignedBytes
                                        : kAlignedBytes;

  auto checkByteAlignment = [&](int64_t dim, StringRef layoutName,
                                StringRef byteExpr) -> LogicalResult {
    if (dim == ShapedType::kDynamic) {
      return success();
    }
    int64_t bytes = dim * static_cast<int64_t>(elemBytes);
    if (bytes % requiredAlignment == 0)
      return success();
    return op->emitOpError()
           << "expects " << name << " " << layoutName
           << " none_box tile " << byteExpr
           << " to be " << requiredAlignment << "-byte aligned, but got "
           << bytes << " bytes";
  };

  if (slayout == static_cast<int32_t>(SLayout::NoneBox)) {
    if (blayout == static_cast<int32_t>(BLayout::RowMajor)) {
      return checkByteAlignment(cols, "row-major",
                                "row byte size (cols * sizeof(dtype))");
    }
    return checkByteAlignment(rows, "col-major",
                              "column byte size (rows * sizeof(dtype))");
  }

  int64_t innerRows = 0;
  int64_t innerCols = 0;
  int32_t fractal = static_cast<int32_t>(cfg.getSFractalSize().getInt());
  switch (fractal) {
  case 1024:
    innerRows = 16;
    innerCols = 16;
    break;
  case 32:
    innerRows = 16;
    innerCols = 2;
    break;
  case 512:
    if (kAlignedBytes % elemBytes != 0) {
      return op->emitOpError() << "expects " << name
                               << " element byte size to divide 32 for boxed "
                                  "fractal-512 tile layout";
    }
    if (slayout == static_cast<int32_t>(SLayout::RowMajor)) {
      innerRows = 16;
      innerCols = kAlignedBytes / static_cast<int64_t>(elemBytes);
    } else if (slayout == static_cast<int32_t>(SLayout::ColMajor)) {
      innerRows = kAlignedBytes / static_cast<int64_t>(elemBytes);
      innerCols = 16;
    }
    break;
  default:
    break;
  }
  if (innerRows <= 0 || innerCols <= 0) {
    return op->emitOpError() << "expects " << name
                             << " to use a supported boxed tile layout";
  }

  auto loc = getPTOMemorySpaceEnum(tb);
  bool allowUnalignedRows =
      (loc && *loc == pto::AddressSpace::VEC) || fractal == 32 || rows == 1;
  if (!allowUnalignedRows && rows != ShapedType::kDynamic &&
      rows % innerRows != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile rows to be a multiple of innerRows (" << innerRows
           << "), but got " << rows;
  }
  if (cols != ShapedType::kDynamic && cols % innerCols != 0) {
    return op->emitOpError()
           << "expects " << name
           << " boxed tile cols to be a multiple of innerCols (" << innerCols
           << "), but got " << cols;
  }

  return success();
}

[[maybe_unused]] static bool isSupportedLoadStoreElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isBF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32 || width == 64;
  }
  return false;
}

static bool isSupportedGatherElemTypeA2A3(Type ty) {
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 16 || width == 32;
  }
  return false;
}

static bool isSupportedGatherElemTypeA5(Type ty) {
  if (isSupportedGatherElemTypeA2A3(ty) || ty.isBF16()) {
    return true;
  }
  if (isPTOHiFloat8Type(ty)) {
    return true;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    unsigned width = ft.getWidth();
    return width == 8;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  }
  return false;
}

static std::optional<pto::Layout> getLogicalViewLayout(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto type = dyn_cast<pto::TensorViewType>(value.getType())) {
    if (auto layout = type.getLayoutAttr()) {
      return layout.getLayout();
    }
  } else if (auto type =
                 dyn_cast<pto::PartitionTensorViewType>(value.getType())) {
    if (auto layout = type.getLayoutAttr()) {
      return layout.getLayout();
    }
  }
  if (auto part = value.getDefiningOp<pto::PartitionViewOp>()) {
    return getLogicalViewLayout(part.getSource());
  }
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    // Prefer the explicit layout attribute when available.  After rank-2 →
    // rank-5 canonicalization, the padded leading strides satisfy the ND
    // (row-major) recurrence even for DN (col-major) data, so inferLayout
    // alone would misclassify DN as ND (the col-major recurrence breaks at
    // the boundary between padded unit-extent dims and real dims).  The
    // layout attribute carries the *intended* memory layout and is the
    // authoritative source — inferLayout is only a fallback for views that
    // lack an explicit layout.
    if (auto layoutAttr = make.getLayoutAttr()) {
      return layoutAttr.getLayout();
    }
    auto tvTy = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!tvTy) {
      return std::nullopt;
    }
    SmallVector<int64_t> shape(tvTy.getShape().begin(), tvTy.getShape().end());
    SmallVector<int64_t> strides;
    strides.reserve(make.getStrides().size());
    for (Value stride : make.getStrides()) {
      auto cst = getConstIndexValue(stride);
      if (!cst) {
        return std::nullopt;
      }
      strides.push_back(*cst);
    }
    return inferLayout5D(shape, strides,
                         getElemByteSize(tvTy.getElementType()));
  }
  return std::nullopt;
}

static bool getLogicalViewShape(Value value, SmallVectorImpl<int64_t> &shape) {
  if (auto make = value.getDefiningOp<pto::MakeTensorViewOp>()) {
    auto type = dyn_cast<pto::TensorViewType>(make.getResult().getType());
    if (!type) {
      return false;
    }
    shape.assign(type.getShape().begin(), type.getShape().end());
    for (auto [index, operand] : llvm::enumerate(make.getShape())) {
      if (shape[index] == ShapedType::kDynamic) {
        if (auto constant = getConstIndexValue(operand)) {
          shape[index] = *constant;
        }
      }
    }
    return true;
  }

  if (auto partition = value.getDefiningOp<pto::PartitionViewOp>()) {
    auto type =
        dyn_cast<pto::PartitionTensorViewType>(partition.getResult().getType());
    if (!type) {
      return false;
    }
    shape.assign(type.getShape().begin(), type.getShape().end());
    for (auto [index, operand] : llvm::enumerate(partition.getSizes())) {
      if (shape[index] == ShapedType::kDynamic) {
        if (auto constant = getConstIndexValue(operand)) {
          shape[index] = *constant;
        }
      }
    }
    return true;
  }

  return false;
}

static std::optional<pto::Layout> getTileBufLogicalLayout(pto::TileBufType type) {
  if (!type) {
    return std::nullopt;
  }
  int32_t sl = type.getSLayoutValueI32();
  int32_t bl = type.getBLayoutValueI32();
  if (sl != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return pto::Layout::NZ;
  }
  if (bl == static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return pto::Layout::ND;
  }
  if (bl == static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return pto::Layout::DN;
  }
  return std::nullopt;
}

static bool isRowMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
}

static bool isColMajorTileBuf(Type ty) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getBLayoutValueI32() ==
                   static_cast<int32_t>(pto::BLayout::ColMajor);
}

static LogicalResult verifyRowReductionSrcLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout != pto::Layout::ND) {
      return op->emitOpError() << "expects " << name
                               << " to use an ND-style tile layout";
    }
  }
  return success();
}

static LogicalResult verifyRowReductionDstLayout(Operation *op, Type ty,
                                                 StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name
                               << " to use the none_box slayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
        tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
      return op->emitOpError() << "expects " << name
                               << " to use the row_major or col_major blayout";
    }
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    auto layout = getTileBufLogicalLayout(tb);
    if (layout && *layout == pto::Layout::DN) {
      auto shape = getShapeVec(ty);
      if (shape.size() == 2 && shape[1] != ShapedType::kDynamic && shape[1] != 1) {
        return op->emitOpError() << "expects DN-style " << name
                                 << " to have shape[1] == 1";
      }
      return success();
    }
    if (layout && *layout == pto::Layout::ND) {
      return success();
    }
    if (layout) {
      return op->emitOpError() << "expects " << name
                               << " to use a DN-style column vector tile or legacy ND-style tile";
    }
  }
  // The dst valid_shape[1] == 1 constraint for row reductions is enforced in
  // verifyRowReductionValidRegion (it must be conditional on the no-op-marker
  // path), so it is intentionally not duplicated here. A previous unreachable
  // copy of that check lived after this return and has been removed.
  return success();
}

static LogicalResult verifyRowReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool allowEmptyMarker) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // A fully-empty dst valid region (0x0) is PyPTO's dual-AIV no-op replay
  // marker: the op writes no elements, so accept it and skip the non-empty
  // structural constraints. Only plain reductions opt in (allowEmptyMarker);
  // arg reductions (trowargmax/trowargmin) still produce a real per-row index,
  // so they stay strict. One-sided empties (only one dim 0) still fall through
  // and are rejected below. Hardware Rv=0 no-op is tracked in pto-isa#143;
  // PTOAS only guarantees the IR is legal here.
  if (allowEmptyMarker && dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
    return op->emitOpError("expects src valid_shape[0] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
    return op->emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op->emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 1) {
    return op->emitOpError("expects dst valid_shape[1] to be 1");
  }
  return success();
}

static bool isSupportedRowReductionElemType(Type elem) {
  return elem.isInteger(16) || elem.isInteger(32) || elem.isF16() ||
         elem.isF32();
}

[[maybe_unused]] static LogicalResult
verifyTRowReductionNoTmpCommon(Operation *op, Type srcTy, Type dstTy,
                               StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true))) {
    return failure();
  }
  if (!isSupportedRowReductionElemType(getElemTy(srcTy))) {
    return op->emitOpError(elemTypeError);
  }
  return success();
}

static LogicalResult verifyTRowReductionWithTmpCommon(Operation *op, Type srcTy,
                                                      Type tmpTy, Type dstTy,
                                                      StringRef elemTypeError) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (getTargetArch(op) != PTOArch::A5 &&
      getElemTy(srcTy) != getElemTy(tmpTy)) {
    return op->emitOpError("expects A2/A3 tmp to have the same element type as src and dst");
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/true))) {
    return failure();
  }
  if (!isSupportedRowReductionElemType(getElemTy(srcTy))) {
    return op->emitOpError(elemTypeError);
  }
  if (getTargetArch(op) != PTOArch::A5 &&
      failed(verifyTmpCapacityAtLeast(op, tmpTy, 32))) {
    return failure();
  }
  return success();
}

static std::optional<int64_t> getVectorRepeatElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 2048 % elemBits != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(2048 / elemBits);
}

static std::optional<int64_t> getVectorBlockElements(Type elemTy) {
  unsigned elemBits = elemTy ? getPTOStorageElemBitWidth(elemTy) : 0;
  if (elemBits == 0 || 256 % elemBits != 0) {
    return std::nullopt;
  }
  return static_cast<int64_t>(256 / elemBits);
}

static int64_t ceilDivInt64(int64_t numerator, int64_t denominator) {
  assert(denominator > 0 && "denominator must be positive");
  assert(numerator >= 0 && "numerator must be non-negative");
  return (numerator + denominator - 1) / denominator;
}

static std::optional<int64_t> getArgReductionTmpMinStride(Type elemTy,
                                                          int64_t srcValidCols) {
  if (srcValidCols == ShapedType::kDynamic || srcValidCols < 0) {
    return std::nullopt;
  }
  auto repeatElems = getVectorRepeatElements(elemTy);
  auto blockElems = getVectorBlockElements(elemTy);
  if (!repeatElems || !blockElems) {
    return std::nullopt;
  }
  int64_t repeats = ceilDivInt64(srcValidCols, *repeatElems);
  return (ceilDivInt64(repeats * 2, *blockElems) +
          ceilDivInt64(repeats, *blockElems)) *
         *blockElems;
}

static bool hasExactKnownValidShape(Type lhsTy, Type rhsTy) {
  return getValidShapeVec(lhsTy) == getValidShapeVec(rhsTy);
}

static LogicalResult verifyTColArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to have rank-2 valid_shape");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTColArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyTColArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src/tmp element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyColArgReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyColReductionValidRegion(op, srcTy, dstTy,
                                           /*requireNonZeroSrc=*/true))) {
    return failure();
  }
  Type srcElemTy = getElemTy(srcTy);
  unsigned srcElemBits = srcElemTy ? getPTOStorageElemBitWidth(srcElemTy) : 0;
  if (!(mlir::isa<IntegerType, FloatType>(srcElemTy) &&
        (srcElemBits == 8 || srcElemBits == 16 || srcElemBits == 32))) {
    return op->emitOpError(
        "expects src element type to be 1, 2, or 4 bytes wide");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTColSumTmpStride(Operation *op, Type srcTy,
                                            Type tmpTy, bool isBinary) {
  if (!isBinary) {
    return success();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  if (srcValid.size() != 2 || tmpShape.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  int64_t srcValidCols = srcValid[1];
  int64_t tmpStride = tmpShape[1];
  if (srcValidCols != ShapedType::kDynamic && tmpStride != ShapedType::kDynamic &&
      tmpStride < srcValidCols) {
    return op->emitOpError()
           << "expects tmp shape[1] to be at least src valid_shape[1] when "
              "isBinary is true; got "
           << tmpStride << " vs " << srcValidCols;
  }
  return success();
}

static LogicalResult verifyTRowArgTmpA2A3(Operation *op, Type srcTy,
                                          Type tmpTy) {
  if (failed(verifyVecTileStorage(op, tmpTy, "tmp")) ||
      failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }

  if (hasExactKnownValidShape(srcTy, tmpTy)) {
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  auto srcShape = getShapeVec(srcTy);
  auto tmpShape = getShapeVec(tmpTy);
  auto srcValid = getValidShapeVec(srcTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (srcShape.size() != 2 || tmpShape.size() != 2 || srcValid.size() != 2 ||
      tmpValid.size() != 2) {
    return op->emitOpError("expects src and tmp to be rank-2 tiles");
  }

  auto repeatElems = getVectorRepeatElements(getElemTy(srcTy));
  if (!repeatElems) {
    return op->emitOpError("failed to infer A2/A3 tmp contract from src element type");
  }

  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] <= *repeatElems) {
    auto tmpTile = dyn_cast<pto::TileBufType>(tmpTy);
    auto layout = tmpTile ? getTileBufLogicalLayout(tmpTile) : std::nullopt;
    if (layout && *layout == pto::Layout::DN) {
      if (tmpShape[1] != ShapedType::kDynamic && tmpShape[1] != 1) {
        return op->emitOpError("expects A2/A3 tmp DN layout to have shape[1] == 1");
      }
      if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] != 1) {
        return op->emitOpError(
            "expects A2/A3 tmp DN layout to have valid_shape[1] == 1");
      }
      if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
          tmpValid[0] < srcValid[0] * 2) {
        return op->emitOpError()
               << "expects A2/A3 tmp DN layout to have valid_shape[0] >= "
               << (srcValid[0] * 2);
      }
      return verifyTmpCapacityAtLeast(op, tmpTy, 32);
    }

    if (!layout || *layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects A2/A3 tmp to use DN 1-col or ND 2-col layout when src valid_shape[1] fits in one repeat");
    }
    if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
        tmpValid[0] < srcValid[0]) {
      return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < 2) {
      return op->emitOpError(
          "expects A2/A3 tmp valid_shape[1] to be at least 2 in the small-col ND path");
    }
    return verifyTmpCapacityAtLeast(op, tmpTy, 32);
  }

  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (srcShape[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] != srcShape[0]) {
    return op->emitOpError("expects A2/A3 tmp shape[0] to match src shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && tmpValid[0] != ShapedType::kDynamic &&
      tmpValid[0] < srcValid[0]) {
    return op->emitOpError("expects A2/A3 tmp valid_shape[0] to cover src valid rows");
  }
  if (srcValid[1] != ShapedType::kDynamic) {
    auto minStride = getArgReductionTmpMinStride(getElemTy(srcTy), srcValid[1]);
    if (!minStride) {
      return op->emitOpError("failed to infer A2/A3 tmp stride from src element type");
    }
    if (tmpValid[1] != ShapedType::kDynamic && tmpValid[1] < *minStride) {
      return op->emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least "
             << *minStride << " for src valid_shape[1] = " << srcValid[1];
    }
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, 32);
}

static LogicalResult verifyTRowArgReductionOpA2A3(Operation *op, Type srcTy,
                                                  Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyTRowArgTmpA2A3(op, srcTy, tmpTy)) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionNoTmp(Operation *op, Type srcTy,
                                                  Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst")) ||
      failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyTRowArgReductionOpA5(Operation *op, Type srcTy,
                                                Type tmpTy, Type dstTy) {
  if (failed(verifyRowReductionSrcLayout(op, srcTy, "src")) ||
      failed(verifyVecTileCommon(op, tmpTy, "tmp")) ||
      failed(verifyRowReductionDstLayout(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyRowReductionValidRegion(op, srcTy, dstTy,
                                           /*allowEmptyMarker=*/false))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  if (!isSupportedRowReductionElemType(srcElem)) {
    return op->emitOpError("expects src element type to be i16/i32/f16/f32");
  }
  auto dstInt = dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!dstInt || dstInt.getWidth() != 32) {
    return op->emitOpError("expects dst element type to be i32 or ui32");
  }
  return success();
}

static LogicalResult verifyNDStyleVecTile(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, ty, name, allowLowPrecision))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError() << "expects " << name << " to use the row_major blayout";
    }
    if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op->emitOpError() << "expects " << name << " to use the none_box slayout";
    }
  }
  return success();
}

static LogicalResult verifyColReductionValidRegion(Operation *op, Type srcTy,
                                                   Type dstTy,
                                                   bool requireNonZeroSrc) {
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  // Col arg reductions (tcolargmax/tcolargmin) never reach this point with a
  // 0x0 dst: verifyColArgReductionDstLayout enforces dst valid_shape[0] == 1
  // first, so they stay strict without needing a flag here (unlike the row
  // path, whose dst-layout check does not constrain valid).
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (requireNonZeroSrc) {
    if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
      return op->emitOpError("expects src valid_shape[0] to be non-zero");
    }
    if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
      return op->emitOpError("expects src valid_shape[1] to be non-zero");
    }
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}

static LogicalResult verifyColArgReductionDstLayout(Operation *op, Type ty,
                                                    StringRef name) {
  if (failed(verifyNDStyleVecTile(op, ty, name))) {
    return failure();
  }
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2) {
    return op->emitOpError() << "expects " << name
                             << " to have rank-2 valid_shape";
  }
  if (valid[0] != ShapedType::kDynamic && valid[0] != 1) {
    return op->emitOpError() << "expects " << name
                             << " valid_shape[0] to be 1";
  }
  return success();
}

static std::optional<int64_t> getConstantIntegerValue(Value value) {
  if (!value) {
    return std::nullopt;
  }
  if (auto arithCst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(arithCst.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

LogicalResult mlir::pto::SectionSimtOp::verify() {
  func::FuncOp func = getOperation()->getParentOfType<func::FuncOp>();
  if (!func) {
    return emitOpError("must be nested under a func.func");
  }

  if (getDimXAttr().getInt() < 0 || getDimYAttr().getInt() < 0 ||
      getDimZAttr().getInt() < 0) {
    return emitOpError("requires non-negative i32 launch dimensions");
  }

  if (func->hasAttr(pto::kPTOSimtEntryAttrName)) {
    return emitOpError("must not appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "'";
  }

  WalkResult nested = getBody().walk([&](SectionSimtOp nestedOp) {
    nestedOp.emitOpError("nested pto.section.simt is not allowed");
    return WalkResult::interrupt();
  });
  if (nested.wasInterrupted()) {
    return failure();
  }

  return success();
}

LogicalResult mlir::pto::FusionRegionOp::verify() {
  Region &bodyRegion = getBody();
  if (bodyRegion.empty()) {
    return emitOpError("expects a non-empty body region");
  }

  Block &body = bodyRegion.front();
  if (body.getNumArguments() != 0) {
    return emitOpError() << "expects body block to have no arguments, got "
                         << body.getNumArguments();
  }

  if (body.empty() || !body.back().hasTrait<OpTrait::IsTerminator>()) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield) {
    return emitOpError("expects body to terminate with pto.yield");
  }

  if (yield.getValues().size() != getOutputs().size()) {
    return emitOpError() << "expects pto.yield to return "
                         << getOutputs().size() << " values, got "
                         << yield.getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(yield.getValues(), getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

LogicalResult mlir::pto::YieldOp::verify() {
  auto parent = dyn_cast_or_null<FusionRegionOp>(getOperation()->getParentOp());
  if (!parent) {
    return emitOpError("expects parent op to be pto.fusion_region");
  }

  if (getValues().size() != parent.getOutputs().size()) {
    return emitOpError() << "expects " << parent.getOutputs().size()
                         << " yielded values to match parent results, got "
                         << getValues().size();
  }

  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(getValues(), parent.getOutputs()))) {
    Value yielded = std::get<0>(pair);
    Value output = std::get<1>(pair);
    if (yielded.getType() != output.getType()) {
      return emitOpError() << "expects yielded value #" << idx << " to have "
                           << "type " << output.getType() << ", got "
                           << yielded.getType();
    }
  }

  return success();
}

static SmallVector<int64_t> getConstantOrDynamicValues(ValueRange values) {
  SmallVector<int64_t> result;
  result.reserve(values.size());
  for (Value value : values) {
    result.push_back(
        getConstIndexValue(value).value_or(ShapedType::kDynamic));
  }
  return result;
}

static SmallVector<int64_t>
getResolvedMakeTensorViewShape(mlir::pto::MakeTensorViewOp op,
                               mlir::pto::TensorViewType resultType) {
  SmallVector<int64_t> shape(resultType.getShape());
  for (auto [index, dim] : llvm::enumerate(shape)) {
    if (dim != ShapedType::kDynamic) {
      continue;
    }
    if (auto constant = getConstIndexValue(op.getShape()[index])) {
      shape[index] = *constant;
    }
  }
  return shape;
}

static LogicalResult
verifyMakeTensorViewLayout(mlir::pto::MakeTensorViewOp op,
                           mlir::pto::TensorViewType resultType) {
  auto opLayoutAttr = op.getLayoutAttr();
  auto typeLayoutAttr = resultType.getLayoutAttr();
  if (opLayoutAttr && typeLayoutAttr && opLayoutAttr != typeLayoutAttr) {
    return op.emitOpError()
           << "layout attribute " << opLayoutAttr
           << " does not match result type layout " << typeLayoutAttr;
  }

  auto layoutAttr = opLayoutAttr ? opLayoutAttr : typeLayoutAttr;
  if (!layoutAttr) {
    return success();
  }

  SmallVector<int64_t> shape =
      getResolvedMakeTensorViewShape(op, resultType);
  SmallVector<int64_t> strides = getConstantOrDynamicValues(op.getStrides());
  Layout layout = layoutAttr.getLayout();
  unsigned storageElemBytes = getElemByteSize(resultType.getElementType());
  if (isLayoutCompatible5D(layout, shape, strides, storageElemBytes)) {
    return success();
  }
  if (layout != Layout::NZ) {
    return op.emitOpError()
           << "user-specified layout=" << stringifyLayout(layout)
           << " is incompatible with the view shape/stride";
  }
  if (resultType.getRank() != static_cast<int64_t>(kPTOLayoutRank)) {
    return op.emitOpError()
           << "user-specified layout=nz requires a rank-5 view, got rank "
           << resultType.getRank();
  }

  auto error =
      getNZViewCompatibilityError(shape, strides, storageElemBytes);
  return op.emitOpError()
         << "user-specified layout=nz is incompatible with shape/stride: "
         << error.value_or("unknown NZ layout mismatch");
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy) {
    return emitOpError("result must be pto.tensor_view<...>");
  }

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  Type ptrElemTy = ptrTy.getElementType();

  if (ptrElemTy != tvTy.getElementType()) {
    return emitOpError() << "ptr element type must match tensor_view element "
                            "type, but got ptr="
                         << ptrElemTy << " view=" << tvTy.getElementType();
  }

  int64_t rank = tvTy.getRank();
  if (static_cast<int64_t>(getShape().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank) {
    return emitOpError()
           << "shape/strides operand counts must match tensor_view rank="
           << rank;
  }
  return verifyMakeTensorViewLayout(*this, tvTy);
}

struct PartitionSourceInfo {
  Type elementType;
  int64_t rank;
  SmallVector<int64_t> shape;
};

static FailureOr<PartitionSourceInfo>
getPartitionSourceInfo(mlir::pto::PartitionViewOp op) {
  PartitionSourceInfo info;
  if (auto tensorView =
          dyn_cast<mlir::pto::TensorViewType>(op.getSource().getType())) {
    info.elementType = tensorView.getElementType();
    info.rank = tensorView.getRank();
    info.shape.assign(tensorView.getShape().begin(), tensorView.getShape().end());
  } else if (auto partitionView =
                 dyn_cast<mlir::pto::PartitionTensorViewType>(
                     op.getSource().getType())) {
    info.elementType = partitionView.getElementType();
    info.rank = partitionView.getRank();
    info.shape.assign(partitionView.getShape().begin(),
                      partitionView.getShape().end());
  } else {
    op.emitOpError("expects tensor_view or partition_tensor_view source");
    return failure();
  }

  SmallVector<int64_t> logicalShape;
  if (getLogicalViewShape(op.getSource(), logicalShape)) {
    info.shape = std::move(logicalShape);
  }
  return info;
}

static LogicalResult verifyPartitionSignature(
    mlir::pto::PartitionViewOp op,
    mlir::pto::PartitionTensorViewType resultType,
    const PartitionSourceInfo &source) {
  if (source.elementType != resultType.getElementType()) {
    return op.emitOpError()
           << "element type mismatch between source and result: src="
           << source.elementType << " result=" << resultType.getElementType();
  }
  if (static_cast<int64_t>(op.getOffsets().size()) != source.rank) {
    return op.emitOpError()
           << "offset count (" << op.getOffsets().size()
           << ") must match source rank (" << source.rank << ")";
  }
  if (static_cast<int64_t>(op.getSizes().size()) != source.rank) {
    return op.emitOpError()
           << "size count (" << op.getSizes().size()
           << ") must match source rank (" << source.rank << ")";
  }
  return success();
}

static LogicalResult verifyPartitionDimension(
    mlir::pto::PartitionViewOp op, int64_t index, int64_t sourceDim,
    std::optional<int64_t> resultDim) {
  auto offset = getConstIndexValue(op.getOffsets()[index]);
  auto size = getConstIndexValue(op.getSizes()[index]);
  if (offset && *offset < 0) {
    return op.emitOpError() << "offset at dim " << index
                            << " must be non-negative, got " << *offset;
  }
  if (size && *size <= 0) {
    return op.emitOpError() << "size at dim " << index
                            << " must be positive, got " << *size;
  }
  if (resultDim && size && *resultDim != ShapedType::kDynamic &&
      *size != *resultDim) {
    return op.emitOpError() << "size/result mismatch at dim " << index
                            << ": size operand=" << *size
                            << " result type dim=" << *resultDim;
  }
  if (sourceDim == ShapedType::kDynamic) {
    return success();
  }
  if (size && *size > sourceDim) {
    return op.emitOpError() << "size at dim " << index << " (" << *size
                            << ") exceeds static source dim (" << sourceDim
                            << ")";
  }
  if (!offset || !size) {
    return success();
  }

  int64_t end = 0;
  if (llvm::AddOverflow(*offset, *size, end)) {
    return op.emitOpError() << "offset+size at dim " << index << " overflows";
  }
  if (end > sourceDim) {
    return op.emitOpError() << "offset+size at dim " << index << " (" << end
                            << ") exceeds static source dim (" << sourceDim
                            << ")";
  }
  return success();
}

static LogicalResult verifyPartitionBounds(
    mlir::pto::PartitionViewOp op,
    mlir::pto::PartitionTensorViewType resultType,
    const PartitionSourceInfo &source) {
  bool sameRank = resultType.getRank() == source.rank;
  for (int64_t index = 0; index < source.rank; ++index) {
    std::optional<int64_t> resultDim;
    if (sameRank) {
      resultDim = resultType.getShape()[index];
    }
    if (failed(
            verifyPartitionDimension(op, index, source.shape[index], resultDim))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyNZPartition(mlir::pto::PartitionViewOp op,
                                       const PartitionSourceInfo &source) {
  if (getLogicalViewLayout(op.getSource()) != Layout::NZ) {
    return success();
  }
  SmallVector<int64_t> offsets =
      getConstantOrDynamicValues(op.getOffsets());
  SmallVector<int64_t> sizes = getConstantOrDynamicValues(op.getSizes());
  if (auto error =
          getNZSubviewCompatibilityError(source.shape, offsets, sizes)) {
    return op.emitOpError(*error);
  }
  return success();
}

LogicalResult mlir::pto::PartitionViewOp::verify() {
  auto resultType =
      dyn_cast<mlir::pto::PartitionTensorViewType>(getResult().getType());
  if (!resultType) {
    return emitOpError("expects partition_tensor_view result");
  }
  FailureOr<PartitionSourceInfo> source = getPartitionSourceInfo(*this);
  if (failed(source)) {
    return failure();
  }
  if (failed(verifyPartitionSignature(*this, resultType, *source)) ||
      failed(verifyPartitionBounds(*this, resultType, *source))) {
    return failure();
  }
  return verifyNZPartition(*this, *source);
}

LogicalResult mlir::pto::AddPtrOp::verify() {
  Value ptr = getOperation()->getOperand(0);
  Value result = getOperation()->getResult(0);

  auto ptrTy = dyn_cast<mlir::pto::PtrType>(ptr.getType());
  if (!ptrTy) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }

  auto resTy = dyn_cast<mlir::pto::PtrType>(result.getType());
  if (!resTy) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  if (ptrTy != resTy) {
    return emitOpError("result type must match ptr operand type");
  }

  return success();
}

static Type getPointerLikeElementType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return ptrTy.getElementType();
  }
  return Type();
}

static bool isEmitCSupportedScalarType(Type type) {
  if (!type) {
    return false;
  }
  if (type.isF16() || type.isBF16() || type.isF32() || type.isF64()) {
    return true;
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    return intTy.getWidth() == 8 || intTy.getWidth() == 16 ||
           intTy.getWidth() == 32 || intTy.getWidth() == 64;
  }
  if (mlir::pto::isPTOFloat8Type(type)) {
    return true;
  }
  if (isa<mlir::pto::HiF8Type, mlir::pto::F4E1M2x2Type,
          mlir::pto::F4E2M1x2Type>(type)) {
    return true;
  }
  return false;
}

LogicalResult mlir::pto::PtrToIntOp::verify() {
  Type resultTy = getResult().getType();
  auto intTy = dyn_cast<IntegerType>(resultTy);
  if (!intTy || intTy.getWidth() != 64) {
    return emitOpError("result must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getPtr().getType())) {
    return emitOpError("ptr operand must be !pto.ptr<...>");
  }
  return success();
}

LogicalResult mlir::pto::IntToPtrOp::verify() {
  auto addrTy = dyn_cast<IntegerType>(getAddr().getType());
  if (!addrTy || addrTy.getWidth() != 64) {
    return emitOpError("address operand must be i64");
  }

  if (!isa<mlir::pto::PtrType>(getResult().getType())) {
    return emitOpError("result must be !pto.ptr<...>");
  }

  Type dstElem = getPointerLikeElementType(getResult().getType());
  if (!isEmitCSupportedScalarType(dstElem)) {
    return emitOpError("result element type is not supported by EmitC: ")
           << dstElem;
  }

  return success();
}

LogicalResult mlir::pto::LocalArrayGetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getResult().getType() != arrayTy.getElementType()) {
    return emitOpError()
           << "result type " << getResult().getType()
           << " does not match array element type "
           << arrayTy.getElementType();
  }
  return success();
}

LogicalResult mlir::pto::LocalArraySetOp::verify() {
  auto arrayTy = getArray().getType();
  int64_t rank = arrayTy.getRank();
  int64_t numIdx = static_cast<int64_t>(getIndices().size());
  if (numIdx != rank) {
    return emitOpError() << "expects " << rank
                         << " indices for !pto.local_array of rank " << rank
                         << ", got " << numIdx;
  }
  if (getValue().getType() != arrayTy.getElementType()) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match array element type "
                         << arrayTy.getElementType();
  }
  return success();
}

// Resolve the field type reached by following a constant `path` of field
// indices from `root`, descending through nested structs. Emits an actionable
// op error and returns failure on an empty path, an out-of-range index, or a
// descent into a non-struct field. On success writes the terminal field type to
// `fieldTyOut`.
static LogicalResult walkStructPath(Operation *op, mlir::pto::StructType root,
                                    llvm::ArrayRef<int64_t> path,
                                    Type &fieldTyOut) {
  if (path.empty()) {
    return op->emitOpError() << "struct path must have at least one index";
  }
  Type cur = root;
  for (auto [depth, idx] : llvm::enumerate(path)) {
    auto st = dyn_cast<mlir::pto::StructType>(cur);
    if (!st) {
      return op->emitOpError()
             << "struct path index " << depth
             << " descends into non-struct field of type " << cur;
    }
    if (idx < 0 || idx >= static_cast<int64_t>(st.getNumFields())) {
      return op->emitOpError()
             << "struct path index " << depth << " (" << idx
             << ") is out of range for " << st << " with " << st.getNumFields()
             << " field(s)";
    }
    cur = st.getFieldType(static_cast<unsigned>(idx));
  }
  fieldTyOut = cur;
  return success();
}

// The declared struct is stack storage owned by the enclosing scope, and the
// value lowers to a pointer to that storage. Letting it reach a terminator
// would publish that address outside the owning scope: `return %s` hands the
// caller a pointer into a dead frame, and `scf.yield %s` carries it out of the
// region that owns it. Both are rejected here rather than emitted as C++ that
// looks fine and is undefined at run time.
LogicalResult mlir::pto::DeclareStructOp::verify() {
  for (Operation *user : getS().getUsers()) {
    if (!user->hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    return emitOpError()
           << "stack-local struct must not escape the scope that declares it, "
              "but its value is passed to '"
           << user->getName()
           << "', which would expose the address of storage that is about to "
              "die; declare the struct in the outer scope and mutate it from "
              "the nested region instead (pto.struct_set mutates in place, "
              "so a struct never needs to be returned or yielded)";
  }
  return success();
}

// Both accessors bottom out at a scalar. A path ending on a nested !pto.struct
// is rejected: the member chain lowers to `emitc.member`, which yields an
// lvalue, and handing a whole aggregate back as an SSA value would mean copying
// it out of the struct — so reaching into a nested struct is spelled as a longer
// path instead.
static LogicalResult verifyStructLeafIsScalar(Operation *op, Type fieldTy) {
  if (!fieldTy.isIntOrFloat()) {
    return op->emitOpError()
           << "struct path must end at a scalar field, but ends at " << fieldTy
           << "; extend the path to reach a scalar inside it";
  }
  return success();
}

LogicalResult mlir::pto::StructGetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(getOperation(), getS().getType(), getPath(),
                            fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "result type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::StructSetOp::verify() {
  Type fieldTy;
  if (failed(walkStructPath(getOperation(), getS().getType(), getPath(),
                            fieldTy))) {
    return failure();
  }
  if (failed(verifyStructLeafIsScalar(getOperation(), fieldTy))) {
    return failure();
  }
  if (getValue().getType() != fieldTy) {
    return emitOpError() << "value type " << getValue().getType()
                         << " does not match field type " << fieldTy
                         << " at the given path";
  }
  return success();
}

LogicalResult mlir::pto::CastPtrOp::verify() {
  Type inputType = getInput().getType();
  Type resultType = getResult().getType();

  auto inputPtrType = dyn_cast<mlir::pto::PtrType>(inputType);
  auto resultPtrType = dyn_cast<mlir::pto::PtrType>(resultType);
  auto inputMemRefType = dyn_cast<BaseMemRefType>(inputType);
  bool inputIsInteger = isa<IntegerType>(inputType);
  bool resultIsInteger = isa<IntegerType>(resultType);

  if (!inputPtrType && !inputMemRefType && !inputIsInteger) {
    return emitOpError("input must be an integer, memref, or !pto.ptr<...>");
  }
  if (!resultPtrType && !resultIsInteger) {
    return emitOpError("result must be an integer or !pto.ptr<...>");
  }

  if (inputIsInteger && resultIsInteger) {
    return emitOpError("integer-to-integer cast is not a ptr cast");
  }

  if (inputMemRefType && resultIsInteger) {
    return emitOpError("memref-to-integer cast is unsupported");
  }

  if (inputMemRefType && resultPtrType) {
    auto memrefSpace = dyn_cast_or_null<mlir::pto::AddressSpaceAttr>(
        inputMemRefType.getMemorySpace());
    auto resultSpace = resultPtrType.getMemorySpace();
    if (memrefSpace && memrefSpace != resultSpace) {
      return emitOpError(
          "memref-to-ptr cast must stay within the same PTO memory space");
    }
  }

  if (inputPtrType && resultPtrType &&
      inputPtrType.getMemorySpace() != resultPtrType.getMemorySpace()) {
    return emitOpError("ptr-to-ptr cast must stay within the same PTO memory space");
  }

  return success();
}




void PTODialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/PTOTypeDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "PTO/IR/PTOOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "PTO/IR/PTOAttrs.cpp.inc"
      >();

  addInterfaces<PTOInlinerInterface>();
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  if (auto ptrType = dyn_cast<PtrType>(type)) {
    return ptrType.getMemorySpace();
  }
  return {};
}

bool mlir::pto::hasExplicitPTOEntryAttr(func::FuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::hasExplicitPTOEntryAttr(LLVM::LLVMFuncOp func) {
  return func && (func->hasAttrOfType<UnitAttr>(kPTOEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyHACCEntryAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kPTOKernelAttrName) ||
                  func->hasAttrOfType<UnitAttr>(kLegacyPTOAICoreAttrName));
}

bool mlir::pto::isPTOEntryFunction(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::isPTOEntryFunction(LLVM::LLVMFuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  return hasExplicitPTOEntryAttr(func);
}

bool mlir::pto::hasExternalArtifactVisibility(func::FuncOp func) {
  if (!func || func.isDeclaration()) {
    return false;
  }
  if (isPTOEntryFunction(func)) {
    return true;
  }
  auto attr = func->getAttrOfType<StringAttr>(kPTOVisibilityAttrName);
  if (!attr) {
    return false;
  }
  return attr.getValue() == kPTOVisibilityExternalValue;
}

void mlir::pto::setExternalArtifactVisibility(func::FuncOp func,
                                              bool isExternal) {
  if (!func) {
    return;
  }
  if (isExternal) {
    func->setAttr(kPTOVisibilityAttrName,
                  StringAttr::get(func.getContext(),
                                  kPTOVisibilityExternalValue));
    return;
  }
  func->removeAttr(kPTOVisibilityAttrName);
}

LogicalResult mlir::pto::validatePTOEntryFunctions(ModuleOp module) {
  if (!module) {
    return success();
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!hasExplicitPTOEntryAttr(func)) {
      continue;
    }
    if (func.isDeclaration()) {
      return func.emitOpError()
             << "`" << kPTOEntryAttrName
             << "` is only valid on function definitions";
    }
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    if (!isPTOEntryFunction(func)) {
      continue;
    }
    if (func.getFunctionType().getNumResults() != 0) {
      return func.emitOpError()
             << "PTO entry functions must return void";
    }
  }
  return success();
}

// A !pto.struct is represented as a pointer to stack storage. Its provenance
// must therefore remain explicit: the value comes directly from
// pto.declare_struct in the owning function. Function arguments/results and
// operations such as arith.select and scf.if must not manufacture or relay a
// struct-typed SSA value, because that alias hides the declaration from
// DeclareStructOp's direct-use escape check. CFG block arguments cannot make a
// declaration safe to forward either: the branch is a terminator and is
// rejected by DeclareStructOp::verify.
LogicalResult mlir::pto::validateStructProvenance(ModuleOp module) {
  if (!module) {
    return success();
  }

  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      for (auto [i, inputTy] :
           llvm::enumerate(func.getFunctionType().getInputs())) {
        if (!isa<StructType>(inputTy)) {
          continue;
        }
        func.emitOpError()
            << "argument " << i << " has type " << inputTy
            << ", but a stack-local struct must not be a function argument; "
               "structs must originate from 'pto.declare_struct' in the same "
               "function";
        return WalkResult::interrupt();
      }
      for (auto [i, resultTy] :
           llvm::enumerate(func.getFunctionType().getResults())) {
        if (!isa<StructType>(resultTy)) {
          continue;
        }
        func.emitOpError()
            << "result " << i << " has type " << resultTy
            << ", but a stack-local struct must not be returned: the value is "
               "a pointer into the callee's frame, and returning it (even "
               "when it merely passes an argument back through) launders its "
               "provenance; keep the struct in its declaring function "
               "(pto.struct_set mutates in place, so a result is never needed)";
        return WalkResult::interrupt();
      }
    }

    if (!isa<DeclareStructOp>(op)) {
      for (auto [i, opResult] : llvm::enumerate(op->getResults())) {
        if (!isa<StructType>(opResult.getType())) {
          continue;
        }
        op->emitOpError()
            << "result " << i << " has type " << opResult.getType()
            << ", but only 'pto.declare_struct' may produce a !pto.struct "
               "result; derived results hide the stack-storage lifetime and "
               "can escape their declaring scope";
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

void mlir::pto::annotatePTOEntryFunctions(ModuleOp module) {
  (void)module;
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//===----------------------------------------------------------------------===//

static std::optional<uint64_t>
getLocalAddressAlignmentBytes(Attribute memorySpace) {
  auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace);
  if (!addrSpace) {
    return std::nullopt;
  }

  // Keep this verifier as a conservative front-line guard for explicit local
  // tile addresses. PTO-ISA's buffer_limits.hpp defines the baseline
  // TASSIGN<Addr> alignment as 32 bytes for local tile memories. For L0 tile
  // bases, PTOAS level3/manual IR historically uses a 4096-bit (512-byte)
  // granularity; fuller per-arch/per-layout bounds checks belong in PTO-ISA.
  switch (addrSpace.getAddressSpace()) {
  case AddressSpace::VEC:
  case AddressSpace::MAT:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
    return 32;
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
    return 512;
  case AddressSpace::GM:
  case AddressSpace::Zero:
    return std::nullopt;
  }
  return std::nullopt;
}

static LogicalResult verifyConstantLocalAddress(Operation *op, Value addr,
                                                Attribute memorySpace,
                                                int addrIndex = -1) {
  std::optional<uint64_t> alignment =
      getLocalAddressAlignmentBytes(memorySpace);
  if (!alignment || *alignment == 0) {
    return success();
  }

  std::optional<int64_t> constantAddr = mlir::getConstantIntValue(addr);
  if (!constantAddr) {
    return success();
  }

  auto emitAddrError = [&]() {
    InFlightDiagnostic diag = op->emitOpError();
    if (addrIndex >= 0) {
      diag << "addr[" << addrIndex << "]";
    } else {
      diag << "addr";
}
    return diag;
  };

  if (*constantAddr < 0) {
    return emitAddrError() << " must be non-negative, got " << *constantAddr;
  }

  uint64_t unsignedAddr = static_cast<uint64_t>(*constantAddr);
  if ((unsignedAddr % *alignment) != 0) {
    return emitAddrError()
           << " must be aligned to " << *alignment
           << " bytes for local tile memory space, got " << unsignedAddr;
  }

  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  if (failed(verifyTileBufLayoutConstraints(*this, ty, "result"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        ty.getMemorySpace()))) {
    return failure();
  }

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("result tile_buf must have rank-2 validShape");
  }

  // TileBuf valid dims use a negative sentinel (e.g. '?' / -1). Be robust to
  // any negative value (some code may materialize MLIR dynamic sentinels).
  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }

  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// AllocMultiTileOp / MultiTileGetOp
//===----------------------------------------------------------------------===//

LogicalResult AllocMultiTileOp::verify() {
  auto mtbTy = getResult().getType();
  if (!mtbTy) {
    return emitOpError("result must be `!pto.multi_tile_buf`");
  }

  TileBufType slotTy = mtbTy.getSlotType();
  if (!slotTy) {
    return emitOpError("multi_tile_buf slot type must be non-null");
  }

  // Reuse the AllocTileOp valid_row/valid_col contract on the slot type.
  Type elemTy = slotTy.getElementType();
  if (isPTOLowPrecisionType(elemTy)) {
    return emitOpError() << "slot dtype " << elemTy
                         << " is not supported by pto.alloc_multi_tile yet";
  }

  if (failed(verifyTileBufLayoutConstraints(*this, slotTy, "slot"))) {
    return failure();
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        slotTy.getMemorySpace()))) {
    return failure();
  }

  // Multi-buffer slots are placed at product(shape) * element_size byte
  // intervals -- both level3 validation and PTOPlanMemory size them that way.
  // `row_plus_one` compaction inflates the
  // major stride by one element per row, so the slot's physical strided
  // footprint exceeds product(shape) and adjacent slots would silently overlap
  // (data corruption). Reject it until the slot stride is derived from the true
  // strided footprint. Non-boxed compact/`normal` and boxed fractal slayouts
  // pack densely (footprint == product(shape)), so they stay supported.
  if (slotTy.getCompactModeI32() ==
      static_cast<int32_t>(mlir::pto::CompactMode::RowPlusOne)) {
    return emitOpError()
           << "multi_tile_buf slot uses row_plus_one compaction, whose padded "
              "storage footprint exceeds product(shape) and would overlap "
              "adjacent multi-buffer slots; use a compact (non-row_plus_one) "
              "slot layout or a single pto.alloc_tile";
  }

  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;
  auto vs = slotTy.getValidShape();
  if (vs.size() != 2) {
    return emitOpError("slot tile_buf must have rank-2 validShape");
  }

  bool needVR = (vs[0] < 0);
  bool needVC = (vs[1] < 0);
  if (hasVR != needVR) {
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because slot v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));
  }
  if (hasVC != needVC) {
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because slot v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));
  }

  // Count bounds are also enforced by MultiTileBufType::verify, but repeat
  // here so the error points at the alloc op the user wrote.
  uint32_t count = mtbTy.getCount();
  if (count < kPtoMultiBufferMinNum || count > kPtoMultiBufferMaxNum) {
    return emitOpError() << "multi_tile_buf count must be in ["
                         << kPtoMultiBufferMinNum << ", "
                         << kPtoMultiBufferMaxNum << "] (got " << count << ")";
  }

  if (Attribute rawAddrs = (*this)->getAttr(kPtoMultiBufferAddrsAttrName)) {
    auto addrs = dyn_cast<DenseI64ArrayAttr>(rawAddrs);
    if (!addrs) {
      return emitOpError() << "expects internal '"
                           << kPtoMultiBufferAddrsAttrName
                           << "' to be a dense i64 array";
    }
    if (getAddr()) {
      return emitOpError() << "cannot carry both base 'addr' and internal '"
                           << kPtoMultiBufferAddrsAttrName << "'";
    }
    if (addrs.size() != count) {
      return emitOpError() << "expects " << count << " planned slot addresses, got "
                           << addrs.size();
    }

    uint64_t elemBytes = getPTOStorageElemByteSize(slotTy.getElementType());
    uint64_t slotBytes = elemBytes;
    for (int64_t dim : slotTy.getShape()) {
      if (dim == ShapedType::kDynamic) {
        return emitOpError(
            "planned multi-buffer addresses require a static slot shape");
      }
      slotBytes *= static_cast<uint64_t>(dim);
    }
    for (auto [lhsIdx, lhs] : llvm::enumerate(addrs.asArrayRef())) {
      if (lhs < 0) {
        return emitOpError("planned slot addresses must be non-negative");
      }
      uint64_t lhsBegin = static_cast<uint64_t>(lhs);
      uint64_t lhsEnd = lhsBegin + slotBytes;
      for (size_t rhsIdx = lhsIdx + 1;
           rhsIdx < static_cast<size_t>(addrs.size()); ++rhsIdx) {
        uint64_t rhsBegin = static_cast<uint64_t>(addrs[rhsIdx]);
        uint64_t rhsEnd = rhsBegin + slotBytes;
        if (std::max(lhsBegin, rhsBegin) < std::min(lhsEnd, rhsEnd)) {
          return emitOpError() << "planned slots " << lhsIdx << " and "
                               << rhsIdx << " overlap";
        }
      }
    }
  }

  return success();
}

LogicalResult MultiTileGetOp::verify() {
  auto srcTy = getSource().getType();
  auto resultTy = getResult().getType();
  if (!srcTy || !resultTy) {
    return emitOpError("source and result types must be non-null");
  }

  if (srcTy.getSlotType() != resultTy) {
    return emitOpError()
           << "result tile_buf must match the multi_tile_buf slot type: "
           << "expected " << srcTy.getSlotType() << ", got " << resultTy;
  }

  // If slot is an `arith.constant`, check it is in range.
  if (auto slotDef = getSlot().getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = llvm::dyn_cast<IntegerAttr>(slotDef.getValue())) {
      int64_t slotVal = attr.getValue().getSExtValue();
      int64_t count = static_cast<int64_t>(srcTy.getCount());
      if (slotVal < 0 || slotVal >= count) {
        return emitOpError()
               << "constant slot " << slotVal
               << " is out of range for multi_tile_buf count=" << count;
      }
    }
  }

  return success();
}

LogicalResult TAssignOp::verify() {
  if (getTile().getType() != getResult().getType()) {
    return emitOpError("result type must match tile operand type");
  }

  auto tileTy = dyn_cast<TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError("expects tile operand and result to be !pto.tile_buf");
  }

  if (failed(verifyConstantLocalAddress(getOperation(), getAddr(),
                                        tileTy.getMemorySpace()))) {
    return failure();
  }

  return success();
}

LogicalResult TLoadOp::verify() {
  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::PartitionTensorViewType, pto::TileBufType>> {
    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
    auto dstTile = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!srcPart || !dstTile) {
      emitOpError("expects src to be !pto.partition_tensor_view and dst to be !pto.tile_buf");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision))) {
      return failure();
    }

    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        emitOpError() << "expects src shape[" << i << "] to be positive";
        return failure();
      }
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        emitOpError() << "expects dst valid_shape[" << i << "] to be non-negative";
        return failure();
      }
    }
    return std::make_pair(srcPart, dstTile);
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common)) {
      return failure();
    }
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tload low-precision element types to be unsupported");
    }
    if (!(dstElem.isInteger(8) || dstElem.isInteger(16) || dstElem.isInteger(32) ||
          dstElem.isInteger(64) || dstElem.isF16() || dstElem.isBF16() || dstElem.isF32())) {
      return emitOpError("expects A2/A3 tload dst element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects A2/A3 tload dst to use loc=vec or loc=mat");
    }

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
      return emitOpError("expects src and dst element types to have the same bitwidth");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common)) {
      return failure();
    }
    auto [srcPart, dstTile] = *common;

    Type srcElem = srcPart.getElementType();
    Type dstElem = dstTile.getElementType();
    unsigned srcBytes = getElemByteSize(srcElem);
    unsigned dstBytes = getElemByteSize(dstElem);
    if (srcBytes != dstBytes) {
      return emitOpError("expects src and dst element types to have the same element size");
    }
    if (!(dstBytes == 1 || dstBytes == 2 || dstBytes == 4 || dstBytes == 8)) {
      return emitOpError("expects A5 tload dst element size to be 1, 2, 4, or 8 bytes");
    }
    if (!isA5TLoadStoreTransferElemType(srcElem)) {
      return emitOpError("expects A5 tload src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    if (!isA5TLoadStoreTransferElemType(dstElem)) {
      return emitOpError("expects A5 tload dst element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }

    if (dstElem.isInteger(64)) {
      auto pad = dstTile.getPadValueI32();
      if (pad != static_cast<int32_t>(pto::PadValue::Null) &&
          pad != static_cast<int32_t>(pto::PadValue::Zero)) {
        return emitOpError("expects A5 i64/u64 tload dst pad to be null or zero");
      }
    }

    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (dstSpace && *dstSpace == pto::AddressSpace::VEC) {
      int32_t bl = dstTile.getBLayoutValueI32();
      int32_t sl = dstTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      if (!isND && !isDN && !isNZ) {
        return emitOpError("expects A5 tload vec dst layout to be ND, DN, or NZ");
      }
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TPrefetchOp::verify() {
  auto verifyImpl = [&](bool allowLowPrecision) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();

    Type srcElem;
    Type dstElem;

    auto srcPart = dyn_cast<pto::PartitionTensorViewType>(srcTy);
    if (!srcPart) {
      return emitOpError("expects src to be !pto.partition_tensor_view");
    }
    auto srcShape = srcPart.getShape();
    for (unsigned i = 0; i < srcShape.size(); ++i) {
      if (srcShape[i] != ShapedType::kDynamic && srcShape[i] <= 0) {
        return emitOpError() << "expects src shape[" << i << "] to be positive";
      }
    }
    srcElem = srcPart.getElementType();

    auto dstTile = dyn_cast<pto::TileBufType>(dstTy);
    if (!dstTile) {
      return emitOpError("expects dst to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(*this, dstTile, "dst", allowLowPrecision))) {
      return failure();
    }
    auto dstValid = dstTile.getValidShape();
    for (unsigned i = 0; i < dstValid.size(); ++i) {
      if (dstValid[i] != ShapedType::kDynamic && dstValid[i] < 0) {
        return emitOpError()
               << "expects dst valid_shape[" << i << "] to be non-negative";
      }
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTile);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to use loc=vec or loc=mat");
    }
    dstElem = dstTile.getElementType();

    if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
      return emitOpError("expects src and dst element types to have the same element size");
    }
    if (!allowLowPrecision &&
        (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem))) {
      return emitOpError("expects A2/A3 tprefetch low-precision element types to be unsupported");
    }
    if (allowLowPrecision &&
        (!isA5TLoadStoreTransferElemType(srcElem) ||
         !isA5TLoadStoreTransferElemType(dstElem))) {
      return emitOpError("expects A5 tprefetch element types to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyImpl(/*allowLowPrecision=*/true);
  };
  switch (getVerifierTargetArch(getOperation())) {
  case VerifierTargetArch::A2A3:
    return verifyA2A3();
  case VerifierTargetArch::A5:
    return verifyA5();
  }
  return failure();
}

LogicalResult MakePrefetchAsyncContextOp::verify() {
  auto ptrTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!ptrTy) {
    return emitOpError("expects workspace to be !pto.ptr<i8>");
  }
  Type elemTy = ptrTy.getElementType();
  if (!isByteIntegerType(elemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }
  return success();
}

LogicalResult TPrefetchAsyncOp::verify() {
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(getOperation(), getSrc(),
                                                   "src"))) {
    return failure();
  }
  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto ptrTy = llvm::dyn_cast<mlir::pto::PtrType>(getFfts().getType());
  if (!ptrTy) {
    return emitOpError("expects a !pto.ptr operand");
  }

  if (!ptrTy.getElementType().isInteger(64) &&
      !ptrTy.getElementType().isInteger(8)) {
    return emitOpError("expects element type i64 (or i8)");
  }

  return mlir::success();
}

ParseResult mlir::pto::SyncSetOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncSetOp::getPipeAttrName(result.name),
                                SyncSetOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncSetOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SyncSetOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic) {
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2) {
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
    }
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    if (IntegerAttr eventIdAttr = getEventIdAttr()) {
      int64_t eventId = eventIdAttr.getInt();
      if (eventId < 0 || eventId > 15) {
        return emitOpError()
               << "A5 sync.set expects static FFTS event_id in [0, 15], but got "
               << eventId;
      }
    }
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.set expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::SyncWaitOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SyncWaitOp::getPipeAttrName(result.name),
                                SyncWaitOp::getEventIdAttrName(result.name));
}

void mlir::pto::SyncWaitOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

ParseResult mlir::pto::SyncAllOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<Type, 2> operandTypes;
  Attribute modeAttr;
  Attribute coreTypeAttr;

  if (parser.parseLParen()) {
    return failure();
  }

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(operands) || parser.parseColonTypeList(operandTypes) ||
        parser.parseRParen()) {
      return failure();
    }
    if (operands.size() != operandTypes.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects the same number of operands and operand types";
    }
  }

  if (parser.parseKeyword("mode") || parser.parseEqual() ||
      parser.parseAttribute(modeAttr) || parser.parseComma() ||
      parser.parseKeyword("core_type") || parser.parseEqual() ||
      parser.parseAttribute(coreTypeAttr)) {
    return failure();
  }

  auto mode = dyn_cast<pto::SyncAllModeAttr>(modeAttr);
  if (!mode) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects mode to be #pto.sync_all_mode<...>";
  }

  auto coreType = dyn_cast<pto::SyncCoreTypeAttr>(coreTypeAttr);
  if (!coreType) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects core_type to be #pto.sync_core_type<...>";
  }

  result.addAttribute("mode", mode);
  result.addAttribute("core_type", coreType);

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  auto addSegmentSizes = [&](int32_t gm, int32_t used) {
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({gm, used}));
  };

  switch (mode.getValue()) {
  case pto::SyncAllMode::Hard:
    if (!operands.empty()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expects hard syncall to have no operands";
    }
    addSegmentSizes(0, 0);
    return success();
  case pto::SyncAllMode::Soft:
    break;
  }

  if (operands.size() != 1 && operands.size() != 2) {
    return parser.emitError(parser.getCurrentLocation())
           << "expects soft syncall to have gm_workspace and optional "
              "used_cores";
  }
  if (parser.resolveOperand(operands[0], operandTypes[0], result.operands)) {
    return failure();
  }
  if (operands.size() == 2 &&
      parser.resolveOperand(operands[1], operandTypes[1], result.operands)) {
    return failure();
  }
  addSegmentSizes(1, operands.size() == 2 ? 1 : 0);
  return success();
}

void mlir::pto::SyncAllOp::print(OpAsmPrinter &p) {
  SmallVector<Value, 2> operands;
  if (getGmWorkspace()) {
    operands.push_back(getGmWorkspace());
  }
  if (getUsedCores()) {
    operands.push_back(getUsedCores());
  }

  p << "(";
  if (!operands.empty()) {
    p.printOperands(operands);
    p << " : ";
    llvm::interleaveComma(operands, p,
                          [&](Value operand) { p.printType(operand.getType()); });
  }
  p << ") mode = " << getMode() << ", core_type = " << getCoreType();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes", "mode",
                                           "core_type"});
}

LogicalResult mlir::pto::SyncWaitOp::verify() {
  bool hasStatic = getEventIdAttr() != nullptr;
  bool hasDynamic = static_cast<bool>(getEventIdDyn());
  if (hasStatic == hasDynamic) {
    return emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  if (IntegerAttr fftsModeAttr = getFftsModeAttr()) {
    int64_t fftsMode = fftsModeAttr.getInt();
    if (fftsMode < 0 || fftsMode > 2) {
      return emitOpError() << "requires ffts_mode in range [0, 2], but got "
                           << fftsMode;
    }
  }

  auto verifyA2A3 = [&]() -> LogicalResult { return success(); };
  auto verifyA5 = [&]() -> LogicalResult {
    if (IntegerAttr eventIdAttr = getEventIdAttr()) {
      int64_t eventId = eventIdAttr.getInt();
      if (eventId < 0 || eventId > 15) {
        return emitOpError()
               << "A5 sync.wait expects static FFTS event_id in [0, 15], but got "
               << eventId;
      }
    }
    switch (getPipe().getPipe()) {
    case PIPE::PIPE_FIX:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_V:
      return success();
    default:
      return emitOpError() << "A5 sync.wait expects pipe to be one of "
                              "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              "<PIPE_MTE3>, <PIPE_V>";
    }
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyNamedSyncEventOp(Operation *op, PipeAttr pipe,
                                            IntegerAttr eventIdAttr,
                                            Value eventIdDyn,
                                            int64_t maxEventId,
                                            StringRef opName) {
  const bool hasStaticEventId = eventIdAttr != nullptr;
  const bool hasDynamicEventId = static_cast<bool>(eventIdDyn);
  if (hasStaticEventId == hasDynamicEventId) {
    return op->emitOpError()
           << "expects exactly one event-id form: static attr or dynamic index operand";
  }
  const bool staticEventIdOutOfRange =
      hasStaticEventId &&
      (eventIdAttr.getInt() < 0 || eventIdAttr.getInt() > maxEventId);
  if (staticEventIdOutOfRange) {
    return op->emitOpError() << "expects static event_id in [0, " << maxEventId
                             << "], but got " << eventIdAttr.getInt();
  }
  switch (pipe.getPipe()) {
  case PIPE::PIPE_FIX:
  case PIPE::PIPE_MTE1:
  case PIPE::PIPE_MTE2:
  case PIPE::PIPE_MTE3:
  case PIPE::PIPE_V:
    return success();
  default:
    return op->emitOpError() << opName << " expects pipe to be one of "
                              << "<PIPE_FIX>, <PIPE_MTE1>, <PIPE_MTE2>, "
                              << "<PIPE_MTE3>, <PIPE_V>";
  }
}

ParseResult mlir::pto::SetCrossBlockOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetCrossBlockOp::getPipeAttrName(result.name),
                                SetCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetCrossBlockOp::verify() {
  if (IntegerAttr mode = getFftsModeAttr()) {
    int64_t modeValue = mode.getInt();
    if (modeValue != 0) {
      return emitOpError() << "requires ffts_mode 0, but got "
                           << modeValue;
    }
  }
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.set_cross_block");
}

ParseResult mlir::pto::WaitCrossBlockOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitCrossBlockOp::getPipeAttrName(result.name),
                                WaitCrossBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitCrossBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitCrossBlockOp::verify() {
  return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                getEventIdDyn(), 15, "pto.wait_cross_block");
}

ParseResult mlir::pto::SetIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                SetIntraBlockOp::getPipeAttrName(result.name),
                                SetIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::SetIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::SetIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.set_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.set_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::WaitIntraBlockOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  return parseSyncEventOpCommon(parser, result,
                                WaitIntraBlockOp::getPipeAttrName(result.name),
                                WaitIntraBlockOp::getEventIdAttrName(result.name));
}

void mlir::pto::WaitIntraBlockOp::print(OpAsmPrinter &p) {
  printSyncEventOpCommon(p, getOperation(), getPipe(), getEventIdAttr(),
                         getEventIdDyn(), getPipeAttrName().getValue(),
                         getEventIdAttrName().getValue());
}

LogicalResult mlir::pto::WaitIntraBlockOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 15,
                                  "pto.wait_intra_block");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyNamedSyncEventOp(getOperation(), getPipe(), getEventIdAttr(),
                                  getEventIdDyn(), 31,
                                  "pto.wait_intra_block");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TStoreOp::verify() {
  const bool hasFp = static_cast<bool>(getFp());
  const bool hasPreQuant = static_cast<bool>(getPreQuantScalar());
  if (hasFp && hasPreQuant) {
    return emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && getStPhase() != pto::STPhase::Unspecified) {
    return emitOpError("expects fp form to use the default stPhase");
  }

  auto verifyCommon =
      [&](bool allowLowPrecision)
      -> FailureOr<std::pair<pto::TileBufType, pto::PartitionTensorViewType>> {
    auto srcTile = dyn_cast<pto::TileBufType>(getSrc().getType());
    auto dstPart = dyn_cast<pto::PartitionTensorViewType>(getDst().getType());
    if (!srcTile || !dstPart) {
      emitOpError("expects src to be !pto.tile_buf and dst to be !pto.partition_tensor_view");
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, srcTile, "src", allowLowPrecision))) {
      return failure();
    }
    if (hasFp) {
      Type fpTy = getFp().getType();
      if (failed(verifyTileBufCommon(*this, fpTy, "fp", allowLowPrecision))) {
        return failure();
      }
      auto fpSpace = getPTOMemorySpaceEnum(fpTy);
      if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
        emitOpError("expects fp to use loc=scaling");
        return failure();
      }
    }
    for (auto [idx, dim] : llvm::enumerate(dstPart.getShape())) {
      if (dim != ShapedType::kDynamic && dim <= 0) {
        emitOpError() << "expects dst shape[" << idx << "] to be positive";
        return failure();
      }
    }
    auto srcValid = srcTile.getValidShape();
    for (auto [idx, dim] : llvm::enumerate(srcValid)) {
      if (dim != ShapedType::kDynamic && dim < 0) {
        emitOpError() << "expects src valid_shape[" << idx << "] to be non-negative";
        return failure();
      }
    }

    // Keep TSTORE contract explicit while preserving existing legal layout
    // reinterpretation paths (e.g. 1x1024 <-> 32x32, 5D partition views).
    // When both sides are fully static, require equal element counts between
    // dst shape and src valid_shape.
    auto getStaticElemCount = [](ArrayRef<int64_t> shape) -> std::optional<int64_t> {
      int64_t total = 1;
      for (int64_t dim : shape) {
        if (dim == ShapedType::kDynamic) {
          return std::nullopt;
        }
        if (dim <= 0) {
          return std::nullopt;
        }
        if (total > std::numeric_limits<int64_t>::max() / dim) {
          return std::nullopt;
        }
        total *= dim;
      }
      return total;
    };

    auto dstElemCount = getStaticElemCount(dstPart.getShape());
    auto srcValidElemCount = getStaticElemCount(srcValid);
    if (!hasFp && dstElemCount && srcValidElemCount &&
        *dstElemCount != *srcValidElemCount) {
      emitOpError() << "expects dst static element count (" << *dstElemCount
                    << ") to match src valid_shape static element count ("
                    << *srcValidElemCount << ")";
      return failure();
    }
    return std::make_pair(srcTile, dstPart);
  };

  auto isLoadStoreElemType = [&](Type ty) -> bool {
    return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
           ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32();
  };
  auto isI8Like = [&](Type ty) -> bool { return ty.isInteger(8); };
  auto reluMode = getReluPreMode();

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/false);
    if (failed(common)) {
      return failure();
    }
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::MAT &&
                      *srcSpace != pto::AddressSpace::ACC)) {
      return emitOpError("expects A2/A3 tstore src to use loc=vec, loc=mat, or loc=acc");
    }
    if ((hasFp || hasPreQuant) && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
    }
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects reluPreMode form to use loc=acc src");
    }

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC || *srcSpace == pto::AddressSpace::MAT) {
      if (hasFp || hasPreQuant) {
        return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
      }
      if (isPTOLowPrecisionType(dstElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore low-precision dst element types to be unsupported");
      }
      if (!isLoadStoreElemType(srcElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore src element type to be i8/i16/i32/i64/u64/f16/bf16/f32");
      }
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
        return emitOpError("expects A2/A3 vec/mat tstore src and dst element types to have the same bitwidth");
      }
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32())) {
      return emitOpError("expects A2/A3 acc tstore src element type to be i32 or f32");
    }
    if (hasPreQuant) {
      if (srcElem.isInteger(32)) {
        if (!(isI8Like(dstElem) || dstElem.isF16())) {
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8/f16");
        }
      } else if (srcElem.isF32()) {
        if (!isI8Like(dstElem)) {
          return emitOpError("expects A2/A3 acc preQuantScalar tstore dst type to be i8/ui8");
        }
      }
    } else if (!hasFp) {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16())) {
        return emitOpError("expects A2/A3 acc tstore dst element type to be i32/f32/f16/bf16");
      }
    }

    auto srcShape = srcTile.getShape();
    if (srcShape[1] != ShapedType::kDynamic &&
        (srcShape[1] < 1 || srcShape[1] > 4095)) {
      return emitOpError("expects A2/A3 acc tstore src cols to be in [1, 4095]");
    }
    auto srcValid = srcTile.getValidShape();
    if (srcValid[1] != ShapedType::kDynamic &&
        (srcValid[1] < 0 || srcValid[1] > 4095)) {
      return emitOpError("expects A2/A3 acc tstore src valid_shape[1] to be in [0, 4095]");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto common = verifyCommon(/*allowLowPrecision=*/true);
    if (failed(common)) {
      return failure();
    }
    auto [srcTile, dstPart] = *common;
    auto srcSpace = getPTOMemorySpaceEnum(srcTile);
    if (!srcSpace || (*srcSpace != pto::AddressSpace::VEC &&
                      *srcSpace != pto::AddressSpace::ACC)) {
      return emitOpError("expects A5 tstore src to use loc=vec or loc=acc");
    }
    if ((hasFp || hasPreQuant) && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
    }
    if (reluMode != pto::ReluPreMode::NoRelu && *srcSpace != pto::AddressSpace::ACC) {
      return emitOpError("expects reluPreMode form to use loc=acc src");
    }

    Type srcElem = srcTile.getElementType();
    Type dstElem = dstPart.getElementType();
    if (*srcSpace == pto::AddressSpace::VEC) {
      if (hasFp || hasPreQuant) {
        return emitOpError("expects fp/preQuantScalar form to use loc=acc src");
      }
      if (!isA5TLoadStoreTransferElemType(srcElem)) {
        return emitOpError("expects A5 vec tstore src element type to be i8/i16/i32/i64/f16/bf16/f32/f8/hif8/fp4");
      }
      if (getElemByteSize(srcElem) != getElemByteSize(dstElem)) {
        return emitOpError("expects A5 vec tstore src and dst element types to have the same bitwidth");
      }

      int32_t bl = srcTile.getBLayoutValueI32();
      int32_t sl = srcTile.getSLayoutValueI32();
      bool isND = (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isDN = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::NoneBox));
      bool isNZ = (bl == static_cast<int32_t>(pto::BLayout::ColMajor) &&
                   sl == static_cast<int32_t>(pto::SLayout::RowMajor));
      auto srcShape = srcTile.getShape();
      bool isSpecialCase = (srcShape.size() == 2 && (srcShape[0] == 1 || srcShape[1] == 1));
      if (!isSpecialCase && !isND && !isDN && !isNZ) {
        return emitOpError("expects A5 vec tstore src layout to be ND, DN, or NZ (or special case with 1 row/col)");
      }
      return success();
    }

    if (!(srcElem.isInteger(32) || srcElem.isF32())) {
      return emitOpError("expects A5 acc tstore src element type to be i32 or f32");
    }
    if (hasPreQuant) {
      if (!isA5AccStorePreQuantDstType(srcElem, dstElem)) {
        return emitOpError("expects A5 acc preQuantScalar tstore dst type to be i8/ui8/f16/bf16/f32/hif8/f8E4M3");
      }
    } else if (!hasFp) {
      if (!(dstElem.isInteger(32) || dstElem.isF32() || dstElem.isF16() ||
            dstElem.isBF16())) {
        return emitOpError("expects A5 acc tstore dst element type to be i32/f32/f16/bf16");
      }
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAbsOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }

  Type elemTy;
  if (auto tb = dyn_cast<pto::TileBufType>(srcTy)) {
    elemTy = tb.getElementType();
  }
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return mlir::isa<RankedTensorType, pto::TensorViewType, pto::TileBufType,
                pto::PartitionTensorViewType>(ty);
}

static bool isTileLikeType(Type ty) {
  return isa<pto::TileBufType>(ty);
}

static Type getElemTy(Type ty) {
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return tt.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return tv.getElementType();
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return tb.getElementType();
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return tv.getElementType();
  }
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto tt = mlir::dyn_cast<RankedTensorType>(ty)) {
    return SmallVector<int64_t, 4>(tt.getShape().begin(), tt.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::TensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  if (auto tb = mlir::dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getShape().begin(), tb.getShape().end());
  }
  if (auto tv = mlir::dyn_cast<pto::PartitionTensorViewType>(ty)) {
    return SmallVector<int64_t, 4>(tv.getShape().begin(), tv.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return SmallVector<int64_t, 4>(tb.getValidShape().begin(), tb.getValidShape().end());
  }
  return getShapeVec(ty);
}

static int64_t getLogicalTileDim(int64_t rawDim, Type elemTy,
                                 std::optional<pto::BLayout> blayout,
                                 unsigned dimIdx) {
  if (rawDim == ShapedType::kDynamic || !isPTOFloat4PackedType(elemTy)) {
    return rawDim;
  }
  pto::BLayout layout = blayout.value_or(pto::BLayout::RowMajor);
  unsigned packedDim = layout == pto::BLayout::ColMajor ? 0 : 1;
  return dimIdx == packedDim ? rawDim * 2 : rawDim;
}

static std::optional<pto::BLayout> getTileBufBLayout(Type ty) {
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    return static_cast<pto::BLayout>(tb.getBLayoutValueI32());
  }
  return std::nullopt;
}

static SmallVector<int64_t, 4> getLogicalTileExtentVec(Type ty,
                                                       bool useValidShape) {
  SmallVector<int64_t, 4> dims =
      useValidShape ? getValidShapeVec(ty) : getShapeVec(ty);
  if (!isTileLikeType(ty) || dims.size() != 2) {
    return dims;
  }

  Type elemTy = getElemTy(ty);
  auto blayout = getTileBufBLayout(ty);
  for (unsigned i = 0; i < dims.size(); ++i) {
    dims[i] = getLogicalTileDim(dims[i], elemTy, blayout, i);
  }
  return dims;
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  if (!value) {
    return {};
  }
  auto valid = getValidShapeVec(value.getType());
  return valid;
}

static SmallVector<int64_t, 4> getMatmulLogicalShapeVec(Type ty) {
  auto shape = getShapeVec(ty);
  auto valid = getValidShapeVec(ty);
  if (!isa<pto::TileBufType>(ty) || shape.size() != valid.size()) {
    return shape;
  }

  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    if (valid[i] != ShapedType::kDynamic) {
      shape[i] = valid[i];
    }
  }
  return shape;
}

static bool isByteIntegerType(Type ty) {
  auto intTy = dyn_cast<IntegerType>(ty);
  return intTy && intTy.getWidth() == 8;
}

static LogicalResult verifyAsyncFlatContiguous1DGMViewLike(Operation *op,
                                                           Value value,
                                                           StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    return op->emitOpError()
           << "expects " << name << " to be a tensor_view or partition_view";
  }

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      return op->emitOpError() << "expects " << name
                               << " to have a static shape";
    }
  }

  bool logical1D = true;
  for (int i = 0, e = static_cast<int>(shape.size()) - 1; i < e; ++i) {
    logical1D &= shape[i] == 1;
  }
  if (!logical1D) {
    return op->emitOpError()
           << "expects " << name
           << " to be a static flat contiguous logical 1D GM view";
  }

  return success();
}

static bool isCommGlobalLikeType(Type ty) {
  return isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty);
}

enum class CommGlobalShapePolicy {
  StaticOnly,
  AllowDynamicPartitionView,
};

static LogicalResult verifyCommGlobalLike(
    Operation *op, Value value, StringRef name,
    CommGlobalShapePolicy policy = CommGlobalShapePolicy::StaticOnly) {
  Type ty = value.getType();
  if (!isCommGlobalLikeType(ty)) {
    return op->emitOpError()
           << "expects " << name << " to be a tensor_view or partition_view";
  }

  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }

  bool opAllowsDynamic =
      policy == CommGlobalShapePolicy::AllowDynamicPartitionView;
  bool isAllowedDynamicType = isa<pto::PartitionTensorViewType>(ty);
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      if (!opAllowsDynamic) {
        return op->emitOpError()
               << "does not support dynamic dimensions on " << name;
      }
      if (!isAllowedDynamicType) {
        return op->emitOpError() << "allows dynamic dimensions on " << name
                                 << " only for partition_tensor_view";
      }
      continue;
    }
    if (dim <= 0) {
      return op->emitOpError() << "expects every static dimension of " << name
                               << " to be positive";
    }
  }
  return success();
}

static LogicalResult verifyCommSignalLike(Operation *op, Value value,
                                          StringRef name) {
  if (failed(verifyCommGlobalLike(op, value, name))) {
    return failure();
  }
  Type elemTy = getElemTy(value.getType());
  if (!elemTy || !elemTy.isSignlessInteger(32)) {
    return op->emitOpError() << "expects " << name
                             << " element type to be i32";
  }
  return success();
}

static LogicalResult verifyCommStagingTileLike(Operation *op, Value value,
                                               StringRef name) {
  Type ty = value.getType();
  if (!isa<pto::TileBufType>(ty)) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf";
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name
                             << " to be in vec address space";
  }
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim <= 0) {
      return op->emitOpError() << "expects " << name
                               << " to have a positive static shape";
    }
  }
  return success();
}

static LogicalResult verifyCommGlobalGroup(Operation *op, ValueRange group,
                                           StringRef name) {
  if (group.empty()) {
    return op->emitOpError() << "expects at least one " << name << " operand";
  }
  Type groupTy = group.front().getType();
  for (auto it : llvm::enumerate(group)) {
    if (failed(verifyCommGlobalLike(op, it.value(),
                                    (name + "[" + Twine(it.index()) + "]").str()))) {
      return failure();
    }
    if (it.value().getType() != groupTy) {
      return op->emitOpError() << "expects all " << name
                               << " operands to have identical types";
    }
  }
  return success();
}

static LogicalResult verifyCommPingPongSameType(Operation *op, Value ping,
                                                Value pong, StringRef pingName,
                                                StringRef pongName) {
  if (!pong) {
    return success();
  }
  if (failed(verifyCommStagingTileLike(op, ping, pingName)) ||
      failed(verifyCommStagingTileLike(op, pong, pongName))) {
    return failure();
  }
  if (ping.getType() != pong.getType()) {
    return op->emitOpError() << "expects " << pingName << " and " << pongName
                             << " to have identical types";
  }
  return success();
}

static std::optional<uint64_t> getStaticByteSize(Type ty) {
  SmallVector<int64_t, 4> shape = getShapeVec(ty);
  if (shape.empty()) {
    return std::nullopt;
  }
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
  }

  Type elemTy = getElemTy(ty);
  uint64_t elemBytes = getElemByteSize(elemTy);
  if (elemBytes == 0) {
    return std::nullopt;
  }

  uint64_t total = elemBytes;
  for (int64_t dim : shape) {
    total *= static_cast<uint64_t>(dim);
  }
  return total;
}

static LogicalResult verifyTmpCapacityAtLeast(Operation *op, Type tmpTy,
                                              uint64_t requiredBytes,
                                              StringRef tmpName) {
  auto actualBytes = getStaticByteSize(tmpTy);
  if (!actualBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " to have statically known byte capacity";
  }
  if (*actualBytes < requiredBytes) {
    return op->emitOpError()
           << "expects " << tmpName << " capacity to be at least "
           << requiredBytes << " bytes, but got " << *actualBytes << " bytes";
  }
  return success();
}

static std::optional<pto::AddressSpace> getPTOMemorySpaceEnum(Type ty) {
  if (auto ptr = dyn_cast<pto::PtrType>(ty)) {
    return ptr.getMemorySpace().getAddressSpace();
  }
  if (auto tb = dyn_cast<pto::TileBufType>(ty)) {
    if (auto as = dyn_cast_or_null<pto::AddressSpaceAttr>(tb.getMemorySpace())) {
      return as.getAddressSpace();
    }
    return std::nullopt;
  }
  return std::nullopt;
}

TMovForm mlir::pto::classifyTMovForm(Value fp) {
  if (!fp)
    return TMovForm::NoTileAux;
  auto space = getPTOMemorySpaceEnum(fp.getType());
  if (!space)
    return TMovForm::XToZz;
  return *space == pto::AddressSpace::SCALING ? TMovForm::Fp
                                              : TMovForm::XToZz;
}

[[maybe_unused]] static bool isRank2TileBuf(Type ty) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  return tb && tb.getRank() == 2 && tb.getValidShape().size() == 2;
}

static bool isSupportedVecElemType(Type ty, bool allowBf16,
                                   bool allowInt8) {
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (allowBf16 && ty.isBF16()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    switch (it.getWidth()) {
    case 32:
    case 16:
      return true;
    case 8:
      return allowInt8;
    default:
      return false;
    }
  }
  return false;
}

static bool isSupportedMGatherMScatterIndexElemType(Type ty) {
  auto it = dyn_cast<IntegerType>(ty);
  if (!it || it.getWidth() != 32) {
    return false;
  }
  return true;
}

static bool isSupportedMGatherMScatterPayloadElemType(Operation *op, Type ty) {
  if (isSupportedVecElemType(ty, /*allowBf16=*/true, /*allowInt8=*/true)) {
    return true;
  }
  if (!isTargetArchA5(op)) {
    return false;
  }
  return isPTOHiFloat8Type(ty) || isPTOFloat8Type(ty);
}

static bool isSupportedMScatterAtomicPayloadElemType(Type ty,
                                                     pto::ScatterAtomicOp atomic) {
  auto intTy = dyn_cast<IntegerType>(ty);
  switch (atomic) {
  case pto::ScatterAtomicOp::None:
    return true;
  case pto::ScatterAtomicOp::Add:
    return ty.isF16() || ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  case pto::ScatterAtomicOp::Max:
  case pto::ScatterAtomicOp::Min:
    return ty.isF32() ||
           (intTy && intTy.getWidth() == 32);
  }
  llvm_unreachable("Unknown ScatterAtomicOp");
}

static LogicalResult verifyMGatherMScatterMemOperand(Operation *op,
                                                     Value memValue,
                                                     Type dataElemTy,
                                                     StringRef dataOperandLabel) {
  Type memTy = memValue.getType();
  Type memElem = getElemTy(memTy);
  if (!memElem || memElem != dataElemTy) {
    return op->emitOpError() << "expects mem element type to match "
                             << dataOperandLabel << " element type";
  }

  if (!isa<pto::PartitionTensorViewType>(memTy)) {
    return op->emitOpError("expects mem to be !pto.partition_tensor_view");
  }
  if (auto layout = getLogicalViewLayout(memValue)) {
    if (*layout != pto::Layout::ND) {
      return op->emitOpError(
          "expects mem partition view to use ND logical layout when layout "
          "can be inferred");
    }
  }
  return success();
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs);
static bool isKnownUnitExtent(int64_t value);
static bool isKnownZeroOrUnitExtent(int64_t value);
static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs);

static LogicalResult verifyMGatherMScatterTileShape(Operation *op, Type dataTy,
                                                    Type idxTy,
                                                    StringRef dataName,
                                                    std::optional<pto::Coalesce> coalesce) {
  auto dataValid = getValidShapeVec(dataTy);
  auto idxValid = getValidShapeVec(idxTy);
  if (dataValid.size() != 2 || idxValid.size() != 2) {
    return op->emitOpError() << "expects " << dataName
                             << " and idx to have rank-2 valid_shape";
  }

  auto idxTile = dyn_cast<pto::TileBufType>(idxTy);
  if (!idxTile) {
    return op->emitOpError("expects idx to be a tile_buf type");
  }

  const bool idxRowMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::RowMajor);
  const bool idxColMajor =
      idxTile.getBLayoutValueI32() ==
      static_cast<int32_t>(pto::BLayout::ColMajor);

  const bool rowCoalesce1xR =
      idxRowMajor && isKnownZeroOrUnitExtent(idxValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[0]);
  const bool rowCoalesceRx1 =
      idxColMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool baseRowCoalesce =
      idxRowMajor && hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      isKnownZeroOrUnitExtent(idxValid[1]);
  const bool elemCoalesce =
      hasCompatibleKnownExtent(idxValid[0], dataValid[0]) &&
      hasCompatibleKnownExtent(idxValid[1], dataValid[1]);

  if (!coalesce) {
    if (baseRowCoalesce || elemCoalesce) {
      return success();
    }
    return op->emitOpError()
           << "expects idx valid_shape to be [" << dataName
           << ".valid_row, 0|1] or match " << dataName
           << " valid_shape when coalesce is omitted";
  }

  if (*coalesce == pto::Coalesce::Row && (rowCoalesce1xR || rowCoalesceRx1)) {
    return success();
  }

  if (*coalesce == pto::Coalesce::Elem && elemCoalesce) {
    return success();
  }

  if (*coalesce == pto::Coalesce::Row) {
    return op->emitOpError()
           << "expects row-coalesce idx valid_shape to be [0|1, " << dataName
           << ".valid_row] or [" << dataName << ".valid_row, 0|1]";
  }

  return op->emitOpError()
         << "expects elem-coalesce idx valid_shape to match " << dataName
         << " valid_shape";
}

template <typename AttrT>
static AttrT getPTOOpAttr(Operation *op, StringRef name) {
  if (Attribute propsAttr = op->getPropertiesAsAttribute()) {
    if (auto props = dyn_cast<DictionaryAttr>(propsAttr)) {
      if (auto attr = dyn_cast_or_null<AttrT>(props.get(name))) {
        return attr;
      }
    }
  }
  return dyn_cast_or_null<AttrT>(op->getRawDictionaryAttrs().get(name));
}

template <typename OpTy>
static ParseResult parsePTOInherentAttrs(OpAsmParser &parser,
                                         OperationState &result,
                                         NamedAttrList &parsedAttrs,
                                         ArrayRef<StringRef> inherentAttrNames) {
  auto attrLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(parsedAttrs)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<typename OpTy::Properties>();
  OpTy::populateDefaultProperties(result.name, properties);
  if (failed(OpTy::setPropertiesFromAttr(
          properties, parsedAttrs.getDictionary(parser.getContext()), [&] {
            return parser.emitError(attrLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          }))) {
    return failure();
  }

  for (StringRef attrName : inherentAttrNames) {
    parsedAttrs.erase(attrName);
  }
  result.attributes = parsedAttrs;
  return success();
}

static NamedAttrList getNonInherentAttrs(Operation *op,
                                         ArrayRef<StringRef> inherentAttrNames) {
  NamedAttrList attrs;
  for (NamedAttribute attr : op->getRawDictionaryAttrs()) {
    if (llvm::is_contained(inherentAttrNames, attr.getName().getValue())) {
      continue;
    }
    attrs.append(attr);
  }
  return attrs;
}

static pto::CoalesceAttr getMGatherCoalesceAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::GatherOOBAttr getMGatherGatherOobAttrIfPresent(pto::MGatherOp op) {
  return dyn_cast_or_null<pto::GatherOOBAttr>(op.getProperties().gatherOob);
}

static pto::GatherOOB getGatherOobOrDefault(pto::MGatherOp op) {
  if (auto attr = getMGatherGatherOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::GatherOOB::Undefined;
}

static pto::CoalesceAttr getMScatterCoalesceAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::CoalesceAttr>(op.getProperties().coalesce);
}

static pto::ScatterAtomicOpAttr
getMScatterScatterAtomicOpAttrIfPresent(pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterAtomicOpAttr>(
      op.getProperties().scatterAtomicOp);
}

static pto::ScatterOOBAttr getMScatterScatterOobAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterOOBAttr>(op.getProperties().scatterOob);
}

static pto::ScatterConflictAttr getMScatterScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return dyn_cast_or_null<pto::ScatterConflictAttr>(
      op.getProperties().scatterConflict);
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MGatherOp op) {
  if (auto attr = getMGatherCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static std::optional<pto::Coalesce> getCoalesceIfPresent(pto::MScatterOp op) {
  if (auto attr = getMScatterCoalesceAttrIfPresent(op)) {
    return attr.getValue();
  }
  return std::nullopt;
}

static pto::ScatterAtomicOp getScatterAtomicOpOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterAtomicOpAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterAtomicOp::None;
}

static pto::ScatterOOB getScatterOobOrDefault(pto::MScatterOp op) {
  if (auto attr = getMScatterScatterOobAttrIfPresent(op)) {
    return attr.getValue();
  }
  return pto::ScatterOOB::Undefined;
}

static pto::ScatterConflictAttr getScatterConflictAttrIfPresent(
    pto::MScatterOp op) {
  return getMScatterScatterConflictAttrIfPresent(op);
}

static Value getTPrintTmpIfPresent(pto::TPrintOp op) {
  return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
}

static LogicalResult verifyMGatherMScatterIdxTile(Operation *op, Type ty,
                                                  StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name
                             << " to be in the vec address space";
  }
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return op->emitOpError() << "expects " << name << " to be a tile_buf type";
  }
  int32_t blayout = tb.getBLayoutValueI32();
  if (blayout != static_cast<int32_t>(pto::BLayout::RowMajor) &&
      blayout != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError() << "expects " << name
                             << " to use row_major or col_major blayout";
  }
  if (tb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op->emitOpError() << "expects " << name
                             << " to use the none_box slayout";
  }
  return success();
}

static bool isA5TLoadStoreTransferElemType(Type ty) {
  return ty.isInteger(8) || ty.isInteger(16) || ty.isInteger(32) ||
         ty.isInteger(64) || ty.isF16() || ty.isBF16() || ty.isF32() ||
         isPTOLowPrecisionType(ty);
}

static bool isA5AccStorePreQuantDstType(Type srcElem, Type dstElem) {
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  if (!srcElem.isF32()) {
    return false;
  }
  return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
         dstElem.isF32() || isPTOHiFloat8Type(dstElem) ||
         isPTOFloat8E4M3LikeType(dstElem);
}

static bool isA5LowPrecisionTCvtPair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return isPTOFloat8Type(dstElem) || isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isF16()) {
    return isPTOHiFloat8Type(dstElem);
  }
  if (srcElem.isBF16()) {
    return isPTOFloat4PackedType(dstElem);
  }
  if (isPTOFloat4PackedType(srcElem)) {
    return dstElem.isBF16();
  }
  if (isPTOFloat8Type(srcElem) || isPTOHiFloat8Type(srcElem)) {
    return dstElem.isF32();
  }
  return false;
}

static bool isA5SupportedTCvtPair(Type srcElem, Type dstElem) {
  if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
    return isA5LowPrecisionTCvtPair(srcElem, dstElem);
  }
  return true;
}

static LogicalResult verifyTileBufCommon(Operation *op, Type ty, StringRef name,
                                         bool allowLowPrecision) {
  auto tb = dyn_cast<pto::TileBufType>(ty);
  if (tb) {
    if (tb.getRank() != 2) {
      return op->emitOpError() << "expects " << name << " to be a rank-2 tile_buf";
    }
    Type elemTy = tb.getElementType();
    if (!allowLowPrecision && isPTOLowPrecisionType(elemTy)) {
      return op->emitOpError() << name << ": dtype " << elemTy
                               << " is not supported by this op yet";
    }
  } else {
    return op->emitOpError() << "expects " << name << " to be a !pto.tile_buf";
  }

  auto validShape = getValidShapeVec(ty);
  if (validShape.size() != 2) {
    return op->emitOpError() << "expects " << name << " to have a rank-2 valid_shape";
  }
  auto shape = getShapeVec(ty);
  for (unsigned i = 0; i < 2; ++i) {
    if (shape[i] != ShapedType::kDynamic && validShape[i] != ShapedType::kDynamic &&
        validShape[i] > shape[i]) {
      return op->emitOpError() << "expects " << name << " to satisfy valid_shape[" << i
                               << "] <= shape[" << i << "]";
    }
  }
  return success();
}

static LogicalResult verifyTileBufSameElemType(Operation *op, Type lhs, Type rhs,
                                               StringRef lhsName,
                                               StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to be !pto.tile_buf";
  }
  if (getElemTy(lhs) != getElemTy(rhs)) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same element type";
  }
  return success();
}

static LogicalResult verifyTileBufSameValidShape(Operation *op, Type lhs, Type rhs,
                                                 StringRef lhsName, StringRef rhsName) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }
  auto lhsValid = getValidShapeVec(lhs);
  auto rhsValid = getValidShapeVec(rhs);
  for (size_t i = 0; i < lhsValid.size() && i < rhsValid.size(); ++i) {
    if (lhsValid[i] != ShapedType::kDynamic && rhsValid[i] != ShapedType::kDynamic &&
        lhsValid[i] != rhsValid[i]) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
  }
  if (lhsValid.size() != rhsValid.size()) {
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have the same valid_shape";
  }
  return success();
}

static LogicalResult verifyTileBufSameLogicalExtent(Operation *op, Type lhs,
                                                    Type rhs, StringRef lhsName,
                                                    StringRef rhsName,
                                                    bool compareValidShape) {
  if (!isTileLikeType(lhs) || !isTileLikeType(rhs)) {
    return success();
  }

  auto lhsExtent = getLogicalTileExtentVec(lhs, compareValidShape);
  auto rhsExtent = getLogicalTileExtentVec(rhs, compareValidShape);
  auto emitMismatch = [&]() -> LogicalResult {
    if (compareValidShape) {
      return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                               << " to have the same valid_shape";
    }
    return op->emitOpError() << "expects " << lhsName << " and " << rhsName
                             << " to have compatible shapes";
  };
  if (lhsExtent.size() != rhsExtent.size()) {
    return emitMismatch();
  }

  for (size_t i = 0, e = lhsExtent.size(); i < e; ++i) {
    if (lhsExtent[i] != ShapedType::kDynamic &&
        rhsExtent[i] != ShapedType::kDynamic && lhsExtent[i] != rhsExtent[i]) {
      return emitMismatch();
    }
  }
  return success();
}

static LogicalResult verifyPartialValidPattern(Operation *op, Type src0Ty,
                                               Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };
  auto equalsKnown = [](ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
    for (auto [a, b] : llvm::zip(lhs, rhs)) {
      if (a != ShapedType::kDynamic && b != ShapedType::kDynamic && a != b) {
        return false;
      }
    }
    return true;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i])) {
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
    }
  }
  if (!equalsKnown(src0Valid, dstValid) && !equalsKnown(src1Valid, dstValid)) {
    return op->emitOpError(
        "expects at least one of src0/src1 valid_shape to match dst valid_shape");
  }
  return success();
}

static LogicalResult verifyPartialValidPatternLoose(Operation *op, Type src0Ty,
                                                    Type src1Ty, Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  auto lessEqualKnown = [](int64_t lhs, int64_t rhs) {
    return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs <= rhs;
  };

  for (unsigned i = 0; i < 2; ++i) {
    if (!lessEqualKnown(src0Valid[i], dstValid[i]) ||
        !lessEqualKnown(src1Valid[i], dstValid[i])) {
      return op->emitOpError(
          "expects src0/src1 valid_shape to be less than or equal to dst valid_shape");
    }
  }
  return success();
}

[[maybe_unused]] static bool hasKnownZeroValidRegion(Type ty) {
  auto valid = getValidShapeVec(ty);
  if (valid.size() != 2) {
    return false;
  }
  return valid[0] == 0 || valid[1] == 0;
}

static LogicalResult verifyScalarTileOp(Operation *op, Type srcTy, Type dstTy,
                                        StringRef srcName, StringRef dstName,
                                        bool requireValidRowsEqual,
                                        bool requireValidColsEqual) {
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << srcName
                             << " to be in the vec address space";
  }
  if (!dstSpace || *dstSpace != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << dstName
                             << " to be in the vec address space";
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }

  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have rank-2 valid_shape";
  }
  if (requireValidRowsEqual &&
      srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[0]";
  }
  if (requireValidColsEqual &&
      srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects " << srcName << " and " << dstName
           << " to have the same valid_shape[1]";
  }
  return success();
}

static FailureOr<Type>
verifyMatchingRowMajorBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                         Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

static FailureOr<Type>
verifyNumericScalarTileOpCommon(Operation *op, Type srcTy, Type dstTy,
                                Type scalarTy, bool requireValidRowsEqual) {
  if (failed(verifyScalarTileOp(op, srcTy, dstTy, "src", "dst",
                                requireValidRowsEqual,
                                /*requireValidColsEqual=*/true))) {
    return failure();
  }
  if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
    op->emitOpError("scalar must be a scalar type (integer/float)");
    return failure();
  }
  return getElemTy(srcTy);
}

static FailureOr<Type>
verifyShiftLikeBinaryTileOpCommon(Operation *op, Type src0Ty, Type src1Ty,
                                   Type dstTy) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  Type e0 = getElemTy(src0Ty);
  Type e1 = getElemTy(src1Ty);
  Type ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    op->emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op->emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, src1Ty, dstTy, "src1", "dst"))) {
    return failure();
  }
  return e0;
}

static FailureOr<Type> verifyDistinctRowMajorUnaryTileOpCommon(
    Operation *op, Value src, Value dst, StringRef srcName = "src",
    StringRef dstName = "dst") {
  if (src == dst) {
    op->emitOpError("expects src and dst to use different storage");
    return failure();
  }
  Type srcTy = src.getType();
  Type dstTy = dst.getType();
  if (failed(verifyTileBufCommon(op, srcTy, srcName)) ||
      failed(verifyTileBufCommon(op, dstTy, dstName))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    op->emitOpError("failed to get element type for src/dst");
    return failure();
  }
  if (srcElem != dstElem) {
    op->emitOpError("expects src and dst to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects src and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  return srcElem;
}

static LogicalResult verifyArithmeticElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool supported = elemTy.isInteger(32) || elemTy.isInteger(16) ||
                   elemTy.isF16() || elemTy.isF32();
  if (targetArch == PTOArch::A5) {
    supported = supported || (allowInt8OnA5 && elemTy.isInteger(8)) ||
                (allowBf16OnA5 && elemTy.isBF16());
  }
  if (supported) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyArithmeticBinaryTileOpWithArchDispatch(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    FailureOr<Type> elemOr =
        verifyMatchingRowMajorBinaryTileOpCommon(op, src0Ty, src1Ty, dstTy);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyArithmeticScalarTileOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, Type scalarTy, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error,
    bool requireValidRowsEqualOnA2A3 = true,
    bool requireValidRowsEqualOnA5 = false) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireValidRowsEqual) -> LogicalResult {
    FailureOr<Type> elemOr = verifyNumericScalarTileOpCommon(
        op, srcTy, dstTy, scalarTy, requireValidRowsEqual);
    if (failed(elemOr)) {
      return failure();
    }
    return verifyArithmeticElemTypeForArch(op, *elemOr, targetArch,
                                           allowInt8OnA5, allowBf16OnA5,
                                           a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireValidRowsEqualOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireValidRowsEqualOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static LogicalResult verifyTColReductionElemTypeForArch(
    Operation *op, Type elemTy, PTOArch targetArch, bool allowInt8OnA5,
    bool allowBf16OnA5, StringRef a2a3Error, StringRef a5Error) {
  bool ok = elemTy.isF16() || elemTy.isF32() || elemTy.isInteger(16) ||
            elemTy.isInteger(32);
  if (targetArch == PTOArch::A5) {
    ok = ok || (allowInt8OnA5 && elemTy.isInteger(8)) ||
         (allowBf16OnA5 && elemTy.isBF16());
  }
  if (ok) {
    return success();
  }
  return op->emitOpError(targetArch == PTOArch::A5 ? a5Error : a2a3Error);
}

static LogicalResult verifyTColReductionOpWithArchDispatch(
    Operation *op, Type srcTy, Type dstTy, bool requireNonZeroSrcOnA2A3,
    bool requireNonZeroSrcOnA5, bool allowInt8OnA5, bool allowBf16OnA5,
    StringRef a2a3Error, StringRef a5Error) {
  auto verifyByArch = [&](PTOArch targetArch,
                          bool requireNonZeroSrc) -> LogicalResult {
    if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
      return failure();
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return op->emitOpError("expects src and dst to have the same element type");
    }
    if (failed(verifyColReductionValidRegion(op, srcTy, dstTy, requireNonZeroSrc))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    return verifyTColReductionElemTypeForArch(op, elem, targetArch, allowInt8OnA5,
                                              allowBf16OnA5, a2a3Error, a5Error);
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A3, requireNonZeroSrcOnA2A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch(PTOArch::A5, requireNonZeroSrcOnA5);
  };
  return dispatchVerifierByArch(op, verifyA2A3, verifyA5);
}

static bool hasCompatibleKnownExtent(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic || lhs == rhs;
}

static bool isKnownUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 1;
}

static bool isKnownZeroOrUnitExtent(int64_t value) {
  return value == ShapedType::kDynamic || value == 0 || value == 1;
}

static bool hasCompatibleKnownExtentOrZero(int64_t lhs, int64_t rhs) {
  return lhs == ShapedType::kDynamic || rhs == ShapedType::kDynamic ||
         lhs == 0 || lhs == rhs;
}

static LogicalResult verifyVecTileStorage(Operation *op, Type ty, StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  return success();
}
static LogicalResult verifyVecTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto tb = dyn_cast<pto::TileBufType>(ty);
  auto as = getPTOMemorySpaceEnum(ty);
  if (as && *as != pto::AddressSpace::VEC) {
    return op->emitOpError() << "expects " << name << " to be in the vec address space";
  }
  if (tb && tb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError() << "expects " << name << " to use the row_major blayout";
  }
  return success();
}

static LogicalResult verifyVecTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyVecTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyVecTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyVecTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyVecTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyVecTileUnaryOp(Operation *op, Type srcTy, Type dstTy,
                                          StringRef srcName,
                                          StringRef dstName,
                                          bool allowBf16,
                                          bool allowInt8) {
  if (failed(verifyVecTileCommon(op, srcTy, srcName)) ||
      failed(verifyVecTileCommon(op, dstTy, dstName))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, dstTy, srcName, dstName))) {
    return failure();
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), allowBf16, allowInt8)) {
    return op->emitOpError() << "expects vec tile element types to be supported";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA2A3(Operation *op, Type ty,
                                             StringRef name) {
  if (failed(verifyTileBufCommon(op, ty, name))) {
    return failure();
  }
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::ACC) {
    return op->emitOpError() << "expects " << name << " to be in the acc address space";
  }
  return success();
}

static LogicalResult verifyAccTileCommonA5(Operation *op, Type ty,
                                           StringRef name) {
  return verifyAccTileCommonA2A3(op, ty, name);
}

static LogicalResult verifyAccTileCommon(Operation *op, Type ty, StringRef name) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyAccTileCommonA2A3(op, ty, name);
  case VerifierTargetArch::A5:
    return verifyAccTileCommonA5(op, ty, name);
  }
  return failure();
}

static LogicalResult verifyMatTileOperandsA2A3(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy,
                                               bool allowLowPrecision) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", allowLowPrecision)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }
  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] || lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 0 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 0 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 0 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [0, 4095]");
    }
  }
  return success();
}

static LogicalResult verifyMatTileOperandsA5(Operation *op, Type lhsTy,
                                             Type rhsTy, Type dstTy,
                                             bool allowLowPrecision) {
  if (failed(verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                       allowLowPrecision))) {
    return failure();
  }

  auto lhsTb = mlir::dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = mlir::dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = mlir::dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyMatTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                           Type dstTy,
                                           bool allowLowPrecision) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatTileOperandsA2A3(op, lhsTy, rhsTy, dstTy,
                                     allowLowPrecision);
  case VerifierTargetArch::A5:
    return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                   allowLowPrecision);
  }
  return failure();
}

static LogicalResult verifyGemvTileOperandsA2A3(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs")) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs")) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  if (!lhsSpace || !rhsSpace) {
    return op->emitOpError("expects lhs and rhs to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT) {
    return op->emitOpError(
        "expects lhs and rhs to use the left and right address spaces");
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1) {
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  }
  if (isa<pto::TileBufType>(dstTy) && dstValid[0] != ShapedType::kDynamic &&
      dstValid[0] != 1) {
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  }
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0]) {
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  }
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  }
  return success();
}

static LogicalResult verifyGemvTileOperandsA5(Operation *op, Type lhsTy,
                                              Type rhsTy, Type dstTy) {
  if (failed(verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy))) {
    return failure();
  }
  return verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy);
}

static LogicalResult verifyGemvTileOperands(Operation *op, Type lhsTy, Type rhsTy,
                                            Type dstTy) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyGemvTileOperandsA2A3(op, lhsTy, rhsTy, dstTy);
  case VerifierTargetArch::A5:
    return verifyGemvTileOperandsA5(op, lhsTy, rhsTy, dstTy);
  }
  return failure();
}

static LogicalResult verifyA5MxMatTileOperands(Operation *op, Type lhsTy,
                                               Type rhsTy, Type dstTy) {
  if (failed(verifyMatTileOperandsA5(op, lhsTy, rhsTy, dstTy,
                                     /*allowLowPrecision=*/true))) {
    return failure();
  }

  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  if (lhsShape.size() == 2 && rhsShape.size() == 2) {
    int64_t lhsK = lhsShape[1];
    int64_t rhsK = rhsShape[0];
    auto checkPhysicalK = [&](int64_t value, StringRef name) -> LogicalResult {
      if (value != ShapedType::kDynamic && (value < 1 || (value % 64) != 0)) {
        return op->emitOpError() << "expects " << name
                                 << " physical K shape to be a positive multiple of 64 on A5";
      }
      return success();
    };
    if (failed(checkPhysicalK(lhsK, "lhs"))) {
      return failure();
    }
    if (failed(checkPhysicalK(rhsK, "rhs"))) {
      return failure();
    }
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }
  return success();
}

static int64_t ceilDivKnown(int64_t value, int64_t divisor) {
  if (value == ShapedType::kDynamic) {
    return ShapedType::kDynamic;
  }
  return (value + divisor - 1) / divisor;
}

static LogicalResult verifyA5MxMatScaleTile(Operation *op, Type scaleTy,
                                            Type lhsTy, Type rhsTy,
                                            StringRef scaleName,
                                            bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING) {
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";
  }

  auto checkDims = [&](ArrayRef<int64_t> scaleDims, ArrayRef<int64_t> lhsDims,
                       ArrayRef<int64_t> rhsDims, StringRef dimsName) -> LogicalResult {
    if (scaleDims.size() != 2 || lhsDims.size() != 2 || rhsDims.size() != 2) {
      return op->emitOpError() << "expects " << scaleName << ", lhs, and rhs to have rank-2 "
                               << dimsName;
    }

    int64_t m = lhsDims[0];
    int64_t k = lhsDims[1];
    int64_t n = rhsDims[1];
    int64_t scaleK = ceilDivKnown(k, 32);
    int64_t expectedRows = isLeftScale ? m : scaleK;
    int64_t expectedCols = isLeftScale ? scaleK : n;
    if (!hasCompatibleKnownExtent(scaleDims[0], expectedRows) ||
        !hasCompatibleKnownExtent(scaleDims[1], expectedCols)) {
      return op->emitOpError()
             << "expects " << scaleName << " " << dimsName << " to be "
             << (isLeftScale ? "[M, ceil(K/32)]" : "[ceil(K/32), N]");
    }
    return success();
  };

  if (failed(checkDims(getShapeVec(scaleTy), getShapeVec(lhsTy), getShapeVec(rhsTy),
                       "shape"))) {
    return failure();
  }
  if (failed(checkDims(getValidShapeVec(scaleTy), getValidShapeVec(lhsTy),
                       getValidShapeVec(rhsTy), "valid_shape"))) {
    return failure();
  }

  auto scaleTb = dyn_cast<pto::TileBufType>(scaleTy);
  if (!scaleTb) {
    return success();
  }
  if (scaleTb.getBLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::BLayout::RowMajor
                                       : pto::BLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " blayout on A5";
  }
  if (scaleTb.getSLayoutValueI32() !=
      static_cast<int32_t>(isLeftScale ? pto::SLayout::RowMajor
                                       : pto::SLayout::ColMajor)) {
    return op->emitOpError()
           << "expects " << scaleName << " to use the "
           << (isLeftScale ? "row_major" : "col_major")
           << " slayout on A5";
  }
  if (scaleTb.getSFractalSizeI32() != 32) {
    return op->emitOpError() << "expects " << scaleName
                             << " to use fractal=32 on A5";
  }
  return success();
}

static LogicalResult verifyA5MxMatScaleTiles(Operation *op, Type lhsScaleTy,
                                             Type rhsScaleTy, Type lhsTy,
                                             Type rhsTy) {
  if (failed(verifyA5MxMatScaleTile(op, lhsScaleTy, lhsTy, rhsTy, "a_scale",
                                    /*isLeftScale=*/true))) {
    return failure();
  }
  return verifyA5MxMatScaleTile(op, rhsScaleTy, lhsTy, rhsTy, "b_scale",
                                /*isLeftScale=*/false);
}

static LogicalResult verifyA5MxGemvTileOperands(Operation *op, Type lhsTy,
                                                Type rhsTy, Type dstTy) {
  if (failed(verifyTileBufCommon(op, lhsTy, "lhs", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(op, rhsTy, "rhs", /*allowLowPrecision=*/true)) ||
      failed(verifyAccTileCommon(op, dstTy, "dst"))) {
    return failure();
  }

  auto lhsSpace = getPTOMemorySpaceEnum(lhsTy);
  auto rhsSpace = getPTOMemorySpaceEnum(rhsTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!lhsSpace || !rhsSpace || !dstSpace) {
    return op->emitOpError("expects lhs, rhs, and dst to have explicit address spaces");
  }
  if (*lhsSpace != pto::AddressSpace::LEFT || *rhsSpace != pto::AddressSpace::RIGHT ||
      *dstSpace != pto::AddressSpace::ACC) {
    return op->emitOpError(
        "expects lhs, rhs, and dst to use the left, right, and acc address spaces");
  }

  auto lhsShape = getMatmulLogicalShapeVec(lhsTy);
  auto rhsShape = getMatmulLogicalShapeVec(rhsTy);
  auto dstShape = getMatmulLogicalShapeVec(dstTy);
  if ((lhsShape[0] != dstShape[0] || rhsShape[1] != dstShape[1] ||
       lhsShape[1] != rhsShape[0])) {
    return op->emitOpError(
        "expects static matmul tile shapes lhs[M,K], rhs[K,N], and dst[M,N]");
  }

  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (lhsValid.size() == 2 && rhsValid.size() == 2) {
    int64_t m = lhsValid[0];
    int64_t k = lhsValid[1];
    int64_t n = rhsValid[1];
    if ((m != ShapedType::kDynamic && (m < 1 || m > 4095)) ||
        (k != ShapedType::kDynamic && (k < 1 || k > 4095)) ||
        (n != ShapedType::kDynamic && (n < 1 || n > 4095))) {
      return op->emitOpError("expects m, k, and n valid sizes to be in [1, 4095]");
    }
  }

  if (lhsValid[0] != ShapedType::kDynamic && lhsValid[0] != 1) {
    return op->emitOpError("expects lhs valid_shape[0] to be 1 for tgemv");
  }
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] != 1) {
    return op->emitOpError("expects dst valid_shape[0] to be 1 for tgemv");
  }
  if (lhsValid[1] != ShapedType::kDynamic && rhsValid[0] != ShapedType::kDynamic &&
      lhsValid[1] != rhsValid[0]) {
    return op->emitOpError()
           << "expects lhs valid_shape[1] to equal rhs valid_shape[0], but got "
           << lhsValid[1] << " vs " << rhsValid[0];
  }
  if (rhsValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      rhsValid[1] != dstValid[1]) {
    return op->emitOpError()
           << "expects rhs valid_shape[1] to equal dst valid_shape[1], but got "
           << rhsValid[1] << " vs " << dstValid[1];
  }

  auto lhsTb = dyn_cast<pto::TileBufType>(lhsTy);
  auto rhsTb = dyn_cast<pto::TileBufType>(rhsTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!lhsTb || !rhsTb || !dstTb) {
    return success();
  }

  if (lhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects lhs to use the col_major blayout on A5");
  }
  if (rhsTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op->emitOpError("expects rhs to use the row_major blayout on A5");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor)) {
    return op->emitOpError("expects dst to use the col_major blayout on A5");
  }

  if (lhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects lhs to use the row_major slayout on A5");
  }
  if (rhsTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
    return op->emitOpError("expects rhs to use the col_major slayout on A5");
  }
  if (dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op->emitOpError("expects dst to use the row_major slayout on A5");
  }
  return success();
}

static LogicalResult verifyA5MxGemvScaleTile(Operation *op, Type scaleTy,
                                             Type lhsTy, Type rhsTy,
                                             StringRef scaleName,
                                             bool isLeftScale) {
  if (failed(verifyTileBufCommon(op, scaleTy, scaleName,
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto scaleSpace = getPTOMemorySpaceEnum(scaleTy);
  if (!scaleSpace || *scaleSpace != pto::AddressSpace::SCALING) {
    return op->emitOpError() << "expects " << scaleName
                             << " to be in the scaling address space";
  }

  auto scaleShape = getShapeVec(scaleTy);
  auto scaleValid = getValidShapeVec(scaleTy);
  auto lhsShape = getShapeVec(lhsTy);
  auto rhsShape = getShapeVec(rhsTy);
  auto lhsValid = getValidShapeVec(lhsTy);
  auto rhsValid = getValidShapeVec(rhsTy);
  if (scaleShape.size() != 2 || scaleValid.size() != 2 ||
      lhsShape.size() != 2 || rhsShape.size() != 2 || lhsValid.size() != 2 ||
      rhsValid.size() != 2) {
    return op->emitOpError() << "expects " << scaleName
                             << ", lhs, and rhs to have rank-2 shape/valid_shape";
  }

  int64_t logicalM = lhsValid[0];
  int64_t logicalK = lhsValid[1];
  int64_t logicalN = rhsValid[1];
  int64_t scaleK = ceilDivKnown(logicalK, 32);

  int64_t expectedShapeRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedShapeCols = isLeftScale ? scaleK : rhsShape[1];
  int64_t expectedValidRows = isLeftScale ? logicalM : scaleK;
  int64_t expectedValidCols = isLeftScale ? scaleK : logicalN;

  if (!hasCompatibleKnownExtent(scaleShape[0], expectedShapeRows) ||
      !hasCompatibleKnownExtent(scaleShape[1], expectedShapeCols) ||
      !hasCompatibleKnownExtent(scaleValid[0], expectedValidRows) ||
      !hasCompatibleKnownExtent(scaleValid[1], expectedValidCols)) {
    if (isLeftScale) {
      return op->emitOpError()
             << "expects " << scaleName
             << " shape/valid_shape to be [M, ceil(K/32)]";
    }
    return op->emitOpError()
           << "expects " << scaleName
           << " shape/valid_shape to be [ceil(K/32), aligned_N]/[ceil(K/32), N]";
  }
  return success();
}

static LogicalResult verifyMatBiasTileA2A3(Operation *op, Type biasTy, Type dstTy,
                                           bool requireFloatBias) {
  if (failed(verifyTileBufCommon(op, biasTy, "bias"))) {
    return failure();
  }
  auto biasSpace = getPTOMemorySpaceEnum(biasTy);
  if (!biasSpace || *biasSpace != pto::AddressSpace::BIAS) {
    return op->emitOpError("expects bias to be in the bias address space");
  }
  auto biasShape = getShapeVec(biasTy);
  if (biasShape[0] != ShapedType::kDynamic && biasShape[0] != 1) {
    return op->emitOpError("expects bias to have 1 row");
  }
  if (requireFloatBias) {
    if (!getElemTy(biasTy).isF32()) {
      return op->emitOpError("expects bias to have element type f32");
    }
  } else if (getElemTy(biasTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects bias and dst to have the same element type");
  }
  return success();
}

static LogicalResult verifyMatBiasTileA5(Operation *op, Type biasTy, Type dstTy,
                                         bool requireFloatBias) {
  if (failed(verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias))) {
    return failure();
  }
  if (auto biasTb = dyn_cast<pto::TileBufType>(biasTy)) {
    if (biasTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op->emitOpError("expects bias to use the row_major blayout on A5");
    }
  }
  return success();
}

static LogicalResult verifyMatBiasTile(Operation *op, Type biasTy, Type dstTy,
                                       bool requireFloatBias) {
  switch (getVerifierTargetArch(op)) {
  case VerifierTargetArch::A2A3:
    return verifyMatBiasTileA2A3(op, biasTy, dstTy, requireFloatBias);
  case VerifierTargetArch::A5:
    return verifyMatBiasTileA5(op, biasTy, dstTy, requireFloatBias);
  }
  return failure();
}

static LogicalResult verifyMatmulTypeTriple(Operation *op, Type lhsElemTy,
                                            Type rhsElemTy, Type dstElemTy) {
  bool isA5 = getVerifierTargetArch(op) == VerifierTargetArch::A5;
  auto isInt8 = [](Type ty) {
    return ty.isInteger(8);
  };
  if (dstElemTy.isInteger(32) && isInt8(lhsElemTy) && isInt8(rhsElemTy)) {
    return success();
  }

  auto isSupportedFpInput = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isF32();
  };
  if (dstElemTy.isF32() && lhsElemTy == rhsElemTy && isSupportedFpInput(lhsElemTy)) {
    return success();
  }

  auto isA5TMatmulFp8Type = [](Type ty) {
    return isPTOFloat8Type(ty);
  };
  if (isA5 && dstElemTy.isF32()) {
    if (isA5TMatmulFp8Type(lhsElemTy) && isA5TMatmulFp8Type(rhsElemTy)) {
      return success();
    }
    if (isPTOHiFloat8Type(lhsElemTy) && lhsElemTy == rhsElemTy) {
      return success();
    }
  }

  return op->emitOpError()
         << "expects (dst, lhs, rhs) element types to match one of "
            "(i32, i8, i8), (f32, f16, f16), (f32, bf16, bf16), (f32, f32, f32)"
            << (isA5 ? ", (f32, fp8, fp8), or (f32, hif8, hif8)" : "");
}

LogicalResult pto::TAddOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadd element type to be i32/i16/f16/f32",
      "expects A5 tadd element type to be i32/i16/i8/f16/bf16/f32");
}

LogicalResult pto::TAddReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    Type elemTy = *elemOr;
    if (elemTy.isInteger(16) || elemTy.isF16() || elemTy.isF32()) {
      return success();
    }
    return emitOpError("expects element type to be i16/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return emitOpError("taddrelu is only supported on A2/A3 targets");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddCOp::verify() {
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type t2 = getSrc2().getType();
  Type td = getDst().getType();

  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) ||
      !isPTOShapedLike(t2) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/src2/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto s2 = getShapeVec(t2);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != s2 || s0 != sd) {
    return emitOpError("expects src0/src1/src2/dst to have the same shape");
  }
  return success();
}
LogicalResult pto::TAddSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tadds element type to be i32/i16/f16/f32",
      "expects A5 tadds element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

LogicalResult pto::TAxpyOp::verify() {
  auto verifyCommon = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type scalarTy = getScalar().getType();
    Type srcElem = getElemTy(srcTy);
    if (scalarTy != srcElem) {
      return emitOpError("expects scalar type to match src element type");
    }
    if (getShapeVec(srcTy) != getShapeVec(dstTy)) {
      return emitOpError("expects src and dst to have the same shape");
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32())) {
      return emitOpError("expects A2/A3 taxpy dst element type to be f16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32())) {
      return emitOpError("expects A2/A3 taxpy src element type to be f16/f32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    Type srcElem = getElemTy(getSrc().getType());
    Type dstElem = getElemTy(getDst().getType());
    bool sameType = srcElem == dstElem;
    bool widenF16ToF32 = srcElem.isF16() && dstElem.isF32();
    if (!(sameType || widenF16ToF32)) {
      return emitOpError(
          "expects dst/src element types to match, or dst=f32 and src=f16");
    }
    if (!(dstElem.isF16() || dstElem.isF32() || dstElem.isBF16())) {
      return emitOpError("expects A5 taxpy dst element type to be f16/bf16/f32");
    }
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isBF16())) {
      return emitOpError("expects A5 taxpy src element type to be f16/bf16/f32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAddSCOp::verify() {
  Type ts0 = getSrc0().getType();
  Type ts1 = getSrc1().getType();
  Type td = getDst().getType();
  if (!isPTOShapedLike(ts0) || !isPTOShapedLike(ts1) || !isPTOShapedLike(td)) {
    return emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  auto s0 = getShapeVec(ts0);
  auto s1 = getShapeVec(ts1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    return emitOpError("expects src0/src1/dst to have the same shape");
  }
  return success();
}

LogicalResult pto::TAndOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tand src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tand src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTConcatCommon(TConcatOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }

  auto v0 = getValidShapeVec(op.getSrc0());
  auto v1 = getValidShapeVec(op.getSrc1());
  auto vd = getValidShapeVec(op.getDst());
  if (v0.size() != 2 || v1.size() != 2 || vd.size() != 2) {
    return op.emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  // validRow must match dst (when known).
  if (v0[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v0[0] != vd[0]) {
    return op.emitOpError("expects src0 valid row to match dst valid row");
  }
  if (v1[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic && v1[0] != vd[0]) {
    return op.emitOpError("expects src1 valid row to match dst valid row");
  }

  // Total valid columns must fit within dst static cols (when known).
  auto sd = getShapeVec(td);
  if (sd.size() == 2 && sd[1] != ShapedType::kDynamic &&
      v0[1] != ShapedType::kDynamic && v1[1] != ShapedType::kDynamic) {
    if (v0[1] + v1[1] > sd[1]) {
      return op.emitOpError("expects src0.valid_col + src1.valid_col <= dst.cols");
    }
  }

  return e0;
}

static LogicalResult verifyTConcatElemType(TConcatOp op, Type elem) {
  if (elem.isF16() || elem.isF32() || elem.isBF16()) {
    return success();
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it ||
      (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
    return op.emitOpError("expects element type to be i8, i16, i32, f16, f32, or bf16");
  }
  return success();
}

static LogicalResult verifyTConcatLocVec(TConcatOp op, Type ty, StringRef name) {
  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::VEC) {
    return op.emitOpError() << "expects " << name << " to use loc=vec";
  }
  return success();
}

static LogicalResult verifyTConcatA2A3(TConcatOp op) {
  FailureOr<Type> elemOr = verifyTConcatCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst"))) {
    return failure();
  }
  return verifyTConcatElemType(op, *elemOr);
}

static LogicalResult verifyTConcatA5(TConcatOp op) {
  FailureOr<Type> elemOr = verifyTConcatCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTConcatLocVec(op, op.getSrc0().getType(), "src0")) ||
      failed(verifyTConcatLocVec(op, op.getSrc1().getType(), "src1")) ||
      failed(verifyTConcatLocVec(op, op.getDst().getType(), "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(op.getSrc0().getType()) || !isRowMajorTileBuf(op.getSrc1().getType()) ||
      !isRowMajorTileBuf(op.getDst().getType())) {
    return op.emitOpError("expects src0, src1, and dst to use row-major layout");
  }
  return verifyTConcatElemType(op, *elemOr);
}

mlir::LogicalResult mlir::pto::TConcatOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTConcatA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTConcatA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TConcatidxOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<std::pair<Type, Type>> {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type ti0 = getSrc0Idx().getType();
    Type ti1 = getSrc1Idx().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, ti0, "src0Idx")) ||
        failed(verifyTileBufCommon(*this, ti1, "src1Idx")) ||
        failed(verifyTileBufCommon(*this, td, "dst"))) {
      return failure();
    }

    // Check data element type consistency.
    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      emitOpError("failed to get element type for data operands");
      return failure();
    }
    if (e0 != e1 || e0 != ed) {
      emitOpError("expects src0, src1, and dst to have the same element type");
      return failure();
    }

    // Check index element type consistency.
    Type ei0 = getElemTy(ti0);
    Type ei1 = getElemTy(ti1);
    if (!ei0 || !ei1) {
      emitOpError("failed to get element type for index operands");
      return failure();
    }
    if (ei0 != ei1) {
      emitOpError("expects src0Idx and src1Idx to have the same element type");
      return failure();
    }

    // All five tiles must be rank-2.
    auto v0  = getValidShapeVec(getSrc0());
    auto v1  = getValidShapeVec(getSrc1());
    auto vi0 = getValidShapeVec(getSrc0Idx());
    auto vi1 = getValidShapeVec(getSrc1Idx());
    auto vd  = getValidShapeVec(getDst());
    if (v0.size() != 2 || v1.size() != 2 || vi0.size() != 2 ||
        vi1.size() != 2 || vd.size() != 2) {
      return emitOpError("expects all operands to have rank-2 valid_shape");
    }

    // validRow must match dst (when known).
    auto checkValidRow = [&](const auto &v, StringRef name) -> LogicalResult {
      if (v[0] != ShapedType::kDynamic && vd[0] != ShapedType::kDynamic &&
          v[0] != vd[0]) {
        return emitOpError("expects ") << name << " valid row to match dst valid row";
      }
      return success();
    };
    if (failed(checkValidRow(v0, "src0")) ||
        failed(checkValidRow(v1, "src1")) ||
        failed(checkValidRow(vi0, "src0Idx")) ||
        failed(checkValidRow(vi1, "src1Idx"))) {
      return failure();
    }

    // Index tile must have cols >= 1 (when known).
    if (vi0[1] != ShapedType::kDynamic && vi0[1] < 1) {
      return emitOpError("expects src0Idx valid_col >= 1");
    }
    if (vi1[1] != ShapedType::kDynamic && vi1[1] < 1) {
      return emitOpError("expects src1Idx valid_col >= 1");
    }

    return std::make_pair(e0, ei0);
  };

  auto verifyElementTypes = [&](Type dataElem, Type idxElem) -> LogicalResult {
    // Data element type: f16, f32, bf16, i8, i16, i32 (signless).
    if (!dataElem.isF16() && !dataElem.isF32() && !dataElem.isBF16()) {
      auto it = mlir::dyn_cast<IntegerType>(dataElem);
      if (!it || !it.isSignless() ||
          (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
        return emitOpError()
               << "expects data element type to be i8, i16, i32, f16, f32, or bf16";
      }
    }

    // Index element type: i8, i16, i32 (signless).
    auto it = mlir::dyn_cast<IntegerType>(idxElem);
    if (!it || !it.isSignless() ||
        (it.getWidth() != 8 && it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError()
             << "expects index element type to be i8, i16, or i32";
    }
    return success();
  };

  auto verifyLocVec = [&](Type ty, StringRef name) -> LogicalResult {
    auto as = getPTOMemorySpaceEnum(ty);
    if (!as || *as != pto::AddressSpace::VEC) {
      return emitOpError() << "expects " << name << " to use loc=vec";
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst"))) {
      return failure();
    }
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  auto verifyA5 = [&]() -> LogicalResult {
    auto elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyLocVec(getSrc0().getType(), "src0")) ||
        failed(verifyLocVec(getSrc1().getType(), "src1")) ||
        failed(verifyLocVec(getSrc0Idx().getType(), "src0Idx")) ||
        failed(verifyLocVec(getSrc1Idx().getType(), "src1Idx")) ||
        failed(verifyLocVec(getDst().getType(), "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc0().getType()) ||
        !isRowMajorTileBuf(getSrc1().getType()) ||
        !isRowMajorTileBuf(getSrc0Idx().getType()) ||
        !isRowMajorTileBuf(getSrc1Idx().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError(
          "expects all operands to use row-major layout");
    }
    return verifyElementTypes(elemOr->first, elemOr->second);
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TAndSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16)) {
      return emitOpError(
          "expects A2/A3 tands src, scalar, and dst element type to be i8/i16");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tands src, scalar, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static ParseResult parseTCILikeOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand s, tmp, dst;
  Type sTy, tmpTy, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(s)) {
    return failure();
  }

  bool hasTmp = succeeded(parser.parseOptionalComma());
  if (hasTmp && parser.parseOperand(tmp)) {
    return failure();
  }

  if (parser.parseColonType(sTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(s, sTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printTCILikeOp(OpAsmPrinter &p, Operation *op, Value s, Value tmp,
                           Value dst) {
  p << " ins(" << s;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << s.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TCIOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTCILikeOp(parser, result);
}

void mlir::pto::TCIOp::print(OpAsmPrinter &p) {
  printTCILikeOp(p, getOperation(), getS(), getTmp(), getDst());
}

LogicalResult pto::TCIOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (getTmp() && failed(verifyTileBufCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }

  auto elemTy = mlir::dyn_cast<IntegerType>(getElemTy(dstTy));
  if (!elemTy) {
    return emitOpError("expects dst element type to be integer");
  }

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32) {
    return emitOpError("expects dst element type to be i16/i32");
  }

  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto tmpTy = mlir::dyn_cast<TileBufType>(getTmp().getType());
    if (!tmpTy) {
      return emitOpError("expects tmp to be a tile buffer");
    }
    auto tmpSpace =
        mlir::dyn_cast_or_null<AddressSpaceAttr>(tmpTy.getMemorySpace());
    if (!tmpSpace || tmpSpace.getAddressSpace() != AddressSpace::VEC) {
      return emitOpError("expects tmp to be in vec address space");
    }
    Type tmpElemTy = tmpTy.getElementType();
    if (!(tmpElemTy.isF32() || tmpElemTy.isInteger(32))) {
      return emitOpError("expects A2/A3 tmp element type to be a 4-byte type");
    }
    if (tmpTy.getBLayoutValueI32() != static_cast<int32_t>(BLayout::RowMajor)) {
      return emitOpError("expects tmp blayout to be row_major");
    }
    if (tmpTy.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox)) {
      return emitOpError("expects tmp slayout to be none_box");
    }
    if (tmpTy.getSFractalSizeI32() != 512) {
      return emitOpError("expects tmp fractal size to be 512");
    }
    auto tmpBytes = getStaticByteSize(tmpTy);
    if (!tmpBytes) {
      return emitOpError("expects tmp to have static byte size");
    }
    uint64_t minTmpBytes = bw == 32 ? 768 : 1792;
    if (*tmpBytes < minTmpBytes) {
      return emitOpError("expects A2/A3 tmp capacity to be at least ")
             << minTmpBytes << " bytes for " << bw
             << "-bit dst element type";
    }
  }

  auto sTy = mlir::dyn_cast<IntegerType>(getOperand(0).getType());
  if (!sTy) {
    return emitOpError("expects S to be integer");
  }

  if (sTy != elemTy) {
    return emitOpError("expects S and dst element type to be exactly the same type");
  }
  auto shape = getShapeVec(dstTy);
  if (shape.size() != 2) {
    return emitOpError("expects dst to be rank-2");
  }
  if (shape[1] != ShapedType::kDynamic && shape[1] == 1) {
    return emitOpError("expects dst cols to be different from 1");
  }

  return success();
}

LogicalResult pto::TTriOp::verify() {
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  auto diagonalTy = mlir::dyn_cast<IntegerType>(getDiagonal().getType());
  if (!diagonalTy) {
    return emitOpError("expects diagonal to be an integer operand");
  }

  int32_t upperOrLower = getUpperOrLower();
  if (upperOrLower != 0 && upperOrLower != 1) {
    return emitOpError("expects upperOrLower to be 0 (lower) or 1 (upper)");
  }

  Type elemTy = getElemTy(dstTy);
  return dispatchVerifierByArch(
      getOperation(),
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/false,
                                    /*allowInt8=*/false)) {
          return emitOpError()
                 << "expects A2/A3 dst element type to be f16/f32/i16/i32/u16/u32";
        }
        return success();
      },
      [&]() -> LogicalResult {
        if (!isSupportedVecElemType(elemTy, /*allowBf16=*/true,
                                    /*allowInt8=*/true)) {
          return emitOpError()
                 << "expects A5 dst element type to be f16/f32/bf16/i8/i16/i32/u8/u16/u32";
        }
        return success();
      });
}

LogicalResult pto::TCmpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyVecTileStorage(*this, t0, "src0")) ||
        failed(verifyVecTileStorage(*this, t1, "src1")) ||
        failed(verifyVecTileStorage(*this, td, "dst"))) {
      return failure();
    }

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      return emitOpError("failed to get element type for src0/src1/dst");
    }
    if (e0 != e1) {
      return emitOpError("expects src0 and src1 to have the same element type");
    }
    if (!(e0.isInteger(32) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tcmp input element type to be i32/f16/f32");
    }
    if (!ed.isInteger(8)) {
      return emitOpError("expects dst element type to be i8");
    }

    auto valid0 = getValidShapeVec(t0);
    auto valid1 = getValidShapeVec(t1);
    auto validd = getValidShapeVec(td);
    if (valid0.size() != 2 || valid1.size() != 2 || validd.size() != 2) {
      return emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
    }
    if (!hasCompatibleKnownExtent(valid0[0], valid1[0])) {
      return emitOpError("expects src0 and src1 to have the same valid row");
    }
    if (!hasCompatibleKnownExtent(valid0[1], valid1[1])) {
      return emitOpError("expects src0 and src1 to have the same valid column");
    }
    if (!hasCompatibleKnownExtent(valid0[0], validd[0])) {
      return emitOpError("expects src0 valid row to equal dst valid row");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    if (failed(verifyTileBufCommon(*this, t0, "src0")) ||
        failed(verifyTileBufCommon(*this, t1, "src1")) ||
        failed(verifyTileBufCommon(*this, td, "dst"))) {
      return failure();
    }

    Type e0 = getElemTy(t0);
    Type e1 = getElemTy(t1);
    Type ed = getElemTy(td);
    if (!e0 || !e1 || !ed) {
      return emitOpError("failed to get element type for src0/src1/dst");
    }
    if (e0 != e1) {
      return emitOpError("expects src0 and src1 to have the same element type");
    }
    bool inputOk = e0.isF16() || e0.isF32() || e0.isBF16() ||
                   e0.isInteger(8) || e0.isInteger(16) || e0.isInteger(32);
    if (!inputOk) {
      return emitOpError("expects A5 tcmp input element type to be i8/i16/i32/f16/bf16/f32");
    }
    if (auto it = dyn_cast<IntegerType>(ed)) {
      if (it.getWidth() != 8) {
        return emitOpError("expects dst element type to be i8");
      }
    } else {
      return emitOpError("expects dst element type to be i8");
    }

    if (getShapeVec(t0) != getShapeVec(t1) || getShapeVec(t0) != getShapeVec(td)) {
      return emitOpError("expects src0, src1, and dst to have the same shape");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32())) {
      return emitOpError("expects A2/A3 tcmps input element type to be i16/i32/f16/f32");
    }

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat())) {
      return emitOpError("expects scalar to be integer, index, or float");
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0]) {
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32())) {
      return emitOpError("expects A5 tcmps input element type to be i8/i16/i32/f16/f32");
    }

    auto scalarTy = getScalar().getType();
    if (!(scalarTy.isIntOrIndexOrFloat())) {
      return emitOpError("expects scalar to be integer, index, or float");
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError("expects src and dst to have rank-2 valid_shape");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != dstValid[0]) {
      return emitOpError("expects src and dst to have the same valid_shape[0]");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
    return failure();
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return emitOpError("expects tcolexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(getSrc());
  auto dstValid = getValidShapeVec(getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      srcValid[1] != dstValid[1]) {
    return emitOpError("expects src and dst to have the same valid_shape[1]");
  }
  return success();
}
static LogicalResult verifyTColExpandBinaryLikeOp(Operation *op, Type t0, Type t1,
                                                  Type td, PTOArch targetArch,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td)) {
    return op->emitOpError("expects src0/src1/dst to be PTO shaped-like types");
  }

  Type e0 = getElemTy(t0);
  Type e1 = getElemTy(t1);
  Type ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for src0/src1/dst");
  }

  auto isSupportedElem = [allowIntegerTypes, targetArch](Type elemTy) {
    if (elemTy.isF16() || elemTy.isF32()) {
      return true;
    }
    if (!allowIntegerTypes) {
      return false;
    }
    if (elemTy.isInteger(16) || elemTy.isInteger(32)) {
      return true;
    }
    return targetArch == PTOArch::A5 && elemTy.isInteger(8);
  };
  if (!isSupportedElem(e0) || !isSupportedElem(e1) || !isSupportedElem(ed)) {
    if (!allowIntegerTypes) {
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    }
    if (targetArch == PTOArch::A5) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    }
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (getShapeVec(t0) != getShapeVec(td)) {
    return op->emitOpError("expects src0/dst to have same shape");
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst"))) {
    return failure();
  }

  if (auto src0TileTy = dyn_cast<TileBufType>(t0)) {
    if (src0TileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects src0 to use row-major layout");
    }
  }

  if (auto src1TileTy = dyn_cast<TileBufType>(t1)) {
    if (src1TileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects src1 to use row-major layout");
    }
  }
  if (auto dstTileTy = dyn_cast<TileBufType>(td)) {
    if (dstTileTy.getBLayoutValueI32() != 0) {
      return op->emitOpError("expects dst to use row-major layout");
    }
  }

  auto src1Valid = getValidShapeVec(t1);
  auto dstValid = getValidShapeVec(td);
  if (src1Valid.size() == 2 && dstValid.size() == 2 &&
      src1Valid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
      src1Valid[1] != dstValid[1]) {
    return op->emitOpError("expects src1 valid_shape[1] to equal dst valid_shape[1]");
  }

  return success();
}
LogicalResult pto::TColExpandMulOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmul",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandAddOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandadd",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    bool allowIntegerTypes = (targetArch == PTOArch::A5);
    return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        targetArch, "tcolexpanddiv",
                                        /*allowIntegerTypes=*/allowIntegerTypes);
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult pto::TColExpandSubOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandsub",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandExpdifOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandexpdif",
                                      /*allowIntegerTypes=*/false);
}
LogicalResult pto::TColExpandMaxOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmax",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColExpandMinOp::verify() {
  PTOArch arch = getTargetArch(getOperation());
  return verifyTColExpandBinaryLikeOp(getOperation(), getSrc0().getType(),
                                      getSrc1().getType(), getDst().getType(),
                                      arch, "tcolexpandmin",
                                      /*allowIntegerTypes=*/true);
}
LogicalResult pto::TColMaxOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmax element type to be f16/f32/i16/i32",
      "expects A5 tcolmax element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMaxOp::verify() {
  if (!getTmp()) {
    return verifyTColArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColMinOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/true,
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolmin element type to be f16/f32/i16/i32",
      "expects A5 tcolmin element type to be i8/i16/i32/f16/bf16/f32");
}

LogicalResult pto::TColArgMinOp::verify() {
  if (!getTmp()) {
    return verifyTColArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTColArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}



ParseResult mlir::pto::TColSumOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  OpAsmParser::UnresolvedOperand dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  // Parse: ins(%src : type) or ins(%src, %tmp {isBinary = ...}: type, type)
  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }

  // Check for optional tmp operand (format 2)
  if (succeeded(parser.parseOptionalComma())) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type)
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;

    // Parse attributes (isBinary)
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }

    // Parse types: : type, type
    if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  } else {
    // Format 1: ins(%src : type)
    if (parser.parseColonType(srcTy)) {
      return failure();
    }
  }

  if (parser.parseRParen()) {
    return failure();
  }

  // Parse: outs(%dst : type)
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }

  // Parse any remaining attributes (for format 1)
  if (!hasTmp) {
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
  }

  // Resolve operands
  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }

  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }

  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TColSumOp::print(OpAsmPrinter &p) {
  if (getTmp()) {
    // Format 2: ins(%src, %tmp {isBinary = ...}: type, type) outs(%dst : type)
    p << " ins(" << getSrc() << ", " << getTmp();
    // Print isBinary attribute if present
    SmallVector<StringRef, 2> elidedAttrs = {"operandSegmentSizes"};
    if (!getIsBinaryAttr() || getIsBinaryAttr().getValue() == false) {
      elidedAttrs.push_back("isBinary");
    }
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    p << " : " << getSrc().getType() << ", " << getTmp().getType() << ")";
  } else {
    // Format 1: ins(%src : type) outs(%dst : type)
    p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  }

  p << " outs(" << getDst() << " : " << getDst().getType() << ")";

  // Print remaining attributes for format 1 (excluding isBinary)
  if (!getTmp()) {
    SmallVector<StringRef, 2> elidedAttrs = {"isBinary", "operandSegmentSizes"};
    p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  }
}

LogicalResult pto::TColSumOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
      return failure();
    }
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp && !hasIsBinary) {
      return emitOpError("tmp operand requires isBinary attribute");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy)) {
        return emitOpError("expects src/tmp/dst element types to match");
      }
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary()))) {
        return failure();
      }
      if (getIsBinary()) {
        auto srcValid = getValidShapeVec(srcTy);
        auto elemBytes = getElemByteSize(getElemTy(srcTy));
        if (srcValid.size() != 2 || srcValid[0] == ShapedType::kDynamic ||
            srcValid[1] == ShapedType::kDynamic || elemBytes == 0) {
          return emitOpError(
              "expects static src valid_shape and element size to verify tcolsum tmp");
        }
        uint64_t requiredBytes =
            static_cast<uint64_t>(ceilDivInt64(srcValid[0], 2)) *
            static_cast<uint64_t>(srcValid[1]) * elemBytes;
        if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, requiredBytes))) {
          return failure();
        }
      }
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return emitOpError("expects src/dst element types to match");
    }
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/false))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError("expects A2/A3 tcolsum element type to be f16/f32/i16/i32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyNDStyleVecTile(*this, srcTy, "src")) ||
        failed(verifyNDStyleVecTile(*this, dstTy, "dst"))) {
      return failure();
    }
    bool hasTmp = (bool)getTmp();
    bool hasIsBinary = (bool)getIsBinaryAttr();
    if (hasTmp && !hasIsBinary) {
      return emitOpError("tmp operand requires isBinary attribute");
    }
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyNDStyleVecTile(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(srcTy) != getElemTy(dstTy) || getElemTy(srcTy) != getElemTy(tmpTy)) {
        return emitOpError("expects src/tmp/dst element types to match");
      }
      if (failed(verifyTColSumTmpStride(*this, srcTy, tmpTy, getIsBinary()))) {
        return failure();
      }
      if (getIsBinary()) {
        auto srcValid = getValidShapeVec(srcTy);
        auto elemBytes = getElemByteSize(getElemTy(srcTy));
        if (srcValid.size() != 2 || srcValid[0] == ShapedType::kDynamic ||
            srcValid[1] == ShapedType::kDynamic || elemBytes == 0) {
          return emitOpError(
              "expects static src valid_shape and element size to verify tcolsum tmp");
        }
        uint64_t requiredBytes =
            static_cast<uint64_t>(ceilDivInt64(srcValid[0], 2)) *
            static_cast<uint64_t>(srcValid[1]) * elemBytes;
        if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, requiredBytes))) {
          return failure();
        }
      }
    }
    if (getElemTy(srcTy) != getElemTy(dstTy)) {
      return emitOpError("expects src/dst element types to match");
    }
    if (failed(verifyColReductionValidRegion(*this, srcTy, dstTy,
                                             /*requireNonZeroSrc=*/true))) {
      return failure();
    }
    Type elem = getElemTy(srcTy);
    if (!(elem.isF16() || elem.isF32() || elem.isBF16() || elem.isInteger(8) ||
          elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError("expects A5 tcolsum element type to be i8/i16/i32/f16/bf16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult pto::TColProdOp::verify() {
  return verifyTColReductionOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(),
      /*requireNonZeroSrcOnA2A3=*/false, /*requireNonZeroSrcOnA5=*/false,
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/true,
      "expects A2/A3 tcolprod element type to be f16/f32/i16/i32",
      "expects A5 tcolprod element type to be i16/ui16/i32/ui32/f16/bf16/f32");
}

static bool tcvtIsResolvedSubview(Value value) {
  auto alloc = value.getDefiningOp<pto::AllocTileOp>();
  auto semantics =
      alloc ? alloc->getAttrOfType<StringAttr>("pto.view_semantics")
            : StringAttr();
  return semantics && semantics.getValue() == "subview";
}

static bool tcvtNeedsTmp(TCvtOp op, Type srcElem, Type dstElem) {
  if (op.getSatMode() != pto::SaturationMode::OFF) {
    return false;
  }
  return (srcElem.isF32() && dstElem.isInteger(16)) ||
         (srcElem.isF16() &&
          (dstElem.isInteger(16) || dstElem.isInteger(8)));
}

static int64_t computeTCvtTmpRequiredBytes(Type srcElem, Type dstElem,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> dstValid) {
  int64_t rows = dstValid[0], cols = dstValid[1];
  int64_t requiredBytes = 0;
  if (rows > 0 && cols > 0 && srcElem.isF32()) {
    int64_t head = 4 * 64 * std::min<int64_t>(cols / 64, 255);
    int64_t remainder = cols % 64;
    int64_t tail = remainder == 0
                       ? 0
                       : 32 * ((std::min<int64_t>(rows, 255) - 1) *
                                   (srcShape[1] / 8) +
                               llvm::divideCeil(remainder, int64_t{8}));
    requiredBytes = std::max(head, tail);
  } else if (cols > 0 && srcElem.isF16()) {
    int64_t width = std::min<int64_t>(cols, 64);
    int64_t halfToI16 = 32 * llvm::divideCeil(width, int64_t{8});
    int64_t halfToI8 = std::max<int64_t>(
        halfToI16,
        128 + 32 * static_cast<int64_t>(
                       llvm::divideCeil(width, int64_t{16})));
    requiredBytes = dstElem.isInteger(8) ? halfToI8 : halfToI16;
  }
  return requiredBytes;
}

static LogicalResult verifyTCvtTmp(TCvtOp op, Type srcTy, Type dstTy,
                                   Type srcElem, Type dstElem) {
  if (!op.getTmp()) {
    return success();
  }
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!tcvtNeedsTmp(op, srcElem, dstElem)) {
    return success();
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcShape.size() != 2 || dstValid.size() != 2 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) ||
      llvm::is_contained(dstValid, ShapedType::kDynamic)) {
    return op.emitOpError(
        "expects static src shape and dst valid_shape to verify tcvt tmp");
  }
  int64_t requiredBytes =
      computeTCvtTmpRequiredBytes(srcElem, dstElem, srcShape, dstValid);
  auto tmpBytes = getStaticByteSize(tmpTy);
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(requiredBytes)) {
    return op.emitOpError()
           << "expects tcvt tmp capacity to be at least " << requiredBytes
           << " bytes";
  }
  return success();
}

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true)) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
    return failure();
  }
  // A resolved subview keeps its parent's physical shape so the generated
  // Tile retains the parent stride. Its valid shape is the logical tcvt
  // extent, so comparing physical shapes would reject a valid sliced tile.
  if (!tcvtIsResolvedSubview(getSrc()) && !tcvtIsResolvedSubview(getDst()) &&
      failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameLogicalExtent(*this, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (isPTOLowPrecisionType(srcElem) || isPTOLowPrecisionType(dstElem)) {
      return emitOpError("expects A2/A3 tcvt low-precision element types to be unsupported");
    }
    return verifyTCvtTmp(*this, srcTy, dstTy, srcElem, dstElem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!isA5SupportedTCvtPair(srcElem, dstElem)) {
      return emitOpError("expects A5 tcvt low-precision type pairs to match PTO-ISA support");
    }
    if (getTmp() && failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

llvm::LogicalResult mlir::pto::TRandomOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("trandom is only supported for A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects dst to use row-major layout");
    }

    Type elemTy = getElemTy(dstTy);
    if (!elemTy.isInteger(32)) {
      return emitOpError("expects dst element type to be i32 or ui32");
    }

    auto checkWord = [&](Value v, StringRef name) -> LogicalResult {
      auto ty = dyn_cast<IntegerType>(v.getType());
      if (!ty || ty.getWidth() != 32) {
        return emitOpError() << "expects " << name << " to be i32/ui32";
      }
      return success();
    };
    if (failed(checkWord(getKey0(), "key0")) ||
        failed(checkWord(getKey1(), "key1")) ||
        failed(checkWord(getCounter0(), "counter0")) ||
        failed(checkWord(getCounter1(), "counter1")) ||
        failed(checkWord(getCounter2(), "counter2")) ||
        failed(checkWord(getCounter3(), "counter3"))) {
      return failure();
    }

    int32_t rounds = getRounds();
    if (rounds != 7 && rounds != 10) {
      return emitOpError("expects rounds to be 7 or 10");
    }

    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32())) {
      return emitOpError("expects A2/A3 tdiv element type to be f16 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto elem0 = *elemOr;
    if (!(elem0.isF16() || elem0.isF32() || elem0.isInteger(16) || elem0.isInteger(32))) {
      return emitOpError("expects A5 tdiv element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto isTileLike = [](Type ty) -> bool {
    return isa<mlir::pto::TileBufType, RankedTensorType,
               mlir::pto::PartitionTensorViewType>(ty);
  };
  auto isScalarLike = [](Type ty) -> bool {
    return mlir::isa<IntegerType, FloatType>(ty);
  };

  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type rhsTy = getScalar().getType();
    Type dstTy = getDst().getType();

    bool srcTile = isTileLike(srcTy);
    bool rhsTile = isTileLike(rhsTy);
    bool srcScalar = isScalarLike(srcTy);
    bool rhsScalar = isScalarLike(rhsTy);

    if (!(srcTile && rhsScalar) && !(srcScalar && rhsTile)) {
      return emitOpError("expects one tile-like operand and one scalar operand in ins(...)");
    }

    Type tileTy = srcTile ? srcTy : rhsTy;
    Type scalarTy = srcTile ? rhsTy : srcTy;

    if (failed(verifyScalarTileOp(*this, tileTy, dstTy, "src", "dst",
                                  /*requireValidRowsEqual=*/true,
                                  /*requireValidColsEqual=*/true))) {
      return failure();
    }
    if (!mlir::isa<IntegerType, FloatType>(scalarTy)) {
      return emitOpError("scalar must be a scalar type (integer/float)");
    }
    Type elem = getElemTy(tileTy);
    if (targetArch == PTOArch::A3 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return emitOpError("expects A2/A3 tdivs element type to be i32/i16/f16/f32");
    }
    if (targetArch == PTOArch::A5 &&
        !(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tdivs element type to be i32/i16/i8/f16/f32");
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                    /*allowBf16=*/false, /*allowInt8=*/false))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type srcElem = getElemTy(srcTy);
    if (!srcElem.isF16() && !srcElem.isF32()) {
      return emitOpError("expects element type to be f16 or f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem) {
      return emitOpError("expects scalar type == dst element type");
    }
    if (*dstSpace == pto::AddressSpace::VEC && !isRowMajorTileBuf(dstTy)) {
      return emitOpError("expects vec dst to use row-major layout on A2/A3");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 16 || w == 32) {
        return mlir::success();
      }
    }
    return emitOpError("expects A2/A3 texpands dst element type to be i16/i32/f16/bf16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || (*dstSpace != pto::AddressSpace::VEC &&
                      *dstSpace != pto::AddressSpace::MAT)) {
      return emitOpError("expects dst to be in the vec or mat address space");
    }
    Type dstElem = getElemTy(dstTy);
    Type scalarTy = getScalar().getType();
    if (scalarTy != dstElem) {
      return emitOpError("expects scalar type == dst element type");
    }
    if (dstElem.isF16() || dstElem.isBF16() || dstElem.isF32()) {
      return mlir::success();
    }
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
      unsigned w = it.getWidth();
      if (w == 8 || w == 16 || w == 32) {
        return mlir::success();
      }
    }
    return emitOpError("expects A5 texpands dst element type to be i8/i16/i32/f16/bf16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isA2A3AccCastExtractTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  }
  return false;
}

static bool isA5AccCastExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(32);
  }
  return false;
}

static bool isA5AccQuantExtractTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16() ||
           dstElem.isF32() ||
           (llvm::isa<FloatType>(dstElem) &&
            llvm::cast<FloatType>(dstElem).getWidth() == 8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static bool hasMatExtractSourceLayoutA2A3(pto::TileBufType srcTy) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  return bl == static_cast<int32_t>(pto::BLayout::RowMajor) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool hasMatExtractSourceLayoutA5(pto::TileBufType srcTy,
                                        pto::AddressSpace dstSpace) {
  int32_t bl = srcTy.getBLayoutValueI32();
  int32_t sl = srcTy.getSLayoutValueI32();
  if (dstSpace == pto::AddressSpace::LEFT) {
    return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
           (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
            sl == static_cast<int32_t>(pto::SLayout::RowMajor)) ||
           bl == static_cast<int32_t>(pto::BLayout::RowMajor);
  }
  return (bl == static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::ColMajor)) ||
         (bl != static_cast<int32_t>(pto::BLayout::RowMajor) &&
          sl == static_cast<int32_t>(pto::SLayout::RowMajor));
}

static bool isA2A3ExtractElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

static bool isA5ExtractElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isRowMajorNoneBoxND(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::NoneBox);
}

struct TExtractCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TExtractCommon> verifyTExtractCommon(TExtractOp op,
                                                      bool allowLowPrecision) {
  const bool hasFp = static_cast<bool>(op.getFp());
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb) {
    return op.emitOpError("expects src and dst to be !pto.tile_buf");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(),
          /*includeIndexAndIntOpsInConstFold=*/hasFp)) ||
      failed(verifyExtractStaticBoundsCommon(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(), srcTy, dstTy,
          /*includeIndexAndIntOpsInConstFold=*/hasFp))) {
    return failure();
  }
  if (hasFp) {
    Type fpTy = op.getFp().getType();
    if (failed(verifyTileBufCommon(op, fpTy, "fp", allowLowPrecision))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to use loc=scaling");
    }
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("expects src and dst to have element types");
  }
  if ((!srcSpace || *srcSpace != pto::AddressSpace::ACC) && srcElem != dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  return TExtractCommon{srcTy, dstTy, srcTb, dstTb,
                        srcElem, dstElem, srcSpace, dstSpace};
}

static LogicalResult
verifyTExtractFpFormLoc(TExtractOp op,
                        std::optional<pto::AddressSpace> srcSpace) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("expects fp and preQuantScalar to be mutually exclusive");
  }
  if (hasFp && !srcIsAcc) {
    return op.emitOpError("expects fp form to use loc=acc src");
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && !srcIsAcc) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Acc(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use loc=mat");
  }
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A2/A3 acc-source textract dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError("expects A2/A3 acc-source textract dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 acc textract element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTExtractA2A3Mat(TExtractOp op,
                                           const TExtractCommon &c) {
  if (*c.srcSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 textract src to use loc=mat, loc=acc, or loc=vec");
  }
  if (*c.dstSpace != pto::AddressSpace::LEFT &&
      *c.dstSpace != pto::AddressSpace::RIGHT) {
    return op.emitOpError("expects A2/A3 textract dst to use loc=left, loc=right, loc=mat, or loc=vec");
  }
  if (!hasMatExtractSourceLayoutA2A3(c.srcTb)) {
    return op.emitOpError("expects A2/A3 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A2/A3 left dst to use row_major blayout and row_major slayout");
    }
  } else {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A2/A3 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA2A3(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA2A3ExtractElemType(c.dstElem) && !(hasFp && c.dstElem.isInteger(16))) {
    return op.emitOpError("expects A2/A3 textract element type to be i8/f16/bf16/f32");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr()) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec->vec textract to use the base form without preQuantScalar or reluPreMode");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA2A3Acc(op, c);
  }
  return verifyTExtractA2A3Mat(op, c);
}

static bool isTExtractA5SupportedPair(pto::AddressSpace srcSpace,
                                      pto::AddressSpace dstSpace) {
  return (srcSpace == pto::AddressSpace::MAT &&
          (dstSpace == pto::AddressSpace::LEFT ||
           dstSpace == pto::AddressSpace::RIGHT ||
           dstSpace == pto::AddressSpace::SCALING)) ||
         (srcSpace == pto::AddressSpace::VEC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC)) ||
         (srcSpace == pto::AddressSpace::ACC &&
          (dstSpace == pto::AddressSpace::MAT ||
           dstSpace == pto::AddressSpace::VEC));
}

static LogicalResult verifyTExtractA5Mat(TExtractOp op,
                                         const TExtractCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError("expects mat-source textract to use the base form without preQuantScalar or reluPreMode");
  }
  if (!hasMatExtractSourceLayoutA5(c.srcTb, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract src to use a supported mat blayout/slayout combination");
  }
  if (*c.dstSpace == pto::AddressSpace::LEFT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 left dst to use col_major blayout and row_major slayout");
    }
  } else if (*c.dstSpace == pto::AddressSpace::RIGHT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::ColMajor)) {
      return op.emitOpError("expects A5 right dst to use row_major blayout and col_major slayout");
    }
  }
  return success();
}

static LogicalResult verifyTExtractA5Acc(TExtractOp op,
                                         const TExtractCommon &c) {
  if (c.srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
      c.srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects A5 acc-source textract src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (c.dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        c.dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError("expects A5 acc-source textract dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    if (!isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError("expects A5 acc->vec textract dst to use ND layout (blayout=row_major, slayout=none_box)");
    }
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA5AccQuantExtractTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A5 acc preQuantScalar textract element types to be "
          "(src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)");
    }
  } else if (!isA5AccCastExtractTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A5 acc textract element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)");
  }
  return success();
}

static LogicalResult verifyTExtractA5(TExtractOp op) {
  auto common = verifyTExtractCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TExtractCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (!isA5ExtractElemType(c.dstElem)) {
    return op.emitOpError("expects A5 textract element type to be an fp8/f16/bf16/f32 or int8 family type");
  }
  if (failed(verifyTExtractFpFormLoc(op, c.srcSpace))) {
    return failure();
  }
  if (op.getAccToVecModeAttr() &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec textract forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects src and dst to have explicit loc");
  }
  if (!isTExtractA5SupportedPair(*c.srcSpace, *c.dstSpace)) {
    return op.emitOpError("expects A5 textract to use a supported src/dst loc pair");
  }
  if (*c.srcSpace == pto::AddressSpace::MAT) {
    return verifyTExtractA5Mat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError("expects vec-source textract to use the base form without preQuantScalar or reluPreMode");
    }
    if (!isRowMajorNoneBoxND(c.srcTb) || !isRowMajorNoneBoxND(c.dstTb)) {
      return op.emitOpError(
          "expects A5 vec->vec textract src/dst to use ND layout "
          "(blayout=row_major, slayout=none_box)");
    }
    return success();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC) {
    return verifyTExtractA5Acc(op, c);
  }
  return success();
}

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTExtractA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTExtractA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem);
static bool isA2A3AccCastInsertTypePair(Type srcElem, Type dstElem) {
  return srcElem.isF32() && (dstElem.isF16() || dstElem.isBF16());
}

static bool isA2A3AccQuantInsertTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8);
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isInteger(16);
  }
  return false;
}

static bool isColMajorRowMajorNZ(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5SupportedVecElemType(Type ty) {
  if (isPTOFloat8Type(ty) || isPTOHiFloat8Type(ty) || isPTOFloat4PackedType(ty)) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    return it.getWidth() == 8 || it.getWidth() == 32;
  }
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8 || ft.isF16() || ft.isBF16() || ft.isF32();
  }
  return false;
}

static bool isA2A3VecInsertElemType(Type ty) {
  return ty.isInteger(8) || ty.isF16() || ty.isBF16() || ty.isF32();
}

struct TInsertCommon {
  Type srcTy;
  Type dstTy;
  pto::TileBufType srcTb;
  pto::TileBufType dstTb;
  Type srcElem;
  Type dstElem;
  std::optional<pto::AddressSpace> srcSpace;
  std::optional<pto::AddressSpace> dstSpace;
};

static FailureOr<TInsertCommon> verifyTInsertCommon(TInsertOp op,
                                                    bool allowLowPrecision) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!srcTb || !dstTb) {
    return op.emitOpError("expects src and dst to be !pto.tile_buf");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowLowPrecision)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowLowPrecision)) ||
      failed(verifyNonNegativeIndexRowCol(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(),
          /*includeIndexAndIntOpsInConstFold=*/true)) ||
      failed(verifyInsertStaticBoundsCommon(
          *op.getOperation(), op.getIndexRow(), op.getIndexCol(), srcTy, dstTy,
          /*includeIndexAndIntOpsInConstFold=*/true))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  return TInsertCommon{srcTy, dstTy, srcTb, dstTb,
                       srcElem, dstElem, srcSpace, dstSpace};
}

static LogicalResult
verifyTInsertOptionalFp(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                        bool isA5) {
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool reluNonDefault = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool srcIsAcc = srcSpace && *srcSpace == pto::AddressSpace::ACC;
  if (hasFp && hasPreQuantScalar) {
    return op.emitOpError("fp and preQuantScalar are mutually exclusive");
  }
  if (hasFp) {
    if (!srcIsAcc) {
      return op.emitOpError("fp is only valid with src loc=acc");
    }
    auto fpTy = op.getFp().getType();
    auto fpTb = dyn_cast<pto::TileBufType>(fpTy);
    if (!fpTb) {
      return op.emitOpError("expects fp to be !pto.tile_buf");
    }
    if (failed(verifyTileBufCommon(op, fpTy, "fp", /*allowLowPrecision=*/isA5))) {
      return failure();
    }
    auto fpSpace = getPTOMemorySpaceEnum(fpTy);
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError("expects fp to be loc=scaling");
    }
  }
  if (hasPreQuantScalar && !srcIsAcc) {
    return op.emitOpError("preQuantScalar is only valid with src loc=acc");
  }
  if (reluNonDefault && !srcIsAcc) {
    return op.emitOpError("reluPreMode is only valid with src loc=acc");
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalAttrs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                           std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasAccToVecMode) {
    if (!isA5) {
      return op.emitOpError("accToVecMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::ACC ||
        *dstSpace != pto::AddressSpace::VEC)
      return op.emitOpError("accToVecMode is only valid with src=acc, dst=vec");
  }
  if (hasInsertMode) {
    if (!isA5) {
      return op.emitOpError("tinsertMode is only supported on A5");
    }
    if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
        *dstSpace != pto::AddressSpace::MAT) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) is only valid with src=vec, dst=mat");
    }
    auto srcTb = dyn_cast<pto::TileBufType>(op.getSrc().getType());
    if (!srcTb || !isColMajorRowMajorNZ(srcTb)) {
      return op.emitOpError(
          "tinsertMode (SPLIT2/SPLIT4) requires src NZ layout "
          "(blayout=col_major, slayout=row_major)");
    }
  }
  return success();
}

static LogicalResult
verifyTInsertOptionalArgs(TInsertOp op, std::optional<pto::AddressSpace> srcSpace,
                          std::optional<pto::AddressSpace> dstSpace, bool isA5) {
  if (failed(verifyTInsertOptionalFp(op, srcSpace, isA5))) {
    return failure();
  }
  return verifyTInsertOptionalAttrs(op, srcSpace, dstSpace, isA5);
}

static LogicalResult verifyTInsertA2A3AccMat(TInsertOp op,
                                             const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A2/A3 tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A2/A3 tinsert dst to use blayout=col_major and slayout=row_major");
  }
  if (c.dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError("expects A2/A3 tinsert dst fractal size to be 512");
  }
  const bool hasFp = static_cast<bool>(op.getFp());
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  if (hasFp || hasPreQuantScalar) {
    if (!isA2A3AccQuantInsertTypePair(c.srcElem, c.dstElem)) {
      return op.emitOpError(
          "expects A2/A3 acc fp/preQuantScalar tinsert element types to be "
          "(src=f32,dst=i8) or (src=i32,dst=i8/f16/i16)");
    }
  } else if (!isA2A3AccCastInsertTypePair(c.srcElem, c.dstElem)) {
    return op.emitOpError(
        "expects A2/A3 tinsert element types to be src=f32, dst=f16/bf16");
  }
  return success();
}

static LogicalResult verifyTInsertA2A3(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/false))) {
    return failure();
  }
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
      *c.dstSpace == pto::AddressSpace::VEC) {
    if (hasPreQuantScalar || hasRelu) {
      return op.emitOpError(
          "expects vec->vec tinsert to use the base form without "
          "preQuantScalar or reluPreMode");
    }
    if (c.srcElem != c.dstElem || !isA2A3VecInsertElemType(c.srcElem)) {
      return op.emitOpError(
          "expects A2/A3 vec->vec tinsert src/dst to have same supported dtype "
          "(i8/f16/bf16/f32)");
    }
    return success();
  }
  if (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
      *c.dstSpace != pto::AddressSpace::MAT) {
    return op.emitOpError("expects A2/A3 tinsert to use acc->mat or vec->vec");
  }
  return verifyTInsertA2A3AccMat(op, c);
}

static LogicalResult verifyTInsertA5Acc(TInsertOp op, const TInsertCommon &c) {
  if (!isColMajorRowMajorNZ(c.srcTb)) {
    return op.emitOpError("expects A5 acc->mat tinsert src to use blayout=col_major and slayout=row_major");
  }
  if (*c.dstSpace == pto::AddressSpace::MAT) {
    if (!isColMajorRowMajorNZ(c.dstTb)) {
      return op.emitOpError("expects A5 acc->mat tinsert dst to use blayout=col_major and slayout=row_major");
    }
  } else {
    bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
    bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
    if (!dstIsND && !dstIsNZ) {
      return op.emitOpError(
          "expects A5 acc->vec tinsert dst to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
    }
  }
  const bool hasQuant =
      static_cast<bool>(op.getFp()) || static_cast<bool>(op.getPreQuantScalar());
  bool okTypes;
  if (hasQuant) {
    okTypes = isA5VectorPreQuantTypePair(c.srcElem, c.dstElem);
  } else {
    okTypes = (c.srcElem.isF32() &&
               (c.dstElem.isF16() || c.dstElem.isBF16() || c.dstElem.isF32())) ||
              (c.srcElem.isInteger(32) && c.dstElem.isInteger(32));
  }
  if (!okTypes) {
    return op.emitOpError(
        "expects A5 acc-source tinsert element types to be "
        "(src=f32,dst=f16/bf16/f32) or (src=i32,dst=i32)" +
        (hasQuant ? std::string("; with fp/scalar: (src=f32,dst=i8/fp8/f16/bf16/f32) or (src=i32,dst=i8/f16/bf16)") : std::string()));
  }
  return success();
}

static LogicalResult verifyTInsertA5VecMat(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->mat tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  if (!isColMajorRowMajorNZ(c.dstTb)) {
    return op.emitOpError("expects A5 vec->mat tinsert dst to use blayout=col_major and slayout=row_major");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  if (!srcIsND && !srcIsNZ) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src to use ND(row_major/none_box) or NZ(col_major/row_major) layout");
  }
  if (hasTInsertMode && !srcIsNZ) {
    return op.emitOpError("expects tinsertMode vec->mat tinsert src to use NZ(col_major/row_major) layout");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->mat tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5VecVec(TInsertOp op, const TInsertCommon &c) {
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  if (hasPreQuantScalar || hasRelu) {
    return op.emitOpError(
        "expects vec->vec tinsert to use the base form without "
        "preQuantScalar or reluPreMode");
  }
  bool srcIsND = isRowMajorNoneBoxND(c.srcTb);
  bool dstIsND = isRowMajorNoneBoxND(c.dstTb);
  bool srcIsNZ = isColMajorRowMajorNZ(c.srcTb);
  bool dstIsNZ = isColMajorRowMajorNZ(c.dstTb);
  if (srcIsND && dstIsND) {
    // ND->ND path
  } else if (srcIsNZ && dstIsNZ) {
    // NZ->NZ path
  } else {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst layouts to match: "
        "both ND(row_major/none_box) or both NZ(col_major/row_major)");
  }
  if (c.srcElem != c.dstElem || !isA5SupportedVecElemType(c.srcElem)) {
    return op.emitOpError(
        "expects A5 vec->vec tinsert src/dst to have same supported dtype "
        "(fp8/f16/bf16/f32/i8/i32)");
  }
  return success();
}

static LogicalResult verifyTInsertA5(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  const bool hasRelu = op.getReluPreMode() != pto::ReluPreMode::NoRelu;
  const bool hasAccToVecMode = static_cast<bool>(op.getAccToVecModeAttr());
  const bool hasTInsertMode = static_cast<bool>(op.getTinsertModeAttr());
  if (hasPreQuantScalar && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects preQuantScalar form to use loc=acc src");
  }
  if (hasRelu && (!c.srcSpace || *c.srcSpace != pto::AddressSpace::ACC)) {
    return op.emitOpError("expects reluPreMode form to use loc=acc src");
  }
  if (hasAccToVecMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::ACC ||
       *c.dstSpace != pto::AddressSpace::VEC)) {
    return op.emitOpError("expects accToVecMode only on A5 acc->vec tinsert forms");
  }
  if (hasTInsertMode &&
      (!c.srcSpace || !c.dstSpace || *c.srcSpace != pto::AddressSpace::VEC ||
       *c.dstSpace != pto::AddressSpace::MAT)) {
    return op.emitOpError("expects tinsertMode only on A5 vec->mat tinsert forms");
  }
  if (!c.srcSpace || !c.dstSpace) {
    return op.emitOpError("expects A5 tinsert src/dst to have explicit loc");
  }
  if (failed(verifyTInsertOptionalArgs(op, c.srcSpace, c.dstSpace, /*isA5=*/true))) {
    return failure();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC &&
      (*c.dstSpace == pto::AddressSpace::MAT || *c.dstSpace == pto::AddressSpace::VEC)) {
    return verifyTInsertA5Acc(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::MAT) {
    return verifyTInsertA5VecMat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
    return verifyTInsertA5VecVec(op, c);
  }
  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, acc->vec, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTInsertA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTInsertA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty)) {
    return ft.getWidth() == 8;
  }
  return false;
}

static bool isA5MxFp8InputType(Type ty) {
  return ty && isa<Float8E4M3FNType, Float8E5M2Type>(ty);
}

static bool isA5MxInputTypePair(Type lhsTy, Type rhsTy) {
  return (isA5MxFp8InputType(lhsTy) && isA5MxFp8InputType(rhsTy)) ||
         (isPTOFloat4PackedType(lhsTy) && isPTOFloat4PackedType(rhsTy));
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputTypePair(lhsElem, rhsElem)) {
    return op->emitOpError()
           << "expects A5 mx " << lhsName << "/" << rhsName
           << " element types to be a supported fp8/fp8 or fp4/fp4 pair";
  }

  if (!dstElem.isF32()) {
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";
  }

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
    return dstElem.isInteger(8) || isA5Fp8LikeType(dstElem) ||
           isPTOHiFloat8Type(dstElem) || dstElem.isF16() ||
           dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(32)) {
    return dstElem.isInteger(8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy,
                                              Type dstTy) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op->emitError("expects src/dst to be PTO shaped-like types");
  }

  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (srcShape.size() != 2 || dstShape.size() != 2) {
    return op->emitError("expects rank-2 shaped types for src/dst");
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    unsigned elemBytes = getPTOStorageElemByteSize(t);
    return elemBytes == 0 ? -1 : static_cast<int64_t>(elemBytes);
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0) {
    return op->emitError("unsupported element type (expects int/float element types)");
  }
  if (srcB != dstB) {
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  }
  if (!(srcB == 1 || srcB == 2 || srcB == 4)) {
    return op->emitError("expects element size to be 1, 2, or 4 bytes");
  }

  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);

  bool expanded = false;
  for (auto [srcDim, dstDim] : llvm::zip_equal(srcShape, dstShape)) {
    if (srcDim == dstDim) {
      continue;
    }
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim)) {
      return op->emitError("cannot infer TFILLPAD lowering from mismatched "
                           "dynamic physical shapes");
    }
    if (srcDim > dstDim) {
      return op->emitError(
          "expects each dst physical shape dimension to be >= src");
    }
    expanded = true;
  }
  if (expanded &&
      (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
       *dstSpace != pto::AddressSpace::VEC)) {
    return op->emitError("expects expanded TFILLPAD only for loc=vec");
  }

  // pto.tfillpad lowers to TFILLPAD(dst, src). For loc=mat, pto-isa only
  // exposes the homogeneous overload, so src/dst must use the same Tile<...>
  // specialization (including valid_shape and pad).
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  if (srcTb && dstTb && srcSpace && dstSpace &&
      *srcSpace == mlir::pto::AddressSpace::MAT &&
      *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
    auto dimToStr = [](int64_t dim) -> std::string {
      return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
    };
    SmallVector<std::string, 4> mismatchFields;
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() == 2 && dstValid.size() == 2) {
      if (srcValid[0] != dstValid[0]) {
        mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                 dimToStr(dstValid[0]) + ")");
      }
      if (srcValid[1] != dstValid[1]) {
        mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                 dimToStr(dstValid[1]) + ")");
      }
    }
    if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
      mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                               " vs " + std::to_string(dstTb.getPadValueI32()) +
                               ")");
    }

    auto diag = op->emitError()
                << "expects src/dst tile types to be lowerable to TFILLPAD "
                   "for loc=mat";
    if (!mismatchFields.empty()) {
      diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
    }
    diag << "\n  src: " << srcTy;
    diag << "\n  dst: " << dstTy;
    diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
    return failure();
  }

  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null) {
      return op->emitError("expects dst PadVal != Null for tfillpad");
    }
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  if (getOperation()->getAttr("mode")) {
    return emitOpError("does not accept 'mode'; PTOAS infers TFILLPAD lowering "
                       "from physical shape and planned addresses");
  }

  if (failed(verifyTFillPadLike(getOperation(), getSrc().getType(),
                                getDst().getType()))) {
    return failure();
  }

  if (auto padValueAttr = getPadValueAttr()) {
    auto dstSpace = getPTOMemorySpaceEnum(getDst().getType());
    if (!dstSpace || *dstSpace != pto::AddressSpace::MAT) {
      return emitOpError("expects padValue attribute only for loc=mat tfillpad");
    }
    auto dstTileTy = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!dstTileTy) {
      return emitOpError("expects dst to be tile_buf when padValue is specified");
    }
    if (dstTileTy.getPadValueI32() != static_cast<int32_t>(padValueAttr.getValue())) {
      return emitOpError("expects padValue attribute to match dst tile pad configuration");
    }
  }

  return success();
}


static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (isPTOFloat8Type(ty)) {
    return true;
  }
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == 8 || width == 16 || width == 32;
  }
  return false;
}

static unsigned getMaskGatherTimes(mlir::pto::MaskPatternAttr mp) {
  switch (mp.getValue()) {
  case mlir::pto::MaskPattern::P1111:
    return 1;
  case mlir::pto::MaskPattern::P0101:
  case mlir::pto::MaskPattern::P1010:
    return 2;
  default:
    return 4;
  }
}

static LogicalResult verifyTGatherMaskShapes(TGatherOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tgather to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tgather to provide maskPattern");
  }
  const unsigned times = getMaskGatherTimes(mp);
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != static_cast<int64_t>(dstValid[1] * times)) {
      return op.emitOpError("expects src valid cols to equal dst valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        srcValid[0] != static_cast<int64_t>(dstValid[0] * times)) {
      return op.emitOpError("expects src valid rows to equal dst valid rows times the mask expansion factor for col direction");
    }
  } else {
    return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return success();
}

static LogicalResult verifyTGatherMaskForm(TGatherOp op, bool allowA5MaskTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5MaskTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5MaskTypes))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src and dst to be in the vec address space");
  }
  unsigned srcElemBytes = getPTOStorageElemByteSize(srcElem);
  unsigned dstElemBytes = getPTOStorageElemByteSize(dstElem);
  if (srcElemBytes == 0 || dstElemBytes == 0) {
    return op.emitOpError("failed to get element size for src/dst");
  }
  if (srcElemBytes != dstElemBytes) {
    return op.emitOpError("expects src and dst element sizes to match");
  }
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  if (failed(verifyTGatherMaskShapes(op))) {
    return failure();
  }
  if (allowA5MaskTypes) {
    if (!(srcElemBytes == 1 || srcElemBytes == 2 || srcElemBytes == 4)) {
      return op.emitOpError("expects A5 mask-pattern gather element size to be 1, 2, or 4 bytes");
    }
    if (!isSupportedGatherElemTypeA5(srcElem) || !isSupportedGatherElemTypeA5(dstElem)) {
      return op.emitOpError(
          "expects A5 mask-pattern gather src/dst element type to be i8/i16/i32/f16/bf16/f32/fp8-like");
    }
  } else {
    if (!(srcElemBytes == 2 || srcElemBytes == 4)) {
      return op.emitOpError("expects A2/A3 mask-pattern gather element size to be 2 or 4 bytes");
    }
  }
  return success();
}

static LogicalResult verifyTGatherIndexTypes(TGatherOp op, bool allow16BitIndices,
                                             bool allowA5ElemTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", allowA5ElemTypes)) ||
      failed(verifyTileBufCommon(op, idxTy, "indices"))) {
    return failure();
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return op.emitOpError("failed to get element type for src/dst");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (allowA5ElemTypes) {
    if (!isSupportedGatherElemTypeA5Index(srcElem) ||
        !isSupportedGatherElemTypeA5Index(dstElem)) {
      return op.emitOpError(
          "expects A5 gather src/dst element type to be i8/i16/i32/f16/f32");
    }
  } else if (!isSupportedGatherElemTypeA2A3(srcElem) ||
             !isSupportedGatherElemTypeA2A3(dstElem)) {
    return op.emitOpError("expects gather src/dst element type to be i16/i32/f16/f32");
  }
  auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
  if (!idxElem) {
    return op.emitOpError("indices element type must be integer");
  }
  unsigned width = idxElem.getWidth();
  if (!(width == 32 || (allow16BitIndices && width == 16))) {
    return op.emitOpError() << "expects indices element type to be i32"
                            << (allow16BitIndices ? " or i16" : "");
  }
  return success();
}

static LogicalResult verifyTGatherIndexShapes(TGatherOp op, bool allowA5ElemTypes) {
  Type dstTy = op.getDst().getType();
  Type idxTy = op.getIndices().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto dstShape = getShapeVec(dstTy);
  if (dstValid.size() == 2 && dstShape.size() == 2 &&
      dstValid[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
      dstValid[1] != dstShape[1]) {
    return op.emitOpError("expects dst valid_shape[1] to equal dst cols");
  }
  auto idxValid = getValidShapeVec(idxTy);
  auto idxShape = getShapeVec(idxTy);
  if (idxValid.size() == 2 && idxShape.size() == 2 &&
      idxValid[1] != ShapedType::kDynamic && idxShape[1] != ShapedType::kDynamic &&
      idxValid[1] != idxShape[1]) {
    return op.emitOpError("expects indices valid_shape[1] to equal indices cols");
  }
  if (!allowA5ElemTypes) {
    if (failed(verifyTileBufSameValidShape(op, dstTy, idxTy, "dst", "indices"))) {
      return failure();
    }
    if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(idxTy) ||
        !isRowMajorTileBuf(op.getTmp().getType())) {
      return op.emitOpError(
          "expects A2/A3 index-form dst, indices, and tmp to use row-major layout");
    }
    auto idxElem = dyn_cast<IntegerType>(getElemTy(idxTy));
    Type tmpElem = getElemTy(op.getTmp().getType());
    if (tmpElem != idxElem) {
      return op.emitOpError("expects tmp and indices to have the same element type");
    }
    if (failed(verifyTileBufSameValidShape(op, idxTy, op.getTmp().getType(), "indices", "tmp"))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTGatherIndexForm(TGatherOp op, bool allow16BitIndices,
                                            bool allowA5ElemTypes) {
  if (failed(verifyTGatherIndexTypes(op, allow16BitIndices, allowA5ElemTypes))) {
    return failure();
  }
  return verifyTGatherIndexShapes(op, allowA5ElemTypes);
}

static LogicalResult verifyTGatherCompareSrcType(TGatherOp op, Type srcElem,
                                                 pto::CmpMode cmpMode,
                                                 bool allowA5SrcTypes) {
  if (allowA5SrcTypes) {
    if (!(srcElem.isF16() || srcElem.isF32() || srcElem.isInteger(16) ||
          srcElem.isInteger(32))) {
      return op.emitOpError(
          "expects A5 compare-form tgather src element type to be i16/i32/f16/f32");
    }
  } else {
    if (!(srcElem.isF16() || srcElem.isF32() ||
          (srcElem.isInteger(32) && cmpMode == pto::CmpMode::EQ))) {
      return op.emitOpError(
          "expects A2/A3 compare-form tgather src element type to be f16/f32, or i32 when cmpMode=eq");
    }
  }
  return success();
}

static LogicalResult verifyTGatherCompareForm(TGatherOp op, bool allowA5SrcTypes) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type cdstTy = op.getCdst().getType();
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst")) ||
      failed(verifyTileBufCommon(op, cdstTy, "cdst")) ||
      failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type cdstElem = getElemTy(cdstTy);
  if (!srcElem || !dstElem || !cdstElem) {
    return op.emitOpError("failed to get element type for src/dst/cdst");
  }
  auto dstInt = dyn_cast<IntegerType>(dstElem);
  if (!dstInt || dstInt.getWidth() != 32) {
    return op.emitOpError("expects dst element type to be i32");
  }
  if (cdstElem != dstElem) {
    return op.emitOpError("expects cdst to have the same element type as dst");
  }
  if (op.getKValue().getType() != srcElem) {
    return op.emitOpError("expects kValue to have the same type as src element type");
  }
  auto cmpAttr = op.getCmpModeAttr();
  auto cmpMode = cmpAttr ? cmpAttr.getValue() : pto::CmpMode::EQ;
  if (cmpMode != pto::CmpMode::EQ && cmpMode != pto::CmpMode::GT) {
    return op.emitOpError("expects compare-form tgather cmpMode to be eq or gt");
  }
  if (failed(verifyTGatherCompareSrcType(op, srcElem, cmpMode, allowA5SrcTypes))) {
    return failure();
  }
  if (failed(verifyVecTileCommonA2A3(op, srcTy, "src")) ||
      failed(verifyVecTileCommonA2A3(op, dstTy, "dst")) ||
      failed(verifyVecTileCommonA2A3(op, cdstTy, "cdst")) ||
      failed(verifyVecTileCommonA2A3(op, tmpTy, "tmp"))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTGatherA2A3(TGatherOp op) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue()) {
      return op.emitOpError("mask-pattern tgather only allows src and dst operands");
    }
    return verifyTGatherMaskForm(op, /*allowA5MaskTypes=*/false);
  }
  if (op.getAxisAttr()) {
    return op.emitOpError("axis attribute must not be provided without maskPattern");
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp()) {
      return op.emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
    }
    if (op.getIndices()) {
      return op.emitOpError("compare-form tgather does not take indices");
    }
    return verifyTGatherCompareForm(op, /*allowA5SrcTypes=*/false);
  }
  if (!op.getIndices() || !op.getTmp()) {
    return op.emitOpError("index-form tgather expects both indices and tmp");
  }
  return verifyTGatherIndexForm(op, /*allow16BitIndices=*/false, /*allowA5ElemTypes=*/false);
}

static LogicalResult verifyTGatherA5(TGatherOp op) {
  if (op.getMaskPatternAttr()) {
    if (op.getCdst() || op.getIndices() || op.getTmp() || op.getKValue()) {
      return op.emitOpError("mask-pattern tgather only allows src and dst operands");
    }
    return verifyTGatherMaskForm(op, /*allowA5MaskTypes=*/true);
  }
  if (op.getAxisAttr()) {
    return op.emitOpError("axis attribute must not be provided without maskPattern");
  }
  if (op.getCdst() || op.getKValue()) {
    if (!op.getCdst() || !op.getKValue() || !op.getTmp()) {
      return op.emitOpError("compare-form tgather expects dst, cdst, kValue, and tmp");
    }
    if (op.getIndices()) {
      return op.emitOpError("compare-form tgather does not take indices");
    }
    return verifyTGatherCompareForm(op, /*allowA5SrcTypes=*/true);
  }
  if (!op.getIndices()) {
    return op.emitOpError("index-form tgather expects indices");
  }
  return verifyTGatherIndexForm(op, /*allow16BitIndices=*/true, /*allowA5ElemTypes=*/true);
}

llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
static std::optional<unsigned> tgatherbElemBytes(Type ty) {
  unsigned elemBytes = getPTOStorageElemByteSize(ty);
  if (elemBytes == 0) {
    return std::nullopt;
  }
  return elemBytes;
}

static FailureOr<std::pair<Type, Type>> verifyTGatherBCommon(TGatherBOp op) {
  Type srcTy = op.getSrc().getType();
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, offTy, "offsets")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  auto srcElemTy = getElemTy(srcTy);
  auto dstElemTy = getElemTy(dstTy);
  if (!srcElemTy || !dstElemTy) {
    return op.emitOpError() << "failed to get element type for src/dst";
  }
  return std::make_pair(srcElemTy, dstElemTy);
}

static LogicalResult verifyTGatherBA2A3(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type offTy = op.getOffsets().getType();
  Type dstTy = op.getDst().getType();
  Type dstElemTy = elems->second;
  if (failed(
          verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy) || !isRowMajorTileBuf(offTy)) {
    return op.emitOpError()
           << "expects dst and offsets to use row-major layout";
  }
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError()
           << "expects A2/A3 dst element size to be 2 or 4 bytes";
  }
  Type offElemTy = getElemTy(offTy);
  if (!offElemTy.isInteger(32)) {
    return op.emitOpError() << "expects offsets element type to be i32";
  }

  auto dstValid = getValidShapeVec(dstTy);
  auto offValid = getValidShapeVec(offTy);
  if (dstValid.size() != 2 || offValid.size() != 2) {
    return op.emitOpError() << "expects rank-2 src/offsets/dst tile buffers";
  }
  if (dstValid[0] != ShapedType::kDynamic &&
      offValid[0] != ShapedType::kDynamic && dstValid[0] != offValid[0]) {
    return op.emitOpError() << "expects offsets valid rows to match dst valid rows";
  }
  if (dstValid[1] != ShapedType::kDynamic &&
      offValid[1] != ShapedType::kDynamic) {
    int64_t blockElems = 32 / *dstBytes;
    int64_t blocks = (dstValid[1] + blockElems - 1) / blockElems;
    int64_t expectedOffsetCols = ((blocks + 7) / 8) * 8;
    if (offValid[1] != expectedOffsetCols) {
      return op.emitOpError()
             << "expects offsets valid cols to be compact 32B block address "
                "count padded to 8 entries; expected "
             << expectedOffsetCols << ", got " << offValid[1];
    }
  }
  return mlir::success();
}

static LogicalResult verifyTGatherBA5(TGatherBOp op) {
  FailureOr<std::pair<Type, Type>> elems = verifyTGatherBCommon(op);
  if (failed(elems)) {
    return failure();
  }
  Type dstElemTy = elems->second;
  auto dstBytes = tgatherbElemBytes(dstElemTy);
  if (!dstBytes || (*dstBytes != 1 && *dstBytes != 2 && *dstBytes != 4)) {
    return op.emitOpError() << "expects dst element size to be 1, 2, or 4 bytes";
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TGatherBOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTGatherBA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTGatherBA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  auto elemTy = getElemTy(srcTy);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto valid = getValidShapeVec(srcTy);
    if (valid.size() != 2) {
      return emitOpError("expects src to have rank-2 valid_shape");
    }
    if (valid[0] != ShapedType::kDynamic && valid[0] < 0) {
      return emitOpError("expects src valid_shape[0] to be non-negative");
    }
    if (valid[1] != ShapedType::kDynamic && valid[1] < 0) {
      return emitOpError("expects src valid_shape[1] to be non-negative");
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << "expects A2/A3 tlrelu element type to be f16 or f32";
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << "expects A5 tlrelu element type to be f16 or f32";
    }
    if (!getSlope().getType().isF32()) {
      return emitOpError() << "expects slope to have type f32";
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmax element type to be i32/i16/f16/f32",
      "expects A5 tmax element type to be i32/i16/i8/f16/f32");
}

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmaxs element type to be i32/i16/f16/f32",
      "expects A5 tmaxs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmin element type to be i32/i16/f16/f32",
      "expects A5 tmin element type to be i32/i16/i8/f16/bf16/f32");
}

mlir::LogicalResult mlir::pto::TMinSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmins element type to be i32/i16/f16/f32",
      "expects A5 tmins element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

static std::optional<int64_t> tmovCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> tmovCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> tmovAlign16(int64_t value) {
  auto biased = tmovCheckedAdd(value, 15);
  return biased ? tmovCheckedMul(*biased / 16, 16) : std::nullopt;
}

static std::optional<int64_t> tmovCheckedElements(ArrayRef<int64_t> shape) {
  if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0 ||
      (shape[1] != 0 &&
       shape[0] > std::numeric_limits<int64_t>::max() / shape[1])) {
    return std::nullopt;
  }
  return shape[0] * shape[1];
}

static LogicalResult verifyTMovXToZzElemLayout(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  auto srcTb = cast<pto::TileBufType>(srcTy);
  auto dstTb = cast<pto::TileBufType>(dstTy);
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  Type tmpElem = getElemTy(fp.getType());
  if (srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError("expects src, dst, and tmp to share one element type");
  }
  bool validElem = isPTOHiFloat8Type(srcElem) || isPTOF8E8M0Type(srcElem);
  if (auto integer = dyn_cast<IntegerType>(srcElem)) {
    validElem = integer.getWidth() == 8 &&
                integer.getSignedness() == IntegerType::Unsigned;
  }
  if (!validElem) {
    return op.emitOpError("expects element type to be one of ui8, !pto.hif8, !pto.f8E8M0 (i8 lowers to int8_t, which PTO-ISA CommonCheckZZ rejects)");
  }
  if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
    return op.emitOpError("expects src to use blayout=row_major, slayout=none_box");
  }
  if (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError("expects dst to use blayout=row_major, slayout=row_major (ZZ box)");
  }
  return success();
}

static LogicalResult verifyTMovXToZzForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (!isA5) {
    return op.emitOpError("X-to-ZZ tmov is only supported on A5");
  }
  if (op.getNumResults() != 0) {
    return op.emitOpError("expects X-to-ZZ tmov not to have results");
  }
  if (op.getPreQuantScalar() || op.getAccToVecModeAttr() ||
      op.getReluPreMode() != pto::ReluPreMode::NoRelu) {
    return op.emitOpError("expects the X-to-ZZ tmov form not to use preQuantScalar, accToVecMode, or reluPreMode");
  }

  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  auto tmpTb = dyn_cast<pto::TileBufType>(fp.getType());
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  auto tmpSpace = getPTOMemorySpaceEnum(fp.getType());
  if (!srcTb || !dstTb || !tmpTb || !srcSpace || !dstSpace ||
      *srcSpace != pto::AddressSpace::VEC ||
      *dstSpace != pto::AddressSpace::VEC ||
      *tmpSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects X-to-ZZ src/dst/tmp to be vec tiles");
  }
  if (op.getSrc() == op.getDst() || op.getSrc() == fp || op.getDst() == fp) {
    return op.emitOpError("expects X-to-ZZ src, dst, and tmp to be distinct tile values");
  }
  if (srcTb.getRank() != 2 || dstTb.getRank() != 2 || tmpTb.getRank() != 2) {
    return op.emitOpError("expects rank-2 valid_shape for src/dst/tmp");
  }
  auto hasDynamic = [](ArrayRef<int64_t> shape) {
    return llvm::is_contained(shape, ShapedType::kDynamic);
  };
  if (hasDynamic(getValidShapeVec(srcTy)) || hasDynamic(getShapeVec(srcTy)) ||
      hasDynamic(getValidShapeVec(dstTy)) || hasDynamic(getShapeVec(dstTy)) ||
      hasDynamic(getShapeVec(fp.getType()))) {
    return op.emitOpError("expects static valid and physical shapes for src/dst and a static tmp physical shape for X-to-ZZ");
  }
  return verifyTMovXToZzElemLayout(op);
}

static LogicalResult verifyTMovXToZzAxis1(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> dstValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  if (dstValid[1] % 2 != 0) {
    return op.emitOpError("expects ND-to-ZZ dst valid_shape[1] (the grouped exponent column count) to be even");
  }
  if (srcValid[0] != 1 && srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects ND-to-ZZ src valid elements to form a compact prefix (single-row legacy flat or physical row stride equal to valid cols)");
  }
  auto paddedRows = tmovAlign16(dstValid[0]);
  auto required =
      paddedRows ? tmovCheckedMul(*paddedRows, dstValid[1]) : std::nullopt;
  if (!required) {
    return op.emitOpError("cannot compute ND-to-ZZ padded capacity without overflow");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  if (!srcBytes || *srcBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ src physical capacity to cover align16(dst rows) * dst cols because source padding is zeroed in place");
  }
  if (!dstBytes || *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects ND-to-ZZ dst physical capacity to cover align16(dst rows) * dst cols");
  }
  auto rowBlocksBias = tmovCheckedAdd(dstValid[0], 15);
  auto offsetBytes = rowBlocksBias
                         ? tmovCheckedMul(*rowBlocksBias / 16, dstValid[1])
                         : std::nullopt;
  auto tmpRequired =
      offsetBytes ? tmovCheckedAdd(64, *offsetBytes) : std::nullopt;
  if (!tmpRequired) {
    return op.emitOpError("cannot compute ND-to-ZZ tmp capacity without overflow");
  }
  auto tmpBytes = getStaticByteSize(fp.getType());
  if (!tmpBytes || *tmpBytes < static_cast<uint64_t>(*tmpRequired)) {
    return op.emitOpError() << "expects tmp to provide at least " << *tmpRequired
                            << " bytes for ND-to-ZZ (64 + ceil(dst rows / 16) * dst cols)";
  }
  return success();
}

static LogicalResult verifyTMovXToZzAxis0(TMovOp op, ArrayRef<int64_t> srcValid,
                                          ArrayRef<int64_t> srcPhysical) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (srcValid[0] < 2 || srcValid[0] % 2 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[0] to be an even count >= 2; a single row-group produces no output in PTO-ISA");
  }
  if (srcValid[1] % 16 != 0) {
    return op.emitOpError("expects DN-to-ZZ src valid_shape[1] to be a multiple of 16");
  }
  if (srcPhysical[1] != srcValid[1]) {
    return op.emitOpError("expects DN-to-ZZ src physical row stride to equal src valid_shape[1]");
  }
  auto srcBytes = getStaticByteSize(srcTy);
  auto dstBytes = getStaticByteSize(dstTy);
  auto required = tmovCheckedMul(srcValid[0], srcValid[1]);
  if (!required || !srcBytes || !dstBytes ||
      *srcBytes < static_cast<uint64_t>(*required) ||
      *dstBytes < static_cast<uint64_t>(*required)) {
    return op.emitOpError("expects DN-to-ZZ src/dst physical capacity to cover src valid rows * src valid cols");
  }
  return success();
}

static LogicalResult verifyTMovXToZzCapacity(TMovOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto srcPhysical = getShapeVec(srcTy);
  auto srcElements = tmovCheckedElements(srcValid);
  auto dstElements = tmovCheckedElements(dstValid);
  if (!srcElements || !dstElements || *srcElements != *dstElements) {
    return op.emitOpError("expects src and dst to hold the same exponent count");
  }
  const MxGroupAxis axis =
      op.getGrpAxisAttr() ? op.getGrpAxisAttr().getValue() : MxGroupAxis::Axis1;
  if (axis == MxGroupAxis::Axis1) {
    return verifyTMovXToZzAxis1(op, srcValid, dstValid, srcPhysical);
  }
  return verifyTMovXToZzAxis0(op, srcValid, srcPhysical);
}

static LogicalResult verifyTMovXToZz(TMovOp op, bool isA5) {
  if (failed(verifyTMovXToZzForm(op, isA5))) {
    return failure();
  }
  return verifyTMovXToZzCapacity(op);
}

static LogicalResult verifyTMovGenericPreconditions(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  if (op.getGrpAxisAttr()) {
    return op.emitOpError("expects grpAxis only on the X-to-ZZ form with a non-scaling third tile");
  }
  if (failed(verifyTileBufCommon(op, srcTy, "src", /*allowLowPrecision=*/isA5)) ||
      failed(verifyTileBufCommon(op, dstTy, "dst", /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && failed(verifyTileBufCommon(op, fp.getType(), "fp",
                                          /*allowLowPrecision=*/isA5))) {
    return failure();
  }
  if (hasFp && op.getPreQuantScalar()) {
    return op.emitOpError() << "expects fp and preQuantScalar forms to be mutually exclusive";
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (!srcSpace || !dstSpace) {
    return op.emitOpError() << "expects src and dst to have explicit address spaces";
  }
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  if (*srcSpace == pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects mat-source tmov to use matching src/dst shapes";
  }
  if (!isA5 && *srcSpace != pto::AddressSpace::MAT && srcShape != dstShape) {
    return op.emitOpError() << "expects A2/A3 non-mat tmov to use matching src/dst shapes";
  }
  return success();
}

static LogicalResult verifyTMovGenericPairing(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isMatToTile =
      *srcSpace == pto::AddressSpace::MAT &&
      (*dstSpace == pto::AddressSpace::LEFT ||
       *dstSpace == pto::AddressSpace::RIGHT ||
       *dstSpace == pto::AddressSpace::BIAS ||
       *dstSpace == pto::AddressSpace::SCALING);
  const bool isVecToVec = *srcSpace == pto::AddressSpace::VEC &&
                          *dstSpace == pto::AddressSpace::VEC;
  const bool isVecToMat = *srcSpace == pto::AddressSpace::VEC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  bool okPair = isMatToTile || isVecToVec || isAccToMat || isAccToVec;
  if (isA5) {
    okPair = okPair || isVecToMat;
  }
  if (!okPair) {
    return op.emitOpError() << "expects a supported tmov address-space pair for this target";
  }
  if (op.getAccToVecModeAttr() && !isAccToVec) {
    return op.emitOpError() << "expects accToVecMode to be used only for acc-to-vec tmov";
  }
  if (op.getReluPreMode() != pto::ReluPreMode::NoRelu &&
      !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects reluPreMode form to use loc=acc src";
  }
  if (op.getPreQuantScalar() && !(isAccToMat || isAccToVec)) {
    return op.emitOpError() << "expects preQuantScalar form to use loc=acc src";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpLayout(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  const bool hasFp = static_cast<bool>(op.getFp());
  auto reluMode = op.getReluPreMode();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  auto srcTb = dyn_cast<pto::TileBufType>(srcTy);
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (srcTb && *srcSpace == pto::AddressSpace::ACC &&
      (hasFp || reluMode != pto::ReluPreMode::NoRelu)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
        srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor)) {
      return op.emitOpError() << "expects acc-source fp/relu tmov src to use blayout=col_major and slayout=row_major";
    }
  }
  if (hasFp && !isA5 && dstTb && isAccToMat &&
      (dstTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
       dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::RowMajor))) {
    return op.emitOpError() << "expects fp tmov dst to use blayout=col_major and slayout=row_major";
  }
  if (srcTb && dstTb && isAccToMat && !isA5 &&
      dstTb.getSFractalSizeI32() != 512) {
    return op.emitOpError() << "expects A2/A3 acc-to-mat tmov destination fractal to be 512";
  }
  return success();
}

static LogicalResult verifyTMovGenericFpForm(TMovOp op, bool isA5) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Value fp = op.getFp();
  const bool hasFp = static_cast<bool>(fp);
  const bool hasPreQuantScalar = static_cast<bool>(op.getPreQuantScalar());
  auto accToVecModeAttr = op.getAccToVecModeAttr();
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  const bool isAccToMat = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::MAT;
  const bool isAccToVec = *srcSpace == pto::AddressSpace::ACC &&
                          *dstSpace == pto::AddressSpace::VEC;
  if (hasFp) {
    auto fpSpace = getPTOMemorySpaceEnum(fp.getType());
    if (!fpSpace || *fpSpace != pto::AddressSpace::SCALING) {
      return op.emitOpError() << "expects fp to be in the scaling address space";
    }
    auto srcElemTy = getElemTy(srcTy);
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!(srcElemTy.isF32() || (srcIntTy && srcIntTy.getWidth() == 32))) {
      return op.emitOpError() << "expects fp form src to have element type f32, i32";
    }
    if (!(isAccToMat || isAccToVec)) {
      return op.emitOpError() << "expects fp form to use loc=acc src";
    }
  }
  if ((hasFp || hasPreQuantScalar) && accToVecModeAttr) {
    switch (accToVecModeAttr.getValue()) {
    case pto::AccToVecMode::SingleModeVec0:
    case pto::AccToVecMode::SingleModeVec1:
      break;
    case pto::AccToVecMode::DualModeSplitM:
    case pto::AccToVecMode::DualModeSplitN:
      return op.emitOpError() << "expects fp/preQuantScalar acc-to-vec forms to use single-mode accToVecMode";
    }
  }
  return verifyTMovGenericFpLayout(op, isA5);
}

static LogicalResult verifyTMovGeneric(TMovOp op, bool isA5) {
  if (failed(verifyTMovGenericPreconditions(op, isA5)) ||
      failed(verifyTMovGenericPairing(op, isA5)) ||
      failed(verifyTMovGenericFpForm(op, isA5))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTMovImpl(TMovOp op, bool isA5) {
  Value fp = op.getFp();
  if (fp && !getPTOMemorySpaceEnum(fp.getType())) {
    return op.emitOpError("expects the third tile to have an explicit address space");
  }
  if (classifyTMovForm(fp) == TMovForm::XToZz) {
    return verifyTMovXToZz(op, isA5);
  }
  return verifyTMovGeneric(op, isA5);
}

mlir::LogicalResult mlir::pto::TMovOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTMovImpl(*this, /*isA5=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<RankedTensorType>(t)) {
    return s.getRank();
  }
  if (auto tile = dyn_cast<pto::TileBufType>(t)) {
    return tile.getRank();
  }
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) {
    return view.getRank();
  }
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (Tensor 或 Tile 类型)
  bool aValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<RankedTensorType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid) {
    return op->emitOpError("expects inputs/outputs to be tensors or PTO tile types");
  }

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank) {
      return op->emitOpError("expects a and dst to have the same rank");
    }
    if (bRank != -1 && dRank != -1 && bRank != dRank) {
      return op->emitOpError("expects b and dst to have the same rank");
    }
  }

  return success();
}

// ---- LoadScalarOp ----
LogicalResult LoadScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else {
    return emitOpError("expects ptr to be !pto.ptr type");
  }

  if (getValue().getType() != elemTy) {
    return emitOpError("expects result type to match ptr element type");
  }

  return success();
}
// ---- StoreScalarOp ----
LogicalResult StoreScalarOp::verify() {
  Type ptrTy = getPtr().getType();
  Type elemTy;
  if (auto pty = dyn_cast<mlir::pto::PtrType>(ptrTy)) {
    elemTy = pty.getElementType();
  } else {
    return emitOpError("expects ptr to be !pto.ptr type");
  }

  if (getValue().getType() != elemTy) {
    return emitOpError("expects value type to match ptr element type");
  }

  return success();
}

// ---- CmoCacheInvalidOp ----
static bool isGmOrDefaultAddressSpace(pto::AddressSpace space) {
  return space == pto::AddressSpace::GM || space == pto::AddressSpace::Zero;
}

static bool isGmOrDefaultCmoAddressType(Type type) {
  if (auto ptrTy = dyn_cast<mlir::pto::PtrType>(type)) {
    return isGmOrDefaultAddressSpace(ptrTy.getMemorySpace().getAddressSpace());
  }
  if (isa<mlir::pto::TensorViewType, mlir::pto::PartitionTensorViewType>(type)) {
    return true;
  }
  return false;
}

ParseResult CmoCacheInvalidOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  if (succeeded(parser.parseOptionalKeyword("all"))) {
    AddressSpaceAttr spaceAttr;
    if (parser.parseAttribute(spaceAttr, "space", result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    return success();
  }

  OpAsmParser::UnresolvedOperand addr;
  Type addrTy;
  if (parser.parseOperand(addr) ||
      parser.parseKeyword("single_cache_line") ||
      parser.parseColonType(addrTy) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(addr, addrTy, result.operands)) {
    return failure();
  }

  if (!result.attributes.get("space")) {
    result.addAttribute(
        "space", AddressSpaceAttr::get(parser.getContext(), AddressSpace::GM));
  }
  return success();
}

void CmoCacheInvalidOp::print(OpAsmPrinter &p) {
  if (Value addr = getAddr()) {
    p << " " << addr << " single_cache_line";
    p << " : " << addr.getType();
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/{"space"});
    return;
  }

  p << " all " << getSpace();
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"space"});
}

LogicalResult CmoCacheInvalidOp::verify() {
  if (!isGmOrDefaultAddressSpace(getSpace().getAddressSpace())) {
    return emitOpError("only supports GM cache maintenance");
  }

  if (Value addr = getAddr()) {
    if (!isGmOrDefaultCmoAddressType(addr.getType())) {
      return emitOpError("single_cache_line address expects a GM pointer or GM tensor view");
    }
  }

  return success();
}

// ---- GetBufOp / RlsBufOp ----
static LogicalResult verifyBufSyncOp(Operation *op, Attribute opTypeAttr,
                                     IntegerAttr bufIdAttr,
                                     IntegerAttr modeAttr) {
  if (!opTypeAttr) {
    return op->emitOpError("expects 'op_type' attribute");
  }

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe)) {
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
  }

  if (!bufIdAttr) {
    return op->emitOpError("expects 'buf_id' attribute");
  }
  int64_t bufId = bufIdAttr.getInt();
  if (bufId < 0 || bufId > 31) {
    return op->emitOpError("expects 'buf_id' in range [0, 31]");
  }

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0) {
      return op->emitOpError("expects 'mode' to be non-negative");
    }
  }

  return success();
}

LogicalResult GetBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

LogicalResult RlsBufOp::verify() {
  return verifyBufSyncOp(getOperation(), getOpTypeAttr(), getBufIdAttr(),
                         getModeAttr());
}

// ---- GetBufDynOp / RlsBufDynOp ----
static LogicalResult verifyBufDynSyncOp(Operation *op, Attribute opTypeAttr,
                                        Value bufId, IntegerAttr modeAttr) {
  if (!opTypeAttr) {
    return op->emitOpError("expects 'op_type' attribute");
  }

  pto::PIPE pipe = pto::PIPE::PIPE_UNASSIGNED;
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    pipe = pipeAttr.getPipe();
  } else {
    auto opTypeOr = parseSyncOpTypeLikeAttr(opTypeAttr);
    if (failed(opTypeOr)) {
      auto diag = op->emitOpError(
          "expects 'op_type' to be pipe_event_type/sync_op_type/pipe, got ");
      diag << opTypeAttr;
      return failure();
    }
    pipe = mapSyncOpTypeToPipe(*opTypeOr);
  }
  if (!isConcreteSyncPipe(pipe)) {
    return op->emitOpError("expects 'op_type' to map to a concrete pipe, not PIPE_ALL/PIPE_UNASSIGNED");
  }

  if (!bufId) {
    return op->emitOpError("expects 'buf_id' operand");
  }

  if (modeAttr) {
    int64_t mode = modeAttr.getInt();
    if (mode < 0) {
      return op->emitOpError("expects 'mode' to be non-negative");
    }
  }

  return success();
}

LogicalResult GetBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

LogicalResult RlsBufDynOp::verify() {
  return verifyBufDynSyncOp(getOperation(), getOpTypeAttr(), getBufId(),
                            getModeAttr());
}

static ParseResult parseLegacyOrAttrMemBar(OpAsmParser &parser,
                                           MemBarAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeMemBarKind(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid membar token: " << token;
    }
    attr = MemBarAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto memBarAttr = dyn_cast<MemBarAttr>(parsed);
  if (!memBarAttr) {
    return parser.emitError(loc, "expected membar attribute");
  }
  attr = memBarAttr;
  return success();
}

static void printLegacyOrAttrMemBar(OpAsmPrinter &p, MemBarAttr kind,
                                    ArrayRef<NamedAttribute> attrs) {
  p << ' ' << '"' << stringifyMemBarKind(kind.getKind()) << '"';
  p.printOptionalAttrDict(attrs, {"kind"});
}

static ParseResult parseLegacyOrAttrDsbMem(OpAsmParser &parser,
                                           DsbMemAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDsbMem(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dsb memory token: " << token;
    }
    attr = DsbMemAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dsbMemAttr = dyn_cast<DsbMemAttr>(parsed);
  if (!dsbMemAttr) {
    return parser.emitError(loc, "expected dsb_mem attribute");
  }
  attr = dsbMemAttr;
  return success();
}

static void printLegacyOrAttrDsbMem(OpAsmPrinter &printer, Operation *op,
                                    DsbMemAttr mem) {
  (void)op;
  printer << ' ' << '"' << stringifyDsbMem(mem.getKind()) << '"';
}

static ParseResult parseLegacyOrAttrDcciCacheLine(OpAsmParser &parser,
                                                  DcciCacheLineAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciCacheLine(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci cache token: " << token;
    }
    attr = DcciCacheLineAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto cacheAttr = dyn_cast<DcciCacheLineAttr>(parsed);
  if (!cacheAttr) {
    return parser.emitError(loc, "expected dcci_cache_line attribute");
  }
  attr = cacheAttr;
  return success();
}

static void printLegacyOrAttrDcciCacheLine(OpAsmPrinter &printer, Operation *op,
                                           DcciCacheLineAttr cache) {
  (void)op;
  printer << ' ' << '"' << stringifyDcciCacheLine(cache.getKind()) << '"';
}

static ParseResult parseOptionalDcciDst(OpAsmParser &parser,
                                        DcciDstAttr &attr) {
  if (failed(parser.parseOptionalComma())) {
    return success();
  }

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto kind = symbolizeDcciDst(token);
    if (!kind) {
      return parser.emitError(loc) << "invalid dcci dst token: " << token;
    }
    attr = DcciDstAttr::get(parser.getContext(), *kind);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto dstAttr = dyn_cast<DcciDstAttr>(parsed);
  if (!dstAttr) {
    return parser.emitError(loc, "expected dcci_dst attribute");
  }
  attr = dstAttr;
  return success();
}

static void printOptionalDcciDst(OpAsmPrinter &printer, Operation *op,
                                 DcciDstAttr dst) {
  (void)op;
  if (!dst) {
    return;
  }
  printer << ", \"" << stringifyDcciDst(dst.getKind()) << '"';
}

LogicalResult DcciOp::verify() {
  auto space = getPTOMemorySpaceEnum(getPtr().getType());
  if (!space) {
    return emitOpError("expects ptr to have a PTO memory space");
  }
  if (*space != pto::AddressSpace::GM && *space != pto::AddressSpace::VEC) {
    return emitOpError("expects ptr memory space to be gm or ub/vec");
  }

  return success();
}

static ParseResult parseLegacyOrAttrPipe(OpAsmParser &parser, PipeAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto pipe = symbolizePIPE(token);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << token;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto pipe = symbolizePIPE(keyword);
    if (!pipe) {
      return parser.emitError(loc) << "invalid pipe token: " << keyword;
    }
    attr = PipeAttr::get(parser.getContext(), *pipe);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto pipeAttr = dyn_cast<PipeAttr>(parsed);
  if (!pipeAttr) {
    return parser.emitError(loc, "expected pipe attribute");
  }
  attr = pipeAttr;
  return success();
}

static ParseResult parseLegacyOrAttrEvent(OpAsmParser &parser, EventAttr &attr) {
  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    auto event = symbolizeEVENT(token);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << token;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  if (succeeded(parser.parseOptionalLess())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseGreater()) {
      return failure();
    }
    auto event = symbolizeEVENT(keyword);
    if (!event) {
      return parser.emitError(loc) << "invalid event token: " << keyword;
    }
    attr = EventAttr::get(parser.getContext(), *event);
    return success();
  }

  Attribute parsed;
  if (failed(parser.parseAttribute(parsed))) {
    return failure();
  }
  auto eventAttr = dyn_cast<EventAttr>(parsed);
  if (!eventAttr) {
    return parser.emitError(loc, "expected event attribute");
  }
  attr = eventAttr;
  return success();
}

static ParseResult parseI32LiteralAttr(OpAsmParser &parser, IntegerAttr &attr) {
  auto loc = parser.getCurrentLocation();
  int64_t value = 0;
  if (failed(parser.parseInteger(value))) {
    return failure();
  }
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    return parser.emitError(loc, "expected 32-bit integer literal");
  }
  attr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), value);
  return success();
}

static void printLegacySyncTriplet(OpAsmPrinter &p, PipeAttr srcPipe,
                                   PipeAttr dstPipe, EventAttr eventId,
                                   ArrayRef<NamedAttribute> attrs) {
  p << "[<" << stringifyPIPE(srcPipe.getPipe()) << ">, <"
    << stringifyPIPE(dstPipe.getPipe()) << ">, <"
    << stringifyEVENT(eventId.getEvent()) << ">]";
  p.printOptionalAttrDict(attrs, {"src_pipe", "dst_pipe", "event_id"});
}

ParseResult SetFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void SetFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult WaitFlagOp::parse(OpAsmParser &parser, OperationState &result) {
  PipeAttr srcPipe;
  PipeAttr dstPipe;
  EventAttr eventId;
  if (parser.parseLSquare() || parseLegacyOrAttrPipe(parser, srcPipe) ||
      parser.parseComma() || parseLegacyOrAttrPipe(parser, dstPipe) ||
      parser.parseComma() || parseLegacyOrAttrEvent(parser, eventId) ||
      parser.parseRSquare()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("src_pipe", srcPipe);
  result.addAttribute("dst_pipe", dstPipe);
  result.addAttribute("event_id", eventId);
  return success();
}

void WaitFlagOp::print(OpAsmPrinter &p) {
  printLegacySyncTriplet(p, getSrcPipe(), getDstPipe(), getEventId(),
                         (*this)->getAttrs());
}

ParseResult MemBarOp::parse(OpAsmParser &parser, OperationState &result) {
  MemBarAttr kind;
  if (parseLegacyOrAttrMemBar(parser, kind)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("kind", kind);
  return success();
}

void MemBarOp::print(OpAsmPrinter &p) {
  printLegacyOrAttrMemBar(p, getKind(), (*this)->getAttrs());
}

static ParseResult parseBufSyncOp(OpAsmParser &parser, OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr bufIdAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc) << "invalid get_buf/rls_buf token: " << token;
}

    if (parser.parseComma() || parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma() ||
        parseI32LiteralAttr(parser, bufIdAttr)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
    if (parser.parseRSquare()) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("buf_id", bufIdAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                           IntegerAttr bufIdAttr, IntegerAttr modeAttr,
                           ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", "
      << bufIdAttr.getInt() << ", " << modeAttr.getInt();
  } else if (auto pipeEventType = dyn_cast<PipeEventTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else if (auto syncOpType = dyn_cast<SyncOpTypeAttr>(opTypeAttr)) {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  } else {
    p << "[" << opTypeAttr << ", " << bufIdAttr.getInt() << ", "
      << modeAttr.getInt() << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "buf_id", "mode"});
}

ParseResult GetBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void GetBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

ParseResult RlsBufOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufSyncOp(parser, result);
}

void RlsBufOp::print(OpAsmPrinter &p) {
  printBufSyncOp(p, getOpTypeAttr(), getBufIdAttr(), getModeAttr(),
                 (*this)->getAttrs());
}

// ---- GetBufDynOp / RlsBufDynOp parse/print ----
static ParseResult parseBufDynSyncOp(OpAsmParser &parser,
                                     OperationState &result) {
  Attribute opTypeAttr;
  IntegerAttr modeAttr;

  auto loc = parser.getCurrentLocation();
  std::string token;
  if (succeeded(parser.parseOptionalString(&token))) {
    if (auto pipe = symbolizePIPE(token)) {
      opTypeAttr = PipeAttr::get(parser.getContext(), *pipe);
    } else if (auto opType = symbolizeSyncOpType(token)) {
      opTypeAttr = PipeEventTypeAttr::get(parser.getContext(), *opType);
    } else {
      return parser.emitError(loc)
             << "invalid get_buf_dyn/rls_buf_dyn token: " << token;
}

    if (parser.parseComma()) {
      return failure();
    }

    OpAsmParser::UnresolvedOperand bufOperand;
    if (parser.parseOperand(bufOperand)) {
      return failure();
    }
    if (parser.resolveOperand(bufOperand,
                              parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
  } else if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseAttribute(opTypeAttr) || parser.parseComma()) {
      return failure();
    }

    OpAsmParser::UnresolvedOperand bufOperand;
    if (parser.parseOperand(bufOperand)) {
      return failure();
    }
    if (parser.resolveOperand(bufOperand,
                              parser.getBuilder().getIndexType(),
                              result.operands)) {
      return failure();
    }

    if (succeeded(parser.parseOptionalComma())) {
      if (parseI32LiteralAttr(parser, modeAttr)) {
        return failure();
      }
    } else {
      modeAttr = IntegerAttr::get(IntegerType::get(parser.getContext(), 32), 0);
    }
    if (parser.parseRSquare()) {
      return failure();
    }
  } else {
    return parser.emitError(loc, "expected string pipe/op_type or '['");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  result.addAttribute("op_type", opTypeAttr);
  result.addAttribute("mode", modeAttr);
  return success();
}

static void printBufDynSyncOp(OpAsmPrinter &p, Attribute opTypeAttr,
                              Value bufId, IntegerAttr modeAttr,
                              ArrayRef<NamedAttribute> attrs) {
  if (auto pipeAttr = dyn_cast<PipeAttr>(opTypeAttr)) {
    p << " \"" << stringifyPIPE(pipeAttr.getPipe()) << "\", " << bufId << ", "
      << modeAttr.getInt();
  } else {
    p << "[" << opTypeAttr << ", " << bufId << ", " << modeAttr.getInt()
      << "]";
  }
  p.printOptionalAttrDict(attrs, {"op_type", "mode"});
}

ParseResult GetBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void GetBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}

ParseResult RlsBufDynOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBufDynSyncOp(parser, result);
}

void RlsBufDynOp::print(OpAsmPrinter &p) {
  printBufDynSyncOp(p, getOpTypeAttr(), getBufId(), getModeAttr(),
                    (*this)->getAttrs());
}
// ---- TOp ----
LogicalResult TGemvBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getA().getType(), getB().getType(),
                                      getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                         getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGemvMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tgemv.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxGemvTileOperands(*this, getA().getType(), getB().getType(),
                                          getDst().getType())) ||
        failed(verifyA5MxGemvScaleTile(*this, getAScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "a_scale", /*isLeftScale=*/true)) ||
        failed(verifyA5MxGemvScaleTile(*this, getBScale().getType(),
                                       getA().getType(), getB().getType(),
                                       "b_scale", /*isLeftScale=*/false)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    auto biasShape = getShapeVec(getBias().getType());
    auto dstShape = getShapeVec(getDst().getType());
    if (biasShape.size() != 2 || dstShape.size() != 2) {
      return emitOpError("expects bias and dst to be rank-2 for tgemv.mx.bias");
    }
    if (biasShape[1] != ShapedType::kDynamic && dstShape[1] != ShapedType::kDynamic &&
        biasShape[1] != dstShape[1]) {
      return emitOpError("expects bias and dst to have the same column shape");
    }
    if (failed(verifyTileBufSameValidShape(*this, getBias().getType(),
                                           getDst().getType(), "bias", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                     getDst().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getA().getType()),
                                      getElemTy(getB().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyMatTileOperands(*this, getA().getType(), getB().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true)) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType()))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType()))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TMatmulMxAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.acc is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getCIn().getType(), "c_in")) ||
        failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType()))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, getCIn().getType(),
                                         getDst().getType(), "c_in", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, getCIn().getType(),
                                           getDst().getType(), "c_in", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
LogicalResult TMatmulMxBiasOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tmatmul.mx.bias is only supported on A5 targets");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyA5MxMatTileOperands(*this, getA().getType(), getB().getType(),
                                         getDst().getType())) ||
        failed(verifyA5MxMatScaleTiles(*this, getAScale().getType(),
                                       getBScale().getType(), getA().getType(),
                                       getB().getType())) ||
        failed(verifyMatBiasTile(*this, getBias().getType(), getDst().getType(),
                                 /*requireFloatBias=*/true))) {
      return failure();
    }
    if (failed(verifyA5MxTypeTriple(*this, getA().getType(), getB().getType(),
                                    getDst().getType(), "lhs", "rhs", "dst"))) {
      return failure();
    }
    return verifyMatmulLike(*this, getA().getType(), getB().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType()) {
      return emitOpError("expects val type to match dst element type");
    }
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  Type srcTy = getSrc().getType();
  if (!mlir::isa<pto::TileBufType>(srcTy)) {
    return emitOpError("expects src to be tile_buf type");
  }

  // Memory space must be vec (Ascend does not support getval from MAT etc.).
  Attribute memSpace = cast<pto::TileBufType>(srcTy).getMemorySpace();
  auto addrSpaceAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memSpace);
  if (!addrSpaceAttr ||
      addrSpaceAttr.getAddressSpace() != pto::AddressSpace::VEC) {
    if (addrSpaceAttr &&
        addrSpaceAttr.getAddressSpace() == pto::AddressSpace::MAT) {
      return emitOpError(
          "Ascend hardware does not support reading from Mat tile_buf to Scalar unit");
    }
    return emitOpError("expects src memory space to be vec");
  }

  if (getElemTy(srcTy) != getDst().getType()) {
    return emitOpError("expects dst type to match src element type");
  }
  return success();
}

LogicalResult THistogramOp::verify() {
  auto isIntegerWidth = [](Type ty, unsigned width) {
    auto it = dyn_cast<IntegerType>(ty);
    return it && it.getWidth() == width;
  };
  int64_t byte = 1;
  auto byteAttr = getByteAttr();
  if (byteAttr) {
    byte = byteAttr.getInt();
  }
  if (auto legacyIsMSB = (*this)->getAttrOfType<BoolAttr>("isMSB")) {
    int64_t legacyByte = legacyIsMSB.getValue() ? 1 : 0;
    if (byteAttr && byte != legacyByte) {
      return emitOpError("does not allow conflicting 'byte' and legacy 'isMSB' attributes");
    }
    byte = legacyByte;
  }
  if (byte < 0 || byte > 3) {
    return emitOpError("expects byte to be in range [0, 3]");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("thistogram is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type idxTy = getIdx().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
        failed(verifyTileBufCommon(*this, idxTy, "idx")) ||
        failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
      return failure();
    }

    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    auto idxSpace = getPTOMemorySpaceEnum(idxTy);
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects src to be in the vec address space");
    }
    if (!idxSpace || *idxSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects idx to be in the vec address space");
    }
    if (!dstSpace || *dstSpace != pto::AddressSpace::VEC) {
      return emitOpError("expects dst to be in the vec address space");
    }

    auto srcTB = dyn_cast<pto::TileBufType>(srcTy);
    auto idxTB = dyn_cast<pto::TileBufType>(idxTy);
    auto dstTB = dyn_cast<pto::TileBufType>(dstTy);
    if (!srcTB || !idxTB || !dstTB) {
      return emitOpError("expects src, idx, and dst to be tile_buf types");
    }

    if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        srcTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return emitOpError("expects src to use row_major + none_box layout");
    }
    if (dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
        dstTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return emitOpError("expects dst to use row_major + none_box layout");
    }

    bool srcIsUi16 = isIntegerWidth(getElemTy(srcTy), 16);
    bool srcIsUi32 = isIntegerWidth(getElemTy(srcTy), 32);
    if (!srcIsUi16 && !srcIsUi32) {
      return emitOpError("expects src element type to be ui16 or ui32");
    }
    if (!isIntegerWidth(getElemTy(idxTy), 8)) {
      return emitOpError("expects idx element type to be ui8");
    }
    if (!isIntegerWidth(getElemTy(dstTy), 32)) {
      return emitOpError("expects dst element type to be ui32");
    }

    auto srcShape = getShapeVec(srcTy);
    auto idxShape = getShapeVec(idxTy);
    auto dstShape = getShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    auto idxValid = getValidShapeVec(idxTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcShape.size() != 2 || idxShape.size() != 2 || dstShape.size() != 2 ||
        srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError(
          "expects src, idx, and dst to have rank-2 shape and valid_shape");
    }

    if (!hasCompatibleKnownExtent(srcShape[0], dstShape[0]) ||
        !hasCompatibleKnownExtent(srcValid[0], dstValid[0])) {
      return emitOpError("expects dst rows and valid rows to match src");
    }

    if (srcIsUi16) {
      if (byte > 1) {
        return emitOpError("expects byte to be 0 or 1 when src element type is ui16");
      }
      if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::ColMajor) ||
          idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
        return emitOpError(
            "expects idx to use DN layout (col_major + none_box) when src element type is ui16");
      }
      if (!hasCompatibleKnownExtent(srcShape[0], idxShape[0]) ||
          !hasCompatibleKnownExtent(srcValid[0], idxValid[0])) {
        return emitOpError("expects idx rows and valid rows to match src when src element type is ui16");
      }
      if (!isKnownUnitExtent(idxShape[1]) || !isKnownZeroOrUnitExtent(idxValid[1])) {
        return emitOpError("expects idx to have exactly one physical column and 0 or 1 valid column when src element type is ui16");
      }
    } else {
      if (byte != 3) {
        if (idxTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
            idxTB.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
          return emitOpError(
              "expects idx to use row_major + none_box layout when src element type is ui32 and byte is 0, 1, or 2");
        }
        if (!hasCompatibleKnownExtent(srcShape[1], idxShape[1]) ||
            !hasCompatibleKnownExtent(srcValid[1], idxValid[1])) {
          return emitOpError(
              "expects idx cols and valid cols to match src when src element type is ui32 and byte is 0, 1, or 2");
        }

        int64_t expectedIdxRows = 1;
        if (byte == 1) {
          expectedIdxRows = 2;
        } else if (byte == 0) {
          expectedIdxRows = 3;
        }
        if (!hasCompatibleKnownExtent(idxShape[0], expectedIdxRows) ||
            !hasCompatibleKnownExtentOrZero(idxValid[0], expectedIdxRows)) {
          return emitOpError(
              "expects idx rows to match the byte-selected filter depth and idx valid rows to be 0 or match it when src element type is ui32 and byte is 0, 1, or 2");
        }
      }
    }
    if (dstShape[1] != ShapedType::kDynamic && dstShape[1] < 256) {
      return emitOpError("expects dst shape[1] to be at least 256");
    }
    if (dstValid[1] != ShapedType::kDynamic && dstValid[1] != 0 &&
        dstValid[1] < 256) {
      return emitOpError("expects dst valid_shape[1] to be 0 or at least 256");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult TGetScaleAddrOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tget_scale_addr is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyTileBufCommon(*this, srcTy, "src", /*allowLowPrecision=*/true))) {
      return failure();
    }
    if (failed(verifyTileBufCommon(*this, dstTy, "dst", /*allowLowPrecision=*/true))) {
      return failure();
    }
    auto srcSpace = getPTOMemorySpaceEnum(srcTy);
    if (!srcSpace ||
        (*srcSpace != pto::AddressSpace::LEFT && *srcSpace != pto::AddressSpace::RIGHT)) {
      return emitOpError("expects src to be in the left or right address space");
    }
    auto dstSpace = getPTOMemorySpaceEnum(dstTy);
    if (!dstSpace || *dstSpace != pto::AddressSpace::SCALING) {
      return emitOpError("expects dst to be in the scaling address space");
    }
    auto dstShape = getShapeVec(dstTy);
    auto srcShape = getShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    auto srcValid = getValidShapeVec(srcTy);
    if (dstShape.size() != 2 || srcShape.size() != 2 || dstValid.size() != 2 ||
        srcValid.size() != 2) {
      return emitOpError(
          "expects src/dst to have rank-2 shape and valid_shape");
    }
    if (*srcSpace == pto::AddressSpace::LEFT) {
      int64_t mShape = srcShape[0];
      int64_t vk = srcValid[1];
      int64_t expectedScaleK = ceilDivKnown(vk, 32);
      if (!hasCompatibleKnownExtent(dstShape[0], mShape) ||
          !hasCompatibleKnownExtent(dstShape[1], expectedScaleK) ||
          !hasCompatibleKnownExtent(dstValid[0], srcValid[0]) ||
          !hasCompatibleKnownExtent(dstValid[1], expectedScaleK)) {
        return emitOpError("expects dst shape/valid_shape to be [M, ceil(K/32)]");
      }
    } else {
      int64_t k = srcValid[0];
      int64_t n = srcShape[1];
      int64_t vk = srcValid[0];
      int64_t vn = srcValid[1];
      if (!hasCompatibleKnownExtent(dstShape[0], ceilDivKnown(k, 32)) ||
          !hasCompatibleKnownExtent(dstShape[1], n) ||
          !hasCompatibleKnownExtent(dstValid[0], ceilDivKnown(vk, 32)) ||
          !hasCompatibleKnownExtent(dstValid[1], vn)) {
        return emitOpError("expects dst shape/valid_shape to be [ceil(K/32), N]");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

// ---- MScatterOp ----
ParseResult mlir::pto::MScatterOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand idx;
  OpAsmParser::UnresolvedOperand mem;
  Type srcTy, idxTy, memTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(idx) || parser.parseColonType(srcTy) ||
      parser.parseComma() || parser.parseType(idxTy) || parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(mem) || parser.parseColonType(memTy) ||
      parser.parseRParen() ||
      parsePTOInherentAttrs<MScatterOp>(
          parser, result, parsedAttrs,
          {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"})) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands) ||
      parser.resolveOperand(mem, memTy, result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MScatterOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx() << " : "
    << getSrc().getType() << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  p << ") outs(" << getMem() << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ")";

  NamedAttrList attrs = getNonInherentAttrs(
      getOperation(),
      {"coalesce", "scatterAtomicOp", "scatterOob", "scatterConflict"});
  if (auto coalesceAttr = getMScatterCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto scatterAtomicAttr = getMScatterScatterAtomicOpAttrIfPresent(*this);
      scatterAtomicAttr &&
      scatterAtomicAttr.getValue() != pto::ScatterAtomicOp::None) {
    attrs.append("scatterAtomicOp", scatterAtomicAttr);
  }
  if (auto scatterOobAttr = getMScatterScatterOobAttrIfPresent(*this);
      scatterOobAttr &&
      scatterOobAttr.getValue() != pto::ScatterOOB::Undefined) {
    attrs.append("scatterOob", scatterOobAttr);
  }
  if (auto scatterConflictAttr =
          getMScatterScatterConflictAttrIfPresent(*this)) {
    attrs.append("scatterConflict", scatterConflictAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

LogicalResult MScatterOp::verify() {
  Type srcTy = getSrc().getType();
  Type idxTy = getIdx().getType();
  Type memTy = getMem().getType();

  if (getPTOTypeRank(srcTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(memTy) == -1) {
    return emitOpError("expects src, idx, and mem to use supported PTO shapes");
  }

  if (failed(verifyNDStyleVecTile(
          *this, srcTy, "src",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type srcElem = getElemTy(srcTy);
  Type idxElem = getElemTy(idxTy);
  pto::ScatterAtomicOp scatterAtomicOp = getScatterAtomicOpOrDefault(*this);
  pto::ScatterOOB scatterOob = getScatterOobOrDefault(*this);
  if (!srcElem || !idxElem) {
    return emitOpError("failed to resolve element types for src or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), srcElem)) {
    return emitOpError(
        "expects src element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), srcElem,
                                             "src"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), srcTy, idxTy, "src",
                                            coalesce))) {
    return failure();
  }

  if (!coalesce &&
      (scatterAtomicOp != pto::ScatterAtomicOp::None ||
       scatterOob != pto::ScatterOOB::Undefined ||
       getScatterConflictAttrIfPresent(*this))) {
    return emitOpError(
        "expects coalesce when scatterAtomicOp/scatterOob/scatterConflict is specified");
  }

  if (getScatterConflictAttrIfPresent(*this) && !isTargetArchA5(getOperation())) {
    return emitOpError("expects scatterConflict only on A5 targets");
  }

  if (!isSupportedMScatterAtomicPayloadElemType(srcElem, scatterAtomicOp)) {
    return emitOpError(
        "expects scatterAtomicOp-compatible src element type: add supports "
        "i32/ui32/f16/f32, max/min support signless i32/f32");
  }

  return success();
}

// ---- MGatherOp ----
// GM -> L1 (cube Mat) gather verifier. The destination is an L1 (loc=mat) tile
// in NZ layout; the index is a GM tensor (the cube core cannot read UB on A5),
// and Coalesce::Elem carries a contiguous GM scratch workspace. Mirrors the
// pto-isa MGATHER GM -> L1 overloads / MGatherCheckGm2L1.
static LogicalResult verifyMGatherGm2L1(Operation *op, Value mem, Value idx,
                                        Value dst, Value scratch,
                                        std::optional<pto::Coalesce> coalesce) {
  Type dstTy = dst.getType();
  auto dstTb = dyn_cast<pto::TileBufType>(dstTy);
  if (!dstTb) {
    return op->emitOpError("expects GM->L1 mgather dst to be a tile_buf");
  }

  // dst must be an L1 / cube Mat tile in NZ layout (col_major + row_major sub +
  // fractal 512), matching the matmul A/NZ operand a TLOAD would produce.
  if (!isColMajorRowMajorNZTileBuf(dstTb)) {
    return op->emitOpError("expects GM->L1 mgather dst (loc=mat) to use "
                           "blayout=col_major and slayout=row_major (NZ)");
  }
  if (dstTb.getSFractalSizeI32() != 512) {
    return op->emitOpError("expects GM->L1 mgather dst fractal size to be 512");
  }

  Type dstElem = getElemTy(dstTy);
  if (!dstElem) {
    return op->emitOpError("failed to resolve GM->L1 mgather dst element type");
  }
  if (!isSupportedMGatherMScatterPayloadElemType(op, dstElem)) {
    return op->emitOpError(
        "expects GM->L1 mgather dst element type to be "
        "i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 (and on A5 targets also "
        "float8_e4m3/float8_e5m2 family types)");
  }

  // NZ tile shape: padded Cols a multiple of C0 (= 32 / sizeof(elem)) and padded
  // Rows a multiple of FRACTAL_NZ_ROW (= 16).
  unsigned elemBytes =
      std::max<unsigned>(1u, dstElem.getIntOrFloatBitWidth() / 8u);
  int64_t kC0 = 32 / static_cast<int64_t>(elemBytes);
  auto dstShape = getShapeVec(dstTy);
  if (dstShape.size() == 2) {
    if (kC0 > 0 && dstShape[1] != ShapedType::kDynamic &&
        dstShape[1] % kC0 != 0) {
      return op->emitOpError()
             << "expects GM->L1 mgather dst padded cols to be a multiple of "
             << kC0 << " (C0 = 32 / sizeof(elem))";
    }
    if (dstShape[0] != ShapedType::kDynamic && dstShape[0] % 16 != 0) {
      return op->emitOpError("expects GM->L1 mgather dst padded rows to be a "
                             "multiple of 16 (FRACTAL_NZ_ROW)");
    }
  }

  // mem table: GM, element type matches dst.
  if (failed(verifyMGatherMScatterMemOperand(op, mem, dstElem, "dst"))) {
    return failure();
  }

  // idx: GM partition tensor view of i32 -- NOT a UB tile.
  Type idxTy = idx.getType();
  if (isa<pto::TileBufType>(idxTy)) {
    return op->emitOpError("expects GM->L1 mgather idx to be a GM tensor "
                           "partition_tensor_view, not a tile_buf");
  }
  if (!isa<pto::PartitionTensorViewType>(idxTy)) {
    return op->emitOpError(
        "expects GM->L1 mgather idx to be a partition_tensor_view");
  }
  Type idxElem = getElemTy(idxTy);
  if (!idxElem || !isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return op->emitOpError("expects GM->L1 mgather idx element type to be i32");
  }

  // Coalesce must be explicit: the GM index has no UB tile shape to infer from.
  if (!coalesce) {
    return op->emitOpError("expects GM->L1 mgather to specify an explicit "
                           "coalesce attribute (row or elem)");
  }

  if (*coalesce == pto::Coalesce::Elem) {
    // Elem mode stages discrete elements into NZ layout through a GM scratch
    // workspace before the bulk GM -> L1 copy.
    if (!scratch) {
      return op->emitOpError("expects GM->L1 mgather with coalesce=elem to "
                             "provide a GM scratch operand");
    }
    Type scTy = scratch.getType();
    if (!isa<pto::PartitionTensorViewType>(scTy)) {
      return op->emitOpError(
          "expects GM->L1 mgather scratch to be a partition_tensor_view");
    }
    Type scElem = getElemTy(scTy);
    if (!scElem || scElem != dstElem) {
      return op->emitOpError("expects GM->L1 mgather scratch element type to "
                             "match dst element type");
    }
  } else { // Row
    if (scratch) {
      return op->emitOpError("expects GM->L1 mgather with coalesce=row to omit "
                             "the scratch operand");
    }
  }

  return success();
}
ParseResult mlir::pto::MGatherOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> insOperands;
  SmallVector<Type, 3> insTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstTy;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    insOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() < 2 || insOperands.size() > 3) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects mgather ins(mem, idx[, scratch])");
  }

  if (parser.parseColon()) {
    return failure();
  }

  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    insTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (insOperands.size() != insTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of ins operands to match the number of ins types");
  }

  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parsePTOInherentAttrs<MGatherOp>(
          parser, result, parsedAttrs, {"coalesce", "gatherOob"})) {
    return failure();
  }

  if (parser.resolveOperand(insOperands[0], insTypes[0], result.operands) ||
      parser.resolveOperand(insOperands[1], insTypes[1], result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (insOperands.size() == 3 &&
      parser.resolveOperand(insOperands[2], insTypes[2], result.operands)) {
    return failure();
  }
  return success();
}

void mlir::pto::MGatherOp::print(OpAsmPrinter &p) {
  p << " ins(" << getMem() << ", " << getIdx();
  if (auto scratch = getScratch()) {
    p << ", " << scratch;
  }
  p << " : ";
  p.printStrippedAttrOrType(getMem().getType());
  p << ", ";
  p.printStrippedAttrOrType(getIdx().getType());
  if (auto scratch = getScratch()) {
    p << ", ";
    p.printStrippedAttrOrType(scratch.getType());
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"coalesce", "gatherOob"});
  if (auto coalesceAttr = getMGatherCoalesceAttrIfPresent(*this)) {
    attrs.append("coalesce", coalesceAttr);
  }
  if (auto gatherOobAttr = getMGatherGatherOobAttrIfPresent(*this);
      gatherOobAttr &&
      gatherOobAttr.getValue() != pto::GatherOOB::Undefined) {
    attrs.append("gatherOob", gatherOobAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

LogicalResult MGatherOp::verify() {
  Type memTy = getMem().getType();
  Type idxTy = getIdx().getType();
  Type dstTy = getDst().getType();

  if (getPTOTypeRank(memTy) == -1 || getPTOTypeRank(idxTy) == -1 ||
      getPTOTypeRank(dstTy) == -1) {
    return emitOpError("expects mem, idx, and dst to use supported PTO shapes");
  }

  // GM -> L1 (cube Mat) gather: dst is an L1 (loc=mat) tile; idx comes from GM
  // and Coalesce::Elem carries a GM scratch operand.
  if (isa<pto::TileBufType>(dstTy)) {
    if (auto as = getPTOMemorySpaceEnum(dstTy);
        as && *as == pto::AddressSpace::MAT) {
      std::optional<pto::Coalesce> coalesce;
      if (auto coalesceAttr = getCoalesceAttr()) {
        coalesce = coalesceAttr.getValue();
      }
      return verifyMGatherGm2L1(getOperation(), getMem(), getIdx(), getDst(),
                                getScratch(), coalesce);
    }
  }

  // GM -> UB (VEC) gather: the default path. A GM scratch operand is only valid
  // for the GM -> L1 path above.
  if (getScratch()) {
    return emitOpError("expects scratch operand only on GM->L1 (loc=mat) "
                       "mgather");
  }

  if (failed(verifyNDStyleVecTile(
          *this, dstTy, "dst",
          /*allowLowPrecision=*/isTargetArchA5(getOperation()))) ||
      failed(verifyMGatherMScatterIdxTile(getOperation(), idxTy, "idx"))) {
    return failure();
  }

  auto coalesce = getCoalesceIfPresent(*this);

  Type dstElem = getElemTy(dstTy);
  Type idxElem = getElemTy(idxTy);
  pto::GatherOOB gatherOob = getGatherOobOrDefault(*this);
  if (!dstElem || !idxElem) {
    return emitOpError("failed to resolve element types for dst or idx");
  }

  if (!isSupportedMGatherMScatterPayloadElemType(getOperation(), dstElem)) {
    return emitOpError(
        "expects dst element type to be i8/ui8/i16/ui16/i32/ui32/f16/bf16/f32 "
        "(and on A5 targets also float8_e4m3/float8_e5m2 family types)");
  }

  if (!isSupportedMGatherMScatterIndexElemType(idxElem)) {
    return emitOpError("expects idx element type to be signless i32");
  }

  if (failed(verifyMGatherMScatterMemOperand(getOperation(), getMem(), dstElem,
                                             "dst"))) {
    return failure();
  }

  if (failed(verifyMGatherMScatterTileShape(getOperation(), dstTy, idxTy, "dst",
                                            coalesce))) {
    return failure();
  }

  if (gatherOob != pto::GatherOOB::Undefined && !coalesce) {
    return emitOpError("expects coalesce when gatherOob is specified");
  }

  return success();
}

void mlir::pto::TCvtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  Builder builder(getContext());
  NamedAttrList attrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == "sat_mode") {
      attrs.set(builder.getStringAttr("satmode"), attr.getValue());
      continue;
    }
    attrs.set(attr.getName(), attr.getValue());
  }
  p.printOptionalAttrDict(attrs.getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
}

ParseResult mlir::pto::TCvtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs) || parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (auto satmode = attrs.get("satmode")) {
    attrs.erase("satmode");
    if (attrs.get("sat_mode")) {
      return parser.emitError(parser.getCurrentLocation(),
                              "cannot specify both satmode and sat_mode");
    }
    attrs.set("sat_mode", satmode);
  }
  result.attributes = attrs;
  if (parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) || parser.parseRParen()) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TDeInterleaveOp::print(OpAsmPrinter &p) {
  p << " ins(";
  llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
  p << " : ";
  llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type type) { p << type; });
  p << ") outs(" << getDst0() << ", " << getDst1() << " : "
    << getDst0().getType() << ", " << getDst1().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TDeInterleaveOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  bool invalidHeader = parser.parseKeyword("ins") || parser.parseLParen();
  if (invalidHeader) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 2> srcs;
  OpAsmParser::UnresolvedOperand src;
  bool invalidSrc = failed(parser.parseOperand(src));
  if (invalidSrc) {
    return failure();
  }
  srcs.push_back(src);
  while (succeeded(parser.parseOptionalComma())) {
    bool invalidNextSrc = failed(parser.parseOperand(src));
    if (invalidNextSrc) {
      return failure();
    }
    srcs.push_back(src);
  }

  SmallVector<Type, 2> srcTypes;
  Type srcType;
  bool invalidSrcType = failed(parser.parseColonType(srcType));
  if (invalidSrcType) {
    return failure();
  }
  srcTypes.push_back(srcType);
  while (succeeded(parser.parseOptionalComma())) {
    bool invalidNextType = failed(parser.parseType(srcType));
    if (invalidNextType) {
      return failure();
    }
    srcTypes.push_back(srcType);
  }
  bool invalidOutsHeader =
      parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen();
  if (invalidOutsHeader) {
    return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 2> dsts;
  OpAsmParser::UnresolvedOperand dst;
  bool invalidDst = parser.parseOperand(dst) || parser.parseComma();
  if (invalidDst) {
    return failure();
  }
  dsts.push_back(dst);
  bool invalidSecondDst = failed(parser.parseOperand(dst));
  if (invalidSecondDst) {
    return failure();
  }
  dsts.push_back(dst);
  Type dst0Ty;
  Type dst1Ty;
  bool invalidDstTypes = parser.parseColonType(dst0Ty) ||
                         parser.parseComma() || parser.parseType(dst1Ty) ||
                         parser.parseRParen();
  if (invalidDstTypes) {
    return failure();
  }
  bool invalidSourceCount = srcs.size() < 1 || srcs.size() > 2;
  if (invalidSourceCount) {
    return parser.emitError(parser.getCurrentLocation(),
                            "tdeinterleave expects one or two source operands");
  }

  bool unresolvedSources =
      failed(parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(),
                                    result.operands));
  bool unresolvedDsts =
      failed(parser.resolveOperand(dsts[0], dst0Ty, result.operands)) ||
      failed(parser.resolveOperand(dsts[1], dst1Ty, result.operands));
  if (unresolvedSources || unresolvedDsts) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {static_cast<int32_t>(srcs.size()), 2}));
  return parser.parseOptionalAttrDict(result.attributes);
}

void mlir::pto::TMrgSortOp::print(OpAsmPrinter &p) {
  if (isFormat1()) {
    p << " ins(" << getSrc() << ", " << getBlockLen() << " : " << getSrc().getType()
      << ", " << getBlockLen().getType() << ") outs(" << getDst() << " : "
      << getDst().getType() << ")";
  } else if (isFormat2() || isFormat2WithoutTmp()) {
    p << " ins(";
    llvm::interleaveComma(getSrcs(), p, [&](Value src) { p << src; });
    if (getTmp()) {
      p << ", " << getTmp();
    } else {
      p << " no_tmp";
}
    p << " {exhausted = " << (getExhausted() ? "true" : "false") << "} : ";
    llvm::interleaveComma(getSrcs().getTypes(), p, [&](Type ty) { p << ty; });
    if (getTmp()) {
      p << ", " << getTmp().getType();
    }
    p << ") outs(" << getDst() << ", " << getExcuted()
      << " : " << getDst().getType() << ", " << getExcuted().getType() << ")";
  } else {
    llvm::report_fatal_error("TMrgSortOp print expects format1 or format2");
  }
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes", "exhausted"});
}

ParseResult mlir::pto::TMrgSortOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand first, second;
  if (parser.parseOperand(first) || parser.parseComma() || parser.parseOperand(second)) {
    return failure();
  }

  if (parser.parseOptionalColon().succeeded()) {
    Type srcTy, blockLenTy, dstTy;
    if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(blockLenTy) ||
        parser.parseRParen() || parser.parseKeyword("outs") || parser.parseLParen()) {
      return failure();
    }
    OpAsmParser::UnresolvedOperand dstOp;
    if (parser.parseOperand(dstOp) || parser.parseColon() || parser.parseType(dstTy) ||
        parser.parseRParen()) {
      return failure();
    }
    result.addAttribute("operandSegmentSizes",
                        parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, 0, 0}));
    if (parser.resolveOperand(first, srcTy, result.operands) ||
        parser.resolveOperand(second, blockLenTy, result.operands) ||
        parser.resolveOperand(dstOp, dstTy, result.operands)) {
      return failure();
    }
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
    if (!result.attributes.get("exhausted")) {
      result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(false));
    }
    return success();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcs = {first, second};
  while (parser.parseOptionalComma().succeeded()) {
    OpAsmParser::UnresolvedOperand next;
    if (parser.parseOperand(next)) {
      return failure();
    }
    srcs.push_back(next);
  }
  bool noTmp = succeeded(parser.parseOptionalKeyword("no_tmp"));
  if ((noTmp && (srcs.size() < 2 || srcs.size() > 4)) ||
      (!noTmp && (srcs.size() < 3 || srcs.size() > 5))) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "tmrgsort format2 expects 2 to 4 src operands and optional no_tmp marker");
  }
  OpAsmParser::UnresolvedOperand tmpOp;
  if (!noTmp) {
    tmpOp = srcs.pop_back_val();
  }
  bool exhaustedVal = false;
  if (parser.parseOptionalLBrace().succeeded()) {
    if (parser.parseKeyword("exhausted") || parser.parseEqual()) {
      return failure();
    }
    StringRef kw;
    if (parser.parseKeyword(&kw) || parser.parseRBrace()) {
      return failure();
    }
    exhaustedVal = (kw == "true");
  }
  SmallVector<Type, 4> srcTypes;
  srcTypes.reserve(srcs.size());
  if (parser.parseColon()) {
    return failure();
  }
  Type firstSrcTy;
  if (parser.parseType(firstSrcTy)) {
    return failure();
  }
  srcTypes.push_back(firstSrcTy);
  while (parser.parseOptionalComma().succeeded()) {
    Type nextTy;
    if (parser.parseType(nextTy)) {
      return failure();
    }
    srcTypes.push_back(nextTy);
  }
  if (srcTypes.size() != srcs.size() + (noTmp ? 0 : 1) ||
      parser.parseRParen() ||
      parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }
  Type tmpTy;
  if (!noTmp) {
    tmpTy = srcTypes.pop_back_val();
  }
  OpAsmParser::UnresolvedOperand dstOp, excutedOp;
  Type dstTy, excutedTy;
  if (parser.parseOperand(dstOp) || parser.parseComma() || parser.parseOperand(excutedOp) ||
      parser.parseColon() || parser.parseType(dstTy) || parser.parseComma() ||
      parser.parseType(excutedTy) || parser.parseRParen()) {
    return failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(srcs.size()), 0, 1,
                           noTmp ? 0 : 1, 1}));
  if (parser.resolveOperands(srcs, srcTypes, parser.getCurrentLocation(), result.operands) ||
      parser.resolveOperand(dstOp, dstTy, result.operands) ||
      (!noTmp && parser.resolveOperand(tmpOp, tmpTy, result.operands)) ||
      parser.resolveOperand(excutedOp, excutedTy, result.operands)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (!result.attributes.get("exhausted")) {
    result.addAttribute("exhausted", parser.getBuilder().getBoolAttr(exhaustedVal));
  }
  return success();
}

static LogicalResult verifyTMrgSortFormat1(TMrgSortOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "format1 expects PTO shaped-like types for src/dst";
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src/dst to have the same element type";
  }
  if (!getElemTy(srcTy).isF16() && !getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects element type to be f16 or f32";
  }
  auto ss = getShapeVec(srcTy);
  auto ds = getShapeVec(dstTy);
  if (ss.size() != 2 || ds.size() != 2) {
    return op.emitOpError() << "expects src/dst to be rank-2 tile-shaped";
  }
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1) {
    return op.emitOpError() << "expects src rows == 1";
  }
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1) {
    return op.emitOpError() << "expects dst rows == 1";
  }
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1]) {
    return op.emitOpError() << "expects src/dst cols to match";
  }
  if (op.getBlockLen()) {
    if (auto cstOp = op.getBlockLen().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(cstOp.getValue())) {
        int64_t v = intAttr.getValue().getSExtValue();
        if (v <= 0 || (v % 64) != 0) {
          return op.emitOpError() << "expects blockLen > 0 and multiple of 64";
        }
      }
    }
  }
  return mlir::success();
}

static LogicalResult verifyTMrgSortFormat2Basics(TMrgSortOp op) {
  for (Value v : op.getSrcs()) {
    if (!isPTOShapedLike(v.getType())) {
      return op.emitOpError() << "format2 expects PTO shaped-like type for each src";
    }
  }
  if (op.getSrcs().size() < 2u || op.getSrcs().size() > 4u) {
    return op.emitOpError() << "format2 expects 2 to 4 srcs";
  }
  if (op.getDsts().size() != 1u || !op.getExcuted()) {
    return op.emitOpError()
           << "format2 expects 2 to 4 srcs, one dst, and excuted=vector";
  }
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  if (!isPTOShapedLike(dstTy) ||
      (tmpTy && !isPTOShapedLike(tmpTy))) {
    return op.emitOpError() << "format2 dst/tmp must be PTO shaped-like";
  }
  auto excutedTy = mlir::dyn_cast<mlir::VectorType>(op.getExcuted().getType());
  if (!excutedTy || excutedTy.getRank() != 1 || excutedTy.getNumElements() != 4 ||
      !excutedTy.getElementType().isInteger(16)) {
    return op.emitOpError() << "format2 excuted must be vector<4xi16>";
  }
  Type elemTy = getElemTy(dstTy);
  if (tmpTy && elemTy != getElemTy(tmpTy)) {
    return op.emitOpError() << "format2 expects dst/tmp element types to match";
  }
  auto dstShape = getShapeVec(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  if (dstShape.size() != 2 || (tmpTy && tmpShape.size() != 2)) {
    return op.emitOpError() << "format2 expects dst/tmp to be rank-2 tile-shaped";
  }
  if ((dstShape[0] != mlir::ShapedType::kDynamic && dstShape[0] != 1) ||
      (tmpTy && tmpShape[0] != mlir::ShapedType::kDynamic &&
       tmpShape[0] != 1)) {
    return op.emitOpError() << "format2 expects dst/tmp rows == 1";
  }
  if (tmpTy && dstShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < dstShape[1]) {
    return op.emitOpError() << "format2 expects tmp.cols >= dst.cols";
  }
  return success();
}

static LogicalResult verifyTMrgSortFormat2Srcs(TMrgSortOp op) {
  Type dstTy = op.getDst().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type elemTy = getElemTy(dstTy);
  auto tmpShape = tmpTy ? getShapeVec(tmpTy) : SmallVector<int64_t, 4>{};
  int64_t requiredTmpCols = 0;
  for (Value src : op.getSrcs()) {
    Type srcTy = src.getType();
    auto srcShape = getShapeVec(srcTy);
    auto srcValidShape = getValidShapeVec(src);
    if (srcShape.size() != 2 || srcValidShape.size() != 2) {
      return op.emitOpError() << "format2 expects src to be rank-2 tile-shaped";
    }
    if (srcShape[0] != mlir::ShapedType::kDynamic && srcShape[0] != 1) {
      return op.emitOpError() << "format2 expects src rows == 1";
    }
    if (getElemTy(srcTy) != elemTy) {
      return op.emitOpError() << "format2 expects src/dst/tmp element types to match";
    }
    if (srcValidShape[1] == mlir::ShapedType::kDynamic) {
      requiredTmpCols = mlir::ShapedType::kDynamic;
    } else if (requiredTmpCols != mlir::ShapedType::kDynamic) {
      requiredTmpCols += srcValidShape[1];
    }
  }
  if (tmpTy && requiredTmpCols != mlir::ShapedType::kDynamic &&
      tmpShape[1] != mlir::ShapedType::kDynamic &&
      tmpShape[1] < requiredTmpCols) {
    return op.emitOpError()
           << "format2 expects tmp.cols >= sum(src.cols) = "
           << requiredTmpCols;
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  if (isFormat1()) {
    return verifyTMrgSortFormat1(*this);
  }
  if (isFormat2() || isFormat2WithoutTmp()) {
    if (failed(verifyTMrgSortFormat2Basics(*this))) {
      return failure();
    }
    return verifyTMrgSortFormat2Srcs(*this);
  }
  return emitOpError() << "tmrgsort expects format1 (1 src + blockLen + 1 dst) or "
                          "format2 (2 to 4 srcs + tmp, outs dst, excuted)";
}

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tmul element type to be i32/i16/f16/f32",
      "expects A5 tmul element type to be i32/i16/f16/f32");
}

mlir::LogicalResult mlir::pto::TMulSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getDst().getType(),
      getScalar().getType(), /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tmuls element type to be i32/i16/f16/f32",
      "expects A5 tmuls element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}

mlir::LogicalResult mlir::pto::TShlSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }

  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem) {
    return emitOpError() << "failed to get element type for src/dst";
  }
  if (srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!mlir::isa<IntegerType>(srcElem)) {
    return emitOpError() << "expects integral element types";
  }
  if (auto scalarValue = getConstantIntegerValue(getScalar()); scalarValue && *scalarValue < 0) {
    return emitOpError("expects tshls scalar to be non-negative");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TShrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type srcElem = getElemTy(srcTy);
    Type dstElem = getElemTy(dstTy);
    if (!srcElem || !dstElem) {
      emitOpError("failed to get element type for src/dst");
      return failure();
    }
    if (srcElem != dstElem) {
      emitOpError("expects src and dst to have the same element type");
      return failure();
    }
    return srcElem;
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 16 && it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tshrs src and dst element type to be i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tshrs src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(16) || elemTy.isInteger(32) || elemTy.isF16() ||
          elemTy.isF32())) {
      return emitOpError()
             << "expects A2/A3 tneg element type to be i16/i32/f16/f32";
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileStorage(*this, srcTy, "src")) ||
        failed(verifyVecTileStorage(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }

    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() != 2 || dstValid.size() != 2) {
      return emitOpError() << "expects src and dst to have rank-2 valid_shape";
    }
    if (srcValid[1] != ShapedType::kDynamic &&
        dstValid[1] != ShapedType::kDynamic &&
        srcValid[1] != dstValid[1]) {
      return emitOpError()
             << "expects src and dst to have the same valid_shape[1]";
    }

    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32) ||
          elemTy.isF16() || elemTy.isF32() || elemTy.isBF16())) {
      return emitOpError()
             << "expects A5 tneg element type to be i8/i16/i32/f16/f32/bf16";
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      return emitOpError() << "expects src and dst to have the same element type";
    }
    if (!elemTy.isInteger(16)) {
      return emitOpError() << "expects A2/A3 tnot element type to be i16";
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    auto elemTy = getElemTy(srcTy);
    if (elemTy != getElemTy(dstTy)) {
      return emitOpError() << "expects src and dst to have the same element type";
    }
    if (!(elemTy.isInteger(8) || elemTy.isInteger(16) || elemTy.isInteger(32))) {
      return emitOpError() << "expects A5 tnot element type to be i8/i16/i32";
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyMatchingRowMajorBinaryTileOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getDst().getType());
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A2/A3 tor src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tor src0, src1, and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TOrSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16)) {
      return emitOpError(
          "expects A2/A3 tors src and dst element type to be i8/i16");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 tors src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyPTOShapedBinarySameElemAndShape(Operation *op,
                                                              Type src0Ty,
                                                              Type src1Ty,
                                                              Type dstTy) {
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op->emitOpError(
               "expects src0/src1/dst to be tensor/tile_buf/tile_view types"),
           failure();
  }
  Type e0 = getElemTy(src0Ty), e1 = getElemTy(src1Ty), ed = getElemTy(dstTy);
  if (!e0 || !e1 || !ed) {
    return op->emitOpError("failed to get element type for operands"), failure();
  }
  if (e0 != e1 || e0 != ed) {
    return op->emitOpError("expects src0/src1/dst to have the same element type"),
           failure();
  }
  auto s0 = getShapeVec(src0Ty), s1 = getShapeVec(src1Ty), sd = getShapeVec(dstTy);
  if (s0 != s1 || s0 != sd) {
    return op->emitOpError("expects src0/src1/dst to have the same shape"),
           failure();
  }
  return e0;
}

static LogicalResult verifyTPartAddA2A3(TPartAddOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A2/A3 tpartadd element type to be i32/i16/f16/f32");
  }
  return mlir::success();
}

static LogicalResult verifyTPartAddA5(TPartAddOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError() << "expects src0/src1/dst to have the same element type";
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32())) {
    return op.emitOpError("expects A5 tpartadd element type to be i32/i16/i8/f16/bf16/f32");
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartAddA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartAddA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyPartialValidPattern(*this, t0, t1, td))) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tpartmax element type to be i32/i16/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32())) {
      return emitOpError("expects A5 tpartmax element type to be i32/i16/i8/f16/bf16/f32");
    }
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td))) {
      return failure();
    }
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    if (failed(verifyPartialValidPattern(*this, t0, t1, td))) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isF16() || e0.isF32())) {
      return emitOpError("expects A2/A3 tpartmin element type to be i32/i16/f16/f32");
    }
    return mlir::success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    Type t0 = getSrc0().getType();
    Type t1 = getSrc1().getType();
    Type td = getDst().getType();
    FailureOr<Type> elemOr =
        verifyPTOShapedBinarySameElemAndShape(getOperation(), t0, t1, td);
    if (failed(elemOr)) {
      return failure();
    }
    Type e0 = *elemOr;
    if (!(e0.isInteger(32) || e0.isInteger(16) || e0.isInteger(8) ||
          e0.isF16() || e0.isBF16() || e0.isF32())) {
      return emitOpError("expects A5 tpartmin element type to be i32/i16/i8/f16/bf16/f32");
    }
    if (failed(verifyPartialValidPatternLoose(*this, t0, t1, td))) {
      return failure();
    }
    return mlir::success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPartArgOpCommon(Operation *op, Type src0Ty,
                                            Type src1Ty, Type src0IdxTy,
                                            Type src1IdxTy, Type dstTy,
                                            Type dstIdxTy, StringRef opName) {
  FailureOr<Type> dataElemOr =
      verifyPTOShapedBinarySameElemAndShape(op, src0Ty, src1Ty, dstTy);
  if (failed(dataElemOr)) {
    return failure();
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }

  if (!isPTOShapedLike(src0IdxTy) || !isPTOShapedLike(src1IdxTy) ||
      !isPTOShapedLike(dstIdxTy)) {
    return op->emitOpError("expects PTO shaped-like src0Idx/src1Idx/dstIdx");
  }
  Type idxElem = getElemTy(src0IdxTy);
  if (!idxElem || idxElem != getElemTy(src1IdxTy) ||
      idxElem != getElemTy(dstIdxTy)) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx to have the same element type");
  }
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return op->emitOpError(
        "expects src0Idx/src1Idx/dstIdx element type to be i32 or ui32");
  }

  auto dataShape = getShapeVec(src0Ty);
  if (dataShape != getShapeVec(src0IdxTy) ||
      dataShape != getShapeVec(src1IdxTy) ||
      dataShape != getShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects data and index operands to have the same shape");
  }
  if (getValidShapeVec(src0Ty) != getValidShapeVec(src0IdxTy) ||
      getValidShapeVec(src1Ty) != getValidShapeVec(src1IdxTy) ||
      getValidShapeVec(dstTy) != getValidShapeVec(dstIdxTy)) {
    return op->emitOpError(
        "expects each data operand and its index operand to have the same valid_shape");
  }

  Type elem = *dataElemOr;
  PTOArch arch = getTargetArch(op);
  if (arch == PTOArch::A5) {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
          elem.isF16() || elem.isBF16() || elem.isF32())) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i32/i16/i8/f16/bf16/f32";
    }
  } else {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op->emitOpError() << "expects A2/A3 " << opName
                               << " element type to be i32/i16/f16/f32";
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPartArgMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmax");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TPartArgMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    return verifyTPartArgOpCommon(
        getOperation(), getSrc0().getType(), getSrc1().getType(),
        getSrc0Idx().getType(), getSrc1Idx().getType(), getDst().getType(),
        getDstIdx().getType(), "tpartargmin");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

static LogicalResult verifyTPartMulA2A3(TPartMulOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPattern(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
        elem.isF32())) {
    return op.emitOpError(
        "expects A2/A3 tpartmul element type to be i32/i16/f16/f32");
  }
  return mlir::success();
}

static LogicalResult verifyTPartMulA5(TPartMulOp op) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) ||
      !isPTOShapedLike(dstTy)) {
    return op.emitOpError() << "expects PTO shaped-like src0/src1/dst";
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty) ||
      getElemTy(src0Ty) != getElemTy(dstTy)) {
    return op.emitOpError()
           << "expects src0/src1/dst to have the same element type";
  }
  Type elem = getElemTy(src0Ty);
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8) ||
        elem.isF16() || elem.isBF16() || elem.isF32())) {
    return op.emitOpError(
        "expects A5 tpartmul element type to be i32/i16/i8/f16/bf16/f32");
  }
  auto s0 = getShapeVec(src0Ty);
  auto s1 = getShapeVec(src1Ty);
  auto d = getShapeVec(dstTy);
  if (s0.size() != 2 || s1.size() != 2 || d.size() != 2) {
    return op.emitOpError()
           << "expects src0/src1/dst to be rank-2 (tile-shaped)";
  }
  if (failed(verifyPartialValidPatternLoose(op, src0Ty, src1Ty, dstTy))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TPartMulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPartMulA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPartMulA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<std::tuple<Type, Type, Type, Type>>
verifyTPReluCommon(TPReluOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type tt = op.getTmp() ? op.getTmp().getType() : Type{};
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }
  if (tt && failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }

  Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
  if (!e0 || !e1 || !ed) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (e0 != e1 || e0 != ed) {
    op.emitOpError("expects dst/src0/src1 to have the same element type");
    return failure();
  }
  if (!(e0.isF16() || e0.isF32())) {
    op.emitOpError("expects dst/src0/src1 element type to be f16 or f32");
    return failure();
  }
  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) || !isRowMajorTileBuf(td)) {
    op.emitOpError("expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, t0, td, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(op, t1, td, "src1", "dst"))) {
    return failure();
  }

  auto s0 = getShapeVec(t0), s1 = getShapeVec(t1), sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd) {
    op.emitOpError("expects src0/src1/dst to have the same shape");
    return failure();
  }
  return std::make_tuple(t0, t1, tt, td);
}

static LogicalResult verifyTPReluA2A3Tmp(TPReluOp op, Type tt, Type td) {
  Type tmpElem = getElemTy(tt);
  auto tmpIntTy = mlir::dyn_cast<IntegerType>(tmpElem);
  if (!tmpIntTy || tmpIntTy.getWidth() != 8) {
    return op.emitOpError("expects A2/A3 tmp element type to be u8");
  }
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto tmpShape = getShapeVec(tt);
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (tmpShape.size() != 2 || dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  if (dstValid[0] != ShapedType::kDynamic && tmpShape[0] != ShapedType::kDynamic &&
      tmpShape[0] < dstValid[0] + 1) {
    return op.emitOpError()
           << "expects A2/A3 tmp shape[0] to be at least dst valid_shape[0] + 1 ("
           << (dstValid[0] + 1) << ")";
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic) {
    int64_t packedMaskCols = llvm::divideCeil(dstValid[1], int64_t{8});
    if (tmpValid[1] < packedMaskCols) {
      return op.emitOpError()
             << "expects A2/A3 tmp valid_shape[1] to be at least ceil(dst valid_shape[1] / 8) ("
             << packedMaskCols << ")";
    }
  }
  if (dstValid[0] == ShapedType::kDynamic ||
      dstValid[1] == ShapedType::kDynamic) {
    return op.emitOpError(
        "expects A2/A3 tprelu dst valid_shape to be static when tmp is provided");
  }
  int64_t packedCols = std::max<int64_t>(
      32, llvm::divideCeil(llvm::divideCeil(dstValid[1], int64_t{8}),
                           int64_t{32}) *
              32);
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstValid[0] + 1) * static_cast<uint64_t>(packedCols)))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTPReluA2A3(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  if (!tt) {
    return success();
  }
  if (failed(verifyTPReluA2A3Tmp(op, tt, td))) {
    return failure();
  }
  if (auto arch = getVerifierArchName(op.getOperation());
      arch && arch->equals_insensitive("a3")) {
    if (op.getSrc0() == op.getSrc1() || op.getSrc0() == op.getTmp() ||
        op.getSrc0() == op.getDst() || op.getSrc1() == op.getTmp() ||
        op.getSrc1() == op.getDst() || op.getTmp() == op.getDst()) {
      return op.emitOpError(
          "expects A3 src0, src1, tmp, and dst to use different storage");
    }
  }
  return success();
}

static LogicalResult verifyTPReluA5(TPReluOp op) {
  auto tysOr = verifyTPReluCommon(op);
  if (failed(tysOr)) {
    return failure();
  }
  auto [t0, t1, tt, td] = *tysOr;
  (void)t0;
  (void)t1;
  (void)td;
  if (tt && failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPReluOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTPReluA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTPReluA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TQuantOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src, fp, offset, dst, tmp;
  Type srcTy, fpTy, offsetTy, dstTy, tmpTy;
  bool hasOffset = false;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() || parser.parseOperand(fp)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(offset)) {
      return failure();
    }
    hasOffset = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(fpTy)) {
    return failure();
  }
  if (hasOffset) {
    if (parser.parseComma() || parser.parseType(offsetTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(dstTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (failed(parsePTOInherentAttrs<TQuantOp>(
          parser, result, parsedAttrs, {"quant_type", "operandSegmentSizes"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(fp, fpTy, result.operands)) {
    return failure();
  }
  if (hasOffset) {
    if (parser.resolveOperand(offset, offsetTy, result.operands)) {
      return failure();
    }
  }
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  auto &properties = result.getOrAddProperties<TQuantOp::Properties>();
  llvm::copy(ArrayRef<int32_t>({1, 1, hasOffset ? 1 : 0, hasTmp ? 1 : 0, 1}),
             properties.operandSegmentSizes.begin());
  return success();
}

void mlir::pto::TQuantOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getFp();
  if (auto offset = getOffset()) {
    p << ", " << offset << " : " << getSrc().getType() << ", "
      << getFp().getType() << ", " << offset.getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getFp().getType() << ")";
  }
  p << " outs(" << getDst();
  if (auto tmp = getTmp()) {
    p << ", " << tmp << " : " << getDst().getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << getDst().getType() << ")";
  }
  NamedAttrList attrs =
      getNonInherentAttrs(getOperation(), {"quant_type", "operandSegmentSizes"});
  attrs.append("quant_type", getQuantTypeAttr());
  p.printOptionalAttrDict(attrs.getAttrs());
}

ParseResult mlir::pto::TQuantMxOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  Type srcTy;
  SmallVector<OpAsmParser::UnresolvedOperand, 5> outOperands;
  SmallVector<Type, 5> outTypes;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseColonType(srcTy) ||
      parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return failure();
  }

  do {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand)) {
      return failure();
    }
    outOperands.push_back(operand);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseColon()) {
    return failure();
  }

  do {
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    outTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRParen()) {
    return failure();
  }

  if (outOperands.size() != outTypes.size()) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects the number of outs operands to match the number of outs types");
  }
  if (outOperands.size() != 4 && outOperands.size() != 5) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects 4 or 5 operands in outs(...)");
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (!llvm::isa_and_nonnull<pto::QuantTypeAttr>(
          result.attributes.get("quant_type"))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "expects quant_type attribute");
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  for (auto [operand, type] : llvm::zip_equal(outOperands, outTypes)) {
    if (parser.resolveOperand(operand, type, result.operands)) {
      return failure();
    }
  }

  return success();
}

void mlir::pto::TQuantMxOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << " : " << getSrc().getType() << ")";
  p << " outs(" << getDst() << ", " << getExp() << ", " << getMax() << ", "
    << getScaling();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz;
  }
  p << " : " << getDst().getType() << ", " << getExp().getType() << ", "
    << getMax().getType() << ", " << getScaling().getType();
  if (auto expZz = getExpZz()) {
    p << ", " << expZz.getType();
  }
  p << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

static LogicalResult verifyTQuantStructural(TQuantOp op) {
  Type dstElemTy = getElemTy(op.getDst().getType());
  auto dstIntTy = dyn_cast<IntegerType>(dstElemTy);
  if (op.getQuantType() == mlir::pto::QuantType::INT8_SYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_SYM quantization requires an fp operand";
    }
    if (op.getOffset()) {
      return op.emitOpError()
             << "INT8_SYM quantization must not have an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_SYM quantization";
    }
  } else if (op.getQuantType() == mlir::pto::QuantType::INT8_ASYM) {
    if (!op.getFp()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an fp operand";
    }
    if (!op.getOffset()) {
      return op.emitOpError()
             << "INT8_ASYM quantization requires an offset operand";
    }
    if (!dstIntTy || dstIntTy.getWidth() != 8) {
      return op.emitOpError()
             << "expects dst element type i8/ui8 for INT8_ASYM quantization";
    }
  } else {
    return op.emitOpError("expects plain tquant quant_type to be INT8_SYM or INT8_ASYM; use tquant.mx for MX quantization");
  }
  return success();
}

static LogicalResult verifyTQuantInt8Common(TQuantOp op) {
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, fpTy, "fp")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (op.getTmp() && failed(verifyTileBufCommon(op, op.getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (!getElemTy(srcTy).isF32()) {
    return op.emitOpError() << "expects src to have element type f32";
  }
  if (op.getOffset()) {
    Type offsetTy = op.getOffset().getType();
    if (failed(verifyTileBufCommon(op, offsetTy, "offset"))) {
      return failure();
    }
    if (!getElemTy(offsetTy).isF32()) {
      return op.emitOpError() << "expects offset to have element type f32";
    }
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (!getElemTy(tmpTy).isF32()) {
      return op.emitOpError() << "expects tmp to have element type f32";
    }
  }
  return success();
}

static LogicalResult verifyTQuantA2A3Tmp(TQuantOp op, Type srcTy, Type tmpTy) {
  if (failed(verifyTileBufSameElemType(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to use row-major layout";
  }
  if (getShapeVec(srcTy) != getShapeVec(tmpTy)) {
    return op.emitOpError() << "expects A2/A3 tmp to have the same shape as src";
  }
  if (failed(verifyTileBufSameValidShape(op, srcTy, tmpTy, "src", "tmp"))) {
    return failure();
  }
  auto requiredBytes = getStaticByteSize(srcTy);
  if (!requiredBytes) {
    return op.emitOpError(
        "expects A2/A3 tquant src shape to be static when tmp is provided");
  }
  return verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes);
}

static LogicalResult verifyTQuantA2A3Param(TQuantOp op, Type paramTy, Type dstTy,
                                           StringRef paramName) {
  if (isRowMajorTileBuf(paramTy)) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " to use non-row-major layout";
  }
  auto paramValid = getValidShapeVec(paramTy);
  auto dstValid = getValidShapeVec(dstTy);
  if (paramValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " and dst to have rank-2 valid_shape";
  }
  if (paramValid[0] != ShapedType::kDynamic &&
      dstValid[0] != ShapedType::kDynamic && paramValid[0] != dstValid[0]) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[0] to equal dst valid_shape[0]";
  }
  if (paramValid[1] != ShapedType::kDynamic && paramValid[1] != 1) {
    return op.emitOpError() << "expects A2/A3 " << paramName
                            << " valid_shape[1] to be 1";
  }
  return success();
}

static LogicalResult verifyTQuantA2A3(TQuantOp op) {
  if (failed(verifyTQuantInt8Common(op))) {
    return failure();
  }
  Type srcTy = op.getSrc().getType();
  Type fpTy = op.getFp().getType();
  Type dstTy = op.getDst().getType();
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return op.emitOpError()
           << "expects A2/A3 src and dst to use row-major layout";
  }
  if (op.getTmp() &&
      failed(verifyTQuantA2A3Tmp(op, srcTy, op.getTmp().getType()))) {
    return failure();
  }
  if (failed(verifyTQuantA2A3Param(op, fpTy, dstTy, "fp"))) {
    return failure();
  }
  if (op.getOffset() &&
      failed(verifyTQuantA2A3Param(op, op.getOffset().getType(), dstTy, "offset"))) {
    return failure();
  }
  return success();
}

mlir::LogicalResult mlir::pto::TQuantOp::verify() {
  if (failed(verifyTQuantStructural(*this))) {
    return failure();
  }
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTQuantA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantInt8Common(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static std::optional<int64_t> mxCheckedMul(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      (rhs != 0 && lhs > std::numeric_limits<int64_t>::max() / rhs)) {
    return std::nullopt;
  }
  return lhs * rhs;
}

static std::optional<int64_t> mxCheckedAdd(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return std::nullopt;
  }
  return lhs + rhs;
}

static std::optional<int64_t> mxCeilDiv(int64_t value, int64_t divisor) {
  if (value < 0 || divisor <= 0) {
    return std::nullopt;
  }
  auto plus = mxCheckedAdd(value, divisor - 1);
  return plus ? std::optional<int64_t>(*plus / divisor) : std::nullopt;
}

static std::optional<int64_t> mxAlignTo(int64_t value, int64_t alignment) {
  auto quotient = mxCeilDiv(value, alignment);
  if (!quotient || *quotient > std::numeric_limits<int64_t>::max() / alignment) {
    return std::nullopt;
  }
  return *quotient * alignment;
}

static std::optional<int64_t> mxCapacityElems(Type type) {
  auto shape = getShapeVec(type);
  return mxCheckedMul(shape[0], shape[1]);
}

static std::optional<int64_t> mxCapacityBytes(Type type) {
  auto elems = mxCapacityElems(type);
  unsigned bytes = getElemByteSize(getElemTy(type));
  if (!elems || bytes == 0 ||
      *elems >
          std::numeric_limits<int64_t>::max() / static_cast<int64_t>(bytes)) {
    return std::nullopt;
  }
  return *elems * static_cast<int64_t>(bytes);
}

static LogicalResult mxRequireCapacity(TQuantMxOp op, StringRef name, Type type,
                                       int64_t required) {
  auto actual = mxCapacityElems(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " elements";
  }
  return success();
}

static LogicalResult mxRequireCapacityBytes(TQuantMxOp op, StringRef name,
                                            Type type, int64_t required) {
  auto actual = mxCapacityBytes(type);
  if (!actual || *actual < required) {
    return op.emitOpError() << "expects " << name
                            << " physical capacity to cover " << required
                            << " bytes";
  }
  return success();
}

static LogicalResult mxRequireCompact(TQuantMxOp op, StringRef name, Type type) {
  auto valid = getValidShapeVec(type);
  auto physical = getShapeVec(type);
  if (valid[0] != 1 && physical[1] != valid[1]) {
    return op.emitOpError() << "expects " << name
                            << " valid elements to form a compact physical prefix";
  }
  return success();
}

static LogicalResult mxRequireStaticShape(TQuantMxOp op, StringRef name,
                                          Type type) {
  for (int64_t dim : getValidShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  for (int64_t dim : getShapeVec(type)) {
    if (dim == ShapedType::kDynamic) {
      return op.emitOpError() << "expects static valid and physical shapes for "
                              << name << " in MX quantization";
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxTilesAndForm(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  if (failed(verifyNDStyleVecTile(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst", /*allowLowPrecision=*/true)) ||
      failed(verifyNDStyleVecTile(op, expTy, "exp")) ||
      failed(verifyNDStyleVecTile(op, maxTy, "max")) ||
      failed(verifyNDStyleVecTile(op, scalingTy, "scaling"))) {
    return failure();
  }
  if (op.getExpZz() &&
      failed(verifyNDStyleVecTile(op, op.getExpZz().getType(), "exp_zz"))) {
    return failure();
  }
  const bool isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  if (op.getInterleave() && !isDn) {
    return op.emitOpError("expects interleave to be used only with grpAxis=axis0");
  }
  if (op.getExpZz() && isDn) {
    return op.emitOpError("expects the deprecated exp_zz form to use grpAxis=axis1; use pto.tmov with a non-scaling tmp for axis0 exponents");
  }
  auto quantType = op.getQuantType();
  if (quantType != mlir::pto::QuantType::MXFP8 &&
      quantType != mlir::pto::QuantType::MXFP4_E2M1) {
    return op.emitOpError("expects quant_type to be MXFP8 or MXFP4_E2M1");
  }
  if (op.getExpZz() && !op.getStoreMode()) {
    return op.emitOpError("expects storeMode when exp_zz is present");
  }
  if (op.getStoreMode() && !op.getExpZz()) {
    return op.emitOpError("expects exp_zz when storeMode is present");
  }
  if (op.getStoreMode() && op.getQuantScaleAlg() != mlir::pto::QuantScaleAlg::OCP) {
    return op.emitOpError("storeMode form must not override quantScaleAlg");
  }
  return success();
}

static LogicalResult verifyTQuantMxElemTypes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type srcElem = getElemTy(srcTy);
  Type dstElem = getElemTy(op.getDst().getType());
  Type expElem = getElemTy(op.getExp().getType());
  Type maxElem = getElemTy(op.getMax().getType());
  Type scalingElem = getElemTy(op.getScaling().getType());
  if (!(srcElem.isF32() || srcElem.isF16() || srcElem.isBF16())) {
    return op.emitOpError("expects src element type to be f32/f16/bf16");
  }
  if (!expElem.isInteger(8)) {
    return op.emitOpError("expects exp element type to be i8/ui8");
  }
  if (op.getExpZz() && !getElemTy(op.getExpZz().getType()).isInteger(8)) {
    return op.emitOpError("expects exp_zz element type to be i8/ui8");
  }
  if (maxElem != srcElem) {
    return op.emitOpError("expects max element type to match src element type");
  }
  if (scalingElem != srcElem) {
    return op.emitOpError("expects scaling element type to match src element type");
  }
  if (op.getQuantType() == mlir::pto::QuantType::MXFP8) {
    if (!dstElem.isInteger(8)) {
      return op.emitOpError("expects MXFP8 dst element type to be i8/ui8");
    }
  } else {
    if (!isa<pto::F4E2M1x2Type>(dstElem)) {
      return op.emitOpError("expects MXFP4_E2M1 dst element type to be !pto.f4E2M1x2");
    }
    if (!(srcElem.isF16() || srcElem.isBF16())) {
      return op.emitOpError("expects MXFP4_E2M1 src element type to be f16/bf16");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxShapes(TQuantMxOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  Type expTy = op.getExp().getType();
  Type maxTy = op.getMax().getType();
  Type scalingTy = op.getScaling().getType();
  auto srcValid = getValidShapeVec(srcTy);
  auto dstValid = getValidShapeVec(dstTy);
  auto expValid = getValidShapeVec(expTy);
  auto maxValid = getValidShapeVec(maxTy);
  auto scalingValid = getValidShapeVec(scalingTy);
  if (srcValid.size() != 2 || dstValid.size() != 2 || expValid.size() != 2 ||
      maxValid.size() != 2 || scalingValid.size() != 2 ||
      getShapeVec(srcTy).size() != 2 || getShapeVec(dstTy).size() != 2 ||
      getShapeVec(expTy).size() != 2 || getShapeVec(maxTy).size() != 2 ||
      getShapeVec(scalingTy).size() != 2) {
    return op.emitOpError("expects rank-2 valid and physical shapes for MX quantization");
  }
  if (op.getExpZz() && (getValidShapeVec(op.getExpZz().getType()).size() != 2 ||
                        getShapeVec(op.getExpZz().getType()).size() != 2)) {
    return op.emitOpError("expects rank-2 valid and physical shapes for exp_zz");
  }
  for (auto [name, type] : {std::pair<StringRef, Type>("src", srcTy),
                            {"dst", dstTy}, {"exp", expTy}, {"max", maxTy},
                            {"scaling", scalingTy}}) {
    if (failed(mxRequireStaticShape(op, name, type))) {
      return failure();
    }
  }
  if (op.getExpZz()) {
    Type expZzTy = op.getExpZz().getType();
    if (llvm::is_contained(getValidShapeVec(expZzTy), ShapedType::kDynamic) ||
        llvm::is_contained(getShapeVec(expZzTy), ShapedType::kDynamic)) {
      return op.emitOpError("expects static valid and physical shapes for exp_zz in the deprecated fused MX quantization form");
    }
  }
  if (failed(verifyTileBufSameElemType(op, srcTy, maxTy, "src", "max")) ||
      failed(verifyTileBufSameElemType(op, srcTy, scalingTy, "src", "scaling")) ||
      failed(verifyTileBufSameLogicalExtent(op, srcTy, dstTy, "src", "dst",
                                            /*compareValidShape=*/true))) {
    return failure();
  }
  return success();
}

struct TQuantMxA5 {
  Type srcTy, dstTy, expTy, maxTy, scalingTy;
  Type srcElem;
  SmallVector<int64_t, 4> dstValid, expValid, expPhysical;
  SmallVector<int64_t, 4> dstPhysical, maxPhysical, scalingPhysical;
  bool isDn, isMxFp4;
  int64_t srcRows, srcCols, srcPhysicalRows, srcPhysicalCols;
  int64_t pack, dstValidCols, groups;
};

static LogicalResult mxRequireGroupedShape(TQuantMxOp op, const TQuantMxA5 &s,
                                           StringRef name, Type type,
                                           bool allowLegacy) {
  auto valid = getValidShapeVec(type);
  SmallVector<int64_t, 2> canonical = {s.isDn ? s.srcRows / 32 : s.srcRows,
                                       s.isDn ? s.srcCols : s.srcCols / 32};
  if (llvm::equal(valid, canonical)) {
    return success();
  }
  if (!s.isDn && allowLegacy && valid[0] == 1 && valid[1] == s.groups) {
    return success();
  }
  return op.emitOpError()
         << "expects " << name << " valid_shape to match "
         << (s.isDn ? "canonical [M/32, N] for grpAxis=axis0"
                    : "canonical [M, N/32] or legacy flat [1, M*N/32] for grpAxis=axis1");
}

static FailureOr<TQuantMxA5> buildTQuantMxA5State(TQuantMxOp op) {
  TQuantMxA5 s;
  s.srcTy = op.getSrc().getType();
  s.dstTy = op.getDst().getType();
  s.expTy = op.getExp().getType();
  s.maxTy = op.getMax().getType();
  s.scalingTy = op.getScaling().getType();
  s.srcElem = getElemTy(s.srcTy);
  auto srcValid = getValidShapeVec(s.srcTy);
  s.dstValid = getValidShapeVec(s.dstTy);
  s.expValid = getValidShapeVec(s.expTy);
  auto srcPhysical = getShapeVec(s.srcTy);
  s.dstPhysical = getShapeVec(s.dstTy);
  s.expPhysical = getShapeVec(s.expTy);
  s.maxPhysical = getShapeVec(s.maxTy);
  s.scalingPhysical = getShapeVec(s.scalingTy);
  s.isDn = op.getGrpAxis() == pto::MxGroupAxis::Axis0;
  s.srcRows = srcValid[0];
  s.srcCols = srcValid[1];
  s.srcPhysicalRows = srcPhysical[0];
  s.srcPhysicalCols = srcPhysical[1];
  if (s.srcRows <= 0 || s.srcCols <= 0 || s.srcPhysicalRows < s.srcRows ||
      s.srcPhysicalCols < s.srcCols) {
    return op.emitOpError("expects positive source valid shape within physical shape");
  }
  if ((s.isDn ? s.srcRows : s.srcCols) % 32 != 0) {
    return op.emitOpError() << "expects src valid_shape[" << (s.isDn ? 0 : 1)
                            << "] to be a multiple of 32 when grpAxis is "
                            << (s.isDn ? "axis0" : "axis1");
  }
  auto groupsOr = mxCheckedMul(s.srcRows, s.srcCols / 32);
  if (!groupsOr) {
    return op.emitOpError("cannot compute MX quantization group count without overflow");
  }
  s.groups = *groupsOr;
  s.isMxFp4 = op.getQuantType() == mlir::pto::QuantType::MXFP4_E2M1;
  s.pack = s.isMxFp4 ? 2 : 1;
  s.dstValidCols = s.isMxFp4 ? s.srcCols / 2 : s.srcCols;
  return s;
}

static LogicalResult verifyTQuantMxGrouping(TQuantMxOp op, const TQuantMxA5 &s) {
  if (op.getExpZz()) {
    auto expZzElements =
        mxCheckedMul(getValidShapeVec(op.getExpZz().getType())[0],
                     getValidShapeVec(op.getExpZz().getType())[1]);
    if (!expZzElements || *expZzElements != s.groups) {
      return op.emitOpError("expects exp_zz valid element count to equal MX group count");
    }
  }
  if (failed(mxRequireGroupedShape(op, s, "max", s.maxTy, /*allowLegacy=*/true)) ||
      failed(mxRequireGroupedShape(op, s, "scaling", s.scalingTy,
                                   /*allowLegacy=*/true))) {
    return failure();
  }
  if (!op.getInterleave()) {
    return mxRequireGroupedShape(op, s, "exp", s.expTy, /*allowLegacy=*/true);
  }
  if (s.srcRows % 64 != 0) {
    return op.emitOpError("expects src valid rows to be a multiple of 64 when interleave is true");
  }
  if (s.srcPhysicalRows % 64 != 0) {
    return op.emitOpError("expects src physical rows to be a multiple of 64 when interleave is true");
  }
  auto doubledValidCols = mxCheckedMul(s.srcCols, 2);
  if (!doubledValidCols) {
    return op.emitOpError("cannot compute interleaved exp valid shape without overflow");
  }
  if (s.expValid[0] != s.srcRows / 64 || s.expValid[1] != *doubledValidCols) {
    return op.emitOpError("expects exp valid_shape to match [M/64, 2N] for grpAxis=axis0 with interleave=true");
  }
  return success();
}

static LogicalResult verifyTQuantMxDstShape(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4 && s.srcPhysicalCols % 2 != 0) {
    return op.emitOpError("expects MXFP4 src physical cols to be even for packed destination addressing");
  }
  if (s.dstValid[0] != s.srcRows || s.dstValid[1] != s.dstValidCols) {
    return op.emitOpError() << "expects dst valid_shape to be [" << s.srcRows
                            << ", " << s.dstValidCols << "] for MX quantization";
  }
  if (s.dstPhysical[0] < s.srcRows) {
    return op.emitOpError("expects dst physical rows to cover src valid rows");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis0(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.isMxFp4) {
    if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
      return op.emitOpError("expects MXFP4 axis0 dst physical cols to equal src physical cols / 2");
    }
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
    if ((s.srcPhysicalCols / s.pack) % 32 != 0 && s.srcElem.isF16()) {
      return op.emitOpError("does not support FP16 MXFP4 axis0 when packed source stride is not a multiple of 32 bytes");
    }
  } else {
    auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.dstPhysical[1]);
    auto required =
        dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
    if (!required || failed(mxRequireCapacity(op, "dst", s.dstTy, *required))) {
      return failure();
    }
  }
  if (s.maxPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects max physical cols to equal src physical cols for grpAxis=axis0");
  }
  if (s.scalingPhysical[1] != s.srcPhysicalCols) {
    return op.emitOpError("expects scaling physical cols to equal src physical cols for grpAxis=axis0");
  }
  auto auxRequired = mxCheckedMul(s.srcRows / 32, s.srcPhysicalCols);
  if (!auxRequired || failed(mxRequireCapacity(op, "max", s.maxTy, *auxRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, *auxRequired))) {
    return failure();
  }
  if (!op.getInterleave()) {
    if (s.expPhysical[1] != s.srcPhysicalCols) {
      return op.emitOpError("expects exp physical cols to equal src physical cols for grpAxis=axis0");
    }
    if (failed(mxRequireCapacity(op, "exp", s.expTy, *auxRequired))) {
      return failure();
    }
  } else {
    auto doubledPhysicalCols = mxCheckedMul(s.srcPhysicalCols, 2);
    auto alignedPhysicalCols =
        doubledPhysicalCols ? mxAlignTo(*doubledPhysicalCols, 32) : std::nullopt;
    if (!alignedPhysicalCols) {
      return op.emitOpError("cannot compute interleaved exp physical cols without overflow");
    }
    if (s.expPhysical[0] != s.srcPhysicalRows / 64) {
      return op.emitOpError("expects interleaved exp physical rows to be src physical rows / 64");
    }
    if (s.expPhysical[1] != *alignedPhysicalCols) {
      return op.emitOpError("expects interleaved exp physical cols to be align32(2 * src physical cols)");
    }
  }
  if (s.srcElem.isF32() && op.getInterleave()) {
    return op.emitOpError("does not support FP32 interleave with the pinned pto-isa revision");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Flat(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.srcPhysicalCols != s.srcCols) {
    return op.emitOpError("expects axis1 flat exp to use a tight source with physical cols equal to valid cols");
  }
  if (s.expValid[0] != 1 || s.expValid[1] != s.groups) {
    return op.emitOpError("expects axis1 flat exp valid_shape to match legacy flat [1, M*N/32]");
  }
  if (failed(mxRequireCapacity(op, "exp", s.expTy, s.groups))) {
    return failure();
  }
  auto srcElems = mxCheckedMul(s.srcPhysicalRows, s.srcPhysicalCols);
  auto validElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
  bool unroll = srcElems && validElems && *srcElems > 1024 &&
                *srcElems % 256 == 0 && *validElems % 256 == 0;
  if (s.srcElem.isF32()) {
    auto scaleGroups =
        unroll ? mxCheckedMul(s.groups, 2) : std::optional<int64_t>(s.groups);
    auto aligned = scaleGroups ? mxAlignTo(*scaleGroups, 64) : std::nullopt;
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 4) : std::nullopt;
    StringRef scalingName = unroll ? "axis1 flat unrolled f32 scaling"
                                   : "axis1 flat f32 scaling";
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, scalingName, s.scalingTy,
                                      *requiredBytes))) {
      return failure();
    }
  } else if (op.getQuantScaleAlg() == pto::QuantScaleAlg::OCP) {
    auto aligned = mxAlignTo(s.groups, 128);
    auto requiredBytes = aligned ? mxCheckedMul(*aligned, 2) : std::nullopt;
    if (!requiredBytes ||
        failed(mxRequireCapacityBytes(op, "axis1 flat B16 OCP scaling",
                                      s.scalingTy, *requiredBytes))) {
      return failure();
    }
  } else {
    return op.emitOpError("does not support axis1 flat B16 NV quantization with the pinned pto-isa revisions");
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1Canonical(TQuantMxOp op,
                                                  const TQuantMxA5 &s) {
  if (s.expValid[0] == 1) {
    return op.emitOpError("expects legacy flat exp to use physical rows == 1");
  }
  if (s.srcElem.isF16() || s.srcElem.isBF16()) {
    return op.emitOpError("does not support axis1 canonical 2D B16 quantization with the pinned pto-isa revision");
  }
  SmallVector<int64_t, 2> canonicalShape = {s.srcRows, s.srcCols / 32};
  if (!llvm::equal(s.expValid, canonicalShape)) {
    return op.emitOpError("expects exp valid_shape to match canonical [M, N/32] for grpAxis=axis1");
  }
  if (s.expPhysical[0] < s.srcRows || s.expPhysical[1] < s.srcCols / 32) {
    return op.emitOpError("expects axis1 canonical exp physical shape to cover [M, N/32]");
  }
  auto expPrefix = mxCheckedMul(s.srcRows - 1, s.expPhysical[1]);
  auto expRequired =
      expPrefix ? mxCheckedAdd(*expPrefix, s.srcCols / 32) : std::nullopt;
  if (!expRequired || failed(mxRequireCapacity(op, "exp", s.expTy, *expRequired)) ||
      failed(mxRequireCapacity(op, "scaling", s.scalingTy, s.groups))) {
    return failure();
  }
  return success();
}

static LogicalResult verifyTQuantMxAxis1(TQuantMxOp op, const TQuantMxA5 &s) {
  if (s.dstPhysical[1] != s.srcPhysicalCols / s.pack) {
    if (s.isMxFp4) {
      return op.emitOpError("expects MXFP4 axis1 dst physical cols to equal src physical cols / 2");
    }
    return op.emitOpError("expects MXFP8 axis1 dst physical cols to equal src physical cols");
  }
  auto dstPrefix = mxCheckedMul(s.srcRows - 1, s.srcPhysicalCols / s.pack);
  auto dstRequired =
      dstPrefix ? mxCheckedAdd(*dstPrefix, s.dstValidCols) : std::nullopt;
  if (!dstRequired || failed(mxRequireCapacity(op, "dst", s.dstTy, *dstRequired))) {
    return failure();
  }
  if (failed(mxRequireCompact(op, "max", s.maxTy)) ||
      failed(mxRequireCompact(op, "scaling", s.scalingTy)) ||
      failed(mxRequireCapacity(op, "max", s.maxTy, s.groups))) {
    return failure();
  }
  const bool flat = s.expPhysical[0] == 1;
  if (flat) {
    return verifyTQuantMxAxis1Flat(op, s);
  }
  return verifyTQuantMxAxis1Canonical(op, s);
}

static LogicalResult verifyTQuantMxSrcPadding(TQuantMxOp op,
                                              const TQuantMxA5 &s) {
  if ((s.srcElem.isF16() || s.srcElem.isBF16()) &&
      s.srcCols < s.srcPhysicalCols && 128 % s.srcPhysicalCols == 0) {
    auto validPhysicalElems = mxCheckedMul(s.srcRows, s.srcPhysicalCols);
    if (!validPhysicalElems) {
      return op.emitOpError("cannot compute B16 source padding extent without overflow");
    }
    if (*validPhysicalElems % 128 != 0) {
      return op.emitOpError("does not support padded B16 source whose VL-aligned padding store has an incomplete final VL");
    }
  }
  return success();
}

static LogicalResult verifyTQuantMxA5(TQuantMxOp op) {
  if (failed(verifyTQuantMxTilesAndForm(op)) ||
      failed(verifyTQuantMxElemTypes(op)) || failed(verifyTQuantMxShapes(op))) {
    return failure();
  }
  auto stateOr = buildTQuantMxA5State(op);
  if (failed(stateOr)) {
    return failure();
  }
  const TQuantMxA5 &s = *stateOr;
  if (failed(verifyTQuantMxGrouping(op, s)) ||
      failed(verifyTQuantMxDstShape(op, s))) {
    return failure();
  }
  if (s.isDn) {
    if (failed(verifyTQuantMxAxis0(op, s))) {
      return failure();
    }
  } else {
    if (failed(verifyTQuantMxAxis1(op, s))) {
      return failure();
    }
  }
  return verifyTQuantMxSrcPadding(op, s);
}

mlir::LogicalResult mlir::pto::TQuantMxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tquant.mx is only supported on A5");
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTQuantMxA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDequantOp::verify() {
  // Structural checks: src must be i8 or i16, dst/scale/offset must be f32.
  auto verifyStructural = [&]() -> LogicalResult {
    Type srcElemTy = getElemTy(getSrc().getType());
    auto srcIntTy = dyn_cast<IntegerType>(srcElemTy);
    if (!srcIntTy || !(srcIntTy.getWidth() == 8 || srcIntTy.getWidth() == 16)) {
      return emitOpError()
             << "expects src element type i8 or i16";
    }
    if (!getElemTy(getDst().getType()).isF32()) {
      return emitOpError() << "expects dst element type f32";
    }
    if (!getElemTy(getScale().getType()).isF32()) {
      return emitOpError() << "expects scale element type f32";
    }
    if (!getElemTy(getOffset().getType()).isF32()) {
      return emitOpError() << "expects offset element type f32";
    }
    return success();
  };

  if (failed(verifyStructural())) {
    return failure();
  }

  auto verifyCommon = [&]() -> LogicalResult {
    if (failed(verifyTileBufCommon(*this, getSrc().getType(), "src")) ||
        failed(verifyTileBufCommon(*this, getScale().getType(), "scale")) ||
        failed(verifyTileBufCommon(*this, getOffset().getType(), "offset")) ||
        failed(verifyTileBufCommon(*this, getDst().getType(), "dst"))) {
      return failure();
    }
    return success();
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyCommon())) {
      return failure();
    }
    if (!isRowMajorTileBuf(getSrc().getType()) ||
        !isRowMajorTileBuf(getDst().getType())) {
      return emitOpError()
             << "expects A2/A3 src and dst to use row-major layout";
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult { return verifyCommon(); };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  Type elemTy = getElemTy(ts);
  if (!(elemTy.isF16() || elemTy.isF32())) {
    return emitOpError() << "expects element type to be f16 or f32";
  }
  if (auto arch = getVerifierArchName(getOperation());
      arch && arch->equals_insensitive("a3") && getSrc() == getDst()) {
    return emitOpError("expects A3 trecip src and dst to use different storage");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto verifyByArch = [&](StringRef errorMessage) -> LogicalResult {
    Type srcTy = getSrc().getType();
    Type dstTy = getDst().getType();
    if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
        failed(verifyVecTileCommon(*this, dstTy, "dst"))) {
      return failure();
    }
    if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
        failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
      return failure();
    }
    Type elemTy = getElemTy(srcTy);
    if (!(elemTy.isInteger(32) || elemTy.isF16() || elemTy.isF32())) {
      return emitOpError() << errorMessage;
    }
    return success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyByArch("expects A2/A3 trelu element type to be i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyByArch("expects A5 trelu element type to be i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static LogicalResult verifyTRemNoTmp(TRemOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trem element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemTmpA2A3(TRemOp op, Type tmpTy, Type elem) {
  Type dstTy = op.getDst().getType();
  auto dstValid = getValidShapeVec(dstTy);
  auto tmpValid = getValidShapeVec(tmpTy);
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 2) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 2");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(dstTy);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trem dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tmpTy, static_cast<uint64_t>(2) * static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trem element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmpA5(TRemOp op, Type tmpTy, Type elem) {
  if (failed(verifyVecTileCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trem element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemTmp(TRemOp op, Type elem) {
  Type tmpTy = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tmpTy);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemTmpA2A3(op, tmpTy, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemTmpA5(op, tmpTy, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(*this, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameElemType(*this, src0Ty, dstTy, "src0", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
      failed(verifyTileBufSameValidShape(*this, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(src1Ty) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src0, src1, and dst to use row-major layout");
  }

  Type elem = getElemTy(src0Ty);
  if (!getTmp()) {
    return verifyTRemNoTmp(*this, elem);
  }
  return verifyTRemTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/false, /*allowBf16OnA5=*/false,
      "expects A2/A3 tfmod element type to be i32/i16/f16/f32",
      "expects A5 tfmod element type to be i32/i16/f16/f32");
}

static LogicalResult verifyTRemSNoTmp(TRemSOp op, Type elem) {
  auto verifyA2A3NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isF32())) {
      return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
    }
    return success();
  };
  auto verifyA5NoTmp = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() ||
          elem.isF32())) {
      return op.emitOpError(
          "expects A5 trems element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3NoTmp,
                                verifyA5NoTmp);
}

static LogicalResult verifyTRemSTmpA2A3(TRemSOp op, Type tt, Type elem) {
  Type td = op.getDst().getType();
  auto dstValid = getValidShapeVec(td);
  auto tmpValid = getValidShapeVec(tt);
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (getElemTy(tt) != getElemTy(td)) {
    return op.emitOpError("expects tmp and dst to have the same element type");
  }
  if (tmpValid[0] != ShapedType::kDynamic && tmpValid[0] < 1) {
    return op.emitOpError("expects A2/A3 tmp valid_shape[0] to be at least 1");
  }
  if (dstValid[1] != ShapedType::kDynamic && tmpValid[1] != ShapedType::kDynamic &&
      tmpValid[1] < dstValid[1]) {
    return op.emitOpError("expects A2/A3 tmp valid columns to cover dst valid columns");
  }
  auto dstShape = getShapeVec(td);
  auto elemBytes = getElemByteSize(elem);
  if (dstShape.size() != 2 || dstShape[1] == ShapedType::kDynamic ||
      elemBytes == 0) {
    return op.emitOpError(
        "expects A2/A3 trems dst shape and element size to be static when tmp is provided");
  }
  if (failed(verifyTmpCapacityAtLeast(
          op, tt, static_cast<uint64_t>(dstShape[1]) * elemBytes))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isF32())) {
    return op.emitOpError("expects A2/A3 trems element type to be i32/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmpA5(TRemSOp op, Type tt, Type elem) {
  if (failed(verifyVecTileCommon(op, tt, "tmp"))) {
    return failure();
  }
  if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
    return op.emitOpError("expects A5 trems element type to be i32/i16/f16/f32");
  }
  return success();
}

static LogicalResult verifyTRemSTmp(TRemSOp op, Type elem) {
  Type tt = op.getTmp().getType();
  if (failed(verifyTileBufCommon(op, tt, "tmp"))) {
    return failure();
  }
  auto dstValid = getValidShapeVec(op.getDst().getType());
  auto tmpValid = getValidShapeVec(tt);
  if (dstValid.size() != 2 || tmpValid.size() != 2) {
    return op.emitOpError("expects tmp and dst to be rank-2 tiles");
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRemSTmpA2A3(op, tt, elem);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRemSTmpA5(op, tt, elem);
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, ts, td, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(ts) || !isRowMajorTileBuf(td)) {
    return emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = getElemTy(ts);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }
  if (!getTmp()) {
    return verifyTRemSNoTmp(*this, elem);
  }
  return verifyTRemSTmp(*this, elem);
}

mlir::LogicalResult mlir::pto::TFModSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }

  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A2/A3 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!(elem.isInteger(32) || elem.isInteger(16) || elem.isF16() || elem.isF32())) {
      return emitOpError("expects A5 tfmods element type to be i32/i16/f16/f32");
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmpShape(Operation *op, Type tmpTy, Type dstTy) {
  if (failed(verifyTileBufSameElemType(op, tmpTy, dstTy, "tmp", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(tmpTy)) {
    return op->emitOpError("expects tmp to use row-major layout");
  }
  return verifyTileBufSameValidShape(op, tmpTy, dstTy, "tmp", "dst");
}

static LogicalResult verifyTPowElemType(TPowOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpow element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpow element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpow element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowTmp(TPowOp op, Type dstTy, bool isIntElem) {
  if (isIntElem && op.getTmp()) {
    return op.emitOpError(
        "does not accept tmp when element type is integer (the integer pow "
        "lowering uses the 3-operand form TPOW(dst, base, exp))");
  }
  if (auto tmp = op.getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op.getOperation(), tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op.getOperation()) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op.emitOpError(
            "expects A2/A3 tpow dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPowOp::verify() {
  Type baseTy = getBase().getType();
  Type expTy = getExp().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyTileBufCommon(*this, baseTy, "base")) ||
      failed(verifyTileBufCommon(*this, expTy, "exp")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameElemType(*this, baseTy, dstTy, "base", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, expTy, "base", "exp")) ||
      failed(verifyTileBufSameValidShape(*this, baseTy, dstTy, "base", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(baseTy) || !isRowMajorTileBuf(expTy) ||
      !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects base, exp, and dst to use row-major layout");
  }

  Type elem = getElemTy(baseTy);
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowTmp(*this, dstTy, isIntElem);
}

static LogicalResult verifyTPowSElemType(TPowSOp op, Type elem, bool isIntElem) {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      return op.emitOpError(
          "A2/A3 does not support precisionType=high_precision");
    }
    if (!(isIntElem || elem.isF32())) {
      return op.emitOpError(
          "expects A2/A3 tpows element type to be i8/i16/i32 or f32");
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (op.getPrecisionType() == pto::PowPrecision::HighPrecision) {
      if (!(elem.isF16() || elem.isF32() || elem.isBF16())) {
        return op.emitOpError("expects A5 tpows element type to be f16/f32/bf16 "
                              "when precisionType=high_precision");
      }
    } else {
      if (!(isIntElem || elem.isF16() || elem.isF32())) {
        return op.emitOpError(
            "expects A5 tpows element type to be i8/i16/i32/f16/f32 "
            "when precisionType=default");
      }
    }
    return success();
  };
  return dispatchVerifierByArch(op.getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTPowSTmp(TPowSOp op, Type dstTy, bool isIntElem) {
  if (isIntElem && op.getTmp()) {
    return op.emitOpError(
        "does not accept tmp when element type is integer (the integer pows "
        "lowering uses the 3-operand form TPOWS(dst, src, scalar))");
  }
  if (auto tmp = op.getTmp()) {
    Type tmpTy = tmp.getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (failed(verifyTPowTmpShape(op.getOperation(), tmpTy, dstTy))) {
      return failure();
    }
    if (getTargetArch(op.getOperation()) != PTOArch::A5) {
      auto requiredBytes = getStaticByteSize(dstTy);
      if (!requiredBytes) {
        return op.emitOpError(
            "expects A2/A3 tpows dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
  }
  return success();
}

mlir::LogicalResult mlir::pto::TPowSOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type scalarTy = getScalar().getType();
  if (failed(verifyTileBufCommon(*this, srcTy, "src")) ||
      failed(verifyTileBufCommon(*this, dstTy, "dst"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(*this, srcTy, dstTy, "src", "dst")) ||
      failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(srcTy) || !isRowMajorTileBuf(dstTy)) {
    return emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = getElemTy(srcTy);
  if (scalarTy != elem) {
    return emitOpError("expects scalar type to match the tile element type");
  }

  // Same dtype matrix as TPowOp; see comment in TPowOp::verify.
  bool isIntElem = elem.isInteger(32) || elem.isInteger(16) || elem.isInteger(8);
  if (failed(verifyTPowSElemType(*this, elem, isIntElem))) {
    return failure();
  }
  return verifyTPowSTmp(*this, dstTy, isIntElem);
}


static std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t numel = 1;
  for (int64_t d : shape) {
    if (d == ShapedType::kDynamic) {
      return std::nullopt;
    }
    if (d < 0) {
      return std::nullopt;
    }
    numel *= d;
  }
  return numel;
}

static std::optional<int64_t> getElemBytes(Type elemTy) {
  if (!elemTy) {
    return std::nullopt;
  }
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16()) {
      return 2;
    }
    if (ft.isF32()) {
      return 4;
    }
    if (ft.isF64()) {
      return 8;
    }
    return std::nullopt;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bits = it.getWidth();
    if (bits <= 0) {
      return std::nullopt;
    }
    return std::max<int64_t>(1, bits / 8);
  }
  return std::nullopt;
}

static bool isLocallyBoundTileSource(Value value) {
  if (!value || isa<BlockArgument>(value)) {
    return false;
  }

  if (isa<AllocTileOp, DeclareTileOp>(value.getDefiningOp())) {
    return true;
  }

  if (auto bitcast = value.getDefiningOp<BitcastOp>()) {
    return isLocallyBoundTileSource(bitcast.getSrc());
  }
  if (auto reshape = value.getDefiningOp<TReshapeOp>()) {
    return isLocallyBoundTileSource(reshape.getSrc());
  }

  return false;
}

static std::optional<int64_t> getConstIndexLike(Value v) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return cOp.value();
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    return cInt.value();
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      return ia.getInt();
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndexLike(castOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndexLike(extOp.getIn());
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndexLike(truncOp.getIn());
  }
  return std::nullopt;
}

mlir::LogicalResult mlir::pto::SetValidShapeOp::verify() {
  SmallVector<int64_t> shape;
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != 2) {
    return emitOpError("expects rank-2 tile_buf source");
  }

  ArrayRef<int64_t> validShape = srcTy.getValidShape();
  if (validShape.size() != 2) {
    return emitOpError("expects source validShape to be rank-2");
  }
  if (!srcTy.hasDynamicValid()) {
    return emitOpError("expects source tile_buf to have dynamic validShape (?, ?)");
  }

  shape.assign(srcTy.getShape().begin(), srcTy.getShape().end());

  if (!isLocallyBoundTileSource(getSource())) {
    return emitOpError(
        "requires a locally bound tile source; function arguments/results "
        "are unsupported");
  }

  auto checkDim = [&](Value operand, unsigned dimIdx,
                      StringRef dimName) -> LogicalResult {
    int64_t maxStatic = shape[dimIdx];

    auto constVal = getConstIndexLike(operand);
    if (!constVal) {
      return success();
    }

    if (*constVal < 0) {
      return emitOpError() << "expects " << dimName << " operand to be non-negative";
    }
    if (maxStatic != ShapedType::kDynamic && *constVal > maxStatic) {
      return emitOpError() << "expects " << dimName << " operand <= shape dim ("
                           << maxStatic << ")";
    }
    return success();
  };

  if (failed(checkDim(getValidRow(), /*dimIdx=*/0, "row"))) {
    return failure();
  }
  if (failed(checkDim(getValidCol(), /*dimIdx=*/1, "col"))) {
    return failure();
  }

  return success();
}

mlir::LogicalResult mlir::pto::GetValidShapeOp::verify() {
  auto srcTy = getSource().getType();
  if (srcTy.getRank() != 2) {
    return emitOpError("expects rank-2 tile_buf source");
  }
  if (srcTy.getValidShape().size() != 2) {
    return emitOpError("expects source validShape to be rank-2");
  }
  return success();
}


mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  Type ts = getSrc().getType();
  Type tr = getResult().getType();
  auto srcTb = dyn_cast<pto::TileBufType>(ts);
  auto dstTb = dyn_cast<pto::TileBufType>(tr);
  if (!srcTb || !dstTb) {
    return emitOpError("expects src/result to be !pto.tile_buf types");
  }

  if (failed(verifyTileBufCommon(*this, ts, "src")) ||
      failed(verifyTileBufCommon(*this, tr, "dst"))) {
    return failure();
  }

  if (srcTb.getMemorySpace() != dstTb.getMemorySpace()) {
    return emitOpError("expects src and dst to use the same loc");
  }

  Type srcElem = srcTb.getElementType();
  Type dstElem = dstTb.getElementType();
  auto srcElemBytes = getElemBytes(srcElem);
  auto dstElemBytes = getElemBytes(dstElem);
  if (!srcElem || !dstElem || !srcElemBytes.has_value() || !dstElemBytes.has_value()) {
    return emitOpError("failed to get element byte width for src/dst");
  }

  auto srcNumel = getStaticNumElements(getShapeVec(ts));
  auto dstNumel = getStaticNumElements(getShapeVec(tr));
  if (!srcNumel.has_value() || !dstNumel.has_value()) {
    return emitOpError("expects static shapes for treshape");
  }

  if (srcElemBytes.value() * srcNumel.value() !=
      dstElemBytes.value() * dstNumel.value()) {
    return emitOpError("expects src and dst to have the same total byte size");
  }

  bool srcBoxed =
      srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  bool dstBoxed =
      dstTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox);
  if (srcBoxed != dstBoxed) {
    return emitOpError("cannot reshape between boxed and non-boxed tile layouts");
  }

  return success();
}

mlir::LogicalResult mlir::pto::BitcastOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }

  if (srcTy.getMemorySpace() != dstTy.getMemorySpace()) {
    return emitOpError("expects src/result to have the same memorySpace");
  }

  if (srcTy.getElementType() == dstTy.getElementType()) {
    return emitOpError(
        "expects src/result to have different element types; use "
        "pto.treshape for shape/config changes");
  }

  if (srcTy.getShape() != dstTy.getShape()) {
    return emitOpError("expects src/result to have the same shape; use pto.treshape for shape changes");
  }

  if (srcTy.getValidShape() != dstTy.getValidShape()) {
    return emitOpError("expects src/result to have the same validShape");
  }

  auto srcCfg = srcTy.getConfigAttr();
  auto dstCfg = dstTy.getConfigAttr();
  if (srcCfg != dstCfg) {
    return emitOpError("expects src/result to have the same tile config");
  }

  auto numel = getStaticNumElements(srcTy.getShape());
  if (!numel.has_value()) {
    return emitOpError("expects static shapes for bitcast");
  }

  auto srcBytes = getElemBytes(srcTy.getElementType());
  auto dstBytes = getElemBytes(dstTy.getElementType());
  if (!srcBytes.has_value() || !dstBytes.has_value()) {
    return emitOpError("unsupported element type for bitcast");
  }

  int64_t srcTotalBytes = numel.value() * srcBytes.value();
  int64_t dstTotalBytes = numel.value() * dstBytes.value();
  if (dstTotalBytes > srcTotalBytes) {
    return emitOpError("bitcast result requires more bytes than source storage");
  }

  return success();
}


static LogicalResult verifyTRowExpandCommon(TRowExpandOp op) {
  Type srcTy = op.getSrc().getType();
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyNDStyleVecTile(op, dstTy, "dst"))) {
    return failure();
  }
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  if (!srcSpace || *srcSpace != pto::AddressSpace::VEC) {
    return op.emitOpError("expects src to be in the vec address space");
  }
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getSLayoutValueI32() != static_cast<int32_t>(pto::SLayout::NoneBox)) {
      return op.emitOpError("expects src to use the none_box slayout");
    }
  }
  if (getElemTy(srcTy) != getElemTy(dstTy)) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (!isSupportedVecElemType(getElemTy(srcTy), /*allowBf16=*/true,
                              /*allowInt8=*/true)) {
    return op.emitOpError("expects trowexpand element type to be supported");
  }
  auto srcValid = getValidShapeVec(op.getSrc());
  auto dstValid = getValidShapeVec(op.getDst());
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }
  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. The op
  // writes no elements; accept and skip the non-empty constraints. One-sided
  // empties still fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }
  if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      srcValid[0] != dstValid[0]) {
    return op.emitOpError("expects src and dst to have the same valid_shape[0]");
  }
  if (srcValid[0] != ShapedType::kDynamic && srcValid[0] == 0) {
    return op.emitOpError("expects src valid_shape[0] to be non-zero");
  }
  if (srcValid[1] != ShapedType::kDynamic && srcValid[1] == 0) {
    return op.emitOpError("expects src valid_shape[1] to be non-zero");
  }
  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0) {
    return op.emitOpError("expects dst valid_shape[0] to be non-zero");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0) {
    return op.emitOpError("expects dst valid_shape[1] to be non-zero");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto verify = [&]() -> LogicalResult { return verifyTRowExpandCommon(*this); };
  return dispatchVerifierByArch(getOperation(), verify, verify);
}


ParseResult mlir::pto::TSort32Op::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, idx, tmp, dst;
  Type srcTy, dstTy, idxTy, tmpTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(idx)) {
      return failure();
    }
    if (succeeded(parser.parseOptionalComma())) {
      if (parser.parseOperand(tmp)) {
        return failure();
      }
      hasTmp = true;
    }
  } else {
    return failure();
  }
  if (parser.parseColonType(srcTy) || parser.parseComma() || parser.parseType(idxTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(idx, idxTy, result.operands)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TSort32Op::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getIdx();
  if (getTmp()) {
    p << ", " << getTmp();
    p << " : " << getSrc().getType() << ", " << getIdx().getType()
      << ", " << getTmp().getType() << ")";
  } else {
    p << " : " << getSrc().getType() << ", " << getIdx().getType() << ")";
  }
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRsqrtOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() || parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TRsqrtOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs());
}

// TPOW assembly format (mirrors TRsqrt's optional-tmp style):
//   pto.tpow ins(%base, %exp[, %tmp] : !tile, !tile[, !tile])
//            outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand base, exp, tmp, dst;
  Type baseTy, expTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(base) || parser.parseComma() ||
      parser.parseOperand(exp)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(baseTy) || parser.parseComma() || parser.parseType(expTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(base, baseTy, result.operands) ||
      parser.resolveOperand(exp, expTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPowOp::print(OpAsmPrinter &p) {
  p << " ins(" << getBase() << ", " << getExp();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getBase().getType() << ", " << getExp().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

// TPOWS assembly format:
//   pto.tpows ins(%src, %scalar[, %tmp] : !tile, scalar_t[, !tile])
//             outs(%dst : !tile) [attr-dict]
ParseResult mlir::pto::TPowSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src, scalar, tmp, dst;
  Type srcTy, scalarTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(scalar)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(srcTy) || parser.parseComma() || parser.parseType(scalarTy)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(scalar, scalarTy, result.operands) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPowSOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getScalar();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType() << ", " << getScalar().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ")";
  p << " outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseTRowExpandBinaryLikeOp(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() || parser.parseOperand(src1)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColon()) {
    return failure();
  }
  if (parser.parseType(src0Ty) || parser.parseComma() || parser.parseType(src1Ty)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.parseComma() || parser.parseType(tmpTy)) {
      return failure();
    }
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen()) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands)) {
    return failure();
  }
  if (hasTmp) {
    if (parser.resolveOperand(tmp, tmpTy, result.operands)) {
      return failure();
    }
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }

  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printTRowExpandBinaryLikeOp(OpAsmPrinter &p, Operation *op, Value src0,
                                        Value src1, Value tmp, Value dst) {
  p << " ins(" << src0 << ", " << src1;
  if (tmp) {
    p << ", " << tmp;
    p << " : " << src0.getType() << ", " << src1.getType() << ", "
      << tmp.getType() << ")";
  } else {
    p << " : " << src0.getType() << ", " << src1.getType() << ")";
  }
  p << " outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TRowExpandDivOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandDivOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMulOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMulOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandSubOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandSubOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandAddOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandAddOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandExpdifOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandExpdifOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMaxOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMaxOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

ParseResult mlir::pto::TRowExpandMinOp::parse(OpAsmParser &parser,
                                              OperationState &result) {
  return parseTRowExpandBinaryLikeOp(parser, result);
}

void mlir::pto::TRowExpandMinOp::print(OpAsmPrinter &p) {
  printTRowExpandBinaryLikeOp(p, getOperation(), getSrc0(), getSrc1(), getTmp(),
                              getDst());
}

static FailureOr<Type> verifyTRowExpandBinaryCore(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasTmp && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (failed(verifyTileBufSameElemType(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (getElemTy(src0Ty) != getElemTy(src1Ty)) {
    op->emitOpError("expects src0 and src1 to have the same element type");
    return failure();
  }
  if (!isRowMajorTileBuf(dstTy)) {
    op->emitOpError("expects dst to use row-major layout");
    return failure();
  }
  return getElemTy(src0Ty);
}

enum class TRowExpandBinaryMode {
  Unknown,
  Mode1ColMajorScalar,
  Mode2RowMajorBlock,
};

static bool validShapesCompatibleForTRowExpand(ArrayRef<int64_t> lhs,
                                               ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
      return false;
    }
  }
  return true;
}

static TRowExpandBinaryMode classifyTRowExpandBinaryMode(Type src0Ty,
                                                         Type src1Ty,
                                                         Type dstTy) {
  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return TRowExpandBinaryMode::Unknown;
  }

  Type expandedTy;
  ArrayRef<int64_t> expandedValid;
  if (validShapesCompatibleForTRowExpand(src0Valid, dstValid)) {
    expandedTy = src1Ty;
    expandedValid = src1Valid;
  } else if (validShapesCompatibleForTRowExpand(src1Valid, dstValid)) {
    expandedTy = src0Ty;
    expandedValid = src0Valid;
  } else {
    return TRowExpandBinaryMode::Unknown;
  }

  int64_t expandedCols = expandedValid[1];
  if (isColMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic || expandedCols == 1)) {
    return TRowExpandBinaryMode::Mode1ColMajorScalar;
  }

  std::optional<int64_t> elemBytes = getElemBytes(getElemTy(dstTy));
  if (!elemBytes || *elemBytes == 0) {
    return TRowExpandBinaryMode::Unknown;
  }
  int64_t expectedMode2Cols = 32 / *elemBytes;
  if (isRowMajorTileBuf(expandedTy) &&
      (expandedCols == ShapedType::kDynamic ||
       expandedCols == expectedMode2Cols)) {
    return TRowExpandBinaryMode::Mode2RowMajorBlock;
  }

  return TRowExpandBinaryMode::Unknown;
}

static int64_t getTRowExpandTmpMinBytes(int64_t dstValidRows) {
  if (dstValidRows == ShapedType::kDynamic) {
    return 8192;
  }
  if (dstValidRows < 0) {
    return 8192;
  }
  if (dstValidRows < 256) {
    return ceilDivInt64(dstValidRows, 8) * 256;
  }
  return 30 * 256;
}

static std::optional<int64_t> getStaticTileCapacityBytes(Type ty) {
  auto numElems = getStaticNumElements(getShapeVec(ty));
  auto elemBytes = getElemBytes(getElemTy(ty));
  if (!numElems || !elemBytes) {
    return std::nullopt;
  }
  return *numElems * *elemBytes;
}

static LogicalResult verifyTRowExpandImplicitTmpContract(
    Operation *op, Type src0Ty, Type src1Ty, Type dstTy, Type tmpTy,
    bool hasTmp, PTOArch targetArch) {
  if (!hasTmp || targetArch == PTOArch::A5) {
    return success();
  }

  if (classifyTRowExpandBinaryMode(src0Ty, src1Ty, dstTy) !=
      TRowExpandBinaryMode::Mode1ColMajorScalar) {
    return op->emitOpError(
        "expects A2/A3 tmp-form trowexpand to use mode 1 "
        "(ColMajor per-row scalar expanded operand)");
  }

  if (failed(verifyVecTileStorage(op, tmpTy, "tmp"))) {
    return failure();
  }
  if (getElemTy(tmpTy) != getElemTy(dstTy)) {
    return op->emitOpError("expects tmp and dst to have the same element type");
  }

  auto dstValid = getValidShapeVec(dstTy);
  if (dstValid.size() != 2) {
    return op->emitOpError("expects dst to have rank-2 valid_shape");
  }
  int64_t minBytes = getTRowExpandTmpMinBytes(dstValid[0]);
  std::optional<int64_t> tmpBytes = getStaticTileCapacityBytes(tmpTy);
  if (!tmpBytes) {
    return op->emitOpError(
        "expects A2/A3 trowexpand tmp capacity to be statically known");
  }
  if (*tmpBytes < minBytes) {
    return op->emitOpError()
           << "expects A2/A3 trowexpand tmp capacity to be at least "
           << minBytes << " bytes, but got " << *tmpBytes << " bytes";
  }
  return success();
}

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported =
        elem.isF16() || elem.isF32() ||
        (targetArch == PTOArch::A5 &&
         (elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32)));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpanddiv element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError("expects element type to be f16 or f32");
    }
    if (getPrecisionType() == pto::DivPrecision::HighPrecision && !getTmp()) {
      return emitOpError("expects tmp when precisionType is high_precision");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpandmul element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError(
          "expects A2/A3 trowexpandmul element type to be i16/i32/f16/f32");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto verifyByArch = [&](PTOArch targetArch) -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dstTy = getDst().getType();
    FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
        *this, src0Ty, src1Ty, dstTy, getTmp() ? getTmp().getType() : Type{},
        static_cast<bool>(getTmp()));
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                     elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8));
    if (!supported) {
      if (targetArch == PTOArch::A5) {
        return emitOpError(
            "expects A5 trowexpandsub element type to be i8/i16/i32/f16/f32");
      }
      return emitOpError(
          "expects A2/A3 trowexpandsub element type to be i16/i32/f16/f32");
    }
    if (failed(verifyTRowExpandImplicitTmpContract(
            getOperation(), src0Ty, src1Ty, dstTy,
            getTmp() ? getTmp().getType() : Type{}, static_cast<bool>(getTmp()),
            targetArch))) {
      return failure();
    }
    return mlir::success();
  };
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A3); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyByArch(PTOArch::A5); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static FailureOr<Type> verifyTRowExpandAddCore(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  FailureOr<Type> elemOr = verifyTRowExpandBinaryCore(
      op, src0Ty, src1Ty, dstTy, op.getTmp() ? op.getTmp().getType() : Type{},
      static_cast<bool>(op.getTmp()));
  if (failed(elemOr)) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(op, src0Ty, dstTy, "src0", "dst"))) {
    return failure();
  }
  if (!isRowMajorTileBuf(src0Ty)) {
    return op.emitOpError("expects src0 to use row-major layout");
  }
  Type elem = *elemOr;
  bool supported = elem.isF16() || elem.isF32() || elem.isInteger(16) ||
                   elem.isInteger(32) ||
                   (targetArch == PTOArch::A5 && elem.isInteger(8));
  if (!supported) {
    if (targetArch == PTOArch::A5) {
      return op.emitOpError(
          "expects A5 trowexpandadd element type to be i8/i16/i32/f16/f32");
    }
    return op.emitOpError(
        "expects A2/A3 trowexpandadd element type to be i16/i32/f16/f32");
  }
  return elem;
}

static LogicalResult verifyTRowExpandAddSrc1(TRowExpandAddOp op, Type elem,
                                             PTOArch targetArch) {
  Type src0Ty = op.getSrc0().getType();
  Type src1Ty = op.getSrc1().getType();
  Type dstTy = op.getDst().getType();
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src1Valid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src1 and dst to have rank-2 valid_shape");
  }
  if (src1Valid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
      src1Valid[0] != dstValid[0]) {
    return op.emitOpError("expects src1 valid_shape[0] to equal dst valid_shape[0]");
  }
  bool src1IsRowMajor = isRowMajorTileBuf(src1Ty);
  int64_t expectedCol = elem.isInteger(8)
                            ? 32
                            : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
  int64_t src1Col = src1Valid[1];
  if (src1IsRowMajor) {
    if (src1Col != ShapedType::kDynamic && src1Col != expectedCol) {
      return op.emitOpError("expects row-major src1 valid_shape[1] to be 32/sizeof(dtype)");
    }
  } else {
    if (src1Col != ShapedType::kDynamic && src1Col != 1) {
      return op.emitOpError("expects non-row-major src1 valid_shape[1] to be 1");
    }
  }
  if (failed(verifyTRowExpandImplicitTmpContract(
          op.getOperation(), src0Ty, src1Ty, dstTy,
          op.getTmp() ? op.getTmp().getType() : Type{},
          static_cast<bool>(op.getTmp()), targetArch))) {
    return failure();
  }
  return mlir::success();
}

static LogicalResult verifyTRowExpandAddByArch(TRowExpandAddOp op,
                                               PTOArch targetArch) {
  FailureOr<Type> elemOr = verifyTRowExpandAddCore(op, targetArch);
  if (failed(elemOr)) {
    return failure();
  }
  return verifyTRowExpandAddSrc1(op, *elemOr, targetArch);
}

mlir::LogicalResult mlir::pto::TRowExpandAddOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A3);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandAddByArch(*this, PTOArch::A5);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static LogicalResult verifyTRowExpandReduceLikeOp(Operation *op, Type src0Ty,
                                                  Type src1Ty, Type dstTy,
                                                  Type tmpTy, bool hasTmp,
                                                  PTOArch targetArch,
                                                  bool enforceTmpContract,
                                                  StringRef opName,
                                                  bool allowIntegerTypes) {
  if (failed(verifyTileBufCommon(op, src0Ty, "src0")) ||
      failed(verifyTileBufCommon(op, src1Ty, "src1")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (hasTmp) {
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != getElemTy(dstTy)) {
      return op->emitOpError() << "expects tmp and dst to have the same element type";
    }
  }

  Type elem = getElemTy(dstTy);
  if (!elem || getElemTy(src0Ty) != elem || getElemTy(src1Ty) != elem) {
    return op->emitOpError("expects src0, src1, and dst to have the same element type");
  }
  bool supported = elem.isF16() || elem.isF32() ||
                   (allowIntegerTypes &&
                    (elem.isInteger(16) || elem.isInteger(32) ||
                     (targetArch == PTOArch::A5 && elem.isInteger(8))));
  if (!supported) {
    if (!allowIntegerTypes) {
      return op->emitOpError() << "expects " << opName
                               << " element type to be f16 or f32";
    }
    if (targetArch == PTOArch::A5) {
      return op->emitOpError() << "expects A5 " << opName
                               << " element type to be i8/i16/i32/f16/f32";
    }
    return op->emitOpError() << "expects A2/A3 " << opName
                             << " element type to be i16/i32/f16/f32";
  }

  if (!isRowMajorTileBuf(dstTy)) {
    return op->emitOpError("expects dst to use row-major layout");
  }

  auto src0Valid = getValidShapeVec(src0Ty);
  auto src1Valid = getValidShapeVec(src1Ty);
  auto dstValid = getValidShapeVec(dstTy);
  if (src0Valid.size() != 2 || src1Valid.size() != 2 || dstValid.size() != 2) {
    return op->emitOpError("expects src0, src1, and dst to have rank-2 valid_shape");
  }

  // Fully-empty dst valid region (0x0): dual-AIV no-op replay marker. Element
  // type/layout were already checked above; the op writes no elements, so accept
  // and skip the non-empty broadcast/width constraints. One-sided empties still
  // fall through. See pto-isa#143 for hardware Rv=0 no-op.
  if (dstValid[0] == 0 && dstValid[1] == 0) {
    return success();
  }

  if (dstValid[0] != ShapedType::kDynamic && dstValid[0] == 0) {
    return op->emitOpError("expects dst valid_shape[0] to be non-zero");
  }
  if (dstValid[1] != ShapedType::kDynamic && dstValid[1] == 0) {
    return op->emitOpError("expects dst valid_shape[1] to be non-zero");
  }

  auto validShapeMatches = [](ArrayRef<int64_t> lhs,
                              ArrayRef<int64_t> rhs) -> bool {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (auto [l, r] : llvm::zip(lhs, rhs)) {
      if (l != ShapedType::kDynamic && r != ShapedType::kDynamic && l != r) {
        return false;
      }
    }
    return true;
  };

  const bool src0MatchesDst = validShapeMatches(src0Valid, dstValid);
  const bool src1MatchesDst = validShapeMatches(src1Valid, dstValid);

  auto checkBroadcastOperand = [&](Type operandTy, ArrayRef<int64_t> operandValid,
                                   StringRef operandName,
                                   bool requireNonRowMajor) -> LogicalResult {
    if (operandValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        operandValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << operandName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    int64_t expectedCol = elem.isInteger(8) ? 32 : ((elem.isF16() || elem.isInteger(16)) ? 16 : 8);
    int64_t operandCol = operandValid[1];
    bool operandIsRowMajor = isRowMajorTileBuf(operandTy);
    if (requireNonRowMajor && operandIsRowMajor) {
      return op->emitOpError() << "expects " << operandName
                               << " to use a non-row-major layout when tmp is present";
    }
    if (operandIsRowMajor) {
      if (operandCol != ShapedType::kDynamic && operandCol != expectedCol) {
        return op->emitOpError()
               << "expects row-major " << operandName
               << " valid_shape[1] to be 32/sizeof(dtype)";
      }
      return success();
    }
    if (operandCol != ShapedType::kDynamic && operandCol != 1) {
      return op->emitOpError() << "expects non-row-major " << operandName
                               << " valid_shape[1] to be 1";
    }
    return success();
  };

  auto checkFullAndBroadcast = [&](Type fullTy, ArrayRef<int64_t> fullValid,
                                   StringRef fullName, Type broadcastTy,
                                   ArrayRef<int64_t> broadcastValid,
                                   StringRef broadcastName) -> LogicalResult {
    if (!isRowMajorTileBuf(fullTy)) {
      return op->emitOpError() << "expects " << fullName
                               << " to use row-major layout when it matches dst";
    }
    if (fullValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        fullValid[0] != dstValid[0]) {
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[0] to equal dst valid_shape[0]";
    }
    if (fullValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        fullValid[1] != dstValid[1]) {
      return op->emitOpError() << "expects " << fullName
                               << " valid_shape[1] to equal dst valid_shape[1]";
    }
    return checkBroadcastOperand(broadcastTy, broadcastValid, broadcastName,
                                 /*requireNonRowMajor=*/hasTmp &&
                                     targetArch == PTOArch::A3);
  };

  // (A5 tmp-form invariant is checked earlier, before the empty-marker accept.)

  auto verifyTmpContract = [&]() -> LogicalResult {
    if (!enforceTmpContract) {
      return success();
    }
    return verifyTRowExpandImplicitTmpContract(op, src0Ty, src1Ty, dstTy,
                                               tmpTy, hasTmp, targetArch);
  };

  if (src0MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src0Ty, src0Valid, "src0", src1Ty,
                                        src1Valid, "src1")) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }
  if (src1MatchesDst) {
    if (succeeded(checkFullAndBroadcast(src1Ty, src1Valid, "src1", src0Ty,
                                        src0Valid, "src0")) &&
        succeeded(verifyTmpContract())) {
      return success();
    }
  }

  return op->emitOpError() << "expects one of src0/src1 to match dst valid_shape"
                           << " and the other to be a per-row scalar vector";
}

mlir::LogicalResult mlir::pto::TRowExpandExpdifOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/false,
                                        "trowexpandexpdif",
                                        /*allowIntegerTypes=*/false);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMaxOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmax",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowExpandMinOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A3,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowExpandReduceLikeOp(getOperation(), getSrc0().getType(),
                                        getSrc1().getType(), getDst().getType(),
                                        getTmp() ? getTmp().getType() : Type{},
                                        (bool)getTmp(), PTOArch::A5,
                                        /*enforceTmpContract=*/true,
                                        "trowexpandmin",
                                        /*allowIntegerTypes=*/true);
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static ParseResult parseOptionalTmpRowReductionOp(OpAsmParser &parser,
                                                  OperationState &result) {
  OpAsmParser::UnresolvedOperand src, tmp, dst;
  Type srcTy, tmpTy, dstTy;
  bool hasTmp = false;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, hasTmp ? 1 : 0, 1}));
  return success();
}

static void printOptionalTmpRowReductionOp(OpAsmPrinter &p, Operation *op,
                                           Value src, Value tmp, Value dst) {
  p << " ins(" << src;
  if (tmp) {
    p << ", " << tmp;
  }
  p << " : " << src.getType();
  if (tmp) {
    p << ", " << tmp.getType();
  }
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static ParseResult parseOptionalTmpFixedDpsOp(
    OpAsmParser &parser, OperationState &result, unsigned minInputs,
    unsigned maxInputs, ArrayRef<int32_t> noTmpSegments,
    ArrayRef<int32_t> withTmpSegments) {
  SmallVector<OpAsmParser::UnresolvedOperand> inputs;
  SmallVector<Type> inputTypes;
  OpAsmParser::UnresolvedOperand dst;
  Type dstType;
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return failure();
  }
  do {
    inputs.emplace_back();
    if (parser.parseOperand(inputs.back())) {
      return failure();
    }
  } while (succeeded(parser.parseOptionalComma()));
  if (inputs.size() < minInputs || inputs.size() > maxInputs ||
      parser.parseColon()) {
    return failure();
  }
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (i && parser.parseComma()) {
      return failure();
    }
    Type type;
    if (parser.parseType(type)) {
      return failure();
    }
    inputTypes.push_back(type);
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstType) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.resolveOperands(inputs, inputTypes, parser.getCurrentLocation(),
                             result.operands) ||
      parser.resolveOperand(dst, dstType, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          inputs.size() == minInputs ? noTmpSegments : withTmpSegments));
  return success();
}

static void printOptionalTmpFixedDpsOp(OpAsmPrinter &p, Operation *op,
                                       ArrayRef<Value> inputs, Value dst) {
  p << " ins(";
  llvm::interleaveComma(inputs, p, [&](Value value) { p << value; });
  p << " : ";
  llvm::interleaveComma(inputs, p,
                        [&](Value value) { p << value.getType(); });
  p << ") outs(" << dst << " : " << dst.getType() << ")";
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TTransOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 1, 2, {1, 0, 1},
                                    {1, 1, 1});
}
void mlir::pto::TTransOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TPReluOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TPReluOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TRemOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TRemSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 2, 3, {1, 1, 0, 1},
                                    {1, 1, 1, 1});
}
void mlir::pto::TRemSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getSrc(), getScalar()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 3, 4,
                                    {1, 1, 1, 0, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc0(), getSrc1()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TSelSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  return parseOptionalTmpFixedDpsOp(parser, result, 3, 4,
                                    {1, 1, 0, 1, 1}, {1, 1, 1, 1, 1});
}
void mlir::pto::TSelSOp::print(OpAsmPrinter &p) {
  SmallVector<Value> inputs{getMask(), getSrc()};
  if (getTmp()) {
    inputs.push_back(getTmp());
  }
  inputs.push_back(getScalar());
  printOptionalTmpFixedDpsOp(p, getOperation(), inputs, getDst());
}

ParseResult mlir::pto::TColArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TColArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TColArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMaxOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMaxOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMaxOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowMinOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowArgMinOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowArgMinOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowSumOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowSumOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

ParseResult mlir::pto::TRowProdOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseOptionalTmpRowReductionOp(parser, result);
}

void mlir::pto::TRowProdOp::print(OpAsmPrinter &p) {
  printOptionalTmpRowReductionOp(p, getOperation(), getSrc(), getTmp(),
                                 getDst());
}

mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMaxOp::verify() {
  if (!getTmp()) {
    return verifyTRowArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TRowArgMinOp::verify() {
  if (!getTmp()) {
    return verifyTRowArgReductionNoTmp(getOperation(), getSrc().getType(),
                                       getDst().getType());
  }
  auto verifyA2A3 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA2A3(*this, getSrc().getType(),
                                        getTmp().getType(), getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    return verifyTRowArgReductionOpA5(*this, getSrc().getType(),
                                      getTmp().getType(), getDst().getType());
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto verifyByArch = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyByArch, verifyByArch);
}

mlir::LogicalResult mlir::pto::TInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type src1Ty = getSrc1().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }

    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }

    bool mismatchedValidShapes =
        failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedValidShapes) {
      return failure();
    }

    auto validShape = getValidShapeVec(dst0Ty);
    bool hasInvalidRank = validShape.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src0, src1, dst0, and dst1 to have rank-2 valid_shape");
    }
    bool hasOddValidColumns =
        validShape[1] != ShapedType::kDynamic && (validShape[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects valid_shape[1] to be even");
    }

    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TDeInterleaveOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    return emitOpError("tdeinterleave is only supported on A5 targets");
  };

  auto verifyA5 = [&]() -> LogicalResult {
    Type src0Ty = getSrc0().getType();
    Type dst0Ty = getDst0().getType();
    Type dst1Ty = getDst1().getType();

    bool invalidTile =
        failed(verifyVecTileCommon(*this, src0Ty, "src0")) ||
        failed(verifyVecTileCommon(*this, dst0Ty, "dst0")) ||
        failed(verifyVecTileCommon(*this, dst1Ty, "dst1"));
    if (invalidTile) {
      return failure();
    }
    bool mismatchedElementTypes =
        failed(verifyTileBufSameElemType(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
        failed(verifyTileBufSameElemType(*this, src0Ty, dst1Ty, "src0", "dst1"));
    if (mismatchedElementTypes) {
      return failure();
    }
    if (!isSupportedVecElemType(getElemTy(src0Ty), /*allowBf16=*/true,
                                /*allowInt8=*/true)) {
      return emitOpError("expects vec tile element types to be supported");
    }
    bool hasNonRowMajorTile =
        !isRowMajorTileBuf(src0Ty) || !isRowMajorTileBuf(dst0Ty) ||
        !isRowMajorTileBuf(dst1Ty);
    if (hasNonRowMajorTile) {
      return emitOpError("expects src and dst tiles to use row-major layout");
    }

    auto src0Valid = getValidShapeVec(src0Ty);
    auto dst0Valid = getValidShapeVec(dst0Ty);
    auto dst1Valid = getValidShapeVec(dst1Ty);
    bool hasInvalidRank =
        src0Valid.size() != 2 || dst0Valid.size() != 2 || dst1Valid.size() != 2;
    if (hasInvalidRank) {
      return emitOpError("expects src and dst tiles to have rank-2 valid_shape");
    }

    bool hasSecondSource = getSrcs().size() == 2;
    if (hasSecondSource) {
      Type src1Ty = getSrc1().getType();
      bool invalidSource =
          failed(verifyVecTileCommon(*this, src1Ty, "src1")) ||
          failed(verifyTileBufSameElemType(*this, src0Ty, src1Ty, "src0", "src1")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, src1Ty, "src0", "src1")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, dst0Ty, "src0", "dst0")) ||
          failed(verifyTileBufSameValidShape(*this, src0Ty, dst1Ty, "src0", "dst1"));
      if (invalidSource) {
        return failure();
      }
      if (!isRowMajorTileBuf(src1Ty)) {
        return emitOpError("expects src1 to use row-major layout");
      }
      bool hasOddValidColumns =
          src0Valid[1] != ShapedType::kDynamic && (src0Valid[1] & 1) != 0;
      if (hasOddValidColumns) {
        return emitOpError("expects two-source valid_shape[1] to be even");
      }
      return success();
    }

    bool hasRowMismatchWithDst0 =
        src0Valid[0] != ShapedType::kDynamic &&
        dst0Valid[0] != ShapedType::kDynamic &&
        src0Valid[0] != dst0Valid[0];
    if (hasRowMismatchWithDst0) {
      return emitOpError("expects src0 and dst0 to have the same valid_shape[0]");
    }
    bool hasRowMismatchWithDst1 =
        src0Valid[0] != ShapedType::kDynamic &&
        dst1Valid[0] != ShapedType::kDynamic &&
        src0Valid[0] != dst1Valid[0];
    if (hasRowMismatchWithDst1) {
      return emitOpError("expects src0 and dst1 to have the same valid_shape[0]");
    }
    bool hasOddValidColumns =
        src0Valid[1] != ShapedType::kDynamic && (src0Valid[1] & 1) != 0;
    if (hasOddValidColumns) {
      return emitOpError("expects single-source valid_shape[1] to be even");
    }
    bool hasColumnMismatchWithDst0 =
        src0Valid[1] != ShapedType::kDynamic &&
        dst0Valid[1] != ShapedType::kDynamic &&
        dst0Valid[1] != src0Valid[1] / 2;
    if (hasColumnMismatchWithDst0) {
      return emitOpError(
          "expects dst0 valid_shape[1] to be half of src0 valid_shape[1]");
    }
    bool hasColumnMismatchWithDst1 =
        src0Valid[1] != ShapedType::kDynamic &&
        dst1Valid[1] != ShapedType::kDynamic &&
        dst1Valid[1] != src0Valid[1] / 2;
    if (hasColumnMismatchWithDst1) {
      return emitOpError(
          "expects dst1 valid_shape[1] to be half of src0 valid_shape[1]");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

mlir::LogicalResult mlir::pto::TRowProdOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A2/A3 trowprod element type to be i16/i32/f16/f32");
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (!getTmp()) {
      return verifyTRowReductionNoTmpCommon(
          *this, getSrc().getType(), getDst().getType(),
          "expects A5 trowprod element type to be i16/i32/f16/f32");
    }
    return verifyTRowReductionWithTmpCommon(
        *this, getSrc().getType(), getTmp().getType(), getDst().getType(),
        "expects A5 trowprod element type to be i16/i32/f16/f32");
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  Type ts = getSrc().getType();
  Type td = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, ts, td, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, ts, td, "src", "dst"))) {
    return failure();
  }
  auto ft = mlir::dyn_cast<mlir::FloatType>(getElemTy(ts));
  if (!ft || (!ft.isF16() && !ft.isF32())) {
    return emitOpError("expects element type to be f16 or f32");
  }
  if (getPrecisionType() == pto::RsqrtPrecision::HighPrecision && !getTmp()) {
    return emitOpError("expects tmp when precisionType is high_precision");
  }
  if (auto tmp = getTmp()) {
    Type tt = tmp.getType();
    if (failed(verifyVecTileCommon(*this, tt, "tmp"))) {
      return failure();
    }

    auto tmpElemTy = getElemTy(tt);
    auto tmpElemBytes = getElemBytes(tmpElemTy);
    auto tmpNumel = getStaticNumElements(getShapeVec(tt));
    if (!tmpElemBytes.has_value() || !tmpNumel.has_value()) {
      return emitOpError("expects tmp to have a static, byte-addressable tile type");
    }
    if (tmpElemBytes.value() * tmpNumel.value() < 32) {
      return emitOpError("expects tmp to be at least 32 bytes when provided");
    }
  }
  return mlir::success();
}


static bool isTScatterAllowedDataElem(mlir::Type t) {
  if (t.isF16() || t.isF32() || t.isBF16()) {
    return true;
  }
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
    return (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
  }
  return false;
}

static bool isTScatterAllowedIndexElem(mlir::Type t) {
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
    return (it.getWidth() == 16 || it.getWidth() == 32);
  }
  return false;
}

static unsigned getMaskScatterTimes(mlir::pto::MaskPatternAttr mp) {
  switch (mp.getValue()) {
  case mlir::pto::MaskPattern::P1111:
    return 1;
  case mlir::pto::MaskPattern::P0101:
  case mlir::pto::MaskPattern::P1010:
    return 2;
  default:
    return 4;
  }
}

static LogicalResult verifyTScatterIndexedElemTypes(TScatterOp op, Type ts,
                                                    Type ti, Type td) {
  Type srcElem = getElemTy(ts), dstElem = getElemTy(td), idxElem = getElemTy(ti);
  if (!srcElem || !dstElem || !idxElem) {
    return op.emitOpError("failed to get element type for operands");
  }
  if (srcElem != dstElem) {
    return op.emitOpError("expects src/dst to have the same element type");
  }

  if (!isTScatterAllowedDataElem(srcElem)) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  if (!isTScatterAllowedIndexElem(idxElem)) {
    return op.emitOpError("expects indexes element type to be i16/i32");
  }

  auto bwData = getPTOStorageElemBitWidth(srcElem);
  auto bwIdx  = getPTOStorageElemBitWidth(idxElem);
  if (bwData != 8 && bwData != 16 && bwData != 32) {
    return op.emitOpError("unexpected src/dst element bitwidth");
  }

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes) {
    return op.emitOpError("expects indexes element size to match the documented scatter rule");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedShapes(TScatterOp op, Type ts, Type ti,
                                                 Type td) {
  auto srcValid = getValidShapeVec(ts);
  auto idxValid = getValidShapeVec(ti);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != 2 || idxValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src, indexes and dst to have rank-2 valid_shape");
  }

  for (unsigned d = 0; d < 2; ++d) {
    if (srcValid[d] != ShapedType::kDynamic && idxValid[d] != ShapedType::kDynamic &&
        srcValid[d] != idxValid[d]) {
      return op.emitOpError("expects src and indexes to have the same valid_shape");
    }
    if (srcValid[d] != ShapedType::kDynamic && dstValid[d] != ShapedType::kDynamic &&
        dstValid[d] < srcValid[d]) {
      return op.emitOpError("expects dst valid_shape to be >= src valid_shape");
    }
  }
  return mlir::success();
}

static LogicalResult verifyTScatterIndexedForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type ti = op.getIndexes().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, ti, "indexes")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }
  if (failed(verifyTScatterIndexedElemTypes(op, ts, ti, td))) {
    return failure();
  }
  return verifyTScatterIndexedShapes(op, ts, ti, td);
}

static LogicalResult verifyTScatterMaskAxisShapes(TScatterOp op, StringRef axisVal,
                                                  ArrayRef<int64_t> srcValid,
                                                  ArrayRef<int64_t> dstValid,
                                                  unsigned times) {
  if (axisVal == "row") {
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != srcValid[0]) {
      return op.emitOpError("expects dst valid rows to equal src valid rows for row direction");
    }
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != static_cast<int64_t>(srcValid[1] * times)) {
      return op.emitOpError("expects dst valid cols to equal src valid cols times the mask expansion factor for row direction");
    }
  } else if (axisVal == "col") {
    if (srcValid[1] != ShapedType::kDynamic && dstValid[1] != ShapedType::kDynamic &&
        dstValid[1] != srcValid[1]) {
      return op.emitOpError("expects dst valid cols to equal src valid cols for col direction");
    }
    if (srcValid[0] != ShapedType::kDynamic && dstValid[0] != ShapedType::kDynamic &&
        dstValid[0] != static_cast<int64_t>(srcValid[0] * times)) {
      return op.emitOpError("expects dst valid rows to equal src valid rows times the mask expansion factor for col direction");
    }
  } else {
      return op.emitOpError("Invalid axis value, expected \"row\" or \"col\"");
  }
  return mlir::success();
}

static LogicalResult verifyTScatterMaskForm(TScatterOp op) {
  Type ts = op.getSrc().getType();
  Type td = op.getDst().getType();
  if (failed(verifyVecTileCommon(op, ts, "src")) ||
      failed(verifyVecTileCommon(op, td, "dst"))) {
    return failure();
  }

  auto srcTB = dyn_cast<pto::TileBufType>(ts);
  auto dstTB = dyn_cast<pto::TileBufType>(td);
  if (!srcTB || !dstTB) {
    return op.emitOpError("expects src and dst to be tile_buf types");
  }

  if (getElemTy(ts) != getElemTy(td)) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (!isTScatterAllowedDataElem(getElemTy(ts))) {
    return op.emitOpError("expects src/dst element type to be i8/i16/i32/f16/bf16/f32");
  }

  auto srcValid = getValidShapeVec(ts);
  auto dstValid = getValidShapeVec(td);
  if (srcValid.size() != 2 || dstValid.size() != 2) {
    return op.emitOpError("expects src and dst to have rank-2 valid_shape");
  }

  auto axisAttr = op.getAxisAttr();
  if (!axisAttr) {
    return op.emitOpError("expects mask-pattern tscatter to provide axis attribute");
  }
  StringRef axisVal = axisAttr.getValue();
  auto mp = op.getMaskPatternAttr();
  if (!mp) {
    return op.emitOpError("expects mask-pattern tscatter to provide maskPattern");
  }
  const unsigned times = getMaskScatterTimes(mp);
  if (failed(verifyTScatterMaskAxisShapes(op, axisVal, srcValid, dstValid, times))) {
    return failure();
  }

  if (srcTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) ||
      dstTB.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
    return op.emitOpError("expects mask-pattern tscatter to use row_major blayout");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  const bool hasIndexes = static_cast<bool>(getIndexes());
  const bool hasMaskPattern = static_cast<bool>(getMaskPatternAttr());
  if (hasIndexes == hasMaskPattern) {
    return emitOpError(
        "expects exactly one of indexes operand or maskPattern attribute");
  }
  if (hasIndexes && getAxisAttr()) {
    return emitOpError("axis attribute must not be provided with indexes operand");
  }
  auto verifyForm = [&]() -> LogicalResult {
    if (hasMaskPattern) {
      return verifyTScatterMaskForm(*this);
    }
    return verifyTScatterIndexedForm(*this);
  };
  return dispatchVerifierByArch(getOperation(), verifyForm, verifyForm);
}


static FailureOr<Type> verifyTSelCommon(TSelOp op) {
  Type t0 = op.getSrc0().getType();
  Type t1 = op.getSrc1().getType();
  Type td = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, t0, "src0")) ||
      failed(verifyTileBufCommon(op, t1, "src1")) ||
      failed(verifyTileBufCommon(op, td, "dst"))) {
    return failure();
  }
  if (op.getTmp() &&
      failed(verifyVecTileCommon(op, op.getTmp().getType(), "tmp"))) {
    return failure();
  }

  Type srcElem = getElemTy(t0);
  Type src1Elem = getElemTy(t1);
  Type dstElem = getElemTy(td);
  if (!srcElem || !src1Elem || !dstElem) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (srcElem != src1Elem || srcElem != dstElem) {
    op.emitOpError("expects src0, src1, and dst to have the same element type");
    return failure();
  }

  if (!isRowMajorTileBuf(t0) || !isRowMajorTileBuf(t1) ||
      !isRowMajorTileBuf(td)) {
    op.emitOpError(
        "expects src0, src1, and dst to use row-major layout");
    return failure();
  }
  return srcElem;
}

static LogicalResult verifyTSelA2A3(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
    ok = it.getWidth() == 16 || it.getWidth() == 32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsel src0, src1, and dst element type to be i16/i32/f16/bf16/f32");
  }
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemByteSize(getElemTy(tmpTy)) != 4) {
      return op.emitOpError("expects A2/A3 tsel tmp element type to be 4 bytes wide");
    }
    unsigned elemBits = getPTOStorageElemBitWidth(elem);
    if (elemBits != 16 && elemBits != 32) {
      return op.emitOpError("expects A2/A3 tsel data element type to be 16 or 32 bits");
    }
    uint64_t minBytes = elemBits == 16 ? 16 : 8;
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, minBytes))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyTSelA5(TSelOp op) {
  FailureOr<Type> srcElem = verifyTSelCommon(op);
  if (failed(srcElem)) {
    return failure();
  }
  Type elem = *srcElem;
  bool ok = elem.isF16() || elem.isBF16() || elem.isF32();
  if (auto it = dyn_cast<IntegerType>(elem)) {
    ok = it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32;
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsel src0, src1, and dst element type to be i8/i16/i32/f16/bf16/f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


static FailureOr<Type> verifyTSelSCommon(TSelSOp op) {
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tTmp = op.getTmp() ? op.getTmp().getType() : Type{};
  Type tDst = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, tMask, "mask")) ||
      failed(verifyTileBufCommon(op, tSrc, "src")) ||
      failed(verifyTileBufCommon(op, tDst, "dst"))) {
    return failure();
  }
  if (tTmp && failed(verifyTileBufCommon(op, tTmp, "tmp"))) {
    return failure();
  }
  Type eMask = getElemTy(tMask), eSrc = getElemTy(tSrc);
  Type eDst = getElemTy(tDst);
  if (!eMask || !eSrc || !eDst) {
    op.emitOpError("failed to get element type for operands");
    return failure();
  }
  if (eSrc != eDst) {
    return op.emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyTileBufSameValidShape(op, tSrc, tDst, "src", "dst"))) {
    return failure();
  }
  return eDst;
}

static LogicalResult verifyTSelSA2A3(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects src and dst to use row-major layout");
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError("expects A2/A3 tsels tmp to have the same element type as src and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects A2/A3 tsels tmp to use row-major layout");
    }
    auto srcShape = getShapeVec(tSrc);
    if (srcShape.size() != 2 || srcShape[1] == ShapedType::kDynamic) {
      return op.emitOpError(
          "expects A2/A3 tsels src shape to be static when tmp is provided");
    }
    auto elemBytes = getElemByteSize(elem);
    if (elemBytes == 0 ||
        failed(verifyTmpCapacityAtLeast(
            op, tmpTy, static_cast<uint64_t>(srcShape[1]) * elemBytes))) {
      return failure();
    }
  }
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
    ok = (it.getWidth() == 16 || it.getWidth() == 32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A2/A3 tsels src and dst element type to be i16, i32, f16, or f32");
  }
  return success();
}

static LogicalResult verifyTSelSA5(TSelSOp op) {
  FailureOr<Type> elemOr = verifyTSelSCommon(op);
  if (failed(elemOr)) {
    return failure();
  }
  Type tMask = op.getMask().getType();
  Type tSrc = op.getSrc().getType();
  Type tDst = op.getDst().getType();
  if (!isRowMajorTileBuf(tMask) || !isRowMajorTileBuf(tSrc) || !isRowMajorTileBuf(tDst)) {
    return op.emitOpError("expects mask, src, and dst to use row-major layout");
  }
  Type elem = *elemOr;
  bool ok = elem.isF16() || elem.isF32();
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(elem)) {
    ok = (it.getWidth() == 8 || it.getWidth() == 16 || it.getWidth() == 32);
  }
  if (!ok) {
    return op.emitOpError(
        "expects A5 tsels src and dst element type to be i8, i16, i32, f16, or f32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TSelSOp::verify() {
  // Constraints & Verification per PTO_IR_manual.md pto.tsels:
  // - src and dst same element type; A2A3: i16/i32/f16/f32; A5: i8/i16/i32/f16/f32
  // - src and dst row-major; src and dst same valid region
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTSelSA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTSelSA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects tshl src0 and src1 element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto verify = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyShiftLikeBinaryTileOpCommon(
        *this, getSrc0().getType(), getSrc1().getType(), getDst().getType());
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects tshr src0 and src1 element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verify, verify);
}


mlir::LogicalResult mlir::pto::TSort32Op::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  Type idxTy = getIdx().getType();
  if (failed(verifyVecTileCommon(*this, srcTy, "src")) ||
      failed(verifyVecTileCommon(*this, dstTy, "dst")) ||
      failed(verifyVecTileCommon(*this, idxTy, "idx"))) {
    return failure();
  }
  if (getTmp() &&
      failed(verifyVecTileCommon(*this, getTmp().getType(), "tmp"))) {
    return failure();
  }
  if (getTmp() && getTargetArch(getOperation()) != PTOArch::A5) {
    auto requiredBytes = getStaticByteSize(srcTy);
    if (!requiredBytes) {
      return emitOpError(
          "expects A2/A3 tsort32 src shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(*this, getTmp().getType(),
                                        *requiredBytes))) {
      return failure();
    }
  }

  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  if (!srcElem || !dstElem || srcElem != dstElem) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (!(srcElem.isF16() || srcElem.isF32())) {
    return emitOpError() << "expects src and dst element type to be f16 or f32";
  }

  auto idxElem = getElemTy(idxTy);
  auto idxInt = dyn_cast<IntegerType>(idxElem);
  if (!idxInt || idxInt.getWidth() != 32) {
    return emitOpError() << "expects idx element type to be i32/u32";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  Type srcTy = getSrc().getType();
  Type dstTy = getDst().getType();
  if (failed(verifyVecTileUnaryOp(*this, srcTy, dstTy, "src", "dst",
                                  /*allowBf16=*/false, /*allowInt8=*/false))) {
    return failure();
  }
  if (failed(verifyTileBufSameValidShape(*this, srcTy, dstTy, "src", "dst"))) {
    return failure();
  }

  auto srcElem = getElemTy(srcTy);
  if (!(mlir::isa<mlir::FloatType>(srcElem) || mlir::isa<mlir::Float16Type>(srcElem))) {
    return emitOpError() << "expects src and dst element type to be float or half";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TSubOp::verify() {
  return verifyArithmeticBinaryTileOpWithArchDispatch(
      getOperation(), getSrc0().getType(), getSrc1().getType(), getDst().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/false,
      "expects A2/A3 tsub element type to be i32/i16/f16/f32",
      "expects A5 tsub element type to be i32/i16/i8/f16/f32");
}


mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type src2Ty = getSrc2().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(src2Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, src2, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size() || getShapeVec(src2Ty).size() != d.size()) {
    return emitOpError() << "expects all tensors to have the same rank";
  }
  return mlir::success();
}


mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  return verifyArithmeticScalarTileOpWithArchDispatch(
      getOperation(), getSrc().getType(), getDst().getType(), getScalar().getType(),
      /*allowInt8OnA5=*/true, /*allowBf16OnA5=*/true,
      "expects A2/A3 tsubs element type to be i32/i16/f16/f32",
      "expects A5 tsubs element type to be i32/i16/i8/f16/bf16/f32",
      /*requireValidRowsEqualOnA2A3=*/true,
      /*requireValidRowsEqualOnA5=*/true);
}


mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  Type src0Ty = getSrc0().getType();
  Type src1Ty = getSrc1().getType();
  Type dstTy = getDst().getType();
  if (!isPTOShapedLike(src0Ty) || !isPTOShapedLike(src1Ty) || !isPTOShapedLike(dstTy)) {
    return emitOpError() << "expects PTO shaped-like src0, src1, and dst";
  }

  auto d = getShapeVec(dstTy);
  if (getShapeVec(src0Ty).size() != d.size() || getShapeVec(src1Ty).size() != d.size()) {
    return emitOpError() << "expects src0, src1, and dst to have the same rank";
  }
  return mlir::success();
}
static bool ttransUsesTmp(Type srcTy, Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  unsigned elemBytes = getPTOStorageElemByteSize(getElemTy(srcTy));
  if (srcShape.size() != 2 || dstShape.size() != 2 || elemBytes == 0 ||
      llvm::is_contained(srcShape, ShapedType::kDynamic) ||
      llvm::is_contained(dstShape, ShapedType::kDynamic)) {
    return true;
  }
  int64_t rowStride = elemBytes == 1 ? 32 : 16;
  int64_t elemPerBlock = 32 / elemBytes;
  int64_t srcStride = srcShape[1];
  int64_t dstStride = dstShape[1];
  return dstStride % rowStride == 0 && srcStride % elemPerBlock == 0 &&
         srcStride / elemPerBlock <= 255;
}

static bool ttransIsAllowedWidthType(Type ty, unsigned elemBytes) {
  if (elemBytes == 4) {
    return ty.isInteger(32) || ty.isF32();
  }
  if (elemBytes == 2) {
    return ty.isInteger(16) || ty.isF16() || ty.isBF16();
  }
  return ty.isInteger(8);
}

static LogicalResult verifyTTransElemWidth(TTransOp op, Type srcElem,
                                           unsigned &elemBytes) {
  elemBytes = getPTOStorageElemByteSize(srcElem);
  if (elemBytes == 0) {
    return op.emitOpError() << "failed to get transpose element size";
  }
  if (elemBytes != 1 && elemBytes != 2 && elemBytes != 4) {
    return op.emitOpError() << "expects transpose element size to be 1, 2, or 4 bytes";
  }
  if (!ttransIsAllowedWidthType(srcElem, elemBytes)) {
    return op.emitOpError() << "expects transpose element type to match the supported set for its width";
  }
  return success();
}

static LogicalResult verifyTTransAlignedMajor(TTransOp op, Type ty, StringRef name,
                                              unsigned elemBytes) {
  auto tb = mlir::dyn_cast<pto::TileBufType>(ty);
  if (!tb) {
    return success();
  }
  auto shape = getShapeVec(ty);
  if (shape.size() != 2) {
    return success();
  }
  bool rowMajor = tb.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::RowMajor);
  int64_t major = rowMajor ? shape[1] : shape[0];
  if (major != ShapedType::kDynamic && (major * static_cast<int64_t>(elemBytes)) % 32 != 0) {
    return op.emitOpError() << "expects " << name << " major dimension times element size to be 32-byte aligned on A5";
  }
  return success();
}

static LogicalResult verifyTTransA2A3(TTransOp op) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError() << "expects src and dst to have the same element type";
  }
  if (auto srcTb = dyn_cast<pto::TileBufType>(srcTy)) {
    if (srcTb.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor)) {
      return op.emitOpError() << "expects A2/A3 transpose src to use the row_major blayout";
    }
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  if (tmpTy) {
    uint64_t requiredBytes = 32;
    if (ttransUsesTmp(srcTy, dstTy)) {
      auto srcBytes = getStaticByteSize(srcTy);
      if (!srcBytes) {
        return op.emitOpError(
            "expects A2/A3 transpose src shape to be static when tmp is used");
      }
      requiredBytes = *srcBytes;
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, requiredBytes))) {
      return failure();
    }
  }
  return mlir::success();
}

static LogicalResult verifyTTransA5(TTransOp op) {
  Type srcTy = op.getSrc().getType();
  Type tmpTy = op.getTmp() ? op.getTmp().getType() : Type{};
  Type dstTy = op.getDst().getType();
  if (failed(verifyTileBufCommon(op, srcTy, "src")) ||
      failed(verifyTileBufCommon(op, dstTy, "dst"))) {
    return failure();
  }
  if (tmpTy && failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
    return failure();
  }
  Type srcElem = getElemTy(srcTy);
  Type tmpElem = tmpTy ? getElemTy(tmpTy) : srcElem;
  Type dstElem = getElemTy(dstTy);
  if (!srcElem || !tmpElem || !dstElem || srcElem != dstElem || srcElem != tmpElem) {
    return op.emitOpError() << "expects src, tmp, and dst to have the same element type";
  }
  unsigned elemBytes = 0;
  if (failed(verifyTTransElemWidth(op, srcElem, elemBytes))) {
    return failure();
  }
  if (tmpTy && failed(verifyTmpCapacityAtLeast(op, tmpTy, 32))) {
    return failure();
  }
  if (failed(verifyTTransAlignedMajor(op, srcTy, "src", elemBytes)) ||
      failed(verifyTTransAlignedMajor(op, dstTy, "dst", elemBytes))) {
    return failure();
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTTransA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTTransA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TXorOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand src0, src1, tmp, dst;
  Type src0Ty, src1Ty, tmpTy, dstTy;
  bool hasTmp = false;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src0) || parser.parseComma() ||
      parser.parseOperand(src1)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(src0Ty) || parser.parseComma() ||
      parser.parseType(src1Ty)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (parser.resolveOperand(src0, src0Ty, result.operands) ||
      parser.resolveOperand(src1, src1Ty, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TXorOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc0() << ", " << getSrc1();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc0().getType() << ", " << getSrc1().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

ParseResult mlir::pto::TXorSOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  OpAsmParser::UnresolvedOperand src, scalar, tmp, dst;
  Type srcTy, scalarTy, tmpTy, dstTy;
  bool hasTmp = false;
  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src) || parser.parseComma() ||
      parser.parseOperand(scalar)) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }
  if (parser.parseColonType(srcTy) || parser.parseComma() ||
      parser.parseType(scalarTy)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (parser.parseRParen() || parser.parseKeyword("outs") ||
      parser.parseLParen() || parser.parseOperand(dst) ||
      parser.parseColonType(dstTy) || parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  if (parser.resolveOperand(src, srcTy, result.operands) ||
      parser.resolveOperand(scalar, scalarTy, result.operands) ||
      (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) ||
      parser.resolveOperand(dst, dstTy, result.operands)) {
    return failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr({1, 1, hasTmp ? 1 : 0, 1}));
  return success();
}

void mlir::pto::TXorSOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc() << ", " << getScalar();
  if (getTmp()) {
    p << ", " << getTmp();
  }
  p << " : " << getSrc().getType() << ", " << getScalar().getType();
  if (getTmp()) {
    p << ", " << getTmp().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"operandSegmentSizes"});
}

static LogicalResult verifyTXorA2A3(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  Type elem = *elemOr;
  if (op.getTmp()) {
    Type tmpTy = op.getTmp().getType();
    if (failed(verifyTileBufCommon(op, tmpTy, "tmp"))) {
      return failure();
    }
    if (getElemTy(tmpTy) != elem) {
      return op.emitOpError(
          "expects tmp to have the same element type as src0, src1, and dst");
    }
    if (!isRowMajorTileBuf(tmpTy)) {
      return op.emitOpError("expects tmp to use row-major layout");
    }
    if (failed(verifyTileBufSameValidShape(
            op, tmpTy, op.getDst().getType(), "tmp", "dst"))) {
      return failure();
    }
    auto requiredBytes = getStaticByteSize(op.getDst().getType());
    if (!requiredBytes) {
      return op.emitOpError(
          "expects A2/A3 txor dst shape to be static when tmp is provided");
    }
    if (failed(verifyTmpCapacityAtLeast(op, tmpTy, *requiredBytes))) {
      return failure();
    }
  }
  auto it = mlir::dyn_cast<IntegerType>(elem);
  if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
              it.getWidth() != 32)) {
    return op.emitOpError(
        "expects A2/A3 txor src0, src1, tmp, and dst element type to be i8/i16/i32");
  }
  return success();
}

static LogicalResult verifyTXorA5(TXorOp op) {
  FailureOr<Type> elemOr = verifyMatchingRowMajorBinaryTileOpCommon(
      op.getOperation(), op.getSrc0().getType(), op.getSrc1().getType(),
      op.getDst().getType());
  if (failed(elemOr)) {
    return failure();
  }
  auto it = mlir::dyn_cast<IntegerType>(*elemOr);
  if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
              it.getWidth() != 32)) {
    return op.emitOpError(
        "expects A5 txor src0, src1, and dst element type to be i8/i16/i32");
  }
  return success();
}

mlir::LogicalResult mlir::pto::TXorOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult { return verifyTXorA2A3(*this); };
  auto verifyA5 = [&]() -> LogicalResult { return verifyTXorA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}


mlir::LogicalResult mlir::pto::TXorSOp::verify() {
  auto verifyCommon = [&]() -> FailureOr<Type> {
    return verifyDistinctRowMajorUnaryTileOpCommon(getOperation(), getSrc(),
                                                   getDst(), "src", "dst");
  };

  auto verifyA2A3 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    Type elem = *elemOr;
    if (getTmp()) {
      Type tmpTy = getTmp().getType();
      if (failed(verifyTileBufCommon(*this, tmpTy, "tmp"))) {
        return failure();
      }
      if (getElemTy(tmpTy) != elem) {
        return emitOpError(
            "expects tmp to have the same element type as src and dst");
      }
      if (!isRowMajorTileBuf(tmpTy)) {
        return emitOpError("expects tmp to use row-major layout");
      }
      auto requiredBytes = getStaticByteSize(getDst().getType());
      if (!requiredBytes) {
        return emitOpError(
            "expects A2/A3 txors dst shape to be static when tmp is provided");
      }
      if (failed(verifyTmpCapacityAtLeast(*this, tmpTy, *requiredBytes))) {
        return failure();
      }
    }
    auto it = mlir::dyn_cast<IntegerType>(elem);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16)) {
      return emitOpError(
          "expects A2/A3 txors src and dst element type to be i8/i16");
    }
    return success();
  };

  auto verifyA5 = [&]() -> LogicalResult {
    FailureOr<Type> elemOr = verifyCommon();
    if (failed(elemOr)) {
      return failure();
    }
    auto it = mlir::dyn_cast<IntegerType>(*elemOr);
    if (!it || (it.getWidth() != 8 && it.getWidth() != 16 &&
                it.getWidth() != 32)) {
      return emitOpError(
          "expects A5 txors src and dst element type to be i8/i16/i32");
    }
    return success();
  };

  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

ParseResult mlir::pto::TPrintOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand tmp;
  Type srcTy, tmpTy;
  bool hasTmp = false;
  NamedAttrList parsedAttrs;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(src)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(tmp)) {
      return failure();
    }
    hasTmp = true;
  }

  if (parser.parseColonType(srcTy)) {
    return failure();
  }
  if (hasTmp && (parser.parseComma() || parser.parseType(tmpTy))) {
    return failure();
  }
  if (parser.parseRParen()) {
    return failure();
  }
  if (failed(parsePTOInherentAttrs<TPrintOp>(
          parser, result, parsedAttrs, {"printFormat"}))) {
    return failure();
  }

  if (parser.resolveOperand(src, srcTy, result.operands)) {
    return failure();
  }
  if (hasTmp && parser.resolveOperand(tmp, tmpTy, result.operands)) {
    return failure();
  }

  return success();
}

void mlir::pto::TPrintOp::print(OpAsmPrinter &p) {
  p << " ins(" << getSrc();
  if (Value tmp = getTPrintTmpIfPresent(*this)) {
    p << ", " << tmp << " : " << getSrc().getType() << ", "
      << tmp.getType();
  } else {
    p << " : " << getSrc().getType();
  }
  p << ")";
  NamedAttrList attrs = getNonInherentAttrs(getOperation(), {"printFormat"});
  if (auto printFormatAttr =
          dyn_cast_or_null<pto::PrintFormatAttr>(getProperties().printFormat)) {
    attrs.append("printFormat", printFormatAttr);
  }
  p.printOptionalAttrDict(attrs.getAttrs());
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  Value tmp = getTPrintTmpIfPresent(*this);
  if (auto tb = mlir::dyn_cast<mlir::pto::TileBufType>(srcType)) {
    auto elem = tb.getElementType();
    if (!(elem.isF16() || elem.isF32() ||
          elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32))) {
      return emitOpError() << "expects printable tile element type";
}
    auto space = getPTOMemorySpaceEnum(srcType);
    if (!tmp) {
      if (!space || *space != pto::AddressSpace::VEC) {
        return emitOpError() << "expects printable tile_buf without tmp to be in vec address space";
}
      return success();
    }

    if (!space) {
      return emitOpError() << "expects printable tile_buf with tmp to use a supported address space";
}
    if (*space == pto::AddressSpace::MAT && isTargetArchA5(getOperation())) {
      return emitOpError() << "expects mat tile printing with tmp only on A2/A3 targets";
}
    if (*space != pto::AddressSpace::VEC && *space != pto::AddressSpace::MAT &&
        *space != pto::AddressSpace::ACC) {
      return emitOpError() << "expects printable tile_buf with tmp to be in vec/mat/acc address space";
}
    if (failed(verifyMGatherMScatterMemOperand(getOperation(), tmp, elem, "tmp"))) {
      return failure();
}
    return success();
  }
  if (tmp) {
    return emitOpError() << "expects tmp only when src is a tile_buf";
}
  if (mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType)) {
    return mlir::success();
}
  return emitOpError() << "expects tile_buf or partition_tensor_view for src";
}



[[maybe_unused]] static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor ----
  if (auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType())) {
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    if (!rhsTy) {
      return op->emitOpError("expects lhs and rhs to be ranked tensors");
    }

    if (lhsTy.getElementType() != rhsTy.getElementType()) {
      return op->emitOpError()
             << "expects lhs and rhs to have the same element type, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();
    }

    if (biasOpt) {
      auto biasTy = dyn_cast<RankedTensorType>(biasOpt.getType());
      if (!biasTy) {
        return op->emitOpError("expects bias to be a ranked tensor");
      }
      if (biasTy.getElementType() != lhsTy.getElementType()) {
        return op->emitOpError()
               << "expects bias to have the same element type as lhs and rhs, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
      }
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects dst to have the same element type as lhs and rhs, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();
    }

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType()) {
      return op->emitOpError()
             << "expects result to have the same element type as lhs and rhs, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();
    }

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile) {
    return op->emitOpError("expects lhs and rhs to be ranked tensors or !pto.tile");
  }

  if (lhsTile.getElementType() != rhsTile.getElementType()) {
    return op->emitOpError() << "expects lhs and rhs tiles to have the same element type, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();
  }

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2) {
    return op->emitOpError("expects lhs and rhs tiles to be 2D");
  }

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0]) {
    return op->emitOpError() << "expects lhs dim1 to equal rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];
  }

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile) {
      return op->emitOpError("expects bias to be !pto.tile when lhs and rhs are !pto.tile");
    }
    if (biasTile.getElementType() != lhsTile.getElementType()) {
      return op->emitOpError("expects bias to have the same element type as lhs and rhs");
    }
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType()) {
    return op->emitOpError() << "expects dst to have the same element type as lhs and rhs";
  }

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType()) {
    return op->emitOpError() << "expects result to have the same element type as lhs and rhs";
  }

  return success();
}

LogicalResult mlir::pto::TMatmulOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                      getDst().getType()))) {
      return failure();
    }
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    return verifyMatmulLike(*this, getLhs().getType(), getRhs().getType(),
                            getDst().getType());
  };
  auto verifyA5 = [&]() -> LogicalResult { return verifyA2A3(); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TMatmulAccOp::verify() {
  auto verifyA2A3 = [&]() -> LogicalResult {
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType()))) {
      return failure();
    }
    return success();
  };
  auto verifyA5 = [&]() -> LogicalResult {
    if (failed(verifyMatmulTypeTriple(*this, getElemTy(getLhs().getType()),
                                      getElemTy(getRhs().getType()),
                                      getElemTy(getDst().getType())))) {
      return failure();
    }
    if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
        failed(verifyMatTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                     getDst().getType(),
                                     /*allowLowPrecision=*/true))) {
      return failure();
    }
    return success();
  };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

LogicalResult mlir::pto::TGemvAccOp::verify() {
  if (failed(verifyAccTileCommon(*this, getAccIn().getType(), "acc_in")) ||
      failed(verifyGemvTileOperands(*this, getLhs().getType(), getRhs().getType(),
                                    getDst().getType()))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
[[maybe_unused]] static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2) {
    return mlir::Type();
  }

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile) {
    return mlir::Type();
  }

  Type elemTy = lhsTile.getElementType();

  if (operands.size() >= 3) {
    if (auto biasTile = dyn_cast<mlir::pto::TileType>(operands[2].getType())) {
      return mlir::pto::TileType::get(context, biasTile.getShape(), elemTy);
    }
  }

  auto lhsShape = lhsTile.getShape();
  auto rhsShape = rhsTile.getShape();
  if (lhsShape.size() >= 2 && rhsShape.size() >= 2) {
    int64_t M = lhsShape[0];
    int64_t N = rhsShape[1];
    llvm::SmallVector<int64_t, 2> outShape = {M, N};
    return mlir::pto::TileType::get(context, outShape, elemTy);
  }

  return mlir::Type();
}

[[maybe_unused]] static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2) {
    return RankedTensorType();
  }

  auto lhsTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto rhsTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!lhsTy || !rhsTy) {
    return RankedTensorType();
  }

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType())) {
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

[[maybe_unused]] static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty()) {
    return RankedTensorType();
  }
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType())) {
    return accRT;
  }
  return RankedTensorType();
}

namespace mlir {
namespace pto {
static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess()) {
    return failure();
  }

  if (parser.parseDimensionList(shape, allowDynamic)) {
    return failure();
  }

  if (parser.parseType(elementType)) {
    return failure();
  }

  if (parser.parseGreater()) {
    return failure();
  }

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << d;
}
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

static LogicalResult parseViewShapeElemAndLayout(
    AsmParser &parser, SmallVectorImpl<int64_t> &shape, Type &elementType,
    Attribute &layout, bool allowDynamic) {
  bool parseFailed =
      parser.parseLess() || parser.parseDimensionList(shape, allowDynamic) ||
      parser.parseType(elementType);
  if (parseFailed) {
    return failure();
  }
  if (succeeded(parser.parseOptionalComma())) {
    LayoutAttr layoutAttr;
    if (parser.parseAttribute(layoutAttr)) {
      return failure();
    }
    layout = layoutAttr;
  }
  return parser.parseGreater();
}

static void printViewShapeElemAndLayout(AsmPrinter &printer,
                                        ArrayRef<int64_t> shape,
                                        Type elementType, Attribute layout) {
  printer << "<";
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic) {
      printer << "?";
    } else {
      printer << dim;
    }
    printer << "x";
  }
  printer.printType(elementType);
  if (layout) {
    printer << ", ";
    printer.printAttribute(layout);
  }
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  Attribute layout;
  if (failed(parseViewShapeElemAndLayout(
          parser, shape, elemTy, layout, /*allowDynamic=*/true))) {
    return Type();
  }

  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy,
                                      layout);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printViewShapeElemAndLayout(printer, getShape(), getElementType(),
                              getLayout());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true))) {
    return Type();
  }
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- LocalArrayType ----
// Asm form: !pto.local_array<D1 x D2 x ... x Dk x T>
// Static shape only (no '?'). Element type must be a scalar; this is enforced
// by the type verifier below.
Type LocalArrayType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/false))) {
    return Type();
  }
  return LocalArrayType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elemTy);
}

void LocalArrayType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

LogicalResult LocalArrayType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty()) {
    return emitError() << "'!pto.local_array' requires at least one dimension";
  }
  for (auto [i, d] : llvm::enumerate(shape)) {
    if (d <= 0) {
      return emitError()
             << "'!pto.local_array' dimension " << i
             << " must be a positive static size, got " << d;
    }
  }
  if (!elementType.isIntOrFloat()) {
    return emitError()
           << "'!pto.local_array' element type must be a scalar integer or "
              "float, got "
           << elementType;
  }
  return success();
}

// ---- StructType ----
// Asm form: !pto.struct<T0, T1, ..., Tn-1>
// A field type must be "scalar-storable": an exactly-nameable scalar or a
// nested !pto.struct (see the two predicates below). The allowlist deliberately
// excludes the vec/cube types (tile_buf / tensor_view / partition view) and any
// other handle type, keeping the scalar struct world disjoint from the
// fractal/layout world.

// A struct field's scalar type must map onto a C++ scalar that the backend can
// name exactly: integers of width 8/16/32/64 and f16/bf16/f32/f64. Widths the
// backend has no spelling for (i1, i24, ...) and the packed low-precision
// vec/cube formats (f8/f4 variants) would otherwise be emitted as `float`,
// silently changing the field's width and semantics, so reject them here.
static bool isStructScalar(Type t) {
  if (llvm::isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(t)) {
    return true;
  }
  if (auto intTy = llvm::dyn_cast<IntegerType>(t)) {
    unsigned w = intTy.getWidth();
    return w == 8 || w == 16 || w == 32 || w == 64;
  }
  return false;
}

// A field is either such a scalar or a nested !pto.struct. !pto.local_array is
// deliberately NOT allowed: a field is reached with `emitc.member`, whose result
// must be an `!emitc.lvalue`, and `!emitc.lvalue` cannot wrap `!emitc.array`
// (the type an array field lowers to). There is no way to spell the access, so
// the restriction is enforced here rather than failing later in the backend.
static bool isStructStorable(Type t) {
  return isStructScalar(t) || llvm::isa<StructType>(t);
}

Type StructType::parse(AsmParser &parser) {
  SmallVector<Type> fields;
  if (parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            Type t;
            if (parser.parseType(t)) {
              return failure();
            }
            fields.push_back(t);
            return success();
          })) {
    return Type();
  }
  return StructType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), fields);
}

void StructType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::ArrayRef<Type> fields = getFieldTypes();
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer.printType(fields[i]);
  }
  printer << ">";
}

LogicalResult StructType::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    llvm::ArrayRef<Type> fieldTypes) {
  if (fieldTypes.empty()) {
    return emitError() << "'!pto.struct' requires at least one field";
  }
  for (auto [i, f] : llvm::enumerate(fieldTypes)) {
    if (!isStructStorable(f)) {
      return emitError()
             << "'!pto.struct' field " << i << " type " << f
             << " is not scalar-storable; only i8/i16/i32/i64 (signed, "
                "unsigned or signless), f16/bf16/f32/f64, or a nested "
                "!pto.struct are allowed (!pto.local_array cannot be a field "
                "because emitc.member cannot yield an array lvalue; tile_buf / "
                "tensor_view belong to the vec/cube world)";
    }
  }
  return success();
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    if (add.getKind() == AffineExprKind::Add) {
      flattenAddExpr(add.getLHS(), terms);
      flattenAddExpr(add.getRHS(), terms);
      return;
    }
  }
  terms.push_back(expr);
}

// Helper: 从 AffineMap 中提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  // 1. 初始化
  strides.assign(map.getNumDims(), 0);
  
  if (map.getNumResults() != 1) {
    return;
  }
  
  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = llvm::dyn_cast<AffineBinaryOpExpr>(term)) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(lhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(rhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
        
        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = llvm::dyn_cast<AffineDimExpr>(rhs)) {
          if (auto cst = llvm::dyn_cast<AffineConstantExpr>(lhs)) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = llvm::dyn_cast<AffineDimExpr>(term)) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// =============================================================================
// [Critical] Strict Alignment Protocol Helper
// =============================================================================
// This function is the SINGLE source of truth for building the AffineMap.
// Both the Parser and the Op Inference MUST use this exact function.
// It ensures that the order of AffineExpr addition is:
//   0 + (d0*str0 + d1*str1...) + (s0*str0 + s1*str1...)
// This guarantees bitwise-identical AffineMaps for verification.
static AffineMap buildStrictBitwiseAffineMap(MLIRContext *ctx, 
                                             ArrayRef<int64_t> strides, 
                                             bool isMultiDimSymbol) {
  unsigned rank = strides.size();
  
  // Step 1: Initialize with Constant(0)
  AffineExpr totalExpr = getAffineConstantExpr(0, ctx);

  // Step 2: Add Dimensions (d0*str0 + d1*str1...)
  // Strictly in order: 0, 1, 2...
  for (unsigned i = 0; i < rank; ++i) {
    auto dim = getAffineDimExpr(i, ctx);
    auto str = getAffineConstantExpr(strides[i], ctx);
    totalExpr = totalExpr + (dim * str);
  }

  // Step 3: Add Symbols (s0*str0 + s1*str1...)
  // Strictly in order: 0, 1, 2...
  if (isMultiDimSymbol) {
    for (unsigned i = 0; i < rank; ++i) {
      auto sym = getAffineSymbolExpr(i, ctx);
      auto str = getAffineConstantExpr(strides[i], ctx);
      totalExpr = totalExpr + (sym * str);
    }
  } 
  // (Optional: handle single dynamic offset case if needed, omitted for clarity)

  // numSymbols is rank if multi-dim (for offsets), else 0
  unsigned numSymbols = isMultiDimSymbol ? rank : 0;
  return AffineMap::get(rank, numSymbols, totalExpr);
}


// =============================================================================
// Parser Implementation
// =============================================================================

// Helper for parsing [64, 1]
static ParseResult parseStrideList(AsmParser &parser, SmallVectorImpl<int64_t> &strides) {
  if (parser.parseLSquare()) {
    return failure();
  }
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) {
      return failure();
    }
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) {
    return failure();
  }
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
[[maybe_unused]] static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) {
    return failure();
  }
  
  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) {
    return failure();
  }
  
  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) {
      return failure();
    }
    
    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) {
          return failure();
        }
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) {
        return failure();
      }
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }
  
  if (parser.parseGreater()) {
    return failure();
  }

  // 3. Validation
  if (isMultiDim && numSymbols != strides.size()) {
    return parser.emitError(parser.getCurrentLocation(), 
                            "Number of offset symbols must match rank");
  }

  // 4. [CALL SHARED BUILDER]
  // Delegate to the strict builder
  MLIRContext *ctx = parser.getContext();
  AffineMap map = buildStrictBitwiseAffineMap(ctx, strides, isMultiDim);
  
  layout = AffineMapAttr::get(map);
  return success();
}

// =============================================================================
// Printer Implementation
// =============================================================================

[[maybe_unused]] static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) {
    return;
  }
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) {
    return;
  }

  // 1. [核心修改] 反解 Strides
  SmallVector<int64_t> strides;
  decomposeStridedLayout(map, strides);

  printer << ", strided<[";
  // 2. 打印真实的 strides
  llvm::interleaveComma(strides, printer); 
  printer << "]";

  // Print Offset: [?, ?]
  unsigned numSyms = map.getNumSymbols();
  if (numSyms > 0) {
    printer << ", offset: [";
    for (unsigned i = 0; i < numSyms; ++i) {
      printer << "?";
      if (i < numSyms - 1) {
        printer << ", ";
      }
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subview 相关实现

// =============================================================================
// Op Interface Implementation: SubViewOp
// =============================================================================

ParseResult mlir::pto::SubViewOp::parse(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> offsets;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> valids;
  Type sourceTy;
  Type resultTy;
  bool hasExplicitResultTy = false;

  if (parser.parseOperand(source) || parser.parseLSquare() ||
      parser.parseOperandList(offsets) || parser.parseRSquare() ||
      parser.parseKeyword("sizes")) {
    return failure();
  }

  ArrayAttr sizesAttr;
  if (parser.parseAttribute(sizesAttr, "sizes", result.attributes)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("valid"))) {
    OpAsmParser::UnresolvedOperand vrow, vcol;
    if (parser.parseLSquare() || parser.parseOperand(vrow) || parser.parseComma() ||
        parser.parseOperand(vcol) || parser.parseRSquare()) {
      return failure();
    }
    valids.push_back(vrow);
    valids.push_back(vcol);
  }

  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceTy)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseType(resultTy)) {
      return failure();
    }
    hasExplicitResultTy = true;
  }

  if (parser.resolveOperand(source, sourceTy, result.operands)) {
    return failure();
  }

  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsets, indexTy, result.operands)) {
    return failure();
  }
  if (!valids.empty() &&
      parser.resolveOperands(valids, indexTy, result.operands)) {
    return failure();
  }

  int32_t hasValid = valids.empty() ? 0 : 1;
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(offsets.size()), hasValid, hasValid}));

  if (hasExplicitResultTy) {
    result.addTypes(resultTy);
    return success();
  }

  SmallVector<Type> inferredReturnTypes;
  DictionaryAttr attrs = result.attributes.getDictionary(parser.getContext());
  if (failed(SubViewOp::inferReturnTypes(
          parser.getContext(), std::nullopt, result.operands, attrs, nullptr,
          RegionRange(), inferredReturnTypes))) {
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to infer pto.subview result type");
  }
  result.addTypes(inferredReturnTypes);
  return success();
}

void mlir::pto::SubViewOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource() << "[";
  printer.printOperands(getOffsets());
  printer << "] sizes " << getSizes();
  if (getValidRow()) {
    printer << " valid [" << getValidRow() << ", " << getValidCol() << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes",
                                                 "sizes"});
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

// The inferred result type derives valid_shape from `sizes` (or the explicit
// valid operands). With the operand omitted the result type is authoritative for
// the valid extent (any static value, including the v=0 no-op-replay marker or a
// partial valid), so accept a static declared valid that differs from the
// size-inferred one here; SubViewOp::verify() enforces the precise per-path rule
// (operand clamping vs the [0, size] range). Only a dynamic declared valid that
// disagrees with the inferred extent is incompatible -- it needs an explicit
// operand to supply the runtime value. Every other difference (shape, element
// type, address space, config) is still rejected as the default check would.
bool SubViewOp::isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (auto [inferred, declared] : llvm::zip(lhs, rhs)) {
    if (inferred == declared) {
      continue;
    }
    auto inferredTb = dyn_cast<TileBufType>(inferred);
    auto declaredTb = dyn_cast<TileBufType>(declared);
    if (!inferredTb || !declaredTb) {
      return false;
    }
    if (inferredTb.getShape() != declaredTb.getShape() ||
        inferredTb.getElementType() != declaredTb.getElementType() ||
        inferredTb.getMemorySpace() != declaredTb.getMemorySpace() ||
        inferredTb.getConfigAttr() != declaredTb.getConfigAttr()) {
      return false;
    }
    auto inferredValid = inferredTb.getValidShape();
    auto declaredValid = declaredTb.getValidShape();
    if (inferredValid.size() != declaredValid.size()) {
      return false;
    }
    for (auto [inferredDim, declaredDim] : llvm::zip(inferredValid, declaredValid)) {
      // Any static declared valid extent is accepted in place of the inferred
      // one; only a dynamic declared valid that disagrees is incompatible.
      if (inferredDim != declaredDim && declaredDim == ShapedType::kDynamic) {
        return false;
      }
    }
  }
  return true;
}

LogicalResult SubViewOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // 1. 获取 Source Type
  if (operands.empty()) {
    return failure();
  }
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) {
    return failure();
  }

  // 2. 获取 subview 逻辑窗口（sizes）
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubViewOp::Properties *>();
    if (prop) {
      sizeAttr = prop->sizes;
    }
  }
  if (!sizeAttr && attributes) {
    sizeAttr = attributes.getAs<ArrayAttr>("sizes");
  }
  if (!sizeAttr) {
    return failure();
  }

  SmallVector<int64_t> subviewShape;
  for (auto attr : sizeAttr) {
    int64_t dim = llvm::cast<IntegerAttr>(attr).getInt();
    subviewShape.push_back(dim);
  }

  // Design: subview 的结果 tile 类型显式表达逻辑子窗口 shape（sizes）。
  ArrayRef<int64_t> parentShape = sourceType.getShape();
  if (subviewShape.size() != parentShape.size()) {
    return failure();
  }

  // Derive valid shape from explicit valid_row/valid_col when provided.
  // Otherwise default to subview shape (no parent valid-shape inheritance).
  SmallVector<int64_t> validShape;
  constexpr int64_t kDynamicValidDim = -1;
  int64_t rank = static_cast<int64_t>(subviewShape.size());
  Value explicitVRow;
  Value explicitVCol;

  // Robustly decode optional valid operands using AttrSizedOperandSegments:
  //   [source, offsets..., valid_row?, valid_col?]
  if (attributes) {
    if (auto segAttr =
            attributes.getAs<DenseI32ArrayAttr>("operandSegmentSizes")) {
      ArrayRef<int32_t> segs = segAttr.asArrayRef();
      if (segs.size() == 4) {
        int32_t srcSeg = segs[0];
        int32_t offSeg = segs[1];
        int32_t vRowSeg = segs[2];
        int32_t vColSeg = segs[3];
        if (srcSeg == 1 && offSeg >= 0 && (vRowSeg == 0 || vRowSeg == 1) &&
            (vColSeg == 0 || vColSeg == 1)) {
          size_t idx = static_cast<size_t>(srcSeg + offSeg);
          if (vRowSeg == 1 && idx < operands.size()) {
            explicitVRow = operands[idx++];
          }
          if (vColSeg == 1 && idx < operands.size()) {
            explicitVCol = operands[idx];
          }
        }
      }
    }
  }

  // Fallback for legacy callers that may not provide operandSegmentSizes.
  if (!explicitVRow && !explicitVCol && rank == 2) {
    size_t expectedWithoutValid = static_cast<size_t>(1 + rank);
    if (operands.size() >= expectedWithoutValid + 2) {
      explicitVRow = operands[expectedWithoutValid];
      explicitVCol = operands[expectedWithoutValid + 1];
    }
  }

  for (size_t i = 0, e = subviewShape.size(); i < e; ++i) {
    int64_t vdim = subviewShape[i];
    Value explicitV = (i == 0) ? explicitVRow : (i == 1 ? explicitVCol : Value());
    if (explicitV) {
      auto cst = getConstIndexValue(explicitV);
      vdim = cst ? std::min<int64_t>(*cst, subviewShape[i]) : kDynamicValidDim;
    }
    validShape.push_back(vdim);
  }

  // 3. 继承 Config (若为空使用默认)
  auto cfg = sourceType.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(context);
  }

  // 4. 构建 Result Type
  auto canonicalValidShape = canonicalizeTileBufValidShape(validShape);
  auto resultType = TileBufType::get(
      context, subviewShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), canonicalValidShape, cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

// =============================================================================
// SubViewOp verifier
// =============================================================================
static bool getConstIndex(Value v, int64_t &out) {
  if (auto cOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    out = cOp.value();
    return true;
  }
  if (auto cInt = v.getDefiningOp<arith::ConstantIntOp>()) {
    out = cInt.value();
    return true;
  }
  if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(cOp.getValue())) {
      out = ia.getInt();
      return true;
    }
  }
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>()) {
    return getConstIndex(castOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>()) {
    return getConstIndex(extOp.getIn(), out);
  }
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
    return getConstIndex(truncOp.getIn(), out);
  }
  return false;
}

static LogicalResult computeInnerShape(TileBufConfigAttr cfg, Type elemTy,
                                       int64_t &innerRows, int64_t &innerCols,
                                       bool &boxed, int32_t &bl, int32_t &sl) {
  auto readBLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<BLayoutAttr>(attr)) {
      out = static_cast<int32_t>(a.getValue());
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(a.getInt());
      return true;
    }
    return false;
  };
  auto readSLayoutI32 = [](Attribute attr, int32_t &out) -> bool {
    if (auto a = dyn_cast<SLayoutAttr>(attr)) {
      out = static_cast<int32_t>(a.getValue());
      return true;
    }
    if (auto a = dyn_cast<IntegerAttr>(attr)) {
      out = static_cast<int32_t>(a.getInt());
      return true;
    }
    return false;
  };
  bl = 0;
  sl = 0;
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) {
    fr = static_cast<int32_t>(attr.getInt());
  }

  boxed = (sl != 0);
  if (!boxed) {
    innerRows = 1;
    innerCols = 1;
    return success();
  }

  int64_t elemBytes = static_cast<int64_t>(getElemByteSize(elemTy));
  if (elemBytes <= 0) {
    return failure();
  }

  if (fr == 1024) {
    innerRows = 16;
    innerCols = 16;
    return success();
  }
  if (fr == 32) {
    innerRows = 16;
    innerCols = 2;
    return success();
  }
  if (fr == 512) {
    if (sl == 1) {
      innerRows = 16;
      innerCols = 32 / elemBytes;
      return success();
    }
    if (sl == 2) {
      innerRows = 32 / elemBytes;
      innerCols = 16;
      return success();
    }
  }
  return failure();
}

struct SubViewInfo {
  int64_t sizeR = 0, sizeC = 0;
  int64_t offR = 0, offC = 0;
  bool offRConst = false, offCConst = false;
};

static LogicalResult verifySubViewSizesAndOffsets(SubViewOp op,
                                                  SubViewInfo &info) {
  auto sizesAttr = op.getSizes();
  if (!sizesAttr || sizesAttr.size() != 2) {
    return op.emitOpError("subview expects 2D sizes");
  }
  info.sizeR = cast<IntegerAttr>(sizesAttr[0]).getInt();
  info.sizeC = cast<IntegerAttr>(sizesAttr[1]).getInt();
  if (info.sizeR <= 0 || info.sizeC <= 0) {
    return op.emitOpError("subview sizes must be positive");
  }
  if (op.getOffsets().size() != 2) {
    return op.emitOpError("subview expects 2D offsets");
  }

  info.offRConst = getConstIndex(op.getOffsets()[0], info.offR);
  info.offCConst = getConstIndex(op.getOffsets()[1], info.offC);
  if (info.offRConst && info.offR < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  if (info.offCConst && info.offC < 0) {
    return op.emitOpError("subview offsets must be non-negative");
  }
  return success();
}

static LogicalResult verifySubViewValidBounds(SubViewOp op, int64_t sizeR,
                                              int64_t sizeC) {
  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "subview expects valid_row and valid_col to be both present or both absent");
  }

  if (hasValidRow) {
    int64_t vRow = 0, vCol = 0;
    if (getConstIndex(op.getValidRow(), vRow)) {
      if (vRow < 0) {
        return op.emitOpError("valid_row must be non-negative when constant");
      }
      if (vRow > sizeR) {
        return op.emitOpError("valid_row must be <= subview row size");
      }
    }
    if (getConstIndex(op.getValidCol(), vCol)) {
      if (vCol < 0) {
        return op.emitOpError("valid_col must be non-negative when constant");
      }
      if (vCol > sizeC) {
        return op.emitOpError("valid_col must be <= subview col size");
      }
    }
  }
  return success();
}

static LogicalResult verifySubViewShapeAndConfig(SubViewOp op, TileBufType srcTy,
                                                 TileBufType dstTy, int64_t sizeR,
                                                 int64_t sizeC) {
  auto dstShape = dstTy.getShape();
  if (dstShape.size() != 2) {
    return op.emitOpError("expects result to be rank-2");
  }
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2) {
    return op.emitOpError("expects source to be rank-2");
  }
  if (dstShape[0] != sizeR || dstShape[1] != sizeC) {
    return op.emitOpError("expects result shape to match subview sizes");
  }

  if (dstTy.getElementType() != srcTy.getElementType()) {
    return op.emitOpError("expects result element type to match source");
  }
  if (dstTy.getMemorySpace() != srcTy.getMemorySpace()) {
    return op.emitOpError("expects result address space to match source");
  }
  auto srcCfg = srcTy.getConfigAttr();
  if (!srcCfg) {
    srcCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  auto dstCfg = dstTy.getConfigAttr();
  if (!dstCfg) {
    dstCfg = TileBufConfigAttr::getDefault(op.getContext());
  }
  if (dstCfg != srcCfg) {
    return op.emitOpError("expects result tile config to match source");
  }
  return success();
}

static LogicalResult verifySubViewValidShape(SubViewOp op, TileBufType dstTy,
                                             int64_t sizeR, int64_t sizeC) {
  // Design choice: when valid[...] is omitted, infer result valid_shape from
  // subview sizes directly. We intentionally do not constrain it by source
  // valid_shape to allow user-controlled subview semantics.

  auto expectedValidDim = [&](Value explicitValid, int64_t defaultSize) {
    if (!explicitValid) {
      return defaultSize;
    }
    int64_t c = 0;
    if (getConstIndex(explicitValid, c)) {
      return std::min<int64_t>(c, defaultSize);
    }
    return ShapedType::kDynamic;
  };
  int64_t expectedVRow = expectedValidDim(op.getValidRow(), sizeR);
  int64_t expectedVCol = expectedValidDim(op.getValidCol(), sizeC);
  auto dstValid = dstTy.getValidShape();
  if (dstValid.size() != 2) {
    return op.emitOpError("expects result to have rank-2 valid_shape");
  }
  // With the valid operand omitted, the result type is authoritative for the
  // valid extent: accept any static value in [0, size] (this subsumes both the
  // full-size default and the v=0 no-op-replay empty marker). A dynamic result valid still
  // requires an explicit operand to supply the runtime extent, so it stays
  // rejected on this path.
  bool rowInferred = !op.getValidRow() && dstValid[0] != ShapedType::kDynamic &&
                     dstValid[0] >= 0 && dstValid[0] <= sizeR;
  bool colInferred = !op.getValidCol() && dstValid[1] != ShapedType::kDynamic &&
                     dstValid[1] >= 0 && dstValid[1] <= sizeC;
  if (dstValid[0] != expectedVRow && !rowInferred) {
    return op.emitOpError("expects result valid_shape[0] to match inferred/explicit valid_row");
  }
  if (dstValid[1] != expectedVCol && !colInferred) {
    return op.emitOpError("expects result valid_shape[1] to match inferred/explicit valid_col");
  }
  return success();
}

static LogicalResult verifySubViewBoxed(SubViewOp op, TileBufType srcTy,
                                        const SubViewInfo &info) {
  auto cfg = srcTy.getConfigAttr();
  if (!cfg) {
    cfg = TileBufConfigAttr::getDefault(op.getContext());
  }

  int64_t innerRows = 1, innerCols = 1;
  bool boxed = false;
  int32_t bl = 0, sl = 0;
  if (failed(computeInnerShape(cfg, srcTy.getElementType(), innerRows, innerCols,
                               boxed, bl, sl))) {
    return op.emitOpError("unsupported tile layout for subview");
  }

  if (!boxed) {
    return success();
  }

  // Boxed layout: require static 2D sizes with inner alignment. Offsets may be
  // dynamic, but static offsets must be aligned.
  if (info.sizeR % innerRows != 0 || info.sizeC % innerCols != 0) {
    return op.emitOpError("boxed layout subview sizes must be multiples of inner shape");
  }

  if (info.offRConst) {
    if (info.offR % innerRows != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }
  if (info.offCConst) {
    if (info.offC % innerCols != 0) {
      return op.emitOpError("boxed layout subview offsets must be multiples of inner shape");
    }
  }

  (void)bl;
  auto srcShape = srcTy.getShape();
  if (srcShape.size() != 2 ||
      srcShape[0] == ShapedType::kDynamic ||
      srcShape[1] == ShapedType::kDynamic) {
    return op.emitOpError("boxed layout subview requires static source shape");
  }

  return success();
}

mlir::LogicalResult mlir::pto::SubViewOp::verify() {
  auto srcTy = llvm::dyn_cast<TileBufType>(getSource().getType());
  auto dstTy = llvm::dyn_cast<TileBufType>(getResult().getType());
  if (!srcTy || !dstTy) {
    return emitOpError("expects tile_buf src and tile_buf result");
  }
  if (srcTy.getRank() != 2 || dstTy.getRank() != 2) {
    return emitOpError("expects rank-2 tilebuf for src/dst");
  }

  SubViewInfo info;
  if (failed(verifySubViewSizesAndOffsets(*this, info))) {
    return failure();
  }
  if (failed(verifySubViewValidBounds(*this, info.sizeR, info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewShapeAndConfig(*this, srcTy, dstTy, info.sizeR,
                                         info.sizeC))) {
    return failure();
  }
  if (failed(verifySubViewValidShape(*this, dstTy, info.sizeR, info.sizeC))) {
    return failure();
  }
  return verifySubViewBoxed(*this, srcTy, info);
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
 
// =============================================================================
// Helper Functions
// =============================================================================
 
// =============================================================================
// Side Effects Implementation
// =============================================================================
 
// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value
 
// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand) {
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
  }
}
 
// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result) {
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
  }
}

// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

void TPrefetchOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TAbsOp ===
// Read: src, Write: dst
void TAbsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TStoreOp ===
// Read: src, Write: dst (GM)
void TStoreOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty()) {
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (classifyTMovForm(getFp()) == TMovForm::XToZz) {
    const MxGroupAxis axis = getGrpAxisAttr()
                                 ? getGrpAxisAttr().getValue()
                                 : MxGroupAxis::Axis1;
    if (axis == MxGroupAxis::Axis1) {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
      addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
    } else {
      addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    }
    addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
    auto fp = getFpMutable();
    if (axis == MxGroupAxis::Axis1 && !fp.empty()) {
      addEffect(effects, &fp[0], MemoryEffects::Read::get());
      addEffect(effects, &fp[0], MemoryEffects::Write::get());
    }
    return;
  }
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  auto preQuantRange = getPreQuantScalarMutable();
  if (!preQuantRange.empty()) {
    addEffect(effects, &*preQuantRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

#define PTO_ADD_READ(operand) addEffect(effects, &(operand), MemoryEffects::Read::get())
#define PTO_ADD_WRITE(operand) addEffect(effects, &(operand), MemoryEffects::Write::get())

#define PTO_DEFINE_UNARY_EFFECTS(OpClass, srcOperand, dstOperand)                    \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(srcOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_BINARY_EFFECTS(OpClass, lhsOperand, rhsOperand, dstOperand)       \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(lhsOperand);                                                       \
    PTO_ADD_READ(rhsOperand);                                                       \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_TERNARY_EFFECTS(OpClass, op0, op1, op2, dstOperand)               \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

#define PTO_DEFINE_QUATERNARY_EFFECTS(OpClass, op0, op1, op2, op3, dstOperand)      \
  void OpClass::getEffects(                                                         \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) { \
    PTO_ADD_READ(op0);                                                              \
    PTO_ADD_READ(op1);                                                              \
    PTO_ADD_READ(op2);                                                              \
    PTO_ADD_READ(op3);                                                              \
    PTO_ADD_WRITE(dstOperand);                                                      \
  }

void LoadScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getPtrMutable());
}

void StoreScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getPtrMutable());
}

// === Tile/Device ops added for InsertSync ===

// MGATHER: Read(mem, idx) -> Write(dst)
void MGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMemMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  // GM -> L1 Elem mode stages the gathered elements into the GM scratch buffer
  // before the bulk copy: the op clobbers scratch, so model it as a write.
  auto scratchRange = getScratchMutable();
  if (!scratchRange.empty()) {
    addEffect(effects, &*scratchRange.begin(), MemoryEffects::Write::get());
  }
}

// MSCATTER: Read(src, idx) -> Write(mem)
void MScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getMemMutable());
}

// TGETVAL: Read(src) -> scalar result
void TGetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
}

void THistogramOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  PTO_ADD_WRITE(getDstMutable());
}

void TGetScaleAddrOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_WRITE(getDstMutable());
}

// TSETVAL: Write(dst) (single element update)
void TSetValOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// SET_VALIDSHAPE: update runtime valid row/col metadata on source tile in-place.
void SetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getSourceMutable());
}

// GET_VALIDSHAPE: read runtime valid row/col metadata from source tile.
void GetValidShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSourceMutable());
}

// Elementwise + reductions: mostly PIPE_V tilebuf ops
PTO_DEFINE_BINARY_EFFECTS(TAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddReluOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TAddCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAddSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TAddSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TAxpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TAndOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TConcatOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_QUATERNARY_EFFECTS(TConcatidxOp, getSrc0Mutable(), getSrc1Mutable(), getSrc0IdxMutable(), getSrc1IdxMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TAndSOp, getSrcMutable(), getDstMutable())

// TCI: Write(dst) (generates sequence)
void TCIOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TTRI: Write(dst) (generates triangular mask)
void TTriOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TCmpOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TCmpSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_UNARY_EFFECTS(TColExpandOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandExpdifOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TColExpandMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMaxOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColMinOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TColProdOp, getSrcMutable(), getDstMutable())

void TColArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TColArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TColSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TCvtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
void TRandomOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TDivOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TDIVS has custom assembly format; conservatively treat first 2 operands as reads.
void TDivSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getScalarMutable());
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TExpOp, getSrcMutable(), getDstMutable())

// TEXPANDS: Write(dst) (broadcast scalar)
void TExpandsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_WRITE(getDstMutable());
}

// TEXTRACT: Read(src) -> Write(dst)
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// TINSERT: Read(src) -> Write(dst)
void TInsertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  auto fpRange = getFpMutable();
  if (!fpRange.empty()) {
    addEffect(effects, &*fpRange.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

PTO_DEFINE_UNARY_EFFECTS(TFillPadOp, getSrcMutable(), getDstMutable())

void TGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (auto cdst = getCdstMutable(); !cdst.empty()) {
    PTO_ADD_WRITE(cdst[0]);
  }
  if (auto indices = getIndicesMutable(); !indices.empty()) {
    PTO_ADD_READ(indices[0]);
  }
  if (auto tmp = getTmpMutable();
      !tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TGatherBOp, getSrcMutable(), getOffsetsMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLogOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TLReluOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMaxSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMinSOp, getSrcMutable(), getDstMutable())

void TMrgSortOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &opnd : getSrcsMutable()) {
    PTO_ADD_READ(opnd);
  }
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  for (auto &opnd : getDstsMutable()) {
    PTO_ADD_WRITE(opnd);
  }
  auto executed = getExcutedMutable();
  if (!executed.empty()) {
    PTO_ADD_WRITE(executed[0]);
  }
}

PTO_DEFINE_BINARY_EFFECTS(TMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TMulSOp, getSrc0Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNegOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TNotOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TOrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TOrSOp, getSrcMutable(), getDstMutable())

PTO_DEFINE_BINARY_EFFECTS(TPartAddOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMaxOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TPartMinOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
void TInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_WRITE(getDst0Mutable());
  PTO_ADD_WRITE(getDst1Mutable());
}
void TDeInterleaveOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  for (auto &operand : getSrcsMutable()) {
    PTO_ADD_READ(operand);
  }
  for (auto &operand : getDstsMutable()) {
    PTO_ADD_WRITE(operand);
  }
}
void TPartArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
void TPartArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  PTO_ADD_READ(getSrc0IdxMutable());
  PTO_ADD_READ(getSrc1IdxMutable());
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getDstIdxMutable());
}
PTO_DEFINE_BINARY_EFFECTS(TPartMulOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
// TPRELU: Read(src0, src1) -> Write(tmp, dst)
void TPReluOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 pto-isa TPRELU implementation does not consume tmp; modeling tmp as a
  // write-only scratch on A5 incorrectly inflates local-memory planning and
  // can trigger false vec-overflow diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getFpMutable());
  auto offsetRange = getOffsetMutable();
  if (!offsetRange.empty()) {
    PTO_ADD_READ(offsetRange[0]);
  }
  auto tmpRange = getTmpMutable();
  if (!tmpRange.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmpRange[0]);
    PTO_ADD_WRITE(tmpRange[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TQuantMxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  Type srcTy = getSrc().getType();
  auto valid = getValidShapeVec(srcTy);
  auto physical = getShapeVec(srcTy);
  Type elem = getElemTy(srcTy);
  if ((elem.isF16() || elem.isBF16()) && valid.size() == 2 && physical.size() == 2 &&
      valid[1] < physical[1]) {
    addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
    addEffect(effects, &getSrcMutable(), MemoryEffects::Write::get());
  } else {
    addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  }
  PTO_ADD_WRITE(getDstMutable());
  PTO_ADD_WRITE(getExpMutable());
  PTO_ADD_WRITE(getMaxMutable());
  PTO_ADD_WRITE(getScalingMutable());
  auto expZzRange = getExpZzMutable();
  if (!expZzRange.empty()) {
    PTO_ADD_WRITE(expZzRange[0]);
  }
}
PTO_DEFINE_TERNARY_EFFECTS(TDequantOp, getSrcMutable(), getScaleMutable(),
                           getOffsetMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TRecipOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TReluOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TFModOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TFModSOp, getSrcMutable(), getDstMutable())
void TRemOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRemSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPowOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getBaseMutable());
  PTO_ADD_READ(getExpMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPowSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
PTO_DEFINE_UNARY_EFFECTS(TRowExpandOp, getSrcMutable(), getDstMutable())

void TRowExpandDivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandSubOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandExpdifOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowExpandMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Row reductions use tmp scratch tile.
void TRowMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMaxOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMAX; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowArgMinOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TROWARGMIN; modeling tmp as a
  // scratch write inflates local-memory planning and can trigger false
  // vec-overflow diagnostics, mirroring the fixed A5 TPRELU issue.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowSumOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TRowProdOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}
void TRsqrtOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (getIndexes()) {
    auto idx = getIndexesMutable();
    if (!idx.empty()) {
      PTO_ADD_READ(idx[0]);
    }
  }
  PTO_ADD_WRITE(getDstMutable());
}

// Select: Read(mask, src0, src1) -> Write(tmp on A2/A3, dst)
void TSelOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TSEL; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TSELS: Read(mask, src) -> Write(tmp on A2/A3, dst)
void TSelSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getMaskMutable());
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TSELS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_BINARY_EFFECTS(TShlOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TShrOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShlSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TShrSOp, getSrcMutable(), getDstMutable())

// TSORT32: Read(src, idx) -> Write(dst [, tmp])
void TSort32Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  PTO_ADD_READ(getIdxMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty()) {
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

PTO_DEFINE_UNARY_EFFECTS(TSqrtOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())
PTO_DEFINE_TERNARY_EFFECTS(TSubCOp, getSrc0Mutable(), getSrc1Mutable(), getSrc2Mutable(), getDstMutable())
PTO_DEFINE_UNARY_EFFECTS(TSubSOp, getSrcMutable(), getDstMutable())
PTO_DEFINE_BINARY_EFFECTS(TSubSCOp, getSrc0Mutable(), getSrc1Mutable(), getDstMutable())

// TXORS: Read(src) -> Write(tmp on A2/A3, dst)
void TXorSOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  // A5 lowering does not consume tmp for TXORS; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TXOR: Read(src0, src1) -> Write(tmp on A2/A3, dst)
void TXorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrc0Mutable());
  PTO_ADD_READ(getSrc1Mutable());
  // A5 lowering does not consume tmp for TXOR; modeling tmp as a scratch
  // write inflates local-memory planning and can trigger false vec-overflow
  // diagnostics.
  auto tmp = getTmpMutable();
  if (!tmp.empty() && getTargetArch(getOperation()) != PTOArch::A5) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

// TTRANS: Read(src) -> Write(tmp, dst)
void TTransOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  auto tmp = getTmpMutable();
  if (!tmp.empty() && ttransUsesTmp(getSrc().getType(), getDst().getType())) {
    PTO_ADD_READ(tmp[0]);
    PTO_ADD_WRITE(tmp[0]);
  }
  PTO_ADD_WRITE(getDstMutable());
}

void TPrintOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  PTO_ADD_READ(getSrcMutable());
  if (!getTmpMutable().empty()) {
    PTO_ADD_WRITE(getTmpMutable()[0]);
  }
  PTO_ADD_WRITE(getSrcMutable());
}

#undef PTO_DEFINE_TERNARY_EFFECTS
#undef PTO_DEFINE_BINARY_EFFECTS
#undef PTO_DEFINE_UNARY_EFFECTS
#undef PTO_ADD_WRITE
#undef PTO_ADD_READ

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasOp ===
// Read: a, b, bias, Write: dst
void TMatmulBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvOp ===
// Read: lhs, rhs, Write: dst
void TGemvOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvAccOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TGemvAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAccInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvBiasOp ===
// Read: a, b, bias, Write: dst
void TGemvBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxOp ===
// Read: a, a_scale, b, b_scale, Write: dst
void TGemvMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxAccOp ===
// Read: c_in, a, a_scale, b, b_scale, Write: dst
void TGemvMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TGemvMxBiasOp ===
// Read: a, a_scale, b, b_scale, bias, Write: dst
void TGemvMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
void TMatmulMxOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulAccMxOp ===
// Read: acc_in, lhs, rhs, Write: dst
void TMatmulMxAccOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getCInMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulBiasMxOp ===
// Read: a, b, bias, Write: dst
void TMatmulMxBiasOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getAMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAScaleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getBScaleMutable(), MemoryEffects::Read::get());
  // 这里的 bias 是必选的 AnyType:$bias，所以是 Singleton
  addEffect(effects, &getBiasMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

static bool isInsideSectionCube(Operation *op) {
  return op->getParentOfType<pto::SectionCubeOp>() != nullptr;
}

static bool isInsideSectionVector(Operation *op) {
  return op->getParentOfType<pto::SectionVectorOp>() != nullptr;
}

static bool isInsideTileOpHelper(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  return funcOp && funcOp->hasAttr("pto.tileop.helper");
}

static std::optional<FunctionKernelKind>
getEnclosingFunctionKernelKind(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return std::nullopt;
  }

  auto kernelKindAttr =
      funcOp->getAttrOfType<FunctionKernelKindAttr>(
          FunctionKernelKindAttr::name);
  if (!kernelKindAttr) {
    return std::nullopt;
  }

  return kernelKindAttr.getKernelKind();
}

static bool isInsideSectionOrAttributedKernel(Operation *op) {
  return isInsideSectionCube(op) || isInsideSectionVector(op) ||
         isInsideTileOpHelper(op) || getEnclosingFunctionKernelKind(op).has_value();
}

static LogicalResult verifySplitAttr(Operation *op, int64_t split) {
  if (split < 0 || split > 4) {
    return op->emitOpError("expects 'split' to be 0, 1, 2, 3, or 4");
  }
  return success();
}

static bool isOddSplit(int64_t split) {
  return split == 3 || split == 4;
}

static bool isInsideCubeKernelOrSection(Operation *op) {
  if (isInsideSectionCube(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Cube;
}

static bool isInsideVectorKernelOrSection(Operation *op) {
  if (isInsideSectionVector(op)) {
    return true;
  }
  auto kernelKind = getEnclosingFunctionKernelKind(op);
  return kernelKind && *kernelKind == FunctionKernelKind::Vector;
}

static LogicalResult verifyFrontendKernelKind(Operation *op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (isInsideTileOpHelper(op)) {
    return success();
  }
  if (isInsideSectionCube(op)) {
    if (expected == FunctionKernelKind::Cube) {
      return success();
    }
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  if (isInsideSectionVector(op)) {
    if (expected == FunctionKernelKind::Vector) {
      return success();
    }
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }

  std::optional<FunctionKernelKind> kernelKind =
      getEnclosingFunctionKernelKind(op);
  if (!kernelKind || *kernelKind != expected) {
    return op->emitOpError("must be inside a ")
           << kernelName << " kernel function or section";
  }
  return success();
}

static ParseResult parseFrontendInitializePipeOp(OpAsmParser &parser,
                                                 OperationState &result) {
  NamedAttrList attrs;
  bool sawId = false;
  bool sawDirMask = false;
  bool sawSlotSize = false;
  bool sawSlotNum = false;
  bool sawLocalSlotNum = false;
  bool sawNoSplit = false;
  bool sawAccPushEpilogue = false;

  if (parser.parseLBrace()) {
    return failure();
  }

  while (failed(parser.parseOptionalRBrace())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) {
      return failure();
    }

    if (keyword == "id") {
      if (sawId) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'id' clause");
      }
      IntegerAttr idAttr;
      if (parser.parseAttribute(idAttr, parser.getBuilder().getI32Type(), "id",
                                attrs)) {
        return failure();
      }
      sawId = true;
    } else if (keyword == "dir_mask") {
      if (sawDirMask) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'dir_mask' clause");
      }
      IntegerAttr dirMaskAttr;
      if (parser.parseAttribute(dirMaskAttr, parser.getBuilder().getI8Type(),
                                "dir_mask", attrs)) {
        return failure();
      }
      sawDirMask = true;
    } else if (keyword == "slot_size") {
      if (sawSlotSize) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_size' clause");
      }
      IntegerAttr slotSizeAttr;
      if (parser.parseAttribute(slotSizeAttr, parser.getBuilder().getI32Type(),
                                "slot_size", attrs)) {
        return failure();
      }
      sawSlotSize = true;
    } else if (keyword == "slot_num") {
      if (sawSlotNum) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'slot_num' clause");
      }
      IntegerAttr slotNumAttr;
      if (parser.parseAttribute(slotNumAttr, parser.getBuilder().getI32Type(),
                                "slot_num", attrs)) {
        return failure();
      }
      sawSlotNum = true;
    } else if (keyword == "local_slot_num") {
      if (sawLocalSlotNum) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'local_slot_num' clause");
      }
      IntegerAttr localSlotNumAttr;
      if (parser.parseAttribute(localSlotNumAttr, parser.getBuilder().getI32Type(),
                                "local_slot_num", attrs)) {
        return failure();
      }
      sawLocalSlotNum = true;
    } else if (keyword == "nosplit") {
      if (sawNoSplit) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'nosplit' clause");
      }
      BoolAttr noSplitAttr;
      if (parser.parseAttribute(noSplitAttr, "nosplit", attrs)) {
        return failure();
      }
      sawNoSplit = true;
    } else if (keyword == "acc_push_epilogue") {
      if (sawAccPushEpilogue) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'acc_push_epilogue' clause");
      }
      AccPushEpilogueAttr accPushEpilogueAttr;
      if (parser.parseAttribute(accPushEpilogueAttr, "acc_push_epilogue",
                                attrs)) {
        return failure();
      }
      sawAccPushEpilogue = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected keyword '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRBrace())) {
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
  }

  if (!sawDirMask) {
    return parser.emitError(parser.getNameLoc(), "expected 'dir_mask' clause");
  }
  if (!sawSlotSize) {
    return parser.emitError(parser.getNameLoc(), "expected 'slot_size' clause");
  }
  if (!sawId) {
    attrs.set("id", parser.getBuilder().getI32IntegerAttr(0));
  }

  OpAsmParser::UnresolvedOperand gmSlotBuffer;
  OpAsmParser::UnresolvedOperand gmSlotTensor;
  OpAsmParser::UnresolvedOperand c2vConsumerBuf;
  OpAsmParser::UnresolvedOperand v2cConsumerBuf;
  Type gmSlotBufferTy;
  Type gmSlotTensorTy;
  Type c2vConsumerBufTy;
  Type v2cConsumerBufTy;
  bool hasGmSlotBuffer = false;
  bool hasGmSlotTensor = false;
  bool hasC2vConsumerBuf = false;
  bool hasV2cConsumerBuf = false;

  if (parser.parseLParen()) {
    return failure();
  }
  while (failed(parser.parseOptionalRParen())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || parser.parseEqual()) {
      return failure();
    }

    if (keyword == "gm_slot_buffer") {
      if (hasGmSlotBuffer) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_buffer' operand");
      }
      if (parser.parseOperand(gmSlotBuffer) ||
          parser.parseColonType(gmSlotBufferTy)) {
        return failure();
      }
      hasGmSlotBuffer = true;
    } else if (keyword == "gm_slot_tensor") {
      if (hasGmSlotTensor) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'gm_slot_tensor' operand");
      }
      if (parser.parseOperand(gmSlotTensor) ||
          parser.parseColonType(gmSlotTensorTy)) {
        return failure();
      }
      hasGmSlotTensor = true;
    } else if (keyword == "c2v_consumer_buf") {
      if (hasC2vConsumerBuf) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'c2v_consumer_buf' operand");
      }
      if (parser.parseOperand(c2vConsumerBuf) ||
          parser.parseColonType(c2vConsumerBufTy)) {
        return failure();
      }
      hasC2vConsumerBuf = true;
    } else if (keyword == "v2c_consumer_buf") {
      if (hasV2cConsumerBuf) {
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate 'v2c_consumer_buf' operand");
      }
      if (parser.parseOperand(v2cConsumerBuf) ||
          parser.parseColonType(v2cConsumerBufTy)) {
        return failure();
      }
      hasV2cConsumerBuf = true;
    } else {
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected initialize_pipe operand '" << keyword << "'";
    }

    if (succeeded(parser.parseOptionalRParen())) {
      break;
    }
    if (parser.parseComma()) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  result.addAttributes(attrs);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {hasGmSlotBuffer ? 1 : 0, hasGmSlotTensor ? 1 : 0,
                           hasC2vConsumerBuf ? 1 : 0,
                           hasV2cConsumerBuf ? 1 : 0}));
  if (hasGmSlotBuffer &&
      parser.resolveOperand(gmSlotBuffer, gmSlotBufferTy, result.operands)) {
    return failure();
  }
  if (hasGmSlotTensor &&
      parser.resolveOperand(gmSlotTensor, gmSlotTensorTy, result.operands)) {
    return failure();
  }
  if (hasC2vConsumerBuf &&
      parser.resolveOperand(c2vConsumerBuf, c2vConsumerBufTy, result.operands)) {
    return failure();
  }
  if (hasV2cConsumerBuf &&
      parser.resolveOperand(v2cConsumerBuf, v2cConsumerBufTy, result.operands)) {
    return failure();
  }
  return success();
}

template <typename InitOpT>
static void printFrontendInitializePipeOp(InitOpT op, OpAsmPrinter &p) {
  p << " {";
  bool needsComma = false;
  auto printClause = [&](StringRef keyword, auto value) {
    if (needsComma) {
      p << ", ";
    }
    p << keyword << " = " << value;
    needsComma = true;
  };

  printClause("id", op.getId());
  printClause("dir_mask", static_cast<int32_t>(op.getDirMask()));
  printClause("slot_size", op.getSlotSize());
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    printClause("slot_num", slotNumAttr.getInt());
  }
  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    printClause("local_slot_num", localSlotNumAttr.getInt());
  }
  if (auto noSplitAttr = op.getNosplitAttr()) {
    printClause("nosplit", noSplitAttr.getValue() ? "true" : "false");
  }
  if (auto accPushEpilogueAttr = op.getAccPushEpilogueAttr()) {
    printClause("acc_push_epilogue", accPushEpilogueAttr);
  }
  p << "}";

  p << "(";
  bool needsOperandComma = false;
  auto printOperandClause = [&](StringRef keyword, Value value) {
    if (needsOperandComma) {
      p << ", ";
    }
    p << keyword << " = " << value << " : " << value.getType();
    needsOperandComma = true;
  };
  if (op.getGmSlotBuffer()) {
    printOperandClause("gm_slot_buffer", op.getGmSlotBuffer());
  }
  if (op.getGmSlotTensor()) {
    printOperandClause("gm_slot_tensor", op.getGmSlotTensor());
  }
  if (op.getC2vConsumerBuf()) {
    printOperandClause("c2v_consumer_buf", op.getC2vConsumerBuf());
  }
  if (op.getV2cConsumerBuf()) {
    printOperandClause("v2c_consumer_buf", op.getV2cConsumerBuf());
  }
  p << ")";
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"id", "dir_mask", "slot_size", "slot_num",
                       "local_slot_num", "acc_push_epilogue",
                       "nosplit", "operandSegmentSizes"});
}

static std::optional<uint64_t>
getStaticElementCount(ArrayRef<int64_t> shape) {
  uint64_t count = 1;
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic || dim < 0) {
      return std::nullopt;
    }
    count *= static_cast<uint64_t>(dim);
  }
  return count;
}

static bool isSameOrHalfSlotByteSize(uint64_t tensorBytes, uint64_t slotBytes) {
  return tensorBytes == slotBytes || tensorBytes * 2 == slotBytes;
}

static LogicalResult verifyFrontendGlobalSlotTensor(Operation *op, Value tensor,
                                                    int8_t dirMask,
                                                    int32_t slotSize) {
  (void)dirMask;
  auto tvTy = dyn_cast<TensorViewType>(tensor.getType());
  if (!tvTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }

  ArrayRef<int64_t> shape = tvTy.getShape();
  if (shape.empty()) {
    return op->emitOpError(
        "expects 'gm_slot_tensor' to describe one slot entry tensor");
  }

  if (auto elemCount = getStaticElementCount(shape)) {
    uint64_t elemBytes = getElemByteSize(tvTy.getElementType());
    if (elemBytes != 0) {
      uint64_t tensorBytes = *elemCount * elemBytes;
      if (!isSameOrHalfSlotByteSize(tensorBytes,
                                    static_cast<uint64_t>(slotSize))) {
        return op->emitOpError()
               << "expects 'slot_size' to equal gm_slot_tensor byte size "
                  "or twice gm_slot_tensor byte size for split GlobalTensor "
                  "entries (got slot_size = "
               << slotSize << ", gm_slot_tensor byte size = " << tensorBytes
               << ")";
      }
    }
  }

  return success();
}

template <typename InitOpT>
static LogicalResult verifyFrontendInitCommon(InitOpT op,
                                              FunctionKernelKind expected,
                                              StringRef kernelName) {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(op.getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendKernelKind(op.getOperation(), expected, kernelName))) {
    return failure();
  }

  auto funcOp = op->template getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op.emitOpError("must be nested under a func.func");
  }

  if (op.getId() < 0) {
    return op.emitOpError("expects 'id' to be non-negative");
  }

  unsigned sameIdInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == op.getId()) {
        ++sameIdInitCount;
      }
      return;
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == op.getId()) {
        ++sameIdInitCount;
      }
    }
  });
  if (sameIdInitCount > 1) {
    return op.emitOpError(
        "requires 'id' to be unique across frontend initialize_pipe ops in the function");
  }

  int8_t dirMask = op.getDirMask();
  if (dirMask != 1 && dirMask != 2 && dirMask != 3) {
    return op.emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (op.getSlotSize() <= 0) {
    return op.emitOpError("expects 'slot_size' to be greater than 0");
  }
  int32_t slotNum = dirMask == 3 ? 4 : 8;
  if (auto slotNumAttr = op.getSlotNumAttr()) {
    slotNum = slotNumAttr.getInt();
    if (slotNum <= 0) {
      return op.emitOpError("expects 'slot_num' to be greater than 0");
    }
  }
  PTOArch arch = getTargetArch(op.getOperation());

  bool hasGlobalSlotTensor = static_cast<bool>(op.getGmSlotTensor());
  bool hasGmSlotBuffer = static_cast<bool>(op.getGmSlotBuffer());
  bool hasC2vConsumerBuf = static_cast<bool>(op.getC2vConsumerBuf());
  bool hasV2cConsumerBuf = static_cast<bool>(op.getV2cConsumerBuf());
  if (hasGlobalSlotTensor) {
    if (hasGmSlotBuffer) {
      return op.emitOpError(
          "'gm_slot_tensor' cannot be combined with 'gm_slot_buffer'");
    }
    if (hasC2vConsumerBuf || hasV2cConsumerBuf) {
      bool supportsC2V =
          dirMask == 1 && hasC2vConsumerBuf && !hasV2cConsumerBuf;
      bool supportsA2A3V2C =
          arch != PTOArch::A5 &&
          ((dirMask == 2 && !hasC2vConsumerBuf && hasV2cConsumerBuf) ||
           (dirMask == 3 && hasC2vConsumerBuf && hasV2cConsumerBuf));
      if (!supportsC2V && !supportsA2A3V2C) {
        return op.emitOpError(
            "GM-backed tile pipe init supports dir_mask = 1 with "
            "'c2v_consumer_buf' on all targets and dir_mask = 2/3 with "
            "matching consumer buffers only on a2/a3");
      }
    }
    if (!hasC2vConsumerBuf && !hasV2cConsumerBuf) {
      if (op.getLocalSlotNumAttr()) {
        return op.emitOpError(
            "globaltensor pipe init does not use 'local_slot_num'");
      }
      return verifyFrontendGlobalSlotTensor(
          op.getOperation(), op.getGmSlotTensor(), dirMask, op.getSlotSize());
    }
    if (failed(verifyFrontendGlobalSlotTensor(
            op.getOperation(), op.getGmSlotTensor(), dirMask,
            op.getSlotSize()))) {
      return failure();
    }
  }

  if (!hasC2vConsumerBuf && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects local pipe init to provide at least one consumer buffer "
        "operand; use 'gm_slot_tensor' for globaltensor pipe entries");
  }
  if (dirMask == 1 && !hasC2vConsumerBuf) {
    return op.emitOpError(
        "expects 'c2v_consumer_buf' when dir_mask is 1");
  }
  if (dirMask == 2 && !hasV2cConsumerBuf) {
    return op.emitOpError(
        "expects 'v2c_consumer_buf' when dir_mask is 2");
  }
  if (dirMask == 3 && (!hasC2vConsumerBuf || !hasV2cConsumerBuf)) {
    return op.emitOpError(
        "expects both 'c2v_consumer_buf' and 'v2c_consumer_buf' when dir_mask is 3");
  }

  if (auto localSlotNumAttr = op.getLocalSlotNumAttr()) {
    if (arch == PTOArch::A5) {
      return op.emitOpError(
          "'local_slot_num' is only supported for a2/a3 frontend pipe lowering");
    }
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0) {
      return op.emitOpError("expects 'local_slot_num' to be greater than 0");
    }
    if (localSlotNum > slotNum) {
      return op.emitOpError()
             << "expects 'local_slot_num' to be less than or equal to slot_num ("
             << slotNum << ") for dir_mask = " << static_cast<int>(dirMask);
    }
  }

  // Fixpipe validation
  if (auto accPushEpilogue = op.getAccPushEpilogueAttr()) {
    // Rule 1: fixpipe must be C2V only (dir_mask == 1)
    if (dirMask != 1) {
      return op.emitOpError(
          "expects fixpipe pipe (with 'acc_push_epilogue') to have dir_mask = 1 (C2V only)");
    }

    // Rule 2: fixpipe must have nosplit = true
    if (!op.getNosplit()) {
      return op.emitOpError(
          "expects fixpipe pipe (with 'acc_push_epilogue') to have nosplit = true");
    }

    // Rule 5: C2V consumer buf must be traceable to a peerable buffer op.
    // This init-common check only enforces a local def-use precondition. The
    // full peer-contract match (buffer name + peer_func + opposite-side init)
    // is verified later in AicInitializePipeOp::verify() /
    // AivInitializePipeOp::verify().
    if (hasC2vConsumerBuf) {
      Value c2vBuf = op.getC2vConsumerBuf();
      Operation *defOp = c2vBuf.getDefiningOp();
      bool foundReserve = false;
      if (defOp) {
        if (isa<ReserveBufferOp>(defOp) || isa<ImportReservedBufferOp>(defOp)) {
          foundReserve = true;
        }
      }
      if (!foundReserve) {
        return op.emitOpError(
            "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
            "import_reserved_buffer for peer contract verification");
      }
    }

    // Rule 7: relu must be no_relu or normal_relu in v1
    auto relu = accPushEpilogue.getRelu();
    if (relu != pto::FixpipeRelu::NoRelu && relu != pto::FixpipeRelu::NormalRelu) {
      return op.emitOpError(
          "expects 'acc_push_epilogue.relu' to be 'no_relu' or 'normal_relu' in v1");
    }

    // Rule 8: quant must be in v1 allowed set
    auto quant = accPushEpilogue.getQuant();
    bool validQuant = false;
    switch (quant) {
      case pto::FixpipeQuant::NoConvert:
      case pto::FixpipeQuant::F32F16:
      case pto::FixpipeQuant::F32BF16:
      case pto::FixpipeQuant::REQ8Scalar:
      case pto::FixpipeQuant::REQ8Vec:
      case pto::FixpipeQuant::DEQF16Scalar:
      case pto::FixpipeQuant::DEQF16Vec:
      case pto::FixpipeQuant::QF322B8PreScalar:
      case pto::FixpipeQuant::QF322B8PreVec:
      case pto::FixpipeQuant::QF322F16PreScalar:
      case pto::FixpipeQuant::QF322BF16PreScalar:
      case pto::FixpipeQuant::QS322BF16PreScalar:
      case pto::FixpipeQuant::QS322BF16PreVec:
      case pto::FixpipeQuant::QF322HIF8PreScalar:
      case pto::FixpipeQuant::QF322FP8PreScalar:
        validQuant = true;
        break;
    }
    if (!validQuant) {
      return op.emitOpError(
          "expects 'acc_push_epilogue.quant' to be one of the v1 allowed quantization modes");
    }

    // Rule 10: Check A5-only modes
    if (quant == pto::FixpipeQuant::QS322BF16PreScalar ||
        quant == pto::FixpipeQuant::QS322BF16PreVec) {
      if (arch != PTOArch::A5) {
        return op.emitOpError(
            "expects 'qs322bf16_pre_*' quantization modes to be used only on A5 target");
      }
    }
    if (quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
        quant == pto::FixpipeQuant::QF322FP8PreScalar) {
      if (arch != PTOArch::A5) {
        return op.emitOpError(
            "expects 'qf322hif8_pre_scalar'/'qf322fp8_pre_scalar' to be used only on A5 target");
      }
    }

    // Rule 14-15: fixpipe slot_size is interpreted as post-fixpipe
    // consumer-visible entry bytes. This init-common check relies on the
    // later fixpipe peer verifier to resolve consumer tpop tile types and
    // enforce the concrete lower bound against that resolved entry shape.
  }

  return success();
}

ParseResult AicInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AicInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ParseResult AivInitializePipeOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseFrontendInitializePipeOp(parser, result);
}

void AivInitializePipeOp::print(OpAsmPrinter &p) {
  printFrontendInitializePipeOp(*this, p);
}

ReserveBufferOp mlir::pto::findReserveBufferByName(func::FuncOp funcOp,
                                                   StringRef name) {
  ReserveBufferOp found;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() != name) {
      return WalkResult::advance();
    }
    found = reserveOp;
    return WalkResult::interrupt();
  });
  return found;
}

LogicalResult ReserveBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  if (getSize() <= 0) {
    return emitOpError("expects 'size' to be greater than 0");
  }

  auto location = getLocation().getAddressSpace();
  if (location != AddressSpace::VEC && location != AddressSpace::MAT) {
    return emitOpError("expects 'location' to be #pto.address_space<vec> or #pto.address_space<mat>");
  }

  if (!getAutoAlloc() && !getBaseAttr()) {
    return emitOpError("expects 'base' when 'auto' is false");
  }

  if (auto baseAttr = getBaseAttr(); baseAttr && baseAttr.getInt() < 0) {
    return emitOpError("expects 'base' to be non-negative when present");
  }

  unsigned sameNameCount = 0;
  funcOp.walk([&](ReserveBufferOp reserveOp) {
    if (reserveOp.getName() == getName()) {
      ++sameNameCount;
    }
  });
  if (sameNameCount > 1) {
    return emitOpError("requires 'name' to be unique within the function");
  }

  return success();
}

LogicalResult ImportReservedBufferOp::verify() {
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto peerFunc = lookupPeerFuncAcrossContainer(getOperation(), getPeerFuncAttr());
  if (!peerFunc) {
    return emitOpError("expects 'peer_func' to reference an existing func.func");
  }

  unsigned sameImportCount = 0;
  funcOp.walk([&](ImportReservedBufferOp importOp) {
    if (importOp.getName() == getName() &&
        importOp.getPeerFuncAttr() == getPeerFuncAttr()) {
      ++sameImportCount;
    }
  });
  if (sameImportCount > 1) {
    return emitOpError(
        "requires (name, peer_func) to be unique within the function");
  }

  if (!findReserveBufferByName(peerFunc, getName())) {
    return emitOpError("expects matching peer reserve_buffer to exist");
  }

  return success();
}

constexpr llvm::StringLiteral kFrontendPipeIdAttrName = "__pto.frontend_id";
constexpr llvm::StringLiteral kPipePeerOwnerFuncAttrName =
    "__pto.peer_owner_func";
constexpr llvm::StringLiteral kPipePeerReserveNameAttrName =
    "__pto.peer_reserve_name";
constexpr llvm::StringLiteral kPipePeerDirMaskAttrName = "__pto.peer_dir_mask";

struct FixpipeQuantStateResource
    : public SideEffects::Resource::Base<FixpipeQuantStateResource> {
  StringRef getName() final { return "PTOFixpipeQuantState"; }
};

static IntegerAttr getFixpipeQuantStateIdAttr(Operation *op, int32_t id) {
  return IntegerAttr::get(IntegerType::get(op->getContext(), 32), id);
}

static FailureOr<Operation *> lookupFrontendInitOpById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedInit = candidate;
        ++matchedInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static std::optional<int32_t> getFrontendPipeIdFromHandle(Value pipeHandle) {
  if (!pipeHandle) {
    return std::nullopt;
  }
  Operation *defOp = pipeHandle.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto frontendIdAttr = defOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
  if (!frontendIdAttr) {
    return std::nullopt;
  }
  return static_cast<int32_t>(frontendIdAttr.getInt());
}

static pto::AccPushEpilogueAttr getAccPushEpilogueFromInitOp(Operation *initOp) {
  if (!initOp) {
    return {};
  }
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aicInit.getAccPushEpilogueAttr();
  }
  if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
    return aivInit.getAccPushEpilogueAttr();
  }
  if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    return l2lInit.getAccPushEpilogueAttr();
  }
  if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    return l2g2lInit.getAccPushEpilogueAttr();
  }
  return {};
}

static bool matchesLoweredFixpipePeerContract(Operation *initOp,
                                              func::FuncOp expectedOwnerFunc,
                                              StringRef expectedReserveName) {
  if (!initOp || !isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(initOp)) {
    return false;
  }

  auto ownerAttr =
      initOp->getAttrOfType<FlatSymbolRefAttr>(kPipePeerOwnerFuncAttrName);
  auto reserveAttr =
      initOp->getAttrOfType<StringAttr>(kPipePeerReserveNameAttrName);
  auto dirMaskAttr =
      initOp->getAttrOfType<IntegerAttr>(kPipePeerDirMaskAttrName);
  if (!ownerAttr || !reserveAttr || !dirMaskAttr) {
    return false;
  }

  if (ownerAttr.getValue() != expectedOwnerFunc.getSymName() ||
      reserveAttr.getValue() != expectedReserveName ||
      dirMaskAttr.getInt() != 1) {
    return false;
  }

  return static_cast<bool>(getAccPushEpilogueFromInitOp(initOp));
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id);

static FailureOr<Operation *>
lookupFixpipePeerConsumerInit(AicInitializePipeOp producerInit) {
  if (!producerInit.getC2vConsumerBuf()) {
    return failure();
  }
  auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
      producerInit.getC2vConsumerBuf().getDefiningOp());
  if (!importOp) {
    return failure();
  }

  auto peerConsumerFunc =
      lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                    importOp.getPeerFuncAttr());
  if (!peerConsumerFunc) {
    return failure();
  }

  StringRef bufferName = importOp.getName();
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerConsumerFunc.walk([&](Operation *candidate) {
    if (auto aivInit = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (!aivInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }
      auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(
          aivInit.getC2vConsumerBuf().getDefiningOp());
      if (!reserveOp || reserveOp.getName() != bufferName) {
        return WalkResult::advance();
      }
      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, peerConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    return failure();
  }
  return matchedInit;
}

static FailureOr<Operation *>
lookupFixpipePeerProducerInit(AivInitializePipeOp consumerInit,
                              func::FuncOp peerProducerFunc,
                              StringRef bufferName,
                              func::FuncOp currentConsumerFunc) {
  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  peerProducerFunc.walk([&](Operation *candidate) {
    if (auto aicInit = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (!aicInit.getC2vConsumerBuf()) {
        return WalkResult::advance();
      }

      auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(
          aicInit.getC2vConsumerBuf().getDefiningOp());
      if (!importOp) {
        return WalkResult::advance();
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(importOp.getOperation(),
                                        importOp.getPeerFuncAttr());
      if (importOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return WalkResult::advance();
      }

      matchedInit = candidate;
      ++matchedInitCount;
      return WalkResult::advance();
    }

    if (!matchesLoweredFixpipePeerContract(candidate, currentConsumerFunc,
                                           bufferName)) {
      return WalkResult::advance();
    }

    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount != 1) {
    consumerInit.emitOpError()
        << "expects peer producer function to contain a matching "
           "aic_initialize_pipe with the same consumer buffer contract";
    return failure();
  }

  return matchedInit;
}

static FailureOr<Operation *> lookupFrontendOrLoweredInitOpById(
    Operation *op, func::FuncOp funcOp, int32_t id) {
  Operation *matchedFrontendInit = nullptr;
  unsigned matchedFrontendInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (auto aic = dyn_cast<AicInitializePipeOp>(candidate)) {
      if (aic.getId() == static_cast<uint32_t>(id)) {
        matchedFrontendInit = candidate;
        ++matchedFrontendInitCount;
      }
      return WalkResult::advance();
    }
    if (auto aiv = dyn_cast<AivInitializePipeOp>(candidate)) {
      if (aiv.getId() == static_cast<uint32_t>(id)) {
        matchedFrontendInit = candidate;
        ++matchedFrontendInitCount;
      }
      return WalkResult::advance();
    }
    return WalkResult::advance();
  });

  if (matchedFrontendInitCount == 1) {
    return matchedFrontendInit;
  }
  if (matchedFrontendInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend initialize_pipe op in the same function";
    return failure();
  }

  Operation *matchedInit = nullptr;
  unsigned matchedInitCount = 0;
  funcOp.walk([&](Operation *candidate) {
    if (!isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(candidate)) {
      return WalkResult::advance();
    }
    auto frontendIdAttr =
        candidate->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
    if (!frontendIdAttr || frontendIdAttr.getInt() != id) {
      return WalkResult::advance();
    }
    matchedInit = candidate;
    ++matchedInitCount;
    return WalkResult::advance();
  });

  if (matchedInitCount == 0) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match a frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  if (matchedInitCount > 1) {
    op->emitOpError() << "expects 'id' = " << id
                      << " to match exactly one frontend or lowered initialize_pipe op in the same function";
    return failure();
  }
  return matchedInit;
}

static LogicalResult verifyFrontendSplitOp(Operation *op,
                                           FunctionKernelKind expected,
                                           StringRef kernelName,
                                           int32_t id,
                                           int64_t split,
                                           bool expectC2V) {
  if (failed(verifyFrontendKernelKind(op, expected, kernelName))) {
    return failure();
  }
  if (id < 0) {
    return op->emitOpError("expects 'id' to be non-negative");
  }
  if (failed(verifySplitAttr(op, split))) {
    return failure();
  }
  if (!isOddSplit(split)) {
    return success();
  }

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  if (!expectC2V && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }

  auto supportsOddDirection = [&](auto init) {
    PTOArch arch = getTargetArch(op);
    bool hasSupportedGmBacking =
        static_cast<bool>(init.getGmSlotTensor()) ||
        (arch != PTOArch::A5 && static_cast<bool>(init.getGmSlotBuffer()));
    int8_t directionMask = expectC2V ? 1 : 2;
    Value consumerBuffer = expectC2V ? init.getC2vConsumerBuf()
                                     : init.getV2cConsumerBuf();
    return (init.getDirMask() & directionMask) != 0 &&
           hasSupportedGmBacking && static_cast<bool>(consumerBuffer);
  };

  bool supported = false;
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    supported = supportsOddDirection(aic);
  } else {
    supported = supportsOddDirection(cast<AivInitializePipeOp>(*initOr));
  }
  if (!supported) {
    return op->emitOpError()
           << "supports odd split modes (split = 3 or 4) only for a "
              "GM-backed tile pipe whose dir_mask enables "
           << (expectC2V ? "C2V" : "V2C")
           << " and provides the matching consumer buffer";
  }
  return success();
}

static LogicalResult verifyOddSplitTileEntry(Operation *op, int64_t split,
                                             Type entryTy) {
  if (isOddSplit(split) && isa<TensorViewType>(entryTy)) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for tile entries; "
        "the pinned pto-isa does not implement odd GlobalTensor offsets");
  }
  return success();
}

static LogicalResult verifyFullTileSplitParity(Operation *op, int64_t split,
                                               Type entryTy) {
  if (split == 0) {
    return success();
  }

  ArrayRef<int64_t> shape;
  if (auto tileTy = dyn_cast<TileBufType>(entryTy)) {
    shape = tileTy.getValidShape();
  } else if (auto viewTy = dyn_cast<TensorViewType>(entryTy)) {
    shape = viewTy.getShape();
  } else {
    return success();
}
  if (shape.size() != 2) {
    return success();
  }

  bool splitRows = split == 1 || split == 3;
  int64_t axisSize = shape[splitRows ? 0 : 1];
  if (axisSize == ShapedType::kDynamic) {
    return success();
  }

  bool expectOdd = isOddSplit(split);
  if ((axisSize % 2 != 0) != expectOdd) {
    return op->emitOpError()
           << "expects a statically " << (expectOdd ? "odd" : "even")
           << " valid-" << (splitRows ? "row" : "column")
           << " count for split = " << split;
  }
  return success();
}

static LogicalResult verifyAivSubblockIdOperand(Operation *op,
                                                Value aivSubblockId,
                                                int64_t split,
                                                Type pipeEntryType) {
  if (!aivSubblockId) {
    return success();
  }

  if (split == 0) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only when 'split' is 1, 2, 3, or 4");
  }

  if (isa<TensorViewType>(pipeEntryType)) {
    return op->emitOpError(
        "does not support 'aiv_subblockid' for !pto.tensor_view pipe entries");
  }

  auto addrSpace = getPTOMemorySpaceEnum(pipeEntryType);
  if (!addrSpace || *addrSpace != AddressSpace::VEC) {
    return op->emitOpError(
        "expects 'aiv_subblockid' only on AIV-side vector tile pipe entries");
  }

  return success();
}

static FailureOr<int8_t> lookupFrontendInitDirMaskById(Operation *op,
                                                       func::FuncOp funcOp,
                                                       int32_t id) {
  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  if (auto aic = dyn_cast<AicInitializePipeOp>(*initOr)) {
    return aic.getDirMask();
  }
  return cast<AivInitializePipeOp>(*initOr).getDirMask();
}

static LogicalResult verifyFrontendDataOpDirection(Operation *op, int32_t id,
                                                   bool expectC2V) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }

  auto dirMaskOr = lookupFrontendInitDirMaskById(op, funcOp, id);
  if (failed(dirMaskOr)) {
    return failure();
  }

  int8_t dirMask = *dirMaskOr;
  if (expectC2V && dirMask != 1 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 1 or 3";
  }
  if (!expectC2V && dirMask != 2 && dirMask != 3) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with dir_mask = 2 or 3";
  }
  return success();
}

static Value getFrontendInitGmSlotTensor(Operation *initOp) {
  if (auto aic = dyn_cast<AicInitializePipeOp>(initOp)) {
    return aic.getGmSlotTensor();
  }
  return cast<AivInitializePipeOp>(initOp).getGmSlotTensor();
}

static LogicalResult verifyFrontendTensorEntryMatchesInit(Operation *op,
                                                          int32_t id,
                                                          Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return op->emitOpError("must be nested under a func.func");
  }

  auto initOr = lookupFrontendInitOpById(op, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }
  Value gmSlotTensor = getFrontendInitGmSlotTensor(*initOr);
  if (!gmSlotTensor) {
    return op->emitOpError()
           << "expects 'id' = " << id
           << " to reference initialize_pipe with 'gm_slot_tensor' when the "
              "pipe entry is !pto.tensor_view";
  }

  auto slotTensorTy = dyn_cast<TensorViewType>(gmSlotTensor.getType());
  if (!slotTensorTy) {
    return op->emitOpError("expects 'gm_slot_tensor' to be !pto.tensor_view");
  }
  if (slotTensorTy.getElementType() != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match gm_slot_tensor element type";
  }
  if (slotTensorTy.getRank() != entryViewTy.getRank()) {
    return op->emitOpError()
           << "expects pipe entry rank to match gm_slot_tensor rank";
  }

  ArrayRef<int64_t> slotShape = slotTensorTy.getShape();
  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match gm_slot_tensor dimension " << slotDim;
  }
  return success();
}

template <typename FrontendPopOpT>
static LogicalResult verifyFrontendPopOp(FrontendPopOpT op,
                                         FunctionKernelKind expected,
                                         StringRef kernelName,
                                         bool expectC2V) {
  if (failed(verifyFrontendSplitOp(op.getOperation(), expected, kernelName,
                                   op.getId(),
                                   op.getSplit(), expectC2V))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(op.getOperation(), op.getId(),
                                           expectC2V))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(op.getOperation(), op.getSplit(),
                                     op.getTile().getType()))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(op.getOperation(), op.getId(),
                                                  op.getTile().getType()))) {
    return failure();
  }
  if (!expectC2V &&
      failed(verifyFullTileSplitParity(op.getOperation(), op.getSplit(),
                                       op.getTile().getType()))) {
    return failure();
  }

  bool hasValidRow = static_cast<bool>(op.getValidRow());
  bool hasValidCol = static_cast<bool>(op.getValidCol());
  if (hasValidRow != hasValidCol) {
    return op.emitOpError(
        "expects valid_row and valid_col operands to be provided together");
  }
  if (expectC2V && isOddSplit(op.getSplit()) && !hasValidRow &&
      !isa<TensorViewType>(op.getTile().getType())) {
    return op.emitOpError(
        "expects odd C2V split tpop to provide per-sub-core valid_row and "
        "valid_col operands");
  }
  if (!hasValidRow) {
    return success();
  }

  if (isa<TensorViewType>(op.getTile().getType())) {
    return op.emitOpError(
        "does not accept valid_row/valid_col when result is !pto.tensor_view");
  }

  auto tileTy = dyn_cast<TileBufType>(op.getTile().getType());
  if (!tileTy) {
    return op.emitOpError(
        "expects tile result to be !pto.tile_buf when valid_row/valid_col operands are provided");
  }
  if (!tileTy.hasDynamicValid()) {
    return op.emitOpError(
        "expects tile result to have dynamic validShape (?, ?) when valid_row/valid_col operands are provided");
  }
  return success();
}

static bool isScalarFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
  case FixpipeQuant::QF322F16PreScalar:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QF322HIF8PreScalar:
  case FixpipeQuant::QF322FP8PreScalar:
    return true;
  default:
    return false;
  }
}

static bool isVectorFixpipeQuant(FixpipeQuant quant) {
  switch (quant) {
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
  case FixpipeQuant::QS322BF16PreVec:
    return true;
  default:
    return false;
  }
}

static bool matchesFixpipeConsumerLayout(FixpipeLayout layout,
                                         TileBufType tileTy) {
  auto memorySpace = dyn_cast_or_null<AddressSpaceAttr>(tileTy.getMemorySpace());
  if (!memorySpace || memorySpace.getAddressSpace() != AddressSpace::VEC) {
    return false;
  }

  int32_t bLayout = tileTy.getBLayoutValueI32();
  int32_t sLayout = tileTy.getSLayoutValueI32();
  switch (layout) {
  case FixpipeLayout::NZ2ND:
    return bLayout == static_cast<int32_t>(BLayout::RowMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2DN:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::NoneBox);
  case FixpipeLayout::NZ2NZ:
    return bLayout == static_cast<int32_t>(BLayout::ColMajor) &&
           sLayout == static_cast<int32_t>(SLayout::RowMajor);
  }
  llvm_unreachable("unhandled FixpipeLayout");
}

static bool isSignedOrUnsignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.getWidth() == 8 && (intTy.isSigned() || intTy.isUnsigned());
  }
  return false;
}

static bool isSignedI8(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.isSignedInteger(8);
  }
  return false;
}

static bool matchesFixpipeConsumerElementType(FixpipeQuant quant,
                                              Type resultElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return resultElemType.isF32() || resultElemType.isInteger(32);
  case FixpipeQuant::F32F16:
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
  case FixpipeQuant::QF322F16PreScalar:
    return resultElemType.isF16();
  case FixpipeQuant::F32BF16:
  case FixpipeQuant::QF322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return resultElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
  case FixpipeQuant::QF322B8PreScalar:
    return isSignedOrUnsignedI8(resultElemType);
  case FixpipeQuant::REQ8Vec:
  case FixpipeQuant::QF322B8PreVec:
    return isSignedI8(resultElemType);
  case FixpipeQuant::QF322HIF8PreScalar:
    return isa<HiF8Type>(resultElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return isPTOFloat8E4M3LikeType(resultElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isFixpipeQuantPayloadElemType(Type elemTy, PTOArch arch) {
  if (!elemTy) {
    return false;
  }
  if (arch == PTOArch::A3) {
    return elemTy.isUnsignedInteger(64) || elemTy.isSignlessInteger(64) ||
           elemTy.isSignedInteger(64);
  }
  return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32();
}

static bool matchesFixpipeProducerAndConsumerTypes(FixpipeQuant quant,
                                                   Type srcElemType,
                                                   Type dstElemType) {
  switch (quant) {
  case FixpipeQuant::NoConvert:
    return (srcElemType.isF32() || srcElemType.isInteger(32)) &&
           srcElemType == dstElemType;
  case FixpipeQuant::F32F16:
    return srcElemType.isF32() && dstElemType.isF16();
  case FixpipeQuant::F32BF16:
    return srcElemType.isF32() && dstElemType.isBF16();
  case FixpipeQuant::REQ8Scalar:
    return srcElemType.isInteger(32) && isSignedOrUnsignedI8(dstElemType);
  case FixpipeQuant::REQ8Vec:
    return srcElemType.isInteger(32) && isSignedI8(dstElemType);
  case FixpipeQuant::DEQF16Scalar:
  case FixpipeQuant::DEQF16Vec:
    return srcElemType.isInteger(32) && dstElemType.isF16();
  case FixpipeQuant::QF322B8PreScalar:
    return srcElemType.isF32() && isSignedOrUnsignedI8(dstElemType);
  case FixpipeQuant::QF322B8PreVec:
    return srcElemType.isF32() && isSignedI8(dstElemType);
  case FixpipeQuant::QF322F16PreScalar:
    return srcElemType.isF32() && dstElemType.isF16();
  case FixpipeQuant::QF322BF16PreScalar:
    return srcElemType.isF32() && dstElemType.isBF16();
  case FixpipeQuant::QS322BF16PreScalar:
  case FixpipeQuant::QS322BF16PreVec:
    return srcElemType.isInteger(32) && dstElemType.isBF16();
  case FixpipeQuant::QF322HIF8PreScalar:
    return srcElemType.isF32() && isa<HiF8Type>(dstElemType);
  case FixpipeQuant::QF322FP8PreScalar:
    return srcElemType.isF32() && isPTOFloat8E4M3LikeType(dstElemType);
  }
  llvm_unreachable("unhandled FixpipeQuant");
}

static bool isUnpublishedFixpipeFrontendAttrName(StringRef name) {
  return llvm::StringSwitch<bool>(name)
      .Case("stPhase", true)
      .Case("st_phase", true)
      .Case("atomicType", true)
      .Case("atomic_type", true)
      .Case("subBlockId", true)
      .Case("subBlockid", true)
      .Case("sub_blockid", true)
      .Case("clipReluMode", true)
      .Case("clip_relu_mode", true)
      .Case("isChannelSplit", true)
      .Case("is_channel_split", true)
      .Case("channelSplit", true)
      .Case("channel_split", true)
      .Default(false);
}

static LogicalResult verifyNoUnpublishedFixpipeFrontendAttrs(Operation *op) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (!isUnpublishedFixpipeFrontendAttrName(name)) {
      continue;
    }
    return op->emitOpError()
           << "does not allow unpublished fixpipe attr '" << name
           << "'; STPhase / AtomicType / SubBlockId / ClipReluMode / "
              "IsChannelSplit are not part of the PTOIR frontend surface";
  }
  return success();
}

static std::optional<uint64_t> getStaticTileByteSize(TileBufType tileTy) {
  auto shape = getShapeVec(tileTy);
  auto elemCount = getStaticElementCount(shape);
  uint64_t elemBytes = getElemByteSize(tileTy.getElementType());
  if (!elemCount || elemBytes == 0) {
    return std::nullopt;
  }
  return *elemCount * elemBytes;
}

// Helper to verify fixpipe consumer tpop result type consistency
static LogicalResult verifyFixpipeConsumerType(Operation *tpopOp, int32_t id,
                                               Type resultTileType) {
  auto funcOp = tpopOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return success(); // Already checked elsewhere
  }

  // Look up the consumer-side init
  auto initOr = lookupFrontendInitOpById(tpopOp, funcOp, id);
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aivInit = dyn_cast<AivInitializePipeOp>(initOp);
  if (!aivInit) {
    return success(); // Not consumer init, skip
  }

  auto accPushEpilogue = aivInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // Not a fixpipe, skip
  }

  if (auto tpop = dyn_cast<TPopFromAicOp>(tpopOp); tpop && tpop.getSplit() != 0) {
    return tpop.emitOpError("expects fixpipe TPOP to have split = 0");
  }

  // Rule 11: At least one tpop must exist (checked by counting all tpops for this pipe)
  // Rule 12: Verify result element type matches expected type from quant mode
  auto tileTy = dyn_cast<pto::TileBufType>(resultTileType);
  if (!tileTy) {
    return tpopOp->emitOpError(
        "expects fixpipe TPOP result to be a tile type");
  }

  Type resultElemType = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();
  bool mismatchedPeerType = false;
  funcOp.walk([&](TPopFromAicOp otherPop) {
    if (otherPop.getOperation() == tpopOp ||
        otherPop.getId() != static_cast<uint32_t>(id)) {
      return WalkResult::advance();
    }
    if (otherPop.getTile().getType() != resultTileType) {
      mismatchedPeerType = true;
      tpopOp->emitOpError()
          << "expects all tpop_from_aic results for fixpipe pipe id = " << id
          << " to use the same tile type";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (mismatchedPeerType) {
    return failure();
  }

  if (!matchesFixpipeConsumerElementType(quant, resultElemType)) {
    return tpopOp->emitOpError()
           << "expects consumer element type to match acc_push_epilogue.quant "
           << stringifyFixpipeQuant(quant);
  }

  if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(), tileTy)) {
    return tpopOp->emitOpError()
           << "expects consumer tile layout to match acc_push_epilogue.layout "
           << stringifyFixpipeLayout(accPushEpilogue.getLayout());
  }

  return success();
}


static LogicalResult verifyPipeShape(Operation *op, int8_t dirMask, int32_t slotSize,
                                     int32_t slotNum,
                                     std::optional<int32_t> flagBase) {
  constexpr int32_t kMaxHardwareFlagIds = 16;
  if (dirMask != 1 && dirMask != 2 && dirMask != 3) {
    return op->emitOpError("expects 'dir_mask' to be 1, 2, or 3");
  }
  if (slotSize <= 0) {
    return op->emitOpError("expects 'slot_size' to be greater than 0");
  }
  if (slotNum <= 0) {
    return op->emitOpError("expects 'slot_num' to be greater than 0");
  }
  if (flagBase && *flagBase < 0) {
    return op->emitOpError("expects 'flag_base' to be non-negative when present");
  }
  if (flagBase) {
    int32_t flagWidth = dirMask == 3 ? 4 : 2;
    if (*flagBase + flagWidth > kMaxHardwareFlagIds) {
      return op->emitOpError()
             << "requires 'flag_base' and dir_mask to fit within "
             << kMaxHardwareFlagIds << " hardware flag ids";
    }
  }

  return success();
}

static LogicalResult verifyPipeHandleProducer(Operation *op, Value pipeHandle) {
  if (!isa<pto::PipeType>(pipeHandle.getType())) {
    return op->emitOpError("expects pipe operand type !pto.pipe");
  }
  if (!pipeHandle.getDefiningOp<InitializeL2LPipeOp>() &&
      !pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>()) {
    return op->emitOpError(
        "pipe_handle must be produced by pto.initialize_l2l_pipe or "
        "pto.initialize_l2g2l_pipe");
  }
  return success();
}

static LogicalResult verifyInternalOddSplitSupport(Operation *op,
                                                   Value pipeHandle,
                                                   int64_t split,
                                                   bool producerSide) {
  if (!isOddSplit(split)) {
    return success();
  }

  bool isCubeSide = isInsideCubeKernelOrSection(op);
  bool isVectorSide = isInsideVectorKernelOrSection(op);
  bool isC2VSide = producerSide ? isCubeSide : isVectorSide;
  bool isV2CSide = producerSide ? isVectorSide : isCubeSide;
  int8_t directionMask = isC2VSide ? 1 : (isV2CSide ? 2 : 0);
  if (isV2CSide && getTargetArch(op) == PTOArch::A5) {
    return op->emitOpError(
        "supports odd V2C split modes (split = 3 or 4) only on a2/a3");
  }
  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  Value consumerBuffer;
  if (initOp && directionMask != 0) {
    consumerBuffer = directionMask == 2 && initOp.getDirMask() == 3
                         ? initOp.getPeerLocalAddr()
                         : initOp.getLocalAddr();
  }
  if (!initOp || directionMask == 0 ||
      (initOp.getDirMask() & directionMask) == 0 || !consumerBuffer) {
    return op->emitOpError(
        "supports odd split modes (split = 3 or 4) only for a "
        "pto.initialize_l2g2l_pipe whose dir_mask enables the operation "
        "direction and provides its local consumer buffer");
  }
  return success();
}

static bool getTensorLikeElementAndShape(Type ty, Type &elementType,
                                         ArrayRef<int64_t> &shape) {
  if (auto tvTy = dyn_cast<TensorViewType>(ty)) {
    elementType = tvTy.getElementType();
    shape = tvTy.getShape();
    return true;
  }
  return false;
}

static LogicalResult verifyTensorEntryMatchesInternalPipeInit(Operation *op,
                                                              Value pipeHandle,
                                                              Type entryTy) {
  auto entryViewTy = dyn_cast<TensorViewType>(entryTy);
  if (!entryViewTy) {
    return success();
  }

  auto initOp = pipeHandle.getDefiningOp<InitializeL2G2LPipeOp>();
  if (!initOp) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use a pipe produced by "
              "pto.initialize_l2g2l_pipe";
  }
  if (initOp.getLocalAddr()) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use global-only "
              "pto.initialize_l2g2l_pipe without local_addr";
  }

  Type slotElementType;
  ArrayRef<int64_t> slotShape;
  if (!getTensorLikeElementAndShape(initOp.getGmAddr().getType(),
slotElementType, slotShape)) {
    return op->emitOpError()
           << "expects !pto.tensor_view pipe entry to use "
              "pto.initialize_l2g2l_pipe gm_addr with tensor_view slot type";
  }

  if (slotElementType != entryViewTy.getElementType()) {
    return op->emitOpError()
           << "expects pipe entry element type to match initialize_l2g2l_pipe "
              "gm_addr element type";
  }
  if (slotShape.size() != static_cast<size_t>(entryViewTy.getRank())) {
    return op->emitOpError()
           << "expects pipe entry rank to match initialize_l2g2l_pipe gm_addr "
              "rank";
  }

  ArrayRef<int64_t> entryShape = entryViewTy.getShape();
  for (auto [idx, entryDim] : llvm::enumerate(entryShape)) {
    int64_t slotDim = slotShape[idx];
    if (slotDim == ShapedType::kDynamic ||
        entryDim == ShapedType::kDynamic || slotDim == entryDim) {
      continue;
    }
    return op->emitOpError()
           << "expects pipe entry dimension " << idx
           << " to match initialize_l2g2l_pipe gm_addr dimension "
           << slotDim;
  }

  if (auto entryElemCount = getStaticElementCount(entryShape)) {
    uint64_t elemBytes = getElemByteSize(entryViewTy.getElementType());
    uint64_t entryBytes = *entryElemCount * elemBytes;
    if (elemBytes != 0) {
      int8_t split = 0;
      if (auto alloc = dyn_cast<TAllocOp>(op)) {
        split = alloc.getSplit();
      } else if (auto push = dyn_cast<TPushOp>(op)) {
        split = push.getSplit();
      } else if (auto pop = dyn_cast<TPopOp>(op)) {
        split = pop.getSplit();
      } else if (auto free = dyn_cast<TFreeOp>(op)) {
        split = free.getSplit();
      }

      uint64_t slotBytes = static_cast<uint64_t>(initOp.getSlotSize());
      bool isSplitEntry = split != 0;
      bool byteSizeMatches =
          entryBytes == slotBytes || (isSplitEntry && entryBytes * 2 == slotBytes);
      if (!byteSizeMatches) {
        return op->emitOpError()
               << "expects pipe entry byte size to match initialize_l2g2l_pipe "
                  "slot_size"
               << (isSplitEntry ? " or half slot_size for split entries" : "")
               << " (got entry byte size = " << entryBytes
               << ", slot_size = " << initOp.getSlotSize() << ")";
      }
    }
  }

  return success();
}

LogicalResult BuildAsyncSessionOp::verify() {
  Type scratchTy = getScratch().getType();
  if (!isa<pto::TileBufType>(scratchTy)) {
    return emitOpError("expects scratch to be tile_buf type");
  }

  auto scratchSpace = getPTOMemorySpaceEnum(scratchTy);
  if (!scratchSpace || *scratchSpace != pto::AddressSpace::VEC) {
    return emitOpError("expects scratch to be in vec address space");
  }

  auto scratchShape = getShapeVec(scratchTy);
  if (scratchShape.empty() || scratchShape.size() > 2) {
    return emitOpError("expects scratch to be rank-1 or rank-2");
  }
  for (int64_t dim : scratchShape) {
    if (dim == ShapedType::kDynamic) {
      return emitOpError("expects scratch to have a static shape");
    }
  }

  auto scratchBytes = getStaticByteSize(scratchTy);
  if (!scratchBytes) {
    return emitOpError("expects scratch byte size to be statically known");
  }
  if (*scratchBytes < sizeof(uint64_t)) {
    return emitOpError("expects scratch to provide at least 8 bytes");
  }

  auto workspaceTy = dyn_cast<pto::PtrType>(getWorkspace().getType());
  if (!workspaceTy) {
    return emitOpError("expects workspace to be !pto.ptr type");
  }
  Type workspaceElemTy = workspaceTy.getElementType();
  if (!isByteIntegerType(workspaceElemTy)) {
    return emitOpError("expects workspace element type to be an 8-bit integer");
  }

  if (auto syncIdAttr = getSyncIdAttr()) {
    int64_t syncId = syncIdAttr.getInt();
    if (syncId < 0 || syncId > 7) {
      return emitOpError("expects sync_id in range [0, 7]");
    }
  }
  if (auto blockBytesAttr = getBlockBytesAttr()) {
    if (blockBytesAttr.getInt() <= 0) {
      return emitOpError("expects block_bytes to be greater than 0");
    }
  }
  if (auto commBlockOffsetAttr = getCommBlockOffsetAttr()) {
    if (commBlockOffsetAttr.getInt() < 0) {
      return emitOpError("expects comm_block_offset to be non-negative");
    }
  }
  if (auto queueNumAttr = getQueueNumAttr()) {
    if (queueNumAttr.getInt() <= 0) {
      return emitOpError("expects queue_num to be greater than 0");
    }
  }
  if (auto channelGroupIdxAttr = getChannelGroupIdxAttr()) {
    APInt value = channelGroupIdxAttr.getValue();
    if (value.isNegative()) {
      return emitOpError("expects channel_group_idx to be non-negative");
    }
    if (value.ugt(UINT32_MAX)) {
      return emitOpError("expects channel_group_idx to fit in uint32");
    }
  }

  return success();
}

static LogicalResult verifyAsyncTransferOp(Operation *op, Value dst, Value src) {
  Type dstElemTy = getElemTy(dst.getType());
  Type srcElemTy = getElemTy(src.getType());
  if (!dstElemTy || !srcElemTy) {
    return op->emitOpError("expects src and dst to have element types");
  }
  if (dstElemTy != srcElemTy) {
    return op->emitOpError("expects src and dst to have the same element type");
  }
  if (failed(verifyAsyncFlatContiguous1DGMViewLike(op, dst, "dst")) ||
      failed(verifyAsyncFlatContiguous1DGMViewLike(op, src, "src"))) {
    return failure();
  }
  if (getShapeVec(dst.getType()) != getShapeVec(src.getType())) {
    return op->emitOpError("expects src and dst to have the same static shape");
  }
  return success();
}

LogicalResult TPutAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TGetAsyncOp::verify() {
  return verifyAsyncTransferOp(getOperation(), getDst(), getSrc());
}

LogicalResult TPutOp::verify() {
  if (failed(verifyCommGlobalLike(
          *this, getDst(), "dst",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommGlobalLike(
          *this, getSrc(), "src",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong"))) {
    return failure();
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType())) {
    return emitOpError(
        "expects src and dst to have the same static/dynamic shape signature");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src/dst");
  }
  return success();
}

LogicalResult TGetOp::verify() {
  if (failed(verifyCommGlobalLike(
          *this, getDst(), "dst",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommGlobalLike(
          *this, getSrc(), "src",
          CommGlobalShapePolicy::AllowDynamicPartitionView)) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong"))) {
    return failure();
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects src and dst to have the same element type");
  }
  if (getShapeVec(getDst().getType()) != getShapeVec(getSrc().getType())) {
    return emitOpError(
        "expects src and dst to have the same static/dynamic shape signature");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src/dst");
  }
  return success();
}

LogicalResult TNotifyOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto valueTy = dyn_cast<IntegerType>(getValue().getType());
  if (!valueTy || valueTy.getWidth() != 32) {
    return emitOpError("expects value to be i32");
  }
  return success();
}

LogicalResult TWaitOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32) {
    return emitOpError("expects cmp_value to be i32");
  }
  return success();
}

LogicalResult TTestOp::verify() {
  if (failed(verifyCommSignalLike(*this, getSignal(), "signal"))) {
    return failure();
  }
  auto cmpTy = dyn_cast<IntegerType>(getCmpValue().getType());
  if (!cmpTy || cmpTy.getWidth() != 32) {
    return emitOpError("expects cmp_value to be i32");
  }
  return success();
}

static LogicalResult verifySyncAllGmWorkspace(Operation *op, Value workspace,
                                              StringRef name) {
  Type ty = workspace.getType();
  Type elemType;
  SmallVector<int64_t, 4> shape;
  if (auto ptrTy = dyn_cast<pto::PtrType>(ty)) {
    if (ptrTy.getMemorySpace().getAddressSpace() != pto::AddressSpace::GM) {
      return op->emitOpError() << "expects " << name
                               << " to be in GM address space";
    }
    elemType = ptrTy.getElementType();
  } else if (isa<pto::TensorViewType, pto::PartitionTensorViewType>(ty)) {
    elemType = getElemTy(ty);
    shape = getShapeVec(ty);
  } else {
    return op->emitOpError()
           << "expects " << name
           << " to be a GM ptr/tensor_view/partition_view";
  }

  auto elemTy = dyn_cast<IntegerType>(elemType);
  if (!elemTy || elemTy.getWidth() != 32) {
    return op->emitOpError() << "expects " << name << " element type to be i32";
  }

  // A pointer does not carry capacity metadata. It is lowered as the fixed
  // 16 x i32 workspace required by PTO-ISA; allocation size remains a runtime
  // responsibility.
  if (isa<pto::PtrType>(ty)) {
    return success();
  }

  if (shape.empty()) {
    return op->emitOpError() << "expects " << name << " to have rank >= 1";
  }
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic && dim <= 0) {
      return op->emitOpError() << "expects " << name << " shape to be positive";
    }
  }

  constexpr int64_t kMinWorkspaceElements = 16;
  if (!llvm::is_contained(shape, ShapedType::kDynamic)) {
    int64_t staticCapacity = 1;
    for (int64_t dim : shape) {
      int64_t product = 0;
      if (llvm::MulOverflow(staticCapacity, dim, product)) {
        staticCapacity = std::numeric_limits<int64_t>::max();
        break;
      }
      staticCapacity = product;
    }
    if (staticCapacity < kMinWorkspaceElements) {
      return op->emitOpError()
             << "expects " << name << " to contain at least "
             << kMinWorkspaceElements
             << " i32 elements (64 bytes), but static capacity is "
             << staticCapacity;
    }
  }

  return success();
}

LogicalResult SyncAllOp::verify() {
  bool hasGm = static_cast<bool>(getGmWorkspace());
  auto mode = getMode().getValue();

  if (mode == pto::SyncAllMode::Hard) {
    if (hasGm || getUsedCores()) {
      return emitOpError(
          "expects hard syncall to have no gm_workspace or used_cores");
    }
    return success();
  }

  if (!hasGm) {
    return emitOpError("expects soft syncall to provide gm_workspace");
  }
  if (failed(verifySyncAllGmWorkspace(getOperation(), getGmWorkspace(),
                                      "gm_workspace"))) {
    return failure();
  }

  return success();
}

LogicalResult TBroadcastOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getSrc().getType() != getGroup().front().getType()) {
    return emitOpError("expects src type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src");
  }
  return success();
}

LogicalResult CommTGatherOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects dst element type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getDst().getType())) {
    return emitOpError("expects staging tile element type to match dst");
  }
  return success();
}

LogicalResult CommTScatterOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getSrc(), "src")) ||
      failed(verifyCommStagingTileLike(*this, getPing(), "ping")) ||
      failed(verifyCommPingPongSameType(*this, getPing(), getPong(), "ping",
                                        "pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getSrc().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects src element type to match group member type");
  }
  if (getElemTy(getPing().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError("expects staging tile element type to match src");
  }
  return success();
}

LogicalResult TReduceOp::verify() {
  if (failed(verifyCommGlobalLike(*this, getDst(), "dst")) ||
      failed(verifyCommStagingTileLike(*this, getAcc(), "acc")) ||
      failed(verifyCommStagingTileLike(*this, getRecvPing(), "recv_ping")) ||
      failed(verifyCommPingPongSameType(*this, getRecvPing(), getRecvPong(),
                                        "recv_ping", "recv_pong")) ||
      failed(verifyCommGlobalGroup(*this, getGroup(), "group"))) {
    return failure();
  }
  if (getRoot() >= static_cast<uint32_t>(getGroup().size())) {
    return emitOpError("expects root to index into group operands");
  }
  if (getElemTy(getDst().getType()) != getElemTy(getGroup().front().getType())) {
    return emitOpError("expects dst element type to match group member type");
  }
  if (getAcc().getType() != getRecvPing().getType()) {
    return emitOpError("expects acc and recv_ping to have identical types");
  }
  if (getElemTy(getAcc().getType()) != getElemTy(getDst().getType())) {
    return emitOpError("expects accumulator/receive tiles to match dst element type");
  }
  return success();
}

LogicalResult AicInitializePipeOp::verify() {
  if (failed(verifyFrontendInitCommon(*this, FunctionKernelKind::Cube, "cube"))) {
    return failure();
  }

  auto accPushEpilogue = getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success();
  }

  auto peerConsumerInitOr = lookupFixpipePeerConsumerInit(*this);
  if (failed(peerConsumerInitOr)) {
    return emitOpError()
           << "expects peer consumer function to contain a matching "
              "aiv_initialize_pipe with the same consumer buffer contract";
  }

  Operation *peerConsumerInit = *peerConsumerInitOr;
  auto peerAccPushEpilogue = getAccPushEpilogueFromInitOp(peerConsumerInit);
  if (!peerAccPushEpilogue) {
    return emitOpError()
           << "expects peer consumer pipe init to also have "
              "'acc_push_epilogue' for fixpipe contract consistency";
  }

  if (peerAccPushEpilogue.getLayout() != accPushEpilogue.getLayout()) {
    return emitOpError()
           << "expects acc_push_epilogue.layout to match peer consumer "
           << "(producer has " << stringifyFixpipeLayout(accPushEpilogue.getLayout())
           << ", consumer has " << stringifyFixpipeLayout(peerAccPushEpilogue.getLayout())
           << ")";
  }
  if (peerAccPushEpilogue.getQuant() != accPushEpilogue.getQuant()) {
    return emitOpError()
           << "expects acc_push_epilogue.quant to match peer consumer "
           << "(producer has " << stringifyFixpipeQuant(accPushEpilogue.getQuant())
           << ", consumer has " << stringifyFixpipeQuant(peerAccPushEpilogue.getQuant())
           << ")";
  }
  if (peerAccPushEpilogue.getRelu() != accPushEpilogue.getRelu()) {
    return emitOpError()
           << "expects acc_push_epilogue.relu to match peer consumer "
           << "(producer has " << stringifyFixpipeRelu(accPushEpilogue.getRelu())
           << ", consumer has " << stringifyFixpipeRelu(peerAccPushEpilogue.getRelu())
           << ")";
  }

  return success();
}

LogicalResult AivInitializePipeOp::verify() {
  if (failed(verifyFrontendInitCommon(*this, FunctionKernelKind::Vector, "vector"))) {
    return failure();
  }

  // Rule 3 & 22: Peer fixpipe contract verification
  auto accPushEpilogue = getAccPushEpilogueAttr();
  if (accPushEpilogue) {
    // This is a consumer-side fixpipe pipe
    // Need to find the corresponding producer-side init and verify contract match

    // Trace c2v_consumer_buf to reserve_buffer or import_reserved_buffer
    if (!getC2vConsumerBuf()) {
      return emitOpError(
          "expects fixpipe consumer pipe to have 'c2v_consumer_buf'");
    }

    Value c2vBuf = getC2vConsumerBuf();
    Operation *bufDefOp = c2vBuf.getDefiningOp();

    func::FuncOp peerProducerFunc;
    auto currentConsumerFunc = getOperation()->getParentOfType<func::FuncOp>();
    if (!currentConsumerFunc) {
      return emitOpError("must be nested under a func.func");
    }
    StringRef bufferName;

    if (auto importOp = dyn_cast_or_null<ImportReservedBufferOp>(bufDefOp)) {
      return emitOpError(
          "expects consumer-side fixpipe pipe to use reserve_buffer, not import_reserved_buffer");
    } else if (auto reserveOp = dyn_cast_or_null<ReserveBufferOp>(bufDefOp)) {
      bufferName = reserveOp.getName();

      ModuleOp moduleOp = currentConsumerFunc->getParentOfType<ModuleOp>();
      if (!moduleOp) {
        return emitOpError("must be nested under a module for fixpipe contract verification");
      }

      ImportReservedBufferOp matchedImport;
      unsigned matchedImportCount = 0;
      moduleOp.walk([&](ImportReservedBufferOp candidateImport) {
        if (candidateImport.getName() != bufferName) {
          return WalkResult::advance();
        }
        auto peerConsumerFunc =
            lookupPeerFuncAcrossContainer(candidateImport.getOperation(),
                                          candidateImport.getPeerFuncAttr());
        if (peerConsumerFunc != currentConsumerFunc) {
          return WalkResult::advance();
        }
        matchedImport = candidateImport;
        ++matchedImportCount;
        return WalkResult::advance();
      });

      if (matchedImportCount == 0) {
        return emitOpError()
               << "cannot find peer import_reserved_buffer for consumer buffer '"
               << bufferName << "'";
      }
      if (matchedImportCount > 1) {
        return emitOpError()
               << "finds multiple peer import_reserved_buffer ops for consumer buffer '"
               << bufferName << "'";
      }

      peerProducerFunc = matchedImport->getParentOfType<func::FuncOp>();
    } else {
      return emitOpError(
          "expects fixpipe pipe 'c2v_consumer_buf' to trace to reserve_buffer or "
          "import_reserved_buffer for peer contract verification");
    }

    if (!peerProducerFunc) {
      return emitOpError("cannot find peer producer function for fixpipe contract verification");
    }

    auto peerProducerInitOr = lookupFixpipePeerProducerInit(
        *this, peerProducerFunc, bufferName, currentConsumerFunc);
    if (failed(peerProducerInitOr)) {
      return failure();
    }

    Operation *peerProducerInitOp = *peerProducerInitOr;
    auto peerProducerFrontendInit =
        dyn_cast<AicInitializePipeOp>(peerProducerInitOp);
    std::optional<uint32_t> peerProducerId;
    if (peerProducerFrontendInit) {
      if (!peerProducerFrontendInit.getC2vConsumerBuf()) {
        return emitOpError()
               << "expects peer producer aic_initialize_pipe to have 'c2v_consumer_buf'";
      }

      Operation *peerBufDefOp =
          peerProducerFrontendInit.getC2vConsumerBuf().getDefiningOp();
      auto peerImportOp = dyn_cast_or_null<ImportReservedBufferOp>(peerBufDefOp);
      if (!peerImportOp) {
        return emitOpError()
               << "expects peer producer aic_initialize_pipe to use import_reserved_buffer "
               << "for c2v_consumer_buf";
      }

      auto peerConsumerFunc =
          lookupPeerFuncAcrossContainer(peerImportOp.getOperation(),
                                        peerImportOp.getPeerFuncAttr());
      if (peerImportOp.getName() != bufferName ||
          peerConsumerFunc != currentConsumerFunc) {
        return emitOpError()
               << "cannot find matching producer aic_initialize_pipe for buffer '"
               << bufferName << "' in peer function";
      }
      peerProducerId = peerProducerFrontendInit.getId();
    } else if (isa<InitializeL2LPipeOp, InitializeL2G2LPipeOp>(
                   peerProducerInitOp)) {
      auto frontendIdAttr =
          peerProducerInitOp->getAttrOfType<IntegerAttr>(kFrontendPipeIdAttrName);
      if (!frontendIdAttr) {
        return emitOpError()
               << "expects lowered peer producer fixpipe pipe to retain "
               << kFrontendPipeIdAttrName;
      }
      peerProducerId = static_cast<uint32_t>(frontendIdAttr.getInt());
    }
    if (!peerProducerId) {
      return emitOpError()
             << "expects peer producer fixpipe contract to resolve to "
                "frontend or lowered aic_initialize_pipe";
    }

    // Verify peer producer also has acc_push_epilogue
    auto peerAccPushEpilogue = getAccPushEpilogueFromInitOp(peerProducerInitOp);
    if (!peerAccPushEpilogue) {
      return emitOpError()
             << "expects peer producer pipe init to also have "
             << "'acc_push_epilogue' for fixpipe contract consistency";
    }

    // Verify layout/quant/relu match
    if (peerAccPushEpilogue.getLayout() != accPushEpilogue.getLayout()) {
      return emitOpError()
             << "expects acc_push_epilogue.layout to match peer producer "
             << "(consumer has " << stringifyFixpipeLayout(accPushEpilogue.getLayout())
             << ", producer has " << stringifyFixpipeLayout(peerAccPushEpilogue.getLayout())
             << ")";
    }

    if (peerAccPushEpilogue.getQuant() != accPushEpilogue.getQuant()) {
      return emitOpError()
             << "expects acc_push_epilogue.quant to match peer producer "
             << "(consumer has " << stringifyFixpipeQuant(accPushEpilogue.getQuant())
             << ", producer has " << stringifyFixpipeQuant(peerAccPushEpilogue.getQuant())
             << ")";
    }

    if (peerAccPushEpilogue.getRelu() != accPushEpilogue.getRelu()) {
      return emitOpError()
             << "expects acc_push_epilogue.relu to match peer producer "
             << "(consumer has " << stringifyFixpipeRelu(accPushEpilogue.getRelu())
             << ", producer has " << stringifyFixpipeRelu(peerAccPushEpilogue.getRelu())
             << ")";
    }

    SmallVector<TPopFromAicOp> matchingPops;
    currentConsumerFunc.walk([&](TPopFromAicOp pop) {
      if (pop.getId() == getId()) {
        matchingPops.push_back(pop);
      }
      return WalkResult::advance();
    });

    if (matchingPops.empty()) {
      return emitOpError()
             << "expects at least one tpop_from_aic for fixpipe pipe id = "
             << getId() << " to resolve the consumer entry type";
    }

    Type resolvedConsumerTileType = matchingPops.front().getTile().getType();
    for (TPopFromAicOp pop : llvm::drop_begin(matchingPops)) {
      if (pop.getTile().getType() != resolvedConsumerTileType) {
        return emitOpError()
               << "expects all tpop_from_aic results for fixpipe pipe id = "
               << getId() << " to use the same tile type";
      }
    }

    auto resolvedConsumerTileTy =
        dyn_cast<pto::TileBufType>(resolvedConsumerTileType);
    if (!resolvedConsumerTileTy) {
      return emitOpError()
             << "expects fixpipe consumer tpop result to be !pto.tile_buf";
    }

    Type resolvedConsumerElemTy = resolvedConsumerTileTy.getElementType();
    auto quant = accPushEpilogue.getQuant();
    if (!matchesFixpipeConsumerElementType(quant, resolvedConsumerElemTy)) {
      return emitOpError()
             << "expects consumer element type to match acc_push_epilogue.quant "
             << stringifyFixpipeQuant(quant);
    }
    if (!matchesFixpipeConsumerLayout(accPushEpilogue.getLayout(),
                                      resolvedConsumerTileTy)) {
      return emitOpError()
             << "expects consumer tile layout to match acc_push_epilogue.layout "
             << stringifyFixpipeLayout(accPushEpilogue.getLayout());
    }

    auto getInitSlotSize = [](Operation *initOp) -> std::optional<uint32_t> {
      if (auto aicInit = dyn_cast_or_null<AicInitializePipeOp>(initOp)) {
        return aicInit.getSlotSize();
      }
      if (auto aivInit = dyn_cast_or_null<AivInitializePipeOp>(initOp)) {
        return aivInit.getSlotSize();
      }
      if (auto l2lInit = dyn_cast_or_null<InitializeL2LPipeOp>(initOp)) {
        return l2lInit.getSlotSize();
      }
      if (auto l2g2lInit = dyn_cast_or_null<InitializeL2G2LPipeOp>(initOp)) {
        return l2g2lInit.getSlotSize();
      }
      return std::nullopt;
    };

    if (auto requiredSlotBytes = getStaticTileByteSize(resolvedConsumerTileTy)) {
      if (static_cast<uint64_t>(getSlotSize()) < *requiredSlotBytes) {
        return emitOpError()
               << "expects consumer-side fixpipe slot_size to be at least "
               << *requiredSlotBytes
               << " bytes for the resolved post-fixpipe consumer entry";
      }
      auto peerProducerSlotSize = getInitSlotSize(peerProducerInitOp);
      if (peerProducerSlotSize &&
          static_cast<uint64_t>(*peerProducerSlotSize) < *requiredSlotBytes) {
        return emitOpError()
               << "expects peer producer fixpipe slot_size to be at least "
               << *requiredSlotBytes
               << " bytes for the resolved post-fixpipe consumer entry";
      }
    }

    bool producerConsumerTypeMismatch = false;
    peerProducerFunc.walk([&](TPushToAivOp push) {
      if (push.getId() != *peerProducerId) {
        return WalkResult::advance();
      }
      auto srcTileTy = dyn_cast<pto::TileBufType>(push.getTile().getType());
      if (!srcTileTy) {
        return WalkResult::advance();
      }
      if (!matchesFixpipeProducerAndConsumerTypes(
              quant, srcTileTy.getElementType(), resolvedConsumerElemTy)) {
        producerConsumerTypeMismatch = true;
        emitOpError()
            << "expects producer source element type and consumer tpop result "
               "element type to satisfy acc_push_epilogue.quant "
            << stringifyFixpipeQuant(quant);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (producerConsumerTypeMismatch) {
      return failure();
    }
  }

  return success();
}

LogicalResult TAllocToAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
        if (aicInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TALLOC to have split = 0");
          }
        }
      }
    }
  }

  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TAllocToAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                              getEntry().getType());
}

LogicalResult TPushToAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                  getTile().getType()))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }

  // Fixpipe validation
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  auto aicInit = dyn_cast<AicInitializePipeOp>(initOp);
  if (!aicInit) {
    return success(); // Not an AIC init, no fixpipe check needed
  }

  auto accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return success(); // No fixpipe config, normal pipe
  }

  // Rule 9: Source tile must be acc tile
  auto tileTy = dyn_cast<pto::TileBufType>(getTile().getType());
  if (!tileTy) {
    return emitOpError(
        "expects fixpipe TPUSH source tile to be a tile type");
  }
  auto tileSpace = getPTOMemorySpaceEnum(tileTy);
  if (!tileSpace || *tileSpace != pto::AddressSpace::ACC) {
    return emitOpError("expects fixpipe TPUSH source tile to use loc=acc");
  }

  // Rule 2: split must be 0 for fixpipe
  if (getSplit() != 0) {
    return emitOpError(
        "expects fixpipe TPUSH to have split = 0");
  }

  // Rule 10: Check source element type matches quant mode
  Type elemTy = tileTy.getElementType();
  auto quant = accPushEpilogue.getQuant();

  bool srcTypeValid = true;
  if (quant == pto::FixpipeQuant::NoConvert) {
    // no_convert requires f32 or i32, and consumer type must match
    if (!elemTy.isF32() && !elemTy.isInteger(32)) {
      srcTypeValid = false;
    }
  } else if (quant == pto::FixpipeQuant::F32F16 ||
             quant == pto::FixpipeQuant::F32BF16 ||
             quant == pto::FixpipeQuant::QF322B8PreScalar ||
             quant == pto::FixpipeQuant::QF322B8PreVec ||
             quant == pto::FixpipeQuant::QF322F16PreScalar ||
             quant == pto::FixpipeQuant::QF322BF16PreScalar ||
             quant == pto::FixpipeQuant::QF322HIF8PreScalar ||
             quant == pto::FixpipeQuant::QF322FP8PreScalar) {
    // f32-based quant modes
    if (!elemTy.isF32()) {
      srcTypeValid = false;
    }
  } else if (quant == pto::FixpipeQuant::REQ8Scalar ||
             quant == pto::FixpipeQuant::REQ8Vec ||
             quant == pto::FixpipeQuant::DEQF16Scalar ||
             quant == pto::FixpipeQuant::DEQF16Vec ||
             quant == pto::FixpipeQuant::QS322BF16PreScalar ||
             quant == pto::FixpipeQuant::QS322BF16PreVec) {
    // i32-based quant modes
    if (!elemTy.isInteger(32)) {
      srcTypeValid = false;
    }
  }

  if (!srcTypeValid) {
    return emitOpError()
           << "expects fixpipe TPUSH source element type to match "
           << "acc_push_epilogue.quant mode requirements";
  }

  // Rule 16-17: Check for required set_quant_scalar/vector ops
  bool isScalarQuant = isScalarFixpipeQuant(quant);
  bool isVectorQuant = isVectorFixpipeQuant(quant);

  if (isScalarQuant || isVectorQuant) {
    Block *tpushBlock = getOperation()->getBlock();
    bool foundQuantConfig = false;
    for (Operation &op : tpushBlock->getOperations()) {
      if (&op == getOperation()) {
        break;
      }

      if (isScalarQuant) {
        if (auto setQuant = dyn_cast<SetQuantScalarOp>(&op)) {
          if (setQuant.getId() == getId()) {
            foundQuantConfig = true;
          }
        }
      } else if (isVectorQuant) {
        if (auto setQuant = dyn_cast<SetQuantVectorOp>(&op)) {
          if (setQuant.getId() == getId()) {
            foundQuantConfig = true;
          }
        }
      }
    }

    if (!foundQuantConfig) {
      if (isScalarQuant) {
        return emitOpError()
               << "expects a preceding pto.set_quant_scalar with id = "
               << getId() << " in the same block";
      }
      return emitOpError()
             << "expects a preceding pto.set_quant_vector with id = "
             << getId() << " in the same block";
    }
  }

  return success();
}

LogicalResult TPushToAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                  getTile().getType()))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  return verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                    getSplit(), getTile().getType());
}

LogicalResult TPopFromAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendPopOp(*this, FunctionKernelKind::Vector, "vector",
                                 /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  return verifyFixpipeConsumerType(getOperation(), getId(), getTile().getType());
}

LogicalResult TPopFromAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  return verifyFrontendPopOp(*this, FunctionKernelKind::Cube, "cube",
                             /*expectC2V=*/false);
}

LogicalResult TFreeFromAicOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Vector,
                                   "vector", getId(), getSplit(),
                                   /*expectC2V=*/true))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/true))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
        if (aivInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TFREE to have split = 0");
          }
        }
      }
    }
  }

  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry()) {
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  }
  return success();
}

LogicalResult TFreeFromAivOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  if (failed(verifyFrontendSplitOp(getOperation(), FunctionKernelKind::Cube,
                                   "cube", getId(), getSplit(),
                                   /*expectC2V=*/false))) {
    return failure();
  }
  if (failed(verifyFrontendDataOpDirection(getOperation(), getId(),
                                           /*expectC2V=*/false))) {
    return failure();
  }

  // Fixpipe validation: check if this is a fixpipe pipe
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (funcOp) {
    auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
    if (succeeded(initOr)) {
      Operation *initOp = *initOr;
      if (auto aivInit = dyn_cast<AivInitializePipeOp>(initOp)) {
        if (aivInit.getAccPushEpilogueAttr()) {
          // Rule 2: fixpipe requires split = 0
          if (getSplit() != 0) {
            return emitOpError("expects fixpipe TFREE to have split = 0");
          }
        }
      }
    }
  }

  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry()) {
    return verifyFrontendTensorEntryMatchesInit(getOperation(), getId(),
                                                getEntry().getType());
  }
  return success();
}

LogicalResult SetQuantScalarOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  // Look up the referenced pipe
  auto initOr =
      lookupFrontendOrLoweredInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  pto::AccPushEpilogueAttr accPushEpilogue;
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  } else if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    accPushEpilogue = l2lInit.getAccPushEpilogueAttr();
  } else if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    accPushEpilogue = l2g2lInit.getAccPushEpilogueAttr();
  } else {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference an aic_initialize_pipe or lowered producer pipe";
  }

  if (!accPushEpilogue) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a fixpipe pipe (with acc_push_epilogue)";
  }

  // Check that quant mode is scalar
  auto quant = accPushEpilogue.getQuant();
  bool isScalarQuant = isScalarFixpipeQuant(quant);

  if (!isScalarQuant) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with scalar quantization mode, but found non-scalar mode";
  }

  // Verify scale operand is f32 to match SET_QUANT_SCALAR(float)
  if (!getScale().getType().isF32()) {
    return emitOpError("expects 'scale' to be f32");
  }

  return success();
}

LogicalResult SetQuantVectorOp::verify() {
  if (failed(verifyNoUnpublishedFixpipeFrontendAttrs(getOperation()))) {
    return failure();
  }
  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return emitOpError("must be nested under a func.func");
  }

  // Look up the referenced pipe
  auto initOr =
      lookupFrontendOrLoweredInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return failure();
  }

  Operation *initOp = *initOr;
  pto::AccPushEpilogueAttr accPushEpilogue;
  if (auto aicInit = dyn_cast<AicInitializePipeOp>(initOp)) {
    accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  } else if (auto l2lInit = dyn_cast<InitializeL2LPipeOp>(initOp)) {
    accPushEpilogue = l2lInit.getAccPushEpilogueAttr();
  } else if (auto l2g2lInit = dyn_cast<InitializeL2G2LPipeOp>(initOp)) {
    accPushEpilogue = l2g2lInit.getAccPushEpilogueAttr();
  } else {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference an aic_initialize_pipe or lowered producer pipe";
  }

  if (!accPushEpilogue) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a fixpipe pipe (with acc_push_epilogue)";
  }

  // Check that quant mode is vector
  auto quant = accPushEpilogue.getQuant();
  bool isVectorQuant = isVectorFixpipeQuant(quant);

  if (!isVectorQuant) {
    return emitOpError()
           << "expects 'id' = " << getId()
           << " to reference a pipe with vector quantization mode, but found non-vector mode";
  }

  Type scalingTy = getScalingTile().getType();
  if (!isa<pto::TileBufType>(scalingTy)) {
    return emitOpError("expects 'scaling_tile' to be a tile type");
  }
  auto scalingSpace = getPTOMemorySpaceEnum(scalingTy);
  if (!scalingSpace || *scalingSpace != pto::AddressSpace::SCALING) {
    return emitOpError("expects 'scaling_tile' to use loc=scaling");
  }
  Type scalingElemTy = getElemTy(scalingTy);
  PTOArch arch = getTargetArch(getOperation());
  if (!isFixpipeQuantPayloadElemType(scalingElemTy, arch)) {
    if (arch == PTOArch::A3) {
      return emitOpError(
          "expects 'scaling_tile' element type to be packed i64/ui64 on A3");
    }
    return emitOpError(
        "expects 'scaling_tile' element type to be f16, bf16, or f32 on A5");
  }

  return success();
}

LogicalResult InitializeL2G2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  if (!getLocalAddr()) {
    if (getPeerLocalAddr()) {
      return emitOpError("'peer_local_addr' requires 'local_addr'");
    }
    if (getLocalSlotNumAttr()) {
      return emitOpError(
          "'local_slot_num' is only allowed when 'local_addr' is present");
    }
    return success();
  }

  if (auto localSlotNumAttr = getLocalSlotNumAttr()) {
    int32_t localSlotNum = localSlotNumAttr.getInt();
    if (localSlotNum <= 0) {
      return emitOpError("expects 'local_slot_num' to be greater than 0");
    }
    if (static_cast<uint32_t>(localSlotNum) > getSlotNum()) {
      return emitOpError(
          "expects 'local_slot_num' to be less than or equal to slot_num");
    }
  }

  if (getDirMask() == 3 && !getPeerLocalAddr()) {
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  }
  if (getDirMask() != 3 && getPeerLocalAddr()) {
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  }
  return success();
}

LogicalResult InitializeL2LPipeOp::verify() {
  if (failed(verifyPipeShape(getOperation(), getDirMask(), getSlotSize(),
                             getSlotNum(),
                             getFlagBaseAttr()
                                 ? std::optional<int32_t>(getFlagBaseAttr().getInt())
                                 : std::nullopt))) {
    return failure();
  }

  if (getDirMask() == 3 && !getPeerLocalAddr()) {
    return emitOpError("expects 'peer_local_addr' when dir_mask is 3");
  }
  if (getDirMask() != 3 && getPeerLocalAddr()) {
    return emitOpError("'peer_local_addr' is only allowed when dir_mask is 3");
  }
  return success();
}

LogicalResult TPushOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/true))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (isInsideCubeKernelOrSection(getOperation()) &&
      failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType()))) {
    return failure();
  }
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED) {
    return emitOpError("tile type must map to a supported producer pipe");
  }
  return success();
}

LogicalResult TAllocOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/true))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

LogicalResult TPopOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/false))) {
    return failure();
  }
  if (failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getTile().getType()))) {
    return failure();
  }
  if (isInsideCubeKernelOrSection(getOperation()) &&
      failed(verifyFullTileSplitParity(getOperation(), getSplit(),
                                       getTile().getType()))) {
    return failure();
  }
  if (failed(verifyAivSubblockIdOperand(getOperation(), getAivSubblockid(),
                                        getSplit(), getTile().getType()))) {
    return failure();
  }
  if (failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getTile().getType()))) {
    return failure();
  }
  if (!isa<TensorViewType>(getTile().getType()) &&
      getPipe() == pto::PIPE::PIPE_UNASSIGNED) {
    return emitOpError(
        "tile type and target arch must map to a supported consumer pipe");
  }
  return success();
}

LogicalResult TFreeOp::verify() {
  if (!isInsideSectionOrAttributedKernel(getOperation())) {
    return emitOpError("must be inside pto.section.cube/vector or a kernel_kind function");
  }
  if (failed(verifyPipeHandleProducer(getOperation(), getPipeHandle()))) {
    return failure();
  }
  if (failed(verifySplitAttr(getOperation(), getSplit()))) {
    return failure();
  }
  if (failed(verifyInternalOddSplitSupport(
          getOperation(), getPipeHandle(), getSplit(),
          /*producerSide=*/false))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyOddSplitTileEntry(getOperation(), getSplit(),
                                     getEntry().getType()))) {
    return failure();
  }
  if (getEntry() &&
      failed(verifyTensorEntryMatchesInternalPipeInit(
          getOperation(), getPipeHandle(), getEntry().getType()))) {
    return failure();
  }
  return success();
}

ParseResult TFreeOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand first;
  OpAsmParser::UnresolvedOperand pipe;
  Type firstTy;
  Type pipeTy;
  bool hasEntry = false;

  if (parser.parseLParen() || parser.parseOperand(first)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalComma())) {
    hasEntry = true;
    if (parser.parseOperand(pipe) || parser.parseColonType(firstTy) ||
        parser.parseComma() || parser.parseType(pipeTy) || parser.parseRParen()) {
      return failure();
    }
  } else {
    if (parser.parseColonType(pipeTy) || parser.parseRParen()) {
      return failure();
    }
    pipe = first;
  }

  NamedAttrList attrs;
  if (parser.parseLBrace() || parser.parseKeyword("split") ||
      parser.parseEqual()) {
    return failure();
  }
  IntegerAttr splitAttr;
  if (parser.parseAttribute(splitAttr, parser.getBuilder().getI8Type(),
                            "split", attrs) ||
      parser.parseRBrace() || parser.parseOptionalAttrDict(attrs)) {
    return failure();
  }

  result.addAttributes(attrs);
  if (hasEntry &&
      parser.resolveOperand(first, firstTy, result.operands)) {
    return failure();
  }
  if (parser.resolveOperand(pipe, pipeTy, result.operands)) {
    return failure();
  }
  return success();
}

void TFreeOp::print(OpAsmPrinter &p) {
  p << "(";
  if (getEntry()) {
    p << getEntry() << ", " << getPipeHandle() << " : "
      << getEntry().getType() << ", " << getPipeHandle().getType();
  } else {
    p << getPipeHandle() << " : " << getPipeHandle().getType();
  }
  p << ") {split = " << static_cast<int32_t>(getSplit()) << "}";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"split"});
}

static func::FuncOp getParentFunc(Operation *op) {
  return op ? op->getParentOfType<func::FuncOp>() : func::FuncOp();
}

static constexpr int64_t kSimtKeepResumeSlotLimit = 123;

static Operation *getFirstNonConstantLikeOp(Block *block) {
  if (!block) {
    return nullptr;
  }
  for (Operation &op : *block) {
    if (!op.hasTrait<OpTrait::ConstantLike>()) {
      return &op;
    }
  }
  return nullptr;
}

static bool isOpInRange(Operation *op, Operation *first, Operation *last) {
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (cur == op) {
      return true;
    }
    if (cur == last) {
      return false;
    }
  }
  return false;
}

static std::optional<unsigned> getSimtKeepResumeRegisterCount(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (intType.getWidth() <= 32) {
      return 1;
    }
    if (intType.getWidth() == 64) {
      return 2;
    }
    return std::nullopt;
  }
  if (type.isF16() || type.isBF16() || type.isF32()) {
    return 1;
  }
  return std::nullopt;
}

template <typename OpT>
static Type getSimtKeepResumeValueType(OpT op);

template <>
Type getSimtKeepResumeValueType(KeepOp op) {
  return op.getPayload().getType();
}

template <>
Type getSimtKeepResumeValueType(ResumeOp op) {
  return op.getResult().getType();
}

template <typename OpT>
static LogicalResult verifySimtKeepResumeSlotRange(OpT op) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return success();
  }
  int64_t slot = op.getSlot();
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op.emitOpError()
           << "requires slot in range [0, "
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  if (*registerCount == 2) {
    if ((slot % 2) != 0) {
      return op.emitOpError()
             << "requires an even slot for 64-bit keep/resume values";
    }
    if (slot + 1 >= kSimtKeepResumeSlotLimit) {
      return op.emitOpError()
             << "requires slot in range [0, "
             << (kSimtKeepResumeSlotLimit - 2)
             << "] for 64-bit keep/resume values";
    }
  }
  return success();
}

template <typename OpT>
static bool overlapsEarlierSimtKeepResumeSlotUse(OpT op,
                                                 SmallVectorImpl<int64_t> &used) {
  std::optional<unsigned> registerCount =
      getSimtKeepResumeRegisterCount(getSimtKeepResumeValueType(op));
  if (!registerCount) {
    return false;
  }
  int64_t slot = op.getSlot();
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    if (llvm::is_contained(used, word)) {
      return true;
    }
  }
  for (int64_t word = slot; word < slot + *registerCount; ++word) {
    used.push_back(word);
  }
  return false;
}

static LogicalResult verifyUniqueResumeGroupSlots(ResumeOp current,
                                                  Operation *first) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto resume = dyn_cast<ResumeOp>(cur);
    if (!resume) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(resume, slots) &&
        resume.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << resume.getSlot()
             << " in the SIMT resume prologue group";
    }
  }
  return success();
}

static LogicalResult verifyUniqueKeepGroupSlots(KeepOp current,
                                                Operation *first,
                                                Operation *last) {
  SmallVector<int64_t, 4> slots;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    auto keep = dyn_cast<KeepOp>(cur);
    if (!keep) {
      break;
    }
    if (overlapsEarlierSimtKeepResumeSlotUse(keep, slots) &&
        keep.getOperation() == current.getOperation()) {
      return current.emitOpError()
             << "duplicates an earlier slot " << keep.getSlot()
             << " in the SIMT keep epilogue group";
    }
    if (cur == last) {
      break;
    }
  }
  return success();
}

static bool isSupportedSimtKeepResumeType(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() <= 64;
  }
  return type.isF16() || type.isBF16() || type.isF32();
}

static bool isInsideSimtExecutionScope(Operation *op) {
  func::FuncOp func = getParentFunc(op);
  return (func && func->hasAttr(pto::kPTOSimtEntryAttrName)) ||
         op->getParentOfType<pto::SectionSimtOp>();
}

static LogicalResult verifyInsideSimtExecutionScope(Operation *op) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName
           << "' or inside pto.section.simt";
  }
  return success();
}

static LogicalResult verifySimtKeepResumeCommon(Operation *op, int64_t slot) {
  if (!isInsideSimtExecutionScope(op)) {
    return op->emitOpError("must appear inside a function marked with '")
           << pto::kPTOSimtEntryAttrName << "' or inside pto.section.simt";
  }
  if (slot < 0 || slot >= kSimtKeepResumeSlotLimit) {
    return op->emitOpError("requires slot in range [0, ")
           << (kSimtKeepResumeSlotLimit - 1) << "]";
  }
  return success();
}

LogicalResult SyncthreadsOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

LogicalResult ThreadfenceBlockOp::verify() {
  return verifyInsideSimtExecutionScope(getOperation());
}

void SyncthreadsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

void ThreadfenceBlockOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult KeepOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getPayload().getType())) {
    return emitOpError()
           << "supports integer scalar payloads up to 64 bits and "
              "f16/bf16/f32 payloads";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }

  Block *block = getOperation()->getBlock();
  bool insideSection =
      getOperation()->getParentOfType<SectionSimtOp>() != nullptr;
  Operation *lastPayloadOp = nullptr;
  if (insideSection) {
    if (!block->empty()) {
      lastPayloadOp = &block->back();
    }
  } else {
    Operation *terminator = block->getTerminator();
    if (isa<func::ReturnOp>(terminator)) {
      lastPayloadOp = terminator->getPrevNode();
    }
  }
  if (!lastPayloadOp) {
    return emitOpError(
        "must be placed in the SIMT epilogue before func.return or the end "
        "of pto.section.simt");
  }

  Operation *cur = lastPayloadOp;
  while (cur && isa<SyncthreadsOp>(cur)) {
    cur = cur->getPrevNode();
  }
  Operation *lastKeep = cur;
  if (!lastKeep || !isa<KeepOp>(lastKeep)) {
    return emitOpError()
           << "must be placed in the SIMT epilogue before func.return; only "
              "'pto.syncthreads' may appear between the final 'pto.keep' group "
              "and func.return or the end of pto.section.simt";
  }

  Operation *firstKeep = lastKeep;
  while (Operation *prev = firstKeep->getPrevNode()) {
    if (!isa<KeepOp>(prev)) {
      break;
    }
    firstKeep = prev;
  }
  if (!isOpInRange(getOperation(), firstKeep, lastKeep)) {
    return emitOpError()
           << "must be in the contiguous SIMT keep epilogue group immediately "
              "before optional 'pto.syncthreads' and func.return or the end "
              "of pto.section.simt";
  }
  if (failed(verifyUniqueKeepGroupSlots(*this, firstKeep, lastKeep))) {
    return failure();
  }
  return success();
}

LogicalResult ResumeOp::verify() {
  if (failed(verifySimtKeepResumeCommon(getOperation(), getSlot()))) {
    return failure();
  }
  if (!isSupportedSimtKeepResumeType(getResult().getType())) {
    return emitOpError()
           << "supports integer scalar results up to 64 bits and "
              "f16/bf16/f32 results";
  }
  if (failed(verifySimtKeepResumeSlotRange(*this))) {
    return failure();
  }
  Block *block = getOperation()->getBlock();
  Operation *first = getFirstNonConstantLikeOp(block);
  if (!first || !isa<ResumeOp>(first)) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }

  bool found = false;
  for (Operation *cur = first; cur; cur = cur->getNextNode()) {
    if (!isa<ResumeOp>(cur)) {
      break;
    }
    if (cur == getOperation()) {
      found = true;
      break;
    }
  }
  if (!found) {
    return emitOpError()
           << "must be in the contiguous SIMT resume prologue group after "
              "constant-like operations";
  }
  if (failed(verifyUniqueResumeGroupSlots(*this, first))) {
    return failure();
  }
  return success();
}

void BuildAsyncSessionOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScratchMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getWorkspaceMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TGetAsyncOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPutOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TGetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
}

void TNotifyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getValueMutable(), MemoryEffects::Read::get());
}

void TWaitOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
}

void TTestOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSignalMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getCmpValueMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TBroadcastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Write::get());
  }
}

void CommTGatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Read::get());
  }
}

void CommTScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPingMutable(), MemoryEffects::Write::get());
  if (getPong()) {
    auto pongRange = getPongMutable();
    if (auto it = pongRange.begin(); it != pongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Write::get());
  }
}

void TReduceOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getAccMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRecvPingMutable(), MemoryEffects::Write::get());
  if (getRecvPong()) {
    auto recvPongRange = getRecvPongMutable();
    if (auto it = recvPongRange.begin(); it != recvPongRange.end()) {
      addEffect(effects, &*it, MemoryEffects::Read::get());
      addEffect(effects, &*it, MemoryEffects::Write::get());
    }
  }
  for (OpOperand &operand : getGroupMutable()) {
    addEffect(effects, &operand, MemoryEffects::Read::get());
  }
}

void WaitAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TestAsyncEventOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEventMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getSessionMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2G2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getGmAddrMutable(), MemoryEffects::Read::get());
  auto localAddr = getLocalAddrMutable();
  if (!localAddr.empty()) {
    addEffect(effects, &*localAddr.begin(), MemoryEffects::Read::get());
  }
  auto peerLocalAddr = getPeerLocalAddrMutable();
  if (!peerLocalAddr.empty()) {
    addEffect(effects, &*peerLocalAddr.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void InitializeL2LPipeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getLocalAddrMutable(), MemoryEffects::Read::get());
  addEffect(effects, getOperation()->getOpResult(0), MemoryEffects::Write::get());
}

void TPushOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());

  if (auto pipeId = getFrontendPipeIdFromHandle(getPipeHandle())) {
    auto accPushEpilogue =
        getAccPushEpilogueFromInitOp(getPipeHandle().getDefiningOp());
    if (accPushEpilogue &&
        (isScalarFixpipeQuant(accPushEpilogue.getQuant()) ||
         isVectorFixpipeQuant(accPushEpilogue.getQuant()))) {
      effects.emplace_back(MemoryEffects::Read::get(),
                           getFixpipeQuantStateIdAttr(getOperation(), *pipeId),
                           FixpipeQuantStateResource::get());
    }
  }
}

void TPushToAivOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getTileMutable(), MemoryEffects::Read::get());

  auto funcOp = getOperation()->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return;
  }

  auto initOr = lookupFrontendInitOpById(getOperation(), funcOp, getId());
  if (failed(initOr)) {
    return;
  }

  auto aicInit = dyn_cast<AicInitializePipeOp>(*initOr);
  if (!aicInit) {
    return;
  }

  auto accPushEpilogue = aicInit.getAccPushEpilogueAttr();
  if (!accPushEpilogue) {
    return;
  }

  auto quant = accPushEpilogue.getQuant();
  if (isScalarFixpipeQuant(quant) || isVectorFixpipeQuant(quant)) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         getFixpipeQuantStateIdAttr(getOperation(), getId()),
                         FixpipeQuantStateResource::get());
  }
}

void TAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getEntryMutable(), MemoryEffects::Write::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void TPopOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
  auto aivSubblockId = getAivSubblockidMutable();
  if (!aivSubblockId.empty()) {
    addEffect(effects, &*aivSubblockId.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getTileMutable(), MemoryEffects::Write::get());
}

void TFreeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto entry = getEntryMutable();
  if (!entry.empty()) {
    addEffect(effects, &*entry.begin(), MemoryEffects::Read::get());
  }
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getPipeHandleMutable(), MemoryEffects::Write::get());
}

void SetQuantScalarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScaleMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

void SetQuantVectorOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  addEffect(effects, &getScalingTileMutable(), MemoryEffects::Read::get());
  effects.emplace_back(MemoryEffects::Write::get(),
                       getFixpipeQuantStateIdAttr(getOperation(), getId()),
                       FixpipeQuantStateResource::get());
}

static constexpr const char kConvertRoundingKeywords[] = "r/a/f/c/z/o/h";

static ParseResult parseConvertRounding(OpAsmParser &parser,
                                        RoundingAttr &roundingAttr) {
  StringRef roundingKeyword;
  if (parser.parseKeyword("round") || parser.parseLParen() ||
      parser.parseKeyword(&roundingKeyword) || parser.parseRParen()) {
    return failure();
  }
  std::optional<Rounding> rounding = symbolizeRounding(roundingKeyword);
  if (!rounding) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert rounding to be one of "
           << kConvertRoundingKeywords;
  }
  roundingAttr = RoundingAttr::get(parser.getContext(), *rounding);
  return success();
}

static void printConvertRounding(OpAsmPrinter &printer, Operation *op,
                                 RoundingAttr rounding) {
  printer << "round(" << stringifyRounding(rounding.getValue()) << ")";
}

static ParseResult parseConvertSaturation(OpAsmParser &parser,
                                          SaturationAttr &saturationAttr) {
  StringRef saturationKeyword;
  if (parser.parseKeyword(&saturationKeyword)) {
    return failure();
  }
  std::optional<Saturation> saturation =
      symbolizeSaturation(saturationKeyword);
  if (!saturation) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected convert saturation to be sat or nosat";
  }
  saturationAttr = SaturationAttr::get(parser.getContext(), *saturation);
  return success();
}

static void printConvertSaturation(OpAsmPrinter &printer, Operation *op,
                                   SaturationAttr saturation) {
  printer << stringifySaturation(saturation.getValue());
}

static ParseResult parseSignedness(OpAsmParser &parser,
                                   SignednessAttr &signedness) {
  StringRef signednessKeyword;
  if (parser.parseKeyword(&signednessKeyword)) {
    return failure();
  }
  std::optional<Signedness> parsed = symbolizeSignedness(signednessKeyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected signedness to be signed or unsigned";
  }
  signedness = SignednessAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printSignedness(OpAsmPrinter &printer, Operation *op,
                            SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static OptionalParseResult parseOptionalSignedness(OpAsmParser &parser,
                                                   SignednessAttr &signedness) {
  if (succeeded(parser.parseOptionalKeyword("signed"))) {
    signedness = SignednessAttr::get(parser.getContext(), Signedness::Signed);
    return success();
  }
  if (succeeded(parser.parseOptionalKeyword("unsigned"))) {
    signedness =
        SignednessAttr::get(parser.getContext(), Signedness::Unsigned);
    return success();
  }
  return std::nullopt;
}

static void printOptionalSignedness(OpAsmPrinter &printer, Operation *op,
                                    SignednessAttr signedness) {
  printer << stringifySignedness(signedness.getValue());
}

static constexpr const char kLdL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmpref/nakeep/naclean/nadrop/idsfv/idslv/idsprs/"
    "idspref/exfv/exlv/exprs/expref";

static constexpr const char kStL2CacheKeywords[] =
    "nmfv/nmlv/nmprs/nmred/naci/napw/napi/nared/wbhfv/wbhlv/wbhprs/"
    "wbhred/wtsfv/wtslv/wtsprs/wtsred";

static ParseResult parseL1Cache(OpAsmParser &parser, L1CacheAttr &l1cache) {
  if (failed(parser.parseOptionalKeyword("l1cache"))) {
    l1cache = L1CacheAttr::get(parser.getContext(), L1Cache::Cache);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<L1Cache> parsed = symbolizeL1Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected memory l1cache to be cache or uncache";
  }
  l1cache = L1CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printL1Cache(OpAsmPrinter &printer, Operation *op,
                         L1CacheAttr l1cache) {
  if (!l1cache) {
    return;
  }
  printer << "l1cache(" << stringifyL1Cache(l1cache.getValue()) << ")";
}

static ParseResult parseLdL2Cache(OpAsmParser &parser,
                                  LdL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = LdL2CacheAttr::get(parser.getContext(), LdL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<LdL2Cache> parsed = symbolizeLdL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected load L2 cache control to be one of "
           << kLdL2CacheKeywords;
  }
  l2cache = LdL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printLdL2Cache(OpAsmPrinter &printer, Operation *op,
                           LdL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyLdL2Cache(l2cache.getValue()) << ")";
}

static ParseResult parseStL2Cache(OpAsmParser &parser,
                                  StL2CacheAttr &l2cache) {
  if (failed(parser.parseOptionalKeyword("l2cache"))) {
    l2cache = StL2CacheAttr::get(parser.getContext(), StL2Cache::NMFV);
    return success();
  }

  StringRef keyword;
  if (parser.parseLParen() || parser.parseKeyword(&keyword) ||
      parser.parseRParen()) {
    return failure();
  }
  std::optional<StL2Cache> parsed = symbolizeStL2Cache(keyword);
  if (!parsed) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected store L2 cache control to be one of "
           << kStL2CacheKeywords;
  }
  l2cache = StL2CacheAttr::get(parser.getContext(), *parsed);
  return success();
}

static void printStL2Cache(OpAsmPrinter &printer, Operation *op,
                           StL2CacheAttr l2cache) {
  if (!l2cache) {
    return;
  }
  printer << "l2cache(" << stringifyStL2Cache(l2cache.getValue()) << ")";
}

// custom<StructPath>($path): a bracketed list of constant field indices, e.g.
// `[0, 2]`. Backs pto.struct_get / pto.struct_set so they read as `%s[0, 2]`.
static ParseResult parseStructPath(OpAsmParser &parser,
                                   DenseI64ArrayAttr &path) {
  SmallVector<int64_t> indices;
  if (parser.parseLSquare()) {
    return failure();
  }
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t idx = 0;
      if (parser.parseInteger(idx)) {
        return failure();
      }
      indices.push_back(idx);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare()) {
      return failure();
    }
  }
  path = DenseI64ArrayAttr::get(parser.getContext(), indices);
  return success();
}

static void printStructPath(OpAsmPrinter &printer, Operation *op,
                            DenseI64ArrayAttr path) {
  printer << "[";
  llvm::ArrayRef<int64_t> indices = path.asArrayRef();
  for (size_t i = 0; i < indices.size(); ++i) {
    if (i) {
      printer << ", ";
    }
    printer << indices[i];
  }
  printer << "]";
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#include "PTO/IR/VPTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"
