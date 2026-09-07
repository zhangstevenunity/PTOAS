// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VMILowerUnifiedToLegacy.cpp - Lower unified v-ops to legacy ops ----===//
//
// Lowers unified v-prefixed VMI ops to their legacy equivalents under the
// opt-in --vmi-two-stage-lowering flag.
//
// Category A — pure syntactic renames (4 ops):
//   vci             → iota
//   vinterpret_cast → bitcast
//   vsel            → select
//   vbrc            → broadcast  (skipped when num_groups is present)
//
// Category B — elementwise arithmetic / bitwise (18 ops):
//   vadd/vsub/vmul/vdiv/vmin/vmax → legacy type-specific binary op
//   vneg/vabs/vsqrt/vexp/vln/vrelu → legacy unary op
//   vand/vor/vxor/vshl/vshr/vnot → legacy bitwise op
//   vshr selects shrui for unsigned/signless elements and shrsi for
//   explicitly signed elements.
//   Mask/pmode synthesis is intentionally bypassed here so two-stage lowering
//   does not introduce select chains before layout assignment.
//
// Category C1 — compare + seed (2 ops):
//   vcmp  → cmpf/cmpi + mask_and
//   vcmps → broadcast scalar + cmpf/cmpi + mask_and
//
// Category C2 — unified type conversion (1 op):
//   vcvt → type-dispatch to extf/truncf/fptosi/sitofp/extsi/extui/trunci
//   For fp narrowing, unified saturate=SAT is normalized away because the
//   legacy truncf -> VPTO lowering already materializes saturating low-level
//   vcvt forms for supported narrowing result families.
//
// Category C3 — unified load/store (2 ops):
//   vload  → dispatch by dist_mode/group/block_stride to
//            load / deinterleave_load / group_broadcast_load{num_groups=1} / ...
//   vstore → dispatch to store / masked_store / interleave_store / group_store / ...
//   Continuous 1/2/4/8-lane values alias unit-stride
//   group_slot_load/group_store operations.
//   Skipped: dist_mode "unpack" (physical widening, no legacy equivalent).
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C4 — static mask creation (3 ops):
//   pset → create_mask(all lanes)
//   pge  → create_mask(N lanes)
//   plt  → create_mask(min(rem, L))
//
// Category C3 — unified load/store (2 ops, dispatch by dist_mode/group):
//   vload → load / deinterleave_load / group_load
//   vstore → store / masked_store / interleave_store / group_store
//
// Category C6 — unified reduce (3 ops):
//   vcadd → reduce_addf/reduce_addi or group_reduce_addf/group_reduce_addi
//   vcmax → reduce_maxf/reduce_maxi or group_reduce_maxf/group_reduce_maxi
//   vcmin → reduce_minf/reduce_mini or group_reduce_minf/group_reduce_mini
//
// Category C7 — fused multiply-add family → legacy fma (2 ops):
//   vmula → fma               (float only; mask discarded; int → skipped, no legacy int fma)
//   vaxpy → broadcast + fma   (float only)
//
// Category C8 — indexed gather/scatter → legacy gather/scatter (2 ops):
//   vgather  → gather   (pmode="zero": passthru = zero constant)
//   vscatter → scatter
//
// Category C9 — fused activation / softmax (3 ops):
//   vexpdif → kept unified for direct VMI-to-VPTO fused lowering
//   vlrelu  → maxf + minf + broadcast + mulf + addf
//   vprelu  → maxf + minf + mulf + addf
//   Lowered Category C7/C8/C9 ops bypass mask/pmode synthesis here and skip
//   pmode="merge".
//
// Category D — no legacy equivalent (explicitly skipped, 13 ops):
//   vadds/vmuls/vmaxs/vmins/vshls/vshrs
//   vaddc vaddcs vintlv vdintlv vselr vgatherb vmull
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_VMILOWERUNIFIEDTOLEGACY
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kIndexBitWidth = 64;
constexpr int64_t kSingleGroupCount = 1;
constexpr int64_t kDecimalRadix = 10;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Returns the string name of a predicate mode, defaulting to "zero".
static StringRef getPmodeOrDefault(Operation *op, StringRef attrName = "pmode") {
  if (auto attr = op->getAttrOfType<StringAttr>(attrName)) {
    return attr.getValue();
  }
  return "zero";
}

/// Returns true when the pmode on `op` is "merge" — these ops must be
/// skipped because merge semantic (inactive lane preserves OLD_DEST) cannot
/// be expressed in VMI SSA IR.
static bool hasMergePmode(Operation *op) {
  return getPmodeOrDefault(op) == "merge";
}

/// Create a zero-valued VMIConstantOp with the same type as \p vmiType.
static Value createZeroConstant(OpBuilder &builder, Location loc,
                                VMIVRegType vmiType) {
  Type elemType = vmiType.getElementType();
  int64_t laneCount = vmiType.getElementCount();
  auto shapedType = RankedTensorType::get({laneCount}, elemType);

  DenseElementsAttr zeroAttr;
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APFloat::getZero(floatType.getFloatSemantics()));
  } else if (auto intType = dyn_cast<IntegerType>(elemType)) {
    zeroAttr = DenseElementsAttr::get(
        shapedType, APInt::getZero(intType.getWidth()));
  } else {
    llvm_unreachable("unsupported VMI element type for zero constant");
  }
  return builder.create<VMIConstantOp>(loc, vmiType, zeroAttr).getResult();
}


/// Map a unified vcmp `cmp` mode to the predicate string for legacy
/// cmpf/cmpi. Float operands use ordered predicates (olt, oeq, ...);
/// integer operands select signedness from the element type.
static std::string mapCmpPredicate(StringRef cmp, Type elemType,
                                   bool isFloat) {
  if (isFloat) {
    // Already ordered/unordered — pass through.
    if (cmp.starts_with("o") || cmp.starts_with("u")) {
      return cmp.str();
    }
    return ("o" + cmp).str(); // e.g. "lt" → "olt"
  }
  // Integer.
  if (cmp.starts_with("s") || cmp.starts_with("u")) {
    return cmp.str();
  }
  // eq/ne are valid for both fp and int without prefix.
  if (cmp == "eq" || cmp == "ne") {
    return cmp.str();
  }
  auto intType = dyn_cast<IntegerType>(elemType);
  if (intType && !intType.isSigned()) {
    return ("u" + cmp).str(); // e.g. "lt" -> "ult"
  }
  return ("s" + cmp).str();   // e.g. "lt" -> "slt"
}

/// Return true when \p elemType is a floating-point type.
/// Return true for MLIR FloatType and PTO low-precision float-like types
/// (hif8, f8, f4, etc.).
static bool isFloatType(Type elemType) {
  return isa<FloatType>(elemType) || pto::isPTOLowPrecisionType(elemType);
}

/// Return the element type of a VMIVRegType.
static Type getVMIElementType(Value v) {
  return cast<VMIVRegType>(v.getType()).getElementType();
}

/// Return the storage bit width for VMI element types (float / float-like / int).
static unsigned getVMIElementBitWidth(Type type) {
  if (isa<IndexType>(type)) {
    return kIndexBitWidth;
  }
  return pto::getPTOStorageElemBitWidth(type);
}

// Inspect the source and result element types of a vcvt and classify the
// conversion direction as a floating-point or integer widening, narrowing,
// or signedness conversion.
static StringRef classifyCvtDirection(Type srcElem, Type dstElem) {
  bool srcFp = isFloatType(srcElem);
  bool dstFp = isFloatType(dstElem);
  unsigned srcBits = getVMIElementBitWidth(srcElem);
  unsigned dstBits = getVMIElementBitWidth(dstElem);

  if (srcFp && dstFp) {
    return dstBits > srcBits ? "widen_fp" : "narrow_fp";
  }
  if (srcFp && !dstFp) {
    if (auto intTy = dyn_cast<IntegerType>(dstElem))
      return intTy.isSigned() ? "fptosi" : "fptoui";
    return "fptosi";
  }
  if (!srcFp && dstFp) {
    auto intTy = dyn_cast<IntegerType>(srcElem);
    if (!intTy || !intTy.isSigned()) {
      return "unsupported";
    }
    return "sitofp";
  }
  // int → int
  return dstBits > srcBits ? "widen_int" : "narrow_int";
}

//===----------------------------------------------------------------------===//
// Category B: binary elementwise → legacy compute, mask/pmode discarded
//===----------------------------------------------------------------------===//

/// Lower a BINARY unified op (vadd, vsub, ...) to a legacy compute op.
/// Unified mask and pmode are intentionally discarded.
///
/// \p createLegacy is a callable `(Location, Type, Value, Value) -> Value`
/// that emits the legacy binary op.
template <typename UnifiedOp>
static LogicalResult
lowerBinaryIgnoringMask(
    UnifiedOp op,
    function_ref<Value(Location, Type, Value, Value)> createLegacy) {
  if (hasMergePmode(op)) {
    return failure();
  }
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();

  Value raw = createLegacy(loc, resultType, lhs, rhs);
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

/// Lower a UNARY unified op (vneg, vabs, …) to its legacy counterpart.
template <typename UnifiedOp>
static LogicalResult
lowerMaskedUnary(UnifiedOp op, OpBuilder &builder,
                 function_ref<Value(Location, Type, Value)> createLegacy) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();

  Value raw = createLegacy(loc, resultType, source);
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C1 helpers: vcmp / vcmps
//===----------------------------------------------------------------------===//

/// Returns true if `seed` is provably an all-active mask (every lane active),
/// so `mask_and(x, seed)` is the identity and the AND can be skipped. Covers a
/// `pset` (all lanes active by definition) and a `create_mask` whose
/// active_lanes is a constant >= the mask lane count.
static bool isAllActiveSeed(Value seed) {
  Operation *def = seed.getDefiningOp();
  if (!def) {
    return false;
  }
  if (isa<VMIPsetOp>(def)) {
    return true;
  }
  if (auto cm = dyn_cast<VMICreateMaskOp>(def)) {
    auto maskTy = cast<VMIMaskType>(cm.getResult().getType());
    if (auto cst = cm.getActiveLanes().getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        return ia.getInt() >= maskTy.getElementCount();
      }
    }
  }
  return false;
}

static bool isCompactGroupCount(int64_t count) {
  return count == kSingleGroupCount || count == mlir::pto::kValue2 ||
         count == mlir::pto::kValue4 || count == mlir::pto::kValue8;
}

/// Lower vcmp to cmpf/cmpi + mask_and.
static LogicalResult lowerVCmp(VMIVcmpOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type elemType = getVMIElementType(op.getLhs());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, elemType, isFloat);

  // Build legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getLhs(), op.getRhs())
                  .getResult();
  }

  // mask_and with seed — skipped when the seed is all-active (identity AND).
  Value result = rawMask;
  if (!isAllActiveSeed(op.getSeed())) {
    result = builder
                 .create<VMIMaskAndOp>(loc, op.getResult().getType(), rawMask,
                                       op.getSeed())
                 .getResult();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmps to broadcast scalar + cmpf/cmpi + mask_and.
static LogicalResult lowerVCmps(VMIVcmpsOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type srcVmiType = op.getSrc().getType();
  Value scalar = op.getScalar();
  Type elemType = getVMIElementType(op.getSrc());
  bool isFloat = isFloatType(elemType);
  StringRef cmpMode = op.getCmp();
  std::string predicate = mapCmpPredicate(cmpMode, elemType, isFloat);

  // 1. Broadcast scalar to vector.
  Value brc = builder.create<VMIBroadcastOp>(loc, srcVmiType, scalar)
                  .getResult();

  // 2. Legacy cmpf or cmpi.
  Value rawMask;
  if (isFloat) {
    rawMask = builder
                  .create<VMICmpFOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  } else {
    rawMask = builder
                  .create<VMICmpIOp>(loc, op.getResult().getType(),
                                     builder.getStringAttr(predicate),
                                     op.getSrc(), brc)
                  .getResult();
  }

  // 3. mask_and with seed — skipped when the seed is all-active (identity AND).
  Value result = rawMask;
  if (!isAllActiveSeed(op.getSeed())) {
    result = builder
                 .create<VMIMaskAndOp>(loc, op.getResult().getType(), rawMask,
                                       op.getSeed())
                 .getResult();
  }

  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C2 helper: vcvt
//===----------------------------------------------------------------------===//

/// Lower vcvt by dispatching on src→dst element types.
static Value createCvtReplacement(VMICvtOp op, OpBuilder &builder) {
  Type sourceElement = getVMIElementType(op.getSource());
  Type resultElement = getVMIElementType(op.getResult());
  StringRef direction = classifyCvtDirection(sourceElement, resultElement);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  StringAttr saturate = op.getSaturateAttr();
  if (direction == "widen_fp") {
    return builder.create<VMIExtFOp>(loc, resultType, source).getResult();
  }
  if (direction == "narrow_fp") {
    return builder
        .create<VMITruncFOp>(loc, resultType, source, op.getRoundingAttr(),
                             saturate)
        .getResult();
  }
  if (direction == "fptosi") {
    return builder
        .create<VMIFPToSIOp>(loc, resultType, source, op.getRoundingAttr(),
                             saturate)
        .getResult();
  }
  if (direction == "fptoui") {
    return builder
        .create<VMIFPToUIOp>(loc, resultType, source, op.getRoundingAttr(),
                             saturate)
        .getResult();
  }
  if (direction == "sitofp") {
    return builder.create<VMISIToFPOp>(loc, resultType, source).getResult();
  }
  if (direction == "widen_int") {
    auto integerType = dyn_cast<IntegerType>(sourceElement);
    bool useSignedExtension = !integerType || integerType.isSigned();
    return useSignedExtension
               ? builder.create<VMIExtSIOp>(loc, resultType, source).getResult()
               : builder.create<VMIExtUIOp>(loc, resultType, source).getResult();
  }
  if (direction == "narrow_int") {
    return builder
        .create<VMITruncIOp>(loc, resultType, source, saturate)
        .getResult();
  }
  return {};
}

static LogicalResult lowerVCvt(VMICvtOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }
  Value result = createCvtReplacement(op, builder);
  if (!result) {
    return failure();
  }
  op.getResult().replaceAllUsesWith(result);
  op.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C3 helpers: vload / vstore
//===----------------------------------------------------------------------===//

static StringAttr getMaskGranularity(Type elementType, OpBuilder &builder) {
  unsigned bits = mlir::pto::kValue32;
  if (auto integerType = dyn_cast<IntegerType>(elementType)) {
    bits = integerType.getWidth();
  } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
    bits = floatType.getWidth();
  }
  StringRef granularity = bits <= mlir::pto::kValue8
                              ? "b8"
                              : bits <= mlir::pto::kValue16 ? "b16" : "b32";
  return builder.getStringAttr(granularity);
}

static Value createAllActiveMask(VMIVRegType valueType, Location loc,
                                 OpBuilder &builder) {
  auto maskType = VMIMaskType::get(
      builder.getContext(), valueType.getElementCount(),
      getMaskGranularity(valueType.getElementType(), builder),
      valueType.getLayout());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, builder.getIndexAttr(valueType.getElementCount()));
  return builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
}

static LogicalResult lowerGroupedLoad(VMIvLoadOp op, OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  int64_t numGroups = op.getGroupAttr().getInt();
  bool isBroadcast = op.getDistMode() && op.getDistMode() == "brc";
  Value replacement;
  if (isBroadcast) {
    replacement = builder
                      .create<VMIGroupBroadcastLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  } else if (resultType.getElementCount() == numGroups) {
    replacement = builder
                      .create<VMIGroupSlotLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  } else {
    replacement = builder
                      .create<VMIGroupLoadOp>(
                          op.getLoc(), resultType, op.getSource(),
                          op.getOffset(), op.getStride(), op.getGroupAttr())
                      .getResult();
  }
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerBlockStrideLoad(VMIvLoadOp op,
                                          OpBuilder &builder) {
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  Value mask = createAllActiveMask(resultType, op.getLoc(), builder);
  Value replacement =
      builder
          .create<VMIStrideLoadOp>(op.getLoc(), resultType, op.getSource(),
                                   op.getOffset(), op.getBlockStride(), mask)
          .getResult();
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerContinuousLoad(VMIvLoadOp op,
                                         OpBuilder &builder) {
  Location loc = op.getLoc();
  auto resultType = cast<VMIVRegType>(op.getResults().front().getType());
  int64_t numGroups = resultType.getElementCount();
  Value replacement;
  if (isCompactGroupCount(numGroups)) {
    Value unitStride = builder.create<arith::ConstantIndexOp>(loc, 1);
    replacement = builder
                      .create<VMIGroupSlotLoadOp>(
                          loc, resultType, op.getSource(), op.getOffset(),
                          unitStride, builder.getI64IntegerAttr(numGroups))
                      .getResult();
  } else {
    replacement = builder
                      .create<VMILoadOp>(loc, resultType, op.getSource(),
                                         op.getOffset())
                      .getResult();
  }
  op.getResults().front().replaceAllUsesWith(replacement);
  return success();
}

static LogicalResult lowerDeinterleaveLoad(VMIvLoadOp op,
                                           OpBuilder &builder) {
  auto load = builder.create<VMIDeinterleaveLoadOp>(
      op.getLoc(), op.getResults()[0].getType(), op.getResults()[1].getType(),
      op.getSource(), op.getOffset());
  op.getResults()[0].replaceAllUsesWith(load.getLow());
  op.getResults()[1].replaceAllUsesWith(load.getHigh());
  return success();
}

static LogicalResult lowerBroadcastLoad(VMIvLoadOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  Type resultType = op.getResults().front().getType();
  Value stride = builder.create<arith::ConstantOp>(
      loc, builder.getIndexType(), builder.getIndexAttr(0));
  auto load = builder.create<VMIGroupBroadcastLoadOp>(
      loc, resultType, op.getSource(), op.getOffset(), stride,
      builder.getI64IntegerAttr(1));
  op.getResults().front().replaceAllUsesWith(load.getResult());
  return success();
}

static LogicalResult lowerDistributedLoad(VMIvLoadOp op,
                                          OpBuilder &builder) {
  StringAttr modeAttr = op.getDistModeAttr();
  StringRef mode = modeAttr ? modeAttr.getValue() : "continuous";
  if (mode == "continuous") {
    return lowerContinuousLoad(op, builder);
  }
  if (mode == "dintlv") {
    return lowerDeinterleaveLoad(op, builder);
  }
  if (mode == "brc") {
    return lowerBroadcastLoad(op, builder);
  }
  return failure();
}

static LogicalResult lowerVLoad(VMIvLoadOp op, OpBuilder &builder) {
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    result = lowerGroupedLoad(op, builder);
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideLoad(op, builder);
  } else if (hasMergePmode(op)) {
    return failure();
  } else {
    result = lowerDistributedLoad(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
}

static LogicalResult lowerBlockStrideStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  auto valueType = cast<VMIVRegType>(op.getValues()[0].getType());
  Value mask = op.getMask().empty()
                   ? createAllActiveMask(valueType, op.getLoc(), builder)
                   : op.getMask()[0];
  builder.create<VMIStrideStoreOp>(
      op.getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
      op.getBlockStride(), mask);
  return success();
}

static LogicalResult lowerContinuousStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  ValueRange values = op.getValues();
  if (values.empty()) {
    return failure();
  }
  Value mask = op.getMask().empty() ? Value() : op.getMask().front();
  auto valueType = cast<VMIVRegType>(values.front().getType());
  int64_t numGroups = valueType.getElementCount();
  bool canUseGroupStore = isCompactGroupCount(numGroups) &&
                          (!mask || isAllActiveSeed(mask));
  if (canUseGroupStore) {
    Value unitStride = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    builder.create<VMIGroupStoreOp>(
        op.getLoc(), values.front(), op.getDestination(), op.getOffset(),
        unitStride, builder.getI64IntegerAttr(numGroups));
    return success();
  }
  if (mask) {
    builder.create<VMIMaskedStoreOp>(op.getLoc(), values.front(),
                                     op.getDestination(), op.getOffset(), mask);
  } else {
    builder.create<VMIStoreOp>(op.getLoc(), values.front(),
                               op.getDestination(), op.getOffset());
  }
  return success();
}

static LogicalResult lowerDistributedStore(VMIvStoreOp op,
                                           OpBuilder &builder) {
  StringAttr modeAttr = op.getDistModeAttr();
  StringRef mode = modeAttr ? modeAttr.getValue() : "continuous";
  if (mode == "continuous") {
    return lowerContinuousStore(op, builder);
  }
  if (mode == "intlv") {
    ValueRange values = op.getValues();
    if (values.size() < mlir::pto::kValue2) {
      return failure();
    }
    builder.create<VMIInterleaveStoreOp>(
        op.getLoc(), values[0], values[1], op.getDestination(), op.getOffset());
    return success();
  }
  return failure();
}

static LogicalResult lowerVStore(VMIvStoreOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }
  LogicalResult result = success();
  if (op.getGroupAttr()) {
    builder.create<VMIGroupStoreOp>(
        op.getLoc(), op.getValues()[0], op.getDestination(), op.getOffset(),
        op.getStride(), op.getGroupAttr());
  } else if (op.getBlockStride()) {
    result = lowerBlockStrideStore(op, builder);
  } else {
    result = lowerDistributedStore(op, builder);
  }
  if (failed(result)) {
    return failure();
  }
  op.erase();
  return success();
}

static LogicalResult lowerPset(VMIPsetOp op, OpBuilder &builder) {
  // If an all-active consumer (e.g. vcmp) elided its use, drop the seed
  // entirely instead of materialising a dead create_mask.
  if (op.use_empty()) {
    op->erase();
    return success();
  }
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  int64_t laneCount = maskType.getElementCount();
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(laneCount));
  Value result =
      builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower pge "PAT_VLN" → create_mask(N).
/// When {group = C} is present → create_group_mask(N, num_groups=C,
/// group_size = total_lanes / C).
static LogicalResult lowerPge(VMIPgeOp op, OpBuilder &builder) {
  StringRef pattern = op.getPattern();
  // Parse "PAT_VL<num>" or fall back to "PAT_VL16".
  int64_t numLanes = 16;
  if (pattern.starts_with("PAT_VL")) {
    StringRef numStr = pattern.drop_front(6); // strlen("PAT_VL")
    if (!numStr.empty()) {
      int64_t parsed = 0;
      for (char c : numStr) {
        if (c < '0' || c > '9') {
          break;
        }
        parsed = parsed * kDecimalRadix + (c - '0');
      }
      if (parsed > 0) {
        numLanes = parsed;
      }
    }
  }

  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getResult().getType());
  auto indexType = IndexType::get(builder.getContext());
  Value activeLanes = builder.create<arith::ConstantOp>(
      loc, indexType, builder.getIndexAttr(numLanes));

  if (auto groupAttr = op.getGroupAttr()) {
    // Grouped tail mask → create_group_mask
    int64_t numGroups = groupAttr.getInt();
    int64_t totalLanes = maskType.getElementCount();
    int64_t groupSize = totalLanes / numGroups;
    Value result = builder
                       .create<VMICreateGroupMaskOp>(
                           loc, maskType, activeLanes,
                           builder.getI64IntegerAttr(numGroups),
                           builder.getI64IntegerAttr(groupSize))
                       .getResult();
    op.getResult().replaceAllUsesWith(result);
  } else {
    Value result =
        builder.create<VMICreateMaskOp>(loc, maskType, activeLanes).getResult();
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C6 helpers: vcadd / vcmax / vcmin
//===----------------------------------------------------------------------===//

template <typename ReductionOp>
static std::optional<int64_t> getReductionNumGroups(ReductionOp op) {
  if (auto groupAttr = op.getGroupAttr()) {
    return groupAttr.getInt();
  }

  // A full reduction is one logical group. Keep the alias decision local to
  // the reduction instead of relying on a downstream store to mutate it.
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  if (!resultType.getLayoutAttr()) {
    return 1;
  }
  return std::nullopt;
}

/// Lower vcadd to legacy reduce_addf/reduce_addi or
/// group_reduce_addf/group_reduce_addi.  Always succeeds for valid input
/// (vcadd verifier guarantees reassoc for float, and group 整除 source lanes).
static LogicalResult lowerVCadd(VMIvcaddOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    // Group reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIGroupReduceAddFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups),
                                            op.getReassocAttr())
              .getResult();
    } else {
      result =
          builder
              .create<VMIGroupReduceAddIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
  } else {
    // Full reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIReduceAddFOp>(loc, resultType, source, mask,
                                       op.getReassocAttr())
              .getResult();
    } else {
      result =
          builder
              .create<VMIReduceAddIOp>(loc, resultType, source, mask)
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
  }
  op->erase();
  return success();
}

/// Lower vcmax to legacy full or grouped float/integer maximum reduction.
static LogicalResult lowerVcmax(VMIvcmaxOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    // Group reduce path
    Value result;
    if (isFloat) {
      result =
          builder
              .create<VMIGroupReduceMaxFOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
    } else {
      result =
          builder
              .create<VMIGroupReduceMaxIOp>(loc, resultType, source, mask,
                                            builder.getI64IntegerAttr(*numGroups))
              .getResult();
}
    op.getResult().replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  Value result;
  if (isFloat) {
    result = builder
                 .create<VMIReduceMaxFOp>(loc, resultType, source, mask)
                 .getResult();
  } else {
    result = builder
                 .create<VMIReduceMaxIOp>(loc, resultType, source, mask)
                 .getResult();
}
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vcmin to legacy full or grouped float/integer minimum reduction.
static LogicalResult lowerVcmin(VMIvcminOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  auto sourceType = cast<VMIVRegType>(op.getSource().getType());
  Type elemType = sourceType.getElementType();
  bool isFloat = isa<FloatType>(elemType);
  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  Value source = op.getSource();
  Value mask = op.getMask();

  if (std::optional<int64_t> numGroups = getReductionNumGroups(op)) {
    Value result;
    if (isFloat) {
      result = builder
                   .create<VMIGroupReduceMinFOp>(
                       loc, resultType, source, mask,
                       builder.getI64IntegerAttr(*numGroups))
                   .getResult();
    } else {
      result = builder
                   .create<VMIGroupReduceMinIOp>(
                       loc, resultType, source, mask,
                       builder.getI64IntegerAttr(*numGroups))
                   .getResult();
}
    op.getResult().replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  Value result;
  if (isFloat) {
    result = builder
                 .create<VMIReduceMinFOp>(loc, resultType, source, mask)
                 .getResult();
  } else {
    result = builder
                 .create<VMIReduceMinIOp>(loc, resultType, source, mask)
                 .getResult();
}
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C7 helpers: vmula / vaxpy (fused multiply-add → legacy fma)
//===----------------------------------------------------------------------===//

/// Lower vmula (acc = acc + lhs*rhs) to legacy fma (lhs*rhs + acc).
/// The mask operand (if present) is discarded — legacy fma has no mask.
/// Legacy fma is floating-point only; integer vmula has no legacy equivalent
/// and is skipped (falls through to VMIToVPTO).
static LogicalResult lowerVmula(VMIVmulaOp op, OpBuilder &builder) {
  // Legacy FMA has no predicate operand.  Keep masked vmula in the VMI
  // pipeline so VMIToVPTO can preserve its predicate and zero-mode semantics.
  if (hasMergePmode(op) || !op.getMask().empty()) {
    return failure();
  }

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType())) {
    return failure();
  }

  Location loc = op.getLoc();
  // fma computes lhs*rhs + acc, matching vmula's acc + lhs*rhs.
  Value result = builder
                     .create<VMIFmaOp>(loc, resultType, op.getLhs(), op.getRhs(),
                                       op.getAcc())
                     .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vaxpy (alpha*x + y) to broadcast(alpha) + legacy fma.
/// alpha is a scalar float, broadcast to a vector before the fma.
static LogicalResult lowerVaxpy(VMIVaxpyOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  if (!isFloatType(vmiType.getElementType())) {
    return failure();
  }

  Location loc = op.getLoc();
  Value alphaVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getAlpha())
                       .getResult();
  // fma(alpha, x, y) == alpha*x + y.
  Value raw = builder
                  .create<VMIFmaOp>(loc, resultType, alphaVec, op.getX(),
                                    op.getAcc())
                  .getResult();
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

/// Lower plt(rem:i32) -> create_mask(min(rem, L)) + arith remainder chain.
///   %act  = arith.minsi %rem, %cL         // min(rem, L)
///   %aidx = arith.index_cast %act          // i32 -> index
///   %mask = vmi.create_mask %aidx
///   %next = arith.subi %rem, %act          // rem - min(rem, L) = max(rem-L, 0)
static LogicalResult lowerPlt(VMIPltOp op, OpBuilder &builder) {
  Location loc = op.getLoc();
  auto maskType = cast<VMIMaskType>(op.getMask().getType());
  int64_t laneCount = maskType.getElementCount();

  auto i32Type = builder.getIntegerType(32);
  Value cL = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getIntegerAttr(i32Type, laneCount));
  Value act = builder.create<arith::MinSIOp>(loc, i32Type, op.getScalar(), cL);
  Value aidx = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), act);
  Value mask = builder.create<VMICreateMaskOp>(loc, maskType, aidx).getResult();
  Value next = builder.create<arith::SubIOp>(loc, i32Type, op.getScalar(), act);

  op.getMask().replaceAllUsesWith(mask);
  op.getScalarOut().replaceAllUsesWith(next);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C8 helpers: vgather / vscatter
//===----------------------------------------------------------------------===//

/// Lower vgather to legacy gather.  Legacy gather carries an explicit passthru
/// operand for inactive lanes; pmode="zero" is modelled with a zero passthru.
/// pmode="merge" (preserve OLD_DEST) has no SSA passthru and is skipped.
static LogicalResult lowerVgather(VMIVgatherOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  auto resultType = cast<VMIVRegType>(op.getResult().getType());
  // pmode="zero" (default): inactive lanes are zeroed. Legacy gather models
  // inactive lanes with an explicit passthru whose element type must match the
  // result, so synthesise a zero constant of the result type — the offsets
  // vector cannot be reused because its element type (e.g. i32) generally
  // differs from the result element type (e.g. f32).
  Value passthru = createZeroConstant(builder, loc, resultType);
  Value result = builder
                     .create<VMIGatherOp>(loc, resultType, op.getSource(),
                                          op.getOffsets(), op.getMask(),
                                          passthru)
                     .getResult();
  op.getResult().replaceAllUsesWith(result);
  op->erase();
  return success();
}

/// Lower vscatter to legacy scatter.  Legacy scatter only writes active lanes
/// (mask-governed), matching vscatter's default/zero pmode; merge is skipped.
static LogicalResult lowerVscatter(VMIVscatterOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  builder.create<VMIScatterOp>(loc, op.getValue(), op.getDestination(),
                               op.getOffsets(), op.getMask());
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Category C9 helpers: vlrelu / vprelu (fused → legacy chains)
//===----------------------------------------------------------------------===//

/// Lower vlrelu (x>0 ? x : slope*x) to max(x,0) + slope*min(x,0).
/// slope is a scalar float broadcast to a vector.
static LogicalResult lowerVlrelu(VMIVlreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value slopeVec = builder
                       .create<VMIBroadcastOp>(loc, resultType, op.getSlope())
                       .getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, slopeVec, neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

/// Lower vprelu (max(x,0) + alpha*min(x,0)) to legacy max/min/mul/add.
/// alpha is a per-lane vector (no broadcast needed).
static LogicalResult lowerVprelu(VMIVpreluOp op, OpBuilder &builder) {
  if (hasMergePmode(op)) {
    return failure();
  }

  Location loc = op.getLoc();
  Type resultType = op.getResult().getType();
  auto vmiType = cast<VMIVRegType>(resultType);
  Value x = op.getX();

  Value zeroConst = createZeroConstant(builder, loc, vmiType);
  Value pos =
      builder.create<VMIMaxFOp>(loc, resultType, x, zeroConst).getResult();
  Value neg =
      builder.create<VMIMinFOp>(loc, resultType, x, zeroConst).getResult();
  Value scaledNeg =
      builder.create<VMIMulFOp>(loc, resultType, op.getAlpha(), neg).getResult();
  Value raw =
      builder.create<VMIAddFOp>(loc, resultType, pos, scaledNeg).getResult();
  op.getResult().replaceAllUsesWith(raw);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Unified operation dispatch
//===----------------------------------------------------------------------===//

static bool isUnifiedLoweringCandidate(Operation *op) {
  return isa<VMIVciOp, VMIVinterpretCastOp, VMIvSelOp, VMIVbrcOp,
             VMIVaddOp, VMIVsubOp, VMIVmulOp, VMIVdivOp, VMIVminOp,
             VMIVmaxOp, VMIVandOp, VMIVorOp, VMIVxorOp, VMIVshlOp,
             VMIVshrOp, VMIVnegOp, VMIVabsOp, VMIVsqrtOp, VMIVexpOp,
             VMIVlnOp, VMIVreluOp, VMIVnotOp, VMIVcmpOp, VMIVcmpsOp,
             VMICvtOp, VMIvLoadOp, VMIvStoreOp, VMIVsstbOp, VMIPsetOp,
             VMIPgeOp, VMIPltOp, VMIvcaddOp, VMIvcmaxOp, VMIvcminOp,
             VMIVmulaOp, VMIVaxpyOp, VMIVgatherOp, VMIVscatterOp,
             VMIVlreluOp, VMIVpreluOp>(op);
}

static bool requiresDirectVMIToVPTOLowering(Operation *op) {
  return isa<VMIAddSOp, VMIMulSOp, VMIMaxSOp, VMIMinSOp, VMIShlSOp,
             VMIShrSOp, VMIVaddcOp, VMIVaddcsOp, VMIVintlvOp, VMIVdintlvOp,
             VMIVselrOp, VMIVgatherbOp, VMIVmullOp>(op);
}

static Value createIotaReplacement(VMIVciOp vci, OpBuilder &builder) {
  StringAttr orderAttr;
  if (std::optional<StringRef> order = vci.getOrder()) {
    orderAttr = builder.getStringAttr(*order);
  }
  Type resultType = vci.getResult().getType();
  IntegerAttr groupAttr = vci.getGroupAttr();
  bool isGrouped = groupAttr && groupAttr.getInt() > 1;
  if (!isGrouped) {
    return builder
        .create<VMIIotaOp>(vci.getLoc(), resultType, vci.getBase(), orderAttr)
        .getResult();
  }
  auto vmiType = dyn_cast<VMIVRegType>(resultType);
  bool needsLayoutConversion =
      vmiType && vmiType.getLayoutAttr() &&
      !vmiType.getLayoutAttr().isContiguous();
  if (!needsLayoutConversion) {
    return builder
        .create<VMIGroupIotaOp>(vci.getLoc(), resultType, vci.getBase(),
                                orderAttr, groupAttr)
        .getResult();
  }
  Type contiguousType = VMIVRegType::get(
      vci.getContext(), vmiType.getElementCount(), vmiType.getElementType(),
      VMILayoutAttr::getContiguous(vci.getContext()));
  Value contiguous =
      builder
          .create<VMIGroupIotaOp>(vci.getLoc(), contiguousType, vci.getBase(),
                                  orderAttr, groupAttr)
          .getResult();
  return builder
      .create<VMIEnsureLayoutOp>(vci.getLoc(), vmiType, contiguous)
      .getResult();
}

static void lowerVci(VMIVciOp vci, OpBuilder &builder) {
  builder.setInsertionPoint(vci);
  Value replacement = createIotaReplacement(vci, builder);
  vci.getResult().replaceAllUsesWith(replacement);
  vci.erase();
}

static void lowerInterpretCast(VMIVinterpretCastOp castOp,
                               OpBuilder &builder) {
  builder.setInsertionPoint(castOp);
  Value result = builder
                     .create<VMIBitcastOp>(castOp.getLoc(),
                                           castOp.getResult().getType(),
                                           castOp.getSource())
                     .getResult();
  castOp.getResult().replaceAllUsesWith(result);
  castOp.erase();
}

static void lowerSelect(VMIvSelOp select, OpBuilder &builder) {
  builder.setInsertionPoint(select);
  Value result =
      builder
          .create<VMISelectOp>(select.getLoc(), select.getResult().getType(),
                               select.getMask(), select.getTrueValue(),
                               select.getFalseValue())
          .getResult();
  select.getResult().replaceAllUsesWith(result);
  select.erase();
}

static void lowerBroadcast(VMIVbrcOp broadcast, OpBuilder &builder) {
  builder.setInsertionPoint(broadcast);
  Value result = broadcast.getGroupAttr()
                     ? builder
                           .create<VMIGroupBroadcastOp>(
                               broadcast.getLoc(),
                               broadcast.getResult().getType(),
                               broadcast.getValue(), broadcast.getGroupAttr())
                           .getResult()
                     : builder
                           .create<VMIBroadcastOp>(
                               broadcast.getLoc(),
                               broadcast.getResult().getType(),
                               broadcast.getValue())
                           .getResult();
  broadcast.getResult().replaceAllUsesWith(result);
  broadcast.erase();
}

static bool lowerRename(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIVciOp>([&builder](auto vci) {
        lowerVci(vci, builder);
        return true;
      })
      .Case<VMIVinterpretCastOp>([&builder](auto castOp) {
        lowerInterpretCast(castOp, builder);
        return true;
      })
      .Case<VMIvSelOp>([&builder](auto select) {
        lowerSelect(select, builder);
        return true;
      })
      .Case<VMIVbrcOp>([&builder](auto broadcast) {
        lowerBroadcast(broadcast, builder);
        return true;
      })
      .Default([](Operation *) { return false; });
}

static void lowerStrideStore(VMIVsstbOp store, OpBuilder &builder) {
  if (hasMergePmode(store)) {
    return;
  }
  builder.create<VMIStrideStoreOp>(
      store.getLoc(), store.getValue(), store.getDestination(),
      store.getOffset(), store.getBlockStride(), store.getMask());
  store.erase();
}

static bool lowerPredicateAndMemorySpecial(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIPsetOp>([&builder](auto typedOp) {
        (void)lowerPset(typedOp, builder);
        return true;
      })
      .Case<VMIPgeOp>([&builder](auto typedOp) {
        (void)lowerPge(typedOp, builder);
        return true;
      })
      .Case<VMIPltOp>([&builder](auto typedOp) {
        (void)lowerPlt(typedOp, builder);
        return true;
      })
      .Case<VMIVcmpOp>([&builder](auto typedOp) {
        (void)lowerVCmp(typedOp, builder);
        return true;
      })
      .Case<VMIVcmpsOp>([&builder](auto typedOp) {
        (void)lowerVCmps(typedOp, builder);
        return true;
      })
      .Case<VMICvtOp>([&builder](auto typedOp) {
        (void)lowerVCvt(typedOp, builder);
        return true;
      })
      .Case<VMIvLoadOp>([&builder](auto typedOp) {
        (void)lowerVLoad(typedOp, builder);
        return true;
      })
      .Case<VMIvStoreOp>([&builder](auto typedOp) {
        (void)lowerVStore(typedOp, builder);
        return true;
      })
      .Case<VMIVsstbOp>([&builder](auto typedOp) {
        lowerStrideStore(typedOp, builder);
        return true;
      })
      .Default([](Operation *) { return false; });
}

static bool lowerReductionSpecial(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIvcaddOp>([&builder](auto typedOp) {
        (void)lowerVCadd(typedOp, builder);
        return true;
      })
      .Case<VMIvcmaxOp>([&builder](auto typedOp) {
        (void)lowerVcmax(typedOp, builder);
        return true;
      })
      .Case<VMIvcminOp>([&builder](auto typedOp) {
        (void)lowerVcmin(typedOp, builder);
        return true;
      })
      .Default([](Operation *) { return false; });
}

static bool lowerAdvancedSpecial(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIVmulaOp>([&builder](auto typedOp) {
        (void)lowerVmula(typedOp, builder);
        return true;
      })
      .Case<VMIVaxpyOp>([&builder](auto typedOp) {
        (void)lowerVaxpy(typedOp, builder);
        return true;
      })
      .Case<VMIVgatherOp>([&builder](auto typedOp) {
        (void)lowerVgather(typedOp, builder);
        return true;
      })
      .Case<VMIVscatterOp>([&builder](auto typedOp) {
        (void)lowerVscatter(typedOp, builder);
        return true;
      })
      .Case<VMIVlreluOp>([&builder](auto typedOp) {
        (void)lowerVlrelu(typedOp, builder);
        return true;
      })
      .Case<VMIVpreluOp>([&builder](auto typedOp) {
        (void)lowerVprelu(typedOp, builder);
        return true;
      })
      .Default([](Operation *) { return false; });
}

static bool lowerSpecial(Operation *op, OpBuilder &builder) {
  return lowerPredicateAndMemorySpecial(op, builder) ||
         lowerReductionSpecial(op, builder) ||
         lowerAdvancedSpecial(op, builder);
}

template <typename FloatOp, typename IntegerOp, typename UnifiedOp>
static void lowerTypedBinary(UnifiedOp op, OpBuilder &builder) {
  Type elementType = getVMIElementType(op.getResult());
  auto createLegacy = [&builder, elementType](Location loc, Type type,
                                               Value lhs, Value rhs) -> Value {
    if (isFloatType(elementType)) {
      return builder.create<FloatOp>(loc, type, lhs, rhs).getResult();
    }
    return builder.create<IntegerOp>(loc, type, lhs, rhs).getResult();
  };
  (void)lowerBinaryIgnoringMask(op, createLegacy);
}

template <typename LegacyOp, typename UnifiedOp>
static void lowerIntegerBinary(UnifiedOp op, OpBuilder &builder) {
  auto createLegacy = [&builder](Location loc, Type type, Value lhs,
                                  Value rhs) -> Value {
    return builder.create<LegacyOp>(loc, type, lhs, rhs).getResult();
  };
  (void)lowerBinaryIgnoringMask(op, createLegacy);
}

template <typename MaskOp, typename DataOp, typename UnifiedOp>
static void lowerLogicalBinary(UnifiedOp op, OpBuilder &builder) {
  builder.setInsertionPoint(op);
  if (isa<VMIMaskType>(op.getLhs().getType())) {
    Value result = builder
                       .create<MaskOp>(op.getLoc(), op.getResult().getType(),
                                       op.getLhs(), op.getRhs())
                       .getResult();
    op.getResult().replaceAllUsesWith(result);
    op.erase();
    return;
  }
  lowerIntegerBinary<DataOp>(op, builder);
}

static void lowerShiftRight(VMIVshrOp op, OpBuilder &builder) {
  Type elementType = getVMIElementType(op.getLhs());
  auto createLegacy = [&builder, elementType](Location loc, Type type,
                                               Value lhs, Value rhs) -> Value {
    if (!cast<IntegerType>(elementType).isSigned()) {
      return builder.create<VMIShRUIOp>(loc, type, lhs, rhs).getResult();
    }
    return builder.create<VMIShRSIOp>(loc, type, lhs, rhs).getResult();
  };
  (void)lowerBinaryIgnoringMask(op, createLegacy);
}

static bool lowerBinary(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIVaddOp>([&builder](auto typedOp) {
        lowerTypedBinary<VMIAddFOp, VMIAddIOp>(typedOp, builder); return true;
      })
      .Case<VMIVsubOp>([&builder](auto typedOp) {
        lowerTypedBinary<VMISubFOp, VMISubIOp>(typedOp, builder); return true;
      })
      .Case<VMIVmulOp>([&builder](auto typedOp) {
        lowerTypedBinary<VMIMulFOp, VMIMulIOp>(typedOp, builder); return true;
      })
      .Case<VMIVdivOp>([&builder](auto typedOp) {
        lowerIntegerBinary<VMIDivFOp>(typedOp, builder); return true;
      })
      .Case<VMIVminOp>([&builder](auto typedOp) {
        lowerTypedBinary<VMIMinFOp, VMIMinIOp>(typedOp, builder); return true;
      })
      .Case<VMIVmaxOp>([&builder](auto typedOp) {
        lowerTypedBinary<VMIMaxFOp, VMIMaxIOp>(typedOp, builder); return true;
      })
      .Case<VMIVandOp>([&builder](auto typedOp) {
        lowerLogicalBinary<VMIMaskAndOp, VMIAndIOp>(typedOp, builder);
        return true;
      })
      .Case<VMIVorOp>([&builder](auto typedOp) {
        lowerLogicalBinary<VMIMaskOrOp, VMIOrIOp>(typedOp, builder); return true;
      })
      .Case<VMIVxorOp>([&builder](auto typedOp) {
        lowerLogicalBinary<VMIMaskXOrOp, VMIXOrIOp>(typedOp, builder);
        return true;
      })
      .Case<VMIVshlOp>([&builder](auto typedOp) {
        lowerIntegerBinary<VMIShLIOp>(typedOp, builder); return true;
      })
      .Case<VMIVshrOp>([&builder](auto typedOp) {
        lowerShiftRight(typedOp, builder); return true;
      })
      .Default([](Operation *) { return false; });
}

template <typename FloatOp, typename IntegerOp, typename UnifiedOp>
static void lowerTypedUnary(UnifiedOp op, OpBuilder &builder) {
  Type elementType = getVMIElementType(op.getResult());
  auto createLegacy = [&builder, elementType](Location loc, Type type,
                                               Value source) -> Value {
    if (isFloatType(elementType)) {
      return builder.create<FloatOp>(loc, type, source).getResult();
    }
    return builder.create<IntegerOp>(loc, type, source).getResult();
  };
  (void)lowerMaskedUnary(op, builder, createLegacy);
}

template <typename LegacyOp, typename UnifiedOp>
static void lowerSimpleUnary(UnifiedOp op, OpBuilder &builder) {
  auto createLegacy = [&builder](Location loc, Type type,
                                  Value source) -> Value {
    return builder.create<LegacyOp>(loc, type, source).getResult();
  };
  (void)lowerMaskedUnary(op, builder, createLegacy);
}

static Value createAbsoluteValue(VMIVabsOp op, OpBuilder &builder,
                                 Location loc, Type type, Value source) {
  Type elementType = getVMIElementType(op.getResult());
  if (!elementType.isBF16()) {
    return isFloatType(elementType)
               ? builder.create<VMIAbsFOp>(loc, type, source).getResult()
               : builder.create<VMIAbsIOp>(loc, type, source).getResult();
  }
  constexpr uint16_t kBFloat16MagnitudeMask = 0x7FFF;
  auto sourceType = cast<VMIVRegType>(source.getType());
  Type i16 = builder.getIntegerType(mlir::pto::kValue16);
  auto integerType = VMIVRegType::get(
      builder.getContext(), sourceType.getElementCount(), i16,
      sourceType.getLayout());
  Value asInteger = builder.create<VMIBitcastOp>(loc, integerType, source);
  Value mask = builder.create<arith::ConstantOp>(
      loc, i16, builder.getIntegerAttr(i16, kBFloat16MagnitudeMask));
  Value maskVector =
      builder.create<VMIBroadcastOp>(loc, integerType, mask).getResult();
  Value cleared =
      builder.create<VMIAndIOp>(loc, integerType, asInteger, maskVector);
  return builder.create<VMIBitcastOp>(loc, type, cleared).getResult();
}

static void lowerAbsolute(VMIVabsOp op, OpBuilder &builder) {
  auto createLegacy = [&builder, op](Location loc, Type type,
                                      Value source) -> Value {
    return createAbsoluteValue(op, builder, loc, type, source);
  };
  (void)lowerMaskedUnary(op, builder, createLegacy);
}

static void lowerLogicalNot(VMIVnotOp op, OpBuilder &builder) {
  builder.setInsertionPoint(op);
  if (isa<VMIMaskType>(op.getSource().getType())) {
    Value result = builder
                       .create<VMIMaskNotOp>(op.getLoc(),
                                             op.getResult().getType(),
                                             op.getSource())
                       .getResult();
    op.getResult().replaceAllUsesWith(result);
    op.erase();
    return;
  }
  lowerSimpleUnary<VMINotOp>(op, builder);
}

static bool lowerUnary(Operation *op, OpBuilder &builder) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<VMIVnegOp>([&builder](auto typedOp) {
        lowerTypedUnary<VMINegFOp, VMINegIOp>(typedOp, builder); return true;
      })
      .Case<VMIVabsOp>([&builder](auto typedOp) {
        lowerAbsolute(typedOp, builder); return true;
      })
      .Case<VMIVsqrtOp>([&builder](auto typedOp) {
        lowerSimpleUnary<VMISqrtOp>(typedOp, builder); return true;
      })
      .Case<VMIVexpOp>([&builder](auto typedOp) {
        lowerSimpleUnary<VMIExpOp>(typedOp, builder); return true;
      })
      .Case<VMIVlnOp>([&builder](auto typedOp) {
        lowerSimpleUnary<VMILnOp>(typedOp, builder); return true;
      })
      .Case<VMIVreluOp>([&builder](auto typedOp) {
        lowerSimpleUnary<VMIReluOp>(typedOp, builder); return true;
      })
      .Case<VMIVnotOp>([&builder](auto typedOp) {
        lowerLogicalNot(typedOp, builder); return true;
      })
      .Default([](Operation *) { return false; });
}

static void lowerUnifiedOperation(Operation *op) {
  OpBuilder builder(op);
  bool lowered = lowerRename(op, builder) || lowerSpecial(op, builder) ||
                 lowerBinary(op, builder);
  if (lowered) {
    return;
  }
  (void)lowerUnary(op, builder);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct VMILowerUnifiedToLegacyPass
    : public mlir::pto::impl::VMILowerUnifiedToLegacyBase<
          VMILowerUnifiedToLegacyPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMILowerUnifiedToLegacyPass)

  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }
};

} // namespace

void VMILowerUnifiedToLegacyPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Operation *, mlir::pto::kValue128> worklist;
  module.walk([&worklist](Operation *op) {
    if (isUnifiedLoweringCandidate(op)) {
      worklist.push_back(op);
    }
    if (requiresDirectVMIToVPTOLowering(op)) {
      op->emitRemark("VMI unified op has no legacy equivalent — "
                     "requires direct VMIToVPTO 1:N lowering");
    }
  });
  for (Operation *op : llvm::reverse(worklist)) {
    if (op->getBlock()) {
      lowerUnifiedOperation(op);
    }
  }
}

std::unique_ptr<Pass> mlir::pto::createVMILowerUnifiedToLegacyPass() {
  return std::make_unique<VMILowerUnifiedToLegacyPass>();
}
