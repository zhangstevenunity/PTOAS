// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- InferPTOLayout.cpp - Infer layout for global tensor views -----------===//
//
// The pto-isa GlobalTensor ABI expects shape/stride to be represented in a 5D
// right-aligned form (pad leading dims with 1). We infer ND/DN/NZ with the same
// 5D view here, propagate the result through control-flow and function
// boundaries, and attach an optional `layout` attribute to:
//   - memref.reinterpret_cast (lowered from pto.make_tensor_view)
//   - memref.subview          (lowered from pto.partition_view)
//   - pto.tload / pto.tstore  (for fully-static GM memrefs)
//
// EmitC lowering should consume this attribute and avoid re-inferring layout
// when it is available.
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOLayoutUtils.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VMIControlFlowSupport.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_INFERPTOLAYOUT
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static constexpr llvm::StringLiteral kLayoutAttrName = "layout";
static constexpr llvm::StringLiteral kInferredLayoutAttrName =
    "pto.inferred_layout";

static LayoutAttr getViewTypeLayoutAttr(Type type) {
  if (auto tensorView = dyn_cast<TensorViewType>(type)) {
    return tensorView.getLayoutAttr();
  }
  if (auto partitionView = dyn_cast<PartitionTensorViewType>(type)) {
    return partitionView.getLayoutAttr();
  }
  return {};
}

static bool isViewType(Type type) {
  return isa<TensorViewType, PartitionTensorViewType>(type);
}

static Type getViewTypeWithLayout(Type type, Layout layout) {
  Attribute layoutAttr;
  if (layout != Layout::ND) {
    layoutAttr = LayoutAttr::get(type.getContext(), layout);
  }
  if (auto tensorView = dyn_cast<TensorViewType>(type)) {
    return TensorViewType::get(type.getContext(), tensorView.getShape(),
                               tensorView.getElementType(), layoutAttr);
  }
  if (auto partitionView = dyn_cast<PartitionTensorViewType>(type)) {
    return PartitionTensorViewType::get(
        type.getContext(), partitionView.getShape(),
        partitionView.getElementType(), layoutAttr);
  }
  return type;
}

static std::optional<int64_t> getConstInt(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantIntOp>()) {
    return c.value();
  }
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }
  return std::nullopt;
}

static std::optional<int64_t> getConstInt(OpFoldResult ofr) {
  if (isa<Attribute>(ofr)) {
    Attribute attr = cast<Attribute>(ofr);
    if (auto ia = dyn_cast<IntegerAttr>(attr)) {
      return ia.getInt();
    }
    return std::nullopt;
  }
  return getConstInt(cast<Value>(ofr));
}

static unsigned elemByteSize(Type ty) {
  return getPTOStorageElemByteSize(ty);
}

static bool isF8E8M0ElemType(Type ty) {
  return getPTOStorageElemByteSize(ty) == 1 && isa<Type>(ty) &&
         ty.getTypeID() == ty.getTypeID() &&
         [&]() {
           std::string buffer;
           llvm::raw_string_ostream os(buffer);
           os << ty;
           os.flush();
           return buffer == "!pto.f8E8M0";
         }();
}

static bool isGlobalMemRef(MemRefType ty) {
  if (auto asAttr =
          dyn_cast_or_null<pto::AddressSpaceAttr>(ty.getMemorySpace())) {
    auto as = asAttr.getAddressSpace();
    return (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
  }
  // Treat missing memory_space as GM.
  return true;
}

static std::optional<Layout> tileBLayoutToGlobalLayout(Type tileLikeTy) {
  auto tbTy = dyn_cast<TileBufType>(tileLikeTy);
  if (!tbTy) {
    return std::nullopt;
  }
  auto bl = dyn_cast_or_null<BLayoutAttr>(tbTy.getBLayoutAttr());
  if (!bl) {
    return std::nullopt;
  }
  switch (bl.getValue()) {
  case BLayout::RowMajor:
    return Layout::ND;
  case BLayout::ColMajor:
    return Layout::DN;
  }
  return std::nullopt;
}

static bool isVectorTileType(Type tileLikeTy) {
  auto tbTy = dyn_cast<TileBufType>(tileLikeTy);
  if (!tbTy) {
    return false;
  }
  auto ms = dyn_cast_or_null<AddressSpaceAttr>(tbTy.getMemorySpace());
  return ms && ms.getAddressSpace() == AddressSpace::VEC;
}

static bool isMinorColsOne(ArrayRef<int64_t> shape) {
  return !shape.empty() && shape.back() == 1;
}

struct ResolvedLayoutInfo {
  Operation *owner = nullptr;
  std::optional<Layout> layout;
  bool inferred = false;
};

static bool getStaticShapeAndStride(MakeTensorViewOp op,
                                    SmallVectorImpl<int64_t> &shape,
                                    SmallVectorImpl<int64_t> &strides);
static ResolvedLayoutInfo resolveLayoutFromViewValue(Value v);

static void setLayoutAttr(Operation *op, Layout layout, bool inferred) {
  op->setAttr(kLayoutAttrName, LayoutAttr::get(op->getContext(), layout));
  if (inferred) {
    op->setAttr(kInferredLayoutAttrName, BoolAttr::get(op->getContext(), true));
  } else {
    op->removeAttr(kInferredLayoutAttrName);
  }
}

template <typename SignalFailureFn>
static bool verifyExistingLayoutAttr(Operation *op, ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides,
                                     unsigned storageElemBytes,
                                     SignalFailureFn signalFailure) {
  auto existing = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (!existing) {
    return true;
  }

  Layout layout = existing.getLayout();
  if (!isLayoutCompatible5D(layout, shape, strides, storageElemBytes)) {
    if (layout == Layout::NZ) {
      if (shape.size() == kPTOLayoutRank && strides.size() == kPTOLayoutRank) {
        auto error =
            getNZViewCompatibilityError(shape, strides, storageElemBytes);
        op->emitError() << "user-specified layout=nz is incompatible with "
                           "shape/stride: "
                        << error.value_or("unknown NZ layout mismatch");
      } else {
        op->emitError()
            << "user-specified layout=nz requires a rank-5 view, got rank "
            << shape.size();
      }
    } else {
      op->emitError() << "user-specified layout=" << stringifyLayout(layout)
                      << " is incompatible with the view shape/stride";
    }
    signalFailure();
    return false;
  }

  if (op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
    op->removeAttr(kInferredLayoutAttrName);
  }
  return true;
}

template <typename SignalFailureFn>
static void resolveOrInferLayoutAttr(Operation *op, ArrayRef<int64_t> shape,
                                     ArrayRef<int64_t> strides,
                                     unsigned storageElemBytes,
                                     std::optional<Layout> inferred,
                                     SignalFailureFn signalFailure) {
  if (op->getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
    (void)verifyExistingLayoutAttr(op, shape, strides, storageElemBytes,
                                   signalFailure);
    return;
  }
  setLayoutAttr(op, inferred.value_or(Layout::ND), /*inferred=*/true);
}

static std::optional<Layout> inferFromStaticMemRefTy(MemRefType mrTy) {
  if (!mrTy.hasStaticShape() || mrTy.getRank() == 0 ||
      mrTy.getRank() > kPTOLayoutRank) {
    return std::nullopt;
  }
  SmallVector<int64_t> strideInts;
  int64_t offset = ShapedType::kDynamic;
  if (failed(
          mlir::pto::getPTOMemRefStridesAndOffset(mrTy, strideInts, offset))) {
    return std::nullopt;
  }
  if (offset == ShapedType::kDynamic ||
      llvm::any_of(strideInts,
                   [](int64_t s) { return s == ShapedType::kDynamic; })) {
    return std::nullopt;
  }
  return inferLayout5D(mrTy.getShape(), strideInts,
                       elemByteSize(mrTy.getElementType()));
}

static bool isInferredMinor2DAmbiguousLayout(MakeTensorViewOp op) {
  auto inferred = op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName);
  if (!inferred || !inferred.getValue()) {
    return false;
  }
  auto layout = op.getLayoutAttr();
  if (!layout) {
    return false;
  }
  Layout layoutValue = layout.getLayout();
  if (layoutValue != Layout::ND && layoutValue != Layout::DN) {
    return false;
  }

  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  if (!getStaticShapeAndStride(op, shape, strides)) {
    return false;
  }

  bool ambiguous = false;
  (void)inferLayout5D(
      shape, strides,
      elemByteSize(cast<TensorViewType>(op.getResult().getType()).getElementType()),
      std::nullopt, &ambiguous);
  return ambiguous && isMinorColsOne(shape);
}

template <typename LoadStoreOp, typename ViewGetter, typename TileGetter>
static void maybeRepairMinor2DLoadStoreLayout(LoadStoreOp op, ViewGetter getView,
                                              TileGetter getTile) {
  auto tilePref = isVectorTileType(getTile(op).getType())
                      ? tileBLayoutToGlobalLayout(getTile(op).getType())
                      : std::nullopt;
  if (!tilePref || (*tilePref != Layout::ND && *tilePref != Layout::DN)) {
    return;
  }

  auto viewInfo = resolveLayoutFromViewValue(getView(op));
  if (!viewInfo.owner || !viewInfo.layout || !viewInfo.inferred ||
      *viewInfo.layout == *tilePref) {
    return;
  }
  auto tv = dyn_cast<MakeTensorViewOp>(viewInfo.owner);
  if (!tv) {
    return;
  }

  if (isInferredMinor2DAmbiguousLayout(tv)) {
    setLayoutAttr(viewInfo.owner, *tilePref, /*inferred=*/true);
    setLayoutAttr(op.getOperation(), *tilePref, /*inferred=*/true);
  }
}

template <typename LoadStoreOp, typename ViewGetter, typename TileGetter>
static void attachLoadStoreLayout(LoadStoreOp op, ViewGetter getView,
                                  TileGetter getTile) {
  if (op->template getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
    maybeRepairMinor2DLoadStoreLayout(op, getView, getTile);
    return;
  }

  auto viewInfo = resolveLayoutFromViewValue(getView(op));
  if (viewInfo.layout) {
    setLayoutAttr(op.getOperation(), *viewInfo.layout, viewInfo.inferred);
  } else if (auto memTy = dyn_cast<MemRefType>(getView(op).getType());
             memTy && isGlobalMemRef(memTy)) {
    setLayoutAttr(op.getOperation(), inferFromStaticMemRefTy(memTy).value_or(Layout::ND),
                  /*inferred=*/true);
  }

  maybeRepairMinor2DLoadStoreLayout(op, getView, getTile);
}

struct LayoutPreference {
  std::optional<Layout> preferred;
  bool conflict = false;
};

static LayoutPreference collectPreferredLayoutFromConsumers(Value tensorView) {
  LayoutPreference result;
  auto mergePref = [&result](std::optional<Layout> candidate) {
    if (!candidate) {
      return;
    }
    if (!result.preferred) {
      result.preferred = candidate;
      return;
    }
    if (*result.preferred != *candidate) {
      result.preferred = std::nullopt;
      result.conflict = true;
    }
  };

  auto walkUses = [&mergePref](auto &&self, Value v) -> void {
    for (OpOperand &use : v.getUses()) {
      Operation *owner = use.getOwner();
      unsigned operandIndex = use.getOperandNumber();

      if (auto part = dyn_cast<PartitionViewOp>(owner)) {
        if (operandIndex == 0) {
          self(self, part.getResult());
        }
        continue;
      }

      if (auto load = dyn_cast<pto::TLoadOp>(owner)) {
        if (operandIndex == 0) {
          if (auto dstTy = dyn_cast<TileBufType>(load.getDst().getType())) {
            auto dstSpace =
                dyn_cast_or_null<AddressSpaceAttr>(dstTy.getMemorySpace());
            if (dstSpace &&
                dstSpace.getAddressSpace() == AddressSpace::MAT &&
                isF8E8M0ElemType(dstTy.getElementType())) {
              auto cfg = dstTy.getConfigAttr();
              auto bl = dyn_cast_or_null<BLayoutAttr>(cfg.getBLayout());
              auto sl = dyn_cast_or_null<SLayoutAttr>(cfg.getSLayout());
              if (bl && sl) {
                if (bl.getValue() == BLayout::RowMajor &&
                    sl.getValue() == SLayout::RowMajor &&
                    dstTy.getShape().size() == 2 && dstTy.getShape()[0] != 1) {
                  mergePref(Layout::MX_A_ZZ);
                } else if (bl.getValue() == BLayout::ColMajor &&
                           sl.getValue() == SLayout::ColMajor) {
                  mergePref(Layout::MX_B_NN);
                }
              }
            } else if (isVectorTileType(dstTy)) {
              mergePref(tileBLayoutToGlobalLayout(dstTy));
            }
          }
        }
        continue;
      }

      if (auto store = dyn_cast<pto::TStoreOp>(owner)) {
        if (operandIndex == 1 && isVectorTileType(store.getSrc().getType())) {
          mergePref(tileBLayoutToGlobalLayout(store.getSrc().getType()));
        }
        continue;
      }
    }
  };

  walkUses(walkUses, tensorView);
  return result;
}

static std::optional<Layout> inferMakeTensorViewLayout(
    MakeTensorViewOp op, ArrayRef<int64_t> shape, ArrayRef<int64_t> strides,
    bool &isAmbiguous) {
  auto pref = collectPreferredLayoutFromConsumers(op.getResult());
  if (!pref.conflict && pref.preferred &&
      (*pref.preferred == Layout::MX_A_ZZ ||
       *pref.preferred == Layout::MX_B_NN)) {
    return pref.preferred;
  }
  std::optional<Layout> preferredForAmbiguous = std::nullopt;
  if (!pref.conflict && isMinorColsOne(shape)) {
    preferredForAmbiguous = pref.preferred;
  }
  return inferLayout5D(
      shape, strides,
      elemByteSize(
          cast<TensorViewType>(op.getResult().getType()).getElementType()),
      preferredForAmbiguous, &isAmbiguous);
}

static void reconcileAmbiguousTensorViewLayout(MakeTensorViewOp op,
                                               ArrayRef<int64_t> shape) {
  auto pref = collectPreferredLayoutFromConsumers(op.getResult());
  if (!isMinorColsOne(shape)) {
    return;
  }
  if (!op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
    return;
  }
  auto cur = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
  if (cur && pref.preferred && *pref.preferred != cur.getLayout()) {
    setLayoutAttr(op.getOperation(), *pref.preferred, /*inferred=*/true);
  }
}

static bool getShapeAndStride(MakeTensorViewOp op,
                              SmallVectorImpl<int64_t> &shape,
                              SmallVectorImpl<int64_t> &strides,
                              bool &allStatic) {
  auto tvTy = dyn_cast<TensorViewType>(op.getResult().getType());
  if (!tvTy) {
    return false;
  }

  const size_t rank = op.getShape().size();
  if (rank == 0 || rank > kPTOLayoutRank) {
    return false;
  }

  allStatic = true;
  shape.clear();
  shape.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    int64_t dim = tvTy.getShape()[i];
    if (dim == ShapedType::kDynamic) {
      auto v = getConstInt(op.getShape()[i]);
      if (v) {
        dim = *v;
      } else {
        allStatic = false;
      }
    }
    shape.push_back(dim);
  }

  strides.clear();
  strides.reserve(rank);
  for (Value s : op.getStrides()) {
    auto v = getConstInt(s);
    if (v) {
      strides.push_back(*v);
    } else {
      strides.push_back(ShapedType::kDynamic);
      allStatic = false;
    }
  }
  return true;
}

static bool getStaticShapeAndStride(MakeTensorViewOp op,
                                    SmallVectorImpl<int64_t> &shape,
                                    SmallVectorImpl<int64_t> &strides) {
  bool allStatic = false;
  return getShapeAndStride(op, shape, strides, allStatic) && allStatic;
}

static void getFoldResults(ArrayRef<OpFoldResult> values,
                           SmallVectorImpl<int64_t> &result, bool &allStatic) {
  result.clear();
  result.reserve(values.size());
  for (OpFoldResult value : values) {
    auto folded = getConstInt(value);
    if (folded) {
      result.push_back(*folded);
    } else {
      result.push_back(ShapedType::kDynamic);
      allStatic = false;
    }
  }
}

static ResolvedLayoutInfo resolveLayoutFromViewValue(Value v) {
  ResolvedLayoutInfo info;
  if (auto layoutAttr = getViewTypeLayoutAttr(v.getType())) {
    info.owner = v.getDefiningOp();
    info.layout = layoutAttr.getLayout();
    if (info.owner) {
      if (auto inferred =
              info.owner->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
        info.inferred = inferred.getValue();
      }
    }
    return info;
  }
  Operation *def = v.getDefiningOp();
  while (def) {
    if (auto layoutAttr = def->getAttrOfType<LayoutAttr>(kLayoutAttrName)) {
      info.owner = def;
      info.layout = layoutAttr.getLayout();
      if (auto inferred =
              def->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
        info.inferred = inferred.getValue();
      }
      return info;
    }
    if (auto part = dyn_cast<PartitionViewOp>(def)) {
      v = part.getSource();
      def = v.getDefiningOp();
      continue;
    }
    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      v = subview.getSource();
      def = v.getDefiningOp();
      continue;
    }
    if (auto reinterpret = dyn_cast<memref::ReinterpretCastOp>(def)) {
      v = reinterpret.getSource();
      def = v.getDefiningOp();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      v = cast.getSource();
      def = v.getDefiningOp();
      continue;
    }
    break;
  }
  return info;
}

static void getValuesOrDynamic(ValueRange values,
                               SmallVectorImpl<int64_t> &result) {
  result.clear();
  result.reserve(values.size());
  for (Value value : values) {
    result.push_back(getConstInt(value).value_or(ShapedType::kDynamic));
  }
}

static void getFoldResultsOrDynamic(ArrayRef<OpFoldResult> values,
                                    SmallVectorImpl<int64_t> &result) {
  result.clear();
  result.reserve(values.size());
  for (OpFoldResult value : values) {
    result.push_back(getConstInt(value).value_or(ShapedType::kDynamic));
  }
}

static void resolveDynamicShape(ValueRange values,
                                SmallVectorImpl<int64_t> &shape) {
  for (auto [index, value] : llvm::enumerate(values)) {
    if (shape[index] != ShapedType::kDynamic) {
      continue;
    }
    if (auto constant = getConstInt(value)) {
      shape[index] = *constant;
    }
  }
}

static void mergeKnownShape(ArrayRef<int64_t> knownShape,
                            SmallVectorImpl<int64_t> &shape) {
  if (knownShape.size() != shape.size()) {
    return;
  }
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] == ShapedType::kDynamic &&
        knownShape[index] != ShapedType::kDynamic) {
      shape[index] = knownShape[index];
    }
  }
}

static bool getResolvedPTOViewShape(Value value,
                                    SmallVectorImpl<int64_t> &shape) {
  ArrayRef<int64_t> typeShape;
  ValueRange dynamicShape;
  if (auto make = value.getDefiningOp<MakeTensorViewOp>()) {
    auto type = dyn_cast<TensorViewType>(make.getResult().getType());
    if (!type) {
      return false;
    }
    typeShape = type.getShape();
    dynamicShape = make.getShape();
  } else if (auto partition = value.getDefiningOp<PartitionViewOp>()) {
    auto type =
        dyn_cast<PartitionTensorViewType>(partition.getResult().getType());
    if (!type) {
      return false;
    }
    typeShape = type.getShape();
    dynamicShape = partition.getSizes();
  } else if (auto type = dyn_cast<TensorViewType>(value.getType())) {
    typeShape = type.getShape();
  } else if (auto type =
                 dyn_cast<PartitionTensorViewType>(value.getType())) {
    typeShape = type.getShape();
  } else {
    return false;
  }

  shape.assign(typeShape.begin(), typeShape.end());
  resolveDynamicShape(dynamicShape, shape);
  return true;
}

static bool getMixedMemRefShape(Value value,
                                SmallVectorImpl<int64_t> &shape) {
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    getFoldResultsOrDynamic(reinterpret.getMixedSizes(), shape);
    return true;
  }
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    getFoldResultsOrDynamic(subview.getMixedSizes(), shape);
    return true;
  }
  return false;
}

static Value getMemRefCastSource(Value value) {
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    return cast.getSource();
  }
  if (auto cast = value.getDefiningOp<memref::MemorySpaceCastOp>()) {
    return cast.getSource();
  }
  return Value();
}

static bool getResolvedViewShape(Value value, SmallVectorImpl<int64_t> &shape) {
  if (getResolvedPTOViewShape(value, shape)) {
    return true;
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());

    SmallVector<int64_t> knownShape;
    if (getMixedMemRefShape(value, knownShape)) {
      mergeKnownShape(knownShape, shape);
      return true;
    }

    Value castSource = getMemRefCastSource(value);
    if (castSource && getResolvedViewShape(castSource, knownShape)) {
      mergeKnownShape(knownShape, shape);
    }
    return true;
  }

  return false;
}

static bool verifyNZPartitionView(PartitionViewOp op) {
  SmallVector<int64_t> sourceShape;
  if (!getResolvedViewShape(op.getSource(), sourceShape)) {
    op.emitError(
        "cannot resolve the source shape needed to validate NZ partition");
    return false;
  }
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  getValuesOrDynamic(op.getOffsets(), offsets);
  getValuesOrDynamic(op.getSizes(), sizes);
  if (auto error =
          getNZSubviewCompatibilityError(sourceShape, offsets, sizes)) {
    op.emitError(*error);
    return false;
  }
  return true;
}

static bool verifyNZMemRefSubview(memref::SubViewOp op) {
  auto sourceType = dyn_cast<MemRefType>(op.getSource().getType());
  auto resultType = dyn_cast<MemRefType>(op.getType());
  if (!sourceType || !resultType ||
      sourceType.getRank() != static_cast<int64_t>(kPTOLayoutRank) ||
      resultType.getRank() != sourceType.getRank()) {
    op.emitError("NZ subview requires a rank-preserving rank-5 view");
    return false;
  }

  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> steps;
  getFoldResultsOrDynamic(op.getMixedOffsets(), offsets);
  getFoldResultsOrDynamic(op.getMixedSizes(), sizes);
  getFoldResultsOrDynamic(op.getMixedStrides(), steps);
  if (llvm::any_of(steps, [](int64_t step) { return step != 1; })) {
    op.emitError("NZ subview requires unit steps in every dimension");
    return false;
  }
  SmallVector<int64_t> sourceShape;
  if (!getResolvedViewShape(op.getSource(), sourceShape) ||
      sourceShape.size() != static_cast<size_t>(kPTOLayoutRank)) {
    op.emitError("cannot resolve the source shape needed to validate NZ subview");
    return false;
  }
  if (auto error = getNZSubviewCompatibilityError(sourceShape, offsets, sizes)) {
    op.emitError(*error);
    return false;
  }
  return true;
}

template <typename SignalFailureFn>
static void inferMakeTensorViewLayoutAttr(MakeTensorViewOp op,
                                          SignalFailureFn signalFailure) {
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  bool allStatic = false;
  if (!getShapeAndStride(op, shape, strides, allStatic)) {
    return;
  }

  if (!op.getLayoutAttr()) {
    if (auto typeLayout = getViewTypeLayoutAttr(op.getResult().getType())) {
      setLayoutAttr(op.getOperation(), typeLayout.getLayout(),
                    /*inferred=*/false);
    }
  }

  bool isAmbiguous = false;
  std::optional<Layout> inferred;
  if (allStatic) {
    inferred = inferMakeTensorViewLayout(op, shape, strides, isAmbiguous);
  }
  resolveOrInferLayoutAttr(
      op.getOperation(), shape, strides,
      elemByteSize(
          cast<TensorViewType>(op.getResult().getType()).getElementType()),
      inferred, signalFailure);
  if (allStatic && isAmbiguous &&
      op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName)) {
    reconcileAmbiguousTensorViewLayout(op, shape);
  }
}

template <typename SignalFailureFn>
static void inferReinterpretCastLayoutAttr(memref::ReinterpretCastOp op,
                                           SignalFailureFn signalFailure) {
  auto mrTy = dyn_cast<MemRefType>(op.getType());
  if (!mrTy || !isGlobalMemRef(mrTy)) {
    return;
  }

  const size_t rank = op.getMixedSizes().size();
  if (rank == 0 || rank > kPTOLayoutRank) {
    return;
  }

  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  bool allStatic = true;
  getFoldResults(op.getMixedSizes(), shape, allStatic);
  getFoldResults(op.getMixedStrides(), strides, allStatic);

  bool isMinor2DAmbiguous = false;
  std::optional<Layout> inferred;
  if (allStatic) {
    inferred =
        inferLayout5D(shape, strides, elemByteSize(mrTy.getElementType()),
                      std::nullopt, &isMinor2DAmbiguous);
  }
  resolveOrInferLayoutAttr(op.getOperation(), shape, strides,
                           elemByteSize(mrTy.getElementType()), inferred,
                           signalFailure);
}

struct ViewLayoutNode {
  Value value;
  unsigned parent;
  std::optional<Layout> layout;
};

/// Whole-module equivalence solver for layouts carried by tensor-view SSA
/// values. A missing fact is Unknown; uniting different known layouts is a
/// Conflict. Inferred minor-2D ND/DN choices stay unknown until vector
/// consumers are connected. Remaining unknown components use the historical
/// ND default.
class ViewLayoutSolver {
public:
  explicit ViewLayoutSolver(ModuleOp module)
      : module(module), context(module.getContext()) {}

  LogicalResult run() {
    if (failed(collectValuesAndSeeds())) {
      return failure();
    }
    if (failed(addConstraints())) {
      return failure();
    }
    if (failed(resolveAmbiguousConsumerLayouts())) {
      return failure();
    }
    rewriteAmbiguousMakeTensorViewLayouts();
    rewriteValueTypes();
    rewriteFunctionTypes();
    return success();
  }

private:
  unsigned addValue(Value value) {
    if (!value || !isViewType(value.getType())) {
      return ~0U;
    }
    auto [it, inserted] = ids.try_emplace(value, nodes.size());
    if (!inserted) {
      return it->second;
    }

    std::optional<Layout> layout;
    if (LayoutAttr attr = getViewTypeLayoutAttr(value.getType())) {
      layout = attr.getLayout();
    }
    nodes.push_back(ViewLayoutNode{value, it->second, layout});
    return it->second;
  }

  unsigned find(unsigned id) {
    if (nodes[id].parent == id) {
      return id;
    }
    nodes[id].parent = find(nodes[id].parent);
    return nodes[id].parent;
  }

  LogicalResult reportConflict(Operation *op, Layout lhs, Layout rhs) {
    return op->emitError()
           << "view layout conflict: " << stringifyLayout(lhs) << " and "
           << stringifyLayout(rhs)
           << " reach the same control-flow or function-boundary value";
  }

  LogicalResult seed(Value value, Layout layout, Operation *op) {
    unsigned id = addValue(value);
    if (id == ~0U) {
      return success();
    }
    ViewLayoutNode &root = nodes[find(id)];
    if (root.layout && *root.layout != layout) {
      return reportConflict(op, *root.layout, layout);
    }
    root.layout = layout;
    return success();
  }

  LogicalResult unite(Value lhs, Value rhs, Operation *op) {
    unsigned lhsId = addValue(lhs);
    unsigned rhsId = addValue(rhs);
    if (lhsId == ~0U || rhsId == ~0U) {
      return success();
    }

    unsigned lhsRoot = find(lhsId);
    unsigned rhsRoot = find(rhsId);
    if (lhsRoot == rhsRoot) {
      return success();
    }
    if (nodes[lhsRoot].layout && nodes[rhsRoot].layout &&
        nodes[lhsRoot].layout != nodes[rhsRoot].layout) {
      return reportConflict(op, *nodes[lhsRoot].layout,
                            *nodes[rhsRoot].layout);
    }

    nodes[rhsRoot].parent = lhsRoot;
    if (!nodes[lhsRoot].layout) {
      nodes[lhsRoot].layout = nodes[rhsRoot].layout;
    }
    return success();
  }

  void collectOperationValues(Operation *op) {
    for (Value result : op->getResults()) {
      addValue(result);
    }
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          addValue(argument);
        }
      }
    }
  }

  LogicalResult seedMakeTensorViewLayouts() {
    WalkResult result = module.walk([&](MakeTensorViewOp op) {
      auto layout = op.getLayoutAttr();
      if (!layout) {
        return WalkResult::advance();
      }
      if (isInferredMinor2DAmbiguousLayout(op)) {
        ambiguousMakes.push_back(op);
        return WalkResult::advance();
      }
      return failed(seed(op.getResult(), layout.getLayout(), op))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult seedReturnLayouts(func::ReturnOp returnOp,
                                  FunctionType functionType) {
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= functionType.getNumResults()) {
        break;
      }
      auto layout = getViewTypeLayoutAttr(functionType.getResult(index));
      if (!layout) {
        continue;
      }
      if (failed(seed(operand, layout.getLayout(), returnOp))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult seedFunctionResultLayouts(func::FuncOp function) {
    if (function.empty()) {
      return success();
    }
    FunctionType functionType = function.getFunctionType();
    WalkResult result = function.walk([&](func::ReturnOp returnOp) {
      return failed(seedReturnLayouts(returnOp, functionType))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult seedFunctionResultLayouts() {
    WalkResult result = module.walk([&](func::FuncOp function) {
      return failed(seedFunctionResultLayouts(function))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult collectValuesAndSeeds() {
    module.walk([&](Operation *op) { collectOperationValues(op); });
    if (failed(seedMakeTensorViewLayouts())) {
      return failure();
    }
    return seedFunctionResultLayouts();
  }

  LogicalResult reportAmbiguousConsumerConflict(Operation *consumer) {
    consumer->emitError(
        "ambiguous tensor view has conflicting ND and DN consumer layouts");
    return failure();
  }

  LogicalResult recordAmbiguousConsumerLayout(
      Value view, Type tileType, Operation *consumer,
      const llvm::DenseSet<unsigned> &ambiguousRoots,
      DenseMap<unsigned, LayoutPreference> &preferences) {
    auto preference = isVectorTileType(tileType)
                          ? tileBLayoutToGlobalLayout(tileType)
                          : std::nullopt;
    if (!preference || (*preference != Layout::ND &&
                        *preference != Layout::DN)) {
      return success();
    }

    auto it = ids.find(view);
    if (it == ids.end()) {
      return success();
    }
    unsigned root = find(it->second);
    if (!ambiguousRoots.count(root)) {
      return success();
    }
    if (nodes[root].layout) {
      return success();
    }

    LayoutPreference &result = preferences[root];
    if (result.conflict) {
      return reportAmbiguousConsumerConflict(consumer);
    }
    if (!result.preferred) {
      result.preferred = preference;
      return success();
    }
    if (*result.preferred != *preference) {
      result.preferred = std::nullopt;
      result.conflict = true;
      return reportAmbiguousConsumerConflict(consumer);
    }
    return success();
  }

  LogicalResult resolveAmbiguousConsumerLayouts() {
    llvm::DenseSet<unsigned> ambiguousRoots;
    for (MakeTensorViewOp op : ambiguousMakes) {
      auto it = ids.find(op.getResult());
      if (it != ids.end()) {
        ambiguousRoots.insert(find(it->second));
      }
    }
    if (ambiguousRoots.empty()) {
      return success();
    }

    DenseMap<unsigned, LayoutPreference> preferences;
    WalkResult loads = module.walk([&](pto::TLoadOp op) {
      return failed(recordAmbiguousConsumerLayout(
                 op.getSrc(), op.getDst().getType(), op, ambiguousRoots,
                 preferences))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (loads.wasInterrupted()) {
      return failure();
    }

    WalkResult stores = module.walk([&](pto::TStoreOp op) {
      return failed(recordAmbiguousConsumerLayout(
                 op.getDst(), op.getSrc().getType(), op, ambiguousRoots,
                 preferences))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (stores.wasInterrupted()) {
      return failure();
    }

    for (auto &entry : preferences) {
      LayoutPreference &preference = entry.second;
      if (preference.preferred) {
        nodes[entry.first].layout = preference.preferred;
      }
    }
    return success();
  }

  LogicalResult addYieldConstraints(ResultRange results, scf::YieldOp yield,
                                    Operation *op) {
    for (auto [index, result] : llvm::enumerate(results)) {
      if (index >= yield.getNumOperands()) {
        break;
      }
      if (failed(unite(result, yield.getOperand(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addIfConstraints(scf::IfOp ifOp) {
    for (OpResult result : ifOp->getResults()) {
      unsigned index = result.getResultNumber();
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        if (region->empty()) {
          continue;
        }
        auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
        bool hasOperand = yield && index < yield.getNumOperands();
        if (hasOperand &&
            failed(unite(result, yield.getOperand(index), ifOp))) {
          return failure();
        }
      }
    }
    return success();
  }

  LogicalResult addExecuteRegionConstraints(scf::ExecuteRegionOp execute) {
    Operation *executeOp = execute.getOperation();
    WalkResult result = execute.getRegion().walk([&](scf::YieldOp yield) {
      bool belongsToExecute = yield->getParentOp() == executeOp;
      if (!belongsToExecute) {
        return WalkResult::advance();
      }
      return failed(addYieldConstraints(execute->getResults(), yield, execute))
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addIndexSwitchConstraints(scf::IndexSwitchOp indexSwitch) {
    auto addTerminator = [&](Block &block) {
      auto yield = dyn_cast<scf::YieldOp>(block.getTerminator());
      return yield ? addYieldConstraints(indexSwitch->getResults(), yield,
                                         indexSwitch)
                   : success();
    };
    if (failed(addTerminator(indexSwitch.getDefaultBlock()))) {
      return failure();
    }
    for (unsigned index = 0, end = indexSwitch.getNumCases(); index < end;
         ++index) {
      if (failed(addTerminator(indexSwitch.getCaseBlock(index)))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addBranchConstraints(Block *destination, OperandRange operands,
                                     Operation *op) {
    if (!destination) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(operands)) {
      if (index >= destination->getNumArguments()) {
        break;
      }
      if (failed(unite(operand, destination->getArgument(index), op))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addSwitchConstraints(cf::SwitchOp switchOp) {
    if (failed(addBranchConstraints(switchOp.getDefaultDestination(),
                                    switchOp.getDefaultOperands(), switchOp))) {
      return failure();
    }
    for (auto [destination, operands] :
         llvm::zip(switchOp.getCaseDestinations(), switchOp.getCaseOperands())) {
      if (failed(addBranchConstraints(destination, operands, switchOp))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addReturnConstraints(func::ReturnOp returnOp) {
    func::FuncOp function = returnOp->getParentOfType<func::FuncOp>();
    if (!function) {
      return success();
    }
    auto [it, inserted] = firstReturnOperands.try_emplace(
        function, SmallVector<Value>(returnOp.getOperands()));
    if (inserted) {
      return success();
    }
    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (index >= it->second.size()) {
        break;
      }
      if (failed(unite(it->second[index], operand, returnOp))) {
        return failure();
      }
    }
    return success();
  }

  bool hasViewValueTypes(Operation *op) const {
    return llvm::any_of(op->getOperandTypes(), isViewType) ||
           llvm::any_of(op->getResultTypes(), isViewType);
  }

  LogicalResult seedDeclarationBoundary(func::CallOp call,
                                        func::FuncOp callee) {
    FunctionType type = callee.getFunctionType();
    for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
      bool isViewInput =
          index < type.getNumInputs() && isViewType(type.getInput(index));
      if (!isViewInput) {
        continue;
      }
      Layout layout = Layout::ND;
      if (auto attr = getViewTypeLayoutAttr(type.getInput(index))) {
        layout = attr.getLayout();
      }
      if (failed(seed(operand, layout, call))) {
        return failure();
      }
    }
    for (auto [index, result] : llvm::enumerate(call.getResults())) {
      bool isViewResult =
          index < type.getNumResults() && isViewType(type.getResult(index));
      if (!isViewResult) {
        continue;
      }
      Layout layout = Layout::ND;
      if (auto attr = getViewTypeLayoutAttr(type.getResult(index))) {
        layout = attr.getLayout();
      }
      if (failed(seed(result, layout, call))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addCallConstraints(func::CallOp call) {
    if (!hasViewValueTypes(call)) {
      return success();
    }
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    if (!callee) {
      return call.emitError(
          "view-typed call requires a resolvable direct callee");
    }
    if (callee.empty()) {
      return seedDeclarationBoundary(call, callee);
    }

    for (auto [operand, argument] :
         llvm::zip(call.getOperands(), callee.getArguments())) {
      if (failed(unite(operand, argument, call))) {
        return failure();
      }
    }
    WalkResult result = callee.walk([&](func::ReturnOp returnOp) {
      for (auto [index, result] : llvm::enumerate(call.getResults())) {
        bool hasOperand = index < returnOp.getNumOperands();
        if (hasOperand &&
            failed(unite(result, returnOp.getOperand(index), call))) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  LogicalResult addSelectConstraints(arith::SelectOp select) {
    if (failed(unite(select.getTrueValue(), select.getResult(), select))) {
      return failure();
    }
    return unite(select.getFalseValue(), select.getResult(), select);
  }

  LogicalResult addCastConstraints(UnrealizedConversionCastOp castOp) {
    bool hasPairwiseValues =
        castOp->getNumOperands() == castOp->getNumResults();
    if (!hasPairwiseValues) {
      return success();
    }
    for (auto [operand, result] :
         llvm::zip(castOp->getOperands(), castOp->getResults())) {
      if (failed(unite(operand, result, castOp))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult addForConstraints(scf::ForOp forOp) {
    return VMIControlFlowSupport::addForConstraints(
        forOp, [&](Value lhs, Value rhs, Operation *anchor) {
          return unite(lhs, rhs, anchor);
        });
  }

  LogicalResult addWhileConstraints(scf::WhileOp whileOp) {
    return VMIControlFlowSupport::addWhileConstraints(
        whileOp, [&](Value lhs, Value rhs, Operation *anchor) {
          return unite(lhs, rhs, anchor);
        });
  }

  LogicalResult addCondBranchConstraints(cf::CondBranchOp branch) {
    if (failed(addBranchConstraints(branch.getTrueDest(),
                                    branch.getTrueDestOperands(), branch))) {
      return failure();
    }
    return addBranchConstraints(branch.getFalseDest(),
                                branch.getFalseDestOperands(), branch);
  }

  LogicalResult addIndirectCallConstraints(func::CallIndirectOp call) {
    if (!hasViewValueTypes(call)) {
      return success();
    }
    call.emitError("view-typed indirect calls cannot propagate layout");
    return failure();
  }

  LogicalResult addValueFlowConstraint(Operation *op) {
    if (auto partition = dyn_cast<PartitionViewOp>(op)) {
      return unite(partition.getSource(), partition.getResult(), partition);
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      return addSelectConstraints(select);
    }
    if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      return addCastConstraints(castOp);
    }
    return success();
  }

  LogicalResult addSCFConstraint(Operation *op) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return addIfConstraints(ifOp);
    }
    if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
      return addExecuteRegionConstraints(execute);
    }
    if (auto indexSwitch = dyn_cast<scf::IndexSwitchOp>(op)) {
      return addIndexSwitchConstraints(indexSwitch);
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return addForConstraints(forOp);
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      return addWhileConstraints(whileOp);
    }
    return success();
  }

  LogicalResult addCFConstraint(Operation *op) {
    if (auto branch = dyn_cast<cf::BranchOp>(op)) {
      return addBranchConstraints(branch.getDest(), branch.getDestOperands(),
                                  branch);
    }
    if (auto condBranch = dyn_cast<cf::CondBranchOp>(op)) {
      return addCondBranchConstraints(condBranch);
    }
    if (auto switchOp = dyn_cast<cf::SwitchOp>(op)) {
      return addSwitchConstraints(switchOp);
    }
    return success();
  }

  LogicalResult addFunctionConstraint(Operation *op) {
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return addReturnConstraints(returnOp);
    }
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      return addCallConstraints(callOp);
    }
    if (auto indirectCall = dyn_cast<func::CallIndirectOp>(op)) {
      return addIndirectCallConstraints(indirectCall);
    }
    return success();
  }

  LogicalResult addConstraint(Operation *op) {
    if (failed(addValueFlowConstraint(op))) {
      return failure();
    }
    if (failed(addSCFConstraint(op))) {
      return failure();
    }
    if (failed(addCFConstraint(op))) {
      return failure();
    }
    if (failed(addFunctionConstraint(op))) {
      return failure();
    }
    return success();
  }

  LogicalResult addConstraints() {
    WalkResult result = module.walk([&](Operation *op) {
      return failed(addConstraint(op)) ? WalkResult::interrupt()
                                       : WalkResult::advance();
    });
    return failure(result.wasInterrupted());
  }

  Layout getLayout(Value value) {
    auto it = ids.find(value);
    if (it == ids.end()) {
      return Layout::ND;
    }
    return nodes[find(it->second)].layout.value_or(Layout::ND);
  }

  void rewriteAmbiguousMakeTensorViewLayouts() {
    for (MakeTensorViewOp op : ambiguousMakes) {
      setLayoutAttr(op.getOperation(), getLayout(op.getResult()),
                    /*inferred=*/true);
    }
  }

  void rewriteValueTypes() {
    for (ViewLayoutNode &node : nodes) {
      Type newType = getViewTypeWithLayout(node.value.getType(),
                                           getLayout(node.value));
      node.value.setType(newType);
    }
  }

  void rewriteFunctionTypes() {
    module.walk([&](func::FuncOp function) {
      FunctionType oldType = function.getFunctionType();
      SmallVector<Type> inputs;
      if (function.empty()) {
        inputs.assign(oldType.getInputs().begin(), oldType.getInputs().end());
      } else {
        for (BlockArgument argument : function.getArguments()) {
          inputs.push_back(argument.getType());
        }
      }

      SmallVector<Type> results(oldType.getResults().begin(),
                                oldType.getResults().end());
      auto it = firstReturnOperands.find(function);
      if (it != firstReturnOperands.end()) {
        for (auto [index, operand] : llvm::enumerate(it->second)) {
          if (index < results.size()) {
            results[index] = operand.getType();
          }
        }
      }
      function.setFunctionType(FunctionType::get(context, inputs, results));
    });
  }

  ModuleOp module;
  MLIRContext *context;
  DenseMap<Value, unsigned> ids;
  SmallVector<ViewLayoutNode> nodes;
  SmallVector<MakeTensorViewOp> ambiguousMakes;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperands;
};

struct InferPTOLayoutPass
    : public mlir::pto::impl::InferPTOLayoutBase<InferPTOLayoutPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferPTOLayoutPass)

  StringRef getArgument() const final { return "pto-infer-layout"; }

  StringRef getDescription() const final {
    return "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // ------------------------------------------------------------------
    // 1) pto.make_tensor_view (only if it still exists in the pipeline)
    // ------------------------------------------------------------------
    bool inferenceFailed = false;
    module.walk([&](MakeTensorViewOp op) {
      inferMakeTensorViewLayoutAttr(op, [this, &inferenceFailed] {
        inferenceFailed = true;
        signalPassFailure();
      });
    });
    if (inferenceFailed) {
      return;
    }

    // ------------------------------------------------------------------
    // 2) Propagate view layouts through SSA merges and direct calls. This
    // rewrites non-ND view types so downstream structural type conversion can
    // preserve the layout without relying on a defining-op walk.
    // ------------------------------------------------------------------
    ViewLayoutSolver solver(module);
    if (failed(solver.run())) {
      signalPassFailure();
      return;
    }

    // ------------------------------------------------------------------
    // 3) pto.partition_view: validate against the resolved source layout.
    // The logical inheritance is resolved through the source chain and later
    // materialized on the lowered memref.subview. Avoid adding derived
    // attributes here so the source remains the single authority.
    // ------------------------------------------------------------------
    module.walk([&](PartitionViewOp op) {
      auto sourceInfo = resolveLayoutFromViewValue(op.getSource());
      if (!sourceInfo.layout) {
        return;
      }
      if (*sourceInfo.layout == Layout::NZ && !verifyNZPartitionView(op)) {
        signalPassFailure();
        return;
      }
      if (auto existing = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
          existing && existing.getLayout() != *sourceInfo.layout) {
        op.emitError() << "partition layout="
                       << stringifyLayout(existing.getLayout())
                       << " does not match source layout="
                       << stringifyLayout(*sourceInfo.layout);
        signalPassFailure();
        return;
      }
    });

    // ------------------------------------------------------------------
    // 4) memref.reinterpret_cast (lowered from make_tensor_view)
    // ------------------------------------------------------------------
    module.walk([&](memref::ReinterpretCastOp op) {
      inferReinterpretCastLayoutAttr(op, [this] { signalPassFailure(); });
    });

    // ------------------------------------------------------------------
    // 5) memref.subview: preserve layout only across a legal derived view.
    // ------------------------------------------------------------------
    module.walk([&](memref::SubViewOp op) {
      auto resTy = dyn_cast<MemRefType>(op.getType());
      if (!resTy || !isGlobalMemRef(resTy)) {
        return;
      }

      auto sourceInfo = resolveLayoutFromViewValue(op.getSource());
      auto existing = op->getAttrOfType<LayoutAttr>(kLayoutAttrName);
      if (existing && sourceInfo.layout &&
          existing.getLayout() != *sourceInfo.layout) {
        op.emitError() << "subview layout="
                       << stringifyLayout(existing.getLayout())
                       << " does not match source layout="
                       << stringifyLayout(*sourceInfo.layout);
        signalPassFailure();
        return;
      }

      if (existing || sourceInfo.layout) {
        Layout layout = existing ? existing.getLayout() : *sourceInfo.layout;
        auto existingInferred =
            op->getAttrOfType<BoolAttr>(kInferredLayoutAttrName);
        bool inferred =
            existing ? (existingInferred && existingInferred.getValue())
                     : sourceInfo.inferred;
        if (layout == Layout::NZ && !verifyNZMemRefSubview(op)) {
          signalPassFailure();
          return;
        }
        setLayoutAttr(op.getOperation(), layout, inferred);
        return;
      }

      // Fallback: if source memref type is fully static, infer from it.
      auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
      if (!srcTy || !srcTy.hasStaticShape()) {
        setLayoutAttr(op.getOperation(), Layout::ND, /*inferred=*/true);
        return;
      }

      SmallVector<int64_t> strideInts;
      int64_t offset = ShapedType::kDynamic;
      if (failed(mlir::pto::getPTOMemRefStridesAndOffset(srcTy, strideInts,
                                                         offset)) ||
          offset == ShapedType::kDynamic ||
          llvm::any_of(strideInts,
                       [](int64_t s) { return s == ShapedType::kDynamic; })) {
        setLayoutAttr(op.getOperation(), Layout::ND, /*inferred=*/true);
        return;
      }

      auto inferred = inferLayout5D(srcTy.getShape(), strideInts,
                                    elemByteSize(srcTy.getElementType()));
      if (inferred == Layout::NZ && !verifyNZMemRefSubview(op)) {
        signalPassFailure();
        return;
      }
      setLayoutAttr(op.getOperation(), inferred.value_or(Layout::ND),
                    /*inferred=*/true);
    });

    // ------------------------------------------------------------------
    // 6) pto.tload / pto.tstore: attach layout for static GM memrefs so EmitC
    //    doesn't need to infer again in buildGlobalTensorFromMemref().
    // ------------------------------------------------------------------
    module.walk([&](pto::TLoadOp op) {
      attachLoadStoreLayout(op, [](auto load) { return load.getSrc(); },
                            [](auto load) { return load.getDst(); });
    });

    module.walk([&](pto::TStoreOp op) {
      attachLoadStoreLayout(op, [](auto store) { return store.getDst(); },
                            [](auto store) { return store.getSrc(); });
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createInferPTOLayoutPass() {
  return std::make_unique<InferPTOLayoutPass>();
}
