//===- PTO.cpp - C API for PTO dialect -----------------------------------===//
//
// This file provides the C API for the PTO dialect and its custom types.
//
// It must be built into an MLIR CAPI library (e.g. MLIRCAPIPTO) and linked
// by any consumers (e.g. Python extension).
//
//===----------------------------------------------------------------------===//

#include "pto-c/Dialect/PTO.h"

// unwrap/wrap + MLIR dialect registration C-API support.
#include "mlir/CAPI/IR.h"

#include "mlir/CAPI/Registration.h"
#include "llvm/ADT/SmallVector.h"

// IMPORTANT: include the C++ dialect header that declares PtrType/TensorViewType.
// This header should itself include the generated PTOTypeDefs.h.inc.
#include "PTO/IR/PTO.h"

using namespace mlir;

// Dialect registration (provides mlirGetDialectHandle__pto__()).
// NOTE: adjust the third argument if your dialect class name/namespace differs.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(PTO, pto, mlir::pto::PTODialect)

//===----------------------------------------------------------------------===//
// Type queries / constructors for !pto.ptr<elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAPtrType(MlirType type) {
  return isa<mlir::pto::PtrType>(unwrap(type));;
}

MlirType mlirPTOPtrTypeGet(MlirContext ctx, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  return wrap(mlir::pto::PtrType::get(c, elem));
}

MlirType mlirPTOPtrTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::PtrType>(unwrap(type));;
  return wrap(t.getElementType());
}

bool mlirPTOAttrIsAAddressSpaceAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::AddressSpaceAttr>(unwrap(attr));
}

MlirAttribute mlirPTOAddressSpaceAttrGet(MlirContext ctx, int32_t value) {
  auto c = unwrap(ctx);

  // 你的 ODS 里 AddressSpaceAttr 的参数是 EnumParameter<PTO_AddressSpaceEnum>
  // 通常对应 C++ 里是一个 enum class AddressSpace : int32_t
  auto v = static_cast<mlir::pto::AddressSpace>(value);

  return wrap(mlir::pto::AddressSpaceAttr::get(c, v));
}

int32_t mlirPTOAddressSpaceAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::AddressSpaceAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getAddressSpace());
}

//===----------------------------------------------------------------------===//
// Type queries / constructors for !pto.tensor_view<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsATensorViewType(MlirType type) {
  return isa<mlir::pto::TensorViewType>(unwrap(type));
}

MlirType mlirPTOTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                  const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::TensorViewType::get(c, shp, elem));
}

intptr_t mlirPTOTensorViewTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOTensorViewTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::TensorViewType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

//===----------------------------------------------------------------------===//
// !pto.tile_view<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsAPartitionTensorViewType(MlirType type) {
  return isa<mlir::pto::PartitionTensorViewType>(unwrap(type));
}

MlirType mlirPTOPartitionTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::PartitionTensorViewType::get(c, shp, elem));
}

intptr_t mlirPTOPartitionTensorViewTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOPartitionTensorViewTypeGetElementType(MlirType type) {
  auto t = mlir::cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOPartitionTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::PartitionTensorViewType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

//===----------------------------------------------------------------------===//
// !pto.tile<shape x elem>
//===----------------------------------------------------------------------===//

bool mlirPTOTypeIsATileType(MlirType type) {
  return isa<mlir::pto::TileType>(unwrap(type));
}

MlirType mlirPTOTileTypeGet(MlirContext ctx, intptr_t rank,
                            const int64_t *shape, MlirType elementType) {
  auto c = unwrap(ctx);
  auto elem = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));
  return wrap(mlir::pto::TileType::get(c, shp, elem));
}

intptr_t mlirPTOTileTypeGetRank(MlirType type) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  return static_cast<intptr_t>(t.getShape().size());
}

MlirType mlirPTOTileTypeGetElementType(MlirType type) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  return wrap(t.getElementType());
}

const int64_t *mlirPTOTileTypeGetShape(MlirType type, intptr_t *numDimsOut) {
  auto t = cast<mlir::pto::TileType>(unwrap(type));
  auto shape = t.getShape();
  *numDimsOut = static_cast<intptr_t>(shape.size());
  return shape.data();
}

bool mlirPTOTypeIsATileBufType(MlirType type) {
  return unwrap(type).isa<mlir::pto::TileBufType>();
}

MlirType mlirPTOTileBufTypeGet(MlirContext ctx, intptr_t rank,
                               const int64_t *shape, MlirType elementType,
                               MlirAttribute memorySpace) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(c);
  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType), unwrap(memorySpace), llvm::ArrayRef<int64_t>{}, cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithConfig(MlirContext ctx, intptr_t rank,
                                         const int64_t *shape, MlirType elementType,
                                         MlirAttribute memorySpace, MlirAttribute config) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto cfg = unwrap(config).dyn_cast_or_null<mlir::pto::TileBufConfigAttr>();
  if (!cfg) cfg = mlir::pto::TileBufConfigAttr::getDefault(c);
  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType), unwrap(memorySpace), cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithValidShape(MlirContext ctx,
                                             intptr_t rank,
                                             const int64_t *shape,
                                             MlirType elementType,
                                             MlirAttribute memorySpace,
                                             intptr_t validRank,
                                             const int64_t *validShape) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto vs  = llvm::ArrayRef<int64_t>(validShape, validRank);
  auto cfg = mlir::pto::TileBufConfigAttr::getDefault(c);

  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType),
                                       unwrap(memorySpace), vs, cfg);
  return wrap(ty);
}

MlirType mlirPTOTileBufTypeGetWithValidShapeAndConfig(MlirContext ctx,
                                                      intptr_t rank,
                                                      const int64_t *shape,
                                                      MlirType elementType,
                                                      MlirAttribute memorySpace,
                                                      intptr_t validRank,
                                                      const int64_t *validShape,
                                                      MlirAttribute config) {
  MLIRContext *c = unwrap(ctx);
  auto shp = llvm::ArrayRef<int64_t>(shape, rank);
  auto vs  = llvm::ArrayRef<int64_t>(validShape, validRank);
  auto cfg = mlir::cast<mlir::pto::TileBufConfigAttr>(unwrap(config));

  auto ty = mlir::pto::TileBufType::get(c, shp, unwrap(elementType),
                                       unwrap(memorySpace), vs, cfg);
  return wrap(ty);
}

bool mlirPTOAttrIsABLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::BLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirPTOBLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::BLayout>(value);
  return wrap(mlir::pto::BLayoutAttr::get(c, v));
}

int32_t mlirPTOBLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::BLayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsASLayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::SLayoutAttr>(unwrap(attr));
}

MlirAttribute mlirPTOSLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::SLayout>(value);
  return wrap(mlir::pto::SLayoutAttr::get(c, v));
}

int32_t mlirPTOSLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::SLayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirPTOAttrIsAPadValueAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::PadValueAttr>(unwrap(attr));
}

MlirAttribute mlirPTOPadValueAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::PadValue>(value);
  return wrap(mlir::pto::PadValueAttr::get(c, v));
}

int32_t mlirPTOPadValueAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::PadValueAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTORoundModeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::RoundMode>(value);
  return wrap(mlir::pto::RoundModeAttr::get(c, mode));
}

bool mlirPTOAttrIsARoundModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::RoundModeAttr>(unwrap(attr));
}

int32_t mlirPTORoundModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::RoundModeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

MlirAttribute mlirPTOPipeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::PIPE>(value);
  return wrap(mlir::pto::PipeAttr::get(c, v));
}

bool mlirPTOAttrIsAPipeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::PipeAttr>(unwrap(attr));
}

int32_t mlirPTOPipeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::PipeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getPipe());
}

MlirAttribute mlirPTOLayoutAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::Layout>(value);
  return wrap(mlir::pto::LayoutAttr::get(c, v));
}

bool mlirPTOAttrIsALayoutAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::LayoutAttr>(unwrap(attr));
}

int32_t mlirPTOLayoutAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::LayoutAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getLayout());
}

MlirAttribute mlirPTOSyncOpTypeAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::SyncOpType>(value);
  return wrap(mlir::pto::SyncOpTypeAttr::get(c, mode));
}

bool mlirPTOAttrIsASyncOpTypeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::SyncOpTypeAttr>(unwrap(attr));
}

int32_t mlirPTOSyncOpTypeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::SyncOpTypeAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getOpType());
}

MlirAttribute mlirPTOEventAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::EVENT>(value);
  return wrap(mlir::pto::EventAttr::get(c, v));
}

bool mlirPTOAttrIsAEventAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::EventAttr>(unwrap(attr));
}

int32_t mlirPTOEventAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::EventAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getEvent());
}

MlirAttribute mlirPTOMaskPatternAttrGet(MlirContext ctx, int32_t value) {
  auto *c = unwrap(ctx);
  auto v = static_cast<mlir::pto::MaskPattern>(value);
  return wrap(mlir::pto::MaskPatternAttr::get(c, v));
}

bool mlirPTOAttrIsAMaskPatternAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::MaskPatternAttr>(unwrap(attr));
}

int32_t mlirPTOMaskPatternAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::MaskPatternAttr>(unwrap(attr));
  return static_cast<int32_t>(a.getValue());
}

bool mlirAttributeIsAPTOCmpModeAttr(MlirAttribute attr) {
  return mlir::isa<mlir::pto::CmpModeAttr>(unwrap(attr));
}

MlirAttribute mlirPTOCmpModeAttrGet(MlirContext ctx, MlirPTOCmpMode value) {
  auto *c = unwrap(ctx);
  auto mode = static_cast<mlir::pto::CmpMode>(value);
  return wrap(mlir::pto::CmpModeAttr::get(c, mode));
}

MlirPTOCmpMode mlirPTOCmpModeAttrGetValue(MlirAttribute attr) {
  auto a = mlir::cast<mlir::pto::CmpModeAttr>(unwrap(attr));
  return static_cast<MlirPTOCmpMode>(static_cast<uint32_t>(a.getValue()));
}

bool mlirPTOAttrIsATileBufConfigAttr(MlirAttribute attr) {
  return unwrap(attr).isa<mlir::pto::TileBufConfigAttr>();
}

MlirAttribute mlirPTOTileBufConfigAttrGetDefault(MlirContext ctx) {
  auto *c = unwrap(ctx);
  return wrap(mlir::pto::TileBufConfigAttr::getDefault(c));
}

static mlir::pto::BLayoutAttr toBLayoutAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto bl = mlir::dyn_cast<mlir::pto::BLayoutAttr>(a)) return bl;
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
    return mlir::pto::BLayoutAttr::get(c, static_cast<mlir::pto::BLayout>(ia.getInt()));
  return {};
}
static mlir::pto::SLayoutAttr toSLayoutAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto sl = mlir::dyn_cast<mlir::pto::SLayoutAttr>(a)) return sl;
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
    return mlir::pto::SLayoutAttr::get(c, static_cast<mlir::pto::SLayout>(ia.getInt()));
  return {};
}
static mlir::pto::PadValueAttr toPadValueAttr(mlir::MLIRContext *c, mlir::Attribute a) {
  if (auto pv = mlir::dyn_cast<mlir::pto::PadValueAttr>(a)) return pv;
  if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a))
    return mlir::pto::PadValueAttr::get(c, static_cast<mlir::pto::PadValue>(ia.getInt()));
  return {};
}

MlirAttribute mlirPTOTileBufConfigAttrGet(MlirContext ctx,
                                          MlirAttribute bLayout,
                                          MlirAttribute sLayout,
                                          MlirAttribute sFractalSize,
                                          MlirAttribute pad) {
  auto *c = unwrap(ctx);
  auto blA = toBLayoutAttr(c, unwrap(bLayout));
  auto slA = toSLayoutAttr(c, unwrap(sLayout));
  auto pvA = toPadValueAttr(c, unwrap(pad));
  if (!blA || !slA || !pvA)
    return MlirAttribute{nullptr};

  auto sz = mlir::dyn_cast<mlir::IntegerAttr>(unwrap(sFractalSize));
  if (!sz || !sz.getType().isInteger(32))
    return MlirAttribute{nullptr};

  return wrap(mlir::pto::TileBufConfigAttr::get(c, blA, slA, sz, pvA));
}

MlirType mlirPTOGMTypeGet(MlirContext ctx, intptr_t rank, const int64_t *shape,
                          MlirType elementType) {
  auto *c = unwrap(ctx);
  auto elemTy = unwrap(elementType);
  llvm::ArrayRef<int64_t> shp(shape, static_cast<size_t>(rank));

  llvm::SmallVector<int64_t, 8> strides(static_cast<size_t>(rank),
                                        ShapedType::kDynamic);
  if (rank > 0)
    strides[static_cast<size_t>(rank) - 1] = 1;
  auto layout =
      StridedLayoutAttr::get(c, ShapedType::kDynamic, llvm::ArrayRef<int64_t>(strides));
  auto memSpace = mlir::pto::AddressSpaceAttr::get(c, mlir::pto::AddressSpace::GM);

  return wrap(MemRefType::get(shp, elemTy, layout, memSpace));
}
