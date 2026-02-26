#ifndef MLIR_C_DIALECT_PTO_H
#define MLIR_C_DIALECT_PTO_H

#include "mlir-c/IR.h" 

#ifdef __cplusplus
extern "C" {
#endif

// Provides: mlirGetDialectHandle__pto__()
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(PTO, pto);

// ---- !pto.ptr<elem> ----
bool mlirPTOTypeIsAPtrType(MlirType type);
MlirType mlirPTOPtrTypeGet(MlirContext ctx, MlirType elementType);
MlirType mlirPTOPtrTypeGetElementType(MlirType type);

// ---- #pto.address_space<...> ----
bool mlirPTOAttrIsAAddressSpaceAttr(MlirAttribute attr);

// Create: #pto.address_space<ub/gm/...>
MlirAttribute mlirPTOAddressSpaceAttrGet(MlirContext ctx, int32_t value);

// Read back enum value (0..6)
int32_t mlirPTOAddressSpaceAttrGetValue(MlirAttribute attr);

// ---- !pto.tensor_view<shape x elem> ----
bool mlirPTOTypeIsATensorViewType(MlirType type);
MlirType mlirPTOTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                  const int64_t *shape, MlirType elementType);
intptr_t mlirPTOTensorViewTypeGetRank(MlirType type);
MlirType mlirPTOTensorViewTypeGetElementType(MlirType type);
// 返回内部 shape 数组指针（只读）；numDimsOut 返回维度数
const int64_t *mlirPTOTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- !pto.partition_tensor_view<shape x elem> ----
bool mlirPTOTypeIsAPartitionTensorViewType(MlirType type);
MlirType mlirPTOPartitionTensorViewTypeGet(MlirContext ctx, intptr_t rank,
                                           const int64_t *shape, MlirType elementType);
intptr_t mlirPTOPartitionTensorViewTypeGetRank(MlirType type);
MlirType mlirPTOPartitionTensorViewTypeGetElementType(MlirType type);
// 返回内部 shape 数组指针（只读）；numDimsOut 返回维度数
const int64_t *mlirPTOPartitionTensorViewTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- !pto.tile<shape x elem> ----
bool mlirPTOTypeIsATileType(MlirType type);
MlirType mlirPTOTileTypeGet(MlirContext ctx, intptr_t rank,
                            const int64_t *shape, MlirType elementType);
intptr_t mlirPTOTileTypeGetRank(MlirType type);
MlirType mlirPTOTileTypeGetElementType(MlirType type);
const int64_t *mlirPTOTileTypeGetShape(MlirType type, intptr_t *numDimsOut);

// ---- TileBufType ----
MLIR_CAPI_EXPORTED bool mlirPTOTypeIsATileBufType(MlirType type);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGet(
    MlirContext ctx, intptr_t rank, const int64_t *shape,
    MlirType elementType, MlirAttribute memorySpace);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithConfig(
    MlirContext ctx, intptr_t rank, const int64_t *shape,
    MlirType elementType, MlirAttribute memorySpace, MlirAttribute config);
// ---- Enum attrs helpers (BLayout/SLayout/PadValue in mlir::pto) ----
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsABLayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOBLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOBLayoutAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASLayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOSLayoutAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAPadValueAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOPadValueAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED int32_t mlirPTOPadValueAttrGetValue(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTORoundModeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsARoundModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTORoundModeAttrGetValue(MlirAttribute attr);
// ---- Pipe attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOPipeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAPipeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOPipeAttrGetValue(MlirAttribute attr);
// ---- Layout attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOLayoutAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsALayoutAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOLayoutAttrGetValue(MlirAttribute attr);
// ---- SyncOpType attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOSyncOpTypeAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsASyncOpTypeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOSyncOpTypeAttrGetValue(MlirAttribute attr);
// ---- Event attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOEventAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAEventAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOEventAttrGetValue(MlirAttribute attr);
// ---- MaskPattern attr ----
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOMaskPatternAttrGet(MlirContext ctx, int32_t value);
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsAMaskPatternAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED int32_t mlirPTOMaskPatternAttrGetValue(MlirAttribute attr);

// ---- CmpMode (compare mode for cmp/cvt) ----
typedef enum MlirPTOCmpMode {
  MlirPTOCmpMode_EQ = 0,
  MlirPTOCmpMode_NE = 1,
  MlirPTOCmpMode_LT = 2,
  MlirPTOCmpMode_LE = 3,
  MlirPTOCmpMode_GT = 4,
  MlirPTOCmpMode_GE = 5,
} MlirPTOCmpMode;
MLIR_CAPI_EXPORTED bool mlirAttributeIsAPTOCmpModeAttr(MlirAttribute attr);
MLIR_CAPI_EXPORTED MlirAttribute mlirPTOCmpModeAttrGet(MlirContext ctx, MlirPTOCmpMode value);
MLIR_CAPI_EXPORTED MlirPTOCmpMode mlirPTOCmpModeAttrGetValue(MlirAttribute attr);
// ---- TileBufConfigAttr ----
MLIR_CAPI_EXPORTED bool mlirPTOAttrIsATileBufConfigAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTileBufConfigAttrGetDefault(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute mlirPTOTileBufConfigAttrGet(
    MlirContext ctx,
    MlirAttribute bLayout, MlirAttribute sLayout,
    MlirAttribute sFractalSize, MlirAttribute pad);
MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithValidShape(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType,
    MlirAttribute memorySpace, intptr_t validRank, const int64_t *validShape);

MLIR_CAPI_EXPORTED MlirType mlirPTOTileBufTypeGetWithValidShapeAndConfig(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType,
    MlirAttribute memorySpace, intptr_t validRank, const int64_t *validShape,
    MlirAttribute config);

// ---- MemRef helpers ----
MLIR_CAPI_EXPORTED MlirType mlirPTOGMTypeGet(
    MlirContext ctx, intptr_t rank, const int64_t *shape, MlirType elementType);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_PTO_H
