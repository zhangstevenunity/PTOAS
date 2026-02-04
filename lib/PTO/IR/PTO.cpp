//===- PTO.cpp - PTO Dialect ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"

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
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include <numeric>
#include <optional>

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
static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType);
} // namespace pto
} // namespace mlir

// =============================================================================
// TileBufType 的自定义 Shape 解析与打印函数
// =============================================================================

// 解析逻辑：解析形如 "32x32" 的维度列表
static ParseResult parseShape(AsmParser &parser, SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 会解析 "dim x dim x ...", 遇到无法解析为维度的字符停止
  // 参数 allowDynamic=true (允许 ?), withTrailingX=false (不吞掉末尾的 x)
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/false))
    return failure();
  return success();
}

// 打印逻辑：打印形如 "32x32" 的维度列表
static void printShape(AsmPrinter &printer, ArrayRef<int64_t> shape) {
  for (auto it = shape.begin(); it != shape.end(); ++it) {
    if (it != shape.begin()) printer << "x"; // 维度间的分隔符
    if (*it == ShapedType::kDynamic)
      printer << "?";
    else
      printer << *it;
  }
  // 注意：我们不在这里打印末尾的 'x'，因为 assemblyFormat 中已经写了 `x` $elementType
}

#define GET_ENUM_CLASSES
#include "PTO/IR/PTOEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/PTOTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "PTO/IR/PTOAttrs.cpp.inc"

#include "PTO/IR/PTODialect.cpp.inc"

static LogicalResult parseShapeAndElemStable(mlir::AsmParser &parser,
                                             llvm::SmallVectorImpl<int64_t> &shape,
                                             mlir::Type &elementType) {
  if (failed(parser.parseLess()))
    return failure();

  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
    return failure();

  if (failed(parser.parseType(elementType)))
    return failure();

  if (failed(parser.parseGreater()))
    return failure();

  return success();
}

static int64_t getPTOTypeRank(Type type) {
  // 1. 处理标准的 MLIR 类型 (MemRef, Tensor, Vector)
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    if (shapedTy.hasRank())
      return shapedTy.getRank();
    return -1; // Unranked type
  }
  
  // 2. 处理 PTO 自定义类型
  if (auto tvTy = dyn_cast<pto::TensorViewType>(type))
    return tvTy.getRank();

  if (auto tileTy = dyn_cast<pto::TileType>(type))
    return tileTy.getRank();
    
  if (auto tileViewTy = dyn_cast<pto::PartitionTensorViewType>(type))
    return tileViewTy.getRank();

  if (auto tileBufTy = dyn_cast<pto::TileBufType>(type))
    return tileBufTy.getRank();

  // 3. 不支持的类型
  return -1;
}

static mlir::Type parsePTOTypeAllowNoBang(mlir::OpAsmParser &parser) {
  mlir::Type ty;

  mlir::OptionalParseResult opt = parser.parseOptionalType(ty);

  if (opt.has_value()) {         
    if (failed(*opt))
      return mlir::Type();       
    return ty;                    
  }


  llvm::StringRef head;
  if (failed(parser.parseKeyword(&head)))
    return mlir::Type();

  mlir::MLIRContext *ctx = parser.getContext();

  auto parseShapeElemForOpParser =
      [&](llvm::SmallVectorImpl<int64_t> &shape, mlir::Type &elem) -> mlir::LogicalResult {
        if (failed(parser.parseLess()))
          return failure();
        if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/true)))
          return failure();
        if (failed(parser.parseType(elem)))
          return failure();
        if (failed(parser.parseGreater()))
          return failure();
        return success();
      };

  if (head == "pto.tile_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::PartitionTensorViewType::get(ctx, shape, elem);
  }

  if (head == "pto.tile") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TileType::get(ctx, shape, elem);
  }

  if (head == "pto.ptr") {
    if (failed(parser.parseLess()))
      return mlir::Type();
    mlir::Type elem;
    if (failed(parser.parseType(elem)) || failed(parser.parseGreater()))
      return mlir::Type();
    return mlir::pto::PtrType::get(ctx, elem);
  }

  if (head == "pto.tensor_view") {
    llvm::SmallVector<int64_t, 4> shape;
    mlir::Type elem;
    if (failed(parseShapeElemForOpParser(shape, elem)))
      return mlir::Type();
    return mlir::pto::TensorViewType::get(ctx, shape, elem);
  }

  return mlir::Type();
}


ParseResult LoadOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand src;
  if (parser.parseOperand(src))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  Type srcTy = parsePTOTypeAllowNoBang(parser);
  if (!srcTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse source type");

  if (parser.parseArrow()) return failure();

  Type resTy = parsePTOTypeAllowNoBang(parser);
  if (!resTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse result type");

  if (parser.resolveOperand(src, srcTy, result.operands))
    return failure();

  result.addTypes(resTy);
  return success();
}

void LoadOp::print(OpAsmPrinter &p) {
  p << " " << getSrc();

  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getSrc().getType() << " -> " << getResult().getType();
}

mlir::Type TensorViewType::parse(::mlir::AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elementType;
  if (failed(parseShapeAndElem(parser, shape, elementType, /*allowDynamic=*/true)))
    return Type();
  return TensorViewType::get(parser.getContext(), shape, elementType);
}

void TensorViewType::print(::mlir::AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

//===----------------------------------------------------------------------===//
// pto.store custom asm to support `-> ()` exactly (IR text unchanged)
//===----------------------------------------------------------------------===//

ParseResult StoreOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand dst, src;
  Type dstTy, srcTy;

  // %dst, %src
  if (parser.parseOperand(dst) || parser.parseComma() || parser.parseOperand(src))
    return failure();

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // : (type(dst), type(src)) -> ()
  if (parser.parseColon() || parser.parseLParen())
    return failure();

  dstTy = parsePTOTypeAllowNoBang(parser);
  if (!dstTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse dst type");
  if (parser.parseComma()) return failure();
  srcTy = parsePTOTypeAllowNoBang(parser);
  if (!srcTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse src type");
  if (parser.parseRParen()) return failure();
  if (parser.parseArrow() || parser.parseLParen() || parser.parseRParen()) return failure();

  if (parser.resolveOperand(dst, dstTy, result.operands)) return failure();
  if (parser.resolveOperand(src, srcTy, result.operands)) return failure();

  return success();
}

void StoreOp::print(OpAsmPrinter &p) {
  p << " " << getDst() << ", " << getSrc();

  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : (" << getDst().getType() << ", " << getSrc().getType() << ") -> ()";
}

//===----------------------------------------------------------------------===//
// pto.tdivs custom asm to support both:
//   pto.tdivs ins(%src, %scalar : !pto.tile_buf<...>, f32) outs(%dst : !pto.tile_buf<...>)
//   pto.tdivs ins(%scalar, %src : f32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
// The operand order in the op remains (src, scalar, dst); scalar_lhs is derived
// from the textual order when not explicitly specified.
//===----------------------------------------------------------------------===//

ParseResult mlir::pto::TDivSOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand op0, op1, dst;
  Type ty0, ty1, dstTy;

  if (parser.parseKeyword("ins") || parser.parseLParen() ||
      parser.parseOperand(op0) || parser.parseComma() ||
      parser.parseOperand(op1) || parser.parseColonType(ty0) ||
      parser.parseComma() || parser.parseType(ty1) || parser.parseRParen())
    return failure();

  if (parser.parseKeyword("outs") || parser.parseLParen() ||
      parser.parseOperand(dst) || parser.parseColonType(dstTy) ||
      parser.parseRParen())
    return failure();

  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs))
    return failure();

  auto tile0 = dyn_cast<mlir::pto::TileBufType>(ty0);
  auto tile1 = dyn_cast<mlir::pto::TileBufType>(ty1);
  if ((tile0 && tile1) || (!tile0 && !tile1))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one tile_buf operand and one scalar operand");

  if (!dyn_cast<mlir::pto::TileBufType>(dstTy))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected outs type to be !pto.tile_buf<...>");

  const bool scalarFirst = (tile1 != nullptr);
  const bool scalarLhs = scalarFirst;

  if (Attribute a = attrs.get("scalar_lhs")) {
    auto b = dyn_cast<BoolAttr>(a);
    if (!b)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected scalar_lhs to be a bool attribute");
    if (b.getValue() != scalarLhs)
      return parser.emitError(parser.getCurrentLocation(),
                              "scalar_lhs attribute conflicts with operand order");
  } else if (scalarLhs) {
    attrs.set("scalar_lhs", BoolAttr::get(parser.getContext(), true));
  }

  if (!scalarFirst) {
    // ins(%src, %scalar : tile_buf, scalar_ty)
    if (parser.resolveOperand(op0, ty0, result.operands) ||
        parser.resolveOperand(op1, ty1, result.operands))
      return failure();
  } else {
    // ins(%scalar, %src : scalar_ty, tile_buf)
    if (parser.resolveOperand(op1, ty1, result.operands) ||
        parser.resolveOperand(op0, ty0, result.operands))
      return failure();
  }

  if (parser.resolveOperand(dst, dstTy, result.operands))
    return failure();

  result.addAttributes(attrs);
  return success();
}

void mlir::pto::TDivSOp::print(OpAsmPrinter &p) {
  bool scalarLhs = false;
  if (auto a = getScalarLhsAttr())
    scalarLhs = a.getValue();

  p << " ins(";
  if (!scalarLhs) {
    p << getSrc() << ", " << getScalar() << " : "
      << getSrc().getType() << ", " << getScalar().getType();
  } else {
    p << getScalar() << ", " << getSrc() << " : "
      << getScalar().getType() << ", " << getSrc().getType();
  }
  p << ") outs(" << getDst() << " : " << getDst().getType() << ")";

  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"scalar_lhs"});
}

ParseResult mlir::pto::MakeTensorViewOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  OpAsmParser::UnresolvedOperand ptr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> shapeOps;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> strideOps;

  Type resultTy;

  // %ptr
  if (parser.parseOperand(ptr))
    return failure();

  // , shape = [ ... ]
  if (parser.parseComma() || parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(shapeOps) ||
      parser.parseRSquare())
    return failure();

  // strides = [ ... ]
  if (parser.parseKeyword("strides") || parser.parseEqual() ||
      parser.parseLSquare() ||
      parser.parseOperandList(strideOps) ||
      parser.parseRSquare())
    return failure();

  // attr-dict
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // : result-type
  if (parser.parseColonType(resultTy))
    return failure();
  result.addTypes(resultTy);

  auto tvTy = llvm::dyn_cast<mlir::pto::TensorViewType>(resultTy);
  if (!tvTy)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected result type pto.tensor_view<...>");

  Type elemTy = tvTy.getElementType();

  Type ptrTy = mlir::pto::PtrType::get(parser.getContext(), elemTy);

  // resolve %ptr
  if (parser.resolveOperand(ptr, ptrTy, result.operands))
    return failure();

  // resolve shape/strides 为 index
  Type indexTy = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(shapeOps, indexTy, result.operands))
    return failure();
  if (parser.resolveOperands(strideOps, indexTy, result.operands))
    return failure();

  auto segAttr = parser.getBuilder().getDenseI32ArrayAttr(
      {1, (int32_t)shapeOps.size(), (int32_t)strideOps.size()});
  result.addAttribute("operandSegmentSizes", segAttr);

  return success();
}

void mlir::pto::MakeTensorViewOp::print(OpAsmPrinter &p) {
  p << " " << getPtr();

  p << ", shape = [";
  p.printOperands(getShape());
  p << "] ";

  p << "strides = [";
  p.printOperands(getStrides());
  p << "]";

  p.printOptionalAttrDict((*this)->getAttrs(),
                        /*elidedAttrs=*/{"operandSegmentSizes"});

  p << " : " << getResult().getType();
}

LogicalResult mlir::pto::MakeTensorViewOp::verify() {
  auto tvTy = dyn_cast<mlir::pto::TensorViewType>(getResult().getType());
  if (!tvTy)
    return emitOpError("result must be pto.tensor_view<...>");

  auto pty = dyn_cast<mlir::pto::PtrType>(getPtr().getType());
  if (!pty)
    return emitOpError("ptr operand must be !pto.ptr<...>");

  if (pty.getElementType() != tvTy.getElementType())
    return emitOpError() << "ptr element type must match tensor_view element type, but got ptr="
                         << pty.getElementType() << " view=" << tvTy.getElementType();

  int64_t rank = tvTy.getRank();

  if ((int64_t)getShape().size() != rank || (int64_t)getStrides().size() != rank)
    return emitOpError() << "shape/strides operand counts must match tensor_view rank="
                         << rank;

  return success();
}


//===----------------------------------------------------------------------===//
// PTODialect
//===----------------------------------------------------------------------===//

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
}


AddressSpaceAttr mlir::pto::getPTOAddressSpaceAttr(Type type) {
  auto memRefType = dyn_cast<BaseMemRefType>(type);
  assert(memRefType && "input type must be a memref type");
  auto scopeAttr = dyn_cast<AddressSpaceAttr>(memRefType.getMemorySpace());
  assert(scopeAttr && "memory scope should be a pto address scope");
  return scopeAttr;
}

//===----------------------------------------------------------------------===//
// PTO Load/Store/Addf (non-DPS polymorphic) verification + inference.
//  - If operands are memref/tensor: verify strictly.
//  - Otherwise (tile_view/tile etc): accept (so old IR can still parse).
//===----------------------------------------------------------------------===//

static LogicalResult verifyMemrefToTensorLoad(Operation *op, Value src, Value res) {
  auto mr = dyn_cast<MemRefType>(src.getType());
  auto rt = dyn_cast<RankedTensorType>(res.getType());
  if (!mr)
    return success(); // non-memref case: don't block old IR
  if (!rt)
    return op->emitOpError("when src is memref, result must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  if (mr.hasStaticShape()) {
    if (!rt.hasStaticShape())
      return op->emitOpError("memref has static shape but result tensor is not static");
    if (mr.getShape() != rt.getShape())
      return op->emitOpError() << "shape mismatch: memref=" << mr << " tensor=" << rt;
  } else {
    // For dynamic memref dims: if tensor dim is static, allow it; if it's dynamic too, also fine.
    // We only reject when a memref static dim conflicts with tensor static dim.
    for (int64_t i = 0; i < mr.getRank(); ++i) {
      int64_t md = mr.getDimSize(i);
      int64_t td = rt.getDimSize(i);
      if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
        return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
    }
  }
  return success();
}

static LogicalResult verifyMemrefTensorStore(Operation *op, Value dst, Value src) {
  auto mr = dyn_cast<MemRefType>(dst.getType());
  if (!mr)
    return success(); // non-memref case: old tile IR allowed
  auto rt = dyn_cast<RankedTensorType>(src.getType());
  if (!rt)
    return op->emitOpError("when dst is memref, src must be ranked tensor");

  if (mr.getElementType() != rt.getElementType())
    return op->emitOpError() << "memref/tensor element type mismatch: memref="
                             << mr.getElementType() << " tensor=" << rt.getElementType();

  if (mr.getRank() != rt.getRank())
    return op->emitOpError() << "rank mismatch: memref rank=" << mr.getRank()
                             << " tensor rank=" << rt.getRank();

  for (int64_t i = 0; i < mr.getRank(); ++i) {
    int64_t md = mr.getDimSize(i);
    int64_t td = rt.getDimSize(i);
    if (md != ShapedType::kDynamic && td != ShapedType::kDynamic && md != td)
      return op->emitOpError() << "dim mismatch at " << i << ": memref=" << md << " tensor=" << td;
  }
  return success();
}

LogicalResult LoadOp::verify() {
  Type srcType = getSrc().getType();
  int64_t rank = getPTOTypeRank(srcType);

  if (rank == -1) {
    return emitOpError("source type ") << srcType << " does not support PTO type";
  }

  return success();
}

LogicalResult AllocTileOp::verify() {
  auto ty = getResult().getType(); // TileBufType

  // op 上有没有传 operands
  bool hasVR = getValidRow() != nullptr;
  bool hasVC = getValidCol() != nullptr;

  // type 上的 validShape
  auto vs = ty.getValidShape();
  if (vs.size() != 2)
    return emitOpError("result tile_buf must have rank-2 validShape");

  bool needVR = (vs[0] == -1);
  bool needVC = (vs[1] == -1);

  // 你要求的：v_row=?, v_col=? 时必须同时给两个
  // （这条规则由下面两句自然实现）
  if (hasVR != needVR)
    return emitOpError() << "valid_row operand "
                         << (needVR ? "is required" : "must be absent")
                         << " because result type v_row is "
                         << (needVR ? "?" : std::to_string(vs[0]));

  if (hasVC != needVC)
    return emitOpError() << "valid_col operand "
                         << (needVC ? "is required" : "must be absent")
                         << " because result type v_col is "
                         << (needVC ? "?" : std::to_string(vs[1]));

  return success();
}

LogicalResult LoadDpsOp::verify() {
  Type srcType = getSrc().getType();
  int64_t rank = getPTOTypeRank(srcType);

  if (rank == -1) {
    return emitOpError("source type ") << srcType << " does not support PTO type";
  }
  
  return success();
}

LogicalResult TLoadOp ::verify() {
  auto srcType = dyn_cast<pto::PartitionTensorViewType>(getSrc().getType());
  if (!srcType)
    return emitOpError("expects src to be !pto.partition_tensor_view, got ")
           << getSrc().getType();

  auto dstType = dyn_cast<pto::TileBufType>(getDst().getType());
  if (!dstType)
    return emitOpError("expects dst to be !pto.tile_buf, got ")
           << getDst().getType();

  // TileBuf must always be 2D for PTO hardware tiles.
  if (dstType.getShape().size() != 2)
    return emitOpError("dst tile_buf rank must be 2, got ")
           << dstType.getShape().size();
  if (dstType.getValidShape().size() != 2)
    return emitOpError("dst tile_buf valid_shape rank must be 2, got ")
           << dstType.getValidShape().size();

  // Only check element counts when both sides are statically known.
  int64_t partElems = srcType.getNumElements();
  int64_t tileValidElems = mlir::ShapedType::kDynamic;
  if (!llvm::is_contained(dstType.getValidShape(),
                          mlir::ShapedType::kDynamic)) {
    tileValidElems = 1;
    for (int64_t dim : dstType.getValidShape())
      tileValidElems *= dim;
  }

  if (partElems != mlir::ShapedType::kDynamic &&
      tileValidElems != mlir::ShapedType::kDynamic &&
      partElems != tileValidElems) {
    return emitOpError("partition element count (")
           << partElems
           << ") must match tile_buf valid element count ("
           << tileValidElems << ")";
  }

  return success();
}

LogicalResult mlir::pto::SetFFTsOp::verify() {
  auto mr = llvm::dyn_cast<mlir::MemRefType>(getFfts().getType());
  if (!mr)
    return emitOpError("expects a memref operand");

  if (!mr.getElementType().isInteger(64) && !mr.getElementType().isInteger(8))
    return emitOpError("expects element type i64 (or i8)");

  return mlir::success();
}
LogicalResult mlir::pto::LoadOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;

  if (operands.size() != 1)
    return failure();

  Type srcTy = operands[0].getType();

  if (auto mr = dyn_cast<MemRefType>(srcTy)) {
    SmallVector<int64_t, 4> shape(mr.getRank(), ShapedType::kDynamic);
    for (int64_t i = 0; i < mr.getRank(); ++i)
      shape[i] = mr.getDimSize(i);
    inferredReturnTypes.push_back(
        RankedTensorType::get(shape, mr.getElementType()));
    return success();
  }

  if (auto tv = dyn_cast<mlir::pto::PartitionTensorViewType>(srcTy)) {
    SmallVector<int64_t, 4> shape;
    shape.append(tv.getShape().begin(), tv.getShape().end());
    inferredReturnTypes.push_back(
        mlir::pto::TileType::get(context, shape, tv.getElementType()));
    return success();
  }

  return failure();
}


LogicalResult StoreOp::verify() {
  if (getOperation()->getNumResults() != 0)
    return emitOpError("expects 0 results");

  if (failed(verifyMemrefTensorStore(getOperation(), getDst(), getSrc())))
    return failure();


  Type dstType = getDst().getType();
  int64_t rank = getPTOTypeRank(dstType);

  if (rank == -1) {
    return emitOpError("destination type ") << dstType << " does not support PTO type";
  }

  return success();
}

LogicalResult StoreDpsOp::verify() {
  Type dstType = getDst().getType();
  int64_t rank = getPTOTypeRank(dstType);

  if (rank == -1) {
    return emitOpError("destination type ") << dstType << " does not support PTO type";
  }

  return success();
}

LogicalResult TStoreOp::verify() {
  Type dstType = getDst().getType();
  int64_t rank = getPTOTypeRank(dstType);

  if (rank == -1) {
    return emitOpError("destination type ") << dstType << " does not support PTO type";
  }

  return success();
}

LogicalResult mlir::pto::TransOp::inferReturnTypes(
    MLIRContext *context,
    std::optional<Location> location,
    ValueRange operands,
    DictionaryAttr attributes,
    OpaqueProperties properties,
    RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.empty())
    return emitOptionalError(location, "missing operand 'src'");
  Type srcTy = operands[0].getType();
  // 1) 非 ShapedType：没法推 shape，就保持类型不变
  auto shaped = dyn_cast<ShapedType>(srcTy);
  if (!shaped) {
    inferredReturnTypes.push_back(srcTy);
    return success();
  }
  // 2) Unranked：只能推 element type + unranked
  if (!shaped.hasRank()) {
    if (isa<TensorType>(srcTy)) {
      inferredReturnTypes.push_back(UnrankedTensorType::get(shaped.getElementType()));
      return success();
    }
    if (isa<MemRefType>(srcTy)) {
      inferredReturnTypes.push_back(UnrankedMemRefType::get(shaped.getElementType(),
                                                          /*memorySpace=*/Attribute()));
      return success();
    }
    inferredReturnTypes.push_back(srcTy);
    return success();
  }
  int64_t rank = shaped.getRank();
  // 取 optional attr: valid_range
  auto vr = attributes.get("valid_range").dyn_cast_or_null<DenseI64ArrayAttr>();
  // 3) 没给 valid_range：按“类型不变”推导（你也可以改成默认 reverse）
  if (!vr) {
    inferredReturnTypes.push_back(srcTy);
    return success();
  }
  // 4) valid_range 作为 perm：校验长度 & 合法性
  ArrayRef<int64_t> perm = vr.asArrayRef();
  if ((int64_t)perm.size() != rank)
    return emitOptionalError(location, "valid_range size must equal src rank");
  SmallVector<bool> seen(rank, false);
  for (int64_t p : perm) {
    if (p < 0 || p >= rank)
      return emitOptionalError(location, "valid_range axis out of range");
    if (seen[p])
      return emitOptionalError(location, "valid_range has duplicate axis");
    seen[p] = true;
  }
  // 5) 推导输出 shape
  SmallVector<int64_t> outShape(rank, ShapedType::kDynamic);
  ArrayRef<int64_t> inShape = shaped.getShape();
  for (int64_t i = 0; i < rank; ++i)
    outShape[i] = inShape[perm[i]];
  // 6) 构造 result type（分别处理 tensor / memref）
  if (auto rt = dyn_cast<RankedTensorType>(srcTy)) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(outShape, rt.getElementType(), rt.getEncoding()));
    return success();
  }
  if (auto mr = dyn_cast<MemRefType>(srcTy)) {
    // 注意：transpose 会改变 layout/strides，这里最保守做法是给 identity layout；
    // 如果你的语义是“只转 logical shape”，那还需要同步推导 affine map/strides。
    inferredReturnTypes.push_back(
        MemRefType::get(outShape, mr.getElementType(),
                        /*layout=*/AffineMapAttr(),
                        mr.getMemorySpace()));
    return success();
  }
  // 其它 shaped 直接回退
  inferredReturnTypes.push_back(srcTy);
  return success();
}

LogicalResult AddFOp::verify() {
  // 获取操作数类型 (取 lhs 即可，因为 AllTypesMatch 保证了它们一致)
  Type operandType = getLhs().getType();
  int64_t rank = getPTOTypeRank(operandType);

  // 检查是否获取到了有效的 Rank
  if (rank == -1) {
    return emitOpError("operand type ") << operandType 
           << " does not support PTO type (unknown rank or unranked type)";
  }

  return success();
}

LogicalResult AddFDpsOp::verify() {
  Type operandType = getLhs().getType();
  int64_t rank = getPTOTypeRank(operandType);

  if (rank == -1) {
    return emitOpError("operand type ") << operandType 
           << " does not support PTO type (unknown rank or unranked type)";
  }

  return success();
}


LogicalResult mlir::pto::TransOp::verify() {
  if (getOperation()->getNumOperands() != 1 || getOperation()->getNumResults() != 1)
    return emitOpError("expects 1 operands and 1 result");

  return success();
}

LogicalResult pto::MScatterOp::verify() {
  auto srcTy = llvm::dyn_cast<MemRefType>(getSrc().getType());
  auto memTy = llvm::dyn_cast<MemRefType>(getMem().getType());
  auto idxTy = llvm::dyn_cast<MemRefType>(getIdx().getType());
  if (!srcTy || !memTy || !idxTy)
    return emitOpError("expects src, mem, idx to be memref types");

  // element type match: store src element into mem element
  if (srcTy.getElementType() != memTy.getElementType())
    return emitOpError("src element type must match mem element type");

  // idx element type: integer or index
  Type idxElem = idxTy.getElementType();
  if (!idxElem.isa<IntegerType>() && !idxElem.isa<IndexType>())
    return emitOpError("idx element type must be integer or index");

  // idx shape matches src shape (static case strict; otherwise require rank equal)
  if (idxTy.getRank() != srcTy.getRank())
    return emitOpError("idx rank must match src rank");
  if (idxTy.hasStaticShape() && srcTy.hasStaticShape() &&
      idxTy.getShape() != srcTy.getShape())
    return emitOpError("idx shape must match src shape");

  return success();
}
LogicalResult pto::AbsOp_DPS::verify() {
  auto srcTy = llvm::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src and dst to be memref types");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("src and dst rank must match");

  Type se = srcTy.getElementType();
  Type de = dstTy.getElementType();
  if (se != de)
    return emitOpError("src and dst element type must match");

  bool ok = false;
  if (auto it = se.dyn_cast<IntegerType>()) {
    ok = (it.getWidth() == 32 || it.getWidth() == 16) && it.isSignless();
  } else if (se.isF32() || se.isF16()) {
    ok = true;
  }
  if (!ok)
    return emitOpError("element type must be i32, i16, f16(half), or f32");

  return success();
}

LogicalResult pto::AddOp_DPS::verify() {
  auto t0 = llvm::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto t1 = llvm::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto td = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!t0 || !t1 || !td)
    return emitOpError("expects src0/src1/dst to be memref types");

  if (t0.getRank() != t1.getRank() || t0.getRank() != td.getRank())
    return emitOpError("src0/src1/dst rank must match");

  Type e0 = t0.getElementType();
  Type e1 = t1.getElementType();
  Type ed = td.getElementType();
  if (e0 != e1 || e0 != ed)
    return emitOpError("element types must match for src0/src1/dst");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      if (!it.isSignless() && !it.isUnsigned()) return false;
      unsigned w = it.getWidth();
      // A2A3: i32/i16 ; A5: i32/u32/i16/u16/i8/u8
      return (w == 32 || w == 16 || w == 8);
    }
    if (t.isF32() || t.isF16()) return true;
    // A5: bfloat16
    if (t.isBF16()) return true;
    return false;
  };

  if (!isOK(e0))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16(half), bf16, f32");

  return success();
}
LogicalResult pto::AddCOp_DPS::verify() {
  auto t0 = llvm::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto t1 = llvm::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto t2 = llvm::dyn_cast<mlir::MemRefType>(getSrc2().getType());
  auto td = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!t0 || !t1 || !t2 || !td)
    return emitOpError("expects src0/src1/src2/dst to be memref types");

  if (t0.getRank() != t1.getRank() || t0.getRank() != t2.getRank() ||
      t0.getRank() != td.getRank())
    return emitOpError("src0/src1/src2/dst rank must match");

  Type e0 = t0.getElementType();
  if (e0 != t1.getElementType() || e0 != t2.getElementType() || e0 != td.getElementType())
    return emitOpError("element types must match for src0/src1/src2/dst");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(e0))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}
LogicalResult pto::AddSOp_DPS::verify() {
  auto ts = llvm::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto td = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!ts || !td)
    return emitOpError("expects src/dst to be memref types");

  if (ts.getRank() != td.getRank())
    return emitOpError("src and dst rank must match");

  Type elem = ts.getElementType();
  if (elem != td.getElementType())
    return emitOpError("src and dst element type must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elem)
    return emitOpError("scalar type must equal memref element type");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(elem))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}
LogicalResult pto::AddSCOp_DPS::verify() {
  auto ts0 = llvm::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto ts1 = llvm::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto td  = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  
  if (!ts0 || !ts1 || !td)
    return emitOpError("expects src0, src1, and dst to be memref types");

  if (ts0.getRank() != td.getRank() || ts1.getRank() != td.getRank())
    return emitOpError("src0, src1, and dst rank must match");

  Type elem = ts0.getElementType();
  if (elem != td.getElementType())
    return emitOpError("src0 and dst element type must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elem)
    return emitOpError("scalar type must equal memref element type");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(elem))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}

LogicalResult pto::AndOp_DPS::verify() {
  auto m0 = dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto m1 = dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto md = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!m0 || !m1 || !md)
    return emitOpError("expects src0/src1/dst to be memref types");

  Type e0 = m0.getElementType();
  Type e1 = m1.getElementType();
  Type ed = md.getElementType();

  if (!e0.isIntOrIndex() || !e1.isIntOrIndex() || !ed.isIntOrIndex())
    return emitOpError("expects integral element types (int/index) for TAND");

  if (e0 != e1 || e0 != ed)
    return emitOpError("src0/src1/dst element types must match");

  return success();
}

LogicalResult pto::AndSOp_DPS::verify() {
  auto ms = dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto md = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!ms || !md)
    return emitOpError("expects src/dst to be memref types");

  Type es = ms.getElementType();
  Type ed = md.getElementType();
  if (!es.isa<IntegerType>() || !ed.isa<IntegerType>())
    return emitOpError("expects integral element types for TANDS");

  if (es != ed)
    return emitOpError("src and dst element types must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != es)
    return emitOpError("scalar type must match tile element type");

  return success();
}
LogicalResult pto::AssignOp_DPS::verify() {
  auto m = dyn_cast<mlir::pto::TileType>(getObj().getType());
  if (!m)
    return emitOpError("expects obj to be a memref");

  Type addrTy = getAddr().getType();
  if (!addrTy.isa<IndexType>() && !addrTy.isa<IntegerType>())
    return emitOpError("expects addr to be index or integer type");

  return success();
}

LogicalResult pto::CIOp_DPS::verify() {
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!dstTy)
    return emitOpError("expects dst to be a memref");

  auto elemTy = dstTy.getElementType().dyn_cast<IntegerType>();
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32)
    return emitOpError("expects dst element type to be i16 or i32");

  auto sTy = getS().getType().dyn_cast<IntegerType>();
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy.getWidth() != bw)
    return emitOpError("expects S type to match dst element bitwidth");

  return success();
}

LogicalResult pto::CmpOp_DPS::verify() {
  auto dstTy  = dyn_cast<mlir::MemRefType>(getDst().getType());
  auto src0Ty = dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = dyn_cast<mlir::MemRefType>(getSrc1().getType());
  if (!dstTy || !src0Ty || !src1Ty)
    return emitOpError("expects dst/src0/src1 to be memref");

  if (src0Ty.getElementType() != src1Ty.getElementType())
    return emitOpError("expects src0/src1 element types to match");
  return success();
}

// ---- TCMPS verify ----
LogicalResult pto::CmpSOp_DPS::verify() {
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  auto srcTy = dyn_cast<mlir::MemRefType>(getSrc().getType());
  if (!dstTy || !srcTy)
    return emitOpError("expects dst/src to be memref");

  Type elemTy = srcTy.getElementType();
  Type scalarTy = getScalar().getType();

  if (scalarTy != elemTy)
    return emitOpError("expects scalar type to match src element type");

  return success();
}
LogicalResult pto::ColExpandOp_DPS::verify() {
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  auto srcTy = dyn_cast<mlir::MemRefType>(getSrc().getType());
  if (!dstTy || !srcTy)
    return emitOpError("expects dst/src to be memref");

  if (dstTy.getElementType() != srcTy.getElementType())
    return emitOpError("expects dst/src element types to match");


  return success();
}
LogicalResult pto::ColMaxOp_DPS::verify() {
  auto srcTy = dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src/dst to be memref");

  // element type must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src/dst element types to match");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects src/dst to have same rank");

  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");
  }

  return success();
}
LogicalResult pto::ColMinOp_DPS::verify() {
  auto srcTy = dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src/dst to be memref");

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src/dst element types to match");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects src/dst to have same rank");

  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");
  }

  return success();
}
LogicalResult pto::ColSumOp_DPS::verify() {
  auto srcTy = dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto tmpTy = dyn_cast<mlir::MemRefType>(getTmp().getType());
  auto dstTy = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !tmpTy || !dstTy)
    return emitOpError("expects src/tmp/dst to be memref");

  if (srcTy.getElementType() != dstTy.getElementType() ||
      srcTy.getElementType() != tmpTy.getElementType())
    return emitOpError("expects src/tmp/dst element types to match");

  if (srcTy.getRank() != tmpTy.getRank())
    return emitOpError("expects src/tmp to have same rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects dst to have same rank as src");

  if (srcTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");

    int64_t tmpC = tmpTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && tmpC != ShapedType::kDynamic && srcC != tmpC)
      return emitOpError("expects src/tmp to have same number of columns (dim1)");
  }

  if (dstTy.getRank() >= 1) {
    int64_t dstR = dstTy.getShape()[0];
    if (dstR != ShapedType::kDynamic && dstR != 1)
      return emitOpError("expects dst dim0 to be 1 (column-reduction result)");
  }

  return success();
}
//===----------------------------------------------------------------------===//
// PTO_TCvtOp_DPS verification
//===----------------------------------------------------------------------===//

llvm::LogicalResult mlir::pto::CvtOp_DPS::verify() {
  auto srcTy = llvm::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy) return emitOpError("expects memref src/dst");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("src/dst rank mismatch");

  return mlir::success();
}

LogicalResult mlir::pto::DivOp_DPS::verify() {
  auto src0Ty = dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError("expects memref operands");

  // 只支持 half/float（按你给的约束）
  auto elem0 = src0Ty.getElementType();
  auto elem1 = src1Ty.getElementType();
  auto elemd = dstTy.getElementType();
  if (elem0 != elem1 || elem0 != elemd)
    return emitOpError("src0/src1/dst element type must match");

  if (!elem0.isF16() && !elem0.isF32())
    return emitOpError("only supports f16/f32 element type");

  if (src0Ty.getRank() != src1Ty.getRank() || src0Ty.getRank() != dstTy.getRank())
    return emitOpError("src0/src1/dst rank mismatch");

  return success();
}
//===----------------------------------------------------------------------===//
// TDivSOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::DivSOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects memref types for src/dst");

  // shape/rank must match (按你们其他 elemwise op 的习惯)
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects same rank for src and dst");
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects same shape for src and dst");

  // element type must match (TDIVS 标注 scalar 是 TileData::DType)
  Type elemTy = srcTy.getElementType();
  if (dstTy.getElementType() != elemTy)
    return emitOpError("expects src/dst element type to match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elemTy)
    return emitOpError("expects scalar type to match memref element type");

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TExpOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ExpOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects memref types for src/dst");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects same rank for src and dst");
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects same shape for src and dst");

  Type srcElem = srcTy.getElementType();
  Type dstElem = dstTy.getElementType();
  if (srcElem != dstElem)
    return emitOpError("expects src/dst element type to match");

  // spec: float or half
  if (!srcElem.isF16() && !srcElem.isF32())
    return emitOpError("expects element type to be f16 or f32");

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TExpandsOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ExpandsOp_DPS::verify() {
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!dstTy)
    return emitOpError("expects memref dst");

  Type dstElem = dstTy.getElementType();
  Type scalarTy = getScalar().getType();

  if (scalarTy != dstElem)
    return emitOpError("expects scalar type == dst element type");

  if (dstElem.isF16() || dstElem.isF32())
    return mlir::success();

  if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
    unsigned w = it.getWidth();
    if (w == 8 || w == 16 || w == 32)
      return mlir::success();
  }

  return emitOpError("unsupported dst element type for texpands (expect f16/f32 or i8/i16/i32/u8/u16/u32)");
}
//===----------------------------------------------------------------------===//
// TExtractOp_DPS verifier
//===----------------------------------------------------------------------===//


mlir::LogicalResult mlir::pto::ExtractOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects memref src and memref dst");

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 memref for src/dst");

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src element type == dst element type");

  if (!getIndexRow().getType().isIndex() || !getIndexCol().getType().isIndex())
    return emitOpError("expects indexRow/indexCol to be index type");

  auto rowCst = getIndexRow().getDefiningOp<mlir::arith::ConstantOp>();
  auto colCst = getIndexCol().getDefiningOp<mlir::arith::ConstantOp>();
  if (rowCst && colCst) {
    auto rowAttr = mlir::dyn_cast<mlir::IntegerAttr>(rowCst.getValue());
    auto colAttr = mlir::dyn_cast<mlir::IntegerAttr>(colCst.getValue());
    if (rowAttr && colAttr) {
      int64_t r0 = rowAttr.getInt();
      int64_t c0 = colAttr.getInt();
      if (r0 < 0 || c0 < 0)
        return emitOpError("indexRow/indexCol must be non-negative");
    }
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TFillPadOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::FillPadOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects memref src and memref dst");

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 memref for src/dst");

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return it.getWidth() / 8;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(t))
      return ft.getWidth() / 8;
    return -1;
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0)
    return emitOpError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return emitOpError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return emitOpError("expects element size to be 1, 2, or 4 bytes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src and dst to have the same static shape for tfillpad");


  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TGatherOp_DPS verifier
//===----------------------------------------------------------------------===//

// TGather: must provide exactly one of {indices operand, maskPattern attr}.
llvm::LogicalResult mlir::pto::GatherOp_DPS::verify() {
  Value indices = getIndices();                 // optional operand (may be null)
  auto maskAttr = getMaskPatternAttr();         // optional attr (may be null)

  const bool hasIdx  = (bool)indices;
  const bool hasMask = (bool)maskAttr;

  if (hasIdx == hasMask) {
    return emitOpError()
        << "expects exactly one of: 'indices' operand OR 'maskPattern' attribute";
  }

  // Basic type sanity: src0/dst element types should match.
  auto src0Ty = llvm::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy  = llvm::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "src0 and dst must have the same element type";

  // If index-based form, indices must be integer element type (at least).
  if (hasIdx) {
    auto idxTy = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
    if (!idxTy)
      return emitOpError() << "indices must be a memref type";
    if (!llvm::isa<IntegerType>(idxTy.getElementType()))
      return emitOpError() << "indices element type must be integer";
  }

  return success();
}
mlir::LogicalResult mlir::pto::GatherbOp_DPS::verify() {
  auto srcTy = getSrc().getType().dyn_cast<mlir::MemRefType>();
  auto offTy = getOffsets().getType().dyn_cast<mlir::MemRefType>();
  auto dstTy = getDst().getType().dyn_cast<mlir::MemRefType>();
  if (!srcTy || !offTy || !dstTy)
    return emitOpError() << "expects memref types for src/offsets/dst";

  // Require equal ranks (tile shapes should match).
  if (srcTy.getRank() != offTy.getRank() || srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src/offsets/dst to have the same rank";

  // Require identical shapes (static when known).
  if (srcTy.getShape() != offTy.getShape() || srcTy.getShape() != dstTy.getShape())
    return emitOpError() << "expects src/offsets/dst to have the same shape";

  // Offsets are interpreted as uint32 byte offsets by implementation.
  if (!offTy.getElementType().isInteger(32) || offTy.getElementType().isUnsignedInteger() == false)
    return emitOpError() << "expects offsets element type to be ui32";

  // Dst/src element size must be 1/2/4 bytes.
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();

  auto elemBits = [](mlir::Type t) -> std::optional<unsigned> {
    if (auto i = t.dyn_cast<mlir::IntegerType>())
      return i.getWidth();
    if (auto f = t.dyn_cast<mlir::FloatType>())
      return f.getWidth();
    return std::nullopt;
  };

  auto sb = elemBits(srcElemTy);
  auto db = elemBits(dstElemTy);
  if (!sb || !db)
    return emitOpError() << "expects src/dst element types to be int/float/half/bf16-like scalars";

  auto bytesOk = [](unsigned bits) { return bits == 8 || bits == 16 || bits == 32; };
  if (!bytesOk(*sb) || !bytesOk(*db))
    return emitOpError() << "expects src/dst element size to be 1/2/4 bytes";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TLOG verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::LogOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  auto elemTy = srcTy.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (dstTy.getElementType() != elemTy)
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memref)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TLRELU verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::LReluOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memref)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMAX verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MaxOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != dstTy.getElementType() ||
      src1Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memref)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s1[i] != d[i])
      return emitOpError() << "expects src1/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMAXS verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MaxSOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/dst to have the same element type";

  if (src0Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/dst to be rank-2 (tile-shaped memref)";

  auto s0 = src0Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMIN verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MinOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memref)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && s1[i] != mlir::ShapedType::kDynamic &&
        s0[i] != s1[i])
      return emitOpError() << "expects src0/src1 shapes to match";
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMINS verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MinsOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memref)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMOV DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MovDpsOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src/dst to have the same rank";

  if (srcTy.getRank() >= 2) {
    auto ss = srcTy.getShape();
    auto ds = dstTy.getShape();

    int64_t sR = ss[srcTy.getRank() - 2];
    int64_t sC = ss[srcTy.getRank() - 1];
    int64_t dR = ds[dstTy.getRank() - 2];
    int64_t dC = ds[dstTy.getRank() - 1];

    if (sR != mlir::ShapedType::kDynamic && dR != mlir::ShapedType::kDynamic && sR != dR)
      return emitOpError() << "expects src/dst rows to match";
    if (sC != mlir::ShapedType::kDynamic && dC != mlir::ShapedType::kDynamic && sC != dC)
      return emitOpError() << "expects src/dst cols to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MovFPOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto fpTy  = mlir::dyn_cast<mlir::MemRefType>(getFp().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !fpTy || !dstTy)
    return emitOpError() << "expects memref types for src/fp/dst";

  // fp is a scaling tile; keep checks minimal but sanity-check it's 64-bit integer when statically known.
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(fpTy.getElementType())) {
    if (it.getWidth() != 64)
      return emitOpError() << "expects fp element type to be i64/ui64";
  }

  // Shape compatibility: require last-2 dims of src/dst to match when both are rank>=2 and statically known.
  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    auto ss = srcTy.getShape();
    auto ds = dstTy.getShape();

    int64_t sR = ss[srcTy.getRank() - 2];
    int64_t sC = ss[srcTy.getRank() - 1];
    int64_t dR = ds[dstTy.getRank() - 2];
    int64_t dC = ds[dstTy.getRank() - 1];

    if (sR != mlir::ShapedType::kDynamic && dR != mlir::ShapedType::kDynamic && sR != dR)
      return emitOpError() << "expects src/dst rows to match";
    if (sC != mlir::ShapedType::kDynamic && dC != mlir::ShapedType::kDynamic && sC != dC)
      return emitOpError() << "expects src/dst cols to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MrgSortOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  // Element type: half/float per ISA (keep strict when statically known).
  if (!srcTy.getElementType().isF16() && !srcTy.getElementType().isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 memrefs";

  // Rows == 1 (list stored in a single row).
  auto ss = srcTy.getShape();
  auto ds = dstTy.getShape();
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
    return emitOpError() << "expects src rows == 1";
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
    return emitOpError() << "expects dst rows == 1";

  // Cols must match when statically known.
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1])
    return emitOpError() << "expects src/dst cols to match";

  // blockLen must be multiple of 64 (single-list variant constraint).
  int64_t bl = 0;
  if (auto a = getBlockLenAttr())
    bl = a.getInt();
  if (bl <= 0)
    return emitOpError() << "expects blockLen > 0";
  if ((bl % 64) != 0)
    return emitOpError() << "expects blockLen to be a multiple of 64";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MulOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::MulsOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  // scalar is F32Attr by TD; nothing to verify beyond presence.

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::NegOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::NotOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  auto elemTy = srcTy.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise NOT";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::OrOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  auto elemTy = src0Ty.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise OR";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && s1[i] != mlir::ShapedType::kDynamic && s0[i] != s1[i])
      return emitOpError() << "expects src0 shape to match src1 shape";
    if (s0[i] != mlir::ShapedType::kDynamic && d[i]  != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::OrsOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src0/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/dst to have the same element type";

  auto elemTy = srcTy.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise OR";

  auto scalarTy = getScalar().getType();
  auto elemITy = mlir::dyn_cast<mlir::IntegerType>(elemTy);
  auto scalarITy = mlir::dyn_cast<mlir::IntegerType>(scalarTy);
  if (!scalarITy)
    return emitOpError() << "expects integer type for scalar";

  if (elemITy.getWidth() != scalarITy.getWidth())
    return emitOpError() << "expects scalar integer width to match element integer width";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::PartAddOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::PartMaxOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::PartMinOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::PreluOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RecipOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ReluOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RemOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped memrefs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RemSOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped memrefs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  // Scalar must be float-like; element type can be int/float (runtime behavior target-defined).
  if (!mlir::isa<mlir::FloatType>(getScalar().getType()))
    return emitOpError() << "expects scalar to be a float type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRESHAPE DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ReshapeOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowExpandOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same element type";
	
  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDDIV DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowExpandDivOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDMUL DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowExpandMulOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDSUB DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowExpandSubOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowMaxOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/tmp/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if ( elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowMinOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto tmpTy = mlir::dyn_cast<mlir::MemRefType>(getTmp().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !tmpTy || !dstTy)
    return emitOpError() << "expects memref types for src/tmp/dst";

  if (srcTy.getRank() != 2 || tmpTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != tmpTy.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RowSumOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/tmp/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if ( elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::RsqrtOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ScatterOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto idxTy = mlir::dyn_cast<mlir::MemRefType>(getIndexes().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());
  if (!srcTy || !idxTy || !dstTy)
    return emitOpError() << "expects memref types for src/indexes/dst";

  if (srcTy.getRank() != 2 || idxTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src/indexes/dst";

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();
  auto idxElem = idxTy.getElementType();

  if (srcElem != dstElem)
    return emitOpError() << "expects src/dst to have the same element type";

  auto isAllowedDataElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  auto isAllowedIndexElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedDataElem(srcElem))
    return emitOpError() << "expects src/dst element type to be i8/i16/i32/f16/bf16/f32";

  if (!isAllowedIndexElem(idxElem))
    return emitOpError() << "expects indexes element type to be i16 or i32";

  // Enforce size relationship between data dtype and index dtype.
  auto bwData = srcElem.getIntOrFloatBitWidth();
  auto bwIdx  = idxElem.getIntOrFloatBitWidth();
  if (bwData != 8 && bwData != 16 && bwData != 32)
    return emitOpError() << "unexpected src/dst element bitwidth " << bwData;

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;

  // 4B data -> 4B idx, 2B data -> 2B idx, 1B data -> 2B idx
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes)
    return emitOpError() << "expects indexes element size " << expectedIdxBytes
                         << " bytes for data element size " << dataBytes << " bytes";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SelOp_DPS::verify() {
  auto maskTy = mlir::dyn_cast<mlir::MemRefType>(getMask().getType());
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!maskTy || !src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for mask, src0, src1, dst";

  if (maskTy.getRank() != 2 || src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for mask, src0, src1, dst";

  auto maskElem = maskTy.getElementType();
  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32/f16/bf16/f32";

  if (maskElem.isInteger(8) == false)
    return emitOpError() << "expects mask element type to be i8";

  // Validate tile layouts
  if (src0Ty.getShape()[1] != dstTy.getShape()[1])
    return emitOpError() << "expects src0 and dst cols to match";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSELS DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SelsOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32/f16/bf16/f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ShlOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::ShrOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PTO.cpp - Fixes for deprecated and erroneous Type methods
//===----------------------------------------------------------------------===//

mlir::LogicalResult Sort32Op_DPS::verify() {
  // Correcting dyn_cast usage for mlir::MemRefType
  auto srcTy = getSrc().getType();
  if (!mlir::isa<mlir::MemRefType>(srcTy)) {
    return emitOpError() << "Expected mlir::MemRefType for src.";
  }

  auto dstTy = getDst().getType();
  if (!mlir::isa<mlir::MemRefType>(dstTy)) {
    return emitOpError() << "Expected mlir::MemRefType for dst.";
  }

  return success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SqrtOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src and dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 memrefs for src and dst";

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();

  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";

  if (!(srcElem.isa<mlir::FloatType>() || srcElem.isa<mlir::Float16Type>()))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSTORE DPS/memref implementation)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSTORE_FP DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::StoreFPOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto fpTy = mlir::dyn_cast<mlir::MemRefType>(getFp().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!srcTy || !fpTy || !dstTy)
    return emitOpError() << "expects memref types for src, fp, and dst";

  if (srcTy.getRank() != dstTy.getRank() || srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same rank and element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUB DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SubOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects all tensors to have the same rank";

  if (src0Ty.getElementType() != src1Ty.getElementType() || src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBC DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SubCOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto src2Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc2().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !src2Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, src2, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank() || src2Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects all tensors to have the same rank";

  if (src0Ty.getElementType() != src1Ty.getElementType() || src0Ty.getElementType() != src2Ty.getElementType() || src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, src2, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBS DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SubSOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!srcTy || !dstTy)
    return emitOpError() << "expects memref types for src and dst";

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src and dst to have the same rank";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBSC DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SubSCOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";

  if (src0Ty.getElementType() != dstTy.getElementType() || src1Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PTO.cpp  (add TXOR DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::XOROp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects memref types for src0, src1, and dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TXORS DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::XORSOp_DPS::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::MemRefType>(getSrc0().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!src0Ty || !dstTy)
    return emitOpError() << "expects memref types for src0 and dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0 and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSYNC DPS/memref implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::SYNCOp_DPS::verify() {
  auto eventsTy = mlir::dyn_cast<mlir::MemRefType>(getEvents().getType());
  auto dstTy = mlir::dyn_cast<mlir::MemRefType>(getDst().getType());

  if (!eventsTy || !dstTy)
    return emitOpError() << "expects memref types for events and dst";

  if (eventsTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects events and dst to have the same element type";

  return mlir::success();
}

mlir::LogicalResult mlir::pto::PrintOp_DPS::verify() {
  auto srcTy = mlir::dyn_cast<mlir::MemRefType>(getSrc().getType());

  if (!srcTy)
    return emitOpError() << "expects memref types for src";
    return mlir::success();
  }
  

LogicalResult pto::TAbsOp::verify() {
  auto srcTy = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src and dst to be tilebuf types");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("src and dst rank must match");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("src and dst shape must match");

  Type se = srcTy.getElementType();
  Type de = dstTy.getElementType();
  if (se != de)
    return emitOpError("src and dst element type must match");

  bool ok = false;
  if (auto it = se.dyn_cast<IntegerType>()) {
    ok = (it.getWidth() == 32 || it.getWidth() == 16) && it.isSignless();
  } else if (se.isF32() || se.isF16()) {
    ok = true;
  }
  if (!ok)
    return emitOpError("element type must be i32, i16, f16(half), or f32");

  return success();
}
// PTO.cpp

static bool isPTOShapedLike(Type ty) {
  return ty.isa<MemRefType, RankedTensorType,
                pto::TileBufType, pto::PartitionTensorViewType>();
}

static Type getElemTy(Type ty) {
  if (auto mr = ty.dyn_cast<MemRefType>()) return mr.getElementType();
  if (auto tt = ty.dyn_cast<RankedTensorType>()) return tt.getElementType();
  if (auto tb = ty.dyn_cast<pto::TileBufType>()) return tb.getElementType();
  if (auto tv = ty.dyn_cast<pto::PartitionTensorViewType>()) return tv.getElementType();
  return Type();
}

static SmallVector<int64_t, 4> getShapeVec(Type ty) {
  SmallVector<int64_t, 4> s;
  if (auto mr = ty.dyn_cast<MemRefType>())
    return SmallVector<int64_t,4>(mr.getShape().begin(), mr.getShape().end());
  if (auto tt = ty.dyn_cast<RankedTensorType>())
    return SmallVector<int64_t,4>(tt.getShape().begin(), tt.getShape().end());
  if (auto tb = ty.dyn_cast<pto::TileBufType>())
    return SmallVector<int64_t,4>(tb.getShape().begin(), tb.getShape().end());
  if (auto tv = ty.dyn_cast<pto::PartitionTensorViewType>())
    return SmallVector<int64_t,4>(tv.getShape().begin(), tv.getShape().end());
  return {};
}

LogicalResult pto::TAddOp::verify() {
  Type t0 = getSrc0().getType();
  Type t1 = getSrc1().getType();
  Type td = getDst().getType();

  // 1) 允许 memref/tensor/tile_buf/tile_view
  if (!isPTOShapedLike(t0) || !isPTOShapedLike(t1) || !isPTOShapedLike(td))
    return emitOpError("expects src0/src1/dst to be memref/tensor/tile_buf/tile_view types");

  // 2) element type 一致
  Type e0 = getElemTy(t0), e1 = getElemTy(t1), ed = getElemTy(td);
  if (!e0 || !e1 || !ed)
    return emitOpError("failed to get element type for operands");
  if (e0 != e1 || e0 != ed)
    return emitOpError("expects src0/src1/dst to have the same element type, but got ")
           << e0 << ", " << e1 << ", " << ed;

  // 3) shape 一致（如果你希望 tile_view/tile_buf 也严格一致）
  auto s0 = getShapeVec(t0);
  auto s1 = getShapeVec(t1);
  auto sd = getShapeVec(td);
  if (s0 != s1 || s0 != sd)
    return emitOpError("expects src0/src1/dst to have the same shape");

  return success();
}

LogicalResult pto::TAddCOp::verify() {
  auto t0 = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto t1 = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto t2 = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc2().getType());
  auto td = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!t0 || !t1 || !t2 || !td)
    return emitOpError("expects src0/src1/src2/dst to be tilebuf types");

  if (t0.getRank() != t1.getRank() || t0.getRank() != t2.getRank() ||
      t0.getRank() != td.getRank())
    return emitOpError("src0/src1/src2/dst rank must match");

  Type e0 = t0.getElementType();
  if (e0 != t1.getElementType() || e0 != t2.getElementType() || e0 != td.getElementType())
    return emitOpError("element types must match for src0/src1/src2/dst");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(e0))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}
LogicalResult pto::TAddSOp::verify() {
  auto ts = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto td = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!ts || !td)
    return emitOpError("expects src/dst to be tilebuf types");

  if (ts.getRank() != td.getRank())
    return emitOpError("src and dst rank must match");

  Type elem = ts.getElementType();
  if (elem != td.getElementType())
    return emitOpError("src and dst element type must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elem)
    return emitOpError("scalar type must equal memref element type");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(elem))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}
LogicalResult pto::TAddSCOp::verify() {
  auto ts0 = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto ts1 = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto td  = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  
  if (!ts0 || !ts1 || !td)
    return emitOpError("expects src0, src1, and dst to be tilebuf types");

  if (ts0.getRank() != td.getRank() || ts1.getRank() != td.getRank())
    return emitOpError("src0, src1, and dst rank must match");

  Type elem = ts0.getElementType();
  if (elem != td.getElementType())
    return emitOpError("src0 and dst element type must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elem)
    return emitOpError("scalar type must equal memref element type");

  auto isOK = [&](Type t) -> bool {
    if (auto it = t.dyn_cast<IntegerType>()) {
      unsigned w = it.getWidth();
      return (w == 32 || w == 16 || w == 8);
    }
    return t.isF32() || t.isF16() || t.isBF16();
  };

  if (!isOK(elem))
    return emitOpError("element type must be one of: i32/u32, i16/u16, i8/u8, f16, bf16, f32");

  return success();
}

LogicalResult pto::TAndOp::verify() {
  auto m0 = dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto m1 = dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto md = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!m0 || !m1 || !md)
    return emitOpError("expects src0/src1/dst to be tilebuf types");

  Type e0 = m0.getElementType();
  Type e1 = m1.getElementType();
  Type ed = md.getElementType();

  if (!e0.isIntOrIndex() || !e1.isIntOrIndex() || !ed.isIntOrIndex())
    return emitOpError("expects integral element types (int/index) for TAND");

  if (e0 != e1 || e0 != ed)
    return emitOpError("src0/src1/dst element types must match");

  return success();
}

LogicalResult pto::TAndSOp::verify() {
  auto ms = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto md = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!ms || !md)
    return emitOpError("expects src/dst to be tilebuf types");

  Type es = ms.getElementType();
  Type ed = md.getElementType();
  if (!es.isa<IntegerType>() || !ed.isa<IntegerType>())
    return emitOpError("expects integral element types for TANDS");

  if (es != ed)
    return emitOpError("src and dst element types must match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != es)
    return emitOpError("scalar type must match tile element type");

  return success();
}
LogicalResult pto::TAssignOp::verify() {
  auto m = dyn_cast<mlir::pto::TileBufType>(getObj().getType());
  if (!m)
    return emitOpError("expects obj to be a tilebuf");

  Type addrTy = getAddr().getType();
  if (!addrTy.isa<IndexType>() && !addrTy.isa<IntegerType>())
    return emitOpError("expects addr to be index or integer type");

  return success();
}
LogicalResult pto::TCIOp::verify() {
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!dstTy)
    return emitOpError("expects dst to be a tilebuf");

  auto elemTy = dstTy.getElementType().dyn_cast<IntegerType>();
  if (!elemTy)
    return emitOpError("expects dst element type to be integer");

  unsigned bw = elemTy.getWidth();
  if (bw != 16 && bw != 32)
    return emitOpError("expects dst element type to be i16 or i32");

  auto sTy = getS().getType().dyn_cast<IntegerType>();
  if (!sTy)
    return emitOpError("expects S to be integer");

  if (sTy.getWidth() != bw)
    return emitOpError("expects S type to match dst element bitwidth");

  return success();
}
LogicalResult pto::TCmpOp::verify() {
  auto dstTy  = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  auto src0Ty = dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  if (!dstTy || !src0Ty || !src1Ty)
    return emitOpError("expects dst/src0/src1 to be tilebuf");

  if (src0Ty.getElementType() != src1Ty.getElementType())
    return emitOpError("expects src0/src1 element types to match");
  return success();
}

// ---- TCMPS verify ----
LogicalResult pto::TCmpSOp::verify() {
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  auto srcTy = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  if (!dstTy || !srcTy)
    return emitOpError("expects dst/src to be tilebuf");

  Type elemTy = srcTy.getElementType();
  Type scalarTy = getScalar().getType();

  if (scalarTy != elemTy)
    return emitOpError("expects scalar type to match src element type");

  return success();
}
LogicalResult pto::TColExpandOp::verify() {
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  auto srcTy = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  if (!dstTy || !srcTy)
    return emitOpError("expects dst/src to be tilebuf");

  if (dstTy.getElementType() != srcTy.getElementType())
    return emitOpError("expects dst/src element types to match");


  return success();
}
LogicalResult pto::TColMaxOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src/dst to be tilebuf");

  // element type must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src/dst element types to match");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects src/dst to have same rank");

  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");
  }

  return success();
}
LogicalResult pto::TColMinOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects src/dst to be tilebuf");

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src/dst element types to match");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects src/dst to have same rank");

  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");
  }

  return success();
}
LogicalResult pto::TColSumOp::verify() {
  auto srcTy = dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto tmpTy = dyn_cast<mlir::pto::TileBufType>(getTmp().getType());
  auto dstTy = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !tmpTy || !dstTy)
    return emitOpError("expects src/tmp/dst to be tilebuf");

  if (srcTy.getElementType() != dstTy.getElementType() ||
      srcTy.getElementType() != tmpTy.getElementType())
    return emitOpError("expects src/tmp/dst element types to match");

  if (srcTy.getRank() != tmpTy.getRank())
    return emitOpError("expects src/tmp to have same rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects dst to have same rank as src");

  if (srcTy.getRank() >= 2) {
    int64_t srcC = srcTy.getShape()[1];
    int64_t dstC = dstTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && dstC != ShapedType::kDynamic && srcC != dstC)
      return emitOpError("expects src/dst to have same number of columns (dim1)");

    int64_t tmpC = tmpTy.getShape()[1];
    if (srcC != ShapedType::kDynamic && tmpC != ShapedType::kDynamic && srcC != tmpC)
      return emitOpError("expects src/tmp to have same number of columns (dim1)");
  }

  if (dstTy.getRank() >= 1) {
    int64_t dstR = dstTy.getShape()[0];
    if (dstR != ShapedType::kDynamic && dstR != 1)
      return emitOpError("expects dst dim0 to be 1 (column-reduction result)");
  }

  return success();
}
//===----------------------------------------------------------------------===//
// PTO_TCvtOp_DPS verification
//===----------------------------------------------------------------------===//

llvm::LogicalResult mlir::pto::TCvtOp::verify() {
  auto srcTy = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy) return emitOpError("expects tilebuf src/dst");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("src/dst rank mismatch");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("src/dst shape mismatch");

  return mlir::success();
}

LogicalResult mlir::pto::TDivOp::verify() {
  auto src0Ty = dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError("expects tilebuf operands");

  // 只支持 half/float（按你给的约束）
  auto elem0 = src0Ty.getElementType();
  auto elem1 = src1Ty.getElementType();
  auto elemd = dstTy.getElementType();
  if (elem0 != elem1 || elem0 != elemd)
    return emitOpError("src0/src1/dst element type must match");

  if (!elem0.isF16() && !elem0.isF32())
    return emitOpError("only supports f16/f32 element type");

  if (src0Ty.getRank() != src1Ty.getRank() || src0Ty.getRank() != dstTy.getRank())
    return emitOpError("src0/src1/dst rank mismatch");

  if (src0Ty.getShape() != src1Ty.getShape())
    return emitOpError("src0/src1 shape mismatch");

  if (src0Ty.getShape() != dstTy.getShape())
    return emitOpError("src0/dst shape mismatch");

  return success();
}
//===----------------------------------------------------------------------===//
// TDivSOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TDivSOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tilebuf types for src/dst");

  // shape/rank must match (按你们其他 elemwise op 的习惯)
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects same rank for src and dst");
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects same shape for src and dst");

  // element type must match (TDIVS 标注 scalar 是 TileData::DType)
  Type elemTy = srcTy.getElementType();
  if (dstTy.getElementType() != elemTy)
    return emitOpError("expects src/dst element type to match");

  Type scalarTy = getScalar().getType();
  if (scalarTy != elemTy)
    return emitOpError("expects scalar type to match tilebuf element type");

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TExpOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TExpOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tilebuf types for src/dst");

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("expects same rank for src and dst");
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects same shape for src and dst");

  Type srcElem = srcTy.getElementType();
  Type dstElem = dstTy.getElementType();
  if (srcElem != dstElem)
    return emitOpError("expects src/dst element type to match");

  // spec: float or half
  if (!srcElem.isF16() && !srcElem.isF32())
    return emitOpError("expects element type to be f16 or f32");

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TExpandsOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TExpandsOp::verify() {
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!dstTy)
    return emitOpError("expects tilebuf dst");

  Type dstElem = dstTy.getElementType();
  Type scalarTy = getScalar().getType();

  if (scalarTy != dstElem)
    return emitOpError("expects scalar type == dst element type");

  if (dstElem.isF16() || dstElem.isF32())
    return mlir::success();

  if (auto it = mlir::dyn_cast<mlir::IntegerType>(dstElem)) {
    unsigned w = it.getWidth();
    if (w == 8 || w == 16 || w == 32)
      return mlir::success();
  }

  return emitOpError("unsupported dst element type for texpands (expect f16/f32 or i8/i16/i32/u8/u16/u32)");
}
//===----------------------------------------------------------------------===//
// TExtractOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TExtractOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tilebuf src and tilebuf dst");

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 tilebuf for src/dst");

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("expects src element type == dst element type");

  if (!getIndexRow().getType().isIndex() || !getIndexCol().getType().isIndex())
    return emitOpError("expects indexRow/indexCol to be index type");

  auto rowCst = getIndexRow().getDefiningOp<mlir::arith::ConstantOp>();
  auto colCst = getIndexCol().getDefiningOp<mlir::arith::ConstantOp>();
  if (rowCst && colCst) {
    auto rowAttr = mlir::dyn_cast<mlir::IntegerAttr>(rowCst.getValue());
    auto colAttr = mlir::dyn_cast<mlir::IntegerAttr>(colCst.getValue());
    if (rowAttr && colAttr) {
	  int64_t r0 = rowAttr.getInt();
	  int64_t c0 = colAttr.getInt();
	  if (r0 < 0 || c0 < 0)
	    return emitOpError("indexRow/indexCol must be non-negative");
    }
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TFillPadOp_DPS verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects tilebuf src and tilebuf dst");

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError("expects rank-2 tilebuf for src/dst");

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();

  auto getElemBytes = [](mlir::Type t) -> int64_t {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t))
      return it.getWidth() / 8;
    if (auto ft = mlir::dyn_cast<mlir::FloatType>(t))
      return ft.getWidth() / 8;
    return -1;
  };

  int64_t srcB = getElemBytes(srcElem);
  int64_t dstB = getElemBytes(dstElem);
  if (srcB < 0 || dstB < 0)
    return emitOpError("unsupported element type (expects int/float element types)");
  if (srcB != dstB)
    return emitOpError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == 2 || srcB == 4))
    return emitOpError("expects element size to be 1, 2, or 4 bytes");

  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("expects src and dst to have the same static shape for tfillpad");
  

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TGatherOp_DPS verifier
//===----------------------------------------------------------------------===//

// TGather: must provide exactly one of {indices operand, maskPattern attr}.
llvm::LogicalResult mlir::pto::TGatherOp::verify() {
  Value indices = getIndices();                 // optional operand (may be null)
  auto maskAttr = getMaskPatternAttr();         // optional attr (may be null)

  const bool hasIdx  = (bool)indices;
  const bool hasMask = (bool)maskAttr;

  if (hasIdx == hasMask) {
    return emitOpError()
        << "expects exactly one of: 'indices' operand OR 'maskPattern' attribute";
  }

  // Basic type sanity: src0/dst element types should match.
  auto src0Ty = llvm::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy  = llvm::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "src0 and dst must have the same element type";

  // If index-based form, indices must be integer element type (at least).
  if (hasIdx) {
    auto idxTy = llvm::dyn_cast<mlir::pto::TileBufType>(indices.getType());
    if (!idxTy)
      return emitOpError() << "indices must be a tilebuf type";
    if (!llvm::isa<IntegerType>(idxTy.getElementType()))
      return emitOpError() << "indices element type must be integer";
  }

  return success();
}
mlir::LogicalResult mlir::pto::TGatherbOp::verify() {
  auto srcTy = getSrc().getType().dyn_cast<mlir::pto::TileBufType>();
  auto offTy = getOffsets().getType().dyn_cast<mlir::pto::TileBufType>();
  auto dstTy = getDst().getType().dyn_cast<mlir::pto::TileBufType>();
  if (!srcTy || !offTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/offsets/dst";

  // Require equal ranks (tile shapes should match).
  if (srcTy.getRank() != offTy.getRank() || srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src/offsets/dst to have the same rank";

  // Require identical shapes (static when known).
  if (srcTy.getShape() != offTy.getShape() || srcTy.getShape() != dstTy.getShape())
    return emitOpError() << "expects src/offsets/dst to have the same shape";

  // Offsets are interpreted as uint32 byte offsets by implementation.
  if (!offTy.getElementType().isInteger(32) || offTy.getElementType().isUnsignedInteger() == false)
    return emitOpError() << "expects offsets element type to be ui32";

  // Dst/src element size must be 1/2/4 bytes.
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();

  auto elemBits = [](mlir::Type t) -> std::optional<unsigned> {
    if (auto i = t.dyn_cast<mlir::IntegerType>())
      return i.getWidth();
    if (auto f = t.dyn_cast<mlir::FloatType>())
      return f.getWidth();
    return std::nullopt;
  };

  auto sb = elemBits(srcElemTy);
  auto db = elemBits(dstElemTy);
  if (!sb || !db)
    return emitOpError() << "expects src/dst element types to be int/float/half/bf16-like scalars";

  auto bytesOk = [](unsigned bits) { return bits == 8 || bits == 16 || bits == 32; };
  if (!bytesOk(*sb) || !bytesOk(*db))
    return emitOpError() << "expects src/dst element size to be 1/2/4 bytes";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TLOG verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TLogOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  auto elemTy = srcTy.getElementType();
  if (!(elemTy.isF16() || elemTy.isF32()))
    return emitOpError() << "expects element type to be f16 or f32";
  if (dstTy.getElementType() != elemTy)
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebuf)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TLRELU verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TLReluOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebuf)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMAX verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMaxOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != dstTy.getElementType() ||
      src1Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebuf)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s1[i] != d[i])
      return emitOpError() << "expects src1/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMAXS verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMaxSOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/dst to have the same element type";

  if (src0Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/dst to be rank-2 (tile-shaped tilebuf)";

  auto s0 = src0Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMIN verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMinOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebuf)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && s1[i] != mlir::ShapedType::kDynamic &&
        s0[i] != s1[i])
      return emitOpError() << "expects src0/src1 shapes to match";
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s0[i] != d[i])
      return emitOpError() << "expects src0/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// TMINS verifier (PTO.cpp)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMinsOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebuf)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic &&
        s[i] != d[i])
      return emitOpError() << "expects src/dst shapes to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMOV DPS/tilebuf op)
//===----------------------------------------------------------------------===//
mlir::LogicalResult mlir::pto::TMovOp::verify() {
  // [Fix 1] 将 TileType 修改为 TileBufType
  // 您的 ODS 定义限制了参数必须是 TileBufType，所以这里应该 Cast 到 TileBufType
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  // [Fix 2] 修改报错信息，使其准确反映预期
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tile_buf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  // TileBufType 的 getRank() 和 getShape() 方法是在 TypeDef 的 extraClassDeclaration 中定义的
  // 确保您的 C++ 实现能访问到它们
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src/dst to have the same rank";

  if (srcTy.getRank() >= 2) {
    auto ss = srcTy.getShape();
    auto ds = dstTy.getShape();

    // 注意：确保 rank >= 2 再访问 rank-2，否则数组越界
    int64_t sR = ss[srcTy.getRank() - 2];
    int64_t sC = ss[srcTy.getRank() - 1];
    int64_t dR = ds[dstTy.getRank() - 2];
    int64_t dC = ds[dstTy.getRank() - 1];

    if (sR != mlir::ShapedType::kDynamic && dR != mlir::ShapedType::kDynamic && sR != dR)
      return emitOpError() << "expects src/dst rows to match";
    if (sC != mlir::ShapedType::kDynamic && dC != mlir::ShapedType::kDynamic && sC != dC)
      return emitOpError() << "expects src/dst cols to match";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMOV_FP DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMovFPOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto fpTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getFp().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !fpTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/fp/dst";

  // fp is a scaling tile; keep checks minimal but sanity-check it's 64-bit integer when statically known.
  if (auto it = mlir::dyn_cast<mlir::IntegerType>(fpTy.getElementType())) {
    if (it.getWidth() != 64)
      return emitOpError() << "expects fp element type to be i64/ui64";
  }

  // Shape compatibility: require last-2 dims of src/dst to match when both are rank>=2 and statically known.
  if (srcTy.getRank() >= 2 && dstTy.getRank() >= 2) {
    auto ss = srcTy.getShape();
    auto ds = dstTy.getShape();

    int64_t sR = ss[srcTy.getRank() - 2];
    int64_t sC = ss[srcTy.getRank() - 1];
    int64_t dR = ds[dstTy.getRank() - 2];
    int64_t dC = ds[dstTy.getRank() - 1];

    if (sR != mlir::ShapedType::kDynamic && dR != mlir::ShapedType::kDynamic && sR != dR)
      return emitOpError() << "expects src/dst rows to match";
    if (sC != mlir::ShapedType::kDynamic && dC != mlir::ShapedType::kDynamic && sC != dC)
      return emitOpError() << "expects src/dst cols to match";
  }

  return mlir::success();
}
// 辅助函数：获取 Rank，支持 ShapedType 和 PTO TileTypes
static int64_t getRankHelper(Type t) {
  if (auto s = dyn_cast<ShapedType>(t)) return s.getRank();
  if (auto tile = dyn_cast<pto::TileBufType>(t)) return tile.getRank();
  if (auto view = dyn_cast<pto::PartitionTensorViewType>(t)) return view.getRank();
  return -1;
}

static LogicalResult verifyMatmulLike(Operation *op, Type aTy, Type bTy, Type dstTy, bool checkRank = true) {
  // 1. 检查类型 (ShapedType 或 Tile 类型)
  bool aValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(aTy);
  bool bValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(bTy);
  bool dValid = isa<ShapedType, pto::TileBufType, pto::PartitionTensorViewType>(dstTy);

  if (!aValid || !bValid || !dValid)
    return op->emitOpError("expects inputs/outputs to be shaped types or PTO tile types");

  if (checkRank) {
    int64_t aRank = getRankHelper(aTy);
    int64_t bRank = getRankHelper(bTy);
    int64_t dRank = getRankHelper(dstTy);

    // 检查 Rank 一致性
    if (aRank != -1 && dRank != -1 && aRank != dRank)
      return op->emitOpError("expects a and dst to have the same rank");
    if (bRank != -1 && dRank != -1 && bRank != dRank)
      return op->emitOpError("expects b and dst to have the same rank");
  }

  return success();
}
// ---- MGatherDpsOp ----
LogicalResult MGatherDpsOp::verify() {  
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  int64_t dstrank = getPTOTypeRank(getDst().getType());

  if (memrank == -1 || idxrank == -1 || dstrank == -1) {
    return emitOpError("mem, idx and dst does not support PTO type");
  }

  return success();
}
// ---- MScatterDpsOp ----
LogicalResult MScatterDpsOp::verify() {
  int64_t srcrank = getPTOTypeRank(getSrc().getType());
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  
  if (memrank == -1 || idxrank == -1 || srcrank == -1) {
    return emitOpError("src, idx, mem does not support PTO type");
  }

  return success();
}
// ---- GetValDpsOp ----
LogicalResult GetValDpsOp::verify() {
  auto memTy = dyn_cast<MemRefType>(getSrc().getType());
  if (!memTy)
    return emitOpError("expects src to be a tilebuf type");

  // Optional: if dst is a scalar type, require it matches tilebuf element type.
  Type elemTy = memTy.getElementType();
  if (getDst().getType() != elemTy)
    return emitOpError("expects dst type to match src element type");

  return success();
}
// ---- SetValDpsOp ----
LogicalResult SetValDpsOp::verify() {
  auto memTy = dyn_cast<MemRefType>(getDst().getType());
  if (!memTy)
    return emitOpError("expects dst to be a tilebuf type");

  // Optional but useful: val type must match element type.
  Type elemTy = memTy.getElementType();
  if (getVal().getType() != elemTy)
    return emitOpError("expects val type to match dst element type");

  return success();
}
// ---- DPS ----
LogicalResult MatmulBiasDpsOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
LogicalResult MatmulMxDpsOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
LogicalResult MatmulMxAccDpsOp::verify() {
  // treat c_in as shaped too, but only check a/b/dst ranks here
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
LogicalResult MatmulMxBiasDpsOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}

// ---- TOp ----
//LogicalResult TMatmulBiasOp::verify() {
  //return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType(), getValidDims());
//}
LogicalResult TMatmulMxOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
LogicalResult TMatmulMxAccOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
LogicalResult TMatmulMxBiasOp::verify() {
  return verifyMatmulLike(*this, getA().getType(), getB().getType(), getDst().getType());
}
// ---- TSetValOp ----
LogicalResult TSetValOp::verify() {
  // dst can be tile/tensor/tilebuf (PTODpsType). Keep checks minimal.
  if (auto shaped = dyn_cast<ShapedType>(getDst().getType())) {
    if (shaped.getElementType() != getVal().getType())
      return emitOpError("expects val type to match dst element type");
  }
  return success();
}
// ---- TGetValOp ----
LogicalResult TGetValOp::verify() {
  if (auto shaped = dyn_cast<ShapedType>(getSrc().getType())) {
    if (shaped.getElementType() != getDst().getType())
      return emitOpError("expects dst type to match src element type");
  }
  return success();
}

// ---- TMScatterOp ----
LogicalResult TMScatterOp::verify() {
  int64_t srcrank = getPTOTypeRank(getSrc().getType());
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  
  if (memrank == -1 || idxrank == -1 || srcrank == -1) {
    return emitOpError("src, idx, mem does not support PTO type");
  }
  return success();
}

// ---- TMGatherOp ----
LogicalResult TMGatherOp::verify() {
  int64_t memrank = getPTOTypeRank(getMem().getType());
  int64_t idxrank = getPTOTypeRank(getIdx().getType());
  int64_t dstrank = getPTOTypeRank(getDst().getType());

  if (memrank == -1 || idxrank == -1 || memrank == -1) {
    return emitOpError("mem, idx and dst does not support PTO type");
  }

  return success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMRGSORT DPS/tilebuf op)
//===----------------------------------------------------------------------===//
mlir::LogicalResult mlir::pto::TMrgSortOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  // Element type: half/float per ISA (keep strict when statically known).
  if (!srcTy.getElementType().isF16() && !srcTy.getElementType().isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 tilebufs";

  // Rows == 1 (list stored in a single row).
  auto ss = srcTy.getShape();
  auto ds = dstTy.getShape();
  if (ss[0] != mlir::ShapedType::kDynamic && ss[0] != 1)
    return emitOpError() << "expects src rows == 1";
  if (ds[0] != mlir::ShapedType::kDynamic && ds[0] != 1)
    return emitOpError() << "expects dst rows == 1";

  // Cols must match when statically known.
  if (ss[1] != mlir::ShapedType::kDynamic && ds[1] != mlir::ShapedType::kDynamic && ss[1] != ds[1])
    return emitOpError() << "expects src/dst cols to match";

  // blockLen must be multiple of 64 (single-list variant constraint).
  int64_t bl = 0;
  if (auto a = getBlockLenAttr())
    bl = a.getInt();
  if (bl <= 0)
    return emitOpError() << "expects blockLen > 0";
  if ((bl % 64) != 0)
    return emitOpError() << "expects blockLen to be a multiple of 64";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMUL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMulOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TMULS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TMulsOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  // scalar is F32Attr by TD; nothing to verify beyond presence.

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TNEG DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TNegOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TNOT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TNotOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  auto elemTy = srcTy.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise NOT";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TOR DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TOrOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  auto elemTy = src0Ty.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise OR";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && s1[i] != mlir::ShapedType::kDynamic && s0[i] != s1[i])
      return emitOpError() << "expects src0 shape to match src1 shape";
    if (s0[i] != mlir::ShapedType::kDynamic && d[i]  != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TORS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TOrsOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/dst to have the same element type";

  auto elemTy = srcTy.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemTy))
    return emitOpError() << "expects integer element type for bitwise OR";

  auto scalarTy = getScalar().getType();
  auto elemITy = mlir::dyn_cast<mlir::IntegerType>(elemTy);
  auto scalarITy = mlir::dyn_cast<mlir::IntegerType>(scalarTy);
  if (!scalarITy)
    return emitOpError() << "expects integer type for scalar";

  if (elemITy.getWidth() != scalarITy.getWidth())
    return emitOpError() << "expects scalar integer width to match element integer width";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTADD DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TPartAddOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTMAX DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TPartMaxOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPARTMIN DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TPartMinOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TPRELU DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TPreluOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();

  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRECIP DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRecipOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRELU DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TReluOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TREM DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRemOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src0/src1/dst to be rank-2 (tile-shaped tilebufs)";

  auto s0 = src0Ty.getShape();
  auto s1 = src1Ty.getShape();
  auto d  = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s0[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s0[i] != d[i])
      return emitOpError() << "expects src0 shape to match dst shape";
    if (s1[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s1[i] != d[i])
      return emitOpError() << "expects src1 shape to match dst shape";
  }

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TREMS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRemSOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects src/dst to be rank-2 (tile-shaped tilebufs)";

  auto s = srcTy.getShape();
  auto d = dstTy.getShape();
  for (int i = 0; i < 2; ++i) {
    if (s[i] != mlir::ShapedType::kDynamic && d[i] != mlir::ShapedType::kDynamic && s[i] != d[i])
      return emitOpError() << "expects src shape to match dst shape";
  }

  // Scalar must be float-like; element type can be int/float (runtime behavior target-defined).
  if (!mlir::isa<mlir::FloatType>(getScalar().getType()))
    return emitOpError() << "expects scalar to be a float type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRESHAPE DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TReshapeOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPAND DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowExpandOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/dst";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDDIV DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowExpandDivOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tile buf types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDMUL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowExpandMulOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWEXPANDSUB DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowExpandSubOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0/src1/dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0/src1/dst";

  auto elemTy = src0Ty.getElementType();
  if (elemTy != src1Ty.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src0/src1/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWMAX DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowMaxOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/tmp/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWMIN DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowMinOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto tmpTy = mlir::dyn_cast<mlir::pto::TileBufType>(getTmp().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !tmpTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/tmp/dst";

  if (srcTy.getRank() != 2 || tmpTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != tmpTy.getElementType() || elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TROWSUM DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRowSumOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/tmp/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/tmp/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/tmp/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TRSQRT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TRsqrtOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/dst";

  auto elemTy = srcTy.getElementType();
  if (elemTy != dstTy.getElementType())
    return emitOpError() << "expects src/dst to have the same element type";

  if (!elemTy.isF16() && !elemTy.isF32())
    return emitOpError() << "expects element type to be f16 or f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSCATTER DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TScatterOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto idxTy = mlir::dyn_cast<mlir::pto::TileBufType>(getIndexes().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());
  if (!srcTy || !idxTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src/indexes/dst";

  if (srcTy.getRank() != 2 || idxTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src/indexes/dst";

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();
  auto idxElem = idxTy.getElementType();

  if (srcElem != dstElem)
    return emitOpError() << "expects src/dst to have the same element type";

  auto isAllowedDataElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  auto isAllowedIndexElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedDataElem(srcElem))
    return emitOpError() << "expects src/dst element type to be i8/i16/i32/f16/bf16/f32";

  if (!isAllowedIndexElem(idxElem))
    return emitOpError() << "expects indexes element type to be i16 or i32";

  // Enforce size relationship between data dtype and index dtype.
  auto bwData = srcElem.getIntOrFloatBitWidth();
  auto bwIdx  = idxElem.getIntOrFloatBitWidth();
  if (bwData != 8 && bwData != 16 && bwData != 32)
    return emitOpError() << "unexpected src/dst element bitwidth " << bwData;

  unsigned dataBytes = bwData / 8;
  unsigned idxBytes  = bwIdx / 8;

  // 4B data -> 4B idx, 2B data -> 2B idx, 1B data -> 2B idx
  unsigned expectedIdxBytes = (dataBytes == 1) ? 2 : dataBytes;
  if (idxBytes != expectedIdxBytes)
    return emitOpError() << "expects indexes element size " << expectedIdxBytes
                         << " bytes for data element size " << dataBytes << " bytes";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSEL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSelOp::verify() {
  auto maskTy = mlir::dyn_cast<mlir::pto::TileBufType>(getMask().getType());
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!maskTy || !src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for mask, src0, src1, dst";

  if (maskTy.getRank() != 2 || src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for mask, src0, src1, dst";

  auto maskElem = maskTy.getElementType();
  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32/f16/bf16/f32";

  if (maskElem.isInteger(8) == false)
    return emitOpError() << "expects mask element type to be i8";

  // Validate tile layouts
  if (src0Ty.getShape()[1] != dstTy.getShape()[1])
    return emitOpError() << "expects src0 and dst cols to match";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSELS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSelsOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (t.isF16() || t.isF32() || t.isBF16()) return true;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32/f16/bf16/f32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSHL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TShlOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSHR DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TShrOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy  = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, dst";

  if (src0Ty.getRank() != 2 || src1Ty.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src0, src1, dst";

  auto srcElem = src0Ty.getElementType();
  auto dstElem = dstTy.getElementType();
  
  if (srcElem != dstElem)
    return emitOpError() << "expects src0 and dst to have the same element type";

  auto isAllowedElem = [&](mlir::Type t) -> bool {
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
      unsigned w = it.getWidth();
      return (w == 8 || w == 16 || w == 32);
    }
    return false;
  };

  if (!isAllowedElem(srcElem))
    return emitOpError() << "expects src0 and dst element type to be i8/i16/i32";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSORT32 DPS/tilebuf op)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// PTO.cpp - Fixes for deprecated and erroneous Type methods
//===----------------------------------------------------------------------===//

mlir::LogicalResult TSort32Op::verify() {
  // Correcting dyn_cast usage for pto::TileType
  auto srcTy = getSrc().getType();
  if (!mlir::isa<mlir::pto::TileBufType>(srcTy)) {
    return emitOpError() << "Expected pto::TilebufType for src.";
  }

  auto dstTy = getDst().getType();
  if (!mlir::isa<mlir::pto::TileBufType>(dstTy)) {
    return emitOpError() << "Expected pto::TilebufType for dst.";
  }

  return success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add verifier for TSQRT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSqrtOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src and dst";

  if (srcTy.getRank() != 2 || dstTy.getRank() != 2)
    return emitOpError() << "expects rank-2 tilebufs for src and dst";

  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();

  if (srcElem != dstElem)
    return emitOpError() << "expects src and dst to have the same element type";

  if (!(srcElem.isa<mlir::FloatType>() || srcElem.isa<mlir::Float16Type>()))
    return emitOpError() << "expects src and dst element type to be float or half";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSTORE_FP DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TStoreFPOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto fpTy = mlir::dyn_cast<mlir::pto::TileBufType>(getFp().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!srcTy || !fpTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src, fp, and dst";

  if (srcTy.getRank() != dstTy.getRank() || srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same rank and element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUB DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSubOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects all tensors to have the same rank";

  if (src0Ty.getElementType() != src1Ty.getElementType() || src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBC DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSubCOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto src2Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc2().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !src2Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, src2, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank() || src2Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects all tensors to have the same rank";

  if (src0Ty.getElementType() != src1Ty.getElementType() || src0Ty.getElementType() != src2Ty.getElementType() || src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, src2, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBS DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSubSOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!srcTy || !dstTy)
    return emitOpError() << "expects tilebuf types for src and dst";

  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError() << "expects src and dst to have the same rank";

  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSUBSC DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSubSCOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, and dst";

  if (src0Ty.getRank() != dstTy.getRank() || src1Ty.getRank() != dstTy.getRank())
    return emitOpError() << "expects src0, src1, and dst to have the same rank";

  if (src0Ty.getElementType() != dstTy.getElementType() || src1Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}
mlir::LogicalResult mlir::pto::TTransOp::verify() {
  auto srcTy = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto tmpTy = mlir::dyn_cast<mlir::pto::TileBufType>(getTmp().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!srcTy || !tmpTy || !dstTy)
    return emitOpError() << "expects tile types for src and dst";

  if (srcTy.getElementType() != dstTy.getElementType() || srcTy.getElementType() != tmpTy.getElementType())
    return emitOpError() << "expects src and dst to have the same element type";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// PTO.cpp  (add TXOR DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TXOROp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc0().getType());
  auto src1Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc1().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !src1Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0, src1, and dst";

  if (src0Ty.getElementType() != src1Ty.getElementType() ||
      src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0, src1, and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TXORS DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TXORSOp::verify() {
  auto src0Ty = mlir::dyn_cast<mlir::pto::TileBufType>(getSrc().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!src0Ty || !dstTy)
    return emitOpError() << "expects tilebuf types for src0 and dst";

  if (src0Ty.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects src0 and dst to have the same element type";

  return mlir::success();
}
//===----------------------------------------------------------------------===//
// PTO.cpp  (add TSYNC DPS/tilebuf implementation)
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::pto::TSYNCOp::verify() {
  auto eventsTy = mlir::dyn_cast<mlir::pto::TileBufType>(getEvents().getType());
  auto dstTy = mlir::dyn_cast<mlir::pto::TileBufType>(getDst().getType());

  if (!eventsTy || !dstTy)
    return emitOpError() << "expects tilebuf types for events and dst";

  if (eventsTy.getElementType() != dstTy.getElementType())
    return emitOpError() << "expects events and dst to have the same element type";

  return mlir::success();
}

mlir::LogicalResult mlir::pto::TPrintOp::verify() {
  auto srcType = getSrc().getType();
  
  // Support TileBufType and PartitionTensorViewType (replaces legacy TileView).
  if (mlir::dyn_cast<mlir::pto::TileBufType>(srcType) ||
      mlir::dyn_cast<mlir::pto::PartitionTensorViewType>(srcType)) {
    return mlir::success();
  }
  
  return emitOpError() << "expects tile_buf or partition_tensor_view types for src";
}

//===----------------------------------------------------------------------===//
// pto.addf custom asm to support BOTH formats:
//
//  (A) legacy tile IR:
//    %r = pto.addf %a, %b : (T, T) -> T
//
//  (B) newer short form:
//    %r = pto.addf %a, %b : T
//===----------------------------------------------------------------------===//

//ParseResult mlir::pto::AddFOp::parse(OpAsmParser &parser, OperationState &result) {
//  OpAsmParser::UnresolvedOperand lhs, rhs;
//
//  // %lhs, %rhs
//  if (parser.parseOperand(lhs) || parser.parseComma() || parser.parseOperand(rhs))
//    return failure();
//
//  // attr-dict
//  if (parser.parseOptionalAttrDict(result.attributes))
//    return failure();
//
//  // :
//  if (parser.parseColon())
//    return failure();
//
//  Type lhsTy, rhsTy, resTy;
//
//  // legacy form: (T, T) -> T
//  if (succeeded(parser.parseOptionalLParen())) {
//    lhsTy = parsePTOTypeAllowNoBang(parser);
//    if (!lhsTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse lhs type");
//
//    if (parser.parseComma()) return failure();
//
//    rhsTy = parsePTOTypeAllowNoBang(parser);
//    if (!rhsTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse rhs type");
//
//    if (parser.parseRParen() || parser.parseArrow()) return failure();
//
//    resTy = parsePTOTypeAllowNoBang(parser);
//    if (!resTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse result type");
//  } else {
//    // short form: T
//    lhsTy = parsePTOTypeAllowNoBang(parser);
//    if (!lhsTy) return parser.emitError(parser.getCurrentLocation(), "failed to parse type");
//    rhsTy = lhsTy;
//    resTy = lhsTy;
//  }
//
//  // resolve operands
//  if (parser.resolveOperand(lhs, lhsTy, result.operands)) return failure();
//  if (parser.resolveOperand(rhs, rhsTy, result.operands)) return failure();
//
//  result.addTypes(resTy);
//  return success();
//}

//void mlir::pto::AddFOp::print(OpAsmPrinter &p) {
//  p << " " << getLhs() << ", " << getRhs();
//  p.printOptionalAttrDict((*this)->getAttrs());
//
//  p << " : (" << getLhs().getType() << ", " << getRhs().getType() << ") -> "
//    << getResult().getType();
//}

// LogicalResult mlir::pto::AddFOp::inferReturnTypes(
//     MLIRContext *context, std::optional<Location> location,
//     ValueRange operands, DictionaryAttr attributes,
//     OpaqueProperties properties, RegionRange regions,
//     SmallVectorImpl<Type> &inferredReturnTypes) {
//   (void)context;
//   (void)location;
//   (void)attributes;
//   (void)properties;
//   (void)regions;

//   if (operands.size() != 2)
//     return failure();
//   if (operands[0].getType() != operands[1].getType())
//     return failure();
//   inferredReturnTypes.push_back(operands[0].getType());
//   return success();
// }

//===----------------------------------------------------------------------===//
// PTO Matmul* custom verification and type inference (keep your existing code)
//===----------------------------------------------------------------------===//

static LogicalResult verifyMatmulCommon(Operation *op, Value lhs, Value rhs,
                                       Value biasOpt, Type maybeDstElemTy,
                                       Type maybeResultElemTy) {
  // ---- case A: tensor/memref (ShapedType) ----
  if (auto lhsTy = dyn_cast<ShapedType>(lhs.getType())) {
    auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
    if (!rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
      return op->emitOpError("lhs/rhs must be ranked tensor or memref");

    if (lhsTy.getElementType() != rhsTy.getElementType())
      return op->emitOpError()
             << "lhs/rhs element types must match, but got lhs="
             << lhsTy.getElementType() << " rhs=" << rhsTy.getElementType();

    if (biasOpt) {
      auto biasTy = dyn_cast<ShapedType>(biasOpt.getType());
      if (!biasTy || !biasTy.hasRank())
        return op->emitOpError("bias must be ranked tensor or memref");
      if (biasTy.getElementType() != lhsTy.getElementType())
        return op->emitOpError()
               << "bias element type must match lhs/rhs element type, but got bias="
               << biasTy.getElementType() << " vs " << lhsTy.getElementType();
    }

    if (maybeDstElemTy && maybeDstElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "dst element type must match lhs/rhs element type, but got dst="
             << maybeDstElemTy << " vs " << lhsTy.getElementType();

    if (maybeResultElemTy && maybeResultElemTy != lhsTy.getElementType())
      return op->emitOpError()
             << "result element type must match lhs/rhs element type, but got result="
             << maybeResultElemTy << " vs " << lhsTy.getElementType();

    return success();
  }

  // ---- case B: tile ----
  auto lhsTile = dyn_cast<mlir::pto::TileType>(lhs.getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(rhs.getType());
  if (!lhsTile || !rhsTile)
    return op->emitOpError("lhs/rhs must be ranked tensor/memref or !pto.tile");

  if (lhsTile.getElementType() != rhsTile.getElementType())
    return op->emitOpError() << "tile lhs/rhs element types must match, but got lhs="
                             << lhsTile.getElementType() << " rhs=" << rhsTile.getElementType();

  if ((int64_t)lhsTile.getShape().size() != 2 || (int64_t)rhsTile.getShape().size() != 2)
    return op->emitOpError("tile matmul expects 2D tiles");

  if (lhsTile.getShape()[1] != rhsTile.getShape()[0])
    return op->emitOpError() << "tile matmul expects lhs dim1 == rhs dim0, but got "
                             << lhsTile.getShape()[1] << " vs " << rhsTile.getShape()[0];

  if (biasOpt) {
    auto biasTile = dyn_cast<mlir::pto::TileType>(biasOpt.getType());
    if (!biasTile)
      return op->emitOpError("bias must be !pto.tile when lhs/rhs are tile");
    if (biasTile.getElementType() != lhsTile.getElementType())
      return op->emitOpError("bias element type must match tile element type");
  }

  if (maybeDstElemTy && maybeDstElemTy != lhsTile.getElementType())
    return op->emitOpError() << "dst element type mismatch";

  if (maybeResultElemTy && maybeResultElemTy != lhsTile.getElementType())
    return op->emitOpError() << "result element type mismatch";

  return success();
}


static LogicalResult verifyMatmulAccCommon(Operation *op, Value accIn, Value lhs,
                                          Value rhs, Type maybeDstElemTy,
                                          Type maybeResultElemTy) {
  auto accTy = dyn_cast<ShapedType>(accIn.getType());
  auto lhsTy = dyn_cast<ShapedType>(lhs.getType());
  auto rhsTy = dyn_cast<ShapedType>(rhs.getType());
  if (!accTy || !lhsTy || !rhsTy || !accTy.hasRank() || !lhsTy.hasRank() ||
      !rhsTy.hasRank())
    return op->emitOpError("acc_in/lhs/rhs must be ranked tensor or memref");

  Type elem = accTy.getElementType();
  if (lhsTy.getElementType() != elem || rhsTy.getElementType() != elem)
    return op->emitOpError()
           << "acc_in/lhs/rhs element types must match, but got acc_in="
           << elem << " lhs=" << lhsTy.getElementType()
           << " rhs=" << rhsTy.getElementType();

  if (maybeDstElemTy && maybeDstElemTy != elem)
    return op->emitOpError()
           << "dst element type must match acc_in element type, but got dst="
           << maybeDstElemTy << " vs " << elem;

  if (maybeResultElemTy && maybeResultElemTy != elem)
    return op->emitOpError()
           << "result element type must match acc_in element type, but got result="
           << maybeResultElemTy << " vs " << elem;

  if (accTy.getRank() == 2 && lhsTy.getRank() == 2 && rhsTy.getRank() == 2 &&
      accTy.hasStaticShape() && lhsTy.hasStaticShape() &&
      rhsTy.hasStaticShape()) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t K1 = lhsTy.getDimSize(1);
    int64_t K2 = rhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);

    if (K1 != K2)
      return op->emitOpError()
             << "matmul_acc expects lhs dim1 == rhs dim0, but got " << K1
             << " vs " << K2;

    if (accTy.getDimSize(0) != M || accTy.getDimSize(1) != N)
      return op->emitOpError()
             << "acc_in must have shape {M,N} matching matmul result, but got acc_in="
             << accTy << " while expected {" << M << "," << N << "}";
  }

  return success();
}

// non-DPS: pto.matmul
LogicalResult mlir::pto::MatmulOp::verify() {
  if (getOperation()->getNumResults() != 1)
    return emitOpError("expects exactly 1 result for non-dps matmul");

  Type resT = getResult().getType();

  Type resElem;
  if (auto rt = dyn_cast<RankedTensorType>(resT)) {
    resElem = rt.getElementType();
  } else if (auto tt = dyn_cast<mlir::pto::TileType>(resT)) {
    resElem = tt.getElementType();
  } else {
    return emitOpError("result must be ranked tensor or !pto.tile");
  }

  Value bias = getBias();
  return verifyMatmulCommon(getOperation(), getLhs(), getRhs(),
                            (bias ? bias : Value{}),
                            /*dstElem=*/Type{},
                            /*resElem=*/resElem);
}


// DPS: pto.matmul_dps
LogicalResult mlir::pto::MatmulDpsOp::verify() {
  return success();
//  Type dstTy = getDst().getType();

//  // 1. 如果是 TileBufType，直接通过验证 (或者做简单的类型检查)
//  // TileBuf 是我们新引入的类型，不属于 ShapedType (Tensor/MemRef)
//  if (llvm::isa<pto::TileBufType>(dstTy)) {
//     // TODO: 如果需要，这里可以添加针对 TileBuf 的形状检查 (M, N, K)
//     // 目前为了跑通流程，我们假设上游产生的 TileBuf 都是合法的
//     return success();
//  }

//  // 2. 针对 Tensor/MemRef 的检查 (保持不变)
//  bool hasResult = (getOperation()->getNumResults() == 1);

//  auto dstShaped = dyn_cast<ShapedType>(dstTy);
//  if (!dstShaped || !dstShaped.hasRank())
//    return emitOpError("dst must be ranked tensor, memref or tile_buf");

//  Type dstElem = dstShaped.getElementType();
  
//  Type resElem;
//  if (hasResult) {
//    auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
//    if (!resTy)
//      return emitOpError("result must be a ranked tensor type when present");
//    resElem = resTy.getElementType();

//    if (isa<RankedTensorType>(dstTy)) {
//      if (getResult().getType() != dstTy)
//        return emitOpError()
//               << "when dst is a tensor, result type must equal dst type, but got result="
//               << getResult().getType() << " dst=" << dstTy;
//    }
//  }

//  if (isa<MemRefType>(dstTy) && hasResult)
//    return emitOpError(
//        "expects no tensor result when dst is a memref (post-bufferization)");

//  Value bias = getBias();
//  return verifyMatmulCommon(getOperation(), getLhs(), getRhs(),
//                            (bias ? bias : Value{}),
//                            /*dstElem=*/dstElem,
//                            /*resElem=*/resElem);
}

LogicalResult mlir::pto::MatmulAccOp::verify() {
  if (getOperation()->getNumResults() != 1)
    return emitOpError("expects exactly 1 result for non-dps matmul_acc");

  auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!resTy)
    return emitOpError("result must be a ranked tensor type");

  if (auto accRT = dyn_cast<RankedTensorType>(getAccIn().getType())) {
    if (accRT != resTy)
      return emitOpError() << "result type must equal acc_in tensor type, but got result="
                           << resTy << " acc_in=" << accRT;
  }

  return verifyMatmulAccCommon(getOperation(), getAccIn(), getLhs(), getRhs(),
                               /*dstElem=*/Type{},
                               /*resElem=*/resTy.getElementType());
}

LogicalResult mlir::pto::MatmulAccDpsOp::verify() {
  return success();
//  bool hasResult = (getOperation()->getNumResults() == 1);

//  auto dstShaped = dyn_cast<ShapedType>(getDst().getType());
//  if (!dstShaped || !dstShaped.hasRank())
//    return emitOpError("dst must be ranked tensor or memref");

//  Type dstElem = dstShaped.getElementType();

//  Type resElem;
//  if (hasResult) {
//    auto resTy = dyn_cast<RankedTensorType>(getResult().getType());
//    if (!resTy)
//      return emitOpError("result must be a ranked tensor type when present");
//    resElem = resTy.getElementType();

//    if (isa<RankedTensorType>(getDst().getType())) {
//      if (getResult().getType() != getDst().getType())
//        return emitOpError()
//               << "when dst is a tensor, result type must equal dst type, but got result="
//               << getResult().getType() << " dst=" << getDst().getType();
//    }
//  }

//  if (isa<MemRefType>(getDst().getType()) && hasResult)
//    return emitOpError(
//        "expects no tensor result when dst is a memref (post-bufferization)");

//  return verifyMatmulAccCommon(getOperation(), getAccIn(), getLhs(), getRhs(),
//                               /*dstElem=*/dstElem,
//                               /*resElem=*/resElem);
}

//===----------------------------------------------------------------------===//
// inferReturnTypes() for matmul ops (keep your existing code)
//===----------------------------------------------------------------------===
static mlir::Type inferMatmulTileResult2DFromAB(MLIRContext *context, ValueRange operands) {
  if (operands.size() < 2)
    return mlir::Type();

  auto lhsTile = dyn_cast<mlir::pto::TileType>(operands[0].getType());
  auto rhsTile = dyn_cast<mlir::pto::TileType>(operands[1].getType());
  if (!lhsTile || !rhsTile)
    return mlir::Type();

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

static RankedTensorType inferMatmulResult2DFromAB(ValueRange operands) {
  if (operands.size() < 2)
    return RankedTensorType();

  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  auto rhsTy = dyn_cast<ShapedType>(operands[1].getType());
  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank())
    return RankedTensorType();

  Type elemTy = lhsTy.getElementType();

  if (operands.size() >= 3) {
    if (auto biasRT = dyn_cast<RankedTensorType>(operands[2].getType()))
      return RankedTensorType::get(biasRT.getShape(), elemTy);
    if (auto biasMR = dyn_cast<MemRefType>(operands[2].getType())) {
      if (biasMR.hasStaticShape())
        return RankedTensorType::get(biasMR.getShape(), elemTy);
    }
  }

  if (lhsTy.getRank() >= 2 && rhsTy.getRank() >= 2) {
    int64_t M = lhsTy.getDimSize(0);
    int64_t N = rhsTy.getDimSize(1);
    return RankedTensorType::get({M, N}, elemTy);
  }

  return RankedTensorType();
}

LogicalResult mlir::pto::MatmulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;

  if (mlir::Type tileTy = inferMatmulTileResult2DFromAB(context, operands)) {
    inferredReturnTypes.push_back(tileTy);
    return success();
  }

  auto rt2d = inferMatmulResult2DFromAB(operands);
  if (rt2d) {
    inferredReturnTypes.push_back(rt2d);
    return success();
  }

  if (operands.size() < 2)
    return failure();
  auto lhsTy = dyn_cast<ShapedType>(operands[0].getType());
  if (!lhsTy)
    return failure();

  inferredReturnTypes.push_back(
      RankedTensorType::get({ShapedType::kDynamic}, lhsTy.getElementType()));
  return success();
}


LogicalResult mlir::pto::MatmulDpsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)context;
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;

  if (operands.size() < 3)
    return failure();
  Type dstTy = operands[2].getType();
  if (auto rt = dyn_cast<RankedTensorType>(dstTy))
    inferredReturnTypes.push_back(rt);
  return success();
}

static RankedTensorType inferAccReturnFromAccIn(ValueRange operands) {
  if (operands.empty())
    return RankedTensorType();
  if (auto accRT = dyn_cast<RankedTensorType>(operands[0].getType()))
    return accRT;
  return RankedTensorType();
}

LogicalResult mlir::pto::MatmulAccOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)context;
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;

  auto rt = inferAccReturnFromAccIn(operands);
  if (!rt)
    return failure();
  inferredReturnTypes.push_back(rt);
  return success();
}

LogicalResult mlir::pto::MatmulAccDpsOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  (void)context;
  (void)location;
  (void)attributes;
  (void)properties;
  (void)regions;

  if (operands.size() < 4)
    return failure();
  Type dstTy = operands[3].getType();
  if (auto rt = dyn_cast<RankedTensorType>(dstTy))
    inferredReturnTypes.push_back(rt);
  return success();
}

namespace mlir {
namespace pto {

static LogicalResult parseShapeAndElem(AsmParser &parser,
                                       SmallVectorImpl<int64_t> &shape,
                                       Type &elementType,
                                       bool allowDynamic) {
  if (parser.parseLess())
    return failure();

  if (parser.parseDimensionList(shape, allowDynamic))
    return failure();

  if (parser.parseType(elementType))
    return failure();

  if (parser.parseGreater())
    return failure();

  return success();
}

static void printShapeAndElem(AsmPrinter &printer,
                              ArrayRef<int64_t> shape,
                              Type elementType) {
  printer << "<";
  for (auto d : shape) {
    if (d == ShapedType::kDynamic)
      printer << "?";
    else
      printer << d;
    printer << "x";
  }
  printer.printType(elementType);
  printer << ">";
}

// =============================================================================
// PartitionTensorViewType Implementation
// =============================================================================

Type PartitionTensorViewType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  
  return PartitionTensorViewType::get(parser.getContext(), shape, elemTy);
}

void PartitionTensorViewType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// ---- TileType ----
Type TileType::parse(AsmParser &parser) {
  SmallVector<int64_t, 4> shape;
  Type elemTy;
  if (failed(parseShapeAndElem(parser, shape, elemTy, /*allowDynamic=*/true)))
    return Type();
  return TileType::get(parser.getContext(), shape, elemTy);
}

void TileType::print(AsmPrinter &printer) const {
  printShapeAndElem(printer, getShape(), getElementType());
}

// =============================================================================
// Decompose Helper (Reverse Engineering AffineMap -> Strides)
// =============================================================================

// Helper: 递归地将 Add 表达式拆解为单独的项列表
static void flattenAddExpr(AffineExpr expr, SmallVectorImpl<AffineExpr> &terms) {
  if (auto add = expr.dyn_cast<AffineBinaryOpExpr>()) {
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
  
  if (map.getNumResults() != 1) return;
  
  // 2. 摊平表达式
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  // 3. 分析每一项
  for (auto term : terms) {
    // 情况 A: dN * Const 或 Const * dN
    if (auto mul = term.dyn_cast<AffineBinaryOpExpr>()) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();

        // 尝试匹配 LHS=Dim, RHS=Const
        if (auto dim = lhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = rhs.dyn_cast<AffineConstantExpr>()) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
        
        // 尝试匹配 LHS=Const, RHS=Dim (乘法交换律)
        if (auto dim = rhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = lhs.dyn_cast<AffineConstantExpr>()) {
            strides[dim.getPosition()] = cst.getValue();
            continue;
          }
        }
      }
    }
    // 情况 B: 单独的 dN (隐含 Stride = 1)
    else if (auto dim = term.dyn_cast<AffineDimExpr>()) {
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
  if (parser.parseLSquare()) return failure();
  do {
    int64_t stride;
    if (parser.parseInteger(stride)) return failure();
    strides.push_back(stride);
  } while (succeeded(parser.parseOptionalComma()));
  if (parser.parseRSquare()) return failure();
  return success();
}

// The custom attribute parser for: strided<[64, 1], offset: [?, ?]>
static ParseResult parseStridedLayout(AsmParser &parser, Attribute &layout) {
  if (parser.parseLess()) return failure();
  
  // 1. Parse Strides
  SmallVector<int64_t> strides;
  if (parseStrideList(parser, strides)) return failure();
  
  bool isMultiDim = false;
  unsigned numSymbols = 0;

  // 2. Parse Offset
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("offset") || parser.parseColon()) return failure();
    
    // Check for multi-dim syntax: [?, ?]
    if (succeeded(parser.parseOptionalLSquare())) {
      isMultiDim = true;
      do {
        if (parser.parseQuestion()) return failure();
        numSymbols++;
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare()) return failure();
    } else {
      // Fallback for old scalar syntax '?'
      if (parser.parseOptionalQuestion()) { /* handle single scalar */ }
    }
  }
  
  if (parser.parseGreater()) return failure();

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

static void printLayout(AsmPrinter &printer, Attribute layoutAttr) {
  if (!layoutAttr) return;
  auto mapAttr = llvm::dyn_cast<AffineMapAttr>(layoutAttr);
  if (!mapAttr) { printer << ", " << layoutAttr; return; }

  AffineMap map = mapAttr.getValue();
  if (map.isIdentity()) return; 

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
      if (i < numSyms - 1) printer << ", ";
    }
    printer << "]";
  }
  printer << ">";
}

// ---- TileBuf ---


// Tile subset 相关实现

// =============================================================================
// Op Interface Implementation: SubsetOp
// =============================================================================

LogicalResult SubsetOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  // 1. 获取 Source Type
  if (operands.empty()) return failure();
  auto sourceType = llvm::dyn_cast<TileBufType>(operands[0].getType());
  if (!sourceType) return failure();

  // 2. 获取 Result Shape (Sizes)
  ArrayAttr sizeAttr;
  if (properties) {
    const auto *prop = properties.as<SubsetOp::Properties *>();
    if (prop) sizeAttr = prop->sizes;
  }
  if (!sizeAttr) return failure();

  SmallVector<int64_t> resultShape;
  for (auto attr : sizeAttr) {
    resultShape.push_back(llvm::cast<IntegerAttr>(attr).getInt());
  }

  // 3. [核心修改] 不再继承 sourceStrides，而是计算 resultShape 的连续 Strides
  // 假设 Row-Major: [Dim1*Dim2*..., Dim2*..., ..., 1]
  // 对于 2D [M, N] -> Strides [N, 1]
  SmallVector<int64_t> contiguousStrides;
  contiguousStrides.resize(resultShape.size());
  
  int64_t runningStride = 1;
  for (int i = resultShape.size() - 1; i >= 0; --i) {
    contiguousStrides[i] = runningStride;
    runningStride *= resultShape[i];
  }
  
  // 4. 构建 Layout Map
  // 虽然是连续的，但因为 subset 会产生 offset，所以我们依然需要一个 Map 来表示 offset
  // Map = (d0, d1)[s0, s1] -> (d0 * 32 + d1 * 1 + offset)
  // 注意：这里的 stride 变成了 32 (Child Width)，而不是 64 (Parent Width)
  AffineMap newLayout = buildStrictBitwiseAffineMap(context, contiguousStrides, /*isMultiDim=*/true);

  auto cfg = sourceType.getConfigAttr(); // 或 sourceType.getConfig() 视你现在实现
  // 5. 构建 Result Type
  auto resultType = TileBufType::get(
      context, resultShape, sourceType.getElementType(),
      sourceType.getMemorySpace(), sourceType.getValidShape(), cfg);

  inferredReturnTypes.push_back(resultType);
  return success();
}

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;
 
// =============================================================================
// Helper Functions
// =============================================================================
 
static AddressSpace getAddressSpace(Value val) {
  auto type = llvm::dyn_cast<MemRefType>(val.getType());
  if (!type) return AddressSpace::Zero; // Default
 
  // 假设你的 AddressSpaceAttr 存储在 MemRef 的 memorySpace 中
  // 需要根据你的 getPTOAddressSpaceAttr 实现来调整
  auto attr = llvm::dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (attr) return attr.getAddressSpace();
  return AddressSpace::Zero;
}
 
// =============================================================================
// CopyOp Implementation
// =============================================================================
 
PIPE CopyOp::getPipe() {
  AddressSpace srcSpace = getAddressSpace(getSrc());
  AddressSpace dstSpace = getAddressSpace(getDst());
 
  bool isSrcGlobal = (srcSpace == AddressSpace::GM || srcSpace == AddressSpace::MAT); 
  bool isDstGlobal = (dstSpace == AddressSpace::GM || srcSpace == AddressSpace::MAT);

  // L0A->LEFT, L0B->RIGHT, L0C->ACC
  bool isSrcLocal  = (srcSpace == AddressSpace::VEC || srcSpace == AddressSpace::LEFT || srcSpace == AddressSpace::RIGHT || srcSpace == AddressSpace::ACC);
  bool isDstLocal  = (dstSpace == AddressSpace::VEC || dstSpace == AddressSpace::LEFT || dstSpace == AddressSpace::RIGHT || dstSpace == AddressSpace::ACC);

  if (isSrcGlobal && isDstLocal) {
    return PIPE::PIPE_MTE2;
  }
  if (isSrcLocal && isDstGlobal) {
    return PIPE::PIPE_MTE3;
  }
  
  // UB->UB 拷贝，Ascend 910 上通常使用 MTE2 或 Vector
  // 为了安全起见，如果没有明确指令，可以归类为 MTE2 (DataMove)
  return PIPE::PIPE_MTE2; 
}
 
// =============================================================================
// MovDpsOp Implementation
// =============================================================================
 
PIPE MovDpsOp::getPipe() {
  // 简单判断：如果 dst 是 L0 (L0A/L0B)，则是 MTE1
  AddressSpace dstSpace = getAddressSpace(getDst());
  if (dstSpace == AddressSpace::LEFT || dstSpace == AddressSpace::RIGHT) {
    return PIPE::PIPE_MTE1;
  }
  return PIPE::PIPE_V;
}

// =============================================================================
// Side Effects Implementation
// =============================================================================
 
// [Fix] 辅助函数：重载以支持 OpOperand* 和 OpResult，避免直接传 Value
 
// 针对操作数 (Operand) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpOperand *operand, MemoryEffects::Effect *effect) {
  if (operand)
    effects.emplace_back(effect, operand, SideEffects::DefaultResource::get());
}
 
// 针对结果 (Result) 的重载
static void addEffect(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
    OpResult result, MemoryEffects::Effect *effect) {
  if (result)
    effects.emplace_back(effect, result, SideEffects::DefaultResource::get());
}
 
// 1. LoadDpsOp: Read(src) -> Write(dst)
void LoadDpsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // 使用 &getSrcMutable() 获取 OpOperand*
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
 
// 2. StoreDpsOp: Read(src) -> Write(dst)
void StoreDpsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
 
// 3. CopyOp: Read(src) -> Write(dst)
void CopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
 
// 4. MatmulDpsOp: Read(lhs, rhs, bias?) -> Write(dst)
void MatmulDpsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  
  // Bias 是 Optional 的，getBiasMutable() 返回 MutableOperandRange
  auto biasMutable = getBiasMutable();
  if (!biasMutable.empty()) {
    addEffect(effects, &biasMutable[0], MemoryEffects::Read::get());
  }
  
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
 
// 5. AddFDpsOp: Read(lhs, rhs) -> Write(dst)
void AddFDpsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// AbsDpsOp: Read src -> Write dst
void AbsOp_DPS::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// 6. MovDpsOp
void MovDpsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}
 
// === TLoadOp ===
// Read: src, Write: dst
// 针对 OpOperand* 的重载
void TLoadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // [Fix] 单个操作数，直接取地址
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
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMovOp ===
// Read: src, Write: dst
void TMovOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === ReluOp_DPS ===
// Read: src, Write: dst
void ReluOp_DPS::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  addEffect(effects, &getSrcMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getDstMutable(), MemoryEffects::Write::get());
}

// === TMatmulOp ===
// Read: lhs, rhs, (bias), Write: dst
void TMatmulOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  // Singleton -> 直接取地址
  addEffect(effects, &getLhsMutable(), MemoryEffects::Read::get());
  addEffect(effects, &getRhsMutable(), MemoryEffects::Read::get());
  
  // Optional -> 返回的是 Range，需要迭代器
  auto biasMutable = getBiasMutable();
  if (!biasMutable.empty()) {
    // 这里的 Range 迭代器解引用后是 OpOperand&，再取地址
    addEffect(effects, &*biasMutable.begin(), MemoryEffects::Read::get());
  }

  // Singleton
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

void TPrintOp::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), mlir::SideEffects::DefaultResource::get());
}

void PrintOp_DPS::getEffects(SmallVectorImpl<mlir::SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), mlir::SideEffects::DefaultResource::get());
}

// [Include 必须放在最后]
#include "PTO/IR/PTOInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "PTO/IR/PTOOps.cpp.inc"
