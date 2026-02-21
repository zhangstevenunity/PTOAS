//===- PTOToEmitC.cpp - PTO to EmitC conversion pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"                   
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
namespace mlir {
#define GEN_PASS_DEF_EMITPTOMANUAL
#include "PTO/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

static const char *addrSpaceQualifier(pto::AddressSpace as) {
  switch (as) {
  case pto::AddressSpace::Zero:
    return "__gm__";
  case pto::AddressSpace::VEC:
    return "__ubuf__";
  case pto::AddressSpace::GM:
    return "__gm__";
  case pto::AddressSpace::MAT:
    return "__cbuf__";
  case pto::AddressSpace::LEFT:
    return "__ca__";
  case pto::AddressSpace::RIGHT:
    return "__cb__";
  case pto::AddressSpace::ACC:
    return "__cc__";
  case pto::AddressSpace::BIAS:
    // Bias tiles are special in pto-isa; keep a safe fallback qualifier.
    return "__gm__";
  case pto::AddressSpace::SCALING:
    // pto-isa TileType::Scaling maps to __fbuf__ (see pto/common/memory.hpp).
    return "__fbuf__";
  }
  return "__gm__";
}

static Value peelUnrealized(Value v) {
  if (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>())
    return castOp.getOperand(0);
  return v;
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class PTOToEmitCTypeConverter : public TypeConverter {
public:
  PTOToEmitCTypeConverter(MLIRContext *Ctx) {
    // ---------------------------------------------------------
    // 1. 基本类型 (f32, i32, index)
    // ---------------------------------------------------------
    addConversion([Ctx](FloatType type) -> Type {
      if (type.isF32()) return emitc::OpaqueType::get(Ctx, "float");
      if (type.isF16()) return emitc::OpaqueType::get(Ctx, "half");
      if (type.isBF16()) return emitc::OpaqueType::get(Ctx, "__bf16");
      if (type.isF64()) return emitc::OpaqueType::get(Ctx, "double");
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](IntegerType type) -> Type {
      // [关键修改] i1 保持为 i1，不要转为 emitc.opaque<"bool">
      // 这样 emitc.if (接受 i1) 就不会报错。
      // 在打印 C++ 代码时，i1 会自动打印为 bool。
      //if (type.getWidth() == 1) return IntegerType::get(Ctx, 1); 
      if (type.getWidth() == 1) return type; // <--- 保持 i1 不变

      // Prefer fixed-width C types. Preserve signedness if the MLIR integer is
      // explicitly signed/unsigned; treat signless as signed by default.
      const bool isUnsigned = type.isUnsignedInteger();
      switch (type.getWidth()) {
      case 8:
        return emitc::OpaqueType::get(Ctx, isUnsigned ? "uint8_t" : "int8_t");
      case 16:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint16_t" : "int16_t");
      case 32:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint32_t" : "int32_t");
      case 64:
        return emitc::OpaqueType::get(Ctx,
                                      isUnsigned ? "uint64_t" : "int64_t");
      default:
        llvm::errs() << "[Debug] Unsupported IntegerType width: "
                     << type.getWidth() << "\n";
        return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
      }
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });

    // vector<4xi16> (e.g. TMRGSORT executedNumList) -> pto::MrgSortExecutedNumList
    addConversion([Ctx](VectorType type) -> Type {
      if (type.getRank() == 1 && type.getNumElements() == 4 &&
          type.getElementType().isInteger(16))
        return emitc::OpaqueType::get(Ctx, "pto::MrgSortExecutedNumList");
      return Type{};
    });
    
    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([Ctx](emitc::OpaqueType type) { return type; });
    addConversion([Ctx](emitc::PointerType type) { return type; });

    // ---------------------------------------------------------
    // 2.5 PtrType 转换 (指针类型)
    // ---------------------------------------------------------
    addConversion([this, Ctx](pto::PtrType type) -> std::optional<Type> {
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType);
      if (!newElemType)
        return std::nullopt;

      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
        llvm::errs() << "  [Error] PtrType elem type is not OpaqueType: "
                     << newElemType << "\n";
        return std::nullopt;
      }

      std::string qualifier = "__gm__";

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      return emitc::PointerType::get(
          emitc::OpaqueType::get(Ctx, finalTypeStr));
    });

    // ---------------------------------------------------------
    // 3. MemRef 转换 (Debug 重点)
    // ---------------------------------------------------------
    addConversion([this, Ctx](MemRefType type) -> std::optional<Type> {
      llvm::errs() << "[Debug] Converting MemRef: " << type << "\n";

      // A. 转换元素类型
      Type elemType = type.getElementType();
      Type newElemType = convertType(elemType); 
      if (!newElemType) {
        llvm::errs() << "  [Error] Failed to convert element type: " << elemType << "\n";
        return std::nullopt;
      }
      
      // 获取元素类型的字符串
      std::string elemTypeStr;
      if (auto opq = dyn_cast<emitc::OpaqueType>(newElemType)) {
        elemTypeStr = opq.getValue().str();
      } else {
         llvm::errs() << "  [Error] Converted element type is not OpaqueType: " << newElemType << "\n";
         return std::nullopt;
      }

      // B. 处理 Memory Space
      std::string qualifier = "";
      Attribute memorySpace = type.getMemorySpace();
      
      if (!memorySpace) {
         qualifier = "__gm__";
      } else if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
         qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
      } else {
         llvm::errs() << "  [Warning] Unknown MemorySpace Attribute type: " << memorySpace << "\n";
         qualifier = "__gm__"; // Fallback
      }

      std::string finalTypeStr = qualifier + " " + elemTypeStr;
      llvm::errs() << "  [Success] -> " << finalTypeStr << "*\n";
      
      return emitc::PointerType::get(emitc::OpaqueType::get(Ctx, finalTypeStr));
    });

    // ---------------------------------------------------------
    // 4. Function & Materialization
    // ---------------------------------------------------------
    addConversion([this](FunctionType type) -> Type {
      SmallVector<Type> inputs;
      if (failed(convertTypes(type.getInputs(), inputs))) return Type{};
      SmallVector<Type> results;
      if (failed(convertTypes(type.getResults(), results))) return Type{};
      return FunctionType::get(type.getContext(), inputs, results);
    });

    auto materializeCast = [](OpBuilder &Builder, Type ResultType,
                              ValueRange Inputs, Location Loc) -> Value {
      if (Inputs.size() != 1) return Value();
      return Builder.create<UnrealizedConversionCastOp>(Loc, ResultType, Inputs[0]).getResult(0);
    };

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
    // Needed for region/block signature conversions (e.g. CFG block args).
    addArgumentMaterialization(materializeCast);
  }
};

static constexpr unsigned kPTOIndexBitWidth =
    32; // keep consistent with IndexType conversion

// Forward declarations (definitions below).
static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth);
static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth);
static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth);
static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth);
static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal);
static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value);
static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src);
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth);

//===----------------------------------------------------------------------===//
// Arith -> EmitC (full dialect coverage for scalar ops)
//===----------------------------------------------------------------------===//

template <typename ArithOp, typename EmitCOp>
struct ArithSimpleBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getOperands());
    return success();
  }
};

// Integer bitwise ops (andi/ori/xori) on signless integers: perform in unsigned
// to avoid signedness pitfalls, then cast back.
template <typename ArithOp, typename EmitCOp>
struct ArithUnsignedBitwiseBinaryToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = this->getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<EmitCOp>(op, dstTy, adaptor.getLhs(),
                                           adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value resU = rewriter.create<EmitCOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, resU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivUIToEmitC : public OpConversionPattern<arith::DivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::DivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithRemUIToEmitC : public OpConversionPattern<arith::RemUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value remU = rewriter.create<emitc::RemOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, remU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivUIToEmitC : public OpConversionPattern<arith::CeilDivUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value one = makeEmitCIntConstant(rewriter, loc, uTy, 1);
    Value rhsMinusOne = rewriter.create<emitc::SubOp>(loc, uTy, rhsU, one);
    Value num = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsMinusOne);
    Value divU = rewriter.create<emitc::DivOp>(loc, uTy, num, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, divU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCeilDivSIToEmitC : public OpConversionPattern<arith::CeilDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CeilDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsSame =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::eq, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsSame);

    Value qPlusOne = rewriter.create<emitc::AddOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qPlusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithFloorDivSIToEmitC : public OpConversionPattern<arith::FloorDivSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FloorDivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
    Value one = makeEmitCIntConstant(rewriter, loc, dstTy, 1);

    Value q0 = rewriter.create<emitc::DivOp>(loc, dstTy, adaptor.getLhs(),
                                             adaptor.getRhs());
    Value r = rewriter.create<emitc::RemOp>(loc, dstTy, adaptor.getLhs(),
                                            adaptor.getRhs());

    Value rNeZero = rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                                  emitc::CmpPredicate::ne, r,
                                                  zero);
    Value lhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getLhs(),
                                      zero);
    Value rhsLt0 =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::lt, adaptor.getRhs(),
                                      zero);
    Value signsDifferent =
        rewriter.create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                      emitc::CmpPredicate::ne, lhsLt0, rhsLt0);
    Value adjust =
        rewriter.create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                             rNeZero, signsDifferent);

    Value qMinusOne = rewriter.create<emitc::SubOp>(loc, dstTy, q0, one);
    Value result = rewriter.create<emitc::ConditionalOp>(loc, dstTy, adjust,
                                                         qMinusOne, q0);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftLeftToEmitC : public OpConversionPattern<arith::ShLIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShLIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // Compute on u8 and truncate to i1.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseLeftShiftOp>(loc, u8Ty, lhsU8,
                                                            rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseLeftShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightUIToEmitC : public OpConversionPattern<arith::ShRUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value shU =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, shU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithShiftRightSIToEmitC : public OpConversionPattern<arith::ShRSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ShRSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    if (bitWidth == 1) {
      // (x >> y) on i1 is either x (y==0) or 0 (y!=0); approximate in u8.
      auto u8Ty = getUnsignedIntOpaqueType(rewriter.getContext(), 8);
      Value lhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getLhs());
      Value rhsU8 = emitCCast(rewriter, loc, u8Ty, adaptor.getRhs());
      Value sh = rewriter.create<emitc::BitwiseRightShiftOp>(loc, u8Ty, lhsU8,
                                                             rhsU8);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, u8Ty, sh,
                                               makeEmitCIntConstant(rewriter, loc,
                                                                    u8Ty, 1));
      rewriter.replaceOp(op, emitCCast(rewriter, loc, dstTy, masked));
      return success();
    }

    // Signed arithmetic shift; cast RHS to unsigned to interpret shift amount.
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value sh =
        rewriter.create<emitc::BitwiseRightShiftOp>(loc, dstTy, adaptor.getLhs(),
                                                    rhsU);
    rewriter.replaceOp(op, sh);
    return success();
  }
};

struct ArithNegFToEmitC : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::UnaryMinusOp>(op, dstTy, adaptor.getOperand());
    return success();
  }
};

struct ArithRemFToEmitC : public OpConversionPattern<arith::RemFOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::RemFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Use builtin `fmod` when possible. For f16, compute in float and cast back.
    Type callTy = dstTy;
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF16()) {
        auto f32Ty = emitc::OpaqueType::get(rewriter.getContext(), "float");
        lhs = emitCCast(rewriter, loc, f32Ty, lhs);
        rhs = emitCCast(rewriter, loc, f32Ty, rhs);
        callTy = f32Ty;
      }
    }

    // Prefer `__builtin_fmod*` to avoid relying on extra headers.
    llvm::StringRef callee = "__builtin_fmod";
    if (auto opFloatTy = dyn_cast<FloatType>(op.getType())) {
      if (opFloatTy.isF32() || opFloatTy.isF16())
        callee = "__builtin_fmodf";
      else if (opFloatTy.isF64())
        callee = "__builtin_fmod";
    }

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{callTy}, callee, ValueRange{lhs, rhs},
        /*args=*/ArrayAttr{}, /*template_args=*/ArrayAttr{});
    Value result = call.getResult(0);
    if (callTy != dstTy)
      result = emitCCast(rewriter, loc, dstTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithSelectToEmitC : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getCondition().getType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "only scalar i1 conditions supported for arith.select");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    auto cond =
        rewriter.create<emitc::ConditionalOp>(op.getLoc(), dstTy,
                                              adaptor.getCondition(),
                                              adaptor.getTrueValue(),
                                              adaptor.getFalseValue());
    rewriter.replaceOp(op, cond.getResult());
    return success();
  }
};

struct ArithExtUIToEmitC : public OpConversionPattern<arith::ExtUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 -> iN: bool to integer already behaves as 0/1.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithExtSIToEmitC : public OpConversionPattern<arith::ExtSIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::ExtSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // i1 sign-extension: 0 -> 0, 1 -> -1.
    if (srcIntTy.getWidth() == 1) {
      Value zero = makeEmitCIntConstant(rewriter, loc, dstTy, 0);
      Value asInt = emitCCast(rewriter, loc, dstTy, adaptor.getIn());
      Value neg = rewriter.create<emitc::SubOp>(loc, dstTy, zero, asInt).getResult();
      rewriter.replaceOp(op, neg);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

template <typename CastOp>
struct ArithCastToEmitC : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

struct ArithIndexCastUIToEmitC : public OpConversionPattern<arith::IndexCastUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::IndexCastUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // MemRef casts are handled elsewhere; for safety, fall back to emitc.cast.
    if (isa<MemRefType>(op.getIn().getType()) || isa<MemRefType>(op.getType())) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto getBW = [](Type t) -> std::optional<unsigned> {
      if (auto i = dyn_cast<IntegerType>(t))
        return i.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };

    auto srcBW = getBW(op.getIn().getType());
    auto dstBW = getBW(op.getType());
    if (!srcBW || !dstBW)
      return rewriter.notifyMatchFailure(op, "unsupported index_castui types");

    if (*dstBW <= *srcBW) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    auto uSrcTy = getUnsignedIntOpaqueType(rewriter.getContext(), *srcBW);
    auto uDstTy = getUnsignedIntOpaqueType(rewriter.getContext(), *dstBW);
    Value srcU = emitCCast(rewriter, loc, uSrcTy, adaptor.getIn());
    Value extU = emitCCast(rewriter, loc, uDstTy, srcU);
    Value result = emitCCast(rewriter, loc, dstTy, extU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithUIToFPToEmitC : public OpConversionPattern<arith::UIToFPOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::UIToFPOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer input");

    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // Convert via an unsigned integer type of the same width.
    if (srcIntTy.getWidth() == 1) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }
    auto uSrcTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
    Value srcU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                           srcIntTy.getWidth());
    Value fp = rewriter.create<emitc::CastOp>(loc, dstTy, srcU).getResult();
    rewriter.replaceOp(op, fp);
    return success();
  }
};

struct ArithFPToUIToEmitC : public OpConversionPattern<arith::FPToUIOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::FPToUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    if (!dstIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer result");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    auto uDstTy =
        getUnsignedIntOpaqueType(rewriter.getContext(), dstIntTy.getWidth());
    Value asU = rewriter.create<emitc::CastOp>(loc, uDstTy, adaptor.getIn()).getResult();
    Value result = emitCCast(rewriter, loc, dstTy, asU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithBitcastToEmitC : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    // For pointer-like types, a regular cast is fine.
    if (isa<emitc::PointerType>(dstTy)) {
      rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
      return success();
    }

    // Only support scalar int/float/index bitcasts here.
    auto srcTy = op.getIn().getType();
    auto dstOrigTy = op.getType();

    auto getBitWidth = [](Type t) -> std::optional<unsigned> {
      if (auto it = dyn_cast<IntegerType>(t))
        return it.getWidth();
      if (auto ft = dyn_cast<FloatType>(t))
        return ft.getWidth();
      if (isa<IndexType>(t))
        return kPTOIndexBitWidth;
      return std::nullopt;
    };
    auto srcBW = getBitWidth(srcTy);
    auto dstBW = getBitWidth(dstOrigTy);
    if (!srcBW || !dstBW || *srcBW != *dstBW)
      return rewriter.notifyMatchFailure(op, "bitcast requires equal bitwidth");

    // Determine the template argument from the destination type string.
    auto dstOpaque = dyn_cast<emitc::OpaqueType>(dstTy);
    if (!dstOpaque)
      return rewriter.notifyMatchFailure(op, "expected emitc opaque dest type");

    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      dstOpaque.getValue())});
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{dstTy}, "ptoas_bitcast", /*operands=*/ValueRange{adaptor.getIn()},
        /*args=*/ArrayAttr{}, /*template_args=*/templateArgs);
    rewriter.replaceOp(op, call.getResult(0));
    return success();
  }
};

// arith.cmpf lowering with ordered/unordered semantics.
struct ArithCmpFToEmitC : public OpConversionPattern<arith::CmpFOp> {
  using OpConversionPattern::OpConversionPattern;

  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value isNotNaN(ConversionPatternRewriter &rewriter, Location loc,
                        Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::eq,
                              v, v)
        .getResult();
  }

  LogicalResult matchAndRewrite(arith::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getLhs().getType()))
      return rewriter.notifyMatchFailure(op, "cmpf only supported on scalar floats");

    auto loc = op.getLoc();
    auto i1Ty = rewriter.getI1Type();

    bool unordered = false;
    emitc::CmpPredicate pred = emitc::CmpPredicate::eq;

    switch (op.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse: {
      auto cst = makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "false");
      rewriter.replaceOp(op, cst);
      return success();
    }
    case arith::CmpFPredicate::AlwaysTrue: {
      auto cst = makeEmitCOpaqueConstant(rewriter, loc, i1Ty, "true");
      rewriter.replaceOp(op, cst);
      return success();
    }
    case arith::CmpFPredicate::OEQ:
      unordered = false;
      pred = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::OGT:
      unordered = false;
      pred = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::OGE:
      unordered = false;
      pred = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::OLT:
      unordered = false;
      pred = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::OLE:
      unordered = false;
      pred = emitc::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::ONE:
      unordered = false;
      pred = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::ORD: {
      Value ordered = rewriter.create<emitc::LogicalAndOp>(
          loc, i1Ty, isNotNaN(rewriter, loc, adaptor.getLhs()),
          isNotNaN(rewriter, loc, adaptor.getRhs()));
      rewriter.replaceOp(op, ordered);
      return success();
    }
    case arith::CmpFPredicate::UEQ:
      unordered = true;
      pred = emitc::CmpPredicate::eq;
      break;
    case arith::CmpFPredicate::UGT:
      unordered = true;
      pred = emitc::CmpPredicate::gt;
      break;
    case arith::CmpFPredicate::UGE:
      unordered = true;
      pred = emitc::CmpPredicate::ge;
      break;
    case arith::CmpFPredicate::ULT:
      unordered = true;
      pred = emitc::CmpPredicate::lt;
      break;
    case arith::CmpFPredicate::ULE:
      unordered = true;
      pred = emitc::CmpPredicate::le;
      break;
    case arith::CmpFPredicate::UNE:
      unordered = true;
      pred = emitc::CmpPredicate::ne;
      break;
    case arith::CmpFPredicate::UNO: {
      Value unord = rewriter.create<emitc::LogicalOrOp>(
          loc, i1Ty, isNaN(rewriter, loc, adaptor.getLhs()),
          isNaN(rewriter, loc, adaptor.getRhs()));
      rewriter.replaceOp(op, unord);
      return success();
    }
    }

    Value cmp = rewriter
                    .create<emitc::CmpOp>(loc, i1Ty, pred, adaptor.getLhs(),
                                          adaptor.getRhs())
                    .getResult();

    Value unord = rewriter.create<emitc::LogicalOrOp>(
        loc, i1Ty, isNaN(rewriter, loc, adaptor.getLhs()),
        isNaN(rewriter, loc, adaptor.getRhs()));
    Value ord = rewriter.create<emitc::LogicalAndOp>(
        loc, i1Ty, isNotNaN(rewriter, loc, adaptor.getLhs()),
        isNotNaN(rewriter, loc, adaptor.getRhs()));

    if (unordered) {
      Value res =
          rewriter.create<emitc::LogicalOrOp>(loc, i1Ty, unord, cmp).getResult();
      rewriter.replaceOp(op, res);
      return success();
    }

    Value res =
        rewriter.create<emitc::LogicalAndOp>(loc, i1Ty, ord, cmp).getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithAddUIExtendedToEmitC
    : public OpConversionPattern<arith::AddUIExtendedOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::AddUIExtendedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getSum().getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type sumDstTy = newResultTypes[0];
    Type overflowDstTy = newResultTypes[1];

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    auto wideTy = getWiderUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);

    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
    Value rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    Value sumWide =
        rewriter.create<emitc::AddOp>(loc, wideTy, lhsWide, rhsWide).getResult();

    Value sumN = emitCCast(rewriter, loc, uTy, sumWide);
    Value sum = emitCCast(rewriter, loc, sumDstTy, sumN);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value high = rewriter
                     .create<emitc::BitwiseRightShiftOp>(loc, wideTy, sumWide,
                                                         shiftAmt)
                     .getResult();
    Value zeroWide = makeEmitCIntConstant(rewriter, loc, wideTy, 0);
    Value overflow =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, high, zeroWide)
            .getResult();
    overflow = emitCCast(rewriter, loc, overflowDstTy, overflow);

    rewriter.replaceOp(op, {sum, overflow});
    return success();
  }
};

template <typename ArithOp, bool isUnsigned>
struct ArithMulExtendedToEmitC : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getResult(0).getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op,
                                         "expected scalar integer or index operands");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    SmallVector<Type> newResultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      newResultTypes)))
      return failure();
    if (newResultTypes.size() != 2)
      return failure();

    Type lowDstTy = newResultTypes[0];
    Type highDstTy = newResultTypes[1];

    Type wideTy = isUnsigned ? (Type)getWiderUnsignedIntOpaqueType(rewriter.getContext(),
                                                                   bitWidth)
                             : (Type)getWiderSignedIntOpaqueType(rewriter.getContext(),
                                                                 bitWidth);

    Value lhsWide;
    Value rhsWide;
    if constexpr (isUnsigned) {
      Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                      bitWidth);
      Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                      bitWidth);
      lhsWide = emitCCast(rewriter, loc, wideTy, lhsU);
      rhsWide = emitCCast(rewriter, loc, wideTy, rhsU);
    } else {
      lhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getLhs());
      rhsWide = emitCCast(rewriter, loc, wideTy, adaptor.getRhs());
    }

    Value prodWide =
        rewriter.create<emitc::MulOp>(loc, wideTy, lhsWide, rhsWide).getResult();
    Value low = emitCCast(rewriter, loc, lowDstTy, prodWide);

    Value shiftAmt = makeEmitCIntConstant(rewriter, loc, wideTy, bitWidth);
    Value highWide = rewriter
                         .create<emitc::BitwiseRightShiftOp>(loc, wideTy, prodWide,
                                                             shiftAmt)
                         .getResult();
    Value high = emitCCast(rewriter, loc, highDstTy, highWide);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

using ArithMulSIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulSIExtendedOp, /*isUnsigned=*/false>;
using ArithMulUIExtendedToEmitC =
    ArithMulExtendedToEmitC<arith::MulUIExtendedOp, /*isUnsigned=*/true>;

struct ArithMinMaxIToEmitCBase {
  static Value makeSelect(ConversionPatternRewriter &rewriter, Location loc,
                          Type dstTy, Value cond, Value trueV, Value falseV) {
    return rewriter
        .create<emitc::ConditionalOp>(loc, dstTy, cond, trueV, falseV)
        .getResult();
  }
};

struct ArithMaxSIToEmitC : public OpConversionPattern<arith::MaxSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinSIToEmitC : public OpConversionPattern<arith::MinSIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt,
                                           adaptor.getLhs(), adaptor.getRhs())
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMaxUIToEmitC : public OpConversionPattern<arith::MaxUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getRhs(),
                           adaptor.getLhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinUIToEmitC : public OpConversionPattern<arith::MinUIOp>,
                           ArithMinMaxIToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinUIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    Value lhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                           bitWidth);
    Value rhsU =
        castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                           bitWidth);
    Value cond = rewriter
                     .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                           emitc::CmpPredicate::lt, lhsU, rhsU)
                     .getResult();
    Value res = makeSelect(rewriter, loc, dstTy, cond, adaptor.getLhs(),
                           adaptor.getRhs());
    rewriter.replaceOp(op, res);
    return success();
  }
};

// Floating-point max/min variants.
struct ArithFloatMinMaxToEmitCBase {
  static Value isNaN(ConversionPatternRewriter &rewriter, Location loc,
                     Value v) {
    return rewriter
        .create<emitc::CmpOp>(loc, rewriter.getI1Type(), emitc::CmpPredicate::ne,
                              v, v)
        .getResult();
  }

  static Value makeFZero(ConversionPatternRewriter &rewriter, Location loc,
                         Type ty) {
    return makeEmitCOpaqueConstant(rewriter, loc, ty, "0.0f");
  }
};

struct ArithMaxNumFToEmitC : public OpConversionPattern<arith::MaxNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MaxNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value maxNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getRhs(),
                                          adaptor.getLhs())
            .getResult();

    Value rhsOrMax =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          maxNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMax)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ArithMinNumFToEmitC : public OpConversionPattern<arith::MinNumFOp>,
                             ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::MinNumFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Type dstTy = getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value minNoNaN =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, cmpLt, adaptor.getLhs(),
                                          adaptor.getRhs())
            .getResult();

    Value rhsOrMin =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN, adaptor.getLhs(),
                                          minNoNaN)
            .getResult();
    Value res =
        rewriter
            .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN, adaptor.getRhs(),
                                          rhsOrMin)
            .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

template <typename ArithOp, bool isMaximum>
struct ArithMinMaxFPropagateNaNToEmitC : public OpConversionPattern<ArithOp>,
                                        ArithFloatMinMaxToEmitCBase {
  using OpConversionPattern<ArithOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArithOp op, typename ArithOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<FloatType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "expected scalar float type");

    auto loc = op.getLoc();
    Type dstTy = this->getTypeConverter()->convertType(op.getType());
    if (!dstTy)
      return failure();

    Value lhsNaN = isNaN(rewriter, loc, adaptor.getLhs());
    Value rhsNaN = isNaN(rewriter, loc, adaptor.getRhs());

    // Basic compare-based min/max.
    Value cmpLt = rewriter
                      .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                            emitc::CmpPredicate::lt,
                                            adaptor.getLhs(), adaptor.getRhs())
                      .getResult();
    Value candidate = rewriter
                          .create<emitc::ConditionalOp>(
                              loc, dstTy, cmpLt,
                              isMaximum ? adaptor.getRhs() : adaptor.getLhs(),
                              isMaximum ? adaptor.getLhs() : adaptor.getRhs())
                          .getResult();

    // Fix signed zero tie-breaking for equal zeros.
    Value zero = makeFZero(rewriter, loc, dstTy);
    Value eq = rewriter
                   .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                         emitc::CmpPredicate::eq,
                                         adaptor.getLhs(), adaptor.getRhs())
                   .getResult();
    Value lhsZero = rewriter
                        .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                              emitc::CmpPredicate::eq,
                                              adaptor.getLhs(), zero)
                        .getResult();
    Value bothZero = rewriter
                         .create<emitc::LogicalAndOp>(loc, rewriter.getI1Type(),
                                                      eq, lhsZero)
                         .getResult();

    auto floatTy = cast<FloatType>(op.getType());
    auto bitsTy = getUnsignedIntOpaqueType(rewriter.getContext(), floatTy.getWidth());
    auto templateArgs =
        rewriter.getArrayAttr({emitc::OpaqueAttr::get(rewriter.getContext(),
                                                      cast<emitc::OpaqueType>(bitsTy).getValue())});
    Value lhsBits =
        rewriter
            .create<emitc::CallOpaqueOp>(loc, TypeRange{bitsTy}, "ptoas_bitcast",
                                         ValueRange{adaptor.getLhs()},
                                         /*args=*/ArrayAttr{},
                                         /*template_args=*/templateArgs)
            .getResult(0);

    Value oneBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 1);
    Value shAmt = makeEmitCIntConstant(rewriter, loc, bitsTy,
                                       floatTy.getWidth() - 1);
    Value signMask = rewriter
                         .create<emitc::BitwiseLeftShiftOp>(loc, bitsTy, oneBits,
                                                            shAmt)
                         .getResult();
    Value signBit = rewriter
                        .create<emitc::BitwiseAndOp>(loc, bitsTy, lhsBits, signMask)
                        .getResult();
    Value zeroBits = makeEmitCIntConstant(rewriter, loc, bitsTy, 0);
    Value lhsIsNegZero =
        rewriter
            .create<emitc::CmpOp>(loc, rewriter.getI1Type(),
                                  emitc::CmpPredicate::ne, signBit, zeroBits)
            .getResult();

    Value tie =
        rewriter
            .create<emitc::ConditionalOp>(
                loc, dstTy, lhsIsNegZero,
                isMaximum ? adaptor.getRhs() : adaptor.getLhs(),
                isMaximum ? adaptor.getLhs() : adaptor.getRhs())
            .getResult();
    Value noNaN = rewriter
                      .create<emitc::ConditionalOp>(loc, dstTy, bothZero, tie,
                                                    candidate)
                      .getResult();

    // Propagate NaN: if lhs is NaN return lhs, else if rhs is NaN return rhs.
    Value rhsOrNoNaN = rewriter
                           .create<emitc::ConditionalOp>(loc, dstTy, rhsNaN,
                                                         adaptor.getRhs(), noNaN)
                           .getResult();
    Value res = rewriter
                    .create<emitc::ConditionalOp>(loc, dstTy, lhsNaN,
                                                  adaptor.getLhs(), rhsOrNoNaN)
                    .getResult();
    rewriter.replaceOp(op, res);
    return success();
  }
};

using ArithMaximumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MaximumFOp, /*isMaximum=*/true>;
using ArithMinimumFToEmitC =
    ArithMinMaxFPropagateNaNToEmitC<arith::MinimumFOp, /*isMaximum=*/false>;

//===----------------------------------------------------------------------===//
// Arith -> EmitC helpers
//===----------------------------------------------------------------------===//

static emitc::OpaqueType getSignedIntOpaqueType(MLIRContext *ctx,
                                                unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "int8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "int16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "int32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "int64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "__int128");
  default:
    llvm::errs() << "[Debug] Unsupported signed integer bitwidth: " << bitWidth
                 << "\n";
    return emitc::OpaqueType::get(ctx, "int64_t");
  }
}

static emitc::OpaqueType getUnsignedIntOpaqueType(MLIRContext *ctx,
                                                  unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 8:
    return emitc::OpaqueType::get(ctx, "uint8_t");
  case 16:
    return emitc::OpaqueType::get(ctx, "uint16_t");
  case 32:
    return emitc::OpaqueType::get(ctx, "uint32_t");
  case 64:
    return emitc::OpaqueType::get(ctx, "uint64_t");
  case 128:
    return emitc::OpaqueType::get(ctx, "unsigned __int128");
  default:
    llvm::errs() << "[Debug] Unsupported unsigned integer bitwidth: "
                 << bitWidth << "\n";
    return emitc::OpaqueType::get(ctx, "uint64_t");
  }
}

static emitc::OpaqueType getWiderSignedIntOpaqueType(MLIRContext *ctx,
                                                     unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getSignedIntOpaqueType(ctx, 16);
  case 16:
    return getSignedIntOpaqueType(ctx, 32);
  case 32:
    return getSignedIntOpaqueType(ctx, 64);
  case 64:
    return getSignedIntOpaqueType(ctx, 128);
  default:
    return getSignedIntOpaqueType(ctx, 128);
  }
}

static emitc::OpaqueType getWiderUnsignedIntOpaqueType(MLIRContext *ctx,
                                                       unsigned bitWidth) {
  switch (bitWidth) {
  case 1:
  case 8:
    return getUnsignedIntOpaqueType(ctx, 16);
  case 16:
    return getUnsignedIntOpaqueType(ctx, 32);
  case 32:
    return getUnsignedIntOpaqueType(ctx, 64);
  case 64:
    return getUnsignedIntOpaqueType(ctx, 128);
  default:
    return getUnsignedIntOpaqueType(ctx, 128);
  }
}

static Value makeEmitCOpaqueConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, Type type,
                                     llvm::StringRef literal) {
  auto attr = emitc::OpaqueAttr::get(rewriter.getContext(), literal);
  return rewriter.create<emitc::ConstantOp>(loc, type, attr);
}

static Value makeEmitCIntConstant(ConversionPatternRewriter &rewriter,
                                  Location loc, Type type, int64_t value) {
  return makeEmitCOpaqueConstant(rewriter, loc, type, std::to_string(value));
}

static Value emitCCast(ConversionPatternRewriter &rewriter, Location loc,
                       Type dstType, Value src) {
  if (src.getType() == dstType)
    return src;
  return rewriter.createOrFold<emitc::CastOp>(loc, dstType, src);
}

// For signless iN integers lowered to signed C++ types, this creates a value
// representing the same N-bit pattern in an unsigned C++ type of the same
// width. This avoids incorrect sign-extension when later widening to a larger
// unsigned type.
static Value castSignlessIntToUnsignedSameWidth(ConversionPatternRewriter &rewriter,
                                                Location loc, Value v,
                                                unsigned bitWidth) {
  auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
  return emitCCast(rewriter, loc, uTy, v);
}

struct ArithMulIToEmitC : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 mul is equivalent to bitwise AND (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseAndOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value mulU = rewriter.create<emitc::MulOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, mulU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithAddIToEmitC : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 add is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value addU = rewriter.create<emitc::AddOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, addU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, newTy, adaptor.getIn());
    return success();
  }
};

struct ArithSubIToEmitC : public OpConversionPattern<arith::SubIOp> {
  using OpConversionPattern<arith::SubIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SubIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Type opTy = op.getType();
    auto intTy = dyn_cast<IntegerType>(opTy);
    const bool isIndex = isa<IndexType>(opTy);
    if (!intTy && !isIndex)
      return rewriter.notifyMatchFailure(op, "expected scalar integer or index type");

    const unsigned bitWidth =
        intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);

    Type dstTy = getTypeConverter()->convertType(opTy);
    if (!dstTy)
      return failure();

    // i1 sub is equivalent to XOR (mod 2 arithmetic).
    if (bitWidth == 1) {
      rewriter.replaceOpWithNewOp<emitc::BitwiseXorOp>(op, opTy, adaptor.getLhs(),
                                                      adaptor.getRhs());
      return success();
    }

    auto uTy = getUnsignedIntOpaqueType(rewriter.getContext(), bitWidth);
    Value lhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getLhs(),
                                                    bitWidth);
    Value rhsU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getRhs(),
                                                    bitWidth);
    Value subU = rewriter.create<emitc::SubOp>(loc, uTy, lhsU, rhsU);
    Value result = emitCCast(rewriter, loc, dstTy, subU);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ArithDivSIToEmitC : public OpConversionPattern<arith::DivSIOp> {
  using OpConversionPattern<arith::DivSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::DivOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithRemSIToEmitC : public OpConversionPattern<arith::RemSIOp> {
  using OpConversionPattern<arith::RemSIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::RemSIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy)
      return failure();
    rewriter.replaceOpWithNewOp<emitc::RemOp>(op, newTy, adaptor.getLhs(),
                                              adaptor.getRhs());
    return success();
  }
};

struct ArithTruncIToEmitC : public OpConversionPattern<arith::TruncIOp> {
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto dstIntTy = dyn_cast<IntegerType>(op.getType());
    auto srcIntTy = dyn_cast<IntegerType>(op.getIn().getType());
    if (!dstIntTy || !srcIntTy)
      return rewriter.notifyMatchFailure(op, "expected scalar integer types");

    Type dstTy = getTypeConverter()->convertType(dstIntTy);
    if (!dstTy)
      return failure();

    // to-i1 conversions: Arith wants truncation to the low bit, while C/C++
    // casts to bool are equivalent to `v != 0`. Implement as `(bool)(v & 1)`.
    if (dstIntTy.getWidth() == 1) {
      if (srcIntTy.getWidth() == 1) {
        rewriter.replaceOp(op, adaptor.getIn());
        return success();
      }

      auto uSrcTy =
          getUnsignedIntOpaqueType(rewriter.getContext(), srcIntTy.getWidth());
      Value inU = castSignlessIntToUnsignedSameWidth(rewriter, loc, adaptor.getIn(),
                                                     srcIntTy.getWidth());
      Value one = makeEmitCIntConstant(rewriter, loc, uSrcTy, 1);
      Value masked =
          rewriter.create<emitc::BitwiseAndOp>(loc, uSrcTy, inU, one);
      Value asBool = emitCCast(rewriter, loc, dstTy, masked);
      rewriter.replaceOp(op, asBool);
      return success();
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, dstTy, adaptor.getIn());
    return success();
  }
};

		struct ArithConstantToEmitC : public OpConversionPattern<arith::ConstantOp> {
		  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
		
		  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
		                                ConversionPatternRewriter &rewriter) const override {
	    Type newType = getTypeConverter()->convertType(op.getType());
	    if (!newType) return failure();
	
	    // `adaptor.getValue()` may be null if attribute conversion isn't defined.
	    // Use the original attribute as fallback and always cast null-safely.
	    Attribute valueAttr = adaptor.getValue();
	    if (!valueAttr) valueAttr = op.getValue();

		    if (auto floatAttr = dyn_cast_or_null<FloatAttr>(valueAttr)) {
		      SmallString<32> valStr;
		      floatAttr.getValue().toString(valStr);
		      llvm::StringRef s(valStr);
		      // Ensure the literal parses as a floating-point constant in C/C++.
		      // `APFloat::toString` may emit "1" for integral values; make it "1.0".
		      const bool hasFloatMarker =
		          s.contains('.') || s.contains('e') || s.contains('E') ||
		          s.contains('p') || s.contains('P') || s.starts_with("0x") ||
		          s.starts_with("0X") || s.starts_with("nan") ||
		          s.starts_with("-nan") || s.starts_with("inf") ||
		          s.starts_with("-inf");
		      if (!hasFloatMarker)
		        valStr.append(".0");
		      // Suffix: keep `f` for f16/f32; omit for f64.
		      if (!floatAttr.getType().isF64())
		        valStr.append("f");
		      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
		      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
		      return success();
		    }
	
	    if (auto intAttr = dyn_cast_or_null<IntegerAttr>(valueAttr)) {
	      std::string valStr = std::to_string(intAttr.getValue().getSExtValue());
	      auto constAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
	      rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, newType, constAttr);
	      return success();
	    }
	
	    return failure();
	  }
	};
//===----------------------------------------------------------------------===//
// pto.mgather lowering -> MGATHER(dst, mem, idx)
// %dst = pto.mgather %mem, %idx : memref<...>, memref<...> -> memref<...>
//===----------------------------------------------------------------------===//

struct PTOMGatherToMGATHER : public OpConversionPattern<pto::MGatherDpsOp> {
  using OpConversionPattern<pto::MGatherDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MGatherDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value mem = peelUnrealized(adaptor.getMem());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa currently has no NPU implementation for MGATHER/MSCATTER.
    // Fallback to a smoke-friendly lowering to keep compile/run coverage.
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, mem});

     if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, dst);
    }
    return success();
  }
};

struct AffineApplyMulConstToEmitC
    : public OpConversionPattern<affine::AffineApplyOp> {
  using OpConversionPattern<affine::AffineApplyOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto map = op.getAffineMap();

    if (map.getNumDims() != 0 || map.getNumSymbols() != 1)
      return failure();

    auto expr = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin || bin.getKind() != AffineExprKind::Mul)
      return failure();

    auto lhs = bin.getLHS();
    auto rhs = bin.getRHS();

    auto symExpr = dyn_cast<AffineSymbolExpr>(lhs);
    auto constExpr = dyn_cast<AffineConstantExpr>(rhs);
    if (!symExpr || !constExpr)
      return failure();

    Value inputVal = adaptor.getMapOperands()[0];

    std::string valStr = std::to_string(constExpr.getValue());
    auto cstAttr = emitc::OpaqueAttr::get(rewriter.getContext(), valStr);
    auto cstOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), inputVal.getType(), cstAttr);

    rewriter.replaceOpWithNewOp<emitc::MulOp>(
        op, inputVal.getType(), inputVal, cstOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel inference helpers
//===----------------------------------------------------------------------===//

enum class KernelKind { VecAdd, Matmul, Unknown };

static KernelKind inferKernelKind(func::FuncOp f) {
  bool hasAdd = false;
  bool hasMM  = false;
  f.walk([&](Operation *op) {
    if (isa<mlir::pto::AddFDpsOp>(op)) hasAdd = true;
    if (isa<mlir::pto::MatmulDpsOp>(op)) hasMM = true;
    if (isa<mlir::pto::MatmulAccDpsOp>(op)) hasMM = true;
  });
  if (hasMM)  return KernelKind::Matmul;
  if (hasAdd) return KernelKind::VecAdd;
  return KernelKind::Unknown;
}

static void inferTileMNK(func::FuncOp f, int &M, int &N, int &K) {
  M = 32; N = 32; K = 32;
  SmallVector<memref::SubViewOp, 4> subs;
  f.walk([&](memref::SubViewOp sv) { subs.push_back(sv); });

  auto readShape2D = [&](memref::SubViewOp sv, int &d0, int &d1) {
    auto resTy = mlir::cast<MemRefType>(sv.getResult().getType());
    if (resTy.getRank() == 2 && resTy.hasStaticShape()) {
      d0 = (int)resTy.getDimSize(0);
      d1 = (int)resTy.getDimSize(1);
    }
  };

  if (subs.empty()) return;

  int a0=32, a1=32;
  readShape2D(subs[0], a0, a1);
  M = a0; N = a1;

  if (subs.size() >= 2) {
    int b0=32, b1=32;
    readShape2D(subs[0], a0, a1);
    readShape2D(subs[1], b0, b1);
    M = a0; K = a1; N = b1;
  }
}

struct FuncToEmitC : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert the function signature with the type converter.
    Type convertedTy = getTypeConverter()->convertType(op.getFunctionType());
    auto funcType = dyn_cast_or_null<FunctionType>(convertedTy);
    if (!funcType)
      return rewriter.notifyMatchFailure(op, "failed to convert function type");
    if (funcType.getNumResults() > 1)
      return rewriter.notifyMatchFailure(
          op, "EmitC cannot return multiple values");

    // Create the EmitC function with the converted signature.
    auto emitcFunc = rewriter.create<emitc::FuncOp>(op.getLoc(), op.getName(),
                                                    funcType);
    emitcFunc.setSpecifiersAttr(
        rewriter.getStrArrayAttr({"__global__ AICORE"}));

    // Inline the original body, then convert region/block argument types to
    // match the converted signature (also covers CFG blocks introduced by
    // pre-lowering, e.g. scf.while -> cf.br/cf.cond_br).
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(),
                                emitcFunc.end());

    TypeConverter::SignatureConversion entryConv(op.getNumArguments());
    for (unsigned i = 0; i < op.getNumArguments(); ++i)
      entryConv.addInputs(i, funcType.getInput(i));

    if (failed(rewriter.convertRegionTypes(&emitcFunc.getBody(),
                                          *getTypeConverter(), &entryConv)))
      return failure();

    // [Compatibility patch] Preserve existing snippets that rely on `T`.
    {
      Block &entryBlock = emitcFunc.getBody().front();
      rewriter.setInsertionPointToStart(&entryBlock);
      rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SubView lowering to GlobalTensor (keep your existing code)
//===----------------------------------------------------------------------===

enum class Role { A, B, C, Unknown };

static Role inferSubviewRole(memref::SubViewOp sv) {
  for (Operation *u : sv.getResult().getUsers()) {
    if (auto ld = dyn_cast<mlir::pto::LoadDpsOp>(u)) {
      Value ub = ld.getDst();
      if (!ub) continue;
      for (Operation *uu : ub.getUsers()) {
        if (auto mm = dyn_cast<mlir::pto::MatmulDpsOp>(uu)) {
          if (mm.getLhs() == ub) return Role::A;
          if (mm.getRhs() == ub) return Role::B;
        }
        if (auto mmacc = dyn_cast<mlir::pto::MatmulAccDpsOp>(uu)) {
          if (mmacc.getLhs() == ub) return Role::A;
          if (mmacc.getRhs() == ub) return Role::B;
        }
      }
    }

    if (auto st = dyn_cast<mlir::pto::StoreDpsOp>(u)) {
      if (st.getDst() == sv.getResult()) return Role::C;
    }
  }
  return Role::Unknown;
}

// =============================================================================
// 4. MemRef SubView -> Explicit Shape/Stride Construction (Full Implementation)
// =============================================================================
struct SubviewToEmitCPattern : public OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  // 辅助函数：尝试从 OpFoldResult 中提取静态整数值
  std::optional<int64_t> extractStaticInt(OpFoldResult ofr) const {
    if (auto attr = ofr.dyn_cast<Attribute>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    } else {
      Value v = ofr.get<Value>();
      if (auto cOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto iAttr = dyn_cast<IntegerAttr>(cOp.getValue()))
          return iAttr.getInt();
      } else if (auto idxOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
        return idxOp.value();
      }
    }
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    
    // 获取源 MemRef 类型信息
    auto srcType = mlir::cast<MemRefType>(op.getSource().getType());
    int64_t rank = srcType.getRank();

    auto elemTypeToString = [&](Type elemTy) -> std::string {
      if (elemTy.isF16())
        return "half";
      if (elemTy.isF32())
        return "float";
      if (elemTy.isF64())
        return "double";
      if (elemTy.isInteger(8)) {
        if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
          return "int8_t";
        return "uint8_t";
      }
      if (elemTy.isInteger(16)) {
        if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
          return "int16_t";
        return "uint16_t";
      }
      if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
          return "int32_t";
        return "uint32_t";
      }
      if (elemTy.isInteger(64)) {
        return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
      }
      return "float";
    };

    // -------------------------------------------------------------------------
    // Part 1: 指针偏移计算 (Runtime Pointer Arithmetic)
    // -------------------------------------------------------------------------
    
    // 准备类型: unsigned
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    
    // Helper: 创建 unsigned 常量
    auto mkU32 = [&](int64_t v) -> Value {
      return rewriter.create<emitc::ConstantOp>(
          loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(v)));
    };

    // Helper: 将 OpFoldResult 转为 EmitC Value (用于计算)
    auto ofrToEmitCValue = [&](OpFoldResult ofr) -> Value {
      if (auto v = ofr.dyn_cast<Value>()) {
        Value rv = rewriter.getRemappedValue(v);
        // 如果类型不匹配，插入 Cast
        if (rv.getType() != u32Ty)
             return rewriter.create<emitc::CastOp>(loc, u32Ty, rv).getResult();
        return rv;
      }
      if (auto attr = ofr.dyn_cast<Attribute>()) {
         if (auto ia = dyn_cast<IntegerAttr>(attr))
             return mkU32(ia.getValue().getSExtValue());
      }
      return mkU32(0);
    };

    // 1. 获取 Source 的 Strides (支持动态 Stride 收集)
    SmallVector<OpFoldResult> sourceStrides;

    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        sourceStrides = rc.getMixedStrides();
    } else {
        SmallVector<int64_t> strideInts;
        int64_t offset = ShapedType::kDynamic;
        bool useTypeStrides = succeeded(getStridesAndOffset(srcType, strideInts, offset));
        (void)offset;
        if (useTypeStrides) {
            for (int64_t s : strideInts) {
                if (s == ShapedType::kDynamic) {
                    useTypeStrides = false;
                    break;
                }
            }
        }
        if (useTypeStrides) {
            for (int64_t s : strideInts) {
                sourceStrides.push_back(rewriter.getIndexAttr(s));
            }
        } else {
            // Fallback: Compact Layout
            auto shape = srcType.getShape();
            int64_t current = 1;
            sourceStrides.resize(rank);
            for (int i = rank - 1; i >= 0; --i) {
                sourceStrides[i] = rewriter.getIndexAttr(current);
                if (shape[i] != ShapedType::kDynamic) current *= shape[i];
            }
        }
    }

    // 2. 计算运行时 Offset
    auto staticOffsets = op.getStaticOffsets();
    auto dynamicOffsets = adaptor.getOffsets();
    int dynOffIdx = 0;
    Value totalOffset = mkU32(0);

    for (int i = 0; i < rank; ++i) {
        // A. 获取 Offset
        Value offVal;
        if (staticOffsets[i] == ShapedType::kDynamic) {
            Value rawDyn = dynamicOffsets[dynOffIdx++];
            offVal = rewriter.create<emitc::CastOp>(loc, u32Ty, rawDyn);
        } else {
            offVal = mkU32(staticOffsets[i]);
        }

        // B. 获取 Stride (用于指针计算)
        Value strideVal = mkU32(1);
        if (i < (int)sourceStrides.size()) {
            strideVal = ofrToEmitCValue(sourceStrides[i]);
        }

        // C. 累加
        Value term = rewriter.create<emitc::MulOp>(loc, u32Ty, offVal, strideVal);
        totalOffset = rewriter.create<emitc::AddOp>(loc, u32Ty, totalOffset, term);
    }

    // 3. 生成新指针
    //
    // NOTE: Some toolchains may materialize kernel pointer params as `void*` even
    // when the underlying element type is i16. Pointer arithmetic on `void*`
    // is ill-formed in C++, so we explicitly cast to a typed pointer for i16.
    Value sourcePtr = adaptor.getSource();
    Value tileCandidate = sourcePtr;
    if (auto castOp = sourcePtr.getDefiningOp<emitc::CastOp>()) {
      tileCandidate = castOp.getOperand();
    } else if (auto uc =
                   sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      tileCandidate = uc.getOperand(0);
    }
    if (auto ot = dyn_cast<emitc::OpaqueType>(tileCandidate.getType())) {
      auto tyStr = ot.getValue();
      if (tyStr.find("Tile<") != std::string::npos ||
          tyStr.find("ConvTile<") != std::string::npos) {
        std::string elemTok = elemTypeToString(srcType.getElementType());
        std::string qualifier = "__gm__";
        if (auto asAttr =
                dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace()))
          qualifier = addrSpaceQualifier(asAttr.getAddressSpace());
        auto rawPtrTy =
            emitc::OpaqueType::get(ctx, qualifier + " " + elemTok + "*");
        sourcePtr =
            rewriter
                .create<emitc::CallOpaqueOp>(loc, rawPtrTy,
                                             "PTOAS__TILE_DATA", ArrayAttr{},
                                             ArrayAttr{}, ValueRange{tileCandidate})
                .getResult(0);
      }
    }
    Value newPtr;
    {
      auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
      Type elemTy = resTy.getElementType();
      if (elemTy.isInteger(16)) {
        std::string castElemTypeStr = "int16_t";
        if (cast<IntegerType>(elemTy).isUnsigned())
          castElemTypeStr = "uint16_t";

        std::string qualifier = "__gm__";
        if (Attribute ms = srcType.getMemorySpace()) {
          if (auto ptoAttr = dyn_cast<pto::AddressSpaceAttr>(ms)) {
            qualifier = addrSpaceQualifier(ptoAttr.getAddressSpace());
          }
        }

        auto typedPtrTy = emitc::OpaqueType::get(ctx, qualifier + " " + castElemTypeStr + "*");
        Value typedSourcePtr = rewriter.create<emitc::CastOp>(loc, typedPtrTy, sourcePtr);
        newPtr = rewriter.create<emitc::AddOp>(loc, typedPtrTy, typedSourcePtr, totalOffset);
      } else {
        newPtr = rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr, totalOffset);
      }
    }


    // -------------------------------------------------------------------------
    // Part 2: For non-GM memrefs, keep pointer (no GlobalTensor).
    // -------------------------------------------------------------------------
    bool isGlobal = true;
    if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcType.getMemorySpace())) {
      auto as = asAttr.getAddressSpace();
      isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
    }
    if (!isGlobal) {
      Type dstTy = getTypeConverter()->convertType(op.getType());
      if (!dstTy)
        return failure();
      if (newPtr.getType() != dstTy)
        newPtr = rewriter.create<emitc::CastOp>(loc, dstTy, newPtr);
      rewriter.replaceOp(op, newPtr);
      return success();
    }

    // -------------------------------------------------------------------------
    // Part 3: 生成 GlobalTensor 类型 (Shape/Stride Template Generation)
    // -------------------------------------------------------------------------
    
    // When emitting C++ with `declareVariablesAtTop`, value declarations are
    // hoisted before body statements. Avoid introducing local `using` aliases
    // for templated types (Shape/Stride/GlobalTensor) because those aliases
    // would appear after the hoisted declarations and break compilation
    // (`unknown type name`).
    //
    // Instead, use the fully spelled template types as EmitC opaque types.

    auto resTy = mlir::cast<MemRefType>(op.getResult().getType());
    
    // 1. 解析具体元素类型 (完整逻辑，不省略)
    std::string elemTypeStr = "float"; 
    Type elemTy = resTy.getElementType();
    
    if (elemTy.isF16()) {
        elemTypeStr = "half";
    } else if (elemTy.isF32()) {
        elemTypeStr = "float";
    } else if (elemTy.isInteger(8)) {
        // 区分有符号/无符号通常依赖上下文，但在 EmitC 中 int8_t 比较通用
        if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
            elemTypeStr = "int8_t";
        else 
            elemTypeStr = "uint8_t";
    } else if (elemTy.isInteger(16)) {
        if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
            elemTypeStr = "int16_t";
        else
            elemTypeStr = "uint16_t";
    } else if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
            elemTypeStr = "int32_t";
        else 
            elemTypeStr = "uint32_t";
    } else if (elemTy.isInteger(64)) {
        elemTypeStr = cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
    }

    // 2. 生成 Shape 模板参数，之后会右对齐有效维度并补齐到 5 维（高维填 1）
    SmallVector<std::string> shapeParamsVec;
    SmallVector<Value> sizeValues; // 每个维度对应的运行时 size（统一为 unsigned）
    auto resShape = resTy.getShape();
    auto mixedSizes = op.getMixedSizes();
    sizeValues.reserve(rank);
    for (int i = 0; i < resTy.getRank(); ++i) {
      if (resShape[i] == ShapedType::kDynamic) {
        shapeParamsVec.push_back("-1");
      } else {
        shapeParamsVec.push_back(std::to_string(resShape[i]));
      }
      // size 值：优先从 op.getMixedSizes() 取（可动态/静态），否则退化为类型里的静态 shape。
      if (i < (int)mixedSizes.size())
        sizeValues.push_back(ofrToEmitCValue(mixedSizes[i]));
      else
        sizeValues.push_back(
            mkU32(resShape[i] == ShapedType::kDynamic ? 1 : resShape[i]));
    }

    // 3. 生成 Stride 模板参数 + 运行时 stride 值（考虑 subview step）
    SmallVector<std::string> dummyStrideVec;
    SmallVector<Value> strideValues; // 每个维度对应的运行时 stride（统一为 unsigned）
    dummyStrideVec.reserve(rank);
    strideValues.reserve(rank);
    auto subViewSteps = op.getMixedStrides();
    for (int i = 0; i < rank; ++i) {
      OpFoldResult srcStrideOfr =
          (i < (int)sourceStrides.size()) ? sourceStrides[i]
                                          : rewriter.getIndexAttr(1);
      OpFoldResult stepOfr = (i < (int)subViewSteps.size())
                                 ? subViewSteps[i]
                                 : rewriter.getIndexAttr(1);

      auto srcStatic = extractStaticInt(srcStrideOfr);
      auto stepStatic = extractStaticInt(stepOfr);
      if (srcStatic && stepStatic) {
        int64_t finalStride = (*srcStatic) * (*stepStatic);
        dummyStrideVec.push_back(std::to_string(finalStride));
        strideValues.push_back(mkU32(finalStride));
        continue;
      }

      dummyStrideVec.push_back("-1");
      Value srcV = ofrToEmitCValue(srcStrideOfr);
      Value stepV = ofrToEmitCValue(stepOfr);
      // 尽量避免乘以 1 生成冗余指令
      if (stepStatic && *stepStatic == 1)
        strideValues.push_back(srcV);
      else if (srcStatic && *srcStatic == 1)
        strideValues.push_back(stepV);
      else
        strideValues.push_back(
            rewriter.create<emitc::MulOp>(loc, u32Ty, srcV, stepV));
    }

    // 3.1 右对齐到 5 维：shape 补 1；已有维度继承原 stride；
    //      被补出来的高维按“紧密升维”规则连续推导：stride[i] = shape[i+1] * stride[i+1]
    SmallVector<std::string, 5> finalShape(5, "1");
    SmallVector<std::string, 5> finalStride(5, "1");
    Value oneU32 = mkU32(1);
    SmallVector<Value, 5> finalShapeValues(5, oneU32);
    SmallVector<Value, 5> finalStrideValues(5, oneU32);
    int shift = 5 - rank;

    // 先放入原始 shape/stride（保持用户提供的值）
    for (int i = 0; i < rank && i < 5; ++i) {
      finalShape[shift + i] = shapeParamsVec[i];
      finalStride[shift + i] = dummyStrideVec[i];
      finalShapeValues[shift + i] = sizeValues[i];
      finalStrideValues[shift + i] = strideValues[i];
    }

    auto mulOrDyn = [](const std::string &a, const std::string &b) -> std::string {
        if (a == "-1" || b == "-1")
            return "-1";
        int64_t va = 1, vb = 1;
        (void)llvm::to_integer(a, va);
        (void)llvm::to_integer(b, vb);
        return std::to_string(va * vb);
    };

    // 从低维到高维倒推补齐 stride（仅对补出来的前置维度生效）
    for (int i = 3; i >= 0; --i) {
      // 如果该维已由原始 rank 覆盖，则保持原值
      if (i >= shift)
        continue;
      // 补维：shape 已经是 1，stride = shape[i+1] * stride[i+1]（或动态）
      finalStride[i] = mulOrDyn(finalShape[i + 1], finalStride[i + 1]);
      if (finalStride[i] != "-1") {
        int64_t si = 1;
        (void)llvm::to_integer(finalStride[i], si);
        finalStrideValues[i] = mkU32(si);
        continue;
      }
      // 动态推导：stride[i] = shape[i+1] * stride[i+1]
      if (finalShape[i + 1] == "1") {
        finalStrideValues[i] = finalStrideValues[i + 1];
      } else {
        finalStrideValues[i] = rewriter.create<emitc::MulOp>(
            loc, u32Ty, finalShapeValues[i + 1], finalStrideValues[i + 1]);
      }
    }

    auto joinParams = [](llvm::ArrayRef<std::string> vec) {
        std::string out;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) out += ", ";
            out += vec[i];
        }
        return out;
    };

    std::string shapeParams = joinParams(finalShape);
    std::string strideParams = joinParams(finalStride);

    // Spelled-out C++ types.
    std::string shapeCppType = "pto::Shape<" + shapeParams + ">";
    std::string strideCppType = "pto::Stride<" + strideParams + ">";

    // 3.0 计算推导出的 Layout；若用户已指定 layout，保持一致性校验
    auto strToInt = [](const std::string &s, int64_t &out) -> bool {
        return s != "-1" && llvm::to_integer(s, out);
    };
    SmallVector<int64_t, 5> shapeInt(5, -1), strideInt(5, -1);
    bool allStatic = true;
    for (int i = 0; i < 5; ++i) {
        if (!strToInt(finalShape[i], shapeInt[i]) || !strToInt(finalStride[i], strideInt[i]))
            allStatic = false;
    }
    // 推导规则与 InferPTOLayout 保持一致
    int layoutTag = 0; // ND
    auto elemBytes = 4; // default float
    if (elemTypeStr.find("half") != std::string::npos || elemTypeStr.find("f16") != std::string::npos)
        elemBytes = 2;
    else if (elemTypeStr.find("double") != std::string::npos || elemTypeStr.find("f64") != std::string::npos)
        elemBytes = 8;
    if (allStatic) {
        // NZ 判定
        if (shapeInt[2] == 16 && shapeInt[2] * shapeInt[3] * elemBytes == 512 &&
            strideInt[4] == 1 && strideInt[3] == shapeInt[4]) {
            layoutTag = 2; // NZ
        } else {
            bool isRow = strideInt[4] == 1;
            for (int i = 3; i >= 0; --i)
                isRow &= (strideInt[i] == strideInt[i + 1] * shapeInt[i + 1]);
            bool isCol = strideInt[0] == 1;
            for (int i = 0; i < 4; ++i)
                isCol &= (strideInt[i + 1] == strideInt[i] * shapeInt[i]);
            if (isCol) layoutTag = 1; // DN
            else layoutTag = isRow ? 0 : 0; // fallback ND
        }
    }
    // 若源 IR 上存在 layout 属性，则校验是否一致，不一致时报错
    if (auto attr = op->getAttrOfType<mlir::pto::LayoutAttr>("layout")) {
        int userTag = 0;
        switch (attr.getLayout()) {
        case mlir::pto::Layout::ND: userTag = 0; break;
        case mlir::pto::Layout::DN: userTag = 1; break;
        case mlir::pto::Layout::NZ: userTag = 2; break;
        }
        if (userTag != layoutTag) {
            return rewriter.notifyMatchFailure(
                op, "layout mismatch: user-specified layout=" +
                        std::string(attr.getLayout() == mlir::pto::Layout::ND ? "ND" :
                                    attr.getLayout() == mlir::pto::Layout::DN ? "DN" : "NZ") +
                        " but inferred layout tag=" + std::to_string(layoutTag));
        }
        layoutTag = userTag; // align with user
    }
    // 生成枚举字符串
    std::string layoutEnum = "pto::Layout::ND";
    if (layoutTag == 1)
        layoutEnum = "pto::Layout::DN";
    else if (layoutTag == 2)
        layoutEnum = "pto::Layout::NZ";
    // GlobalTensor takes a Layout non-type template parameter; directly use the
    // enum constant.


    // -------------------------------------------------------------------------
    // Part 3: 显式对象实例化 (Explicit Object Instantiation)
    // -------------------------------------------------------------------------

    // A. Instantiate Shape object.
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeCppType);
    SmallVector<Value> shapeArgs;
    // 从 adaptor.getSizes() 获取 subview 的所有 dynamic sizes
    for (Value dynSize : adaptor.getSizes()) {
        shapeArgs.push_back(dynSize);
    }
    
    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        shapeTypeOpaque, // 返回类型
        shapeCppType,    // 调用的“函数名”即类名构造函数
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(shapeArgs)
    );
    
    // B. Instantiate Stride object.
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideCppType);
    // 仅传入动态 stride 维度对应的值，匹配 pto::Stride 的 N-parameter ctor（并满足其 static_assert）。
    SmallVector<Value> strideCtorArgs;
    strideCtorArgs.reserve(5);
    for (int i = 0; i < 5; ++i) {
      if (finalStride[i] == "-1")
        strideCtorArgs.push_back(finalStrideValues[i]);
    }
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, strideTypeOpaque, strideCppType,
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(strideCtorArgs));

    // C. Instantiate GlobalTensor object (ptr + shape + stride).
    std::string gtCppType = "GlobalTensor<" + elemTypeStr + ", " + shapeCppType +
                            ", " + strideCppType + ", " + layoutEnum + ">";
    auto gtType = emitc::OpaqueType::get(ctx, gtCppType);

    // 准备构造参数: [ptr, shape_instance, stride_instance]
    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0)); // 拿到 shape_inst 的 SSA Value
    gtConstructorArgs.push_back(strideInstOp.getResult(0)); // 拿到 stride_inst 的 SSA Value

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        gtType, 
        gtCppType,
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs)
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper: build GlobalTensor from a static MemRef (for TLOAD/TSTORE)
//===----------------------------------------------------------------------===//

static std::string getElemTypeStringForGT(Type elemTy) {
  if (elemTy.isF16()) return "half";
  if (elemTy.isF32()) return "float";
  if (elemTy.isF64()) return "double";
  if (elemTy.isInteger(8)) {
    if (elemTy.isSignlessInteger(8) || elemTy.isSignedInteger(8))
      return "int8_t";
    return "uint8_t";
  }
  if (elemTy.isInteger(16)) {
    if (elemTy.isSignlessInteger(16) || elemTy.isSignedInteger(16))
      return "int16_t";
    return "uint16_t";
  }
  if (elemTy.isInteger(32)) {
    if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
      return "int32_t";
    return "uint32_t";
  }
  if (elemTy.isInteger(64)) {
    return cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";
  }
  return "float";
}

static Value buildGlobalTensorFromMemref(ConversionPatternRewriter &rewriter,
                                         Location loc, Value basePtr,
                                         MemRefType mrTy,
                                         Operation *anchor) {
  auto *ctx = rewriter.getContext();

  // Only handle fully static shapes/strides for now.
  auto shape = mrTy.getShape();
  for (int64_t dim : shape) {
    if (dim == ShapedType::kDynamic)
      return Value();
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(mrTy, strides, offset))) {
    // Fallback: compact row-major
    strides.resize(shape.size());
    int64_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }
    offset = 0;
  }
  if (offset == ShapedType::kDynamic)
    return Value();
  for (int64_t s : strides) {
    if (s == ShapedType::kDynamic)
      return Value();
  }

  // Apply static base offset if needed.
  Value ptr = basePtr;
  if (offset != 0) {
    Type u32Ty = emitc::OpaqueType::get(ctx, "unsigned");
    auto offVal = rewriter.create<emitc::ConstantOp>(
        loc, u32Ty, emitc::OpaqueAttr::get(ctx, std::to_string(offset)));
    ptr = rewriter.create<emitc::AddOp>(loc, basePtr.getType(), basePtr,
                                        offVal);
  }

  std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(anchor));
  std::string shapeTypeName  = "GTShape"  + suffix;
  std::string strideTypeName = "GTStride" + suffix;
  std::string gtTypeName     = "GT"       + suffix;

  std::string elemTypeStr = getElemTypeStringForGT(mrTy.getElementType());

  SmallVector<std::string> shapeParamsVec;
  SmallVector<std::string> strideParamsVec;
  for (int i = 0, e = (int)shape.size(); i < e; ++i) {
    shapeParamsVec.push_back(std::to_string(shape[i]));
    strideParamsVec.push_back(std::to_string(strides[i]));
  }

  // Right-align to 5D (pad leading dims with 1).
  SmallVector<std::string, 5> finalShape(5, "1");
  SmallVector<std::string, 5> finalStride(5, "1");
  int rank = (int)shape.size();
  int shift = 5 - rank;
  for (int i = 0; i < rank && i < 5; ++i) {
    finalShape[shift + i] = shapeParamsVec[i];
    finalStride[shift + i] = strideParamsVec[i];
  }
  auto mulOrDyn = [](const std::string &a, const std::string &b) -> std::string {
    if (a == "-1" || b == "-1")
      return "-1";
    int64_t va = 1, vb = 1;
    (void)llvm::to_integer(a, va);
    (void)llvm::to_integer(b, vb);
    return std::to_string(va * vb);
  };
  for (int i = 3; i >= 0; --i) {
    if (i >= shift)
      continue;
    finalStride[i] = mulOrDyn(finalShape[i + 1], finalStride[i + 1]);
  }

  auto joinParams = [](llvm::ArrayRef<std::string> vec) {
    std::string out;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) out += ", ";
      out += vec[i];
    }
    return out;
  };

  std::string shapeParams = joinParams(finalShape);
  std::string strideParams = joinParams(finalStride);

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + shapeTypeName + " = pto::Shape<" + shapeParams + ">;");
  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + strideTypeName + " = pto::Stride<" + strideParams + ">;");

  // Infer layout (same rules as Subview lowering)
  SmallVector<int64_t, 5> shapeInt(5, -1), strideInt(5, -1);
  for (int i = 0; i < 5; ++i) {
    (void)llvm::to_integer(finalShape[i], shapeInt[i]);
    (void)llvm::to_integer(finalStride[i], strideInt[i]);
  }
  int layoutTag = 0; // ND
  int elemBytes = 4;
  if (elemTypeStr.find("half") != std::string::npos)
    elemBytes = 2;
  else if (elemTypeStr.find("double") != std::string::npos)
    elemBytes = 8;
  if (shapeInt[2] == 16 && shapeInt[2] * shapeInt[3] * elemBytes == 512 &&
      strideInt[4] == 1 && strideInt[3] == shapeInt[4]) {
    layoutTag = 2; // NZ
  } else {
    bool isRow = strideInt[4] == 1;
    for (int i = 3; i >= 0; --i)
      isRow &= (strideInt[i] == strideInt[i + 1] * shapeInt[i + 1]);
    bool isCol = strideInt[0] == 1;
    for (int i = 0; i < 4; ++i)
      isCol &= (strideInt[i + 1] == strideInt[i] * shapeInt[i]);
    if (isCol) layoutTag = 1; // DN
    else layoutTag = isRow ? 0 : 0; // fallback ND
  }
  std::string layoutEnum = "pto::Layout::ND";
  if (layoutTag == 1)
    layoutEnum = "pto::Layout::DN";
  else if (layoutTag == 2)
    layoutEnum = "pto::Layout::NZ";
  std::string layoutConstName = gtTypeName + "_layout";
  rewriter.create<emitc::VerbatimOp>(
      loc, "constexpr pto::Layout " + layoutConstName + " = " + layoutEnum + ";");

  auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeTypeName);
  auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideTypeName);
  auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, shapeTypeOpaque, shapeTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});
  auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
      loc, strideTypeOpaque, strideTypeName, ArrayAttr{}, ArrayAttr{},
      ValueRange{});

  rewriter.create<emitc::VerbatimOp>(
      loc, "using " + gtTypeName + " = GlobalTensor<" + elemTypeStr + ", " +
               shapeTypeName + ", " + strideTypeName + ", " +
               layoutConstName + ">;");
  auto gtType = emitc::OpaqueType::get(ctx, gtTypeName);

  SmallVector<Value> gtArgs;
  gtArgs.push_back(ptr);
  gtArgs.push_back(shapeInstOp.getResult(0));
  gtArgs.push_back(strideInstOp.getResult(0));

  auto gtInst = rewriter.create<emitc::CallOpaqueOp>(
      loc, gtType, gtTypeName, ArrayAttr{}, ArrayAttr{}, ValueRange(gtArgs));

  return gtInst.getResult(0);
}

//===----------------------------------------------------------------------===//
// pto.pointer_cast lowering
//===----------------------------------------------------------------------===
struct PointerCastConversion : public OpConversionPattern<pto::PointerCastOp> {
  static bool getIndexConst(Value v, int64_t &out) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getValue().getSExtValue();
        return true;
      }
    }
    return false;
  }

  using OpConversionPattern<pto::PointerCastOp>::OpConversionPattern;

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias, Scaling };

  static void collectUserOpsThroughCasts(Value v, SmallVectorImpl<Operation *> &out) {
    for (Operation *u : v.getUsers()) {
      if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(u)) {
        for (Value r : castOp.getResults())
          collectUserOpsThroughCasts(r, out);
        continue;
      }
      out.push_back(u);
    }
  }

  static Value peelUnrealized(Value v) {
    while (auto castOp = v.getDefiningOp<UnrealizedConversionCastOp>()) {
      v = castOp.getOperand(0);
    }
    return v;
  }

  static TileRole inferRole(pto::PointerCastOp op) {
    // 1. 优先检查 AddressSpace
    if (auto memRefTy = dyn_cast<MemRefType>(op.getType())) {
      Attribute memorySpace = memRefTy.getMemorySpace();
      if (auto ptoAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace)) {
        switch (ptoAttr.getAddressSpace()) {
          case pto::AddressSpace::LEFT:  return TileRole::Left;
          case pto::AddressSpace::RIGHT: return TileRole::Right;
          case pto::AddressSpace::ACC:   return TileRole::Acc;
          case pto::AddressSpace::BIAS:  return TileRole::Bias; 
          case pto::AddressSpace::MAT:   return TileRole::Mat;
          case pto::AddressSpace::SCALING: return TileRole::Scaling;
          default: break; 
        }
      }
    }

    // 2. 通过 Usage 推导 (Fallback)
    SmallVector<Operation *, 8> users;
    collectUserOpsThroughCasts(op.getResult(), users);

    for (Operation *user : users) {
      if (auto mm = dyn_cast<pto::MatmulDpsOp>(user)) {
        if (mm.getDst() && peelUnrealized(mm.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mm.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mm.getRhs()) == op.getResult()) return TileRole::Right;
      }
      if (auto mmacc = dyn_cast<pto::MatmulAccDpsOp>(user)) {
        if (mmacc.getDst() && peelUnrealized(mmacc.getDst()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getAccIn()) == op.getResult()) return TileRole::Acc;
        if (peelUnrealized(mmacc.getLhs()) == op.getResult()) return TileRole::Left;
        if (peelUnrealized(mmacc.getRhs()) == op.getResult()) return TileRole::Right;
      }
    }

    return TileRole::Vec;
  }

  // [新增] 辅助函数：判断 Value 是否源自 arith.constant
  static bool isConstant(Value v, int64_t &outVal) {
    if (!v) return false;
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
       if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
           outVal = attr.getInt();
           return true;
       }
    }
    return false;
  }

  LogicalResult matchAndRewrite(pto::PointerCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto selfType = mlir::cast<MemRefType>(op.getType());
    ArrayRef<int64_t> shape = selfType.getShape();
    Type elemType = selfType.getElementType();
    
    // 1. 推导 Tile Role
    TileRole role = inferRole(op);

    // 2. 类型字符串生成 (elemTypeStr, dimStr)
    std::string elemTypeStr = "T";
    if (elemType.isF16()) elemTypeStr = "half";
    else if (elemType.isF32()) elemTypeStr = "float";
    else if (elemType.isInteger(8)) elemTypeStr = cast<IntegerType>(elemType).isUnsigned() ? "uint8_t" : "int8_t";
    else if (elemType.isInteger(16)) elemTypeStr = cast<IntegerType>(elemType).isUnsigned() ? "uint16_t" : "int16_t";
    else if (elemType.isInteger(32)) elemTypeStr = cast<IntegerType>(elemType).isUnsigned() ? "uint32_t" : "int32_t";
    else if (elemType.isInteger(64)) elemTypeStr = cast<IntegerType>(elemType).isUnsigned() ? "uint64_t" : "int64_t";

    std::string dimStr;
    auto dimToString = [](int64_t dim, const char* symbol) -> std::string {
        return (dim == ShapedType::kDynamic) ? std::string(symbol) : std::to_string(dim);
    };

    if (role == TileRole::Left) dimStr = dimToString(shape[0], "M") + ", " + dimToString(shape[1], "K");
    else if (role == TileRole::Right) dimStr = dimToString(shape[0], "K") + ", " + dimToString(shape[1], "N");
    else if (role == TileRole::Bias) dimStr = "1, " + dimToString(shape[1], "N");
    else dimStr = dimToString(shape[0], "M") + ", " + dimToString(shape[1], "N");

    // 3. Role Token
    const char *roleTok = "TileType::Vec";
    switch (role) {
      case TileRole::Left:  roleTok = "TileType::Left"; break;
      case TileRole::Right: roleTok = "TileType::Right"; break;
      case TileRole::Acc:   roleTok = "TileType::Acc"; break;
      case TileRole::Bias:  roleTok = "TileType::Bias"; break;
      case TileRole::Mat:   roleTok = "TileType::Mat"; break;
      case TileRole::Vec:   roleTok = "TileType::Vec"; break;
      case TileRole::Scaling: roleTok = "TileType::Scaling"; break;
    }

    // 4. Config & Layout
    std::string layoutParams = "BLayout::RowMajor";
    std::string extraParams = "";
    if (auto configOpt = op.getConfig()) {
        auto config = *configOpt;
        int32_t blVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getBLayout())) blVal = attr.getInt();
        if (blVal == 1) layoutParams = "BLayout::ColMajor";

        int32_t slVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getSLayout())) slVal = attr.getInt();
        std::string slStr = (slVal == 1) ? "SLayout::RowMajor" : (slVal == 2) ? "SLayout::ColMajor" : "SLayout::NoneBox";

        int32_t frVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getSFractalSize())) frVal = attr.getInt();
        
        int32_t padVal = 0;
        if (auto attr = dyn_cast<IntegerAttr>(config.getPad())) padVal = attr.getInt();
        std::string padStr = "PadValue::Null";
        switch (padVal) {
            case 1: padStr = "PadValue::Zero"; break;
            case 2: padStr = "PadValue::Max";  break;
            case 3: padStr = "PadValue::Min";  break;
        }

        if (!slStr.empty()) {
            extraParams += ", " + slStr + ", " + std::to_string(frVal) + ", " + padStr;
        }
    }

    // [核心修改] Valid Dims 处理逻辑 (支持混合静态/动态)
    std::string vrowTok, vcolTok;
    bool useConstructor = false;
    
    // 引入标志位，明确记录哪个维度是动态的
    bool rowIsDynamic = false;
    bool colIsDynamic = false;

    SmallVector<Value> constructorArgs;

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();
    Value vRowEmitC = adaptor.getValidRow(); 
    Value vColEmitC = adaptor.getValidCol();

    int64_t cRow, cCol;

    // --- Row 逻辑 ---
    if (vRow && isConstant(vRow, cRow)) {
        // Case A: 静态常量 (e.g., 32)
        vrowTok = std::to_string(cRow);
    } else if (vRow) {
        // Case B: 动态变量 (e.g., %arg0)
        vrowTok = "-1";
        rowIsDynamic = true; // 标记为动态
        useConstructor = true;
    } else {
        // Case C: 默认静态 (Shape)
        vrowTok = std::to_string(shape[0]);
    }

    // --- Col 逻辑 ---
    if (vCol && isConstant(vCol, cCol)) {
        // Case A: 静态常量
        vcolTok = std::to_string(cCol);
    } else if (vCol) {
        // Case B: 动态变量
        vcolTok = "-1";
        colIsDynamic = true; // 标记为动态
        useConstructor = true;
    } else {
        // Case C: 默认静态
        vcolTok = std::to_string(shape[1]);
    }

    // --- 收集构造参数 ---
    // [修复] 只收集被标记为 Dynamic 的维度的值
    if (useConstructor) {
        if (rowIsDynamic && vRowEmitC) constructorArgs.push_back(vRowEmitC);
        if (colIsDynamic && vColEmitC) constructorArgs.push_back(vColEmitC);
    }

    // 5. 生成 Tile 类型字符串
    std::string tileTypeStr =
      std::string("Tile<") + roleTok + ", " + elemTypeStr + ", " + dimStr + ", " +
      layoutParams + ", " + vrowTok + ", " + vcolTok + extraParams + ">";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value resultValue;

    if (useConstructor) {
        // 使用 CallOpaqueOp 生成构造函数调用 (Tile v = Tile(...))
        auto ctorOp = rewriter.create<emitc::CallOpaqueOp>(
            loc, 
            tileType,        // Result Type
            tileTypeStr,     // Callee Name (类名)
            ArrayAttr{},     // args
            ArrayAttr{},     // template_args
            ValueRange(constructorArgs) // operands
        );
        resultValue = ctorOp.getResult(0);
    } else {
        // 静态情况 (Tile v;)
        auto varOp = rewriter.create<emitc::VariableOp>(
            loc, 
            tileType, 
            emitc::OpaqueAttr::get(ctx, "")
        );
        resultValue = varOp.getResult();
    }

    // TASSIGN: pto-isa expects an integral address.
    Value addr = adaptor.getAddrs()[0];
    if (isa<emitc::PointerType>(addr.getType()) ||
        (isa<emitc::OpaqueType>(addr.getType()) &&
         cast<emitc::OpaqueType>(addr.getType()).getValue().ends_with("*"))) {
      auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
      auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});
      addr = rewriter.create<emitc::CallOpaqueOp>(
                 loc, u64Ty, "reinterpret_cast",
                 /*args=*/ArrayAttr{}, /*templateArgs=*/rcU64,
                 /*operands=*/ValueRange{addr})
                 .getResult(0);
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, addr});

    rewriter.replaceOp(op, resultValue);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_dps / pto.store_dps lowering (FIX: keep optional result)
//===----------------------------------------------------------------------===

struct PTOLoadDpsToTLOAD : public OpConversionPattern<pto::LoadDpsOp> {
  using OpConversionPattern<pto::LoadDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.load_dps");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value srcArg = src;
    if (auto srcMrTy = dyn_cast<MemRefType>(op.getSrc().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(srcMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), src, srcMrTy,
                                                  op.getOperation()))
          srcArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, srcArg});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct PTOStoreDpsToTSTORE : public OpConversionPattern<pto::StoreDpsOp> {
  using OpConversionPattern<pto::StoreDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.store_dps");

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value dstArg = dst;
    if (auto dstMrTy = dyn_cast<MemRefType>(op.getDst().getType())) {
      bool isGlobal = true;
      if (auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(dstMrTy.getMemorySpace())) {
        auto as = asAttr.getAddressSpace();
        isGlobal = (as == pto::AddressSpace::GM || as == pto::AddressSpace::Zero);
      }
      if (isGlobal) {
        if (Value gt = buildGlobalTensorFromMemref(rewriter, op.getLoc(), dst, dstMrTy,
                                                  op.getOperation()))
          dstArg = gt;
      }
    }

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSTORE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dstArg, src});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.addf_dps lowering (TADD requires dst)
//===----------------------------------------------------------------------===

struct PTOAddfDpsToTADD : public OpConversionPattern<pto::AddFDpsOp> {
  using OpConversionPattern<pto::AddFDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddFDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) on pto.addf_dps");

    Value lhs = peelUnrealized(adaptor.getLhs());
    Value rhs = peelUnrealized(adaptor.getRhs());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOMatmulDpsToTMATMUL : public OpConversionPattern<pto::MatmulDpsOp> {
  using OpConversionPattern<pto::MatmulDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Left)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Right)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Acc)

    // 2. 直接生成函数调用 TMATMUL(dst, lhs, rhs)
    // 假设输入已经在对应的 L0 Buffer 中
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.gemv_dps lowering
//===----------------------------------------------------------------------===//
struct PTOGemvDpsToTGEMV : public OpConversionPattern<pto::GemvDpsOp> {
  using OpConversionPattern<pto::GemvDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GemvDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数 (剥离 Cast)
    Value lhs = peelUnrealized(adaptor.getLhs()); // A (Matrix)
    Value rhs = peelUnrealized(adaptor.getRhs()); // B (Vector)
    Value dst = peelUnrealized(adaptor.getDst()); // C (Result)

    // 2. 直接生成函数调用 TGEMV(dst, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.gemv_acc_dps lowering
//===----------------------------------------------------------------------===//
struct PTOGemvAccDpsToTGEMVACC : public OpConversionPattern<pto::GemvAccDpsOp> {
  using OpConversionPattern<pto::GemvAccDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GemvAccDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.gemv_acc_dps");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Matrix)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Vector)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 直接生成函数调用 TGEMV_ACC(dst, accIn, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TGEMV_ACC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.matmul_acc_dps lowering (Simplified: No internal copy/sync)
//===----------------------------------------------------------------------===//
struct PTOMatmulAccDpsToTMATMULACC : public OpConversionPattern<pto::MatmulAccDpsOp> {
  using OpConversionPattern<pto::MatmulAccDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulAccDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.getDst())
      return rewriter.notifyMatchFailure(op, "expected outs(dst) for pto.matmul_acc_dps");

    // 1. 获取操作数
    Value accIn = peelUnrealized(adaptor.getAccIn()); // AccOld
    Value lhs   = peelUnrealized(adaptor.getLhs());   // A (Left)
    Value rhs   = peelUnrealized(adaptor.getRhs());   // B (Right)
    Value dst   = peelUnrealized(adaptor.getDst());   // AccNew

    // 2. 直接生成函数调用 TMATMUL_ACC(dst, accIn, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TMATMUL_ACC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, accIn, lhs, rhs});

    // 3. 处理 Op 替换/删除
    if (op->getNumResults() == 1) {
      rewriter.replaceOp(op, dst);
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Return lowering
//===----------------------------------------------------------------------===

struct ReturnToEmitC : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto vals = adaptor.getOperands();
    if (vals.empty()) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, Value{});
      return success();
    }
    if (vals.size() == 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(op, vals[0]);
      return success();
    }
    return rewriter.notifyMatchFailure(op, "EmitC cannot return multiple values");
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering
//===----------------------------------------------------------------------===

static std::string getPipeName(pto::PIPE pipe) {
  switch (pipe) {
    case pto::PIPE::PIPE_S: return "PIPE_S";
    case pto::PIPE::PIPE_V: return "PIPE_V";
    case pto::PIPE::PIPE_M: return "PIPE_M";
    case pto::PIPE::PIPE_MTE1: return "PIPE_MTE1";
    case pto::PIPE::PIPE_MTE2: return "PIPE_MTE2";
    case pto::PIPE::PIPE_MTE3: return "PIPE_MTE3";
    case pto::PIPE::PIPE_ALL: return "PIPE_ALL";
    case pto::PIPE::PIPE_MTE4: return "PIPE_MTE4";
    case pto::PIPE::PIPE_MTE5: return "PIPE_MTE5";
    case pto::PIPE::PIPE_V2: return "PIPE_V2";
    case pto::PIPE::PIPE_FIX: return "PIPE_FIX";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1A: return "VIRTUAL_PIPE_MTE2_L1A";
    case pto::PIPE::VIRTUAL_PIPE_MTE2_L1B: return "VIRTUAL_PIPE_MTE2_L1B";
    // 默认回退
    default: return "PIPE_ALL"; 
  }
}

//===----------------------------------------------------------------------===//
// pto.barrier lowering -> pipe_barrier(...)
//===----------------------------------------------------------------------===//
struct PTOBarrierToEmitC : public OpConversionPattern<pto::BarrierOp> {
  using OpConversionPattern<pto::BarrierOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BarrierOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();

    // [FIX] op.getPipe() returns PipeAttr. 
    // We must call .getPipe() on the attribute to get the actual Enum value.
    pto::PIPE pipeEnum = op.getPipe().getPipe();

    // Convert Enum to String (e.g., PIPE_ALL -> "PIPE_ALL")
    std::string pipeStr = pto::stringifyPIPE(pipeEnum).str();

    auto args = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, pipeStr)
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        TypeRange{},        // void return
        "pipe_barrier",     // function name
        args,               // arguments
        ArrayAttr{},        // template args
        ValueRange{}        // operands
    );

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sync lowering (robust for bracket form pto.set_flag[...] / pto.wait_flag[...])
// Replace your PTOSyncToRuntimeCall with the code below.
//===----------------------------------------------------------------------===//

static LogicalResult extractSyncTripletTokens(Operation *op,
                                             std::string &srcTok,
                                             std::string &dstTok,
                                             std::string &evtTok,
                                             ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();

  auto pipeToTok = [](mlir::Attribute a, std::string &out) -> bool {
    if (!a) return false;
    if (auto p = dyn_cast<mlir::pto::PipeAttr>(a)) {
      out = mlir::pto::stringifyPIPE(p.getPipe()).str();
      return true;
    }
    if (auto s = dyn_cast<StringAttr>(a)) {
      out = s.getValue().str(); // expects already like "PIPE_MTE2"
      return true;
    }
    return false;
  };

  auto evtToTok = [](mlir::Attribute a, std::string &out) -> bool {
    if (!a) return false;
    if (auto e = dyn_cast<mlir::pto::EventAttr>(a)) {
      out = mlir::pto::stringifyEVENT(e.getEvent()).str();
      return true;
    }
    if (auto s = dyn_cast<StringAttr>(a)) {
      out = s.getValue().str(); // expects already like "EVENT_ID0"
      return true;
    }
    return false;
  };

  auto tryNamed = [&](StringRef s0, StringRef s1, StringRef e0) -> bool {
    std::string st, dt, et;
    if (!pipeToTok(op->getAttr(s0), st)) return false;
    if (!pipeToTok(op->getAttr(s1), dt)) return false;
    if (!evtToTok(op->getAttr(e0), et)) return false;
    srcTok = std::move(st);
    dstTok = std::move(dt);
    evtTok = std::move(et);
    return true;
  };

  // 1) Most common named-attr encodings
  if (tryNamed("src_pipe", "dst_pipe", "event_id")) return success();
  if (tryNamed("srcPipe",  "dstPipe",  "eventId"))  return success();
  if (tryNamed("src",      "dst",      "event"))    return success();

  // 2) Bracket-form / custom-asm often packs them into an ArrayAttr under some key
  auto tryArrayKey = [&](StringRef key) -> bool {
    auto arr = op->getAttrOfType<ArrayAttr>(key);
    if (!arr || arr.size() < 3) return false;

    std::string st, dt, et;
    if (!pipeToTok(arr[0], st)) return false;
    if (!pipeToTok(arr[1], dt)) return false;
    if (!evtToTok(arr[2], et))  return false;
    srcTok = std::move(st);
    dstTok = std::move(dt);
    evtTok = std::move(et);
    return true;
  };

  if (tryArrayKey("args") || tryArrayKey("pipes") || tryArrayKey("sync") ||
      tryArrayKey("triplet") || tryArrayKey("attrs"))
    return success();

  // 3) Last resort: scan everything and pick 2 Pipe + 1 Event in encounter order.
  std::vector<std::string> pipes;
  std::string event;
  for (auto &na : op->getAttrs()) {
    Attribute a = na.getValue();
    std::string tok;
    if (pipeToTok(a, tok)) {
      pipes.push_back(std::move(tok));
      continue;
    }
    if (evtToTok(a, tok)) {
      event = std::move(tok);
      continue;
    }
  }

  if (pipes.size() >= 2 && !event.empty()) {
    srcTok = pipes[0];
    dstTok = pipes[1];
    evtTok = event;
    return success();
  }

  return rewriter.notifyMatchFailure(op, "cannot extract PIPE/PIPE/EVENT tokens from pto.{set,wait}_flag");
}
static inline std::string pipeTokFromPipeEnum(mlir::pto::PIPE p) {
  return mlir::pto::stringifyPIPE(p).str();
}
static inline std::string evtTokFromEventEnum(mlir::pto::EVENT e) {
  return mlir::pto::stringifyEVENT(e).str();
}
static inline std::string pipeTokFromPipeAttr(mlir::pto::PipeAttr a) {
  return mlir::pto::stringifyPIPE(a.getPipe()).str();
}
static inline std::string evtTokFromEventAttr(mlir::pto::EventAttr a) {
  return mlir::pto::stringifyEVENT(a.getEvent()).str();
}

template <typename T, typename = void>
struct HasGetSrcPipe : std::false_type {};
template <typename T>
struct HasGetSrcPipe<T, std::void_t<decltype(std::declval<T>().getSrcPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipe : std::false_type {};
template <typename T>
struct HasGetDstPipe<T, std::void_t<decltype(std::declval<T>().getDstPipe())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventId : std::false_type {};
template <typename T>
struct HasGetEventId<T, std::void_t<decltype(std::declval<T>().getEventId())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetSrcPipeAttr : std::false_type {};
template <typename T>
struct HasGetSrcPipeAttr<T, std::void_t<decltype(std::declval<T>().getSrcPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetDstPipeAttr : std::false_type {};
template <typename T>
struct HasGetDstPipeAttr<T, std::void_t<decltype(std::declval<T>().getDstPipeAttr())>> : std::true_type {};

template <typename T, typename = void>
struct HasGetEventIdAttr : std::false_type {};
template <typename T>
struct HasGetEventIdAttr<T, std::void_t<decltype(std::declval<T>().getEventIdAttr())>> : std::true_type {};

template <typename SyncOpT>
static LogicalResult extractSyncTokens(SyncOpT op,
                                      std::string &srcTok,
                                      std::string &dstTok,
                                      std::string &evtTok,
                                      ConversionPatternRewriter &rewriter) {
  if constexpr (HasGetSrcPipe<SyncOpT>::value &&
                HasGetDstPipe<SyncOpT>::value &&
                HasGetEventId<SyncOpT>::value) {
    auto s = op.getSrcPipe();
    auto d = op.getDstPipe();
    auto e = op.getEventId();

    if constexpr (std::is_same<decltype(s), mlir::pto::PIPE>::value) srcTok = pipeTokFromPipeEnum(s);
    else srcTok = pipeTokFromPipeAttr(s);

    if constexpr (std::is_same<decltype(d), mlir::pto::PIPE>::value) dstTok = pipeTokFromPipeEnum(d);
    else dstTok = pipeTokFromPipeAttr(d);

    if constexpr (std::is_same<decltype(e), mlir::pto::EVENT>::value) evtTok = evtTokFromEventEnum(e);
    else evtTok = evtTokFromEventAttr(e);

    return success();
  }

  if constexpr (HasGetSrcPipeAttr<SyncOpT>::value &&
                HasGetDstPipeAttr<SyncOpT>::value &&
                HasGetEventIdAttr<SyncOpT>::value) {
    auto s = op.getSrcPipeAttr();
    auto d = op.getDstPipeAttr();
    auto e = op.getEventIdAttr();
    srcTok = pipeTokFromPipeAttr(s);
    dstTok = pipeTokFromPipeAttr(d);
    evtTok = evtTokFromEventAttr(e);
    return success();
  }

  return extractSyncTripletTokens(op.getOperation(), srcTok, dstTok, evtTok, rewriter);
}
struct PTOSetFlagToEmitC : public OpConversionPattern<mlir::pto::SetFlagOp> {
  using OpConversionPattern<mlir::pto::SetFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::SetFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "set_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOWaitFlagToEmitC : public OpConversionPattern<mlir::pto::WaitFlagOp> {
  using OpConversionPattern<mlir::pto::WaitFlagOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::pto::WaitFlagOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();

    std::string srcTok, dstTok, evtTok;
    if (failed(extractSyncTokens(op, srcTok, dstTok, evtTok, rewriter)))
      return failure();

    auto argsAttr = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, srcTok),
        emitc::OpaqueAttr::get(ctx, dstTok),
        emitc::OpaqueAttr::get(ctx, evtTok),
    });

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "wait_flag",
        /*args=*/argsAttr,
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{});
    return success();
  }
};

struct PTOSyncSetToEmitC : public OpConversionPattern<mlir::pto::SyncSetOp> {
  using OpConversionPattern<mlir::pto::SyncSetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::SyncSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();
    auto loc = op->getLoc();

    std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, pipeTok), op.getEventIdAttr()});
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "set_intra_block",
                                         /*args=*/argsAttr,
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSyncWaitToEmitC : public OpConversionPattern<mlir::pto::SyncWaitOp> {
  using OpConversionPattern<mlir::pto::SyncWaitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::SyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *ctx = rewriter.getContext();
    auto loc = op->getLoc();

    std::string pipeTok = pipeTokFromPipeAttr(op.getPipe());
    auto argsAttr = rewriter.getArrayAttr(
        {emitc::OpaqueAttr::get(ctx, pipeTok), op.getEventIdAttr()});
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "wait_intra_block",
                                         argsAttr, ArrayAttr{}, ValueRange{});

    rewriter.eraseOp(op);
    return success();
  }
};

// GetBlockIdxOp Lowering (pto.get_block_idx -> get_block_idx())
struct PTOGetBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_idx", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetBlockNumOp Lowering (pto.get_block_num -> get_block_num())
struct PTOGetBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_block_num", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockIdxOp Lowering (pto.get_block_idx -> get_subblockid())
struct PTOGetSubBlockIdxToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockIdxOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockIdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockIdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockid", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

// GetSubBlockNumOp Lowering (pto.get_block_num -> get_subblockdim())
struct PTOGetSubBlockNumToEmitC
    : public OpConversionPattern<mlir::pto::GetSubBlockNumOp> {
  using OpConversionPattern<mlir::pto::GetSubBlockNumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::pto::GetSubBlockNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op.getType(), "get_subblockdim", ValueRange{}, ArrayAttr{},
        ArrayAttr{});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.mscatter lowering -> MSCATTER(mem, src, idx)
// pto.mscatter %src, %mem, %idx : memref<...>, memref<...>, memref<...>
//===----------------------------------------------------------------------===//

struct PTOMScatterToMSCATTER : public OpConversionPattern<pto::MScatterDpsOp> {
  using OpConversionPattern<pto::MScatterDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MScatterDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value mem = peelUnrealized(adaptor.getMem());

    // pto-isa currently has no NPU implementation for MGATHER/MSCATTER.
    // Fallback to a smoke-friendly lowering to keep compile/run coverage.
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSTORE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{mem, src});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOSetValToSETVAL : public OpConversionPattern<pto::SetValDpsOp> {
  using OpConversionPattern<pto::SetValDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetValDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value dst = peelUnrealized(adaptor.getDst());
    Value val = peelUnrealized(adaptor.getVal());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // NOTE: EmitC has no direct member-call op today. We emit a marker call
    // and post-process ptoas output to rewrite it into:
    //   dst.SetValue(offset, val);
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__TILE_SET_VALUE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOGetValToGETVAL : public OpConversionPattern<pto::GetValDpsOp> {
  using OpConversionPattern<pto::GetValDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GetValDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());

    // ---- offset: SSA index operand ----
    Value offset = peelUnrealized(adaptor.getOffset());

    // NOTE: EmitC has no direct member-call op today. We emit a marker call
    // and post-process ptoas output to rewrite it into:
    //   auto x = src.GetValue(offset);
    Type dstTy = getTypeConverter()->convertType(op.getDst().getType());
    if (!dstTy)
      return failure();
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        TypeRange{dstTy},
        "PTOAS__TILE_GET_VALUE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.load_scalar / pto.store_scalar lowering -> ptr[offset]
//===----------------------------------------------------------------------===//

struct PTOLoadScalarToEmitC : public OpConversionPattern<pto::LoadScalarOp> {
  using OpConversionPattern<pto::LoadScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LoadScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());

    Type dstTy = getTypeConverter()->convertType(op.getValue().getType());
    if (!dstTy)
      return failure();

    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{dstTy}, "PTOAS__PTR_LOAD",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset});

    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct PTOStoreScalarToEmitC : public OpConversionPattern<pto::StoreScalarOp> {
  using OpConversionPattern<pto::StoreScalarOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreScalarOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = peelUnrealized(adaptor.getPtr());
    Value offset = peelUnrealized(adaptor.getOffset());
    Value val = peelUnrealized(adaptor.getValue());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "PTOAS__PTR_STORE",
        ArrayAttr{}, ArrayAttr{}, ValueRange{ptr, offset, val});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tabs lowering -> TABS(dst, src)
//===----------------------------------------------------------------------===//

struct PTOAbsToTABS : public OpConversionPattern<pto::AbsOp_DPS> {
  using OpConversionPattern<pto::AbsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AbsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TABS(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TABS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadd lowering -> TADD(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTOAddToTADD : public OpConversionPattern<pto::AddOp_DPS> {
  using OpConversionPattern<pto::AddOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// populate patterns
//===----------------------------------------------------------------------===
struct ReinterpretCastToEmitC : public OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resMrTy = dyn_cast<MemRefType>(op.getType());
    if (!resMrTy)
      return failure();

    auto asAttr = dyn_cast_or_null<pto::AddressSpaceAttr>(resMrTy.getMemorySpace());
    const bool isGm = (!asAttr || asAttr.getAddressSpace() == pto::AddressSpace::GM);

    bool emitAddPtrTrace = op->hasAttr("pto.addptr_trace");
    Value source = peelUnrealized(adaptor.getSource());
    auto offsets = adaptor.getOffsets();
    Value offsetVal = offsets.empty() ? Value() : offsets[0];

    // GM: keep pointer arithmetic.
    if (isGm) {
      if (!offsetVal) {
        rewriter.replaceOp(op, source);
        return success();
      }

      Type resultType = getTypeConverter()->convertType(op.getType());
      if (!resultType)
        return failure();

      auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);
      if (emitAddPtrTrace) {
        rewriter.setInsertionPointAfter(addOp);
        rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{}, "PTOAS__ADDPTR_TRACE",
            ArrayAttr{}, ArrayAttr{},
            ValueRange{addOp.getResult(), source, offsetVal});
      }
      rewriter.replaceOp(op, addOp.getResult());
      return success();
    }

    // UB/L1/L0 tiles: materialize a new Tile view by assigning an adjusted
    // underlying pointer (in elements).
    pto::AddressSpace as = asAttr.getAddressSpace();

    // Element type token.
    std::string elemTok = "float";
    Type elemTy = resMrTy.getElementType();
    int64_t elemBytes = 4;
    if (elemTy.isF16())
      elemBytes = 2,
      elemTok = "half";
    else if (elemTy.isF32())
      elemBytes = 4,
      elemTok = "float";
    else if (elemTy.isInteger(8))
      elemBytes = 1,
      elemTok = cast<IntegerType>(elemTy).isUnsigned() ? "uint8_t" : "int8_t";
    else if (elemTy.isInteger(16))
      elemBytes = 2,
      elemTok = cast<IntegerType>(elemTy).isUnsigned() ? "uint16_t" : "int16_t";
    else if (elemTy.isInteger(32))
      elemBytes = 4,
      elemTok = cast<IntegerType>(elemTy).isUnsigned() ? "uint32_t" : "int32_t";
    else if (elemTy.isInteger(64))
      elemBytes = 8,
      elemTok = cast<IntegerType>(elemTy).isUnsigned() ? "uint64_t" : "int64_t";

    // Tile role.
    const char *roleTok = "TileType::Vec";
    switch (as) {
    case pto::AddressSpace::VEC:
      roleTok = "TileType::Vec";
      break;
    case pto::AddressSpace::MAT:
      roleTok = "TileType::Mat";
      break;
    case pto::AddressSpace::LEFT:
      roleTok = "TileType::Left";
      break;
    case pto::AddressSpace::RIGHT:
      roleTok = "TileType::Right";
      break;
    case pto::AddressSpace::ACC:
      roleTok = "TileType::Acc";
      break;
    case pto::AddressSpace::BIAS:
      roleTok = "TileType::Bias";
      break;
    case pto::AddressSpace::GM:
      roleTok = "TileType::Vec";
      break;
    }

    // Shape (fallback to 32x32).
    int64_t rows = 32, cols = 32;
    if (resMrTy.getRank() >= 2 && resMrTy.hasStaticShape()) {
      rows = resMrTy.getDimSize(0);
      cols = resMrTy.getDimSize(1);
    }

    // Keep a conservative default config for now.
    std::string tileTypeStr =
        std::string("Tile<") + roleTok + ", " + elemTok + ", " +
        std::to_string(rows) + ", " + std::to_string(cols) +
        ", BLayout::RowMajor, " + std::to_string(rows) + ", " +
        std::to_string(cols) + ", SLayout::NoneBox, 512, PadValue::Null>";

    auto tileType = emitc::OpaqueType::get(ctx, tileTypeStr);
    Value tile = rewriter
                     .create<emitc::VariableOp>(loc, tileType,
                                                emitc::OpaqueAttr::get(ctx, ""))
                     .getResult();

    // Compute an integer address and assign it to the new tile.
    // NOTE: pto-isa TASSIGN requires an integral address (not a pointer).
    auto u64Ty = emitc::OpaqueType::get(ctx, "uint64_t");
    auto rcU64 = rewriter.getArrayAttr({emitc::OpaqueAttr::get(ctx, "uint64_t")});

    // Non-GM reinterpret_cast operands come from UB/L1/L0 tiles.
    // We need the underlying address, but `__cce_get_tile_ptr()` is only valid
    // inside `__tf__` functions. Use `tile.data()` (via a post-processed marker)
    // and compute the adjusted address in bytes.
    Value rawPtr = source;
    if (auto ot = dyn_cast<emitc::OpaqueType>(source.getType())) {
      // Only Tiles have a `.data()` member. For plain address-space pointers
      // (e.g. `__ubuf__ float*`), use the pointer value directly.
      if (ot.getValue().starts_with("Tile<")) {
        std::string rawPtrTok =
            std::string(addrSpaceQualifier(as)) + " " + elemTok + "*";
        auto rawPtrTy = emitc::OpaqueType::get(ctx, rawPtrTok);
        rawPtr = rewriter
                     .create<emitc::CallOpaqueOp>(loc, rawPtrTy,
                                                  "PTOAS__TILE_DATA", ArrayAttr{},
                                                  ArrayAttr{}, ValueRange{source})
                     .getResult(0);
      }
    }

    Value baseAddr = rewriter
                         .create<emitc::CallOpaqueOp>(loc, u64Ty, "reinterpret_cast",
                                                      /*args=*/ArrayAttr{},
                                                      /*templateArgs=*/rcU64,
                                                      /*operands=*/ValueRange{rawPtr})
                         .getResult(0);

    Value addr = baseAddr;
    if (offsetVal) {
      Value offU64 = offsetVal;
      if (offU64.getType() != u64Ty)
        offU64 = rewriter.create<emitc::CastOp>(loc, u64Ty, offU64).getResult();

      auto bytesAttr = emitc::OpaqueAttr::get(ctx, std::to_string(elemBytes));
      Value bytesVal = rewriter.create<emitc::ConstantOp>(loc, u64Ty, bytesAttr);
      Value byteOff = rewriter.create<emitc::MulOp>(loc, u64Ty, offU64, bytesVal);
      addr = rewriter.create<emitc::AddOp>(loc, u64Ty, baseAddr, byteOff);
    }

    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "TASSIGN",
                                         /*args=*/ArrayAttr{},
                                         /*templateArgs=*/ArrayAttr{},
                                         /*operands=*/ValueRange{tile, addr});

    rewriter.replaceOp(op, tile);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddc lowering -> TADDC(dst, src0, src1, src2)
//===----------------------------------------------------------------------===//

struct PTOAddCToTADDC : public OpConversionPattern<pto::AddCOp_DPS> {
  using OpConversionPattern<pto::AddCOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddCOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst  = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDC yet.
    // Decompose: dst = src0 + src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tadds lowering -> TADDS(dst, src, scalar)
//===----------------------------------------------------------------------===//

struct PTOAddSToTADDS : public OpConversionPattern<pto::AddSOp_DPS> {
  using OpConversionPattern<pto::AddSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value dst    = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.taddsc lowering -> TADDSC(dst, src0, scalar, src1)
//===----------------------------------------------------------------------===//

struct PTOAddSCToTADDSC : public OpConversionPattern<pto::AddSCOp_DPS> {
  using OpConversionPattern<pto::AddSCOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AddSCOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src0    = peelUnrealized(adaptor.getSrc0());
    Value scalar  = peelUnrealized(adaptor.getScalar());
    Value src1    = peelUnrealized(adaptor.getSrc1());
    Value dst     = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TADDSC yet.
    // Decompose: dst = src0 + scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOAndToEmitC : public OpConversionPattern<pto::AndOp_DPS> {
  using OpConversionPattern<pto::AndOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AndOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a   = peelUnrealized(adaptor.getSrc0());
    Value b   = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TAND",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, a, b});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOAndSToEmitC : public OpConversionPattern<pto::AndSOp_DPS> {
  using OpConversionPattern<pto::AndSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AndSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOCIToEmitC : public OpConversionPattern<pto::CIOp_DPS> {
  using OpConversionPattern<pto::CIOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CIOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value S   = peelUnrealized(adaptor.getS());

    // scalar cpp type token
    std::string scalarTok = "int32_t";
    if (auto it = S.getType().dyn_cast<IntegerType>()) {
      scalarTok = (it.getWidth() == 16) ? "int16_t" : "int32_t";
    }

    // descending -> "0"/"1"
    std::string descTok = op.getDescending() ? "1" : "0";

    ArrayAttr targs;
    if (auto ot = dst.getType().dyn_cast<emitc::OpaqueType>()) {
      std::string tileTok = ot.getValue().str(); // "Tile<...>"
      targs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, tileTok),
          emitc::OpaqueAttr::get(ctx, scalarTok),
          emitc::OpaqueAttr::get(ctx, descTok),
      });
    } else {
      targs = rewriter.getArrayAttr({});
    }

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCI",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, S});

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string cmpModeTok(pto::CmpModeAttr a) {
  // 生成 "CmpMode::GT" 这种 token
  auto m = a.getValue(); // 取 enum
  switch (m) {
    case pto::CmpMode::EQ: return "CmpMode::EQ";
    case pto::CmpMode::NE: return "CmpMode::NE";
    case pto::CmpMode::LT: return "CmpMode::LT";
    case pto::CmpMode::LE: return "CmpMode::LE";
    case pto::CmpMode::GT: return "CmpMode::GT";
    case pto::CmpMode::GE: return "CmpMode::GE";
  }
  return "CmpMode::EQ";
}
struct PTOColExpandToEmitC : public OpConversionPattern<pto::ColExpandOp_DPS> {
  using OpConversionPattern<pto::ColExpandOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ColExpandOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLEXPAND",
        /*args=*/ArrayAttr(),           
        /*templateArgs=*/ArrayAttr(),
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpToEmitC : public OpConversionPattern<pto::CmpOp_DPS> {
  using OpConversionPattern<pto::CmpOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CmpOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
	
    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());

    std::string tok = "CmpMode::EQ";
    if (auto a = op.getCmpModeAttr())
      tok = cmpModeTok(a);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    auto argsAttr = rewriter.getArrayAttr({});

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMP",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOCmpSToEmitC : public OpConversionPattern<pto::CmpSOp_DPS> {
  using OpConversionPattern<pto::CmpSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CmpSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());

    // cmpMode -> token
    auto cmpAttr = op.getCmpModeAttr();          // PTO_CmpModeAttr
    std::string tok = cmpModeTok(cmpAttr);

    auto modeTy = emitc::OpaqueType::get(ctx, "CmpMode");
    Value modeVal = rewriter.create<emitc::ConstantOp>(
        loc, modeTy, emitc::OpaqueAttr::get(ctx, tok));

    rewriter.create<emitc::CallOpaqueOp>(
        loc,
        TypeRange{},
        "TCMPS",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, scalar, modeVal});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOColMaxToEmitC : public OpConversionPattern<pto::ColMaxOp_DPS> {
  using OpConversionPattern<pto::ColMaxOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ColMaxOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMAX(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMAX",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOColMinToEmitC : public OpConversionPattern<pto::ColMinOp_DPS> {
  using OpConversionPattern<pto::ColMinOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ColMinOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // intrinsic: TCOLMIN(dst, src)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCOLMIN",
        /*args=*/ArrayAttr{},          // default: print all operands
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOColSumToEmitC : public OpConversionPattern<pto::ColSumOp_DPS> {
  using OpConversionPattern<pto::ColSumOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ColSumOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // Check if tmp exists before accessing it
    if (op.getTmp()) {
      // Format 2: with tmp and isBinary
      Value tmp = peelUnrealized(adaptor.getTmp());
      bool isBinary = false;
      if (auto a = op.getIsBinaryAttr())
        isBinary = a.getValue();

      auto boolTy = emitc::OpaqueType::get(ctx, "bool");
      auto tok = isBinary ? "true" : "false";
      Value isBinaryVal = rewriter.create<emitc::ConstantOp>(
          loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),             
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src, tmp, isBinaryVal});
    } else {
      // Format 1: without tmp and isBinary
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TCOLSUM",
          /*args=*/ArrayAttr(),             
          /*templateArgs=*/ArrayAttr(),
          /*operands=*/ValueRange{dst, src});
    }

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::CAST_NONE";
  case RM::RINT:      return "RoundMode::CAST_RINT";
  case RM::ROUND:     return "RoundMode::CAST_ROUND";
  case RM::FLOOR:     return "RoundMode::CAST_FLOOR";
  case RM::CEIL:      return "RoundMode::CAST_CEIL";
  case RM::TRUNC:     return "RoundMode::CAST_TRUNC";
  case RM::ODD:       return "RoundMode::CAST_ODD";
  case RM::CAST_RINT: return "RoundMode::CAST_RINT";
  }
  return "RoundMode::CAST_RINT";
}
struct PTOCvtToEmitC : public OpConversionPattern<pto::CvtOp_DPS> {
  using OpConversionPattern<pto::CvtOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::CvtOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    // rmode default: CAST_RINT
    pto::RoundModeAttr rmAttr = op.getRmodeAttr();
    std::string rmTok = rmAttr ? roundModeTok(rmAttr)
                               : std::string("RoundMode::CAST_RINT");

    // 生成: TCVT(dst, src, RoundMode::XXX)
    auto rmodeTy = emitc::OpaqueType::get(ctx, "RoundMode");
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, rmodeTy, emitc::OpaqueAttr::get(ctx, rmTok));

    // 这里 args 被清空，只保留 operands，包括 src, dst 和 rmode
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TCVT",
        /*args=*/ArrayAttr{},                  // 不使用 args
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, rmodeVal}); // 传递 dst, src 和 rmode

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdiv lowering -> TDIV(dst, src0, src1)
//===----------------------------------------------------------------------===//

struct PTODivToTDIV : public OpConversionPattern<pto::DivOp_DPS> {
  using OpConversionPattern<pto::DivOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::DivOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TDIV",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tdivs lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTODivSToEmitC : public OpConversionPattern<pto::DivSOp_DPS> {
  using OpConversionPattern<pto::DivSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::DivSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Check types BEFORE conversion (using original op types, not adaptor types)
    // The adaptor types may already be converted to emitc.opaque
    Value origSrc = op.getSrc();
    Value origScalar = op.getScalar();
    
    // Determine order based on original operand types
    // Check if src is memref/tensor/partition_tensor_view/tile (not scalar)
    bool srcIsMemref = (isa<MemRefType>(origSrc.getType()) || 
                        isa<RankedTensorType>(origSrc.getType()) ||
                        isa<mlir::pto::PartitionTensorViewType>(origSrc.getType()) ||
                        isa<mlir::pto::TileBufType>(origSrc.getType()));
    // Check if scalar is memref/tensor/partition_tensor_view/tile (not scalar)
    bool scalarIsMemref = (isa<MemRefType>(origScalar.getType()) || 
                           isa<RankedTensorType>(origScalar.getType()) ||
                           isa<mlir::pto::PartitionTensorViewType>(origScalar.getType()) ||
                           isa<mlir::pto::TileBufType>(origScalar.getType()));

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    if (srcIsMemref && !scalarIsMemref) {
      // memref/scalar: TDIVS(dst, src, scalar) - normal order
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, src, scalar});
    } else if (!srcIsMemref && scalarIsMemref) {
          // scalar/memref: TDIVS(dst, scalar, src) - swapped order
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, scalar, src});
    } else {
      // This should not happen if verifier is correct, but provide a fallback
      return op.emitError("DivSOp_DPS: expected exactly one memref/tensor operand and one scalar operand");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// pto.tdivs (TDivSOp) lowering -> TDIVS(dst, src, scalar)  or  TDIVS(dst, scalar, src)
// Order is determined by operand types: if src is tile_buf, order is (tile, scalar)
// Otherwise, order is (scalar, tile)
//===----------------------------------------------------------------------===//

struct PTOTDivSToEmitC : public OpConversionPattern<pto::TDivSOp> {
  using OpConversionPattern<pto::TDivSOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TDivSOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    // Determine order based on operand types
    bool srcIsTile = isa<mlir::pto::TileBufType>(src.getType());
    bool scalarIsTile = isa<mlir::pto::TileBufType>(scalar.getType());

    if (srcIsTile && !scalarIsTile) {
      // tile/scalar: TDIVS(dst, src, scalar)
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, src, scalar});
    } else if (!srcIsTile && scalarIsTile) {
      // scalar/tile: TDIVS(dst, scalar, src)
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, scalar, src});
    } else {
      // Default: assume src is tile (should not happen if types are correct)
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, src, scalar});
    }

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texp lowering -> TEXP(dst, src)
//===----------------------------------------------------------------------===//

struct PTOExpToEmitC : public OpConversionPattern<pto::ExpOp_DPS> {
  using OpConversionPattern<pto::ExpOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ExpOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXP",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.texpands lowering -> TEXPANDS(dst, scalar)
//===----------------------------------------------------------------------===//

struct PTOExpandsToEmitC : public OpConversionPattern<pto::ExpandsOp_DPS> {
  using OpConversionPattern<pto::ExpandsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ExpandsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXPANDS",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, scalar});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.textract lowering -> TEXTRACT(dst, src, indexRow, indexCol)
//===----------------------------------------------------------------------===//

struct PTOExtractToEmitC : public OpConversionPattern<pto::ExtractOp_DPS> {
  using OpConversionPattern<pto::ExtractOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ExtractOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value r0  = peelUnrealized(adaptor.getIndexRow());
    Value c0  = peelUnrealized(adaptor.getIndexCol());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TEXTRACT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, r0, c0});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tfillpad lowering -> TFILLPAD_EXPAND(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadToEmitC : public OpConversionPattern<pto::FillPadOp_DPS> {
  using OpConversionPattern<pto::FillPadOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::FillPadOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD_EXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// pto.tgather lowering
// - Index form: TGATHER(dst, src0, indices)
// - Mask form : TGATHER<dstTileTok, srcTileTok, pto::MaskPattern::Pxxxx>(dst, src0)
//===----------------------------------------------------------------------===//

static std::string maskPatternTok(mlir::pto::MaskPatternAttr a) {

  auto v = a.getValue(); // enum
  return (std::string("pto::MaskPattern::") + mlir::pto::stringifyMaskPattern(v).str());
}

struct PTOGatherToEmitC : public OpConversionPattern<pto::GatherOp_DPS> {
  using OpConversionPattern<pto::GatherOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GatherOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst  = peelUnrealized(adaptor.getDst());
    Value src0 = peelUnrealized(adaptor.getSrc());

    // Case 1: index-based TGATHER(dst, src0, indices)
    if (Value idx = adaptor.getIndices()) {
      idx = peelUnrealized(idx);

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TGATHER",
          /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
          /*operands=*/ValueRange{dst, src0, idx});

      rewriter.eraseOp(op);
      return success();
    }

    // Case 2: mask-pattern TGATHER<DstT, SrcT, MaskPattern::P0101>(dst, src0)
    auto mp = op.getMaskPatternAttr();
    if (!mp)
      return rewriter.notifyMatchFailure(op, "expected maskPattern when indices is absent");

    auto getOpaqueTok = [&](Value v, StringRef name) -> FailureOr<std::string> {
      if (auto ot = v.getType().dyn_cast<emitc::OpaqueType>())
        return ot.getValue().str();
      return rewriter.notifyMatchFailure(op, (name + " must be emitc::OpaqueType (tile)").str());
    };

    auto dstTokOr = getOpaqueTok(dst, "dst");
    auto srcTokOr = getOpaqueTok(src0, "src0");
    if (failed(dstTokOr) || failed(srcTokOr))
      return failure();

    // mp is an EnumAttr; stringify name is "P0101" etc.
    // We emit MaskPattern::P0101 (because generated C++ has `using namespace pto;`)
    std::string mpTok = std::string("MaskPattern::") +
                        mlir::pto::stringifyMaskPattern(mp.getValue()).str();

    auto targs = rewriter.getArrayAttr({
        emitc::OpaqueAttr::get(ctx, *dstTokOr),
        emitc::OpaqueAttr::get(ctx, *srcTokOr),
        emitc::OpaqueAttr::get(ctx, mpTok),
    });

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHER",
        /*args=*/ArrayAttr{},
        /*templateArgs=*/targs,
        /*operands=*/ValueRange{dst, src0});

    rewriter.eraseOp(op);
    return success();
  }
};


struct PTOGatherbToEmitC : public OpConversionPattern<pto::GatherbOp_DPS> {
  using OpConversionPattern<pto::GatherbOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GatherbOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src     = peelUnrealized(adaptor.getSrc());
    Value offsets = peelUnrealized(adaptor.getOffsets());
    Value dst     = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TGATHERB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src, offsets});

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TLOG lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOLogToEmitC : public OpConversionPattern<pto::LogOp_DPS> {
  using OpConversionPattern<pto::LogOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::LogOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TLOG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};



//===----------------------------------------------------------------------===//
// TLRELU lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOLReluToEmitC : public OpConversionPattern<pto::LReluOp_DPS> {
	  using OpConversionPattern<pto::LReluOp_DPS>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::LReluOp_DPS op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();
	
	    Value src = peelUnrealized(adaptor.getSrc());
	    Value slope = peelUnrealized(adaptor.getSlope());
	    Value dst = peelUnrealized(adaptor.getDst());

            SmallVector<Value, 3> operands{dst, src, slope};

	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TLRELU",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
	        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAX lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMaxToEmitC : public OpConversionPattern<pto::MaxOp_DPS> {
  using OpConversionPattern<pto::MaxOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MaxOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMAXS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

	struct PTOMaxSToEmitC : public OpConversionPattern<pto::MaxSOp_DPS> {
	  using OpConversionPattern<pto::MaxSOp_DPS>::OpConversionPattern;
	
	  LogicalResult matchAndRewrite(pto::MaxSOp_DPS op, OpAdaptor adaptor,
	                                ConversionPatternRewriter &rewriter) const override {
	    auto loc = op.getLoc();
	
	    Value src0 = peelUnrealized(adaptor.getSrc0());
	    Value scalar = peelUnrealized(adaptor.getScalar());
	    Value dst  = peelUnrealized(adaptor.getDst());

	    SmallVector<Value, 3> operands{dst, src0, scalar};
	    rewriter.create<emitc::CallOpaqueOp>(
	        loc, TypeRange{}, "TMAXS",
	        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// TMIN lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinToEmitC : public OpConversionPattern<pto::MinOp_DPS> {
  using OpConversionPattern<pto::MinOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MinOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TMINS lowering to EmitC (fix APFloat -> FloatAttr)  (PTOConvert.cpp)
//===----------------------------------------------------------------------===//

struct PTOMinsToEmitC : public OpConversionPattern<pto::MinsOp_DPS> {
  using OpConversionPattern<pto::MinsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MinsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMINS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TMOV DPS/memref op -> EmitC)
//===----------------------------------------------------------------------===//

struct PTOMovToEmitC : public OpConversionPattern<pto::MovDpsOp> {
  using OpConversionPattern<pto::MovDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MovDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMOV_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMovFPToEmitC : public OpConversionPattern<pto::MovFPOp_DPS> {
  using OpConversionPattern<pto::MovFPOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MovFPOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());

    // TMOV_FP<DstTileData, AccTile, FbTile>(dstTileData, cTile, fbTile)
    ArrayAttr templateArgs;
    auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
    auto srcOT = src.getType().dyn_cast<emitc::OpaqueType>();
    auto fpOT  = fp.getType().dyn_cast<emitc::OpaqueType>();
    if (dstOT && srcOT && fpOT) {
      templateArgs = rewriter.getArrayAttr({
          emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, srcOT.getValue().str()),
          emitc::OpaqueAttr::get(ctx, fpOT.getValue().str()),
      });
    } else {
      templateArgs = ArrayAttr{};
    }

    SmallVector<Value, 3> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs,
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMRGSORT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMrgSortToEmitC : public OpConversionPattern<pto::MrgSortOp_DPS> {
  using OpConversionPattern<pto::MrgSortOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MrgSortOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    if (op.isFormat1()) {
      Value src = peelUnrealized(adaptor.getSrcs().front());
      Value dst = peelUnrealized(adaptor.getDsts().front());
      Value blockLen = peelUnrealized(adaptor.getBlockLen());

      SmallVector<Value, 3> operands{dst, src, blockLen};
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          ArrayAttr{}, ArrayAttr{}, operands);
    } else if (op.isFormat2()) {
      // pto-isa API:
      //   TMRGSORT<DstTile, TmpTile, Src0, Src1, Src2, Src3, exhausted>(
      //       dst, executedNumList, tmp, src0, src1, src2, src3);
      auto *ctx = rewriter.getContext();

      Value dst = peelUnrealized(adaptor.getDsts()[0]);
      Value tmp = peelUnrealized(adaptor.getDsts()[1]);
      Value excuted = peelUnrealized(adaptor.getExcuted());

      SmallVector<Value, 4> srcs;
      srcs.reserve(4);
      for (Value v : adaptor.getSrcs())
        srcs.push_back(peelUnrealized(v));

      auto dstOT = dst.getType().dyn_cast<emitc::OpaqueType>();
      auto tmpOT = tmp.getType().dyn_cast<emitc::OpaqueType>();
      if (!dstOT || !tmpOT || srcs.size() != 4)
        return op.emitOpError("format2 expects (dst,tmp) tilebufs and exactly 4 srcs");

      SmallVector<Attribute, 8> targs;
      targs.reserve(7);
      targs.push_back(emitc::OpaqueAttr::get(ctx, dstOT.getValue().str()));
      targs.push_back(emitc::OpaqueAttr::get(ctx, tmpOT.getValue().str()));
      for (Value v : srcs) {
        auto ot = v.getType().dyn_cast<emitc::OpaqueType>();
        if (!ot)
          return op.emitOpError("format2 expects tilebuf srcs");
        targs.push_back(emitc::OpaqueAttr::get(ctx, ot.getValue().str()));
      }
      targs.push_back(emitc::OpaqueAttr::get(ctx, op.getExhausted() ? "true" : "false"));
      ArrayAttr templateArgs = rewriter.getArrayAttr(targs);

      SmallVector<Value, 7> operands{dst, excuted, tmp};
      operands.append(srcs.begin(), srcs.end());

      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TMRGSORT",
          /*args=*/ArrayAttr{}, /*templateArgs=*/templateArgs, operands);
    } else {
      return op.emitOpError("unsupported mrgsort_dps format");
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulToEmitC : public OpConversionPattern<pto::MulOp_DPS> {
  using OpConversionPattern<pto::MulOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MulOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TMULS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOMulsToEmitC : public OpConversionPattern<pto::MulsOp_DPS> {
  using OpConversionPattern<pto::MulsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MulsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc0());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());

    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMULS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNEG DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONegToEmitC : public OpConversionPattern<pto::NegOp_DPS> {
  using OpConversionPattern<pto::NegOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::NegOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNEG",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TNOT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTONotToEmitC : public OpConversionPattern<pto::NotOp_DPS> {
  using OpConversionPattern<pto::NotOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::NotOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TNOT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrToEmitC : public OpConversionPattern<pto::OrOp_DPS> {
  using OpConversionPattern<pto::OrOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::OrOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOOrsToEmitC : public OpConversionPattern<pto::OrsOp_DPS> {
  using OpConversionPattern<pto::OrsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::OrsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value dst  = peelUnrealized(adaptor.getDst());
    // NOTE: The conversion type system may materialize integers as emitc.opaque
    // (e.g. "int32_t"). For EmitC call emission we can pass the scalar through
    // directly without arith casts here.
    Value s = adaptor.getScalar();

    SmallVector<Value, 3> operands{dst, src0, s};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTADD DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartAddToEmitC : public OpConversionPattern<pto::PartAddOp_DPS> {
  using OpConversionPattern<pto::PartAddOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PartAddOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMaxToEmitC : public OpConversionPattern<pto::PartMaxOp_DPS> {
  using OpConversionPattern<pto::PartMaxOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PartMaxOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPARTMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPartMinToEmitC : public OpConversionPattern<pto::PartMinOp_DPS> {
  using OpConversionPattern<pto::PartMinOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PartMinOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPARTMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TPRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOPreluToEmitC : public OpConversionPattern<pto::PreluOp_DPS> {
  using OpConversionPattern<pto::PreluOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PreluOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    // pto-isa TPRELU requires a tmp tile argument. Current NPU implementation
    // does not use tmp, so we safely pass dst as tmp for compatibility.
    SmallVector<Value, 4> operands{dst, src0, src1, dst};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRECIP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORecipToEmitC : public OpConversionPattern<pto::RecipOp_DPS> {
  using OpConversionPattern<pto::RecipOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RecipOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRECIP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRELU DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReluToEmitC : public OpConversionPattern<pto::ReluOp_DPS> {
  using OpConversionPattern<pto::ReluOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ReluOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRELU",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemToEmitC : public OpConversionPattern<pto::RemOp_DPS> {
  using OpConversionPattern<pto::RemOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RemOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TREMS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORemSToEmitC : public OpConversionPattern<pto::RemSOp_DPS> {
  using OpConversionPattern<pto::RemSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RemSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value scalar = peelUnrealized(adaptor.getScalar());
    
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TREMS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRESHAPE DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOReshapeToEmitC : public OpConversionPattern<pto::ReshapeOp_DPS> {
  using OpConversionPattern<pto::ReshapeOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ReshapeOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRESHAPE",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPAND DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandToEmitC : public OpConversionPattern<pto::RowExpandOp_DPS> {
  using OpConversionPattern<pto::RowExpandOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowExpandOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPAND",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDDIV DPS/memref op)
//===----------------------------------------------------------------------===//
// Helper: replace or erase based on whether op has results.
static void replaceOrEraseWithOpaqueCall(Operation *op,
                                        StringRef callee,
                                        ArrayRef<Value> args,
                                        ConversionPatternRewriter &rewriter) {
  TypeRange resultTypes = op->getResultTypes();
  auto call = rewriter.create<emitc::CallOpaqueOp>(
      op->getLoc(), resultTypes, callee, ArrayAttr{}, ArrayAttr{}, ValueRange(args));
  if (resultTypes.empty())
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, call.getResults());
}

// ---------- DPS ----------
struct PTOMatmulBiasDpsToTMATMUL_BIAS
    : public OpConversionPattern<pto::MatmulBiasDpsOp> {
  using OpConversionPattern<pto::MatmulBiasDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulBiasDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    // intrinsic: TMATMUL_BIAS(dst, a, b, bias)
    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOMatmulMxDpsToTMATMUL_MX
    : public OpConversionPattern<pto::MatmulMxDpsOp> {
  using OpConversionPattern<pto::MatmulMxDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulMxDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    // intrinsic: TMATMUL_MX(dst, a, a_scale, b, b_scale)
    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOMatmulMxAccDpsToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::MatmulMxAccDpsOp> {
  using OpConversionPattern<pto::MatmulMxAccDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulMxAccDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    // intrinsic: TMATMUL_MX_ACC(dst, c_in, a, a_scale, b, b_scale)
    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX_ACC",
                                {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOMatmulMxBiasDpsToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::MatmulMxBiasDpsOp> {
  using OpConversionPattern<pto::MatmulMxBiasDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::MatmulMxBiasDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    // intrinsic: TMATMUL_MX_BIAS(dst, a, a_scale, b, b_scale, bias)
    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX_BIAS",
                                {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

// ---------- Gemv DPS Ops ----------
struct PTOGemvBiasDpsToTGEMV_BIAS
    : public OpConversionPattern<pto::GemvBiasDpsOp> {
  using OpConversionPattern<pto::GemvBiasDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::GemvBiasDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TGEMV_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

// ---------- TOp ----------
struct PTOTGemvBiasToTGEMV_BIAS
    : public OpConversionPattern<pto::TGemvBiasOp> {
  using OpConversionPattern<pto::TGemvBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TGemvBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TGEMV_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulBiasToTMATMUL_BIAS
    : public OpConversionPattern<pto::TMatmulBiasOp> {
  using OpConversionPattern<pto::TMatmulBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a    = peelUnrealized(adaptor.getA());
    Value b    = peelUnrealized(adaptor.getB());
    Value bias = peelUnrealized(adaptor.getBias());
    Value dst  = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_BIAS",
                                {dst, a, b, bias}, rewriter);
    return success();
  }
};

struct PTOTMatmulMxToTMATMUL_MX
    : public OpConversionPattern<pto::TMatmulMxOp> {
  using OpConversionPattern<pto::TMatmulMxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX",
                                {dst, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMxAccToTMATMUL_MX_ACC
    : public OpConversionPattern<pto::TMatmulMxAccOp> {
  using OpConversionPattern<pto::TMatmulMxAccOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value cIn     = peelUnrealized(adaptor.getCIn());
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX_ACC",
                                {dst, cIn, a, aScale, b, bScale}, rewriter);
    return success();
  }
};

struct PTOTMatmulMxBiasToTMATMUL_MX_BIAS
    : public OpConversionPattern<pto::TMatmulMxBiasOp> {
  using OpConversionPattern<pto::TMatmulMxBiasOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TMatmulMxBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value a       = peelUnrealized(adaptor.getA());
    Value aScale  = peelUnrealized(adaptor.getAScale());
    Value b       = peelUnrealized(adaptor.getB());
    Value bScale  = peelUnrealized(adaptor.getBScale());
    Value bias    = peelUnrealized(adaptor.getBias());
    Value dst     = peelUnrealized(adaptor.getDst());

    replaceOrEraseWithOpaqueCall(op.getOperation(), "TMATMUL_MX_BIAS",
                                {dst, a, aScale, b, bScale, bias}, rewriter);
    return success();
  }
};

struct PTORowExpandDivToEmitC : public OpConversionPattern<pto::RowExpandDivOp_DPS> {
  using OpConversionPattern<pto::RowExpandDivOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowExpandDivOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDDIV",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDMUL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandMulToEmitC : public OpConversionPattern<pto::RowExpandMulOp_DPS> {
  using OpConversionPattern<pto::RowExpandMulOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowExpandMulOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDMUL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWEXPANDSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowExpandSubToEmitC : public OpConversionPattern<pto::RowExpandSubOp_DPS> {
  using OpConversionPattern<pto::RowExpandSubOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowExpandSubOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWEXPANDSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMAX DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMaxToEmitC : public OpConversionPattern<pto::RowMaxOp_DPS> {
  using OpConversionPattern<pto::RowMaxOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowMaxOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMAX",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWMIN DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowMinToEmitC : public OpConversionPattern<pto::RowMinOp_DPS> {
  using OpConversionPattern<pto::RowMinOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowMinOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWMIN",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TROWSUM DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORowSumToEmitC : public OpConversionPattern<pto::RowSumOp_DPS> {
  using OpConversionPattern<pto::RowSumOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RowSumOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TROWSUM",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TRSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTORsqrtToEmitC : public OpConversionPattern<pto::RsqrtOp_DPS> {
  using OpConversionPattern<pto::RsqrtOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::RsqrtOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 2> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TRSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSCATTER DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOScatterToEmitC : public OpConversionPattern<pto::ScatterOp_DPS> {
  using OpConversionPattern<pto::ScatterOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ScatterOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value idx = peelUnrealized(adaptor.getIndexes());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, idx};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSCATTER",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSEL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelToEmitC : public OpConversionPattern<pto::SelOp_DPS> {
  using OpConversionPattern<pto::SelOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SelOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value mask = peelUnrealized(adaptor.getMask());
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, mask, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSEL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSELS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSelSToEmitC : public OpConversionPattern<pto::SelsOp_DPS> {
  using OpConversionPattern<pto::SelsOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SelsOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value selectMode = peelUnrealized(adaptor.getSelectMode());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1, selectMode};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSELS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHL DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShlSToEmitC : public OpConversionPattern<pto::ShlOp_DPS> {
  using OpConversionPattern<pto::ShlOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ShlOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHL",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSHR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOShrSToEmitC : public OpConversionPattern<pto::ShrOp_DPS> {
  using OpConversionPattern<pto::ShrOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ShrOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst  = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering for TSHLS/TSHRS DPS: shift by scalar)
//===----------------------------------------------------------------------===//

struct PTOShlSConstToEmitC : public OpConversionPattern<pto::ShlSOp_DPS> {
  using OpConversionPattern<pto::ShlSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ShlSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHLS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOShrSConstToEmitC : public OpConversionPattern<pto::ShrSOp_DPS> {
  using OpConversionPattern<pto::ShrSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::ShrSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dst    = peelUnrealized(adaptor.getDst());
    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    SmallVector<Value, 3> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSHRS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSORT32 DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSORT32SToEmitC : public OpConversionPattern<pto::Sort32Op_DPS> {
  using OpConversionPattern<pto::Sort32Op_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::Sort32Op_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());
    Value idx = peelUnrealized(adaptor.getIdx());

    SmallVector<Value, 4> operands{dst, src, idx};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSORT32",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSQRT DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSqrtSToEmitC : public OpConversionPattern<pto::SqrtOp_DPS> {
  using OpConversionPattern<pto::SqrtOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SqrtOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSQRT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSTORE_FP DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOStoreFPSToEmitC : public OpConversionPattern<pto::StoreFPOp_DPS> {
  using OpConversionPattern<pto::StoreFPOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::StoreFPOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSTORE_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUB DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSToEmitC : public OpConversionPattern<pto::SubOp_DPS> {
  using OpConversionPattern<pto::SubOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SubOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubCSToEmitC : public OpConversionPattern<pto::SubCOp_DPS> {
  using OpConversionPattern<pto::SubCOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SubCOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBC yet.
    // Decompose: dst = src0 - src1 + src2
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUB",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, src1});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src2});

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSSToEmitC : public OpConversionPattern<pto::SubSOp_DPS> {
  using OpConversionPattern<pto::SubSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SubSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, scalar};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSUBSC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSubSCToEmitC : public OpConversionPattern<pto::SubSCOp_DPS> {
  using OpConversionPattern<pto::SubSCOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SubSCOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa does not provide NPU implementation for TSUBSC yet.
    // Decompose: dst = src0 - scalar + src1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, src0, scalar});
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TADD",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange{dst, dst, src1});

    rewriter.eraseOp(op);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXOR DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORToEmitC : public OpConversionPattern<pto::XOROp_DPS> {
  using OpConversionPattern<pto::XOROp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::XOROp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa TXOR requires a tmp tile argument. Current NPU implementation
    // does not use tmp, so we safely pass dst as tmp for compatibility.
    SmallVector<Value, 4> operands{dst, src0, src1, dst};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXOR",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
struct PTOTransToEmitC : public OpConversionPattern<pto::TransDpsOp> {
  using OpConversionPattern<pto::TransDpsOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TransDpsOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src, tmp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TTRANS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TXORS DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOXORSToEmitC : public OpConversionPattern<pto::XORSOp_DPS> {
  using OpConversionPattern<pto::XORSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::XORSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst = peelUnrealized(adaptor.getDst());

    // pto-isa TXORS requires a tmp tile argument. Current NPU implementation
    // does not use tmp, so we safely pass dst as tmp for compatibility.
    SmallVector<Value, 4> operands{dst, src0, scalar, dst};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TXORS",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};
  struct PTOPrintToTPRINT : public OpConversionPattern<pto::PrintOp_DPS> {
  using OpConversionPattern<pto::PrintOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::PrintOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());

    SmallVector<Value, 4> operands{src};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TPRINT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PTOConvert.cpp  (add lowering + patterns.add for TSYNC DPS/memref op)
//===----------------------------------------------------------------------===//

struct PTOSYNCToEmitC : public OpConversionPattern<pto::SYNCOp_DPS> {
  using OpConversionPattern<pto::SYNCOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SYNCOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value events = peelUnrealized(adaptor.getEvents());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, events};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSYNC",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

    rewriter.eraseOp(op);
    return success();
  }
};

// =============================================================================
// 2. BindTileOp Lowering (FIX: Trace back to physical address)
// =============================================================================
struct PTOBindTileToEmitC : public OpConversionPattern<pto::BindTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::BindTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 获取 Config
    auto configAttr = op.getConfigAttr();

    // [关键修复] 回溯寻找原始物理地址
    // BindTile 的输入 (op.getSource()) 通常是由 alloc -> pointer_cast 产生的。
    // 我们需要那个 pointer_cast 使用的原始整数地址 (I64)，而不是中间的 MemRef。
    
    SmallVector<Value> physAddrs;
    Value source = op.getSource();

    // 1. 尝试跳过可能存在的 Cast
    while (auto castOp = source.getDefiningOp<UnrealizedConversionCastOp>()) {
      source = castOp.getOperand(0);
    }

    // 2. 检查定义该 Value 的 Op 是否为 PointerCastOp
    if (auto upstreamCast = source.getDefiningOp<pto::PointerCastOp>()) {
      // 找到了！窃取它的输入操作数 (即原始 offset/address)
      auto upstreamOperands = upstreamCast.getAddrs();
      physAddrs.append(upstreamOperands.begin(), upstreamOperands.end());
    } else {
      // Fallback: 如果追溯不到 PointerCast (例如源头是函数参数 Function Argument)，
      // 那么只能使用当前的 source (此时它是指针类型)。
      // TASSIGN(tile, ptr) 通常也是合法的。
      physAddrs.push_back(adaptor.getSource());
    }

    Value vRow = op.getValidRow();
    Value vCol = op.getValidCol();

    // 创建 PointerCastOp
    // ODS 定义要求：ValueRange addrs, Value row, Value col, Attribute config
    rewriter.replaceOpWithNewOp<pto::PointerCastOp>(
        op, 
        op.getType(), 
        physAddrs,     
        vRow ? vRow : Value(), // 如果为空，传空 Value()，Builder 会处理
        vCol ? vCol : Value(),          
        configAttr
    );

    return success();
  }
};

// =============================================================================
// Arith CmpI -> EmitC Cmp
// =============================================================================
class ArithCmpIToEmitC : public OpConversionPattern<arith::CmpIOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // 将 arith.cmpi 转换为 emitc.cmp
    // 映射 Predicate: eq -> equal, slt -> less, etc.
    emitc::CmpPredicate emitcPred;
    const bool isUnsignedPred =
        op.getPredicate() == arith::CmpIPredicate::ult ||
        op.getPredicate() == arith::CmpIPredicate::ule ||
        op.getPredicate() == arith::CmpIPredicate::ugt ||
        op.getPredicate() == arith::CmpIPredicate::uge;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  emitcPred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  emitcPred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: emitcPred = emitc::CmpPredicate::ge; break;
      // ... 处理无符号比较 (ult, ule 等) ...
      case arith::CmpIPredicate::ult: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::ule: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: emitcPred = emitc::CmpPredicate::ge; break;
      default: return failure();
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    if (!resTy)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (isUnsignedPred) {
      Type opTy = op.getLhs().getType();
      auto intTy = dyn_cast<IntegerType>(opTy);
      const bool isIndex = isa<IndexType>(opTy);
      if (!intTy && !isIndex)
        return rewriter.notifyMatchFailure(
            op, "expected scalar integer or index operands");

      const unsigned bitWidth =
          intTy ? intTy.getWidth() : static_cast<unsigned>(kPTOIndexBitWidth);
      if (bitWidth != 1) {
        lhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, lhs, bitWidth);
        rhs = castSignlessIntToUnsignedSameWidth(rewriter, loc, rhs, bitWidth);
      }
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/resTy, // i1 -> bool/i1
        emitcPred,
        lhs,
        rhs
    );
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Section Op Lowering
//===----------------------------------------------------------------------===//
template <typename SectionOpTy>
struct SectionToEmitC : public OpConversionPattern<SectionOpTy> {
  using OpConversionPattern<SectionOpTy>::OpConversionPattern;

  std::string getMacroName() const {
    if (std::is_same<SectionOpTy, pto::SectionCubeOp>::value)
      return "__DAV_CUBE__";
    if (std::is_same<SectionOpTy, pto::SectionVectorOp>::value)
      return "__DAV_VEC__";
    return "UNKNOWN_MACRO";
  }

  LogicalResult
  matchAndRewrite(SectionOpTy op, typename SectionOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    std::string startMacro = "\n#if defined(" + getMacroName() + ")";
    rewriter.create<emitc::VerbatimOp>(loc, startMacro);

    Block &innerBlock = op.getBody().front();
    if (!innerBlock.empty()) {
      rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});
    }

    std::string endMacro = "#endif // " + getMacroName() + "\n";
    rewriter.create<emitc::VerbatimOp>(loc, endMacro);

    rewriter.eraseOp(op);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SCF Control-Flow Pre-Lowering
//
// EmitC translation supports `emitc.for`/`emitc.if` plus CFG-style
// `cf.br`/`cf.cond_br`. Upstream SCFToEmitC patterns only cover `scf.for` and
// `scf.if`, so we pre-lower some SCF ops into those supported forms.
//===----------------------------------------------------------------------===//

namespace {

static bool isTriviallyInlineableExecuteRegion(scf::ExecuteRegionOp op) {
  Region &r = op.getRegion();
  if (!r.hasOneBlock())
    return false;
  Block &b = r.front();
  return isa_and_nonnull<scf::YieldOp>(b.getTerminator());
}

static bool needsWholeFunctionSCFToCF(func::FuncOp func) {
  bool needs = false;
  func.walk([&](Operation *op) {
    if (!isa<scf::WhileOp, scf::IndexSwitchOp, scf::ExecuteRegionOp>(op))
      return WalkResult::advance();
    Operation *parentOp = op->getParentOp();

    // `scf.execute_region` can legally appear in single-block parents. Only
    // require whole-function SCFToCF if we need to lower it into CFG blocks
    // (multi-block region / non-trivial terminators).
    if (auto exec = dyn_cast<scf::ExecuteRegionOp>(op)) {
      if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>() &&
          !isTriviallyInlineableExecuteRegion(exec)) {
        needs = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      needs = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return needs;
}

// scf.execute_region is semantically just an inlined region producing results
// via scf.yield. Inline it to the parent block to avoid extra lowering needs.
struct SCFExecuteRegionInline
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Block &innerBlock = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(innerBlock.getTerminator());
    if (!yield)
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Move the body operations before the execute_region op.
    rewriter.inlineBlockBefore(&innerBlock, op.getOperation(), ValueRange{});

    // Replace execute_region results with yielded values, then erase the yield.
    rewriter.replaceOp(op, yield.getOperands());
    rewriter.eraseOp(yield);
    return success();
  }
};

// Lower scf.execute_region into CFG blocks with cf.br/cf.cond_br by inlining the
// region blocks into the parent region and rewriting scf.yield to branch into a
// continuation block carrying results.
//
// Note: This requires the parent region to allow multiple blocks (e.g. the
// function body CFG region). For execute_region nested in single-block regions
// (scf.for/scf.if), run SCFToCF first to eliminate the single-block constraint.
struct SCFExecuteRegionToCF : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp op,
                                PatternRewriter &rewriter) const override {
    if (isTriviallyInlineableExecuteRegion(op))
      return rewriter.notifyMatchFailure(op, "trivially inlineable");

    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.execute_region inside a single-block parent region");
    }

    if (op.getRegion().empty())
      return rewriter.notifyMatchFailure(op, "expected non-empty region");

    Location loc = op.getLoc();
    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    // Split the parent block so we can branch to a continuation block with phi
    // arguments for the execute_region results.
    auto execIt = Block::iterator(op.getOperation());
    Block *continueBlock = rewriter.splitBlock(curBlock, std::next(execIt));

    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(contArgs[it.index()]);

    // Capture blocks before moving the region.
    SmallVector<Block *> movedBlocks;
    movedBlocks.reserve(op.getRegion().getBlocks().size());
    for (Block &b : op.getRegion())
      movedBlocks.push_back(&b);
    Block *entryBlock = &op.getRegion().front();

    // Inline the execute_region blocks into the parent region right before the
    // continuation block.
    rewriter.inlineRegionBefore(op.getRegion(), *parentRegion,
                                continueBlock->getIterator());

    // Replace all scf.yield terminators with a branch to the continuation.
    for (Block *b : movedBlocks) {
      auto yield = dyn_cast<scf::YieldOp>(b->getTerminator());
      if (!yield)
        continue;
      rewriter.setInsertionPoint(yield);
      rewriter.create<cf::BranchOp>(loc, continueBlock, yield.getOperands());
      rewriter.eraseOp(yield);
    }

    // Replace execute_region itself with a branch to the inlined entry block.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, entryBlock, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.index_switch into CFG blocks with cf.cond_br/cf.br so that we can
// avoid `scf.if` result materialization quirks (and avoid relying on cf.switch,
// which is not supported by EmitC C++ translation).
struct SCFIndexSwitchToCF : public OpRewritePattern<scf::IndexSwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  static LogicalResult cloneYieldingBlockAndBranchTo(
      PatternRewriter &rewriter, Location loc, Block &srcBlock, Block *destBlock,
      Block *continueBlock) {
    rewriter.setInsertionPointToEnd(destBlock);

    IRMapping mapping;
    for (Operation &inner : srcBlock.without_terminator())
      rewriter.clone(inner, mapping);

    auto yield = dyn_cast<scf::YieldOp>(srcBlock.getTerminator());
    if (!yield)
      return failure();

    SmallVector<Value> yieldOperands;
    yieldOperands.reserve(yield.getNumOperands());
    for (Value v : yield.getOperands())
      yieldOperands.push_back(mapping.lookupOrDefault(v));

    rewriter.create<cf::BranchOp>(loc, continueBlock, yieldOperands);
    return success();
  }

  LogicalResult matchAndRewrite(scf::IndexSwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.index_switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    // Split the parent block so we can branch to a continuation block with phi
    // arguments for the switch results.
    auto switchIt = Block::iterator(op.getOperation());
    Block *continueBlock = rewriter.splitBlock(curBlock, std::next(switchIt));

    SmallVector<BlockArgument> contArgs;
    contArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      contArgs.push_back(continueBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(contArgs[it.index()]);

    unsigned numCases = op.getCases().size();
    auto insertPt = continueBlock->getIterator();

    SmallVector<Block *> checkBlocks;
    SmallVector<Block *> caseBlocks;
    checkBlocks.reserve(numCases);
    caseBlocks.reserve(numCases);

    // Create check blocks for each case: check_i compares selector to case_i.
    for (unsigned i = 0; i < numCases; ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));

    // Create one block for default and one block per case to execute the body.
    Block *defaultBlock = rewriter.createBlock(parentRegion, insertPt);
    for (unsigned i = 0; i < numCases; ++i)
      caseBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));

    Value selector = op.getArg();
    auto cases = op.getCases();

    // Fill check blocks with chained comparisons.
    for (unsigned i = 0; i < numCases; ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);
      Value caseVal = rewriter.create<arith::ConstantIndexOp>(loc, cases[i]);
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, selector, caseVal);
      Block *falseDest = (i + 1 < numCases) ? checkBlocks[i + 1] : defaultBlock;
      rewriter.create<cf::CondBranchOp>(loc, cond, caseBlocks[i], ValueRange{},
                                        falseDest, ValueRange{});
    }

    // Fill case blocks and default block with cloned bodies + branch to cont.
    for (unsigned i = 0; i < numCases; ++i) {
      if (failed(cloneYieldingBlockAndBranchTo(
              rewriter, loc, op.getCaseBlock(i), caseBlocks[i], continueBlock)))
        return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");
    }
    if (failed(cloneYieldingBlockAndBranchTo(rewriter, loc, op.getDefaultBlock(),
                                             defaultBlock, continueBlock)))
      return rewriter.notifyMatchFailure(op, "expected scf.yield terminator");

    // Replace the original switch op with a branch into the check chain.
    Block *entryDest = numCases ? checkBlocks[0] : defaultBlock;
    rewriter.setInsertionPointAfter(op);
    rewriter.create<cf::BranchOp>(loc, entryDest, ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower scf.while into CFG blocks with cf.br/cf.cond_br.
//
// Note: This requires the parent region to allow multiple blocks. In
// particular, scf.if/scf.for regions are single-block and cannot contain this
// lowering.
struct SCFWhileToCF : public OpRewritePattern<scf::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::WhileOp op,
                                PatternRewriter &rewriter) const override {
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower scf.while inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();

    // Only support the common structured form where the while results are used
    // in the same block after the op.
    for (Value res : op.getResults()) {
      for (auto &use : res.getUses()) {
        if (use.getOwner()->getBlock() != curBlock)
          return rewriter.notifyMatchFailure(
              op, "unsupported: while results used outside the parent block");
      }
    }

    auto loc = op.getLoc();
    auto whileIt = Block::iterator(op.getOperation());
    Block *afterWhileBlock = rewriter.splitBlock(curBlock, std::next(whileIt));

    // Add block args to carry while results into the continuation block.
    SmallVector<Value> exitArgs;
    exitArgs.reserve(op.getNumResults());
    for (Type t : op.getResultTypes())
      exitArgs.push_back(afterWhileBlock->addArgument(t, loc));

    for (auto it : llvm::enumerate(op.getResults()))
      it.value().replaceAllUsesWith(exitArgs[it.index()]);

    // Create the CFG blocks before the continuation block.
    Region *parentRegion = curBlock->getParent();
    auto insertPt = afterWhileBlock->getIterator();

    // Header block arguments match the while init operands.
    SmallVector<Type> headerArgTypes;
    for (Value v : op.getInits())
      headerArgTypes.push_back(v.getType());
    SmallVector<Location> headerArgLocs(headerArgTypes.size(), loc);
    Block *headerBlock =
        rewriter.createBlock(parentRegion, insertPt, headerArgTypes,
                             headerArgLocs);

    // Body block arguments match the "after" region arguments.
    Block &afterRegionBlock = op.getAfter().front();
    SmallVector<Type> bodyArgTypes(afterRegionBlock.getArgumentTypes().begin(),
                                  afterRegionBlock.getArgumentTypes().end());
    SmallVector<Location> bodyArgLocs(bodyArgTypes.size(), loc);
    insertPt = afterWhileBlock->getIterator();
    Block *bodyBlock =
        rewriter.createBlock(parentRegion, insertPt, bodyArgTypes, bodyArgLocs);

    // Move the before/after region bodies into the new CFG blocks.
    rewriter.mergeBlocks(&op.getBefore().front(), headerBlock,
                         headerBlock->getArguments());
    rewriter.mergeBlocks(&afterRegionBlock, bodyBlock, bodyBlock->getArguments());

    // Replace scf.condition in the header with cf.cond_br.
    {
      auto condOp = cast<scf::ConditionOp>(headerBlock->getTerminator());
      rewriter.setInsertionPoint(condOp);
      rewriter.create<cf::CondBranchOp>(loc, condOp.getCondition(),
                                        /*trueDest=*/bodyBlock,
                                        /*trueOperands=*/condOp.getArgs(),
                                        /*falseDest=*/afterWhileBlock,
                                        /*falseOperands=*/condOp.getArgs());
      rewriter.eraseOp(condOp);
    }

    // Replace scf.yield in the body with cf.br back to the header.
    {
      auto yieldOp = cast<scf::YieldOp>(bodyBlock->getTerminator());
      rewriter.setInsertionPoint(yieldOp);
      rewriter.create<cf::BranchOp>(loc, headerBlock, yieldOp.getOperands());
      rewriter.eraseOp(yieldOp);
    }

    // Replace scf.while itself with a branch to the header.
    rewriter.setInsertionPoint(op);
    rewriter.create<cf::BranchOp>(loc, headerBlock, op.getInits());
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower cf.switch into chained comparisons and cf.cond_br/cf.br.
//
// EmitC C++ translation currently supports cf.br/cf.cond_br, but not cf.switch.
struct CFSwitchToCondBr : public OpRewritePattern<cf::SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cf::SwitchOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Operation *parentOp = op->getParentOp();
    if (parentOp && parentOp->hasTrait<OpTrait::SingleBlock>()) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower cf.switch inside a single-block parent region");
    }

    Block *curBlock = op->getBlock();
    Region *parentRegion = curBlock->getParent();

    Value flag = op.getFlag();
    auto flagTy = dyn_cast<IntegerType>(flag.getType());
    if (!flagTy)
      return rewriter.notifyMatchFailure(op, "expected integer switch flag");

    SmallVector<Value> defaultOperands(op.getDefaultOperands().begin(),
                                       op.getDefaultOperands().end());
    Block *defaultDest = op.getDefaultDestination();

    SmallVector<Block *> caseDests(op.getCaseDestinations().begin(),
                                   op.getCaseDestinations().end());
    SmallVector<SmallVector<Value>> caseOperands;
    caseOperands.reserve(caseDests.size());
    for (auto range : op.getCaseOperands())
      caseOperands.emplace_back(range.begin(), range.end());

    if (caseDests.empty()) {
      rewriter.replaceOpWithNewOp<cf::BranchOp>(op, defaultDest, defaultOperands);
      return success();
    }

    std::optional<DenseIntElementsAttr> caseValuesAttr = op.getCaseValues();
    if (!caseValuesAttr)
      return rewriter.notifyMatchFailure(op, "missing case_values");

    SmallVector<APInt> caseValues;
    for (APInt v : caseValuesAttr->getValues<APInt>())
      caseValues.push_back(v);

    if (caseValues.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_values/destinations mismatch");
    if (caseOperands.size() != caseDests.size())
      return rewriter.notifyMatchFailure(op, "case_operands/destinations mismatch");

    // Insert check blocks right after the current block.
    auto insertPt = std::next(curBlock->getIterator());
    SmallVector<Block *> checkBlocks;
    checkBlocks.reserve(caseDests.size());
    for (size_t i = 0; i < caseDests.size(); ++i)
      checkBlocks.push_back(rewriter.createBlock(parentRegion, insertPt));

    // Fill each check block with:
    //   if (flag == caseVal_i) goto caseDest_i else goto nextCheck/default.
    for (size_t i = 0; i < caseDests.size(); ++i) {
      rewriter.setInsertionPointToEnd(checkBlocks[i]);

      APInt caseVal = caseValues[i];
      if (caseVal.getBitWidth() != flagTy.getWidth()) {
        return rewriter.notifyMatchFailure(
            op, "case value bitwidth doesn't match flag type");
      }

      Value caseConst = rewriter.create<arith::ConstantOp>(
          loc, flagTy, rewriter.getIntegerAttr(flagTy, caseVal));
      Value cond = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, flag, caseConst);

      Block *falseDest =
          (i + 1 < checkBlocks.size()) ? checkBlocks[i + 1] : defaultDest;
      ValueRange falseOperands =
          (i + 1 < checkBlocks.size()) ? ValueRange{} : ValueRange(defaultOperands);

      rewriter.create<cf::CondBranchOp>(loc, cond,
                                        /*trueDest=*/caseDests[i],
                                        /*trueOperands=*/caseOperands[i],
                                        /*falseDest=*/falseDest,
                                        /*falseOperands=*/falseOperands);
    }

    // Replace the switch terminator with a branch into the first check block.
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, checkBlocks.front(),
                                              ValueRange{});
    return success();
  }
};

} // namespace

static void populatePTOToEmitCPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter,
                                       MLIRContext *ctx,
                                       DataFlowSolver &solver) {
  (void)solver;
  patterns.add<ArithCmpIToEmitC>(typeConverter, ctx);
  patterns.add<PTOBindTileToEmitC>(typeConverter, ctx);
  patterns.add<PTOSetFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubCSToEmitC>(typeConverter, ctx);
  patterns.add<PTOWaitFlagToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSYNCToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSToEmitC>(typeConverter, ctx);
  patterns.add<PTOXORToEmitC>(typeConverter, ctx);
  patterns.add<PTOReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOScatterToEmitC>(typeConverter, ctx);
  patterns.add<PTOStoreFPSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSubSSToEmitC>(typeConverter, ctx);
  patterns.add<PTOSqrtSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTransToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSConstToEmitC>(typeConverter, ctx);
  patterns.add<PTOSORT32SToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTORsqrtToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandMulToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandDivToEmitC>(typeConverter, ctx);
  patterns.add<PTORowSumToEmitC>(typeConverter, ctx);
  patterns.add<PTORowMinToEmitC>(typeConverter, ctx);
  patterns.add<PTODivSToEmitC>(typeConverter, ctx);
  patterns.add<PTOTDivSToEmitC>(typeConverter, ctx);
  patterns.add<PTORemToEmitC>(typeConverter, ctx);
  patterns.add<PTOReshapeToEmitC>(typeConverter, ctx);
  patterns.add<PTORecipToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulsToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpToEmitC>(typeConverter, ctx);
  patterns.add<PTOPreluToEmitC>(typeConverter, ctx);
  patterns.add<PTORemSToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTONotToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOExpandsToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrToEmitC>(typeConverter, ctx);
  patterns.add<PTOPartAddToEmitC>(typeConverter, ctx);
  patterns.add<PTOExtractToEmitC>(typeConverter, ctx);
  patterns.add<PTOFillPadToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherToEmitC>(typeConverter, ctx);
  patterns.add<PTOGatherbToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovFPToEmitC>(typeConverter, ctx);
  patterns.add<PTOOrsToEmitC>(typeConverter, ctx);
  patterns.add<PTOLogToEmitC>(typeConverter, ctx);
  patterns.add<FuncToEmitC>(typeConverter, ctx);
  patterns.add<PTOMovToEmitC>(typeConverter, ctx);
  patterns.add<ArithConstantToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulSIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulUIExtendedToEmitC>(typeConverter, ctx);
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTOCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL, PTOGetValToGETVAL,
               PTOLoadScalarToEmitC, PTOStoreScalarToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
  patterns.add<ArithSubIToEmitC>(typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::AndIOp, emitc::BitwiseAndOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::OrIOp, emitc::BitwiseOrOp>>(
      typeConverter, ctx);
  patterns.add<ArithUnsignedBitwiseBinaryToEmitC<arith::XOrIOp, emitc::BitwiseXorOp>>(
      typeConverter, ctx);
  patterns.add<ArithShiftLeftToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithShiftRightSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCeilDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithFloorDivSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithRemSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithNegFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::AddFOp, emitc::AddOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::SubFOp, emitc::SubOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::MulFOp, emitc::MulOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithSimpleBinaryToEmitC<arith::DivFOp, emitc::DivOp>>(typeConverter,
                                                                     ctx);
  patterns.add<ArithRemFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaximumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinimumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMaxNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithMinNumFToEmitC>(typeConverter, ctx);
  patterns.add<ArithSelectToEmitC>(typeConverter, ctx);
  patterns.add<ArithCmpFToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithExtSIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<ArithUIToFPToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<ArithFPToUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithCastToEmitC<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<ArithIndexCastUIToEmitC>(typeConverter, ctx);
  patterns.add<ArithBitcastToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddSToTADDS>(typeConverter, ctx);
  patterns.add<PTOColExpandToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMinToEmitC>(typeConverter, ctx);
  patterns.add<PTOLoadDpsToTLOAD>(typeConverter, ctx);
  patterns.add<PTOStoreDpsToTSTORE>(typeConverter, ctx);
  patterns.add<PTOMScatterToMSCATTER>(typeConverter, ctx);
  patterns.add<PTOAddCToTADDC>(typeConverter, ctx);
  patterns.add<PTOMinsToEmitC>(typeConverter, ctx);
  patterns.add<PTOAddfDpsToTADD>(typeConverter, ctx);
  patterns.add<PTOMGatherToMGATHER>(typeConverter, ctx);
  patterns.add<PTOMatmulDpsToTMATMUL>(typeConverter, ctx);
  patterns.add<PTOMatmulAccDpsToTMATMULACC>(typeConverter, ctx);
  patterns.add<PTOGemvDpsToTGEMV>(typeConverter, ctx);
  patterns.add<PTOGemvAccDpsToTGEMVACC>(typeConverter, ctx);
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOAbsToTABS>(typeConverter, ctx);
  patterns.add<PTOAddToTADD>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<ArithTruncIToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionCubeOp>>(typeConverter, ctx);
  patterns.add<SectionToEmitC<pto::SectionVectorOp>>(typeConverter, ctx);
  patterns.add<PTOGetBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockIdxToEmitC>(typeConverter, ctx);
  patterns.add<PTOGetSubBlockNumToEmitC>(typeConverter, ctx);
  patterns.add<PTOPrintToTPRINT>(typeConverter, ctx);
  patterns.add<
    PTOMatmulBiasDpsToTMATMUL_BIAS,
    PTOMatmulMxDpsToTMATMUL_MX,
    PTOMatmulMxAccDpsToTMATMUL_MX_ACC,
    PTOMatmulMxBiasDpsToTMATMUL_MX_BIAS,
    PTOTMatmulBiasToTMATMUL_BIAS,
    PTOTMatmulMxToTMATMUL_MX,
    PTOTMatmulMxAccToTMATMUL_MX_ACC,
    PTOTMatmulMxBiasToTMATMUL_MX_BIAS,
    PTOGemvBiasDpsToTGEMV_BIAS,
    PTOTGemvBiasToTGEMV_BIAS,
    PTOBarrierToEmitC
  >(typeConverter, ctx);

  patterns.add<ReturnToEmitC>(typeConverter, ctx);

  populateSCFToEmitCConversionPatterns(patterns);
  // Keep CFG-style branches type-consistent when block argument types are
  // converted (e.g. after lowering scf.while to cf.br/cf.cond_br).
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct EmitPTOManualPass
    : public PassWrapper<EmitPTOManualPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EmitPTOManualPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    memref::MemRefDialect, affine::AffineDialect,
                    mlir::cf::ControlFlowDialect, mlir::pto::PTODialect>();
  }

	  void runOnOperation() override {
	    llvm::errs() << "DEBUG: Start PTOToEmitC Pass\n";
	    MLIRContext *ctx = &getContext();
	    ModuleOp mop = getOperation();

	    // 1. 插入头文件
	    auto loc = mop->getLoc();
	    OpBuilder builder(ctx);
	    builder.setInsertionPointToStart(mop.getBody());
	    builder.create<emitc::IncludeOp>(
	        loc, builder.getStringAttr("pto/pto-inst.hpp"), /*isAngled=*/nullptr);
	    builder.create<emitc::VerbatimOp>(
	        loc, builder.getStringAttr("using namespace pto;"));
	
	    // Only inject the bitcast helper when we actually lower ops that need it
	    // (e.g. arith.bitcast or arith.maximumf/minimumf tie-breaking on zeros).
	    bool needsBitcastHelper = false;
	    mop.walk([&](Operation *op) {
	      if (isa<arith::BitcastOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
	        needsBitcastHelper = true;
	        return WalkResult::interrupt();
	      }
	      return WalkResult::advance();
	    });
	    if (needsBitcastHelper) {
	      builder.create<emitc::VerbatimOp>(
	          loc, builder.getStringAttr(R"cpp(
		template <typename To, typename From>
		static inline To ptoas_bitcast(From from) {
		  static_assert(sizeof(To) == sizeof(From), "ptoas_bitcast: size mismatch");
		  To to;
		  __builtin_memcpy(&to, &from, sizeof(To));
		  return to;
		}
		)cpp"));
	    }

	    // 1.5 Pre-lower SCF constructs not handled by SCFToEmitC.
	    {
	      // scf.while / scf.index_switch are lowered via CFG blocks. This is not
      // possible inside ops that require single-block regions (e.g. scf.for /
      // scf.if). If we see such nesting, lower the entire function to the
      // ControlFlow dialect first.
      bool needsAnySCFToCF = false;
      for (auto func : mop.getOps<func::FuncOp>()) {
        if (needsWholeFunctionSCFToCF(func)) {
          needsAnySCFToCF = true;
          break;
        }
      }
      if (needsAnySCFToCF) {
        RewritePatternSet scfToCfPatterns(ctx);
        populateSCFToControlFlowConversionPatterns(scfToCfPatterns);
        FrozenRewritePatternSet frozenSCFToCF(std::move(scfToCfPatterns));

        ConversionTarget scfToCfTarget(*ctx);
        // Only eliminate the single-block SCF constructs; we'll pre-lower
        // scf.while/index_switch/execute_region ourselves afterwards.
        scfToCfTarget.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp,
                                   scf::ParallelOp>();
        scfToCfTarget.markUnknownOpDynamicallyLegal(
            [](Operation *) { return true; });

        for (auto func : mop.getOps<func::FuncOp>()) {
          if (!needsWholeFunctionSCFToCF(func))
            continue;
          if (failed(applyPartialConversion(func, scfToCfTarget,
                                            frozenSCFToCF))) {
            func.emitError()
                << "failed to lower nested SCF to ControlFlow (SCFToCF)";
            return signalPassFailure();
          }
        }
      }

      RewritePatternSet scfLoweringPatterns(ctx);
      scfLoweringPatterns.add<SCFExecuteRegionInline, SCFExecuteRegionToCF,
                              SCFIndexSwitchToCF,
                              SCFWhileToCF, CFSwitchToCondBr>(ctx);
      (void)applyPatternsAndFoldGreedily(mop, std::move(scfLoweringPatterns));

      bool hasUnsupportedSCF = false;
      mop.walk([&](Operation *op) {
        if (isa<scf::ExecuteRegionOp, scf::IndexSwitchOp, scf::WhileOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError() << "Unsupported SCF op remained after pre-lowering";
          return WalkResult::interrupt();
        }
        if (isa<cf::SwitchOp>(op)) {
          hasUnsupportedSCF = true;
          op->emitError()
              << "Unsupported CF op remained after pre-lowering: cf.switch";
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasUnsupportedSCF)
        return signalPassFailure();
    }

    // 2. 配置转换目标
    PTOToEmitCTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>(); 
    
    // If we introduced CFG branches (e.g. from scf.while), make sure they are
    // updated to use legalized operand types.
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                  typeConverter);
        });

    // [关键] 允许 Cast 存在，最后统一清理
    target.addLegalOp<UnrealizedConversionCastOp>(); 

    target.addIllegalOp<func::ReturnOp>();
    target.addIllegalOp<func::FuncOp>(); 
    target.addIllegalOp<func::CallOp>();

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalOp<ModuleOp>();

    auto solver = std::make_unique<DataFlowSolver>();
    solver->load<dataflow::DeadCodeAnalysis>();
    solver->load<dataflow::IntegerRangeAnalysis>();
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    RewritePatternSet patterns(ctx);
    populatePTOToEmitCPatterns(patterns, typeConverter, ctx, *solver);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    // 3. 执行转换
    if (failed(applyPartialConversion(mop, target, std::move(patterns)))) {
      llvm::errs() << "Conversion FAILED! Rolling back executed.\n";
      return signalPassFailure();
    }

    // =========================================================================
    // 4. [终极清理] 
    // 顺序至关重要：
    // Step A: 先移除所有 Cast，让 Loop 的 Operand 类型变成底层类型 (如 int32)
    // Step B: 再根据新的 Operand 类型，修复 Loop IV 的类型
    // =========================================================================
    
    // --- Step A: 清理 UnrealizedConversionCastOp ---
    // Prefer dropping redundant/unused casts; otherwise lower to emitc.cast
    // so the C++ emitter can print it.
    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    bool castCleanupFailed = false;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (castCleanupFailed)
        return;

      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
        cast.emitError() << "unsupported unrealized_conversion_cast shape";
        castCleanupFailed = true;
        return;
      }

      Value input = cast.getOperand(0);
      Value output = cast.getResult(0);
      Type inTy = input.getType();
      Type outTy = output.getType();

      if (output.use_empty()) {
        castsToErase.push_back(cast);
        return;
      }

      if (inTy == outTy) {
        output.replaceAllUsesWith(input);
        castsToErase.push_back(cast);
        return;
      }

      if (emitc::isSupportedEmitCType(inTy) && emitc::isSupportedEmitCType(outTy)) {
        OpBuilder builder(cast);
        auto c = builder.create<emitc::CastOp>(cast.getLoc(), outTy, input);
        output.replaceAllUsesWith(c.getResult());
        castsToErase.push_back(cast);
        return;
      }

      cast.emitError() << "cannot lower unrealized_conversion_cast(" << inTy
                       << " -> " << outTy << ") to emitc.cast";
      castCleanupFailed = true;
    });

    for (auto cast : castsToErase)
      cast.erase();

    if (castCleanupFailed)
      return signalPassFailure();

    // --- Step A2: Sink casts of emitc.variable "reads" to their use sites ---
    //
    // SCFToEmitC lowers scf.if/scf.for results via mutable `emitc.variable` and
    // `emitc.assign`. During type conversion, casts from the variable handle to
    // the converted type may be materialized right after the variable
    // declaration, effectively snapshotting the value *before* assignments. That
    // produces wrong C++ (use-before-init / stale reads).
    //
    // Fix by re-materializing the cast at each use site so it reads the variable
    // at the point of use.
    {
      SmallVector<emitc::CastOp> castOpsToSink;
      mop.walk([&](emitc::CastOp castOp) {
        if (castOp.getSource().getDefiningOp<emitc::VariableOp>())
          castOpsToSink.push_back(castOp);
      });

      for (emitc::CastOp castOp : castOpsToSink) {
        Value src = castOp.getSource();
        Type dstTy = castOp.getResult().getType();
        Value oldRes = castOp.getResult();

        // Replace each use with a freshly inserted cast right before the user.
        for (OpOperand &use : llvm::make_early_inc_range(oldRes.getUses())) {
          Operation *user = use.getOwner();
          OpBuilder b(user);
          b.setInsertionPoint(user);
          auto newCast = b.create<emitc::CastOp>(castOp.getLoc(), dstTy, src);
          use.set(newCast.getResult());
        }

        castOp.erase();
      }
    }

    // --- Step B: 修复 Loop 归纳变量 (IV) ---
    // 此时 emitc.for 的 operand 已经是 int32 了，我们检查 IV 是否匹配，不匹配则修正
    mop.walk([&](emitc::ForOp forOp) {
       Type boundTy = forOp.getLowerBound().getType(); 
       BlockArgument iv = forOp.getBody()->getArgument(0); 
       
       if (iv.getType() != boundTy) {
         iv.setType(boundTy); // 强制将 IV 类型 (index) 修改为与边界一致 (int32)
       }
    });
    
    // --- Step C: 消除冗余 Tile 变量 (Dead Code Elimination) [新增] ---
    // 逻辑：如果一个 emitc.variable 没有被读取（use_empty），
    // 那么它自己，以及给它赋值的 TASSIGN 都可以删除。
    // 注意：TASSIGN(v15, v9) 会把 v15 作为 Operand 0 使用，所以 v15 不是严格的 use_empty。
    // 我们需要检查：v15 是否除了 TASSIGN 之外没有其他 User。

    llvm::SmallVector<emitc::VariableOp> deadVars;
    mop.walk([&](emitc::VariableOp varOp) {
        // 检查该变量的所有 User
        bool isRead = false;
        for (Operation* user : varOp.getResult().getUsers()) {
            // 如果 User 是 TASSIGN 且变量是第0个参数(dst)，不算"读取"
            if (auto call = dyn_cast<emitc::CallOpaqueOp>(user)) {
                if (call.getCallee() == "TASSIGN" && call.getOperand(0) == varOp.getResult()) {
                    continue; // 这是一个赋值操作，不算有效使用
                }
            }
            // 如果还有其他用途（如 TLOAD, TMOV, TMATMUL），则该变量有用
            isRead = true;
            break;
        }

        if (!isRead) {
            deadVars.push_back(varOp);
        }
    });

    for (auto varOp : deadVars) {
        // 1. 先删除所有使用该变量的 TASSIGN
        llvm::SmallVector<Operation*> usersToErase;
        for (Operation* user : varOp.getResult().getUsers()) {
             // 我们上面已经确认过，剩下的 user 只能是 TASSIGN
             usersToErase.push_back(user);
        }
        for (auto u : usersToErase) u->erase();

        // 2. 删除变量定义本身
        varOp.erase();
    }

    // =========================================================================
  }
};
} // namespace

std::unique_ptr<Pass> mlir::pto::createEmitPTOManualPass() {
  return std::make_unique<EmitPTOManualPass>();
}
