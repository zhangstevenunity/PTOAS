//===- PTOConvert.cpp - Emit PTO Manual codes -----------------------------===//
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

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
      llvm::errs() << "[Debug] Unsupported FloatType: " << type << "\n";
      return Type{};
    });

    addConversion([Ctx](IntegerType type) -> Type {
      // [关键修改] i1 保持为 i1，不要转为 emitc.opaque<"bool">
      // 这样 emitc.if (接受 i1) 就不会报错。
      // 在打印 C++ 代码时，i1 会自动打印为 bool。
      //if (type.getWidth() == 1) return IntegerType::get(Ctx, 1); 
      if (type.getWidth() == 1) return type; // <--- 保持 i1 不变
      
      if (type.getWidth() == 32) return emitc::OpaqueType::get(Ctx, "int32_t");
      if (type.getWidth() == 64) return emitc::OpaqueType::get(Ctx, "int64_t");
      llvm::errs() << "[Debug] Unsupported IntegerType width: " << type.getWidth() << "\n";
      return emitc::OpaqueType::get(Ctx, "int32_t"); // Fallback
    });

    addConversion([Ctx](IndexType type) -> Type {
      return emitc::OpaqueType::get(Ctx, "int32_t");
    });
    
    // ---------------------------------------------------------
    // 2. PTO 特殊类型 (透传或转换)
    // ---------------------------------------------------------
    addConversion([Ctx](emitc::OpaqueType type) { return type; });
    addConversion([Ctx](emitc::PointerType type) { return type; });

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
         switch (ptoAttr.getAddressSpace()) {
           case pto::AddressSpace::UB: qualifier = "__ub__"; break;
           case pto::AddressSpace::GM: qualifier = "__gm__"; break;
           case pto::AddressSpace::MAT:   qualifier = "__mat__"; break; 
           case pto::AddressSpace::ACC:   qualifier = "__acc__"; break; 
           case pto::AddressSpace::LEFT:  qualifier = "__left__"; break; 
           case pto::AddressSpace::RIGHT: qualifier = "__right__"; break; 
           case pto::AddressSpace::BIAS:  qualifier = "__bias__"; break;
           default: 
             llvm::errs() << "  [Error] Unknown AddressSpace Enum\n";
             return std::nullopt;
         }
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
  }
};

struct ArithMulIToEmitC : public OpConversionPattern<arith::MulIOp> {
  using OpConversionPattern<arith::MulIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::MulIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) return failure();
    rewriter.replaceOpWithNewOp<emitc::MulOp>(op, newTy, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ArithAddIToEmitC : public OpConversionPattern<arith::AddIOp> {
  using OpConversionPattern<arith::AddIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::AddIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (!newTy) return failure();
    rewriter.replaceOpWithNewOp<emitc::AddOp>(op, newTy, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct ArithCastOPToEmitC : public OpConversionPattern<arith::IndexCastOp> {
  using OpConversionPattern<arith::IndexCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::IndexCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, op.getType(), op.getIn());

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
    Value idx = peelUnrealized(adaptor.getIdx());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MGATHER",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, mem, idx});

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
    MLIRContext *ctx = op.getContext();

    // 1. 准备函数参数类型 (使用具体类型替换泛型 T)
    SmallVector<Type> newArgTypes;
    newArgTypes.reserve(op.getNumArguments());

    for (Type argType : op.getArgumentTypes()) {
        Type newType;
        if (auto memRefTy = dyn_cast<MemRefType>(argType)) {
            std::string typeStr;
            Type elemTy = memRefTy.getElementType();
            if (elemTy.isF16()) typeStr = "half";
            else if (elemTy.isF32()) typeStr = "float";
            else if (elemTy.isInteger(8)) typeStr = "int8_t";
            else if (elemTy.isInteger(32)) typeStr = "int32_t";
            else typeStr = "void"; 

            std::string addrSpaceStr = "__gm__ "; 
            if (auto attr = dyn_cast_or_null<pto::AddressSpaceAttr>(memRefTy.getMemorySpace())) {
                if (attr.getAddressSpace() == pto::AddressSpace::UB) addrSpaceStr = "__ub__ ";
            }

            newType = emitc::PointerType::get(
                emitc::OpaqueType::get(ctx, addrSpaceStr + typeStr)
            );
        } else {
            newType = getTypeConverter()->convertType(argType);
        }
        
        if (!newType) return failure();
        newArgTypes.push_back(newType);
    }

    // 2. 创建单一 EmitC 函数
    auto funcType = FunctionType::get(ctx, newArgTypes, /*results=*/{});
    auto emitcFunc = rewriter.create<emitc::FuncOp>(
        op.getLoc(), 
        op.getName(), 
        funcType, 
        ArrayRef<NamedAttribute>{},
        ArrayRef<DictionaryAttr>{}
    );

    // 3. 设置 Specifiers
    emitcFunc.setSpecifiersAttr(rewriter.getStrArrayAttr({
        "__global__ AICORE"
    }));

    // 4. 迁移函数体
    rewriter.inlineRegionBefore(op.getBody(), emitcFunc.getBody(), emitcFunc.end());

    // =========================================================================
    // 5. [修复] 修正 Block 参数类型，并插入 Cast 桥接
    // =========================================================================
    Block &entryBlock = emitcFunc.getBody().front();
    
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&entryBlock);

    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
        BlockArgument arg = entryBlock.getArgument(i);
        Type oldType = arg.getType(); // MemRefType
        Type newType = newArgTypes[i]; // EmitC Pointer Type

        if (oldType != newType) {
            // A. 修改参数类型为新的 EmitC 指针
            arg.setType(newType);

            // B. 创建一个 Cast: NewType -> OldType
            // 这样函数体内的旧 Op (期望 MemRef) 就可以使用这个 Cast 的结果
            auto cast = rewriter.create<UnrealizedConversionCastOp>(
                op.getLoc(), oldType, arg);
            
            // C. 将 arg 的所有其他使用者替换为使用 cast 的结果
            // (除了 cast 自己，否则会死循环)
            arg.replaceAllUsesExcept(cast.getResult(0), cast);
        }
    }

    // 6. [兼容性补丁] 注入 "using T = float;"
    {
        rewriter.setInsertionPointToStart(&entryBlock); // 再次确保在最前面
        rewriter.create<emitc::VerbatimOp>(op.getLoc(), "using T = float;");
    }

    // 7. 删除旧 Op
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
    SmallVector<Value> dynamicStrideValues; // 用于后续构造 StrideDim 对象

    if (auto rc = op.getSource().getDefiningOp<memref::ReinterpretCastOp>()) {
        sourceStrides = rc.getMixedStrides();
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
    Value sourcePtr = adaptor.getSource();
    Value newPtr = rewriter.create<emitc::AddOp>(loc, sourcePtr.getType(), sourcePtr, totalOffset);


    // -------------------------------------------------------------------------
    // Part 2: 生成 GlobalTensor 类型 (Shape/Stride Template Generation)
    // -------------------------------------------------------------------------
    
    std::string suffix = "_" + std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
    std::string shapeTypeName  = "GTShape"  + suffix;
    std::string strideTypeName = "GTStride" + suffix;
    std::string gtTypeName     = "GT"       + suffix;

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
    } else if (elemTy.isInteger(32)) {
        if (elemTy.isSignlessInteger(32) || elemTy.isSignedInteger(32))
            elemTypeStr = "int32_t";
        else 
            elemTypeStr = "uint32_t";
    } else if (elemTy.isInteger(64)) {
        elemTypeStr = "int64_t";
    }

    // 2. 生成 Shape 模板参数，之后会右对齐有效维度并补齐到 5 维（高维填 1）
    SmallVector<std::string> shapeParamsVec;
    auto resShape = resTy.getShape();
    for (int i = 0; i < resTy.getRank(); ++i) {
        if (resShape[i] == ShapedType::kDynamic) {
            shapeParamsVec.push_back("-1");
        } else {
            shapeParamsVec.push_back(std::to_string(resShape[i]));
        }
    }

    // 3. 生成 Stride 动态值收集（保留，用于动态 stride 传参）
    SmallVector<std::string> dummyStrideVec;
    auto subViewSteps = op.getMixedStrides();
    for (int i = 0; i < rank; ++i) {
        int64_t finalStride = 1;
        bool isDynamic = false;
        if (i < (int)sourceStrides.size()) {
            if (auto val = extractStaticInt(sourceStrides[i])) {
                finalStride = *val;
            } else {
                isDynamic = true;
                if (auto v = sourceStrides[i].dyn_cast<Value>()) {
                    dynamicStrideValues.push_back(rewriter.getRemappedValue(v));
                }
            }
        }
        if (i < (int)subViewSteps.size()) {
            if (auto val = extractStaticInt(subViewSteps[i])) {
                finalStride *= *val;
            }
        }
        dummyStrideVec.push_back(isDynamic ? "-1" : std::to_string(finalStride));
    }

    // 3.1 右对齐有效维度，前部补 1；再按连续规则重新推导 5 维 stride
    // 选择“有效”维度（非 1 或 动态），右对齐到 5 维尾部
    SmallVector<std::string, 5> finalShape(5, "1");
    SmallVector<std::string> effectiveDims;
    for (const auto &d : shapeParamsVec) {
        if (d != "1")
            effectiveDims.push_back(d);
    }
    int eff = std::min<int>(effectiveDims.size(), 5);
    int shift = 5 - eff;
    for (int i = 0; i < eff; ++i) {
        finalShape[shift + i] = effectiveDims[effectiveDims.size() - eff + i];
    }

    SmallVector<std::string, 5> finalStride(5, "1");

    auto mulOrDyn = [](const std::string &a, const std::string &b) -> std::string {
        if (a == "-1" || b == "-1")
            return "-1";
        int64_t va = 1, vb = 1;
        (void)llvm::to_integer(a, va);
        (void)llvm::to_integer(b, vb);
        return std::to_string(va * vb);
    };

    // 最低维 stride 固定为 1（或动态）
    finalStride[4] = (finalShape[4] == "-1") ? "-1" : "1";
    for (int i = 3; i >= 0; --i) {
        finalStride[i] = mulOrDyn(finalStride[i + 1], finalShape[i + 1]);
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

    // 4. 发射 typedef 语句
    rewriter.create<emitc::VerbatimOp>(loc, "using " + shapeTypeName + " = pto::Shape<" + shapeParams + ">;");
    rewriter.create<emitc::VerbatimOp>(loc, "using " + strideTypeName + " = pto::Stride<" + strideParams + ">;");


    // -------------------------------------------------------------------------
    // Part 3: 显式对象实例化 (Explicit Object Instantiation)
    // -------------------------------------------------------------------------

    // A. 实例化 Shape 对象
    // C++: GTShape_... shape_inst(size_v0, size_v1...);
    auto shapeTypeOpaque = emitc::OpaqueType::get(ctx, shapeTypeName);
    SmallVector<Value> shapeArgs;
    // 从 adaptor.getSizes() 获取 subview 的所有 dynamic sizes
    for (Value dynSize : adaptor.getSizes()) {
        shapeArgs.push_back(dynSize);
    }
    
    auto shapeInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        shapeTypeOpaque, // 返回类型
        shapeTypeName,   // 调用的“函数名”即类名构造函数
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(shapeArgs)
    );
    
    // B. 实例化 Stride 对象
    // C++: GTStride_... stride_inst(stride_v0...);
    auto strideTypeOpaque = emitc::OpaqueType::get(ctx, strideTypeName);
    auto strideInstOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, 
        strideTypeOpaque, 
        strideTypeName, 
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{}, 
        /*operands=*/ValueRange(dynamicStrideValues) // 传入之前收集的动态stride
    );

    // C. 实例化 GlobalTensor 对象 (传入 Shape 和 Stride 对象)
    // C++: GT_... gt_inst(ptr, shape_inst, stride_inst);
    
    // 发射 GlobalTensor typedef
    rewriter.create<emitc::VerbatimOp>(
        loc, 
        "using " + gtTypeName + " = GlobalTensor<" + elemTypeStr + ", " + shapeTypeName + ", " + strideTypeName + ">;"
    );
    auto gtType = emitc::OpaqueType::get(ctx, gtTypeName);

    // 准备构造参数: [ptr, shape_instance, stride_instance]
    SmallVector<Value> gtConstructorArgs;
    gtConstructorArgs.push_back(newPtr);
    gtConstructorArgs.push_back(shapeInstOp.getResult(0)); // 拿到 shape_inst 的 SSA Value
    gtConstructorArgs.push_back(strideInstOp.getResult(0)); // 拿到 stride_inst 的 SSA Value

    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, 
        gtType, 
        gtTypeName,
        /*args=*/ArrayAttr{}, 
        /*templateArgs=*/ArrayAttr{},
        /*operands=*/ValueRange(gtConstructorArgs)
    );

    return success();
  }
};

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

  enum class TileRole { Vec, Mat, Left, Right, Acc, Bias };

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
    else if (elemType.isInteger(8)) elemTypeStr = "int8_t";
    else if (elemType.isInteger(32)) elemTypeStr = "int32_t";

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

    // TASSIGN
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{resultValue, adaptor.getAddrs()[0]});

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

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TLOAD",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

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

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TSTORE",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src});

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
    Value idx = peelUnrealized(adaptor.getIdx());

    // intrinsic: MSCATTER(dst(mem), src, indexes)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "MSCATTER",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, mem, idx});

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

    // intrinsic: SETVAL(dst, offset, val)
    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "SETVAL",
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

    // intrinsic: dst = GETVAL(src, offset)
    auto call = rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(),
        TypeRange{op.getDst().getType()},   
        "GETVAL",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{src, offset});

    rewriter.replaceOp(op, call.getResults());
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
    // 1. 获取源指针 (Workspace Base)
    Value source = adaptor.getSource();
    
    // 2. 获取偏移量 (Offsets)
    // memref.reinterpret_cast 的 offset 是一个列表。
    // 在我们的 2D 平铺场景下，之前的 Lowering 已经将 offset 线性化了 (比如 0 和 32)。
    // 所以我们只需要取第一个 offset。
    auto offsets = adaptor.getOffsets();
    
    // 如果没有 offset，或者是静态 offset 0 (需要检查 attributes)，则直接返回 source
    if (offsets.empty()) {
       // 注意：更严谨的实现应该检查 static_offsets 属性，但基于你的 IR，offset 是动态参数
       rewriter.replaceOp(op, source);
       return success();
    }

    Value offsetVal = offsets[0];

    // 3. 生成指针加法: new_ptr = source + offset
    // 我们使用 emitc.add 生成 C++ 的 "+" 运算符。
    // C++ 允许 指针 + 整数，所以这是合法的。
    
    // 获取结果类型 (转换后的 emitc 指针类型)
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();

    auto loc = op.getLoc();
    
    // 创建加法指令
    auto addOp = rewriter.create<emitc::AddOp>(loc, resultType, source, offsetVal);

    // 4. 用新的指针 (加法结果) 替换原 Op
    rewriter.replaceOp(op, addOp.getResult());
    
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
    Value src0 = peelUnrealized(adaptor.getSrc0());
    Value src1 = peelUnrealized(adaptor.getSrc1());
    Value src2 = peelUnrealized(adaptor.getSrc2());
    Value dst  = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADDC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, src1, src2});

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
    Value src0    = peelUnrealized(adaptor.getSrc0());
    Value scalar  = peelUnrealized(adaptor.getScalar());
    Value src1    = peelUnrealized(adaptor.getSrc1());
    Value dst     = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TADDSC",
        ArrayAttr{}, ArrayAttr{},
        ValueRange{dst, src0, scalar, src1});

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
struct PTOAssignToEmitC : public OpConversionPattern<pto::AssignOp_DPS> {
  using OpConversionPattern<pto::AssignOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::AssignOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    Value dst  = peelUnrealized(adaptor.getDst());
    Value addr = peelUnrealized(adaptor.getAddr());

    rewriter.create<emitc::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "TASSIGN",
        ArrayAttr{}, ArrayAttr{}, ValueRange{dst, addr});

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

     auto modeTy = emitc::OpaqueType::get(ctx, "auto");
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

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());

    // cmpMode -> token
    auto cmpAttr = op.getCmpModeAttr();          // PTO_CmpModeAttr
    std::string tok = cmpModeTok(cmpAttr);

    auto argsAttr = rewriter.getArrayAttr({
      emitc::OpaqueAttr::get(ctx, tok),
    });

    Type dstTy = getTypeConverter()->convertType(adaptor.getDst().getType());
    if (!dstTy)
      return rewriter.notifyMatchFailure(op, "cannot convert dst type");

    auto call = rewriter.create<emitc::CallOpaqueOp>(
      loc,
      TypeRange{dstTy},
      "TCMPS",
      /*args=*/argsAttr,
      /*templateArgs=*/ArrayAttr{},
      /*operands=*/ValueRange{src, scalar}
    );

    rewriter.replaceOp(op, call.getResults());
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
        /*args=*/rewriter.getArrayAttr({}),          // no extra tokens
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
        /*args=*/rewriter.getArrayAttr({}),
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
    Value tmp = peelUnrealized(adaptor.getTmp());
    Value dst = peelUnrealized(adaptor.getDst());

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

    rewriter.eraseOp(op);
    return success();
  }
};
static std::string roundModeTok(mlir::pto::RoundModeAttr attr) {
  using RM = mlir::pto::RoundMode;
  switch (attr.getValue()) {
  case RM::NONE:      return "RoundMode::NONE";
  case RM::RINT:      return "RoundMode::RINT";
  case RM::ROUND:     return "RoundMode::ROUND";
  case RM::FLOOR:     return "RoundMode::FLOOR";
  case RM::CEIL:      return "RoundMode::CEIL";
  case RM::TRUNC:     return "RoundMode::TRUNC";
  case RM::ODD:       return "RoundMode::ODD";
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
    auto boolTy = emitc::OpaqueType::get(ctx, "auto");
    auto tok = rmTok; // 默认是 "RoundMode::CAST_RINT"
    Value rmodeVal = rewriter.create<emitc::ConstantOp>(
        loc, boolTy, emitc::OpaqueAttr::get(ctx, tok));

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
//===----------------------------------------------------------------------===//

struct PTODivSToEmitC : public OpConversionPattern<pto::DivSOp_DPS> {
  using OpConversionPattern<pto::DivSOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::DivSOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src    = peelUnrealized(adaptor.getSrc());
    Value scalar = peelUnrealized(adaptor.getScalar());
    Value dst    = peelUnrealized(adaptor.getDst());

    bool scalarLhs = false;
    if (auto a = op.getScalarLhsAttr())
      scalarLhs = a.getValue();

    if (!scalarLhs) {
      // tile/scalar: TDIVS(dst, src, scalar)
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, src, scalar});
    } else {
      // scalar/tile: TDIVS(dst, scalar, src)
      rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{}, "TDIVS",
          ArrayAttr{}, ArrayAttr{},
          ValueRange{dst, scalar, src});
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
// pto.tfillpad lowering -> TFILLPAD(dst, src)
//===----------------------------------------------------------------------===//

struct PTOFillPadToEmitC : public OpConversionPattern<pto::FillPadOp_DPS> {
  using OpConversionPattern<pto::FillPadOp_DPS>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::FillPadOp_DPS op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TFILLPAD",
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

    Value src = peelUnrealized(adaptor.getSrc());
    Value fp  = peelUnrealized(adaptor.getFp());
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src, fp};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMOV_FP",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
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

    Value src = peelUnrealized(adaptor.getSrc());
    Value dst = peelUnrealized(adaptor.getDst());

    auto i32Ty = rewriter.getI32Type();
    int64_t bl = op.getBlockLenAttr().getInt();
    Value blockLen = rewriter.create<arith::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(static_cast<int32_t>(bl)));

    SmallVector<Value, 3> operands{dst, src, blockLen};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TMRGSORT",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

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

    SmallVector<Value, 3> operands{dst, src0, src1};
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

// ---------- TOp ----------
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
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src};
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
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 3> operands{dst, src};
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

    SmallVector<Value, 4> operands{dst, src0, src1, src2};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBC",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

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

    SmallVector<Value, 4> operands{dst, src0, scalar, src1};
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "TSUBSC",
        /*args=*/ArrayAttr{}, /*templateArgs=*/ArrayAttr{},
        /*operands=*/operands);

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

    SmallVector<Value, 4> operands{dst, src0, src1};
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

    SmallVector<Value, 4> operands{dst, tmp, src};
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
    Value scalar = adaptor.getScalar();
    Value dst = peelUnrealized(adaptor.getDst());

    SmallVector<Value, 4> operands{dst, src0, scalar};
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
    // 将 arith.cmpi 转换为 emitc.cmp
    // 映射 Predicate: eq -> equal, slt -> less, etc.
    emitc::CmpPredicate emitcPred;
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:  emitcPred = emitc::CmpPredicate::eq; break;
      case arith::CmpIPredicate::ne:  emitcPred = emitc::CmpPredicate::ne; break;
      case arith::CmpIPredicate::slt: emitcPred = emitc::CmpPredicate::lt; break;
      case arith::CmpIPredicate::sle: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::sgt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::sge: emitcPred = emitc::CmpPredicate::ge; break;
      // ... 处理无符号比较 (ult, ule 等) ...
      case arith::CmpIPredicate::ult: emitcPred = emitc::CmpPredicate::lt; break; // EmitC CmpOp 不区分有符号/无符号，依赖操作数类型
      case arith::CmpIPredicate::ule: emitcPred = emitc::CmpPredicate::le; break;
      case arith::CmpIPredicate::ugt: emitcPred = emitc::CmpPredicate::gt; break;
      case arith::CmpIPredicate::uge: emitcPred = emitc::CmpPredicate::ge; break;
      default: return failure();
    }

    rewriter.replaceOpWithNewOp<emitc::CmpOp>(
        op, 
        /*resultType=*/op.getType(), // i1 -> bool/i1
        emitcPred,
        adaptor.getLhs(), 
        adaptor.getRhs()
    );
    return success();
  }
};

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
  patterns.add<PTOAssignToEmitC>(typeConverter, ctx);
  patterns.add<PTOSelSToEmitC>(typeConverter, ctx);
  patterns.add<PTOColMinToEmitC>(typeConverter, ctx);
  patterns.add<PTORowExpandSubToEmitC>(typeConverter, ctx);
  patterns.add<PTOShrSToEmitC>(typeConverter, ctx);
  patterns.add<PTOShlSToEmitC>(typeConverter, ctx);
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
  patterns.add<AffineApplyMulConstToEmitC>(typeConverter, ctx);
  patterns.add<PTONegToEmitC>(typeConverter, ctx);
  patterns.add<PTOCIToEmitC>(typeConverter, ctx);
  patterns.add<PTOCmpToEmitC>(typeConverter, ctx);
  patterns.add<PTOColSumToEmitC>(typeConverter, ctx);
  patterns.add<PTOLReluToEmitC>(typeConverter, ctx);
  patterns.add<PTOMrgSortToEmitC>(typeConverter, ctx);
  patterns.add<SubviewToEmitCPattern>(typeConverter, ctx);
  patterns.add<PointerCastConversion>(typeConverter, ctx);
  patterns.add<PTOSetValToSETVAL, PTOGetValToGETVAL>(typeConverter, ctx);
  patterns.add<PTOAndToEmitC>(typeConverter, ctx);
  patterns.add<PTOMulToEmitC>(typeConverter, ctx);
  patterns.add<PTOAndSToEmitC>(typeConverter, ctx);
  patterns.add<PTOCvtToEmitC>(typeConverter, ctx);
  patterns.add<PTODivToTDIV>(typeConverter, ctx);
  patterns.add<PTOMaxToEmitC>(typeConverter, ctx);
  patterns.add<PTOMaxSToEmitC>(typeConverter, ctx);
  patterns.add<ArithMulIToEmitC>(typeConverter, ctx);
  patterns.add<ArithAddIToEmitC>(typeConverter, ctx);
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
  patterns.add<ReinterpretCastToEmitC>(typeConverter, ctx);
  patterns.add<PTOAbsToTABS>(typeConverter, ctx);
  patterns.add<PTOAddToTADD>(typeConverter, ctx);
  patterns.add<PTOAddSCToTADDSC>(typeConverter, ctx);
  patterns.add<ArithCastOPToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncSetToEmitC>(typeConverter, ctx);
  patterns.add<PTOSyncWaitToEmitC>(typeConverter, ctx);
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
    PTOBarrierToEmitC
  >(typeConverter, ctx);

  patterns.add<ReturnToEmitC>(typeConverter, ctx);

  populateSCFToEmitCConversionPatterns(patterns);
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
                    mlir::pto::PTODialect>();
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

    // 2. 配置转换目标
    PTOToEmitCTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<pto::PTODialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<mlir::scf::SCFDialect>(); 
    
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
    
    // --- Step A: 移除 Cast ---
    llvm::SmallVector<UnrealizedConversionCastOp> castsToErase;
    mop.walk([&](UnrealizedConversionCastOp cast) {
      if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) return;
      
      Value input = cast.getOperand(0);
      Value output = cast.getResult(0);
      
      // 暴力替换：无论类型是否匹配，直接用 Input 替换 Output
      // 这会让 emitc.for 的 operand 从 cast(index) 变为 const(int32)
      output.replaceAllUsesWith(input);
      castsToErase.push_back(cast);
    });

    for (auto cast : castsToErase) {
      cast.erase();
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
