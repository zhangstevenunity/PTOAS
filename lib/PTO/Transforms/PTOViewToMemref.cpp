/**
 * PTOViewToMemref.cpp
 * * 功能：将 PTO Dialect 的高层 Tile 操作降级为标准的 MemRef 操作。
 * 核心机制：
 * 1. 类型转换：!pto.tile_buf -> memref<..., offset: ?>
 * 2. 元数据保留：使用 pto.bind_tile 将 TileConfig 绑定到 SSA Value 上。
 * 3. 动态回溯：计算算子通过 lookupConfig 回溯 SSA 链条获取硬件配置。
 */

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils.h" // 假设包含一些通用的工具函数

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOVIEWTOMEMREF

namespace {

// =============================================================================
// Helper: Metadata Backtracking (核心机制)
// =============================================================================
// 从一个 MemRef Value 向上回溯，找到它绑定的 TileBufConfig。
// 这解决了 "Type Erasure" 问题：memref 类型本身不包含 config，但 SSA 定义链包含。
static mlir::pto::TileBufConfigAttr lookupConfig(Value v) {
  // 1. 最直接的情况：它就是 bind_tile 的结果
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    return bind.getConfig();
  }
  
  // 2. 穿透 View 操作 (SubView, Cast 等) 向上查找
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    return lookupConfig(subview.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    return lookupConfig(cast.getSource());
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    return lookupConfig(cast.getSource());
  }
  
  // 如果追溯到 BlockArgument (函数参数) 或其他无法穿透的 Op，则返回空
  return {}; 
}

// =============================================================================
// Helper: Valid dims backtracking (v_row / v_col)
// =============================================================================
static void lookupValidDims(Value v, Value &vRow, Value &vCol) {
  if (auto bind = v.getDefiningOp<mlir::pto::BindTileOp>()) {
    vRow = bind.getValidRow();
    vCol = bind.getValidCol();
    return;
  }
  if (auto subview = v.getDefiningOp<memref::SubViewOp>()) {
    lookupValidDims(subview.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::ReinterpretCastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  if (auto cast = v.getDefiningOp<memref::CastOp>()) {
    lookupValidDims(cast.getSource(), vRow, vCol);
    return;
  }
  vRow = Value();
  vCol = Value();
}

// =============================================================================
// Helper Functions for Layout Normalization
// =============================================================================

struct TileLayoutInfo {
  int64_t rowStride = 1;
  int64_t colStride = 1;
  int64_t innerRows = 1;
  int64_t innerCols = 1;
  bool boxed = false; // slayout != NoneBox
};

static int64_t getElemBytes(Type elemTy) {
  if (auto ft = elemTy.dyn_cast<FloatType>()) {
    if (ft.isF16() || ft.isBF16()) return 2;
    if (ft.isF32()) return 4;
    if (ft.isF64()) return 8;
  }
  if (auto it = elemTy.dyn_cast<IntegerType>()) {
    int64_t bytes = it.getWidth() / 8;
    return bytes > 0 ? bytes : 1;
  }
  return -1;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<BLayoutAttr>(attr)) {
    out = (int32_t)a.getValue();
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = (int32_t)a.getInt();
    return true;
  }
  return false;
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<SLayoutAttr>(attr)) {
    out = (int32_t)a.getValue();
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = (int32_t)a.getInt();
    return true;
  }
  return false;
}

static bool getConstIndexValue(Value v, int64_t &out) {
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
  if (auto castOp = v.getDefiningOp<arith::IndexCastOp>())
    return getConstIndexValue(castOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtSIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto extOp = v.getDefiningOp<arith::ExtUIOp>())
    return getConstIndexValue(extOp.getIn(), out);
  if (auto truncOp = v.getDefiningOp<arith::TruncIOp>())
    return getConstIndexValue(truncOp.getIn(), out);
  return false;
}

static bool computeTileLayoutInfo(mlir::pto::TileBufConfigAttr cfg, Type elemTy,
                                  ArrayRef<int64_t> shape,
                                  TileLayoutInfo &info) {
  if (shape.size() != 2) return false;
  if (shape[0] == ShapedType::kDynamic || shape[1] == ShapedType::kDynamic)
    return false;

  int64_t rows = shape[0];
  int64_t cols = shape[1];

  int32_t bl = 0; // RowMajor
  int32_t sl = 0; // NoneBox
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize())) fr = (int32_t)attr.getInt();

  // Inner shape
  if (sl == 0) {
    info.innerRows = 1;
    info.innerCols = 1;
    info.boxed = false;
  } else {
    info.boxed = true;
    int64_t elemBytes = getElemBytes(elemTy);
    if (elemBytes <= 0) return false;
    if (fr == 1024) {
      info.innerRows = 16;
      info.innerCols = 16;
    } else if (fr == 32) {
      info.innerRows = 16;
      info.innerCols = 2;
    } else if (fr == 512) {
      if (sl == 1) {
        info.innerRows = 16;
        info.innerCols = 32 / elemBytes;
      } else if (sl == 2) {
        info.innerRows = 32 / elemBytes;
        info.innerCols = 16;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  // Strides for pointer offset (block-aligned for boxed layouts).
  if (sl == 0) {
    if (bl == 1) {
      info.rowStride = 1;
      info.colStride = rows;
    } else {
      info.rowStride = cols;
      info.colStride = 1;
    }
  } else {
    if (bl == 1) {
      // ColMajor + InnerRowMajor (NZ) is supported. InnerColMajor is unsupported.
      if (sl != 1) return false;
      info.rowStride = info.innerCols;
      info.colStride = rows;
    } else {
      // RowMajor (ZZ/ZN)
      info.rowStride = cols;
      info.colStride = info.innerRows;
    }
  }

  return true;
}

// Helper: 递归拆解 AffineExpr
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

// Helper: 从 AffineMap 提取 Strides
static void decomposeStridedLayout(AffineMap map, SmallVectorImpl<int64_t> &strides) {
  strides.assign(map.getNumDims(), 0);
  if (map.getNumResults() != 1) return;
  
  SmallVector<AffineExpr, 4> terms;
  flattenAddExpr(map.getResult(0), terms);

  for (auto term : terms) {
    if (auto mul = term.dyn_cast<AffineBinaryOpExpr>()) {
      if (mul.getKind() == AffineExprKind::Mul) {
        AffineExpr lhs = mul.getLHS();
        AffineExpr rhs = mul.getRHS();
        if (auto dim = lhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = rhs.dyn_cast<AffineConstantExpr>())
            strides[dim.getPosition()] = cst.getValue();
        } else if (auto dim = rhs.dyn_cast<AffineDimExpr>()) {
          if (auto cst = lhs.dyn_cast<AffineConstantExpr>())
            strides[dim.getPosition()] = cst.getValue();
        }
      }
    } else if (auto dim = term.dyn_cast<AffineDimExpr>()) {
      strides[dim.getPosition()] = 1;
    }
  }
}

// 确保 Value 是 Index 类型
static Value ensureIndex(IRRewriter &rewriter, Location loc, Value v,
                         Operation *anchorOp) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), v);
  if (anchorOp)
    anchorOp->emitError() << "expected index or integer, but got " << v.getType();
  return Value();
}

static Value computeSubsetValidDim(IRRewriter &rewriter, Location loc,
                                   Value parentValid, Value offset,
                                   int64_t size, Operation *anchorOp) {
  Value sizeVal = rewriter.create<arith::ConstantIndexOp>(loc, size);
  if (!parentValid)
    return sizeVal;

  int64_t pvConst = 0, offConst = 0;
  if (getConstIndexValue(parentValid, pvConst) &&
      getConstIndexValue(offset, offConst)) {
    int64_t diff = 0;
    if (pvConst > 0) {
      int64_t offMod = offConst % pvConst;
      if (offMod < 0)
        offMod += pvConst;
      diff = pvConst - offMod; // in [1, pvConst] when pvConst>0
    }
    if (diff < 0)
      diff = 0;
    int64_t clipped = std::min<int64_t>(size, diff);
    return rewriter.create<arith::ConstantIndexOp>(loc, clipped);
  }

  Value pv = ensureIndex(rewriter, loc, parentValid, anchorOp);
  Value off = ensureIndex(rewriter, loc, offset, anchorOp);

  // Use the same "periodic valid dims" rule as SubsetOp::inferReturnTypes:
  // diff = pv - (off % pv), so offsets that land on the next tile (off == pv)
  // still produce a full valid dim (diff == pv), instead of 0.
  Type i64Ty = rewriter.getI64Type();
  Value pvI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, pv);
  Value offI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, off);
  Value remI64 = rewriter.create<arith::RemUIOp>(loc, offI64, pvI64);
  Value diffI64 = rewriter.create<arith::SubIOp>(loc, pvI64, remI64);
  Value diff = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   diffI64);

  Value lt = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, diff,
                                            sizeVal);
  return rewriter.create<arith::SelectOp>(loc, lt, diff, sizeVal);
}

static void dumpPretty(Operation *op, llvm::raw_ostream &os) {
  OpPrintingFlags flags;
  flags.useLocalScope();            
  AsmState state(op, flags);
  op->print(os, state);
  os << "\n";
  os.flush();
}

// =============================================================================
// Type Converter Logic
// =============================================================================

static Type convertPTOTypeToMemRef(Type t) {
  // 1. 处理 !pto.ptr<T>
  if (auto pty = dyn_cast<mlir::pto::PtrType>(t)) {
    return MemRefType::get({ShapedType::kDynamic}, pty.getElementType(),
                           MemRefLayoutAttrInterface(), Attribute());
  }
  
  // 2. 处理 !pto.tile_buf<...>
  if (auto tbTy = dyn_cast<mlir::pto::TileBufType>(t)) {
    SmallVector<int64_t> strides;
    
    // Try layout-aware strides first (BLayout/SLayout-aware).
    auto shape = tbTy.getShape();
    TileLayoutInfo info;
    bool gotLayout = false;
    if (computeTileLayoutInfo(tbTy.getConfigAttr(), tbTy.getElementType(), shape,
                              info)) {
      strides = {info.rowStride, info.colStride};
      gotLayout = true;
    }

    // Fallback: Row-Major contiguous strides.
    if (!gotLayout) {
      strides.resize(shape.size());
      int64_t s = 1;
      for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        if (shape[i] != ShapedType::kDynamic)
          s *= shape[i];
        else
          s = ShapedType::kDynamic;
      }
    }

    // 构造归一化的 Strided Layout
    // 【关键】Offset 设为 Dynamic (?)。
    // 这对于 Subview 出来的 MemRef 和 Alloc 出来的 MemRef 都必须一致，
    // 否则 TAdd 的两个输入类型不匹配会报错。
    auto layoutAttr = StridedLayoutAttr::get(t.getContext(), 
                                             ShapedType::kDynamic, // offset: ?
                                             strides);

    return MemRefType::get(
        tbTy.getShape(), 
        tbTy.getElementType(), 
        layoutAttr,
        tbTy.getMemorySpace()
    );
  }
  // 其他类型透传
  return t;
}

// =============================================================================
// The Pass Implementation
// =============================================================================

struct PTOViewToMemrefPass
    : public PassWrapper<PTOViewToMemrefPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOViewToMemrefPass)

  StringRef getArgument() const final { return "pto-view-to-memref"; }
  StringRef getDescription() const final {
    return "Lower PTO views to memref with Metadata Binding";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::pto::PTODialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    // Debug output before pass
    // dumpPretty(mod.getOperation(), llvm::errs());

    for (auto func : mod.getOps<func::FuncOp>()) {
      if (func.isExternal()) continue;

      Block &entry = func.front();
      auto fnTy = func.getFunctionType();

      // ------------------------------------------------------------------
      // Stage 0: Rewrite Function Signature
      // ------------------------------------------------------------------
      SmallVector<Type> newInputs;
      for (Type t : fnTy.getInputs()) newInputs.push_back(convertPTOTypeToMemRef(t));

      SmallVector<Type> newResults;
      for (Type t : fnTy.getResults()) newResults.push_back(convertPTOTypeToMemRef(t));

      // Update entry block arguments
      for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
        if (entry.getArgument(i).getType() != newInputs[i]) {
            entry.getArgument(i).setType(newInputs[i]);
        }
      }

      // Update function type
      func.setFunctionType(FunctionType::get(ctx, newInputs, newResults));

      // ------------------------------------------------------------------
      // Stage 0.5: lower pto.alloc_tile -> memref.alloc + pto.bind_tile
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::AllocTileOp, 8> allocTiles;
      func.walk([&](mlir::pto::AllocTileOp op) { allocTiles.push_back(op); });

      for (auto op : allocTiles) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        auto tbTy = dyn_cast<mlir::pto::TileBufType>(op.getResult().getType());
        if (!tbTy) continue;

        // 1. 获取 Shape 和 ElementType
        SmallVector<int64_t, 4> shape(tbTy.getShape().begin(), tbTy.getShape().end());
        Type elemTy = tbTy.getElementType();

        // 2. 计算 Strides (layout-aware when possible)
        SmallVector<int64_t> strides;
        TileLayoutInfo info;
        if (computeTileLayoutInfo(tbTy.getConfigAttr(), elemTy, shape, info)) {
          strides = {info.rowStride, info.colStride};
        } else {
          strides.resize(shape.size());
          int64_t s = 1;
          for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            if (shape[i] != ShapedType::kDynamic) s *= shape[i];
          }
        }

        // 3. 构造 [Alloc 专用] 的静态类型 (Offset: 0)
        // memref.alloc 要求明确的 layout，不能是动态 offset
        auto allocLayout = StridedLayoutAttr::get(ctx, 0, strides); // offset = 0
        auto allocType = MemRefType::get(shape, elemTy, allocLayout, tbTy.getMemorySpace());

        // 4. 构造 [BindTile 输出] 的动态类型 (Offset: ?)
        // 这必须与 convertPTOTypeToMemRef 返回的类型一致，以便与 Subview 兼容
        auto targetLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, strides); // offset = ?
        auto targetType = MemRefType::get(shape, elemTy, targetLayout, tbTy.getMemorySpace());

        // 5. 创建 AllocOp (使用静态类型)
        // 这样就不会报 "symbol operand count" 错误了
        Value alloc = rewriter.create<memref::AllocOp>(loc, allocType);

        // 6. Preserve tile valid dims (v_row / v_col).
        //
        // `pto.alloc_tile` encodes the valid shape in the result TileBufType
        // (e.g. acc tile may be rows=16 but v_row=1). The alloc op itself does
        // not necessarily carry explicit operands for static valid dims, so we
        // must materialize them from the type to keep them through
        // tile_buf -> memref lowering.
        //
        // For dynamically valid tiles (validShape == [-1, -1]), preserve the
        // runtime operands if present.
        Value vRow = op.getValidRow();
        Value vCol = op.getValidCol();
        ArrayRef<int64_t> validShape = tbTy.getValidShape();
        if (!tbTy.hasDynamicValid()) {
          // TileBuf valid dims use a negative sentinel (e.g. '?' / -1), which is
          // distinct from MLIR's ShapedType::kDynamic (INT64_MIN). Treat any
          // negative value as dynamic here.
          if (validShape.size() >= 1 && validShape[0] >= 0) {
            vRow = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[0]))
                       .getResult();
          }
          if (validShape.size() >= 2 && validShape[1] >= 0) {
            vCol = rewriter
                       .create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                                  rewriter.getIndexAttr(validShape[1]))
                       .getResult();
          }
        }

        // 7. 获取 Config (保持不变)
        auto configAttr = tbTy.getConfigAttr();
        if (!configAttr) configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        // 8. 创建 BindTileOp
        // BindTileOp 的 Builder 会自动处理空的 Value，将其视为静态维度
        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, 
            targetType,     
            alloc,          
            vRow ? vRow : Value(), // 显式传值
            vCol ? vCol : Value(),
            configAttr
        );

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 1: Lower pto.make_tensor_view -> memref.reinterpret_cast
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::MakeTensorViewOp, 8> makeViews;
      func.walk([&](mlir::pto::MakeTensorViewOp op) { makeViews.push_back(op); });

      for (auto op : makeViews) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value baseBuf = op.getOperand(0);
        OpFoldResult off0 = rewriter.getIndexAttr(0);

        // Fold pto.addptr chains into the view base to avoid nested reinterpret_cast.
        bool foldedAddPtr = false;
        {
          Value cur = baseBuf;
          Value totalOffset;
          while (auto add = cur.getDefiningOp<mlir::pto::AddPtrOp>()) {
            foldedAddPtr = true;
            Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
            if (totalOffset)
              totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
            else
              totalOffset = off;
            cur = add.getOperand(0);
          }
          if (cur != baseBuf) {
            baseBuf = cur;
            off0 = totalOffset ? OpFoldResult(totalOffset) : off0;
          }
        }

        auto baseMr = dyn_cast<BaseMemRefType>(baseBuf.getType());
        if (!baseMr) {
             op.emitError("make_tensor_view base must be memref"); signalPassFailure(); return;
        }

        // [修复] 获取动态 Rank (根据 shape 输入的数量)
        size_t rank = op.getShape().size(); 

        // Construct target type with dynamic offset/strides
        Type elemTy = baseMr.getElementType();
        int64_t dyn = ShapedType::kDynamic;
        
        // [修复] 构建 N 维 Strided Layout
        // strides 数组长度必须等于 rank
        SmallVector<int64_t> dynStrides(rank, dyn);
        auto layout = StridedLayoutAttr::get(ctx, /*offset=*/dyn, /*strides=*/dynStrides);
        
        // [修复] 构建 N 维 Shape
        SmallVector<int64_t> dynShape(rank, dyn);
        auto mrTy = MemRefType::get(dynShape, elemTy, layout, baseMr.getMemorySpace());

        SmallVector<OpFoldResult, 4> sizes;
        for (Value v : op.getShape()) sizes.push_back(ensureIndex(rewriter, loc, v, op));

        SmallVector<OpFoldResult, 4> strides;
        for (Value v : op.getStrides()) strides.push_back(ensureIndex(rewriter, loc, v, op));

        auto rc = rewriter.create<memref::ReinterpretCastOp>(
            loc, mrTy, baseBuf, off0, sizes, strides);
        if (foldedAddPtr) {
          rc->setAttr("pto.addptr_trace", rewriter.getUnitAttr());
        }

        rewriter.replaceOp(op, rc.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 1.5: Fold pto.addptr chains into load/store_scalar.
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::LoadScalarOp, 8> loadScalars;
      func.walk([&](mlir::pto::LoadScalarOp op) { loadScalars.push_back(op); });

      for (auto op : loadScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          auto newOp = rewriter.create<pto::LoadScalarOp>(
              loc, op.getValue().getType(), base, totalOffset);
          rewriter.replaceOp(op, newOp.getValue());
        }
      }

      SmallVector<mlir::pto::StoreScalarOp, 8> storeScalars;
      func.walk([&](mlir::pto::StoreScalarOp op) { storeScalars.push_back(op); });

      for (auto op : storeScalars) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        Value base = op.getPtr();
        Value totalOffset = ensureIndex(rewriter, loc, op.getOffset(), op);

        bool foldedAddPtr = false;
        while (auto add = base.getDefiningOp<mlir::pto::AddPtrOp>()) {
          foldedAddPtr = true;
          Value off = ensureIndex(rewriter, loc, add.getOperand(1), add);
          if (totalOffset)
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, off);
          else
            totalOffset = off;
          base = add.getOperand(0);
        }

        if (foldedAddPtr) {
          rewriter.create<pto::StoreScalarOp>(
              loc, base, totalOffset, op.getValue());
          rewriter.eraseOp(op);
        }
      }

      // Clean up: addptr should be folded into make_tensor_view.
      SmallVector<Operation *, 8> addPtrs;
      func.walk([&](mlir::pto::AddPtrOp op) { addPtrs.push_back(op.getOperation()); });
      bool changed = true;
      while (changed) {
        changed = false;
        for (auto &op : addPtrs) {
          if (!op)
            continue;
          if (op->use_empty()) {
            op->erase();
            op = nullptr;
            changed = true;
          }
        }
      }
      for (auto *op : addPtrs) {
        if (!op)
          continue;
        op->emitError("addptr must feed make_tensor_view or load/store_scalar for lowering");
        signalPassFailure();
        return;
      }

      // ------------------------------------------------------------------
      // Stage 2: Lower pto.partition_tensor_view -> memref.subview
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::PartitionViewOp, 8> partitiontensorviews;
      func.walk([&](mlir::pto::PartitionViewOp op) { partitiontensorviews.push_back(op); });

      for (auto op : partitiontensorviews) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();
        Value src = op.getOperand(0);
        auto srcMrTy = dyn_cast<MemRefType>(src.getType());
        int64_t rank = srcMrTy.getRank();

        // =====================================================================
        // 1. 处理 Sizes (智能区分 Static/Dynamic)
        // =====================================================================
        ValueRange sizeValues = op.getSizes(); 
        SmallVector<int64_t> staticSizes;     // 用于构建 Result MemRefType
        SmallVector<OpFoldResult> mixedSizes; // 用于传给 memref.subview

        for (Value s : sizeValues) {
            // [关键修改] 检查 Value 是否源自常量 Op
            IntegerAttr constAttr;
            bool isStatic = false;

            // 检查 arith.constant (index or int)
            if (auto cOp = s.getDefiningOp<arith::ConstantIndexOp>()) {
                constAttr = rewriter.getIndexAttr(cOp.value());
                isStatic = true;
            } else if (auto cInt = s.getDefiningOp<arith::ConstantIntOp>()) {
                constAttr = rewriter.getIndexAttr(cInt.value());
                isStatic = true;
            }

            if (isStatic) {
                // Case A: 静态常量 -> 存 Attribute
                mixedSizes.push_back(constAttr);
                staticSizes.push_back(constAttr.getInt());
            } else {
                // Case B: 动态变量 -> 存 Value
                mixedSizes.push_back(ensureIndex(rewriter, loc, s, op));
                staticSizes.push_back(ShapedType::kDynamic);
            }
        }

        // =====================================================================
        // 2. 处理 Offsets (同样应用智能区分)
        // =====================================================================
        // Offsets 也需要同样的逻辑，否则也会报类似的 mismatch
        ValueRange offsValues = op.getOffsets();
        SmallVector<OpFoldResult> mixedOffsets;
        
        for (Value o : offsValues) {
            IntegerAttr constAttr;
            bool isStatic = false;
            
            if (auto cOp = o.getDefiningOp<arith::ConstantIndexOp>()) {
                constAttr = rewriter.getIndexAttr(cOp.value());
                isStatic = true;
            } else if (auto cInt = o.getDefiningOp<arith::ConstantIntOp>()) {
                constAttr = rewriter.getIndexAttr(cInt.value());
                isStatic = true;
            }

            if (isStatic) {
                mixedOffsets.push_back(constAttr);
            } else {
                mixedOffsets.push_back(ensureIndex(rewriter, loc, o, op));
            }
        }

        // =====================================================================
        // 3. 构建 Result MemRefType
        // =====================================================================
        int64_t dyn = ShapedType::kDynamic;
        SmallVector<int64_t> dynStrides(rank, dyn);
        auto layout = StridedLayoutAttr::get(ctx, dyn, dynStrides);
        
        auto resTy = MemRefType::get(staticSizes, srcMrTy.getElementType(), layout, srcMrTy.getMemorySpace());

        // =====================================================================
        // 4. 处理 Strides (默认全 1)
        // =====================================================================
        SmallVector<OpFoldResult> mixedStrides;
        for (int i = 0; i < rank; ++i) {
            mixedStrides.push_back(rewriter.getIndexAttr(1));
        }

        // =====================================================================
        // 5. 创建 memref.subview
        // =====================================================================
        auto sv = rewriter.create<memref::SubViewOp>(
            loc, 
            resTy, 
            src, 
            mixedOffsets, 
            mixedSizes, 
            mixedStrides
        );
        
        rewriter.replaceOp(op, sv.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 2.5: lower pto.subset -> memref.subview + bind_tile
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::SubsetOp, 8> subsets;
      func.walk([&](mlir::pto::SubsetOp op) { subsets.push_back(op); });

      for (auto op : subsets) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        // 1. Source must be memref already
        Value src = op->getOperand(0);
        auto srcMrTy = dyn_cast<MemRefType>(src.getType());
        if (!srcMrTy) {
          op.emitError("pto.subset source must be lowered to memref first");
          signalPassFailure();
          return;
        }

        // 2. Sizes (static)
        ArrayAttr sizeAttr = op.getSizes();
        SmallVector<int64_t> staticSizes;
        SmallVector<OpFoldResult> mixedSizes;
        staticSizes.reserve(sizeAttr.size());
        mixedSizes.reserve(sizeAttr.size());
        for (Attribute attr : sizeAttr) {
          int64_t s = cast<IntegerAttr>(attr).getInt();
          staticSizes.push_back(s);
          mixedSizes.push_back(rewriter.getIndexAttr(s));
        }

        // 3. Offsets (mixed)
        SmallVector<OpFoldResult> mixedOffsets;
        for (Value o : op.getOffsets()) {
          IntegerAttr constAttr;
          bool isStatic = false;
          if (auto cOp = o.getDefiningOp<arith::ConstantIndexOp>()) {
            constAttr = rewriter.getIndexAttr(cOp.value());
            isStatic = true;
          } else if (auto cInt = o.getDefiningOp<arith::ConstantIntOp>()) {
            constAttr = rewriter.getIndexAttr(cInt.value());
            isStatic = true;
          }
          if (isStatic)
            mixedOffsets.push_back(constAttr);
          else
            mixedOffsets.push_back(ensureIndex(rewriter, loc, o, op));
        }

        // 3.1 Layout-aware checks for boxed tiles (SLayout != NoneBox)
        auto configAttr = lookupConfig(src);
        if (!configAttr) configAttr = pto::TileBufConfigAttr::getDefault(ctx);

        TileLayoutInfo layoutInfo;
        bool hasLayout =
            computeTileLayoutInfo(configAttr, srcMrTy.getElementType(),
                                  srcMrTy.getShape(), layoutInfo);
        if (!hasLayout) {
          op.emitError("unsupported tile layout for pto.subset");
          signalPassFailure();
          return;
        }

        if (layoutInfo.boxed) {
          if (staticSizes.size() != 2 || op.getOffsets().size() != 2) {
            op.emitError("boxed layout subset expects 2D sizes/offsets");
            signalPassFailure();
            return;
          }

          auto checkMul = [&](int64_t v, int64_t m, StringRef name) -> bool {
            if (m <= 0) return false;
            if (v % m != 0) {
              op.emitError("boxed layout requires ") << name << " multiple of "
                                                   << m << ", got " << v;
              return false;
            }
            return true;
          };

          if (!checkMul(staticSizes[0], layoutInfo.innerRows, "row size") ||
              !checkMul(staticSizes[1], layoutInfo.innerCols, "col size")) {
            signalPassFailure();
            return;
          }

          int64_t off0 = 0, off1 = 0;
          bool off0Const = getConstIndexValue(op.getOffsets()[0], off0);
          bool off1Const = getConstIndexValue(op.getOffsets()[1], off1);
          if (off0Const) {
            if (!checkMul(off0, layoutInfo.innerRows, "row offset")) {
              signalPassFailure();
              return;
            }
          }
          if (off1Const) {
            if (!checkMul(off1, layoutInfo.innerCols, "col offset")) {
              signalPassFailure();
              return;
            }
          }

          int32_t bl = 0;
          (void)readBLayoutI32(configAttr.getBLayout(), bl);

          auto srcShape = srcMrTy.getShape();
          if (srcShape.size() == 2) {
            if (bl == 0) {
              if (staticSizes[1] != srcShape[1]) {
                op.emitError("boxed RowMajor subset must keep full cols");
                signalPassFailure();
                return;
              }
              if (!off1Const || off1 != 0) {
                op.emitError("boxed RowMajor subset requires static col offset = 0");
                signalPassFailure();
                return;
              }
            } else {
              if (staticSizes[0] != srcShape[0]) {
                op.emitError("boxed ColMajor subset must keep full rows");
                signalPassFailure();
                return;
              }
              if (!off0Const || off0 != 0) {
                op.emitError("boxed ColMajor subset requires static row offset = 0");
                signalPassFailure();
                return;
              }
            }
          }
        }

        // 4. Result layout inherits source strides (offset is dynamic)
        SmallVector<int64_t> srcStrides;
        int64_t srcOffset = ShapedType::kDynamic;
        if (failed(getStridesAndOffset(srcMrTy, srcStrides, srcOffset))) {
          // Fallback: compact row-major
          auto shape = srcMrTy.getShape();
          srcStrides.resize(shape.size());
          int64_t s = 1;
          for (int i = shape.size() - 1; i >= 0; --i) {
            srcStrides[i] = s;
            if (shape[i] != ShapedType::kDynamic) s *= shape[i];
          }
        }
        (void)srcOffset;

        auto resultLayout = StridedLayoutAttr::get(ctx, ShapedType::kDynamic, srcStrides);
        auto resultMemRefType =
            MemRefType::get(staticSizes, srcMrTy.getElementType(), resultLayout,
                            srcMrTy.getMemorySpace());

        // 5. Strides for subview: keep same stride (use 1)
        SmallVector<OpFoldResult> mixedStrides;
        mixedStrides.reserve(staticSizes.size());
        for (size_t i = 0; i < staticSizes.size(); ++i)
          mixedStrides.push_back(rewriter.getIndexAttr(1));

        auto sv = rewriter.create<memref::SubViewOp>(
            loc, resultMemRefType, src, mixedOffsets, mixedSizes, mixedStrides);

        // 6. Re-bind tile metadata (config + valid dims)
        Value parentVRow;
        Value parentVCol;
        lookupValidDims(src, parentVRow, parentVCol);

        Value vRow;
        Value vCol;
        if (!staticSizes.empty())
          vRow = computeSubsetValidDim(rewriter, loc, parentVRow,
                                       op.getOffsets()[0], staticSizes[0], op);
        if (staticSizes.size() > 1)
          vCol = computeSubsetValidDim(rewriter, loc, parentVCol,
                                       op.getOffsets()[1], staticSizes[1], op);

        auto bindOp = rewriter.create<pto::BindTileOp>(
            loc, resultMemRefType, sv.getResult(),
            vRow ? vRow : Value(), vCol ? vCol : Value(), configAttr);

        rewriter.replaceOp(op, bindOp.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 3: Rewrite Compute Ops 
      // [关键] 全面使用 op->getOperand(i) 避免 Typed Accessor Crash
      // ------------------------------------------------------------------
      
      // --- TLoadOp [Src, Dst] ---
      SmallVector<mlir::pto::TLoadOp, 8> loads;
      func.walk([&](mlir::pto::TLoadOp op) { loads.push_back(op); });
      for (auto op : loads) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          
          Value src = op->getOperand(0); 
          Value dst = op->getOperand(1);
          
          auto config = lookupConfig(dst); // Config on Tile

          rewriter.replaceOpWithNewOp<pto::LoadDpsOp>(op, TypeRange{}, src, dst);
      }

      // --- TStoreOp [Src, Dst] ---
      SmallVector<mlir::pto::TStoreOp, 8> storeops;
      func.walk([&](mlir::pto::TStoreOp op) { storeops.push_back(op); });
      for (auto op : storeops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op->getOperand(0); 
        Value dst = op->getOperand(1);

        auto config = lookupConfig(src); // Config on Tile

        rewriter.replaceOpWithNewOp<pto::StoreDpsOp>(op, TypeRange{}, src, dst);
      }

       // --- TTransOp [Src, Tmp, Dst] ---
      SmallVector<mlir::pto::TTransOp, 8> trans;
      func.walk([&](mlir::pto::TTransOp op) { trans.push_back(op); });
      for (auto op : trans) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::TransDpsOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      // --- TExpOp [Src, Dst] ---
      SmallVector<mlir::pto::TExpOp, 8> exp;
      func.walk([&](mlir::pto::TExpOp op) { exp.push_back(op); });
      for (auto op : exp) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::ExpOp_DPS>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1));
      }

      // --- TMulOp [Src, Scalar, Dst] ---
      SmallVector<mlir::pto::TMulOp, 8> mul;
      func.walk([&](mlir::pto::TMulOp op) { mul.push_back(op); });
      for (auto op : mul) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MulOp_DPS>(
            op, op->getOperand(0), op.getOperand(1), op->getOperand(2));
      }

      // --- TMulsOp [Src, Scalar, Dst] ---
      SmallVector<mlir::pto::TMulsOp, 8> muls;
      func.walk([&](mlir::pto::TMulsOp op) { muls.push_back(op); });
      for (auto op : muls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MulsOp_DPS>(
            op, op->getOperand(0), op.getScalar(), op->getOperand(2));
      }

      SmallVector<mlir::pto::TRowExpandMulOp, 8> rowemul;
      func.walk([&](mlir::pto::TRowExpandMulOp op) { rowemul.push_back(op); });

      for (auto op : rowemul) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RowExpandMulOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TRowMinOp, 8> rowmin;
      func.walk([&](mlir::pto::TRowMinOp op) { rowmin.push_back(op); });

      for (auto op : rowmin) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value tmp = op.getTmp();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !tmpTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RowMinOp_DPS>(
            op,
            src,
            tmp,
            dst);
      }

      // --- TAddOp [Src0, Src1, Dst] ---
      SmallVector<mlir::pto::TAddOp, 8> addops;
      func.walk([&](mlir::pto::TAddOp op) { addops.push_back(op); });
      for (auto op : addops) {
          IRRewriter rewriter(ctx);
          rewriter.setInsertionPoint(op);
          
          Value src0 = op->getOperand(0);
          auto config = lookupConfig(src0);
          
          rewriter.replaceOpWithNewOp<pto::AddFDpsOp>(
              op, TypeRange{}, 
              op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      // --- TMatmulOp [Lhs, Rhs, Bias?, Dst] ---
      SmallVector<mlir::pto::TMatmulOp , 8> matmuls;
      func.walk([&](mlir::pto::TMatmulOp  op) { matmuls.push_back(op); });
      for (auto op : matmuls) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value bias, dst;

        if (op->getNumOperands() == 4) {
            bias = op->getOperand(2);
            dst  = op->getOperand(3);
        } else {
            bias = Value(); 
            dst  = op->getOperand(2);
        }

        auto config = lookupConfig(lhs);

        rewriter.replaceOpWithNewOp<pto::MatmulDpsOp>(
          op, TypeRange{}, lhs, rhs, bias, dst);
      }

      // --- TMatmulAccOp [Acc, Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TMatmulAccOp , 8> matmulAccs;
      func.walk([&](mlir::pto::TMatmulAccOp  op) { matmulAccs.push_back(op); });
      for (auto op : matmulAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MatmulAccDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TMatmulBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      SmallVector<mlir::pto::TMatmulBiasOp , 8> matmulBiass;
      func.walk([&](mlir::pto::TMatmulBiasOp  op) { matmulBiass.push_back(op); });
      for (auto op : matmulBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MatmulBiasDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TMatmulMxOp---
      SmallVector<mlir::pto::TMatmulMxOp , 8> matmulMxs;
      func.walk([&](mlir::pto::TMatmulMxOp  op) { matmulMxs.push_back(op); });
      for (auto op : matmulMxs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MatmulMxDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4));
      }

      // --- TMatmulMxAccOp  ---
      SmallVector<mlir::pto::TMatmulMxAccOp , 8> matmulMxAccs;
      func.walk([&](mlir::pto::TMatmulMxAccOp  op) { matmulMxAccs.push_back(op); });
      for (auto op : matmulMxAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MatmulMxAccDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TMatmulMxBiasOp ---
      SmallVector<mlir::pto::TMatmulMxBiasOp , 8> matmulMxBiass;
      func.walk([&](mlir::pto::TMatmulMxBiasOp  op) { matmulMxBiass.push_back(op); });
      for (auto op : matmulMxBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MatmulMxBiasDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3), op->getOperand(4), op->getOperand(5));
      }

      // --- TGemvOp [Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TGemvOp , 8> gemvs;
      func.walk([&](mlir::pto::TGemvOp  op) { gemvs.push_back(op); });
      for (auto op : gemvs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        Value dst = op->getOperand(2);

        auto config = lookupConfig(lhs);

        rewriter.replaceOpWithNewOp<pto::GemvDpsOp>(
          op, TypeRange{}, lhs, rhs, dst);
      }

      // --- TGemvAccOp [Acc, Lhs, Rhs, Dst] ---
      SmallVector<mlir::pto::TGemvAccOp , 8> gemvAccs;
      func.walk([&](mlir::pto::TGemvAccOp  op) { gemvAccs.push_back(op); });
      for (auto op : gemvAccs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::GemvAccDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TGemvBiasOp [Acc, Lhs, Rhs, Bias, Dst] ---
      SmallVector<mlir::pto::TGemvBiasOp , 8> gemvBiass;
      func.walk([&](mlir::pto::TGemvBiasOp  op) { gemvBiass.push_back(op); });
      for (auto op : gemvBiass) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::GemvBiasDpsOp>(
          op, TypeRange{}, 
          op->getOperand(0), op->getOperand(1), op->getOperand(2), op->getOperand(3));
      }

      // --- TMovOp [Src, Dst] ---
      SmallVector<mlir::pto::TMovOp , 8> movs;
      func.walk([&](mlir::pto::TMovOp  op) { movs.push_back(op); });
      for (auto op : movs) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::MovDpsOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1));
      }

      // --- Row Ops (ExpandDiv, ExpandSub, Max, Sum) ---
      SmallVector<mlir::pto::TRowExpandDivOp, 8> rowediv;
      func.walk([&](mlir::pto::TRowExpandDivOp op) { rowediv.push_back(op); });
      for (auto op : rowediv) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::RowExpandDivOp_DPS>(
            op, op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      SmallVector<mlir::pto::TRowExpandSubOp, 8> rowesub;
      func.walk([&](mlir::pto::TRowExpandSubOp op) { rowesub.push_back(op); });
      for (auto op : rowesub) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::RowExpandSubOp_DPS>(
            op, op->getOperand(0), op->getOperand(1), op->getOperand(2));
      }

      SmallVector<mlir::pto::TRowMaxOp, 8> rowmax;
      func.walk([&](mlir::pto::TRowMaxOp op) { rowmax.push_back(op); });
      for (auto op : rowmax) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::RowMaxOp_DPS>(
            op, TypeRange{}, op.getSrc(), op.getTmp(), op.getDst());
      }

      SmallVector<mlir::pto::TRowSumOp, 8> rowsum;
      func.walk([&](mlir::pto::TRowSumOp op) { rowsum.push_back(op); });
      for (auto op : rowsum) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::RowSumOp_DPS>(
            op, TypeRange{}, op.getSrc(), op.getTmp(), op.getDst());
      }

      SmallVector<mlir::pto::TAbsOp, 8> abseops;
      func.walk([&](mlir::pto::TAbsOp op) { abseops.push_back(op); });

      for (auto op : abseops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AbsOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TAddCOp, 8> addcops;
      func.walk([&](mlir::pto::TAddCOp op) { addcops.push_back(op); });

      for (auto op : addcops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value src2 = op.getSrc2();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto src2Ty = dyn_cast<MemRefType>(src2.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !src2Ty ||!dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AddCOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            src2,
            dst);
      }

      SmallVector<mlir::pto::TAddSOp, 8> addsops;
      func.walk([&](mlir::pto::TAddSOp op) { addsops.push_back(op); });

      for (auto op : addsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AddSOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TAddSCOp, 8> addscops;
      func.walk([&](mlir::pto::TAddSCOp op) { addscops.push_back(op); });

      for (auto op : addscops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value scalar = op.getScalar();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AddSCOp_DPS>(
            op,
            TypeRange{},
            src0,
            scalar,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TAndOp, 8> andops;
      func.walk([&](mlir::pto::TAndOp op) { andops.push_back(op); });

      for (auto op : andops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AndOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TAndSOp, 8> andsops;
      func.walk([&](mlir::pto::TAndSOp op) { andsops.push_back(op); });

      for (auto op : andsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::AndSOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TCIOp, 8> ciops;
      func.walk([&](mlir::pto::TCIOp op) { ciops.push_back(op); });

      for (auto op : ciops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value s= op.getS();
        Value dst = op.getDst();
        bool descending = op.getDescending();

        auto sTy = dyn_cast<IntegerType>(s.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!sTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::CIOp_DPS>(
            op,
            TypeRange{},
            s,
            dst,
            descending);
      }

      SmallVector<mlir::pto::TCmpOp, 8> cmpops;
      func.walk([&](mlir::pto::TCmpOp op) { cmpops.push_back(op); });

      for (auto op : cmpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

         auto newOp = rewriter.create<pto::CmpOp_DPS>(
            op.getLoc(),
            TypeRange{},
            src0,
            src1,
            dst);
         
          if (auto a = op.getCmpModeAttr())
            newOp->setAttr("cmpMode", a);

        rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
      }

      SmallVector<mlir::pto::TCmpSOp, 8> cmpsops;
      func.walk([&](mlir::pto::TCmpSOp op) { cmpsops.push_back(op); });

      for (auto op : cmpsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scalarTy = dyn_cast<FloatType>(scalar.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scalarTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        auto newOp = rewriter.create<pto::CmpSOp_DPS>(
            op.getLoc(),
            TypeRange{},
            src,
            scalar,
            dst);

        if (auto a = op.getCmpModeAttr())
          newOp->setAttr("cmpMode", a);

        rewriter.replaceOp(op, newOp->getResults()); // 0 results -> OK
      }

      SmallVector<mlir::pto::TColExpandOp, 8> colexpand;
      func.walk([&](mlir::pto::TColExpandOp op) { colexpand.push_back(op); });

      for (auto op : colexpand) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ColExpandOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TColMaxOp, 8> colmaxops;
      func.walk([&](mlir::pto::TColMaxOp op) { colmaxops.push_back(op); });

      for (auto op : colmaxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ColMaxOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TColMinOp, 8> colminops;
      func.walk([&](mlir::pto::TColMinOp op) { colminops.push_back(op); });

      for (auto op : colminops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if ( !srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ColMinOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TColSumOp, 8> colsumops;
      func.walk([&](mlir::pto::TColSumOp op) { colsumops.push_back(op); });

      for (auto op : colsumops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value tmp = op.getTmp();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("src/dst are not memref yet");
          signalPassFailure();
          return;
        }

        // If tmp exists, it must have isBinary attribute
        if (tmp) {
          auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
          if (!tmpTy) {
            op.emitError("tmp is not memref yet");
            signalPassFailure();
            return;
          }

          // Get isBinary attribute (should exist if tmp exists)
          BoolAttr isBinaryAttr = op.getIsBinaryAttr();
          if (!isBinaryAttr) {
            isBinaryAttr = BoolAttr::get(ctx, false);
          }

          rewriter.replaceOpWithNewOp<pto::ColSumOp_DPS>(
              op,
              TypeRange{},
              src,
              tmp,
              dst,
              isBinaryAttr);
        } else {
          // Format 1: no tmp, no isBinary
          // Use generic builder to avoid adding default isBinary attribute
          SmallVector<Value> operands = {src, dst};
          SmallVector<NamedAttribute> attrs;
          // Copy all attributes except isBinary
          for (auto attr : op->getAttrs()) {
            if (attr.getName() != "isBinary") {
              attrs.push_back(attr);
            }
          }
          rewriter.replaceOpWithNewOp<pto::ColSumOp_DPS>(
              op,
              TypeRange{},
              operands,
              attrs);
        }
      }

      SmallVector<mlir::pto::TCvtOp, 8> cvtops;
      func.walk([&](mlir::pto::TCvtOp op) { cvtops.push_back(op); });

      for (auto op : cvtops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        auto rmodeAttr = op.getRmodeAttr(); // PTO_RoundModeAttr

        auto newOp = rewriter.create<pto::CvtOp_DPS>(
            op.getLoc(),
            TypeRange{},
            src,
            dst);

       if (rmodeAttr)
         newOp->setAttr("rmode", rmodeAttr);
 
         rewriter.replaceOp(op, newOp->getResults());
      }

      SmallVector<mlir::pto::TDivOp, 8> divops;
      func.walk([&](mlir::pto::TDivOp op) { divops.push_back(op); });

      for (auto op : divops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::DivOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TDivSOp, 8> divsops;
      func.walk([&](mlir::pto::TDivSOp op) { divsops.push_back(op); });

      for (auto op : divsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scale = op.getScalar();
        Value dst = op.getDst();

        // Check types - they might still be TileBufType or already converted to MemRefType
        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto srcTileTy = dyn_cast<mlir::pto::TileBufType>(src.getType());
        auto scaleTy = dyn_cast<FloatType>(scale.getType());
        auto scaleTileTy = dyn_cast<mlir::pto::TileBufType>(scale.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto dstTileTy = dyn_cast<mlir::pto::TileBufType>(dst.getType());
        
        // Determine which operand is the tile/memref and which is the scalar
        // DivSOp_DPS expects (memref, scalar, dst) internally, so we need to ensure correct order
        // Check if src is memref/tensor/tile (not scalar)
        bool srcIsMemref = (srcTy != nullptr || srcTileTy != nullptr || 
                            isa<RankedTensorType>(src.getType()) ||
                            isa<mlir::pto::PartitionTensorViewType>(src.getType()));
        // Check if scale is memref/tensor/tile (not scalar)
        bool scaleIsMemref = (isa<MemRefType>(scale.getType()) || 
                              scaleTileTy != nullptr ||
                              isa<RankedTensorType>(scale.getType()) ||
                              isa<mlir::pto::PartitionTensorViewType>(scale.getType()));

        // Type validation - ensure we have the right types
        if (!srcIsMemref && !scaleIsMemref) {
          op.emitError("at least one operand (src or scale) must be tile_buf or memref");
          signalPassFailure();
          return;
        }
        if (srcIsMemref && scaleIsMemref) {
          op.emitError("exactly one operand (src or scale) must be tile_buf or memref, the other must be scalar");
          signalPassFailure();
          return;
        }

        if (!dstTy && !dstTileTy) {
          op.emitError("dst operand must be tile_buf or memref");
          signalPassFailure();
          return;
        }
        Value memrefOperand, scalarOperand;
        if (srcIsMemref) {
          // Normal order: (src=tile/memref, scale=scalar, dst)
          memrefOperand = src;
          scalarOperand = scale;
        } else {
          // Swapped order: (src=scalar, scale=tile/memref, dst)
          // Need to swap to (memref, scalar, dst) for DivSOp_DPS
          memrefOperand = scale;
          scalarOperand = src;
        }

        rewriter.replaceOpWithNewOp<pto::DivSOp_DPS>(
            op,
            TypeRange{},
            memrefOperand,
            scalarOperand,
            dst);
      }

      SmallVector<mlir::pto::TExpandsOp, 8> expandsops;
      func.walk([&](mlir::pto::TExpandsOp op) { expandsops.push_back(op); });

      for (auto op : expandsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ExpandsOp_DPS>(
            op,
            TypeRange{},
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TExtractOp, 8> extractops;
      func.walk([&](mlir::pto::TExtractOp op) { extractops.push_back(op); });

      for (auto op : extractops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indexRow = op.getIndexRow();
        Value indexCol = op.getIndexCol();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto indexRowTy = dyn_cast<IndexType>(indexRow.getType());
        auto indexColTy = dyn_cast<IndexType>(indexCol.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !indexRowTy || !indexColTy || !dstTy) {
          op.emitError("ins/outs are not correct yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ExtractOp_DPS>(
            op,
            TypeRange{},
            src,
            indexRow,
            indexCol,
            dst);
      }

      SmallVector<mlir::pto::TFillPadOp, 8> fillpadops;
      func.walk([&](mlir::pto::TFillPadOp op) { fillpadops.push_back(op); });

      for (auto op : fillpadops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::FillPadOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      // --- TSetValOp [Dst, Offset, Val] ---
      // Lower tile-world scalar write to memref-world SETVAL DPS op.
      SmallVector<mlir::pto::TSetValOp, 8> tsetvalops;
      func.walk([&](mlir::pto::TSetValOp op) { tsetvalops.push_back(op); });

      for (auto op : tsetvalops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value dst = op.getDst();
        Value offset = op.getOffset();
        Value val = op.getVal();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!dstTy) {
          op.emitError("dst is not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SetValDpsOp>(
            op,
            TypeRange{},
            dst,
            offset,
            val);
      }

      // --- TGetValOp [Src, Offset] -> Scalar ---
      // Lower tile-world scalar read to memref-world GETVAL DPS op.
      SmallVector<mlir::pto::TGetValOp, 8> tgetvalops;
      func.walk([&](mlir::pto::TGetValOp op) { tgetvalops.push_back(op); });

      for (auto op : tgetvalops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value offset = op.getOffset();
        Type dstType = op.getDst().getType();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        if (!srcTy) {
          op.emitError("src is not memref yet");
          signalPassFailure();
          return;
        }

        auto newOp = rewriter.create<pto::GetValDpsOp>(
            op.getLoc(),
            dstType,
            src,
            offset);
        rewriter.replaceOp(op, newOp.getDst());
      }

      SmallVector<mlir::pto::TGatherOp, 8> gatherops;
      func.walk([&](mlir::pto::TGatherOp op) { gatherops.push_back(op); });

      for (auto op : gatherops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value indices = op.getIndices();
        auto maskPattern = op.getMaskPatternAttr();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());

        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        if (indices) {
          auto indicesTy = dyn_cast<MemRefType>(indices.getType());
          if (!indicesTy) {
            op.emitError("indices must be memref yet");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::GatherOp_DPS>(
              op,
              TypeRange{},
              src,
              dst,
              indices);
        } else {
          if (!maskPattern) {
            op.emitError("expects maskPattern when indices is absent");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::GatherOp_DPS>(
              op,
              TypeRange{},
              src,
              dst,
              /*indices=*/Value(),
              /*maskPattern=*/maskPattern);
        }
      }

      SmallVector<mlir::pto::TGatherbOp, 8> gatherbops;
      func.walk([&](mlir::pto::TGatherbOp op) { gatherbops.push_back(op); });

      for (auto op : gatherbops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value offsets = op.getOffsets();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto offsetsTy = dyn_cast<MemRefType>(offsets.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !offsetsTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::GatherbOp_DPS>(
            op,
            TypeRange{},
            src,
            offsets,
            dst);
      }

      SmallVector<mlir::pto::TLogOp, 8> logops;
      func.walk([&](mlir::pto::TLogOp op) { logops.push_back(op); });

      for (auto op : logops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::LogOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TLReluOp, 8> lreluops;
      func.walk([&](mlir::pto::TLReluOp op) { lreluops.push_back(op); });

      for (auto op : lreluops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value slope = op.getSlope();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto slopeTy = dyn_cast<FloatType>(slope.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !slopeTy || !dstTy) {
          op.emitError("ins/outs are not correct type yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::LReluOp_DPS>(
            op,
            TypeRange{},
            src,
            slope,
            dst);
      }

      SmallVector<mlir::pto::TMaxOp, 8> maxops;
      func.walk([&](mlir::pto::TMaxOp op) { maxops.push_back(op); });

      for (auto op : maxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MaxOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TMaxSOp, 8> maxsops;
      func.walk([&](mlir::pto::TMaxSOp op) { maxsops.push_back(op); });

      for (auto op : maxsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scalarTy = dyn_cast<FloatType>(scalar.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scalarTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MaxSOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TMinOp, 8> minops;
      func.walk([&](mlir::pto::TMinOp op) { minops.push_back(op); });

      for (auto op : minops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MinOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TMinsOp, 8> minsops;
      func.walk([&](mlir::pto::TMinsOp op) { minsops.push_back(op); });

      for (auto op : minsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scalarTy = dyn_cast<FloatType>(scalar.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scalarTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MinsOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TMovFPOp, 8> movfpops;
      func.walk([&](mlir::pto::TMovFPOp op) { movfpops.push_back(op); });

      for (auto op : movfpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto fpTy = dyn_cast<MemRefType>(fp.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !fpTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MovFPOp_DPS>(
            op,
            TypeRange{},
            src,
            fp,
            dst);
      }

      SmallVector<mlir::pto::TMrgSortOp, 8> mrgsortops;
      func.walk([&](mlir::pto::TMrgSortOp op) { mrgsortops.push_back(op); });

      for (auto op : mrgsortops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        if (op.isFormat1()) {
          Value src = op.getSrc();
          Value dst = op.getDst();
          Value blockLenVal = op.getBlockLen();

          auto srcTy = dyn_cast<MemRefType>(src.getType());
          auto dstTy = dyn_cast<MemRefType>(dst.getType());
          if (!srcTy || !dstTy) {
            op.emitError("ins/outs are not memref yet");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::MrgSortOp_DPS>(
              op,
              TypeRange{},
              ValueRange{src},
              blockLenVal,
              ValueRange{dst},
              Value() /*excuted*/,
              op.getExhaustedAttr());
        } else if (op.isFormat2()) {
          bool allMemRef = true;
          for (Value v : op.getSrcs())
            if (!dyn_cast<MemRefType>(v.getType())) { allMemRef = false; break; }
          if (!allMemRef) {
            op.emitError("format2 ins/outs are not memref yet");
            signalPassFailure();
            return;
          }
          if (op.getDsts().size() != 2u) {
            op.emitError("format2 expects outs(dst, tmp) tile buffers");
            signalPassFailure();
            return;
          }

          Value dst = op.getDst();
          Value tmp = op.getTmp();
          Value excuted = op.getExcuted();
          if (!dyn_cast<MemRefType>(dst.getType()) || !dyn_cast<MemRefType>(tmp.getType())) {
            op.emitError("format2 outs(dst/tmp) must be memref");
            signalPassFailure();
            return;
          }
          if (!dyn_cast<VectorType>(excuted.getType())) {
            op.emitError("format2 outs(excuted) must be vector");
            signalPassFailure();
            return;
          }

          rewriter.replaceOpWithNewOp<pto::MrgSortOp_DPS>(
              op,
              TypeRange{},
              op.getSrcs(),
              Value() /*blockLen*/,
              ValueRange{dst, tmp},
              excuted,
              op.getExhaustedAttr());
        } else {
          op.emitError("tmrgsort must be format1 or format2");
          signalPassFailure();
          return;
        }
      }

      SmallVector<mlir::pto::TNegOp, 8> negops;
      func.walk([&](mlir::pto::TNegOp op) { negops.push_back(op); });

      for (auto op : negops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::NegOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TNotOp, 8> notops;
      func.walk([&](mlir::pto::TNotOp op) { notops.push_back(op); });

      for (auto op : notops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::NotOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TOrOp, 8> orops;
      func.walk([&](mlir::pto::TOrOp op) { orops.push_back(op); });

      for (auto op : orops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::OrOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TOrsOp, 8> orsops;
      func.walk([&](mlir::pto::TOrsOp op) { orsops.push_back(op); });

      for (auto op : orsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scalarTy = dyn_cast<IntegerType>(scalar.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scalarTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::OrsOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TPartAddOp, 8> partaddops;
      func.walk([&](mlir::pto::TPartAddOp op) { partaddops.push_back(op); });

      for (auto op : partaddops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::PartAddOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TPartMaxOp, 8> partmaxops;
      func.walk([&](mlir::pto::TPartMaxOp op) { partmaxops.push_back(op); });

      for (auto op : partmaxops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::PartMaxOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TPartMinOp, 8> partminops;
      func.walk([&](mlir::pto::TPartMinOp op) { partminops.push_back(op); });

      for (auto op : partminops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::PartMinOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TPreluOp, 8> preluops;
      func.walk([&](mlir::pto::TPreluOp op) { preluops.push_back(op); });

      for (auto op : preluops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::PreluOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TRecipOp , 8> recipops;
      func.walk([&](mlir::pto::TRecipOp  op) { recipops.push_back(op); });

      for (auto op : recipops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RecipOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TReluOp , 8> reluops;
      func.walk([&](mlir::pto::TReluOp  op) { reluops.push_back(op); });

      for (auto op : reluops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ReluOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TRemOp, 8> remops;
      func.walk([&](mlir::pto::TRemOp op) { remops.push_back(op); });

      for (auto op : remops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RemOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TRemSOp, 8> rems;
      func.walk([&](mlir::pto::TRemSOp op) { rems.push_back(op); });

      for (auto op : rems) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value scale = op.getScalar();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<FloatType>(scale.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RemSOp_DPS>(
            op,
            src,
            scale,
            dst);
      }

      SmallVector<mlir::pto::TReshapeOp , 8> reshapeops;
      func.walk([&](mlir::pto::TReshapeOp  op) { reshapeops.push_back(op); });

      for (auto op : reshapeops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ReshapeOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TRowExpandOp , 8> rowexpandops;
      func.walk([&](mlir::pto::TRowExpandOp  op) { rowexpandops.push_back(op); });

      for (auto op : rowexpandops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RowExpandOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TRsqrtOp , 8> rsqrtops;
      func.walk([&](mlir::pto::TRsqrtOp  op) { rsqrtops.push_back(op); });

      for (auto op : rsqrtops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::RsqrtOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TScatterOp, 8> scatters;
      func.walk([&](mlir::pto::TScatterOp op) { scatters.push_back(op); });

      for (auto op : scatters) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indexes = op.getIndexes();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto indexesTy = dyn_cast<MemRefType>(indexes.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !indexesTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ScatterOp_DPS>(
            op,
            src,
            indexes,
            dst);
      }

      SmallVector<mlir::pto::TSelOp, 8> selops;
      func.walk([&](mlir::pto::TSelOp op) { selops.push_back(op); });

      for (auto op : selops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value mask = op.getMask();
        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto maskTy = dyn_cast<MemRefType>(mask.getType());
        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!maskTy || !src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SelOp_DPS>(
            op,
            mask,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TSelsOp, 8> selsops;
      func.walk([&](mlir::pto::TSelsOp op) { selsops.push_back(op); });

      for (auto op : selsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value selectMode = op.getSelectMode();
        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto selectModeTy = dyn_cast<IntegerType>(selectMode.getType());
        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!selectModeTy || !src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SelsOp_DPS>(
            op,
            src0,
            src1,
            selectMode,
            dst);
      }

      SmallVector<mlir::pto::TShlOp, 8> shlps;
      func.walk([&](mlir::pto::TShlOp op) { shlps.push_back(op); });

      for (auto op : shlps) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ShlOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }
      
      SmallVector<mlir::pto::TShrOp, 8> shrps;
      func.walk([&](mlir::pto::TShrOp op) { shrps.push_back(op); });

      for (auto op : shrps) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::ShrOp_DPS>(
            op,
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TShlSOp, 8> shlsops;
      func.walk([&](mlir::pto::TShlSOp op) { shlsops.push_back(op); });
      for (auto op : shlsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::ShlSOp_DPS>(
            op, op.getSrc(), op.getScalar(), op.getDst());
      }

      SmallVector<mlir::pto::TShrSOp, 8> shrsops;
      func.walk([&](mlir::pto::TShrSOp op) { shrsops.push_back(op); });
      for (auto op : shrsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::ShrSOp_DPS>(
            op, op.getSrc(), op.getScalar(), op.getDst());
      }

      SmallVector<mlir::pto::TSort32Op , 8> sort32ops;
      func.walk([&](mlir::pto::TSort32Op  op) { sort32ops.push_back(op); });

      for (auto op : sort32ops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value idx = op.getIdx();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto idxTy = dyn_cast<MemRefType>(idx.getType());
        if (!srcTy || !dstTy || !idxTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::Sort32Op_DPS>(
            op,
            TypeRange{},
            src,
            dst,
            idx);
      }

      SmallVector<mlir::pto::TSqrtOp , 8> sqrtops;
      func.walk([&](mlir::pto::TSqrtOp  op) { sqrtops.push_back(op); });

      for (auto op : sqrtops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SqrtOp_DPS>(
            op,
            TypeRange{},
            src,
            dst);
      }

      SmallVector<mlir::pto::TStoreFPOp, 8> storefpops;
      func.walk([&](mlir::pto::TStoreFPOp op) { storefpops.push_back(op); });

      for (auto op : storefpops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value fp = op.getFp();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto fpTy = dyn_cast<MemRefType>(fp.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !fpTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::StoreFPOp_DPS>(
            op,
            TypeRange{},
            src,
            fp,
            dst);
      }

      SmallVector<mlir::pto::TSubOp, 8> subops;
      func.walk([&](mlir::pto::TSubOp op) { subops.push_back(op); });

      for (auto op : subops) {
        Type s0Ty = op.getSrc0().getType();
        Type s1Ty = op.getSrc1().getType();
        if (s0Ty != s1Ty)
          continue;

        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::SubOp_DPS>(
            op,
            TypeRange{},
            op.getSrc0(),
            op.getSrc1(),
            op.getDst());
      }

      SmallVector<mlir::pto::TSubCOp, 8> subcops;
      func.walk([&](mlir::pto::TSubCOp op) { subcops.push_back(op); });

      for (auto op : subcops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value src2 = op.getSrc2();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto src2Ty = dyn_cast<MemRefType>(src2.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !src2Ty ||!dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SubCOp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            src2,
            dst);
      }

      SmallVector<mlir::pto::TSubSOp, 8> subsops;
      func.walk([&](mlir::pto::TSubSOp op) { subsops.push_back(op); });

      for (auto op : subsops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value scalar = op.getScalar();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SubSOp_DPS>(
            op,
            TypeRange{},
            src,
            scalar,
            dst);
      }

      SmallVector<mlir::pto::TSubSCOp, 8> subscops;
      func.walk([&](mlir::pto::TSubSCOp op) { subscops.push_back(op); });

      for (auto op : subscops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value scalar = op.getScalar();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::SubSCOp_DPS>(
            op,
            TypeRange{},
            src0,
            scalar,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TXORSOp, 8> xors;
      func.walk([&](mlir::pto::TXORSOp op) { xors.push_back(op); });

      for (auto op : xors) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value dst = op.getDst();
        Value scale = op.getScalar();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::XORSOp_DPS>(
            op,
            src,
            scale,
            dst);
      }

      SmallVector<mlir::pto::TXOROp, 8> xorops;
      func.walk([&](mlir::pto::TXOROp op) { xorops.push_back(op); });

      for (auto op : xorops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src0 = op.getSrc0();
        Value src1 = op.getSrc1();
        Value dst = op.getDst();

        auto src0Ty = dyn_cast<MemRefType>(src0.getType());
        auto src1Ty = dyn_cast<MemRefType>(src1.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!src0Ty || !src1Ty || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::XOROp_DPS>(
            op,
            TypeRange{},
            src0,
            src1,
            dst);
      }

      SmallVector<mlir::pto::TMGatherOp, 8> mgatherops;
      func.walk([&](mlir::pto::TMGatherOp op) { mgatherops.push_back(op); });

      for (auto op : mgatherops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value dst = op.getDst();
        Value idx = op.getIdx();
        Value mem = op.getMem();

        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        auto idxTy = dyn_cast<MemRefType>(idx.getType());
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!dstTy || !idxTy || !memTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MGatherDpsOp>(
            op,
            TypeRange{},
            mem,
            idx,
            dst);
      }

      SmallVector<mlir::pto::TMScatterOp, 8> mascatterops;
      func.walk([&](mlir::pto::TMScatterOp op) { mascatterops.push_back(op); });

      for (auto op : mascatterops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value idx = op.getIdx();
        Value mem = op.getMem();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto idxTy = dyn_cast<MemRefType>(idx.getType());
        auto memTy = dyn_cast<MemRefType>(mem.getType());
        if (!srcTy || !idxTy || !memTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::MScatterDpsOp>(
            op,
            TypeRange{},
            src,
            idx,
            mem);
      }
      SmallVector<mlir::pto::TPrintOp, 8> printops;
      func.walk([&](mlir::pto::TPrintOp op) { printops.push_back(op); });

      for (auto op : printops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        if (!srcTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::PrintOp_DPS>(
            op,
            TypeRange{},
            src);
      }
    }
    
    // Debug Output
    dumpPretty(mod.getOperation(), llvm::errs());
  }
};

} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<PTOViewToMemrefPass>();
}

} // namespace pto
} // namespace mlir
