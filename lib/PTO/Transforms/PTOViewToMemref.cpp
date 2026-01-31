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
// Helper Functions for Layout Normalization
// =============================================================================

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
    return MemRefType::get({ShapedType::kDynamic}, pty.getElementType());
  }
  
  // 2. 处理 !pto.tile_buf<...>
  if (auto tbTy = dyn_cast<mlir::pto::TileBufType>(t)) {
    SmallVector<int64_t> strides;
    
    // 默认计算 Contiguous Strides (因为 Tile 通常是 Dense 的)
    // 例如 32x64 -> [64, 1]
    bool isAllZero = true; // 简化逻辑，这里强制重算 strides 以保证一致性
    
    if (isAllZero) {
        auto shape = tbTy.getShape();
        strides.resize(shape.size());
        int64_t s = 1;
        // Row-Major 倒序计算
        for (int i = shape.size() - 1; i >= 0; --i) {
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

        // 2. 计算 Strides (假设 Row-Major 连续)
        // 例如 32x32 -> strides [32, 1]
        SmallVector<int64_t> strides;
        strides.resize(shape.size());
        int64_t s = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            if (shape[i] != ShapedType::kDynamic) s *= shape[i];
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

        // 6. [Design 2 关键修正] 独立透传 valid_row 和 valid_col
        // 不要检查 "if (row && col)"，而是分别获取。
        Value vRow = op.getValidRow(); // 如果 AllocTile 有 row，则是动态值；否则为 null
        Value vCol = op.getValidCol(); // 如果 AllocTile 有 col，则是动态值；否则为 null

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

        OpFoldResult off0 = rewriter.getIndexAttr(0);
        
        SmallVector<OpFoldResult, 4> sizes;
        for (Value v : op.getShape()) sizes.push_back(ensureIndex(rewriter, loc, v, op));

        SmallVector<OpFoldResult, 4> strides;
        for (Value v : op.getStrides()) strides.push_back(ensureIndex(rewriter, loc, v, op));

        auto rc = rewriter.create<memref::ReinterpretCastOp>(
            loc, mrTy, baseBuf, off0, sizes, strides);

        rewriter.replaceOp(op, rc.getResult());
      }

      // ------------------------------------------------------------------
      // Stage 2: Lower pto.subview -> memref.subview
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::SubviewOp, 8> subviews;
      func.walk([&](mlir::pto::SubviewOp op) { subviews.push_back(op); });

      for (auto op : subviews) {
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
      // Stage 2.5: lower pto.subset -> memref.reinterpret_cast
      // ------------------------------------------------------------------
      SmallVector<mlir::pto::SubsetOp, 8> subsets;
      func.walk([&](mlir::pto::SubsetOp op) { subsets.push_back(op); });

      for (auto op : subsets) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        Location loc = op.getLoc();

        // 1. 获取 Base Pointer (作为通用 MemRef)
        Value src = op->getOperand(0); 
        auto srcMrTy = dyn_cast<MemRefType>(src.getType());
        if (!srcMrTy) {
          op.emitError("pto.subset source must be lowered to memref first");
          signalPassFailure(); return;
        }

        // 2. 计算线性偏移 (Linear Offset)
        // 假设: subset %src[%i, %j] 中的 %i, %j 是用来计算 Block 偏移的
        // Offset = i * Stride0 + j * Stride1
        // 注意：这里的 Stride 是父块为了定位 Tile 而定义的 "Tile Stride"，
        // 而不是 MemRef 的 Element Stride。
        // 如果我们没有额外信息，我们暂时假设输入的 %i, %j 已经构成了我们需要的偏移，
        // 或者我们需要从父块的 Layout 中反解出 Tile Stride。
        
        // 【重要】为了实现 "Pointer Cast + Offset"，我们需要一个单一的动态 Offset。
        // 这里我们做一个简化的假设：PTO模型中，Subset 的 offset 计算已经由上层保证
        // 或者我们简单地将多维 offset 线性化。
        
        // 既然你提到 "Pointer Cast + Offset"，最直接的映射是：
        // BaseOffset = i * dim1_size + j  (如果是 2D 坐标)
        // 但由于 Tile 是连续的，通常 Offset = TileIndex * TileSize。
        
        // 让我们先实现通用的线性化逻辑 (Row-Major Linearization of Indices):
        Value totalOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto offsets = op.getOffsets(); // [%i, %j]
        
        // 我们需要父块的 Shape 来计算线性偏移吗？
        // 如果 Tile 也是连续存放的 (Tile Array)，那么 Shape 可能是 [NumTiles, TileSize...]
        // 这里我们采用最通用的做法：
        // 如果 layout 是 strided，我们利用 layout map 计算 offset。
        // 如果没有 layout，我们假设这是紧凑的，按 Shape 计算 strides。
        
        // 获取父块的 Strides (用于计算 Offset)
        // 注意：这里的 Strides 是用来计算 "怎么跳到目标位置"，
        // 而不是结果 MemRef 的 Strides。
        SmallVector<int64_t> srcStrides;
        int64_t dummyOffset; // 1. 声明一个哑变量
        
        // 2. 传进去占位
        if (failed(getStridesAndOffset(srcMrTy, srcStrides, dummyOffset))) {
             // 如果获取失败，假设 Row-Major 紧凑布局
             auto shape = srcMrTy.getShape();
             srcStrides.resize(shape.size());
             int64_t s = 1;
             for(int k=shape.size()-1; k>=0; --k) {
                 srcStrides[k] = s;
                 if(shape[k] != ShapedType::kDynamic) s *= shape[k];
             }
        }
        
        // 计算运行时 Offset: sum( index[k] * stride[k] )
        for (unsigned k = 0; k < offsets.size(); ++k) {
            Value idx = ensureIndex(rewriter, loc, offsets[k], op);
            Value strideVal = rewriter.create<arith::ConstantIndexOp>(loc, srcStrides[k]);
            Value dimOff = rewriter.create<arith::MulIOp>(loc, idx, strideVal);
            totalOffset = rewriter.create<arith::AddIOp>(loc, totalOffset, dimOff);
        }

        // 3. 准备 reinterpret_cast 的参数
        
        // A. Sizes (静态: [32, 32])
        SmallVector<OpFoldResult> newSizes;
        ArrayAttr sizeAttr = op.getSizes();
        for (Attribute attr : sizeAttr) {
            int64_t s = cast<IntegerAttr>(attr).getInt();
            newSizes.push_back(rewriter.getIndexAttr(s));
        }

        // B. Strides (强制连续: [32, 1])
        // 这就是 "Address Space Continuous" 的核心体现
        SmallVector<OpFoldResult> newStrides;
        // 计算 Row-Major Strides: [Dim1, 1]
        // 假设 Rank=2
        int64_t dim1Size = cast<IntegerAttr>(sizeAttr[1]).getInt();
        newStrides.push_back(rewriter.getIndexAttr(dim1Size)); // Stride 0 = Width
        newStrides.push_back(rewriter.getIndexAttr(1));        // Stride 1 = 1

        // 4. 构建结果类型
        // memref<32x32xf32, strided<[32, 1], offset: ?>>
        // 注意：这里的 Stride 是 [32, 1] 而不是 [64, 1] 了！
        // 因为我们把它当做独立的连续块来看待。
        auto resultLayout = StridedLayoutAttr::get(ctx, 
                                                   ShapedType::kDynamic, // Offset is dynamic
                                                   {dim1Size, 1});       // Strides are fixed contiguous
        
        auto resultMemRefType = MemRefType::get(
            {cast<IntegerAttr>(sizeAttr[0]).getInt(), dim1Size},
            srcMrTy.getElementType(),
            resultLayout,
            srcMrTy.getMemorySpace()
        );

        // 5. 创建 memref.reinterpret_cast
        // 语义：给我一个指针(src)，加上偏移(totalOffset)，
        // 然后把它强制看作是 32x32 的连续内存。
        auto castOp = rewriter.create<memref::ReinterpretCastOp>(
            loc,
            resultMemRefType,
            src,
            totalOffset,  // Dynamic Offset
            newSizes,     // Target Sizes
            newStrides    // Target Strides (Contiguous)
        );

        rewriter.replaceOp(op, castOp.getResult());
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
            op, op->getOperand(0), op->getOperand(1));
      }

      SmallVector<mlir::pto::TRowSumOp, 8> rowsum;
      func.walk([&](mlir::pto::TRowSumOp op) { rowsum.push_back(op); });
      for (auto op : rowsum) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<pto::RowSumOp_DPS>(
            op, op->getOperand(0), op->getOperand(1));
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

        rewriter.replaceOpWithNewOp<pto::ColExpandOp_DPS>(
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

        rewriter.replaceOpWithNewOp<pto::ColExpandOp_DPS>(
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

        rewriter.replaceOpWithNewOp<pto::ColSumOp_DPS>(
            op,
            TypeRange{},
            src,
            tmp,
            dst);
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
        BoolAttr scalar_lhs = op.getScalarLhsAttr();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto scaleTy = dyn_cast<FloatType>(scale.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !scaleTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::DivSOp_DPS>(
            op,
            TypeRange{},
            src,
            scale,
            dst,
            scalar_lhs);
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

      SmallVector<mlir::pto::TGatherOp, 8> gatherops;
      func.walk([&](mlir::pto::TGatherOp op) { gatherops.push_back(op); });

      for (auto op : gatherops) {
        IRRewriter rewriter(ctx);
        rewriter.setInsertionPoint(op);

        Value src = op.getSrc();
        Value indices = op.getIndices();
        Value dst = op.getDst();

        auto srcTy = dyn_cast<MemRefType>(src.getType());
        auto indicesTy = dyn_cast<MemRefType>(indices.getType());
        auto dstTy = dyn_cast<MemRefType>(dst.getType());
        if (!srcTy || !indicesTy || !dstTy) {
          op.emitError("ins/outs are not memref yet");
          signalPassFailure();
          return;
        }

        rewriter.replaceOpWithNewOp<pto::GatherOp_DPS>(
            op,
            TypeRange{},
            src,
            dst,
            indices);
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

        Value src = op.getSrc();
        Value dst = op.getDst();
        uint32_t blockLen = op.getBlockLen();

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
            src,
            dst,
            blockLen);
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
            idx,
            mem,
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
