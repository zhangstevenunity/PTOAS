#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h" 
#include "mlir/Dialect/MemRef/IR/MemRef.h" 
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h" // 用于 LLVM_DEBUG

#define DEBUG_TYPE "pto-convert-to-dps"

using namespace mlir;
using namespace mlir::pto;

namespace {

// 辅助函数：将 TileType 转换为对应的 TileBufType，并指定 Address Space
static Type getTileBufType(Type type, pto::AddressSpace addrSpace) {
  if (auto tile = llvm::dyn_cast<pto::TileType>(type)) {
    MLIRContext *ctx = tile.getContext();
    
    // 使用传入的 addrSpace 创建属性
    auto addrSpaceAttr = pto::AddressSpaceAttr::get(ctx, addrSpace);

    return pto::TileBufType::get(ctx, tile.getShape(), tile.getElementType(), addrSpaceAttr);
  }
  return Type();
}

// ============================================================================
// Pattern 1: LoadOp -> LoadDpsOp
// ============================================================================
// 逻辑：Load 总是产生数据。
// 如果原 Load 返回 Tensor -> 创建 tensor.empty 作为 outs，新 Op 返回 Tensor。
// 如果原 Load 返回其他（暂不支持），则报错或忽略。
struct LoadToDPSPattern : public OpRewritePattern<pto::LoadOp> {
  using OpRewritePattern<pto::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::LoadOp op, PatternRewriter &rewriter) const override {
    Type resType = op.getResult().getType();

    // === Path A: Tensor 流程 (Tensor Empty) ===
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resType)) {
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), tensorType.getShape(), tensorType.getElementType());
      
      // 创建带返回值的 DPS Op
      rewriter.replaceOpWithNewOp<pto::TLoadOp >(
          op, tensorType, op.getSrc(), initTensor);
      return success();
    }

    // === Path B: Tile 流程 (Alloc Tile) ===
    if (auto tileBufType = getTileBufType(resType, pto::AddressSpace::VEC)) {
      // 1. 显式分配 Buffer
      Value alloc = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType, Value(), /*valid_col*/Value());
      
      // 2. 创建无返回值的 DPS Op (Void return)
      rewriter.create<pto::TLoadOp >(
          op.getLoc(),
          TypeRange{}, // 无返回值
          op.getSrc(), // ins
          alloc        // outs
      );

      // 3. 用 alloc 替换原来的 SSA Value
      rewriter.replaceOp(op, alloc);
      return success();
    }

    return failure();
  }
};

// ============================================================================
// Pattern 2: StoreOp -> StoreDpsOp
// ============================================================================
// 逻辑：Store 是副作用。
// 如果 dst 是 MemRef -> StoreDpsOp 无返回值 (void)。
// 如果 dst 是 Tensor -> StoreDpsOp 返回新 Tensor (函数式更新)。
struct StoreToDPSPattern : public OpRewritePattern<pto::StoreOp> {
  using OpRewritePattern<pto::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::StoreOp op, PatternRewriter &rewriter) const override {
    Value src = op.getSrc(); // 数据 (Tile/Tensor)
    Value dst = op.getDst(); // 目标 (TileView/MemRef)
    Type dstType = dst.getType();

    // === Path A: MemRef (已经 Bufferized) ===
    if (mlir::isa<MemRefType>(dstType)) {
      rewriter.create<pto::TStoreOp>(
          op.getLoc(), TypeRange{}, src, dst);
      rewriter.eraseOp(op);
      return success();
    }

    // === Path B: TileView (High-Level) ===
    // 如果 dst 是 PartitionTensorViewType，那么 src 应该是 TileBufType (因为之前的 Pattern 已经把 Tile 替换为 Alloc 了)
    if (mlir::isa<pto::PartitionTensorViewType>(dstType)) {
      // 确认 src 是否已经被转换为了 Buffer (AllocTile 的结果)
      // 如果 src 还是 TileType (值语义)，这说明上游的 Op 还没被转换，或者转换顺序有问题。
      // 但 greedy pattern rewrite 会自动处理依赖。
      
      rewriter.create<pto::TStoreOp>(
          op.getLoc(),
          TypeRange{}, 
          src, // ins: tile_buf
          dst  // outs: tile_view
      );
      rewriter.eraseOp(op);
      return success();
    }
    
    // === Path C: Tensor (Copy语义) ===
    // 通常 Store 不用于 Tensor，除非做 InsertSlice。这里暂不处理。

    return failure();
  }
};

// ============================================================================
// Pattern: TAddOp -> AddOp_DPS
// ============================================================================
struct TAddToAddDPSPattern : public OpRewritePattern<pto::TAddOp> {
  using OpRewritePattern<pto::TAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TAddOp op, PatternRewriter &rewriter) const override {
    Value a   = op.getSrc0();
    Value b   = op.getSrc1();
    Value out = op.getDst();

    mlir::BoolAttr init = op.getInitOutBufferAttr(); 

    rewriter.replaceOpWithNewOp<pto::AddOp_DPS>(
        op,
        a, b, out,
        init 
    );
    return success();
  }
};


// ============================================================================
// Pattern 4: MatmulOp -> MatmulDpsOp
// ============================================================================
struct MatmulToDPSPattern : public OpRewritePattern<pto::MatmulOp> {
  using OpRewritePattern<pto::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MatmulOp op, PatternRewriter &rewriter) const override {
    Type resType = op.getResult().getType();
    Value bias = op.getBias(); // 获取 Bias (Optional)

    // === Path A: Tensor 流程 ===
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resType)) {
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), tensorType.getShape(), tensorType.getElementType());

      // 注意参数顺序：lhs, rhs, bias, dst
      rewriter.replaceOpWithNewOp<pto::TMatmulOp >(
          op, tensorType, 
          op.getLhs(), op.getRhs(), bias, initTensor);
      return success();
    }

    // === Path B: Tile 流程 ===
    if (auto tileBufType = getTileBufType(resType, pto::AddressSpace::MAT)) {
      Value alloc = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType,Value(), /*valid_col*/Value());

      // 创建无返回值的 Op
      rewriter.create<pto::TMatmulOp>(
          op.getLoc(),
          TypeRange{}, // Void result
          op.getLhs(), op.getRhs(), bias, // inputs
          alloc                          // output
      );

      rewriter.replaceOp(op, alloc);
      return success();
    }

    return failure();
  }
};

// ============================================================================
// Pattern 5: MatmulAccOp -> MatmulAccDpsOp [新增]
// ============================================================================
struct MatmulAccToDPSPattern : public OpRewritePattern<pto::MatmulAccOp> {
  using OpRewritePattern<pto::MatmulAccOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MatmulAccOp op,
                                PatternRewriter &rewriter) const override {
    Type resType = op.getResult().getType();

    // === Path A: Tensor 流程 (保持兼容) ===
    // 逻辑：创建 tensor.empty 作为 outs，返回新的 Tensor
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resType)) {
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), tensorType.getShape(), tensorType.getElementType());

      // DPS Op 返回 Tensor
      rewriter.replaceOpWithNewOp<pto::TMatmulAccOp >(
          op, 
          tensorType,        // Result Type
          op.getAccIn(),     // acc_in
          op.getLhs(),       // lhs
          op.getRhs(),       // rhs
          initTensor        // dst (outs)
      );
      return success();
    }


    // === Path B: Tile 流程 (Explicit Buffer Allocation) ===
    // 逻辑：创建 pto.alloc_tile 作为 outs，不返回值 (Void)，原结果替换为 alloc
    if (auto tileBufType = getTileBufType(resType, pto::AddressSpace::MAT)) {
      // 1. 显式分配 Output Buffer
      Value alloc = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType,Value(), /*valid_col*/Value());

      // 2. 创建无返回值的 MatmulAccDpsOp
      //    acc_in 在这里已经是上游转换过的 TileBuf (或者原始 TileView，取决于 ODS 约束)
      //    lhs, rhs 同理
      //    alloc 作为 dst 传入
      rewriter.create<pto::TMatmulAccOp >(
          op.getLoc(),
          TypeRange{},       // Void Return
          op.getAccIn(),     // acc_in (ins)
          op.getLhs(),       // lhs (ins)
          op.getRhs(),       // rhs (ins)
          alloc             // dst (outs)
      );

      // 3. 将原 Op 的结果 (Value Semantics) 替换为 alloc (Buffer Semantics)
      rewriter.replaceOp(op, alloc);
      return success();
    }

    return failure();
  }
};

// ============================================================================
// Pattern 6: MovOp -> MovDpsOp
// ============================================================================
struct MovToDPSPattern : public OpRewritePattern<pto::MovOp> {
  using OpRewritePattern<pto::MovOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::MovOp op, PatternRewriter &rewriter) const override {
    Type resType = op.getResult().getType();

    // === Path A: Tensor ===
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resType)) {
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), tensorType.getShape(), tensorType.getElementType());
      
      rewriter.replaceOpWithNewOp<pto::TMovOp >(
          op, tensorType, op.getSrc(), initTensor);
      return success();
    }

    // === Path B: Tile (Alloc) ===
    if (auto tileBufType = getTileBufType(resType, pto::AddressSpace::VEC)) {
      // 1. 分配目标 Buffer
      Value alloc = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType,Value(), /*valid_col*/Value());

      // 2. 创建 DPS Mov
      rewriter.create<pto::TMovOp >(
          op.getLoc(),
          TypeRange{}, // Void
          op.getSrc(), // ins (已经变成 tile_buf)
          alloc        // outs
      );

      // 3. 替换
      rewriter.replaceOp(op, alloc);
      return success();
    }

    return failure();
  }
};

// ============================================================================
// Pattern 7: TransOp -> TransDpsOp
// ============================================================================
struct TransposeToDPSPattern : public OpRewritePattern<pto::TransOp> {
  using OpRewritePattern<pto::TransOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TransOp op, PatternRewriter &rewriter) const override {
    Type resType = op.getResult().getType();

    // === Path A: Tensor 流程 (Tensor Empty) ===
    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(resType)) {
      Value initTensor = rewriter.create<tensor::EmptyOp>(
          op.getLoc(), tensorType.getShape(), tensorType.getElementType());
      
      // 创建带返回值的 DPS Op
      rewriter.replaceOpWithNewOp<pto::TTransOp>(
          op, tensorType, op.getSrc(), op.getTmp(), initTensor);
      return success();
    }

    // === Path B: Tile 流程 (Alloc Tile) ===
    if (auto tileBufType = getTileBufType(resType, pto::AddressSpace::VEC)) {
      // 1. 显式分配 Buffer
      Value alloc = rewriter.create<pto::AllocTileOp>(op.getLoc(), tileBufType,Value(), /*valid_col*/Value());
      
      // 2. 创建无返回值的 DPS Op (Void return)
      rewriter.create<pto::TTransOp>(
          op.getLoc(),
          TypeRange{}, // 无返回值
          op.getSrc(), // ins
          op.getTmp(),
          alloc        // outs
      );

      // 3. 用 alloc 替换原来的 SSA Value
      rewriter.replaceOp(op, alloc);
      return success();
    }

    return failure();
  }
};

// ============================================================================
// Pass Definition
// ============================================================================
struct PTOConvertToDPSPass : public PassWrapper<PTOConvertToDPSPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOConvertToDPSPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  StringRef getArgument() const override { return "pto-convert-to-dps"; }
  StringRef getDescription() const override { return "Convert PTO ops to DPS ops (Load/Store/AddF/Matmul)"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    llvm::errs() << "Begin Convert to DPS!\n";

    RewritePatternSet patterns(context);
    // [注册所有 Pattern]
    patterns.add<LoadToDPSPattern, 
                 StoreToDPSPattern, 
                 TAddToAddDPSPattern,
                 MatmulToDPSPattern,
                 MatmulAccToDPSPattern,
                 MovToDPSPattern,
                 TransposeToDPSPattern>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    func.print(llvm::errs());
    llvm::errs() << "\n// ==================================// \n\n";
  }
};
} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTOConvertToDPSPass() {
  return std::make_unique<PTOConvertToDPSPass>();
}
} // namespace pto
} // namespace mlir