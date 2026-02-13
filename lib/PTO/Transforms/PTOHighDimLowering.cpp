#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "PTO/IR/PTO.h" 

using namespace mlir;
using namespace mlir::pto;

namespace {

// ==============================================================================
// Helper: Create Rank-Reduced Slice (Shared Logic)
// ==============================================================================
static Value createSlice(OpBuilder &builder, Location loc, Value operand, MemRefType type, 
                         ValueRange ivs, int64_t outerDimsCount, int64_t rank) {
  SmallVector<OpFoldResult> offsets, sizes, strides;
  SmallVector<int64_t> resultShape;

  // Outer Dims: Offset=iv, Size=1
  for (int i = 0; i < outerDimsCount; ++i) {
    offsets.push_back(ivs[i]);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));
  }

  // Inner Dims: Offset=0, Size=Full
  for (int i = rank - 2; i < rank; ++i) {
    offsets.push_back(builder.getIndexAttr(0));
    if (type.isDynamicDim(i)) {
      sizes.push_back(builder.create<memref::DimOp>(loc, operand, i).getResult());
      resultShape.push_back(ShapedType::kDynamic);
    } else {
      int64_t dimSize = type.getDimSize(i);
      sizes.push_back(builder.getIndexAttr(dimSize));
      resultShape.push_back(dimSize);
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  auto resType = memref::SubViewOp::inferRankReducedResultType(
      resultShape, type, offsets, sizes, strides);
   
  return builder.create<memref::SubViewOp>(
      loc, cast<MemRefType>(resType), operand, offsets, sizes, strides);
}

// ==============================================================================
// Helper: Build Loop Nest with ValidDims Optimization
// 统一封装循环构建逻辑，支持 valid_dims 优化
// ==============================================================================
static void buildOptimizedLoopNest(
    PatternRewriter &rewriter, Location loc, 
    int64_t outerDimsCount, Value srcMemRef, MemRefType srcType,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder) {

  SmallVector<Value> lbs, ubs, steps;
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  for (int i = 0; i < outerDimsCount; ++i) {
    lbs.push_back(c0);
    steps.push_back(c1);
    

    // Fallback: 如果没有 valid_dims，则遍历整个 Shape
    if (srcType.isDynamicDim(i)) {
      ubs.push_back(rewriter.create<memref::DimOp>(loc, srcMemRef, i));
    } else {
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, srcType.getDimSize(i)));
    }
    
  }

  scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, bodyBuilder);
}

// --------------------------------------------------------------------------
// Specialized Lowering for TLoadOp
// --------------------------------------------------------------------------
struct HighDimLoadDPSLowering : public OpRewritePattern<pto::TLoadOp> {
  using OpRewritePattern<pto::TLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TLoadOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    llvm::errs() << "\n=== [PTOHighDim] Start Matching TLoadOp ===\n";

    Value src = op.getSrc();
    Value dst = op.getDst();
    
    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());

    if (!srcType || !dstType) return failure();

    int64_t rank = srcType.getRank();
    if (rank <= 2 || dstType.getRank() != rank) return failure();

    int64_t outerDimsCount = rank - 2;

    Value padValue = op.getPadValue();
    Value leftPad = op.getLeftPaddingNum();
    Value rightPad = op.getRightPaddingNum();
    Value initCond = op.getInitCondition();
    auto padMode = op.getPadModeAttr();
    bool initOut = op.getInitOutBuffer();

    // [OPTIMIZATION] 使用优化的 Loop 构建器
    buildOptimizedLoopNest(
        rewriter, loc, outerDimsCount, src, srcType,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value subSrc = createSlice(builder, loc, src, srcType, ivs, outerDimsCount, rank);
          Value subDst = createSlice(builder, loc, dst, dstType, ivs, outerDimsCount, rank);

          builder.create<pto::TLoadOp>(
              loc, TypeRange{}, subSrc, subDst, padMode, padValue, leftPad, rightPad, initOut, initCond
          );
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// --------------------------------------------------------------------------
// Specialized Lowering for TStoreOp
// --------------------------------------------------------------------------
struct HighDimStoreDPSLowering : public OpRewritePattern<pto::TStoreOp> {
  using OpRewritePattern<pto::TStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(pto::TStoreOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    llvm::errs() << "\n=== [PTOHighDim] Start Matching TStoreOp ===\n";

    Value src = op.getSrc();
    Value dst = op.getDst();

    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto dstType = dyn_cast<MemRefType>(dst.getType());

    if (!srcType || !dstType) return failure();

    int64_t rank = srcType.getRank();
    if (rank <= 2 || dstType.getRank() != rank) return failure();

    int64_t outerDimsCount = rank - 2;

    // [OPTIMIZATION] 使用优化的 Loop 构建器
    buildOptimizedLoopNest(
        rewriter, loc, outerDimsCount, src, srcType,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) {
          Value subSrc = createSlice(builder, loc, src, srcType, ivs, outerDimsCount, rank);
          Value subDst = createSlice(builder, loc, dst, dstType, ivs, outerDimsCount, rank);

          builder.create<pto::TStoreOp>(
              loc, TypeRange{}, subSrc, subDst
          );
        });

    rewriter.eraseOp(op);
    return success();
  }
};

// --------------------------------------------------------------------------
// Generic Lowering for Elementwise Ops (e.g. Add)
// --------------------------------------------------------------------------
template <typename OpType>
struct HighDimElementwiseDPSLowering : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    llvm::errs() << "\n=== [PTOHighDim] Start Matching Op ===\n";
    op.dump();

    if (op->getNumOperands() == 0) return failure();

    Value driverOperand = op->getOperand(0);
    auto driverType = dyn_cast<MemRefType>(driverOperand.getType());
    if (!driverType) return failure();

    int64_t driverRank = driverType.getRank();
    if (driverRank <= 2) return failure();
    
    int64_t outerDimsCount = driverRank - 2;

    SmallVector<Value> originalOperands;
    for (auto v : op->getOperands()) originalOperands.push_back(v);

    DictionaryAttr opAttrs = op->getAttrDictionary();

    // [OPTIMIZATION] 使用优化的 Loop 构建器
    buildOptimizedLoopNest(
        rewriter, loc, outerDimsCount, driverOperand, driverType,
        [&](OpBuilder &builder, Location loc, ValueRange ivs) -> void {
          SmallVector<Value> newOperands;
          
          for (size_t idx = 0; idx < originalOperands.size(); ++idx) {
            Value operand = originalOperands[idx];
            auto operandType = dyn_cast<MemRefType>(operand.getType());
            
            if (!operandType || operandType.getRank() <= 2) {
                newOperands.push_back(operand);
                continue;
            }

            int64_t opRank = operandType.getRank();
            if (opRank != driverRank) {
                newOperands.push_back(operand);
                continue;
            }

            Value subView = createSlice(builder, loc, operand, operandType, ivs, outerDimsCount, opRank);
            newOperands.push_back(subView);
          }

          builder.create<OpType>(loc, TypeRange{}, newOperands, opAttrs.getValue());
        });

    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOHighDimLoweringPass : public PassWrapper<PTOHighDimLoweringPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOHighDimLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, 
                    memref::MemRefDialect, 
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // 1. Elementwise Ops
    patterns.add<HighDimElementwiseDPSLowering<pto::AddFDpsOp>>(&getContext());
    
    // 2. Load Op
    patterns.add<HighDimLoadDPSLowering>(&getContext());

    // 3. Store Op
    patterns.add<HighDimStoreDPSLowering>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::errs() << "\n// === [PTOHighDimLowering] Result Dump Start ===\n";
    getOperation()->print(llvm::errs());
    llvm::errs() << "\n// === [PTOHighDimLowering] Result Dump End ===\n\n";
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTOHighDimLoweringPass() {
  return std::make_unique<PTOHighDimLoweringPass>();
}
} // namespace pto
} // namespace mlir