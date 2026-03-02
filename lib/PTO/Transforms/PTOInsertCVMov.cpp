#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" 

using namespace mlir;
using namespace mlir::pto;

namespace {

enum class ComputeDomain {
  VECTOR, 
  CUBE,   
  OTHER   
};

class PTOInsertCVMovPass : public PassWrapper<PTOInsertCVMovPass, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOInsertCVMovPass)

  StringRef getArgument() const override { return "pto-insert-cv-mov"; }
  StringRef getDescription() const override { return "Insert mov instructions between Cube and Vector domains"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    llvm::errs() << "[PTO-Insert-CV-Mov] Running...\n";
    func::FuncOp func = getOperation();
    
    // 缓存已生成的 Mov 指令
    llvm::DenseMap<std::pair<Value, ComputeDomain>, Value> movCache;

    func.walk([&](Operation *op) {
      ComputeDomain consumerDomain = getOpDomain(op);
      if (consumerDomain == ComputeDomain::OTHER) return;

      OpBuilder builder(op); 

      for (OpOperand &operand : op->getOpOperands()) {
        Value producerVal = operand.get();
        Operation *producerOp = producerVal.getDefiningOp();
        
        if (!producerOp) continue;

        ComputeDomain producerDomain = getOpDomain(producerOp);

        if (producerDomain == ComputeDomain::OTHER) continue;

        if (producerDomain != consumerDomain) {
          auto cacheKey = std::make_pair(producerVal, consumerDomain);
          
          if (movCache.count(cacheKey)) {
            operand.set(movCache[cacheKey]);
          } else {
            auto movOp = builder.create<pto::TMovOp>(
                op->getLoc(), 
                producerVal.getType(), 
                producerVal
            );
            
            Value movResult = movOp.getResult();
            operand.set(movResult);
            movCache[cacheKey] = movResult;
          }
        }
      }
    });

    func.print(llvm::errs());
    llvm::errs() << "\n================ [PTOInsertCVMov] end! ==================\n";
  }

private:
  ComputeDomain getOpDomain(Operation *op) {
    if (llvm::isa<pto::TMatmulOp>(op)) return ComputeDomain::CUBE;
    if (llvm::isa<pto::TMatmulAccOp>(op)) return ComputeDomain::CUBE;
    if (llvm::isa<pto::TAddOp>(op)) return ComputeDomain::VECTOR;
    return ComputeDomain::OTHER;
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTOInsertCVMovPass() {
  return std::make_unique<PTOInsertCVMovPass>();
}
} // namespace pto
} // namespace mlir