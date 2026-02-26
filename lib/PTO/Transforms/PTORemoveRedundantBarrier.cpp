#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h" 
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
 
using namespace mlir;
using namespace mlir::pto;
 
namespace {
 
// ==========================================================
// 更严格的活跃性分析
// ==========================================================
 
// 辅助：判断是否是实质性的资源操作 (Resource Op)
// Wait 和 Set 不算作实质性操作。
// 只有真正消耗计算或带宽的指令才算"活跃"。
bool isResourceOp(Operation *op, Attribute targetPipe) {
    if (auto loadOp = dyn_cast<pto::TLoadOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE2) == targetPipe;
    if (auto storeOp = dyn_cast<pto::TStoreOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_MTE3) == targetPipe;
    if (auto addfOp = dyn_cast<pto::AddFOp>(op)) 
        return pto::PipeAttr::get(op->getContext(), pto::PIPE::PIPE_V) == targetPipe;
    return false;
}
 
// 递归检查 Region 内是否有实质性操作
// 用于深入 scf.if / scf.for 内部查找
bool isPipeUsedInRegion(Region &region, Attribute targetPipe) {
    for (Block &block : region) {
        for (Operation &op : block) {
            // 1. 如果是实质性操作，返回 True
            if (isResourceOp(&op, targetPipe)) return true;
            
            // 2. 递归检查嵌套 (if/for)
            for (Region &nestedRegion : op.getRegions()) {
                if (isPipeUsedInRegion(nestedRegion, targetPipe)) return true;
            }
        }
    }
    return false;
}
 
// 向后扫描：检查 targetPipe 在当前 Block 后续是否"真正"活跃
// WaitOp 不再被视为活跃标志。
// 如果一个 Pipe 后面只剩 Wait，说明它已经完成了工作，发给它的信号是多余的。
bool isPipelineActiveFuture(Block *block, Block::iterator startIt, Attribute targetPipe) {
    for (auto it = startIt; it != block->end(); ++it) {
        Operation *op = &*it;
        
        // 1. 遇到实质性操作 -> 活跃
        if (isResourceOp(op, targetPipe)) return true;
 
        // [注意] 这里故意跳过了 WaitOp 的检查。
        // WaitOp 只是同步原语，不代表该 Pipeline 在"干活"。
 
        // 2. 递归检查嵌套区域 (scf.if, scf.for)
        for (Region &region : op->getRegions()) {
            if (isPipeUsedInRegion(region, targetPipe)) return true;
        }
 
        // 3. 处理 Terminator (跨 Block 检查)
        if (op->hasTrait<OpTrait::IsTerminator>()) {
            // 如果是 Return，肯定死了
            if (isa<func::ReturnOp>(op)) return false;
            
            // 如果是 Yield (scf.if / scf.for)，我们需要看 Parent Block 的后续
            // 这是一个简单的单层 Lookahead，防止 Set 被误删
            if (auto parentOp = block->getParentOp()) {
                Block *parentBlock = parentOp->getBlock();
                if (parentBlock) {
                    // 从 Parent Op 的下一条指令开始查
                    for (auto pIt = std::next(parentOp->getIterator()); pIt != parentBlock->end(); ++pIt) {
                        if (isResourceOp(&*pIt, targetPipe)) return true;
                        
                        // 如果外面还有嵌套，理论上要继续递归，这里保守返回 true
                        if (pIt->getNumRegions() > 0) return true; 
                        
                        // 如果遇到 Return，说明后面真没了
                        if (isa<func::ReturnOp>(&*pIt)) return false;
                    }
                }
            }
            return false; // 如果 Parent Block 后面也没东西，那就是死了
        }
    }
    return false;
}
 
// ==========================================================
// Pass 实现
// ==========================================================
struct PTORemoveRedundantBarrierPass : public PassWrapper<PTORemoveRedundantBarrierPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTORemoveRedundantBarrierPass)
 
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = &getContext();
    
    Attribute attrMTE2 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2);
    Attribute attrMTE3 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE3);
    Attribute attrVec  = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
 
    auto getOpPipe = [&](Operation *op) -> Attribute {
      if (isa<pto::TLoadOp>(op)) return attrMTE2;
      if (isa<pto::TStoreOp>(op)) return attrMTE3;
      if (isa<pto::AddFOp>(op)) return attrVec;
      return {};
    };
 
    llvm::SmallVector<Operation*> opsToErase;
 
    func.walk([&](Block *block) {
      // 记录 Block 内脏状态 (Intra-Block Dirty State)
      // 用于判断是否需要发广播
      llvm::DenseSet<Attribute> intraPipeDirtySet;
 
      for (auto it = block->begin(); it != block->end(); ++it) {
        Operation *op = &*it;
        Attribute pipe = getOpPipe(op);
 
        // === 1. 状态更新 ===
        if (pipe) {
            intraPipeDirtySet.insert(pipe);
            continue; 
        }
 
        // === 2. Barrier 消除 ===
        if (auto barrierOp = dyn_cast<pto::BarrierOp>(op)) {
            Attribute bPipe = barrierOp.getPipe();
            
            // 规则 A: Dead Pipeline
            // 后面没活干了 -> 删除 (保护空气没有意义)
            if (!isPipelineActiveFuture(block, std::next(it), bPipe)) {
                opsToErase.push_back(op);
                continue;
            }
            // 规则 B: Clean Pipeline
            // 管线本来就是干净的 -> 删除
            if (!intraPipeDirtySet.count(bPipe)) {
                opsToErase.push_back(op);
                continue;
            }
            // 规则 C: Subsumed by Set
            // 紧跟 Set -> 删除 (Set 隐含 Barrier)
            auto nextIt = std::next(it);
            if (nextIt != block->end()) {
                if (auto setOp = dyn_cast<pto::SetFlagOp>(&*nextIt)) {
                    if (setOp.getSrcPipe() == bPipe) {
                        opsToErase.push_back(op);
                        continue;
                    }
                }
            }
            // 如果 Barrier 留下了，管线变干净
            intraPipeDirtySet.erase(bPipe);
        }
 
        // === 3. Wait 消除 (幽灵 Wait 消除) ===
        if (auto waitOp = dyn_cast<pto::WaitFlagOp>(op)) {
            Attribute dst = waitOp.getDstPipe();
            
            // 规则: Dead Consumer
            // 如果 dst 后面没有 Resource Op，这个 Wait 是毫无意义的阻塞。
            // 即使逻辑上需要等，但如果等完不干活，等它干嘛？
            if (!isPipelineActiveFuture(block, std::next(it), dst)) {
                opsToErase.push_back(op);
                continue;
            }
        }
 
        // === 4. Set 消除 (死信 & 陈旧广播消除) ===
        if (auto setOp = dyn_cast<pto::SetFlagOp>(op)) {
            Attribute src = setOp.getSrcPipe();
            Attribute dst = setOp.getDstPipe();
 
            // 规则 A: Dead Receiver (死信)
            // 如果 dst 后面没有 Resource Op，发信号也没人用。
            // 注意：因为 isPipelineActiveFuture 忽略了 WaitOp，
            // 所以如果后面只有 Wait <Src, Dst> 而没有 Dst 的实质操作，这里也会判定为 Dead，
            // 从而删除 Set。上面的 Wait 消除逻辑会删除那个 Wait。完美闭环。
            if (!isPipelineActiveFuture(block, std::next(it), dst)) {
                opsToErase.push_back(op);
                continue;
            }
 
            // 规则 B: Stale Broadcast (陈旧广播)
            // 如果 Src 在当前 Block 没脏过 (没干活)，就不要发广播。
            // 这精准删除了 scf.if 中 MTE2->MTE3 的冗余广播，因为 MTE2 在分支里通常是不动的。
            if (!intraPipeDirtySet.count(src)) {
                 opsToErase.push_back(op);
                 continue;
            }
        }
      }
    });
 
    for (Operation *op : opsToErase) op->erase();
  }
};
 
} // namespace
 
namespace mlir {
namespace pto {
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass() {
  return std::unique_ptr<Pass>(new PTORemoveRedundantBarrierPass());
}
} // namespace pto
} // namespace mlir