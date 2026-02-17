#include "PTO/Transforms/SyncCodegen.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;
 
// ==============================================================================
// 1. Helper Functions
// ==============================================================================
 
static pto::PipeAttr getPipeAttr(Builder &builder, PipelineType pipe) {
  auto odsPipeVal = static_cast<pto::PIPE>(pipe);
  return pto::PipeAttr::get(builder.getContext(), odsPipeVal);
}
 
static pto::EventAttr getEventAttr(Builder &builder, int id) {
  auto odsEventVal = static_cast<pto::EVENT>(id);
  return pto::EventAttr::get(builder.getContext(), odsEventVal);
}
 
static bool IsSyncExist(const SyncOps &list, SyncOperation *newSync) {
  for (auto *existing : list) {
    if (existing == newSync) return true;
    if (existing->GetType() != newSync->GetType()) continue;
    if (existing->GetActualSrcPipe() != newSync->GetActualSrcPipe()) continue;
    if (existing->GetActualDstPipe() != newSync->GetActualDstPipe()) continue;
    if (newSync->isSyncSetType() || newSync->isSyncWaitType()) {
       if (existing->eventIds != newSync->eventIds) continue;
    }
    return true;
  }
  return false;
}
 
static void MergeSyncList(SyncOps &dstList, const SyncOps &srcList) {
  for (auto *sync : srcList) {
    if (!IsSyncExist(dstList, sync)) {
      dstList.push_back(sync);
    }
  }
}
 
// ==============================================================================
// 2. SyncCodegen Implementation
// ==============================================================================
 
void SyncCodegen::Run() {
  MLIRContext *ctx = func_->getContext();
  IRRewriter rewriter(ctx);
  
  UpdateOpInsertSync(rewriter);
 
  // [Optional Debug] 这里的 Debug 打印可以保留或注释掉
  // ...
 
  func_->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op2InsertSync.count(op)) {
      // 处理 PRE Sync
      for (auto &syncBefore : op2InsertSync[op].pipeBefore) {
        SyncInsert(rewriter, op, syncBefore, true);
      }
      // 处理 POST Sync (逆序遍历，为了保持插入后的顺序正确)
      for (auto &syncAfter : llvm::reverse(op2InsertSync[op].pipeAfter)) {
        SyncInsert(rewriter, op, syncAfter, false);
      }
    }
  });
}
 
void SyncCodegen::UpdateOpInsertSync(IRRewriter &rewriter) {
  for (auto &nowElement : syncIR_) {
    if (auto *compoundElement = dyn_cast<CompoundInstanceElement>(nowElement.get())) {
      UpdateCompoundOpInsertSync(compoundElement);
    } else if (auto *placeHolder = dyn_cast<PlaceHolderInstanceElement>(nowElement.get())) {
      updatePlaceHolderOpInsertSync(placeHolder);
    } else if (auto *loopElement = dyn_cast<LoopInstanceElement>(nowElement.get())) {
      UpdateLoopOpInsertSync(loopElement);
    } else if (auto *branchElement = dyn_cast<BranchInstanceElement>(nowElement.get())) {
      UpdateBranchOpInsertSync(branchElement);
    }
  }
}
 
void SyncCodegen::UpdateCompoundOpInsertSync(CompoundInstanceElement *nowCompound) {
  auto &pipeBuild = op2InsertSync[nowCompound->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, nowCompound->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, nowCompound->pipeAfter);
}
 
void SyncCodegen::UpdateLoopOpInsertSync(LoopInstanceElement *nowElement) {
  if (nowElement->getLoopKind() == KindOfLoop::LOOP_END) {
    auto *loopBegin = dyn_cast<LoopInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, loopBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::UpdateBranchOpInsertSync(BranchInstanceElement *nowElement) {
  if (nowElement->getBranchKind() == KindOfBranch::IF_END) {
    auto *branchBegin = dyn_cast<BranchInstanceElement>(syncIR_[nowElement->beginId].get());
    auto &pipeBuild = op2InsertSync[nowElement->elementOp];
    MergeSyncList(pipeBuild.pipeBefore, branchBegin->pipeBefore);
    MergeSyncList(pipeBuild.pipeAfter, nowElement->pipeAfter);
  }
}
 
void SyncCodegen::updatePlaceHolderOpInsertSync(PlaceHolderInstanceElement *placeHolder) {
  // 1. 处理 Virtual Else
  if (placeHolder->isVirtualElse) {
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->parentIfOp);
      if (!ifOp) return;
 
      // 如果还没有 else block，创建一个
      if (!ifOp.elseBlock()) {
          OpBuilder builder(ifOp.getContext());
          // 只有当确实有 Sync 指令需要插入时才创建
          if (!placeHolder->pipeBefore.empty() || !placeHolder->pipeAfter.empty()) {
               Region &elseRegion = ifOp.getElseRegion();
               Block *elseBlock = new Block();
               elseRegion.push_back(elseBlock);
               builder.setInsertionPointToEnd(elseBlock);
               builder.create<scf::YieldOp>(ifOp.getLoc());
          }
      }
      
      // 更新映射：将 Virtual Placeholder 映射到新创建的 Yield Op
      if (ifOp.elseBlock()) {
          placeHolder->elementOp = ifOp.getElseRegion().front().getTerminator();
      } else {
          // 依然没有 Sync 需要插入，直接返回
          return;
      }
  } 
  // 2. 处理 Normal PlaceHolder (Then End or Existing Else End)
  else if (placeHolder->elementOp == placeHolder->parentIfOp) {
      // 之前的 Translator 逻辑把 Normal Placeholder 也映射到了 ifOp
      // 我们需要修正它指向 Yield
      auto ifOp = dyn_cast<scf::IfOp>(placeHolder->elementOp);
      // 判断是 Then 还是 Else
      // 简单判断：看 index。或者 Translator 里直接存 Yield Op。
      // 这里假设 Translator 存的是 IfOp，我们需要找到对应的 Yield。
      // ... 
      // 建议在 Translator 里直接让 elementOp 指向 Yield Op（如果存在）。
  }
 
  // 执行常规的 Sync 插入
  if (!placeHolder->elementOp) return;
  auto &pipeBuild = op2InsertSync[placeHolder->elementOp];
  MergeSyncList(pipeBuild.pipeBefore, placeHolder->pipeBefore);
  MergeSyncList(pipeBuild.pipeAfter, placeHolder->pipeAfter);
}
 
void SyncCodegen::SyncInsert(IRRewriter &rewriter, Operation *op,
                             SyncOperation *sync, bool beforeInsert) {
  if (sync->uselessSync) return;

  // [Fix] 处理补偿逻辑的强制插入点
  Operation *insertAnchorOp = op;
  bool forceBefore = beforeInsert;

  if (sync->isCompensation) {
      // 策略：补偿指令必须插在控制流块的末尾（Terminator 之前）
      
      // Case 1: Anchor 是 scf.if (Virtual Else 的情况)
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          // 我们需要确定是插在 Then 还是 Else。
          // 通常 Analysis 会根据 context 知道，但这里 op 只是 anchor。
          // 我们利用 SyncOperation 的上下文推断，或者更简单地：
          // 如果是 Virtual Else，PTOIRTranslator 应该已经处理了 Block 创建。
          // 如果这里还是 IfOp，说明我们必须进入 Else Region。
          
          if (!ifOp.elseBlock()) {
              // 再次兜底：创建 Else Block
              OpBuilder b(ifOp.getContext());
              Block *elseBlock = new Block();
              ifOp.getElseRegion().push_back(elseBlock);
              b.setInsertionPointToEnd(elseBlock);
              b.create<scf::YieldOp>(ifOp.getLoc());
          }
          
          // 将插入点重定向到 Else Block 的 Yield
          insertAnchorOp = ifOp.getElseRegion().front().getTerminator();
      }
      // Case 2: Anchor 已经是 Terminator (YieldOp)
      else if (op->hasTrait<OpTrait::IsTerminator>()) {
          insertAnchorOp = op;
      }
      // Case 3: 其他情况 (Anchor 指向了 Block 内的某条指令)
      else {
          // 找到该 Block 的 Terminator
          insertAnchorOp = op->getBlock()->getTerminator();
      }

      // 强制在 Terminator 之前插入
      forceBefore = true;
  }

  // 分发创建逻辑，传入修正后的 insertAnchorOp 和 forceBefore
  if (sync->GetType() == SyncOperation::TYPE::PIPE_BARRIER) {
    CreateBarrierOp(rewriter, insertAnchorOp, sync, forceBefore);
  } else if (sync->isSyncSetType() || sync->isSyncWaitType()) {
    if (sync->eventIds.size() == 1) {
      CreateSetWaitOpForSingleBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    } else {
      CreateSetWaitOpForMultiBuffer(rewriter, insertAnchorOp, sync, forceBefore);
    }
  } 
}
 
// [核心修改] 加强版 CreateBarrierOp
void SyncCodegen::CreateBarrierOp(IRRewriter &rewriter, Operation *op,
                                  SyncOperation *sync, bool beforeInsert) {
  // [Fix] 判定是否需要前置插入：如果是显式 Before，或者 Op 是 Terminator (如 Yield)
  bool insertAtPos = beforeInsert || op->hasTrait<OpTrait::IsTerminator>();
 
  // 1. 设置插入点
  if (insertAtPos) {
    rewriter.setInsertionPoint(op);
  } else {
    rewriter.setInsertionPointAfter(op);
  }
 
  // 2. 获取上下文
  Block *block = rewriter.getInsertionBlock();
  Block::iterator ip = rewriter.getInsertionPoint();
  auto currentPipeAttr = getPipeAttr(rewriter, sync->GetActualSrcPipe());
 
  // 3. 窥孔优化 (双向检查)
  // 注意：如果是 Terminator 导致的强制前置插入，我们也应该检查 Prev，因为它是插在末尾
  if (insertAtPos) {
    // PRE 插入：检查前一条指令
    if (ip != block->begin()) {
      if (auto prevBarrier = dyn_cast<pto::BarrierOp>(&*std::prev(ip))) {
        if (prevBarrier.getPipe() == currentPipeAttr) return; // Dedup
      }
    }
  } else {
    // POST 插入：检查当前/下一条指令
    if (ip != block->end()) {
      if (auto nextBarrier = dyn_cast<pto::BarrierOp>(&*ip)) {
        if (nextBarrier.getPipe() == currentPipeAttr) return; // Dedup
      }
    }
  }
 
  // 4. 创建指令
  rewriter.create<pto::BarrierOp>(op->getLoc(), currentPipeAttr);
}
 
void SyncCodegen::CreateSetWaitOpForSingleBuffer(IRRewriter &rewriter,
                                                 Operation *op,
                                                 SyncOperation *sync,
                                                 bool beforeInsert) {
  // [Fix] Terminator 强制前置插入
  if (beforeInsert || op->hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(op);
  } else {
      rewriter.setInsertionPointAfter(op);
  }
 
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  auto eventId = getEventAttr(rewriter, sync->eventIds[0]);
 
  if (sync->isSyncWaitType()) {
    rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  } else {
    rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId);
  }
}
 
void SyncCodegen::CreateSetWaitOpForMultiBuffer(IRRewriter &rewriter,
                                                Operation *op,
                                                SyncOperation *sync,
                                                bool beforeInsert) {
  // 获取多缓冲选择条件
  Value cond = GetBufferSelectCond(rewriter, op, sync);
  if (!cond || sync->eventIds.size() < 2) {
    CreateSetWaitOpForSingleBuffer(rewriter, op, sync, beforeInsert);
    return;
  }
  
  // [Fix] Terminator 强制前置插入
  if (beforeInsert || op->hasTrait<OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(op);
  } else {
      rewriter.setInsertionPointAfter(op);
  }
 
  auto srcPipe = getPipeAttr(rewriter, sync->GetActualSrcPipe());
  auto dstPipe = getPipeAttr(rewriter, sync->GetActualDstPipe());
  auto eventId0 = getEventAttr(rewriter, sync->eventIds[0]);
  auto eventId1 = getEventAttr(rewriter, sync->eventIds[1]);

  auto ifOp = rewriter.create<scf::IfOp>(op->getLoc(), cond, /*withElse=*/true);
  {
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    if (sync->isSyncWaitType()) {
      rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId0);
    } else {
      rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId0);
    }
  }
  {
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    if (sync->isSyncWaitType()) {
      rewriter.create<pto::WaitFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId1);
    } else {
      rewriter.create<pto::SetFlagOp>(op->getLoc(), srcPipe, dstPipe, eventId1);
    }
  }
}
 
Value SyncCodegen::GetBufferSelectCond(IRRewriter &rewriter, Operation *op,
                                       SyncOperation *sync) {
  if (SyncIndex2SelectCond.count(sync->GetSyncIndex())) {
    return SyncIndex2SelectCond[sync->GetSyncIndex()];
  }
 
  auto parentLoop = op->getParentOfType<scf::ForOp>();
  if (!parentLoop) return nullptr;
 
  Value counter;
  if (loop2BufferCounter.count(parentLoop)) {
    counter = loop2BufferCounter[parentLoop];
  } else {
    rewriter.setInsertionPointToStart(parentLoop.getBody());
    Value iv = parentLoop.getInductionVar();
    auto loc = op->getLoc();
    auto i64Ty = rewriter.getI64Type();
    Value ivI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, iv);
    Value c2 = rewriter.create<arith::ConstantIntOp>(loc, 2, 64);
    counter = rewriter.create<arith::RemSIOp>(loc, ivI64, c2);
    loop2BufferCounter[parentLoop] = counter;
  }
 
  rewriter.setInsertionPointAfter(counter.getDefiningOp());
  Value isZero = rewriter.create<arith::CmpIOp>(
      op->getLoc(), arith::CmpIPredicate::eq, counter,
      rewriter.create<arith::ConstantIntOp>(op->getLoc(), 0, 64));
  
  SyncIndex2SelectCond[sync->GetSyncIndex()] = isZero;
  return isZero;
}
