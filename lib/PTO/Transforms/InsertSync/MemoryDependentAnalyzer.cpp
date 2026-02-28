#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Debug.h"
 
#define DEBUG_TYPE "pto-inject-sync"
 
using namespace mlir;
using namespace mlir::pto;

static bool isTraceEnabled() {
  return isInsertSyncDebugEnabled(InsertSyncDebugLevel::Trace);
}
 
// [Debug] 打印 Value 详细信息
static void printValueDebug(const char* tag, Value v) {
  if (!isTraceEnabled())
    return;
  llvm::errs() << tag << ": ";
  if (!v) {
    llvm::errs() << "NULL\n";
    return;
  }
  
  if (auto *op = v.getDefiningOp()) {
    llvm::errs() << "OpResult defined by [" << op->getName() << "]";
  } else {
    llvm::errs() << "BlockArgument";
  }
  llvm::errs() << " | Type: " << v.getType() << "\n";
}
 
// [Fix & Debug] 增强版 GetRealRoot
static Value GetRealRoot(Value v) {
  const bool trace = isTraceEnabled();
  if (trace) {
    llvm::errs() << "  [Trace] GetRealRoot Start:\n";
    printValueDebug("    Current", v);
  }
  
  int depth = 0;
  const int maxDepth = 20;
 
  while (v && depth++ < maxDepth) {
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
        if (trace)
          llvm::errs() << "    -> Reached BlockArgument. Stop.\n";
        break; 
    }
 
    if (auto op = dyn_cast<memref::CollapseShapeOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit CollapseShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ExpandShapeOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ExpandShapeOp. Peel off.\n";
        v = op.getSrc();
        continue;
    }
    if (auto op = dyn_cast<memref::ViewOp>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ViewOp. Peel off.\n";
        v = op.getSource();
        continue;
    }
    if (auto view = dyn_cast<ViewLikeOpInterface>(defOp)) {
        if (trace)
          llvm::errs() << "    -> Hit ViewLikeInterface. Peel off.\n";
        v = view.getViewSource();
        continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(defOp)) {
        v = cast.getSource();
        continue;
    }
    if (auto reCast = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
        v = reCast.getSource();
        continue;
    }
 
    if (trace) {
      llvm::errs() << "    -> Hit Alloc/Other [" << defOp->getName()
                   << "]. Stop.\n";
    }
    break;
  }
  return v;
}
 
bool MemoryDependentAnalyzer::DepBetween(
    const SmallVector<const BaseMemInfo *> &a,
    const SmallVector<const BaseMemInfo *> &b,
    DepBaseMemInfoPairVec &depBaseMemInfosVec) {
  
  // [Debug Log] 关键入口信息
  if (isTraceEnabled()) {
    llvm::errs() << "\n[DepBetween] Checking dependency...\n";
    llvm::errs() << "  Vec A Size: " << a.size() << "\n";
    llvm::errs() << "  Vec B Size: " << b.size() << "\n";
  }
 
  bool hasAlias = false;
  for (auto &i : a) {
    for (auto &j : b) {
      if (MemAlias(i, j)) {
        depBaseMemInfosVec.push_back(std::make_pair(i, j));
        hasAlias = true;
      }
    }
  }
  return hasAlias;
}
 
bool MemoryDependentAnalyzer::MemAlias(const BaseMemInfo *a,
                                       const BaseMemInfo *b) {
  pto::AddressSpace as = a->scope;
  pto::AddressSpace bs = b->scope;
 
  // [Debug Log] 打印比较对象
  if (isTraceEnabled()) {
    llvm::errs() << "  [MemAlias Check]\n";
    printValueDebug("    Root A", a->rootBuffer);
    printValueDebug("    Root B", b->rootBuffer);
    llvm::errs() << "    Scope A: " << (int)as << ", Scope B: " << (int)bs
                 << "\n";
  }
 
  if (as != bs) {
    if (isTraceEnabled())
      llvm::errs() << "    -> Scope Mismatch. False.\n";
    return false;
  }
 
  // 1. GM 内存
  if (as == pto::AddressSpace::GM) {
    return isGMBufferOverlap(a, b);
  }
 
  // 2. Local Memory (UB/L1)
  
  if (a->rootBuffer == b->rootBuffer) {
    if (a->baseAddresses.empty() || b->baseAddresses.empty()) return true;
    return isBufferAddressRangeOverlap(a, b);
  }
 
  // 2.2 深层比较：穿透 View
  Value realRootA = GetRealRoot(a->rootBuffer);
  Value realRootB = GetRealRoot(b->rootBuffer);
 
  if (isTraceEnabled()) {
    llvm::errs() << "    [Deep Check] Surface Roots differ. Digging deeper...\n";
    printValueDebug("      Real Root A", realRootA);
    printValueDebug("      Real Root B", realRootB);
  }
 
  if (realRootA == realRootB && realRootA != nullptr) {
      if (isTraceEnabled())
        llvm::errs() << "      -> MATCH! Real roots are the same.\n";
      return true;
  } else {
      if (isTraceEnabled())
        llvm::errs() << "      -> Mismatch. Real roots differ.\n";
  }
 
  return false;
}
 
bool MemoryDependentAnalyzer::isGMBufferOverlap(const BaseMemInfo *a,
                                                const BaseMemInfo *b) {
  if (a->rootBuffer != b->rootBuffer) {
    Value realRootA = GetRealRoot(a->rootBuffer);
    Value realRootB = GetRealRoot(b->rootBuffer);
    
    if (realRootA != realRootB) {
        return false;
    }
    return true; 
  }
 
  if (a->baseAddresses.empty() || b->baseAddresses.empty()) return true; 
  if (a->allocateSize == 0 || b->allocateSize == 0) return true;
 
  return isBufferAddressRangeOverlap(a, b);
}
 
bool MemoryDependentAnalyzer::isBufferAddressRangeOverlap(
    const BaseMemInfo *a, const BaseMemInfo *b) {
  int aBaseAddressesSize = static_cast<int>(a->baseAddresses.size());
  int bBaseAddressesSize = static_cast<int>(b->baseAddresses.size());
  
  for (int i = 0; i < aBaseAddressesSize; i++) {
    for (int j = 0; j < bBaseAddressesSize; j++) {
      if (isBufferOverlap(a, b, i, j)) {
        return true;
      }
    }
  }
  return false;
}
 
bool MemoryDependentAnalyzer::isBufferOverlap(const BaseMemInfo *a,
                                              const BaseMemInfo *b, int aIndex,
                                              int bIndex) {
  uint64_t aStart = a->baseAddresses[aIndex];
  uint64_t bStart = b->baseAddresses[bIndex];
  uint64_t aEnd = aStart + a->allocateSize;
  uint64_t bEnd = bStart + b->allocateSize;
 
  uint64_t maxStart = std::max(aStart, bStart);
  uint64_t minEnd = std::min(aEnd, bEnd);
 
  return maxStart < minEnd;
}
