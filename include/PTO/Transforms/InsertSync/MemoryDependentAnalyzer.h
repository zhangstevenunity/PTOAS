#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_MEMORYDEPENDENTANALYZER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_MEMORYDEPENDENTANALYZER_H
 
#include "PTO/Transforms/InsertSync/SyncCommon.h"
 
namespace mlir {
namespace pto {
 
class MemoryDependentAnalyzer {
public:
  MemoryDependentAnalyzer() = default;
  ~MemoryDependentAnalyzer() = default;
 
  // 检查两组内存信息之间是否存在依赖
  bool DepBetween(const SmallVector<const BaseMemInfo *> &a,
                  const SmallVector<const BaseMemInfo *> &b,
                  DepBaseMemInfoPairVec &depBaseMemInfosVec);
 
  // 检查两个具体的 MemInfo 是否别名
  bool MemAlias(const BaseMemInfo *a, const BaseMemInfo *b);
 
private:
  bool isGMBufferOverlap(const BaseMemInfo *a, const BaseMemInfo *b);
  
  bool isBufferAddressRangeOverlap(const BaseMemInfo *a, const BaseMemInfo *b);
  
  bool isBufferOverlap(const BaseMemInfo *a, const BaseMemInfo *b, 
                       int aIndex, int bIndex);
};
 
} // namespace pto
} // namespace mlir
 
#endif
