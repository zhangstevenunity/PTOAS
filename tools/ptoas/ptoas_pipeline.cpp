// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===--- ptoas_pipeline.cpp ---------------------------------------------------------===//
// PTOAS pipeline construction: arch resolution, build-level validation,
// auto-sync/frontend passes, VPTO/EmitC backend pipelines, and the
// compilePTOASModule entry.
//===----------------------------------------------------------------------===//

#include "ptoas_internal.h"

#include "ptoas.h"

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/VMIUtils.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "PTO/Transforms/CppPostprocess.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "VPTOHostStubEmission.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "ptobc/ptobc_decode.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>

#include <sys/types.h>
#include <unistd.h>


using namespace mlir;
using namespace pto;

namespace {
/// Materialize the implicit no-inline semantics of `pto.simt_entry` using the
/// standard Func dialect attribute understood by MLIR's public inliner
/// extension and by later LLVM lowering.
struct ApplySIMTEntryNoInlinePass final
    : public PassWrapper<ApplySIMTEntryNoInlinePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ApplySIMTEntryNoInlinePass)

  void runOnOperation() final {
    for (func::FuncOp func : getOperation().getOps<func::FuncOp>()) {
      if (func->hasAttr(pto::kPTOSimtEntryAttrName)) {
        func->setAttr("no_inline", UnitAttr::get(func.getContext()));
      }
    }
  }
};

/// LLVM 21 runs the EmitC expression patterns without the greedy driver's
/// generic operation folding. LLVM 19 cannot disable that folding, which can
/// erase an expression while the EmitC pattern is rewriting it. Apply the
/// same EmitC rewrite directly so PTOAS retains LLVM 21 expression semantics.
///
/// LLVM 19's C++ emitter also loses the enclosing precedence after it adds
/// parentheses around a nested expression. Keep conditional expressions as
/// explicit temporaries when another C expression consumes them so a ternary
/// can never be flattened into an arithmetic expression with changed meaning.
struct FormEmitCExpressionsCompatPass final
    : public PassWrapper<FormEmitCExpressionsCompatPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormEmitCExpressionsCompatPass)

  static bool containsConditionalOperator(emitc::ExpressionOp expression) {
    for (Operation &op : expression.getBody()->without_terminator()) {
      if (isa<emitc::ConditionalOp>(op)) {
        return true;
      }
    }
    return false;
  }

  static bool foldExpression(emitc::ExpressionOp expression,
                             IRRewriter &rewriter) {
    bool changed = false;
    for (Operation &op : llvm::make_early_inc_range(
             expression.getBody()->without_terminator())) {
      auto apply = dyn_cast<emitc::ApplyOp>(op);
      const bool isAddressOf =
          apply && apply.getApplicableOperator() == "&";
      if (isAddressOf) {
        continue;
      }

      for (OpOperand &operand : llvm::make_early_inc_range(op.getOpOperands())) {
        auto producer = operand.get().getDefiningOp<emitc::ExpressionOp>();
        if (!producer || !producer.getResult().hasOneUse() ||
            producer.hasSideEffects()) {
          continue;
        }

        if (producer.getDoNotInline()) {
          continue;
        }

        if (containsConditionalOperator(producer)) {
          producer.setDoNotInline(true);
          changed = true;
          continue;
        }

        rewriter.setInsertionPoint(&op);
        IRMapping mapper;
        for (Operation &toClone : producer.getBody()->without_terminator()) {
          Operation *clone = rewriter.clone(toClone, mapper);
          mapper.map(&toClone, clone);
        }

        Operation *clonedRoot = mapper.lookup(producer.getRootOp());
        assert(clonedRoot && clonedRoot->getNumResults() == 1 &&
               "expected a cloned single-result EmitC expression root");
        rewriter.replaceOp(producer, clonedRoot->getResults());
        changed = true;
      }
    }
    return changed;
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());
    module.walk([&builder](Operation *op) {
      const bool isTopLevelSingleResultExpression =
          op->hasTrait<OpTrait::emitc::CExpression>() &&
          !op->getParentOfType<emitc::ExpressionOp>() &&
          op->getNumResults() == 1;
      if (isTopLevelSingleResultExpression) {
        emitc::createExpression(op, builder);
      }
    });

    IRRewriter rewriter(&getContext());
    bool changed;
    do {
      changed = false;
      module.walk<WalkOrder::PostOrder>(
          [&changed, &rewriter](emitc::ExpressionOp expression) {
            if (foldExpression(expression, rewriter)) {
              changed = true;
            }
          });
    } while (changed);
  }
};

static std::string normalizeArch(llvm::StringRef arch) {
  std::string normalized = arch.str();
  for (char &c : normalized) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return normalized;
}

static bool isA2A3Arch(llvm::StringRef arch) {
  std::string normalized = normalizeArch(arch);
  return normalized == "a2" || normalized == "a3";
}

static bool isSupportedPTOASTargetArch(llvm::StringRef arch) {
  std::string normalized = normalizeArch(arch);
  return normalized == "a2" || normalized == "a3" || normalized == "a5";
}

static std::optional<std::string> getModuleTargetArchAttr(ModuleOp module) {
  auto attr = module->getAttrOfType<mlir::StringAttr>("pto.target_arch");
  if (!attr) {
    return std::nullopt;
  }
  std::string arch = normalizeArch(attr.getValue());
  if (!isSupportedPTOASTargetArch(arch)) {
    return std::nullopt;
  }
  return arch;
}

static std::string resolveEffectiveTargetArch(ModuleOp module,
                                              llvm::StringRef fallbackArch) {
  if (std::optional<std::string> arch = getModuleTargetArchAttr(module)) {
    return *arch;
  }

  std::optional<std::string> childArch;
  for (ModuleOp child : module.getOps<ModuleOp>()) {
    std::optional<std::string> arch = getModuleTargetArchAttr(child);
    if (!arch) {
      continue;
    }
    if (!childArch) {
      childArch = std::move(arch);
      continue;
    }
    if (*childArch != *arch) {
      return normalizeArch(fallbackArch);
    }
  }
  if (childArch) {
    return *childArch;
  }

  std::string fallback = normalizeArch(fallbackArch);
  if (!isSupportedPTOASTargetArch(fallback)) {
    return "a3";
  }
  return fallback;
}
} // namespace

static LogicalResult applyConfiguredPassManagerCLOptions(
    PassManager &pm, llvm::StringRef pipelineName,
    llvm::raw_ostream &diagOS = llvm::errs()) {
  if (succeeded(mlir::applyPassManagerCLOptions(pm))) {
    return success();
  }
  diagOS << "Error: failed to apply MLIR pass manager command-line options for "
         << pipelineName << ".\n";
  return failure();
}

struct EmitCFunctionOrder {
  SmallVector<emitc::FuncOp> declarations;
  SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;
  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> outgoing;
};

static void collectEmitCFunctions(ModuleOp module, EmitCFunctionOrder &order) {
  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      order.declarations.push_back(func);
      continue;
    }
    order.definitions.push_back(func);
    order.definitionsByName[func.getSymNameAttr()] = func;
    order.indegree[func.getOperation()] = 0;
  }
}

static LogicalResult buildEmitCFunctionCallGraph(EmitCFunctionOrder &order) {
  for (auto caller : order.definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, kSeenCalleeInlineCapacity> seenCallees;
    bool hasCycle = false;
    caller.walk([&callerOp, &order, &seenCallees,
                 &hasCycle](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr) {
        return;
      }
      auto it = order.definitionsByName.find(calleeAttr.getLeafReference());
      if (it == order.definitionsByName.end()) {
        return;
      }
      Operation *calleeOp = it->second.getOperation();
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second) {
        return;
      }
      order.outgoing[calleeOp].push_back(callerOp);
      ++order.indegree[callerOp];
    });
    if (hasCycle) {
      return caller.emitOpError()
             << "recursive function calls are not supported for EmitC C++ "
                "emission";
    }
  }
  return success();
}

static LogicalResult sortEmitCFunctionDefinitions(
    ModuleOp module, const EmitCFunctionOrder &order,
    SmallVectorImpl<emitc::FuncOp> &sortedDefinitions) {
  llvm::DenseMap<Operation *, unsigned> remaining = order.indegree;
  SmallVector<Operation *> ready;
  for (auto func : order.definitions) {
    if (remaining[func.getOperation()] == 0) {
      ready.push_back(func.getOperation());
    }
  }

  while (!ready.empty()) {
    Operation *next = ready.front();
    ready.erase(ready.begin());
    sortedDefinitions.push_back(cast<emitc::FuncOp>(next));
    auto outgoingIt = order.outgoing.find(next);
    if (outgoingIt == order.outgoing.end()) {
      continue;
    }
    for (Operation *user : outgoingIt->second) {
      unsigned &userIndegree = remaining[user];
      if (--userIndegree == 0) {
        ready.push_back(user);
      }
    }
  }

  const bool hasSortedAllDefinitions =
      sortedDefinitions.size() == order.definitions.size();
  if (hasSortedAllDefinitions) {
    return success();
  }
  return module.emitError()
         << "cyclic function call graph is not supported for EmitC C++ emission";
}

static void moveEmitCFunctionsToOrder(
    ModuleOp module, ArrayRef<emitc::FuncOp> desiredOrder) {
  Block &body = module.getBodyRegion().front();
  Operation *anchor = nullptr;
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op)) {
      anchor = &op;
      break;
    }
  }
  if (!anchor) {
    return;
  }

  auto advanceAnchor = [&anchor]() {
    while (anchor) {
      anchor = anchor->getNextNode();
      if (!anchor || isa<emitc::FuncOp>(anchor)) {
        break;
      }
    }
  };
  for (emitc::FuncOp func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceAnchor();
      continue;
    }
    if (anchor) {
      func->moveBefore(anchor);
    } else {
      func->moveBefore(&body, body.end());
    }
  }
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  EmitCFunctionOrder order;
  collectEmitCFunctions(module, order);
  if (failed(buildEmitCFunctionCallGraph(order))) {
    return failure();
  }
  SmallVector<emitc::FuncOp> sortedDefinitions;
  if (failed(sortEmitCFunctionDefinitions(module, order, sortedDefinitions))) {
    return failure();
  }
  const bool onlyOneDefinition = order.definitions.size() <= 1;
  const bool noReorderingNeeded = order.declarations.empty() && onlyOneDefinition;
  if (noReorderingNeeded) {
    return success();
  }

  SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(order.declarations.begin(), order.declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());
  moveEmitCFunctionsToOrder(module, desiredOrder);
  return success();
}

// --------------------------------------------------------------------------
// Command Line Options


enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

static PTOBuildLevel defaultBuildLevel() {
  return PTOBuildLevel::Level2;
}

static bool parseBuildLevel(llvm::StringRef levelStr, PTOBuildLevel &out) {
  std::string s = levelStr.str();
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (s == "level1") {
    out = PTOBuildLevel::Level1;
    return true;
  }
  if (s == "level2") {
    out = PTOBuildLevel::Level2;
    return true;
  }
  if (s == "level3") {
    out = PTOBuildLevel::Level3;
    return true;
  }
  return false;
}

struct ReserveBufferMemSpec {
  uint64_t capacityBytes = 0;
  uint64_t alignmentBytes = 1;
};

static ReserveBufferMemSpec getReserveBufferMemSpec(PTOArch arch,
                                                    AddressSpace space) {
  switch (space) {
  case AddressSpace::VEC:
    return {arch == PTOArch::A5 ? 253952uLL : 196608uLL, 256};
  case AddressSpace::MAT:
    return {524288uLL, 256};
  case AddressSpace::LEFT:
  case AddressSpace::RIGHT:
  case AddressSpace::ACC:
  case AddressSpace::BIAS:
  case AddressSpace::SCALING:
  case AddressSpace::GM:
  case AddressSpace::Zero:
    break;
  }
  return {};
}

static LogicalResult validateReserveBufferBase(pto::ReserveBufferOp op,
                                               PTOArch arch) {
  auto baseAttr = op.getBaseAttr();
  if (!baseAttr) {
    return op.emitError("expects explicit 'base'");
  }

  int64_t signedBase = baseAttr.getInt();
  if (signedBase < 0) {
    return op.emitError("expects 'base' to be non-negative when present");
  }

  ReserveBufferMemSpec spec =
      getReserveBufferMemSpec(arch, op.getLocation().getAddressSpace());
  uint64_t base = static_cast<uint64_t>(signedBase);
  if (base % spec.alignmentBytes != 0) {
    return op.emitError("expects 'base' to be aligned to ")
           << spec.alignmentBytes << " bytes for "
           << stringifyEnum(op.getLocation().getAddressSpace());
  }

  uint64_t size = static_cast<uint64_t>(op.getSize());
  if (base > spec.capacityBytes || size > spec.capacityBytes - base) {
    return op.emitError("reserved range exceeds ")
           << stringifyEnum(op.getLocation().getAddressSpace())
           << " capacity: base " << base << " + size " << size
           << " > " << spec.capacityBytes << " bytes";
  }

  return success();
}

static bool validateReserveBufferLevelRules(ModuleOp module,
                                            PTOBuildLevel level) {
  bool failed = false;
  PTOArch arch = getTargetArch(module);
  module.walk([&arch, level, &failed](pto::ReserveBufferOp op) {
    if (level != PTOBuildLevel::Level3) {
      if (op.getAutoAlloc()) {
        if (op.getBaseAttr()) {
          op.emitError("unexpected 'base' on auto reserve_buffer: "
                       "level1/level2 assign it in pto-plan-memory");
          failed = true;
        }
        return;
      }

      if (op.getBaseAttr()) {
        (void)validateReserveBufferBase(op, arch);
      }
      op.emitError("pto.reserve_buffer with explicit 'base' (auto = false) is "
                   "not supported when --pto-level=level1 or level2; use "
                   "--pto-level=level3 or set auto = true");
      failed = true;
      return;
    }

    if (op.getAutoAlloc() || !op.getBaseAttr()) {
      op.emitError("pto.reserve_buffer requires 'auto = false' and explicit "
                   "'base' when --pto-level=level3");
      failed = true;
      return;
    }

    if (mlir::failed(validateReserveBufferBase(op, arch))) {
      failed = true;
    }
  });
  return !failed;
}

static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";

static bool parseAutoSyncTailHint(llvm::StringRef hintStr, std::string &normalized) {
  std::string s = hintStr.str();
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (s == "barrier-all" || s == "barrier_all" || s == "default") {
    normalized = kAutoSyncTailPolicyBarrierAll.str();
    return true;
  }
  if (s == "mte3-to-s-event0" || s == "mte3_to_s_event0" ||
      s == "setwait-mte3-to-s-event0" ||
      s == "setwait_mte3_to_s_event0") {
    normalized = kAutoSyncTailPolicyMte3ToSEvent0.str();
    return true;
  }
  return false;
}

static LogicalResult emitSharedPreBackendSeamIR(ModuleOp module,
                                                llvm::StringRef outputPath) {
  if (outputPath.empty()) {
    return success();
  }

  if (outputPath == "-") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
    return success();
  }

  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Error: failed to open seam IR file '" << outputPath
                 << "': " << ec.message() << "\n";
    return failure();
  }

  module->print(outputFile.os());
  outputFile.os() << "\n";
  outputFile.keep();
  return success();
}

static void printSharedPreBackendSeamIR(ModuleOp module) {
  module->print(llvm::errs());
  llvm::errs() << "\n";
}

static bool hasUnexpandedTileOps(ModuleOp module) {
  bool found = false;
  module.walk([&found](Operation *op) {
    if (found) {
      return;
    }
    if (isa<pto::OpPipeInterface>(op)) {
      found = true;
      return;
    }

    // A pure PTODSL tileop can contain only low-level compute plus a SIMT
    // launch, so it has no high-level TileOp interface to trigger this path.
    // It still needs tile-handle materialization, backend-helper inlining, and
    // tile_buf_addr folding before VPTO emission.
    if (auto func = dyn_cast<func::FuncOp>(op);
        func && func->hasAttr("pto.tileop.helper")) {
      found = true;
    }
  });
  return found;
}

namespace {
struct NarrowUnusedMultiResultProvenancePass
    : public PassWrapper<NarrowUnusedMultiResultProvenancePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      NarrowUnusedMultiResultProvenancePass)

  void runOnOperation() override {
    narrowUnusedMultiResultProvenanceLocs(getOperation());
  }
};
} // namespace

static std::unique_ptr<Pass> createNarrowUnusedMultiResultProvenancePass() {
  return std::make_unique<NarrowUnusedMultiResultProvenancePass>();
}

namespace {
static SmallVector<func::FuncOp> collectSharedPipelineFunctions(ModuleOp module) {
  SmallVector<func::FuncOp> functions;
  // Object compilation promotes backend children to top-level compile units.
  // Preserve recursive traversal only for user-visible IR modes, which retain
  // the authored container shape for debugging.
  if (emitMlirIR) {
    module.walk([&functions](func::FuncOp funcOp) { functions.push_back(funcOp); });
  } else {
    llvm::append_range(functions, module.getOps<func::FuncOp>());
  }
  return functions;
}

struct SerialAutoSyncPass
    : public PassWrapper<SerialAutoSyncPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SerialAutoSyncPass)

  enum class Mode { InsertSync, Bufid, BarrierAll };

  SerialAutoSyncPass(Mode mode, bool enableBufidDebug)
      : mode(mode), enableBufidDebug(enableBufidDebug) {}

  void runOnOperation() override {
    OpPassManager functionPM(func::FuncOp::getOperationName());
    switch (mode) {
    case Mode::InsertSync:
      functionPM.addPass(pto::createPTOInsertSyncPass());
      break;
    case Mode::Bufid: {
      PTOBufidSyncOptions options;
      options.enableBufidSyncDebug = enableBufidDebug;
      functionPM.addPass(pto::createPTOBufidSyncPass(options));
      break;
    }
    case Mode::BarrierAll:
      functionPM.addPass(pto::createPTOInjectBarrierAllSyncPass());
      break;
    }

    for (func::FuncOp funcOp :
         collectSharedPipelineFunctions(getOperation())) {
      if (failed(runPipeline(functionPM, funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  Mode mode;
  bool enableBufidDebug;
};
} // namespace

namespace {
struct SerialFrontendPipeLoweringPass
    : public PassWrapper<SerialFrontendPipeLoweringPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SerialFrontendPipeLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, pto::PTODialect>();
  }

  void runOnOperation() override {
    OpPassManager functionPM(func::FuncOp::getOperationName());
    functionPM.addPass(pto::createPTOAssignDefaultFrontendPipeIdPass());
    functionPM.addPass(pto::createPTOLowerFrontendPipeOpsPass());

    // Fixpipe frontend verifiers resolve peer contracts by inspecting sibling
    // functions. Running this function pipeline through a regular nested pass
    // adaptor allows one function to be verified while another function is
    // still mutating its pipe ops. Keep these two small passes serial so every
    // verifier observes either the complete frontend or complete lowered form.
    for (func::FuncOp funcOp :
         collectSharedPipelineFunctions(getOperation())) {
      if (failed(runPipeline(functionPM, funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

static std::unique_ptr<Pass> createSerialFrontendPipeLoweringPass() {
  return std::make_unique<SerialFrontendPipeLoweringPass>();
}

static bool shouldDeclareVariablesAtTop(ModuleOp module) {
  auto hasMultiBlockFunc = [](auto func) { return func.getBlocks().size() > 1; };
  return llvm::any_of(module.getOps<func::FuncOp>(), hasMultiBlockFunc) ||
         llvm::any_of(module.getOps<emitc::FuncOp>(), hasMultiBlockFunc);
}

static void appendVMISemanticPipeline(OpPassManager &pm);

static VPTOSchedulerCLIMode
getEffectiveVPTOSchedulerMode(llvm::StringRef arch) {
  if (vptoSchedulerMode.getNumOccurrences() == 0) {
    return arch == "a5" ? VPTOSchedulerCLIMode::On : VPTOSchedulerCLIMode::Off;
  }
  return vptoSchedulerMode;
}

static void prepareVPTOForEmission(PassManager &pm,
                                   VPTOSchedulerCLIMode schedulerMode) {
  auto &kernelModulePM = pm.nest<ModuleOp>();
  // VPTO LLVM emission lowers pto.barrier to the backend barrier intrinsic.
  // A5 does not support a standalone PIPE_V barrier; vector barriers are either
  // unnecessary or must be removed before LLVM emission. Upper-level
  // programming frameworks may still produce pto.barrier(PIPE_V) from generic
  // storage-sync constructs, so run sync-to-pipe legalization here and let the
  // backend checks catch any illegal barrier that still leaks through.
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createLoweringSyncToPipePass());
  // Persistent fragment loops must be fully unrolled before fragment
  // analysis/materialization; promote them ahead of the unroll pass.
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOPromotePersistentFragmentLoopsPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOUnrollLoopsPass());
  kernelModulePM.addPass(createSCCPPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(createCSEPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOAnalyzeSIMTPersistentFragmentPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOMaterializeSIMTPersistentFragmentPass());
  kernelModulePM.addPass(pto::createPTOOutlineSIMTSectionsPass());
  kernelModulePM.addPass(pto::createVPTOPtrNormalizePass());
  kernelModulePM.addPass(pto::createVPTOPtrCastCleanupPass());
  kernelModulePM.addPass(pto::createVPTOOptimizeVcvtPass());
  kernelModulePM.addPass(pto::createVPTOMaskSimplifyPass());
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      createVPTOExpandWrapperOpsPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTOInferVPTOVecScopePass());
  if (enableSoftPostUpdate) {
    kernelModulePM.addPass(pto::createVPTOSoftPostUpdatePass());
  }
  // Hoist loop-invariant guarded address chains out of scf.if regions before
  // the generic LICM (which only inspects top-level loop-body operations).
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createVPTOGuardedLICMPass());
  kernelModulePM.addPass(createLoopInvariantCodeMotionPass());
  kernelModulePM.addNestedPass<func::FuncOp>(
      pto::createPTONarrowVPTOLoopCountersPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(createCSEPass());
  // SoftOps are materialized only after all VPTO optimization and layout
  // decisions.  The materializer creates a temporary func.call; inline it
  // immediately so the final legality check sees the actual VPTO sequence.
  kernelModulePM.addPass(pto::createPTOExpandSoftLibPass());
  kernelModulePM.addPass(pto::createPTOInlineLibCallPass());
  kernelModulePM.addPass(createCanonicalizerPass());
  kernelModulePM.addPass(createCSEPass());
  // Reconstruct the optimized reduction tree before scheduling so the
  // scheduler sees the final MI instruction set and dependencies.
  kernelModulePM.addPass(pto::createVPTOCombineReductionsPass());
  kernelModulePM.addPass(createCSEPass());
  if (schedulerMode != VPTOSchedulerCLIMode::Off) {
    pto::VPTOSchedulerOptions schedulerOptions;
    schedulerOptions.mode =
        schedulerMode == VPTOSchedulerCLIMode::Analyze ? "analyze" : "on";
    schedulerOptions.trace = vptoSchedulerTrace;
    kernelModulePM.addPass(pto::createVPTOSchedulerPass(schedulerOptions));
  }
  kernelModulePM.addPass(pto::createPTOValidateVPTOEmissionIRPass());
}

static void appendA5VPTOPostLoweringFusionPipeline(OpPassManager &kernelModulePM) {
  kernelModulePM.addPass(pto::createPTOLowLevelLoopFusionPass());
  kernelModulePM.addPass(mlir::createCanonicalizerPass());
  kernelModulePM.addPass(mlir::createCSEPass());
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOFusionPredicateElisionPass());
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOFusionLoadStoreElisionPass());
  if (enableVexpdifFusion) {
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOVexpdifFusionPass());
  }
  if (enableUnrollAfterLoopFusion) {
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOUnrollAfterLoopFusionPass());
    kernelModulePM.addPass(mlir::createCanonicalizerPass());
    kernelModulePM.addPass(mlir::createCSEPass());
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionLoadStoreElisionPass());
    // Unrolling and cleanup can expose new vsub + vexp patterns, so run
    // vexpdif fusion again before flattening the regions.
    if (enableVexpdifFusion) {
      kernelModulePM.addNestedPass<mlir::func::FuncOp>(
          pto::createPTOVexpdifFusionPass());
    }
  }
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOFlattenFusionRegionPass());
  kernelModulePM.addPass(mlir::createCSEPass());
}

static void lowerPTOToVPTOBackend(PassManager &pm, ModuleOp module) {
  auto &kernelModulePM = pm.nest<ModuleOp>();
  auto moduleArchAttr =
      module->getAttrOfType<mlir::StringAttr>("pto.target_arch");
  const bool isA2A3 = moduleArchAttr && isA2A3Arch(moduleArchAttr.getValue());
  const bool opFusionEnabled = enableOpFusion == llvm::cl::BOU_TRUE;
  const bool enableA5VPTOPostLoweringFusionLifecycle =
      opFusionEnabled && moduleArchAttr && moduleArchAttr.getValue() == "a5";

  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createLowerPTOToUBufOpsPass());
  if (isA2A3) {
    kernelModulePM.addNestedPass<mlir::func::FuncOp>(
        memref::createExpandStridedMetadataPass());
    kernelModulePM.addPass(mlir::createCanonicalizerPass());
    return;
  }

  kernelModulePM.addPass(pto::createExpandTileOpPass());

  kernelModulePM.addPass(pto::createPTOInlineLibCallPass());
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createFoldTileBufIntrinsicsPass("shape-only"));
  if (enableA5VPTOPostLoweringFusionLifecycle) {
    appendA5VPTOPostLoweringFusionPipeline(kernelModulePM);
  }
  kernelModulePM.addNestedPass<mlir::func::FuncOp>(
      pto::createFoldTileBufIntrinsicsPass("addr-only"));
  kernelModulePM.addPass(mlir::createSCCPPass());
  kernelModulePM.addPass(mlir::createCanonicalizerPass());
}

static pto::VPTOEmissionOptions
buildVPTOEmissionOptions(const pto::CANNVersion &cannVersion,
                         llvm::StringRef targetArch) {
  pto::VPTOEmissionOptions options;
  options.dumpVPTOIR = false;
  options.targetTriple = "hiipu64-hisilicon-cce";
  options.cannVersion = cannVersion;
  std::string arch = normalizeArch(targetArch);
  if (isA2A3Arch(arch)) {
    options.march = "dav-c220-vec";
  }
  return options;
}

static int emitVPTOBackendResult(ModuleOp module, PTOASCompileResult &result,
                                 bool emitHostStub,
                                 const pto::CANNVersion &cannVersion) {
  if (emitVPTO) {
    result.kind = PTOASCompileResultKind::Text;
    llvm::raw_string_ostream os(result.textOutput);
    module.print(os);
    os << "\n";
    os.flush();
    return 0;
  }

  if (emitVPTOLLVMDialect) {
    result.kind = PTOASCompileResultKind::Text;
    pto::VPTOEmissionOptions options =
        buildVPTOEmissionOptions(
            cannVersion, resolveEffectiveTargetArch(module, ptoTargetArch));
    if (failed(pto::lowerVPTOModuleToLLVMIRText(
            module, options, result.textOutput, llvm::errs()))) {
      llvm::errs() << "Error: Failed to lower VPTO to LLVM IR.\n";
      return 1;
    }
    return 0;
  }

  pto::VPTOEmissionOptions options =
      buildVPTOEmissionOptions(
          cannVersion, resolveEffectiveTargetArch(module, ptoTargetArch));
  std::string stubSource;
  if (emitHostStub) {
    if (failed(pto::emitVPTOHostStubSource(module, stubSource, llvm::errs()))) {
      llvm::errs() << "Error: Failed to emit VPTO host stub source.\n";
      return 1;
    }
  }

  if (failed(
          pto::lowerVPTOModuleToLLVMModules(module, options,
                                            result.vptoCubeModule,
                                            result.vptoVectorModule,
                                            llvm::errs()))) {
    llvm::errs() << "Error: Failed to lower VPTO to LLVM modules.\n";
    return 1;
  }

  result.vptoStubSource = std::move(stubSource);
  result.kind = PTOASCompileResultKind::VPTOObject;
  return 0;
}

static LogicalResult runVPTOBackendPipeline(OwningOpRef<ModuleOp> &module,
                                            bool hasTileOpsToExpand) {
  PassManager pm(module->getContext());
  pm.enableVerifier();
  if (!hasTileOpsToExpand) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOCanonicalizeIRPass());
  }
  pm.addPass(pto::createVPTOSplitCVModulePass());
  pm.addPass(pto::createVPTONormalizeContainerPass());
  if (hasTileOpsToExpand) {
    lowerPTOToVPTOBackend(pm, module.get());
  }
  auto &kernelModulePM = pm.nest<ModuleOp>();
  // Inline legal direct calls before VMI layout assignment so private helper
  // bodies participate in one caller-local layout decision. The Func
  // inliner honors `no_inline` on either the callee or call site. Materialize
  // the implicit no-inline semantics of `pto.simt_entry` first so the rest of
  // the pipeline can use MLIR's standard Func inliner implementation.
  kernelModulePM.addPass(std::make_unique<ApplySIMTEntryNoInlinePass>());
  kernelModulePM.addPass(createInlinerPass());
  appendVMISemanticPipeline(kernelModulePM);
  VPTOSchedulerCLIMode schedulerMode = getEffectiveVPTOSchedulerMode(
      resolveEffectiveTargetArch(*module, ptoTargetArch));
  prepareVPTOForEmission(pm, schedulerMode);
  if (failed(applyConfiguredPassManagerCLOptions(
          pm, "VPTO unified emission pipeline")))
    return failure();
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "Error: VPTO emission pipeline failed.\n";
    return failure();
  }
  return success();
}

static void appendVMISemanticPipeline(OpPassManager &pm) {
  // Materialize unsigned carriers for sign-sensitive VMI ops before any
  // verifier, layout, or lowering pass sees signless integer element types.
  pm.addNestedPass<func::FuncOp>(
      pto::createVMINormalizeSignlessIntToUnsignedPass());
  // Expand unified VMI ops before layout assignment so grouped vci becomes
  // the contiguous-only legacy group_iota producer. Layout assignment can
  // then materialize any consumer-requested non-contiguous use explicitly.
  pm.addPass(pto::createVMILowerUnifiedToLegacyPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(pto::createVMILegalizeArithSelectPass());
  pm.addPass(pto::createPTOValidateVMIIRPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMIPreAssignmentCombinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMILegalizeArithSelectPass());
  pm.addPass(pto::createVMIMaskGranularityAssignmentPass());
  pm.addPass(pto::createVMILayoutRematerializeWeakProducersPass());
  pm.addPass(pto::createVMILayoutAssignmentPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMILayoutRematerializePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMILayoutFoldPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMILayoutSinkMaterializationPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(pto::createVMILegalizeArithSelectPass());
  pm.addPass(pto::createPTOValidateVMILayoutIRPass());
  pm.addPass(pto::createVMIToVPTOPass());
  pm.addPass(pto::createVPTOStatefulStreamFusionPass());
}

/// Reject statically invalid scf.for steps at the PTOAS input boundary.
/// LLVM 19 intentionally does not follow SSA values from scf.for verifiers,
/// so the generic MLIR verifier cannot enforce this semantic constraint.
static LogicalResult validateSCFForConstantSteps(ModuleOp module) {
  WalkResult result = module.walk([](scf::ForOp forOp) -> WalkResult {
    std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
    if (!step || *step > 0) {
      return WalkResult::advance();
    }

    forOp.emitOpError("constant step operand must be positive");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

struct CompilePipelineState {
  std::string arch;
  PTOBuildLevel level = PTOBuildLevel::Level2;
  bool isA2A3 = false;
  bool opFusionEnabled = false;
  bool enableA5EmitCFusionPath = false;
  bool enableA5VPTOFusionPath = false;
  bool hasTileOpsToExpand = false;
  FunctionBlockArgHintMap functionBlockArgHints;
};

static LogicalResult validateCompileBackendFlags(PTOBackend backend,
                                                 llvm::StringRef arch) {
  VPTOSchedulerCLIMode schedulerMode = getEffectiveVPTOSchedulerMode(arch);
  if (backend != PTOBackend::VPTO &&
      (emitVPTO || emitVPTOLLVMDialect || ptoPrintSeamIR ||
       !ptoSeamIRFile.empty())) {
    llvm::errs() << "Error: VPTO-specific flags require "
                    "--pto-backend=vpto or pto.backend = \"vpto\".\n";
    return failure();
  }
  if (enableBufidSync && arch != "a5") {
    llvm::errs() << "Error: --enable-bufid_sync requires --pto-arch=a5.\n";
    return failure();
  }
  if (schedulerMode != VPTOSchedulerCLIMode::Off && arch != "a5") {
    llvm::errs() << "Error: --vpto-scheduler requires --pto-arch=a5.\n";
    return failure();
  }
  if (vptoSchedulerTrace && schedulerMode != VPTOSchedulerCLIMode::On) {
    llvm::errs() << "Error: --vpto-scheduler-trace requires "
                    "--vpto-scheduler=on.\n";
    return failure();
  }
  return success();
}

static LogicalResult initializeCompilePipelineState(
    OwningOpRef<ModuleOp> &module, PTOASContext &context, PTOBackend backend,
    CompilePipelineState &state) {
  const bool invalidScfStep = failed(validateSCFForConstantSteps(*module));
  const bool invalidStructProvenance =
      failed(pto::validateStructProvenance(*module));
  if (invalidScfStep || invalidStructProvenance) {
    return failure();
  }
  state.arch = resolveEffectiveTargetArch(*module, context.getArch());
  state.functionBlockArgHints = collectFunctionBlockArgNameHints(*module);
  if (failed(validateCompileBackendFlags(backend, state.arch))) {
    return failure();
  }
  state.level = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, state.level)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
    return failure();
  }
  state.isA2A3 = isA2A3Arch(state.arch);
  module->getOperation()->setAttr(
      "pto.target_arch",
      mlir::StringAttr::get(module->getContext(), state.arch));
  if (failed(mlir::verify(module.get()))) {
    llvm::errs() << "Error: input module verification failed.\n";
    return failure();
  }
  state.opFusionEnabled = enableOpFusion == llvm::cl::BOU_TRUE;
  return success();
}

static LogicalResult validateFusionConfiguration(const CompilePipelineState &state,
                                                 PTOBackend backend) {
  if (state.opFusionEnabled && state.arch != "a5") {
    llvm::errs() << "Error: --enable-op-fusion=true requires --pto-arch=a5.\n";
    return failure();
  }
  if (state.opFusionEnabled && state.level == PTOBuildLevel::Level1) {
    llvm::errs() << "Warning: --enable-op-fusion=true is ignored because "
                    "--pto-level=level2 or level3 is required.\n";
  }
  if (enableUnrollAfterLoopFusion &&
      !(state.opFusionEnabled && state.arch == "a5")) {
    llvm::errs() << "Error: --enable-unroll-after-loop-fusion requires "
                    "--pto-arch=a5 and --enable-op-fusion.\n";
    return failure();
  }
  if (enableUnrollAfterLoopFusion && backend != PTOBackend::VPTO) {
    llvm::errs() << "Error: --enable-unroll-after-loop-fusion requires "
                    "--pto-backend=vpto; the pass is VPTO-only and is not "
                    "inserted under other backends.\n";
    return failure();
  }
  if (enableUnrollAfterLoopFusion && !enableVfSimCostmodelOptimization) {
    llvm::errs() << "Warning: --enable-unroll-after-loop-fusion consumes "
                    "pto.fusion.row/col_unroll_factor, which is produced by "
                    "--enable-vfsim-costmodel-optimization.\n";
  }
  return success();
}

static void setFusionPipelineFlags(CompilePipelineState &state,
                                   PTOBackend backend) {
  const bool enabled = state.opFusionEnabled && state.arch == "a5" &&
                       state.level != PTOBuildLevel::Level1;
  state.enableA5EmitCFusionPath = enabled && backend == PTOBackend::EmitC;
  state.enableA5VPTOFusionPath = enabled && backend == PTOBackend::VPTO;
  if (enableVfSimCostmodelOptimization &&
      !(state.enableA5EmitCFusionPath || state.enableA5VPTOFusionPath)) {
    llvm::errs() << "Warning: --enable-vfsim-costmodel-optimization is ignored "
                    "because the A5 tile-fusion pipeline is not enabled; "
                    "requires --pto-arch=a5, --pto-level=level2 or level3, "
                    "and op fusion enabled.\n";
  }
  if (enableVfSimCostmodelOptimization && state.enableA5EmitCFusionPath) {
    llvm::errs() << "Warning: --enable-vfsim-costmodel-optimization may "
                    "annotate costmodel attributes on the EmitC fusion path, "
                    "but current VfSim unroll attributes are consumed only by "
                    "the VPTO backend; use --pto-backend=vpto for unroll "
                    "consumption.\n";
  }
  if (enableVfSimCostmodelOptimization && state.enableA5VPTOFusionPath &&
      !enableUnrollAfterLoopFusion) {
    llvm::errs() << "Warning: --enable-vfsim-costmodel-optimization may "
                    "annotate pto.fusion.row/col_unroll_factor, but "
                    "--enable-unroll-after-loop-fusion is not enabled; unroll "
                    "attributes will not be consumed by the VPTO backend.\n";
  }
}

static LogicalResult validateAutoSyncTailHints(ModuleOp module) {
  bool invalid = false;
  module.walk([&invalid, &module](mlir::func::FuncOp func) {
    auto hintAttr =
        func->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint");
    if (!hintAttr) {
      return;
    }
    std::string normalizedHint;
    if (!parseAutoSyncTailHint(hintAttr.getValue(), normalizedHint)) {
      func.emitError("invalid pto.auto_sync_tail_hint '")
          << hintAttr.getValue()
          << "'. Expected 'barrier-all' (or 'default') or "
             "'mte3-to-s-event0'.";
      invalid = true;
      return;
    }
    func->setAttr("pto.auto_sync_tail_hint",
                  mlir::StringAttr::get(module.getContext(), normalizedHint));
  });
  return invalid ? failure() : success();
}

static bool moduleHasTAssign(ModuleOp module) {
  bool hasTAssign = false;
  module.walk([&hasTAssign](pto::TAssignOp) { hasTAssign = true; });
  return hasTAssign;
}

static LogicalResult validateTAssignConfiguration(ModuleOp module,
                                                  PTOBuildLevel level) {
  const bool hasTAssign = moduleHasTAssign(module);
  if (hasTAssign && level != PTOBuildLevel::Level3) {
    llvm::errs() << "Error: pto.tassign is only supported when "
                    "--pto-level=level3.\n";
    return failure();
  }
  if (hasTAssign && enableInsertSync) {
    llvm::errs() << "Error: pto.tassign requires --enable-insert-sync to be "
                    "disabled.\n";
    return failure();
  }
  const int enabledAutoSyncModes =
      (enableInsertSync ? 1 : 0) + (enableBufidSync ? 1 : 0) +
      (enableInjectBarrierAllSync ? 1 : 0);
  if (enabledAutoSyncModes > 1) {
    llvm::errs() << "Error: --enable-insert-sync, --enable-bufid_sync, "
                    "and --enable-inject-barrier-all-sync are mutually "
                    "exclusive.\n";
    return failure();
  }
  if (hasTAssign && enableInjectBarrierAllSync) {
    llvm::errs() << "Error: pto.tassign requires "
                    "--enable-inject-barrier-all-sync to be disabled.\n";
    return failure();
  }
  if (hasTAssign && enableBufidSync) {
    llvm::errs() << "Error: pto.tassign requires --enable-bufid_sync to be "
                    "disabled.\n";
    return failure();
  }
  return success();
}

static LogicalResult validateAllocationConfiguration(ModuleOp module,
                                                     PTOBuildLevel level) {
  bool hasUserPlannedMultiAddrs = false;
  module.walk([&hasUserPlannedMultiAddrs](pto::AllocMultiTileOp op) {
    if (op->hasAttr(pto::kPtoMultiBufferAddrsAttrName)) {
      op.emitError() << "attribute '" << pto::kPtoMultiBufferAddrsAttrName
                     << "' is reserved for pto-plan-memory";
      hasUserPlannedMultiAddrs = true;
    }
  });
  if (hasUserPlannedMultiAddrs) {
    return failure();
  }

  bool invalidAddress = false;
  if (level == PTOBuildLevel::Level3) {
    module.walk([&invalidAddress](pto::AllocTileOp op) {
      if (!op.getAddr()) {
        op.emitError("requires 'addr' operand when --pto-level=level3");
        invalidAddress = true;
      }
    });
    module.walk([&invalidAddress](pto::AllocMultiTileOp op) {
      if (!op.getAddr()) {
        op.emitError("pto.alloc_multi_tile requires a base 'addr' operand when "
                     "--pto-level=level3");
        invalidAddress = true;
      }
    });
  } else {
    module.walk([&invalidAddress](pto::AllocTileOp op) {
      if (op.getAddr()) {
        op.emitError("unexpected 'addr' operand: only supported when "
                     "--pto-level=level3");
        invalidAddress = true;
      }
    });
    module.walk([&invalidAddress](pto::AllocMultiTileOp op) {
      if (op.getAddr()) {
        op.emitError("unexpected 'addr' operand on pto.alloc_multi_tile: only "
                     "supported when --pto-level=level3");
        invalidAddress = true;
      }
    });
  }
  return invalidAddress ? failure() : success();
}

static LogicalResult validateCompileOptions(ModuleOp module,
                                            PTOBuildLevel level) {
  return failed(validateAutoSyncTailHints(module)) ||
                 failed(validateTAssignConfiguration(module, level)) ||
                 failed(validateAllocationConfiguration(module, level)) ||
                 !validateReserveBufferLevelRules(module, level)
             ? failure()
             : success();
}

static LogicalResult runPreBackendNormalization(ModuleOp module) {
  PassManager preBackendPM(module.getContext());
  preBackendPM.enableVerifier();
  preBackendPM.addPass(pto::createPTOMaterializeTileOpSectionsPass());
  preBackendPM.addPass(pto::createPTONormalizeUncoveredTileSectionsPass());
  preBackendPM.addPass(pto::createPTOValidatePhysicalSectionBoundariesPass());
  if (failed(preBackendPM.run(module))) {
    llvm::errs() << "Error: failed to normalize uncovered PTO tile sections.\n";
    return failure();
  }
  return success();
}

/// Build and run the shared mainline lowering pipeline (everything between the
/// VPTO fast path and backend-specific emission). On success the caller checks
/// `handled`: when true the pipeline fully answered the request (`--emit-mlir-ir`
/// text or a VPTO backend result) and `exitCode` carries the process exit code;
/// when false the caller continues with EmitC emission. Failure means the caller
/// must propagate the error.
/// Keep frontend fusion on tile-native PTO IR and annotate last_use directly
/// on scheduled block-local spans before the shared mainline lowers tiles.
/// The shape-inference switch drives FusionPlan only: that is where the
/// iteration-domain decisions (static vs ShapeConstraintSolver) are made.
/// FusionRegionGen consumes only the shared pre-fusion dataflow graph (cached
/// by the analysis manager and built once by FusionPlan) plus the resulting
/// pto.fusion.group_id/order metadata; it never consults the domain classes,
/// so it takes no option here.
static void appendFusionFrontendPassesForBackend(
    PassManager &pm, const pto::FusionPlanOptions &fusionPlanOpts) {
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createFusionPlanPass(fusionPlanOpts));
  pm.addNestedPass<mlir::func::FuncOp>(pto::createOpSchedulingPass());
}

static LogicalResult appendFusionFrontendPasses(
    PassManager &pm, bool isA2A3, bool enableA5EmitCFusionPath,
    bool enableA5VPTOFusionPath) {
  if (isA2A3) {
    return success();
  }
  pto::FusionPlanOptions fusionPlanOpts;
  fusionPlanOpts.enableShapeInference = enableShapeInference;
  fusionPlanOpts.enableVfSimCostmodelOptimization =
      enableVfSimCostmodelOptimization;
  fusionPlanOpts.dumpVfSimUnrollTest = dumpVfSimUnrollTest;
  if (enableA5EmitCFusionPath) {
    appendFusionFrontendPassesForBackend(pm, fusionPlanOpts);
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOMarkLastUsePass());
    return success();
  }
  if (enableA5VPTOFusionPath) {
    appendFusionFrontendPassesForBackend(pm, fusionPlanOpts);
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionRegionGenPass());
  }
  return success();
}

static LogicalResult appendPlanMemoryPasses(PassManager &pm,
                                            PTOBuildLevel effectiveLevel) {
  if (planMemoryImpl != "legacy" && planMemoryImpl != "modern") {
    llvm::errs() << "Error: invalid --plan-memory-impl='" << planMemoryImpl
                 << "', expected 'legacy' or 'modern'.\n";
    return failure();
  }

  if (effectiveLevel != PTOBuildLevel::Level3) {
    pto::PlanMemoryOptions planMemoryOptions;
    planMemoryOptions.memMode = "local";
    bool effectivePlanMemoryOrderBySize = planMemoryOrderBySize;
    if (planMemoryImpl == "modern" &&
        planMemoryOrderBySize.getNumOccurrences() == 0) {
      effectivePlanMemoryOrderBySize = true;
    }
    planMemoryOptions.orderBySize = effectivePlanMemoryOrderBySize;
    if (planMemoryImpl == "legacy") {
      pm.addPass(pto::createPlanMemoryPass(planMemoryOptions));
    } else {
      pm.addPass(pto::createPlanMemoryModernPass(planMemoryOptions));
    }
  }
  return success();
}

/// Conditionally add one automatic synchronization mode. Barrier-all is a
/// conservative standalone pass; InsertSync is a set/wait solver, while
/// BufidSync is A5-only get_buf/rls_buf synchronization. Sync
/// runs BEFORE PTOResolveBufferSelect so it sees per-use `pto.multi_tile_get`
/// operations and keeps their slot identity for alias and event-id analysis.
static void appendAutoSyncPasses(PassManager &pm) {
  if (enableInsertSync) {
    if (emitMlirIR) {
      pm.addPass(std::make_unique<SerialAutoSyncPass>(
          SerialAutoSyncPass::Mode::InsertSync, false));
    } else {
      pm.addNestedPass<func::FuncOp>(pto::createPTOInsertSyncPass());
    }
  }
  else if (enableBufidSync) {
    if (emitMlirIR) {
      pm.addPass(std::make_unique<SerialAutoSyncPass>(
          SerialAutoSyncPass::Mode::Bufid, enableBufidSyncDebug));
    } else {
      PTOBufidSyncOptions options;
      options.enableBufidSyncDebug = enableBufidSyncDebug;
      pm.addNestedPass<func::FuncOp>(pto::createPTOBufidSyncPass(options));
    }
  } else if (enableInjectBarrierAllSync) {
    if (emitMlirIR) {
      pm.addPass(std::make_unique<SerialAutoSyncPass>(
          SerialAutoSyncPass::Mode::BarrierAll, false));
    } else {
      pm.addNestedPass<func::FuncOp>(
          pto::createPTOInjectBarrierAllSyncPass());
    }
  }
}

static LogicalResult runMainLoweringPipeline(
    OwningOpRef<ModuleOp> &module, PTOASContext &context,
    PTOBackend effectiveBackend, const CompilePipelineState &state,
    PTOASCompileResult &result, bool emitVPTOHostStub, bool &handled,
    int &exitCode) {
  handled = false;
  exitCode = 0;
  const bool enableA5EmitCFusionPath = state.enableA5EmitCFusionPath;
  const bool enableA5VPTOFusionPath = state.enableA5VPTOFusionPath;
  const bool isA2A3 = state.isA2A3;
  const bool hasTileOpsToExpand = state.hasTileOpsToExpand;
  const PTOBuildLevel effectiveLevel = state.level;

  // Main PassManager
  PassManager pm(module->getContext());

  if (failed(applyPassManagerCLOptions(pm))) {
    return failure();
  }

  // Rank-2 → rank-5 view canonicalization is currently gated on the VPTO
  // backend to limit blast radius.  A3/A5 EmitC codegen already pads strides
  // to rank-5 via InferPTOLayout and buildGlobalTensorShapeAndStride, so it
  // does not need the canonicalization pass at the IR level.  When VPTO
  // validation is complete and the pass is proven stable, the gate can be
  // lifted to make it unconditional for all backends.
  if (effectiveBackend == PTOBackend::VPTO) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOCanonicalizeIRPass());
  }
  pm.addPass(createSerialFrontendPipeLoweringPass());
  pm.addPass(pto::createPTOInferValidatePipeInitPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
  if (!disableInferLayout) {
    pm.addPass(pto::createInferPTOLayoutPass());
  }
  // PTOViewToMemref is generic view lowering required by both backends; keep it
  // outside the local-memory planning gate so default A2/A3 EmitC still lowers
  // pto.make_tensor_view before backend legalization.
  if (!isA2A3) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOA5NormalizeTMovPass());
  }
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOValidateIntToPtrUsesPass());

  // PTODSL legality discovery happens on tile-native PTO IR before fusion.
  // Fusion may later filter the ordered `candidates` array; ExpandTileOp
  // consumes the first candidate that remains.
  if (!isA2A3 && effectiveBackend == PTOBackend::VPTO && hasTileOpsToExpand) {
    pm.addPass(pto::createInsertTemplateAttributesPass());
  }

  if (failed(appendFusionFrontendPasses(pm, isA2A3, enableA5EmitCFusionPath,
                                         enableA5VPTOFusionPath))) {
    return failure();
  }

  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTOMaterializeImplicitTmpPass(
          effectiveLevel == PTOBuildLevel::Level3));
  pm.addNestedPass<mlir::func::FuncOp>(
      pto::createPTORematerializeFixpipeVectorQuantPass());

  if (failed(appendPlanMemoryPasses(pm, effectiveLevel))) {
    return failure();
  }
  pm.addPass(pto::createPTOResolveReservedBuffersPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTORemoveIdentityTMovPass());

  appendAutoSyncPasses(pm);

  // Materialize each `pto.multi_tile_get` as an addressed `pto.alloc_tile`;
  // dynamic selections use an `arith.select` chain over planned addresses.
  pm.addPass(pto::createPTOResolveBufferSelectPass());
  if (effectiveBackend == PTOBackend::EmitC) {
    pm.addPass(createNarrowUnusedMultiResultProvenancePass());
  }

  if (emitMlirIR) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return failure();
    }
    result.kind = PTOASCompileResultKind::Text;
    llvm::raw_string_ostream os(result.textOutput);
    module->print(os);
    os.flush();
    handled = true;
    exitCode = 0;
    return success();
  }

  pm.addPass(createCSEPass());
  // PTODSL backend helpers already use the tile-native ABI.
  pm.addPass(pto::createPTOInlineBackendHelpersPass());
  if (effectiveBackend == PTOBackend::EmitC) {
    pm.addPass(createNarrowUnusedMultiResultProvenancePass());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(applyConfiguredPassManagerCLOptions(pm, "main PTOAS pipeline"))) {
    return failure();
  }

  if (effectiveBackend == PTOBackend::VPTO) {
    if (failed(pm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return failure();
    }

    if (ptoPrintSeamIR) {
      printSharedPreBackendSeamIR(*module);
    }
    if (ptoPrintSeamIR) {
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }
    if (failed(emitSharedPreBackendSeamIR(*module, ptoSeamIRFile))) {
      return failure();
    }

    if (failed(runVPTOBackendPipeline(module, hasTileOpsToExpand))) {
      return failure();
    }
    handled = true;
    exitCode = emitVPTOBackendResult(*module, result, emitVPTOHostStub,
                                     context.getCANNVersionOrDefault());
    return success();
  }

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return failure();
  }
  return success();
}

// VPTO fast path: when no TileOp expansion is needed the shared mainline
// pipeline can be skipped entirely.
static int runVPTOSkipMainlinePipeline(OwningOpRef<ModuleOp> &module,
                                       PTOASContext &context,
                                       const CompilePipelineState &state,
                                       PTOASCompileResult &result,
                                       bool emitVPTOHostStub) {
  if (ptoPrintSeamIR || !ptoSeamIRFile.empty()) {
    llvm::errs() << "Error: shared pre-backend seam IR is unavailable when "
                    "skipping the shared PTO-to-VPTO lowering pipeline.\n";
    return 1;
  }
  if (failed(runVPTOBackendPipeline(module, state.hasTileOpsToExpand))) {
    return 1;
  }
  return emitVPTOBackendResult(*module, result, emitVPTOHostStub,
                               context.getCANNVersionOrDefault());
}

static LogicalResult runEmitCPreparationPipeline(OwningOpRef<ModuleOp> &module,
                                                 llvm::StringRef arch) {
  PassManager emitcPM(module->getContext());
  emitcPM.enableVerifier();
  if (isA2A3Arch(arch)) {
    emitcPM.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A3));
  } else {
    emitcPM.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A5));
  }
  emitcPM.addPass(std::make_unique<FormEmitCExpressionsCompatPass>());
  emitcPM.addPass(mlir::createCSEPass());
  if (failed(applyConfiguredPassManagerCLOptions(
          emitcPM, "EmitC backend pipeline"))) {
    return failure();
  }
  if (failed(emitcPM.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return failure();
  }
  return success();
}

// Post-process the emitted C++ text. Markers, provenance comments, and scalar
// constant hoisting must run in this fixed order.
static void runCppPostRewrites(std::string &cppOutput) {
  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAsyncEventMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteScalarGMStoreFlushMarkers(cppOutput);
  rewriteEventIdArrayMarkers(cppOutput);
  rewriteGlobalTensorMetadataMarkers(cppOutput);
  pto::rewriteLastUseMarkersInCpp(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteMalformedVerbatimSemicolons(cppOutput);
  rewriteScalarConstantDecls(cppOutput);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  rewriteNameHintMarkers(cppOutput);
}

static int runEmitCTextEmission(OwningOpRef<ModuleOp> &module,
                                const CompilePipelineState &state,
                                PTOASCompileResult &result) {
  if (ptoPrintSeamIR) {
    printSharedPreBackendSeamIR(*module);
  }
  if (failed(emitSharedPreBackendSeamIR(*module, ptoSeamIRFile))) {
    return 1;
  }

  narrowUnusedMultiResultProvenanceLocs(module.get());
  splitDerivedSingleResultProvenanceLocs(module.get());

  if (failed(runEmitCPreparationPipeline(module, state.arch))) {
    return 1;
  }

  applyFunctionBlockArgNameHintsToEmitC(*module, state.functionBlockArgHints);
  splitDerivedSingleResultProvenanceLocs(module.get());
  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());
  normalizeEmitCIntegerAttrsForCppEmission(module.get());
  if (failed(reorderEmitCFunctions(module.get()))) {
    llvm::errs() << "Error: Failed to order emitted functions for C++ emission.\n";
    return 1;
  }
  annotateEmitCProvenanceHints(*module);

  // Emit C++ to string, then post-process, then hand the text to the caller.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  const bool declareVariablesAtTop = shouldDeclareVariablesAtTop(*module);
  if (failed(emitc::translateToCpp(*module, cppOS,
                                   /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  runCppPostRewrites(cppOutput);

  result.kind = PTOASCompileResultKind::Text;
  result.textOutput = std::move(cppOutput);
  return 0;
}

int mlir::pto::compilePTOASModule(
    OwningOpRef<ModuleOp> &module, PTOASContext &context,
    PTOBackend effectiveBackend, PTOASCompileResult &result,
    bool emitVPTOHostStub) {
  result.reset();
  CompilePipelineState state;
  if (failed(initializeCompilePipelineState(module, context, effectiveBackend,
                                            state)) ||
      failed(validateFusionConfiguration(state, effectiveBackend))) {
    return 1;
  }
  setFusionPipelineFlags(state, effectiveBackend);
  if (failed(validateCompileOptions(*module, state.level))) {
    return 1;
  }
  if (failed(runPreBackendNormalization(*module))) {
    return 1;
  }
  state.hasTileOpsToExpand = hasUnexpandedTileOps(*module);

  // The state is assembled and validated once above, so backend branches
  // cannot observe partially validated option combinations.
  if (effectiveBackend == PTOBackend::VPTO && !state.hasTileOpsToExpand) {
    return runVPTOSkipMainlinePipeline(module, context, state, result,
                                       emitVPTOHostStub);
  }

  bool mainPipelineHandled = false;
  int mainPipelineExitCode = 0;
  if (failed(runMainLoweringPipeline(module, context, effectiveBackend, state,
                                     result, emitVPTOHostStub,
                                     mainPipelineHandled,
                                     mainPipelineExitCode))) {
    return 1;
  }
  if (mainPipelineHandled) {
    return mainPipelineExitCode;
  }
  if (effectiveBackend == PTOBackend::VPTO) {
    return 0;
  }
  return runEmitCTextEmission(module, state, result);
}
