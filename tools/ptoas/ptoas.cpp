// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas.h"

#include "ptoas_internal.h"

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




/// LLVM 19's Func inliner interface accepts every call and callable, including
/// operations carrying the standard `no_inline` attribute. Keep the upstream
/// terminator handling while honoring the attribute used for PTO SIMT entry
/// functions.
struct PTOASFuncInlinerInterface final : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    (void)wouldBeCloned;
    return !call->hasAttr("no_inline") && !callable->hasAttr("no_inline");
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    auto returnOp = dyn_cast<func::ReturnOp>(op);
    if (!returnOp) {
      return;
    }
    OpBuilder builder(op);
    builder.create<cf::BranchOp>(op->getLoc(), newDest,
                                 returnOp.getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<func::ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

static void registerPTOASFuncInlinerExtension(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<PTOASFuncInlinerInterface>();
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();
  });
}

} // namespace

int main(int argc, char **argv);

void mlir::pto::registerPTOASDialects(DialectRegistry &registry) {
  registerPTOASFuncInlinerExtension(registry);
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();

  registry.insert<mlir::pto::PTODialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
}

void mlir::pto::registerPTOASPassesAndCLOptions() {
  mlir::registerConversionPasses();
  mlir::arith::registerArithPasses();
  mlir::func::registerFuncPasses();
  mlir::math::registerMathPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerSCFPasses();
  mlir::tensor::registerTensorPasses();
  mlir::registerTransformsPasses();

  mlir::pto::registerPTOPasses();
  mlir::pto::registerPTOInlineLibCall();
  mlir::pto::registerFoldTileBufIntrinsics();
  mlir::pto::registerLowerPTOToUBufOps();
  mlir::registerPassManagerCLOptions();
}

void mlir::pto::loadPTOASDialects(MLIRContext &context) {
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}


llvm::cl::opt<VPTOSchedulerCLIMode> vptoSchedulerMode(
    "vpto-scheduler",
    llvm::cl::desc("VPTO scheduler mode"),
    llvm::cl::values(
        clEnumValN(VPTOSchedulerCLIMode::Off, "off", "Disable scheduling"),
        clEnumValN(VPTOSchedulerCLIMode::Analyze, "analyze",
                   "Report scheduling analysis without changing IR"),
        clEnumValN(VPTOSchedulerCLIMode::On, "on", "Run scheduler in on mode")),
    llvm::cl::init(VPTOSchedulerCLIMode::Off));

llvm::cl::opt<bool> vptoSchedulerTrace(
    "vpto-scheduler-trace",
    llvm::cl::desc("Print detailed VPTO on-mode scheduling results"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

llvm::cl::opt<bool> planMemoryOrderBySize(
    "plan-memory-order-by-size",
    llvm::cl::desc("Plan larger local buffers first inside one AddressSpace "
                   "before applying the basic SPEC_LEVEL_0 reuse strategy. "
                   "Defaults to true when no explicit value is given"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableBufidSync(
    "enable-bufid_sync",
    llvm::cl::desc("Enable A5 buffer-id synchronization insertion pass"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableBufidSyncDebug(
    "enable-bufid-sync-debug",
    llvm::cl::desc("Enable verbose debug printing for --enable-bufid_sync"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableInjectBarrierAllSync(
    "enable-inject-barrier-all-sync",
    llvm::cl::desc("Enable conservative synchronization by inserting "
                   "pto.barrier PIPE_ALL before memory-effecting PTO pipe ops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableTileOpExpand(
    "enable-tile-op-expand",
    llvm::cl::desc(
        "Deprecated compatibility flag. TileOp expansion is controlled by "
        "--pto-backend=vpto."),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableVexpdifFusion(
    "enable-vexpdif-fusion",
    llvm::cl::desc("Enable vsub + vexp fusion into vexpdif"),
    llvm::cl::init(true));

llvm::cl::opt<llvm::cl::boolOrDefault> enableOpFusion(
    "enable-op-fusion",
    llvm::cl::desc("Control A5 tile fusion on level2/level3. Disabled by "
                   "default; pass --enable-op-fusion=true to opt in. EmitC "
                   "uses last-use annotation; VPTO uses fusion-region "
                   "lifecycle."),
    llvm::cl::init(llvm::cl::BOU_UNSET));

llvm::cl::opt<bool> enableUnrollAfterLoopFusion(
    "enable-unroll-after-loop-fusion",
    llvm::cl::desc("Partial-unroll the innermost scf.for in pto.fusion_region "
                   "by pto.fusion.row/col_unroll_factor. VPTO backend only; "
                   "requires --pto-arch=a5 and --enable-op-fusion."),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableShapeInference(
    "enable-shape-inference",
    llvm::cl::desc("Enable shape inference (ShapeConstraintSolver) for A5 tile "
                  "fusion. On by default: uses the ShapeConstraintSolver for "
                  "iteration-domain inference; pass --enable-shape-inference=false "
                  "to fall back to static/direct-bound inference."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableVfSimCostmodelOptimization(
    "enable-vfsim-costmodel-optimization",
    llvm::cl::desc("Enable optional VfSimulator costmodel-driven fusion "
                   "optimization. Requires the A5 tile-fusion pipeline. This "
                   "may annotate pto.fusion.row/col_unroll_factor; pass "
                   "--enable-unroll-after-loop-fusion to consume those "
                   "attributes in the VPTO backend."),
    llvm::cl::init(false));

llvm::cl::opt<bool> dumpVfSimUnrollTest(
    "dump-vfsim-unroll-test",
    llvm::cl::desc("Print VfSimulator unroll candidate timings for accepted "
                   "fusion groups. Debug dump only; does not enable or disable "
                   "the VfSimulator planner."),
    llvm::cl::init(false));

llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableSoftPostUpdate(
    "enable-vpto-soft-postupdate",
    llvm::cl::desc("Enable VPTO soft post-update optimization (default: true)"),
    llvm::cl::init(true));

llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitMlirIR(
    "emit-pto-ir",
    llvm::cl::desc("Emit PTO IR after lowering instead of C++"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoTargetArch(
    "pto-arch",
    llvm::cl::desc("Target Ascend architecture for codegen: a2, a3, or a5 (default: a3)"),
    llvm::cl::value_desc("a2|a3|a5"),
    llvm::cl::init("a3"));

llvm::cl::opt<std::string> ptoBuildLevel(
    "pto-level",
    llvm::cl::desc("Build level for pass pipeline: level1, level2, or level3 (default: level2)"),
    llvm::cl::value_desc("level1|level2|level3"),
    llvm::cl::init("level2"));

llvm::cl::opt<std::string> mlir::pto::ptoBackend(
    "pto-backend",
    llvm::cl::desc("Final PTOAS backend: emitc or vpto (default: emitc)"),
    llvm::cl::value_desc("emitc|vpto"), llvm::cl::init("emitc"));

llvm::cl::opt<bool> mlir::pto::emitVPTO(
    "emit-vpto",
    llvm::cl::desc("Write final post-pass VPTO IR to -o"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitVPTOLLVMDialect(
    "emit-vpto-llvm-ir",
    llvm::cl::desc("Write translated VPTO LLVM IR to -o"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoPrintIR(
    "vpto-print-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> vptoLoweringStrategy(
    "vpto-lowering-strategy",
    llvm::cl::desc("VPTO vector lowering strategy: post-update or no-post-update"),
    llvm::cl::value_desc("post-update|no-post-update"),
    llvm::cl::init("post-update"));

llvm::cl::opt<bool> dumpVPTOIR(
    "dump-vpto-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::ptoPrintSeamIR(
    "pto-print-seam-ir",
    llvm::cl::desc("Print shared pre-backend seam IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoSeamIRFile(
    "pto-seam-ir-file",
    llvm::cl::desc("Write shared pre-backend seam IR to a file"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

llvm::cl::opt<std::string> mlir::pto::cannOutputVersion(
    "cann-output-version",
    llvm::cl::desc("Override the CANN version used for lowering and public ABI output selection; examples: 9.0.0, 9.0.0-beta.1"),
    llvm::cl::value_desc("version"), llvm::cl::init(""));

llvm::cl::opt<mlir::pto::VFSIMTSizeFixMode> mlir::pto::vptoFixVFSIMTSize(
    "vpto-fix-vfsimt-size",
    llvm::cl::desc("Validate or repair VF_SIMT code sizes in VPTO vector objects"),
    llvm::cl::value_desc("auto|off|verify"),
    llvm::cl::values(
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Auto, "auto",
                   "Repair the known invalid 0xffff size (default)"),
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Off, "off",
                   "Skip VF_SIMT size validation and repair"),
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Verify, "verify",
                   "Validate VF_SIMT sizes without repairing them")),
    llvm::cl::init(mlir::pto::VFSIMTSizeFixMode::Auto));
