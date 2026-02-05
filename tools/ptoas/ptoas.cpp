//===- ptoas.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace pto;

// #define ADD_CANONICALIZER_PASS \
//    CanonicalizerOptions options; \
//    options.enableExtendedPattern = true; \
//    std::vector<std::string> disabledPatterns{}; \
//    options.disabledPatterns = disabledPatterns; \
//    pm.addPass(createCanonicalizerPass(options))

// #define ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS \
//    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass(options))

// static void canonicalizationPipeline(OpPassManager &pm) {
//    pm.addPass(createArithToAffineConversionPass());
//    ADD_CANONICALIZER_PASS;
//    pm.addPass(createSCFForLoopCanonicalizationPass());
//    pm.addPass(createCSEPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    //pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
// }

static void bufferizationPipeline(OpPassManager &pm) {
  bufferization::OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShotOptions));
  // pm.addPass(bufferization::createOneShotBufferizePass());

  // if (hivmPipelineOptions.enableVfMerge) {
  //    pm.addPass(hfusion::createMergeVecScopePass());
  // }
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(createConvertToPTOOpPass());
}

// --------------------------------------------------------------------------
// Command Line Options
// --------------------------------------------------------------------------
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o",
                                                 llvm::cl::desc("Output filename"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("-"));

static llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(true)); // 默认关闭，需显式开启

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  registry.insert<mlir::pto::PTODialect>();
  //mlir::registerAllDialects(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  //func::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  llvm::SourceMgr sourceMgr;
  // Use inputFilename from cl options
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: " 
                 << fileOrErr.getError().message() << "\n";
    return 1;
  }
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  
  MLIRContext context(registry);
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: Failed to parse MLIR.\n";
    return 1;
  }

  // [Fix] ToolOutputFile Usage
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  // Main PassManager
  PassManager pm(&context);
  
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertCVMovPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOConvertToDPSPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertLoadStoreForMixCVPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());
  
  pm.addPass(pto::createPTOViewToMemrefPass());
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  // bufferizationPipeline(pm);
  //pm.addPass(createInferPTOMemScopePass());
  
  PlanMemoryOptions planMemoryOption;
  planMemoryOption.memMode = MemPlanMode::GLOBAL_WORKSPACE_PLAN;
  planMemoryOption.enableGlobalReuse = false;
  planMemoryOption.enablePrintMemoryAllocatedSize = false;
  pm.addPass(pto::createPlanMemoryPass());

  // Conditionally add Sync pass based on flag
  if (enableInsertSync) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
  }

  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTORemoveRedundantBarrierPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOHighDimLoweringPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVFloopGatherPass());
  
  pm.addPass(createCSEPass());
  pm.addPass(pto::createEmitPTOManualPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  
  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  // llvm::outs() << "\n===== EmitC IR (before translateToCpp) =====\n";
  // module->print(llvm::outs());
  // llvm::outs() << "\n===== End EmitC IR =====\n";

  // Emit C++ to the configured output file
  if (failed(emitc::translateToCpp(*module, outputFile.os()))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }

  outputFile.keep(); // Success, keep the file
  llvm::outs() << "PTO Driver Success!!!\n";

  return 0;
}
