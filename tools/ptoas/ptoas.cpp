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
#include <cctype>
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

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

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
//
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                  -> obj.data()
// --------------------------------------------------------------------------
static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    size_t markerPos = cpp.find(marker.str(), searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t lparenPos = markerPos + marker.size();
    if (lparenPos >= cpp.size() || cpp[lparenPos] != '(') {
      searchPos = markerPos + marker.size();
      continue;
    }

    // Find the matching ')' for this call, tracking nested parentheses.
    size_t argsBegin = lparenPos + 1;
    int parenDepth = 0;
    size_t rparenPos = std::string::npos;
    for (size_t i = argsBegin; i < cpp.size(); ++i) {
      char c = cpp[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth == 0) {
          rparenPos = i;
          break;
        }
        --parenDepth;
      }
    }
    if (rparenPos == std::string::npos) {
      // Unbalanced parentheses; stop trying to rewrite.
      break;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    llvm::SmallVector<llvm::StringRef, 4> args;
    size_t partBegin = 0;
    parenDepth = 0;
    for (size_t i = 0; i < argsRef.size(); ++i) {
      char c = argsRef[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth > 0)
          --parenDepth;
      } else if (c == ',' && parenDepth == 0) {
        args.push_back(argsRef.slice(partBegin, i).trim());
        partBegin = i + 1;
      }
    }
    if (partBegin <= argsRef.size())
      args.push_back(argsRef.drop_front(partBegin).trim());

    if (args.size() != expectedNumArgs) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement;
    replacement.reserve(marker.size() + argsRef.size() + 16);
    replacement.append(args[0].str());
    replacement.push_back('.');
    replacement.append(memberName.str());
    replacement.push_back('(');
    if (expectedNumArgs == 1) {
      // no args
    } else if (expectedNumArgs == 2) {
      replacement.append(args[1].str());
    } else if (expectedNumArgs == 3) {
      replacement.append(args[1].str());
      replacement.append(", ");
      replacement.append(args[2].str());
    }
    replacement.push_back(')');

    cpp.replace(markerPos, (rparenPos - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  // Keep applying until fixed-point in case rewrites shift subsequent matches.
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_SET_VALUE", "SetValue", /*expectedNumArgs=*/3);
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_GET_VALUE", "GetValue", /*expectedNumArgs=*/2);
    changed |= rewriteMarkerCallToMember(
        cpp, "PTOAS__TILE_DATA", "data", /*expectedNumArgs=*/1);
  }
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    size_t markerPos = cpp.find("PTOAS__ADDPTR_TRACE", searchPos);
    if (markerPos == std::string::npos)
      break;

    size_t lparenPos = markerPos + (sizeof("PTOAS__ADDPTR_TRACE") - 1);
    if (lparenPos >= cpp.size() || cpp[lparenPos] != '(') {
      searchPos = markerPos + 1;
      continue;
    }

    size_t argsBegin = lparenPos + 1;
    int parenDepth = 0;
    size_t rparenPos = std::string::npos;
    for (size_t i = argsBegin; i < cpp.size(); ++i) {
      char c = cpp[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth == 0) {
          rparenPos = i;
          break;
        }
        --parenDepth;
      }
    }
    if (rparenPos == std::string::npos) {
      break;
    }

    llvm::StringRef argsRef(cpp.data() + argsBegin, rparenPos - argsBegin);
    llvm::SmallVector<llvm::StringRef, 4> args;
    size_t partBegin = 0;
    parenDepth = 0;
    for (size_t i = 0; i < argsRef.size(); ++i) {
      char c = argsRef[i];
      if (c == '(') {
        ++parenDepth;
      } else if (c == ')') {
        if (parenDepth > 0)
          --parenDepth;
      } else if (c == ',' && parenDepth == 0) {
        args.push_back(argsRef.slice(partBegin, i).trim());
        partBegin = i + 1;
      }
    }
    if (partBegin <= argsRef.size())
      args.push_back(argsRef.drop_front(partBegin).trim());

    if (args.size() != 3) {
      searchPos = rparenPos + 1;
      continue;
    }

    std::string replacement;
    if (showTrace) {
      replacement.reserve(64 + argsRef.size());
      replacement.append("/* ADDPTR_TRACE: ");
      replacement.append(args[0].str());
      replacement.append(" = ");
      replacement.append(args[1].str());
      replacement.append(" + ");
      replacement.append(args[2].str());
      replacement.append(" */");
    }

    size_t replaceEnd = rparenPos;
    if (!showTrace) {
      size_t i = rparenPos + 1;
      while (i < cpp.size() && std::isspace(static_cast<unsigned char>(cpp[i])))
        ++i;
      if (i < cpp.size() && cpp[i] == ';')
        replaceEnd = i;
    }

    cpp.replace(markerPos, (replaceEnd - markerPos) + 1, replacement);
    changed = true;
    searchPos = markerPos + replacement.size();
  }
  return changed;
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so a
  // hoisted declaration like:
  //   GlobalTensor<...> v42;
  // fails to compile. Initialize those hoisted temporaries with a null pointer
  // so they are constructible:
  //   GlobalTensor<...> v42(nullptr);
  //
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + 64);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    if (trimmed.starts_with("GlobalTensor<") && trimmed.ends_with(";") &&
        !trimmed.contains('=') && !trimmed.contains('(')) {
      llvm::StringRef decl = trimmed.drop_back().rtrim();
      size_t lastWs = decl.find_last_of(" \t");
      if (lastWs != llvm::StringRef::npos) {
        llvm::StringRef varName = decl.drop_front(lastWs + 1);
        if (varName.starts_with("v") && varName.size() > 1) {
          bool allDigits = true;
          for (char c : varName.drop_front(1)) {
            if (c < '0' || c > '9') {
              allDigits = false;
              break;
            }
          }
          if (allDigits) {
            size_t indentLen = line.find_first_not_of(" \t");
            if (indentLen == std::string::npos)
              indentLen = 0;
            llvm::StringRef indent = line.take_front(indentLen);

            out.append(indent.str());
            out.append(decl.str());
            out.append("(nullptr);");
            rewritten = true;
          }
        }
      }
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

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
  pm.addPass(mlir::createCSEPass());
  
  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  // llvm::outs() << "\n===== EmitC IR (before translateToCpp) =====\n";
  // module->print(llvm::outs());
  // llvm::outs() << "\n===== End EmitC IR =====\n";

  // Emit C++ to string, then post-process, then write to output file.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  if (failed(emitc::translateToCpp(*module, cppOS,
                                  /*declareVariablesAtTop=*/true))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  outputFile.os() << cppOutput;

  outputFile.keep(); // Success, keep the file

  return 0;
}
