// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Internal shared declarations for the PTOAS compiler translation units
// split out of ptoas.cpp (name hints, C++ rewriting, pipeline). Not part of
// the public ptoas.h API.
//===----------------------------------------------------------------------===//

#ifndef PTOAS_PTOAS_INTERNAL_H
#define PTOAS_PTOAS_INTERNAL_H

#include "ptoas.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include <string>

// Shared inline capacities and marker sizes (moved verbatim from ptoas.cpp).
constexpr unsigned kSeenCalleeInlineCapacity = 8;
constexpr unsigned kStringRefInlineCapacity = 4;
constexpr unsigned kEmptyExpressionInlineCapacity = 8;
constexpr unsigned kBranchInlineCapacity = 16;
constexpr unsigned kNameHintInlineCapacity = 4;
constexpr unsigned kBlockInlineCapacity = 8;
constexpr unsigned kProvenanceInlineCapacity = 8;
constexpr unsigned kProvenanceMarkerInlineCapacity = 32;
constexpr unsigned kFunctionLineInlineCapacity = 32;
constexpr unsigned kRawStringInlineCapacity = 128;
constexpr unsigned kIndexBitWidth = 64;
constexpr unsigned kEmitCIntegerRadix = 10;
constexpr size_t kCppIdentifierReserveExtra = 4;
constexpr size_t kMarkerReplacementReserveExtra = 8;
constexpr size_t kMarkerCallReserveExtra = 16;
constexpr size_t kRewriteOutputReserveExtra = 64;
constexpr size_t kMarkerRewriteMinArgCount = 2;
constexpr size_t kMarkerRewriteTernaryArgCount = 3;
constexpr size_t kThirdMarkerArgumentIndex = 2;
constexpr size_t kCommentTerminatorLength = 2;
constexpr unsigned kHexNibbleBitWidth = 4;
constexpr unsigned kHexNibbleMask = 0xF;

using StringRefVector =
    llvm::SmallVector<llvm::StringRef, kStringRefInlineCapacity>;

using FunctionBlockArgHintMap = llvm::StringMap<
    llvm::SmallVector<llvm::SmallVector<std::string, kNameHintInlineCapacity>,
                      kNameHintInlineCapacity>>;

enum class VPTOSchedulerCLIMode { Off, Analyze, On };

// Command-line options defined in ptoas.cpp and read by the pipeline TU.
extern llvm::cl::opt<VPTOSchedulerCLIMode> vptoSchedulerMode;
extern llvm::cl::opt<bool> vptoSchedulerTrace;
extern llvm::cl::opt<bool> enableInsertSync;
extern llvm::cl::opt<bool> enableBufidSync;
extern llvm::cl::opt<bool> enableBufidSyncDebug;
extern llvm::cl::opt<bool> enableInjectBarrierAllSync;
extern llvm::cl::opt<llvm::cl::boolOrDefault> enableOpFusion;
extern llvm::cl::opt<bool> enableUnrollAfterLoopFusion;
extern llvm::cl::opt<bool> enableVexpdifFusion;
extern llvm::cl::opt<bool> enableShapeInference;
extern llvm::cl::opt<bool> enableVfSimCostmodelOptimization;
extern llvm::cl::opt<bool> dumpVfSimUnrollTest;
extern llvm::cl::opt<bool> planMemoryOrderBySize;
extern llvm::cl::opt<std::string> ptoBuildLevel;
extern llvm::cl::opt<bool> disableInferLayout;
extern llvm::cl::opt<bool> enableSoftPostUpdate;
extern llvm::cl::opt<bool> emitAddPtrTrace;
extern llvm::cl::opt<bool> dumpVPTOIR;

// Name-hint machinery (ptoas_name_hints.cpp).
FunctionBlockArgHintMap collectFunctionBlockArgNameHints(
    mlir::ModuleOp module);
void applyFunctionBlockArgNameHintsToEmitC(
    mlir::ModuleOp module, const FunctionBlockArgHintMap &blockArgHints);
void annotateEmitCProvenanceHints(mlir::ModuleOp module);
void narrowUnusedMultiResultProvenanceLocs(mlir::Operation *root);
void splitDerivedSingleResultProvenanceLocs(mlir::Operation *root);
void rewriteNameHintMarkers(std::string &cpp);

// C++ text post-processing (ptoas_cpp_rewrite.cpp).
bool isGeneratedValueName(llvm::StringRef name);
void dropEmptyEmitCExpressions(mlir::Operation *rootOp);
void materializeControlFlowOperands(mlir::Operation *rootOp);
void normalizeEmitCIntegerAttrsForCppEmission(mlir::Operation *rootOp);
bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace);
void rewriteTileGetSetValueMarkers(std::string &cpp);
void rewriteAsyncEventMarkers(std::string &cpp);
void rewritePtrScalarMarkers(std::string &cpp);
void rewriteScalarGMStoreFlushMarkers(std::string &cpp);
void rewriteEventIdArrayMarkers(std::string &cpp);
void rewriteGlobalTensorMetadataMarkers(std::string &cpp);
void rewriteMalformedVerbatimSemicolons(std::string &cpp);
void rewriteScalarConstantDecls(std::string &cpp);
void rewriteHoistedGlobalTensorDecls(std::string &cpp);

#endif // PTOAS_PTOAS_INTERNAL_H
