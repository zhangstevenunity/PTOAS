// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

// The VPTO backend is emitted from tools/ptoas rather than a TableGen pass;
// these registrations continue to describe the shared pre-backend pipeline.

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];
 
  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect" 
  ];
}

def PTOInjectBarrierAllSync : Pass<"pto-inject-barrier-all-sync", "func::FuncOp"> {
  let summary = "Conservatively insert PIPE_ALL barriers for PTO dialect";
  let description = [{
    Inserts a `pto.barrier <PIPE_ALL>` before each PTO pipe operation that has
    MemoryEffects read/write side effects. This mode does not reuse set/wait
    dependency analysis and does not replace the internal synchronization
    protocol of composite pipe ops such as `pto.tpush`, `pto.tpop`, and
    `pto.tfree`; it only adds conservative barriers at operation boundaries.
  }];

  let constructor = "mlir::pto::createPTOInjectBarrierAllSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOBufidSync : Pass<"pto-bufid-sync", "func::FuncOp"> {
  let summary = "Insert get_buf/rls_buf synchronization for PTO dialect";
  let description = [{
    Analyzes cross-pipeline local buffer dependencies and wraps dependent
    operations with A5 buffer-id synchronization. Unlike pto-insert-sync, this
    pass emits get_buf/rls_buf pairs instead of set_flag/wait_flag events.
  }];

  let constructor = "mlir::pto::createPTOBufidSyncPass()";

  let options = [
    Option<"enableBufidSyncDebug", "enable-bufid-sync-debug",
           "bool", "false",
           "Enable verbose debug printing for the bufid_sync pass">
  ];

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "ModuleOp"> {
  let summary = "Infer and propagate GlobalTensor layout (ND/DN/NZ)";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules, propagate it through SCF/CF control flow and
    direct function calls, and diagnose conflicting layouts at merge points.
    Non-ND layouts are encoded in tensor-view SSA types for backend conversion.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect",
                           "cf::ControlFlowDialect", "func::FuncDialect",
                           "scf::SCFDialect"];
}

def PTOA5NormalizeTMov : Pass<"pto-a5-normalize-tmov", "func::FuncOp"> {
  let summary = "Normalize risky A5 vec->vec col_major TMOV into row-major reinterpret + TMOV";
  let description = [{
    Rewrites A5 `pto.tmov` patterns that use vec->vec col_major/none_box tiles:
      src(col_major) -> pto.treshape(row_major) -> pto.tmov(row_major->row_major)
      dst(col_major) -> pto.treshape(row_major)
    This avoids unsupported A5 PTO-ISA vec->vec col_major TMOV paths while
    preserving alias semantics via SSA treshape (no real data movement).
  }];
  let constructor = "mlir::pto::createPTOA5NormalizeTMovPass()";
  let dependentDialects = ["pto::PTODialect", "func::FuncDialect"];
}

def PTORemoveIdentityTMov : Pass<"pto-remove-identity-tmov", "func::FuncOp"> {
  let summary = "Remove plain PTO TMOV operations that copy a tile onto itself";
  let description = [{
    Removes side-effect-free `pto.tmov` operations whose source and destination
    are proven to refer to the same physical tile address/range. Except for
    exact same-SSA self-copies, the proof goes through InsertSync's
    `BaseMemInfo` exact-range analysis for statically addressable values.
    Same-address copies are removed only when the TMOV destination/result is
    dead and no later same-address alias use is visible, so a live copy cannot
    be the sole synchronization bridge. This pass runs before InsertSync so
    deleting the identity TMOV also prevents
    synchronization from being generated only for that no-op copy.
  }];
  let constructor = "mlir::pto::createPTORemoveIdentityTMovPass()";
  let dependentDialects = [
    "pto::PTODialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "memref::MemRefDialect"
  ];
}


def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PTORematerializeFixpipeVectorQuant
    : Pass<"pto-rematerialize-fixpipe-vector-quant", "func::FuncOp"> {
  let summary =
      "Rematerialize fixpipe vector quant bindings before each producer tpush";
  let description = [{
    Rewrites logical fixpipe vector quant state (`pto.set_quant_vector`) into
    explicit per-`pto.tpush` bindings in the same block. This keeps the
    scaling-tile read visible to downstream memref passes such as PlanMemory,
    so the scaling tile's address is not reused before a later fixpipe producer
    push still needs it.
  }];
  let constructor =
      "mlir::pto::createPTORematerializeFixpipeVectorQuantPass()";
  let dependentDialects = ["mlir::pto::PTODialect", "mlir::func::FuncDialect"];
}

def PTOMaterializeImplicitTmp
    : Pass<"pto-materialize-implicit-tmp", "func::FuncOp"> {
  let summary = "Materialize implicit tmp tiles for PTO ops before memplan";
  let description = [{
    Rewrites PTO ops with optional tmp operands into explicit tmp forms when
    the backend tmp-aware overload is required. The synthesized tmp is emitted
    as tile-native `pto.alloc_tile(no addr)` so PlanMemory can assign its local
    address together with other tile buffers.
  }];
  let constructor = "mlir::pto::createPTOMaterializeImplicitTmpPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
 let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";

  let options = [
    Option<"memMode", "mem-mode", "std::string", "\"local\"",
           "Planning mode. Supported value today: local">,
    Option<"orderBySize", "order-by-size", "bool", "false",
           "Plan larger buffers first inside one AddressSpace before applying "
           "SPEC_LEVEL_0 first-fit reuse">
  ];

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
}

def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PreFusionAnalysis : Pass<"pto-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Analyze block-local tile-fusion candidates before planning";
  let description = [{
    Builds reusable pre-planning analysis state for tile fusion in tile-native
    PTO IR. This pass is analysis-only and does not emit grouping or
    scheduling decisions.
  }];
  let constructor = "mlir::pto::createPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PrintPreFusionAnalysis
    : Pass<"pto-print-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Print pre-fusion analysis results for testing and debugging";
  let description = [{
    Computes the pre-fusion analysis result and prints a stable textual dump of
    block-local compute nodes, edges, liveness, iteration-domain classes, and
    local boundaries.
  }];
  let constructor = "mlir::pto::createPrintPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def FusionPlan : Pass<"pto-fusion-plan", "func::FuncOp"> {
  let summary = "Build conservative tile-fusion planning groups from block-local analysis";
  let description = [{
    Consumes PreFusionAnalysis results, forms conservative block-local fusion
    groups for currently supported tile-native compute ops, and annotates
    accepted group members with:
      - pto.fusion.group_id
      - pto.fusion.order
  }];
  let constructor = "mlir::pto::createFusionPlanPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
  let options = [
    Option<"enableShapeInference", "enable-shape-inference",
           "bool", /*default=*/"false",
           "Enable shape inference (ShapeConstraintSolver) for tile fusion. "
           "When disabled (default), fall back to static/direct-bound "
           "iteration-domain inference.  Only FusionPlan consumes this option; "
           "FusionRegionGen consumes only the shared pre-fusion dataflow graph "
           "(cached by the analysis manager) and never consults iteration-domain "
           "classes, so it does not need the option.">,
    Option<"enableVfSimCostmodelOptimization",
           "enable-vfsim-costmodel-optimization",
           "bool", /*default=*/"false",
           "Invoke the optional VfSimulator costmodel planner after PTOAS has "
           "emitted FusionPlan group/order metadata.">,
    Option<"dumpVfSimUnrollTest", "dump-vfsim-unroll-test",
           "bool", /*default=*/"false",
           "Print VfSimulator unroll candidate timings for accepted fusion "
           "groups. This is a debug dump only and does not enable or disable "
           "the VfSimulator planner.">
  ];
}

def OpScheduling : Pass<"pto-op-scheduling", "func::FuncOp"> {
  let summary = "Compact planned fusion groups into block-local contiguous spans";
  let description = [{
    Consumes fusion planning metadata emitted by FusionPlanPass, performs
    block-local instruction scheduling, and normalizes the final physical runs
    into contiguous groups. Fragmented groups degrade into independent spans;
    singleton spans lose fusion metadata.
  }];
  let constructor = "mlir::pto::createOpSchedulingPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOMarkLastUse : Pass<"pto-mark-last-use", "func::FuncOp"> {
  let summary = "Mark scheduled tile-fusion operand last-use slots";
  let description = [{
    Walks scheduled tile-fusion spans identified by pto.fusion.group_id /
    pto.fusion.order and annotates each op with a stable per-input last-use
    bit mask. The analysis considers both later in-span uses and later
    post-span uses in the enclosing block.
  }];
  let constructor = "mlir::pto::createPTOMarkLastUsePass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOFusionRegionGen : Pass<"pto-fusion-region-gen", "func::FuncOp"> {
  let summary = "Wrap one scheduled fusion span into one pto.fusion_region";
  let description = [{
    Consumes scheduled tile-fusion metadata and encapsulates one contiguous
    fusion group span into one pto.fusion_region without redefining group
    membership or physical order.
  }];
  let constructor = "mlir::pto::createPTOFusionRegionGenPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOWrapFunctionsInSections : Pass<"pto-wrap-functions-in-sections", "func::FuncOp"> {
  let summary = "Wrap attributed single-core functions in PTO cube/vector sections";
  let description = [{
    Rewrites frontend-marked functions into the existing PTO section form:
    - `pto.kernel_kind = #pto.kernel_kind<cube>` -> wrap the function body in
      `pto.section.cube`
    - `pto.kernel_kind = #pto.kernel_kind<vector>` -> wrap the function body
      in `pto.section.vector`
  }];

  let constructor = "mlir::pto::createPTOWrapFunctionsInSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTONormalizeUncoveredTileSections
    : Pass<"pto-normalize-uncovered-tile-sections", "ModuleOp"> {
  let summary = "Collect uncovered top-level PTO tile segments before section normalization";
  let description = [{
    Scans function top-level blocks for contiguous operation segments that are
    not already covered by `pto.section.cube` / `pto.section.vector`.

    This pass is the shared entry point for inferred-section normalization:
    explicit sections remain in place, while uncovered top-level segments that
    contain section-relevant operations are identified for later wrapping.
  }];

  let constructor = "mlir::pto::createPTONormalizeUncoveredTileSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOValidatePhysicalSectionBoundaries
    : Pass<"pto-validate-physical-section-boundaries", "ModuleOp"> {
  let summary = "Verify SSA isolation at physical section boundaries";
  let description = [{
    Verifies that values defined inside a `pto.section.cube` or
    `pto.section.vector` remain inside that lexical physical section. Values
    defined outside a section may be captured by it, but values cannot be
    passed implicitly between sibling sections or from a section to its
    enclosing function. This validation is run after inferred sections are
    materialized, so explicit and automatically inferred sections obey the
    same boundary rules.
  }];

  let constructor =
      "mlir::pto::createPTOValidatePhysicalSectionBoundariesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOMaterializeTileOpSections
    : Pass<"pto-materialize-tileop-sections", "ModuleOp"> {
  let summary = "Verify PTODSL tileop helpers and materialize their single compute section";
  let description = [{
    Verifies functions marked with `pto.tileop.helper` have Tile/scalar ABI and
    a pure single-core body. The pass infers one Vector or Cube compute kind,
    rejects MTE, pipe synchronization, direct SIMT operations, high-level
    TileOps, and mixed Vector/Cube evidence, then wraps the complete helper
    body in exactly one `pto.section.vector` or `pto.section.cube`.

    Dynamic `pto.tile_valid_rows` / `pto.tile_valid_cols` metadata reads remain
    legal structural operations. When either value is dynamic, the pass
    appends internal index arguments to the helper and its call chain before
    TileBuf parameters are converted to memrefs. The pass records one
    read/write effect entry per helper argument for synchronization analysis;
    internal metadata arguments have no memory effect.

    TileOp helpers are void; their authored Tile/scalar arguments remain
    implicitly visible inside the original operand-free section container,
    and results are written through mutable Tile parameters.
  }];

  let constructor = "mlir::pto::createPTOMaterializeTileOpSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def VPTOSplitCVModule : Pass<"vpto-split-cv-module", "ModuleOp"> {
  let summary = "Split a VPTO module with cube/vector sections into kernel modules";
  let description = [{
    Rewrites a single-module VPTO input where one `pto.kernel` function
    contains `pto.section.cube` / `pto.section.vector` regions into the
    normalized fatobj input form with one child module per kernel kind.

    The pass clones the original module for each target kind, removes the
    opposite section kind, and inlines the target section body. Section regions
    have no results, so MLIR SSA scoping prevents direct scalar values from
    escaping one section into another; cross-core handoff must use caller-owned
    Tile, MTE, and synchronization. After section removal, unreferenced
    `pto.simt_entry` helpers are removed from the clone.
  }];

  let constructor = "mlir::pto::createVPTOSplitCVModulePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def VPTONormalizeContainer : Pass<"vpto-normalize-container", "ModuleOp"> {
  let summary = "Normalize and verify VPTO kernel container shape";
  let description = [{
    Normalizes a VPTO kernel module carrying `pto.kernel_kind` into the canonical
    outer container form. Existing containers are left in place. The pass
    verifies that the resulting top-level module contains only child modules and
    that every child module carries `pto.kernel_kind`.
  }];

  let constructor = "mlir::pto::createVPTONormalizeContainerPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOLowerFrontendPipeOps : Pass<"pto-lower-frontend-pipe-ops", "func::FuncOp"> {
  let summary = "Lower frontend TPUSH/TPOP pipe ops to internal pipe ops";
  let description = [{
    Converts frontend `aic/aiv_initialize_pipe`, `tpush_to_*`,
    `tpop_from_*`, and `tfree_from_*` ops into the internal unified pipe IR:
    `initialize_l2l_pipe` / `initialize_l2g2l_pipe`, `tpush`,
    `declare_tile`, `tpop`, and `tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerFrontendPipeOpsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOAssignDefaultFrontendPipeId : Pass<"pto-assign-default-frontend-pipe-id", "func::FuncOp"> {
  let summary = "Materialize default id = 0 on frontend TPUSH/TPOP pipe ops";
  let description = [{
    Preserves backward compatibility with older frontend pipe syntax that
    omitted `id`. This pass rewrites missing `id` attrs on
    `aic/aiv_initialize_pipe`, `tpush_to_*`, `tpop_from_*`, and
    `tfree_from_*` to an explicit `id = 0` before frontend pipe lowering.
  }];

  let constructor = "mlir::pto::createPTOAssignDefaultFrontendPipeIdPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOInferValidatePipeInit : Pass<"pto-infer-validate-pipe-init", "ModuleOp"> {
  let summary = "Infer and validate internal pipe init nosplit configuration";
  let description = [{
    Runs after frontend pipe lowering and before memory planning. For each
    logical pipe, this pass:
    - validates that downstream `pto.tpush` / `pto.tpop` / `pto.tfree` users
      do not mix `split = 0` with `split = 1/2/3/4`
    - preserves explicit `nosplit` attrs on internal pipe init ops and rejects
      conflicts across peer pipe init pairs
    - infers missing `nosplit` attrs from downstream split usage for backward
      compatibility with older IR that omitted init-level `nosplit`
    - propagates the resolved `nosplit` value across peer pipe init pairs so
      both ends of one logical pipe agree before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOInferValidatePipeInitPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOResolveReservedBuffers : Pass<"pto-resolve-reserved-buffers", "ModuleOp"> {
  let summary = "Resolve reserved local buffer addresses and peer pipe flag bases";
  let description = [{
    Runs after `pto-plan-memory`. Assumes `pto.reserve_buffer` base addresses
    have already been planned, then:
    - aligns missing `flag_base` attrs for peer internal pipe init ops
    - rejects internal pipe init ops without explicit `flag_base` when their
      `local_addr` cannot be traced back to `pto.reserve_buffer` /
      `pto.import_reserved_buffer`
    - replaces `pto.reserve_buffer` / `pto.import_reserved_buffer` results with
      resolved constant local addresses
    - erases the frontend-only reserve/import ops before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOResolveReservedBuffersPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def InsertTemplateAttributes
    : Pass<"pto-insert-template-attributes", "ModuleOp"> {
  let summary = "Attach legal PTODSL template candidates to tile operations";
  let description = [{
    Queries the process-wide PTODSL TileLib runtime for legal template
    candidates and stores the compact candidate list on each tile operation as
    the `candidates` attribute. Each candidate contains only id, name,
    loop_depth, postupdate, and tail metadata.
  }];
  let constructor = "mlir::pto::createInsertTemplateAttributesPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def ExpandTileOp : Pass<"pto-expand-tile-op", "ModuleOp"> {
  let summary = "Expand tile ops into calls to TileLib template functions";
  let description = [{
    Expands tile-level operations (pto.tadd, pto.tsub, etc.) by asking the
    process-wide PTODSL TileLib runtime to instantiate template libraries in
    the current operation's MLIRContext. The generated template functions use
    tile_buf parameters and contain vector-level implementations (pto.vecscope,
    pto.vlds, pto.vadd, pto.vsts, etc.).

    Each tile op is replaced by a func.call to the generated template function,
    with tile_buf operands passed directly (no type bridging).

    After this pass, the Inline pass inlines template bodies, and
    FoldTileBufIntrinsics resolves tile_buf_addr / tile_valid_rows /
    tile_valid_cols.
  }];
  let constructor = "mlir::pto::createExpandTileOpPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
}

def FoldTileBufIntrinsics : Pass<"pto-fold-tile-buf-intrinsics", "mlir::func::FuncOp"> {
  let summary = "Fold structured-view intrinsics after template inlining";
  let description = [{
    After PTODSL template functions are inlined, the IR contains
    structured-view intrinsics whose operands are now bound to concrete values.

    This pass resolves them:
    - pto.tile_buf_addr → replaced by pto.castptr from the active materialized
      tile handle address
    - pto.tile_valid_rows → folded to arith.constant if v_row is static,
      or replaced with the dynamic valid_row operand carried by the
      materialized tile handle
    - pto.tile_valid_cols → same as above for v_col

    tensor_view family:
    - pto.tensor_view_addr → traces through unrealized_conversion_cast →
      subview → reinterpret_cast, then folds to the base memref or to
      pto.castptr/pto.addptr on the base memref
    - pto.get_tensor_view_dim → folded to arith.constant for static subview
      sizes, or to the subview size SSA operand for dynamic dims
    - pto.get_tensor_view_stride → folded to the reinterpret_cast stride
      operand, multiplied by the subview stride when needed

    Dead unrealized_conversion_cast, memref.subview, and
    memref.reinterpret_cast ops exposed by folding are cleaned up after the
    rewrite.
  }];
  let constructor = "mlir::pto::createFoldTileBufIntrinsicsPass()";
  let options = [
    Option<"foldMode", "fold-mode", "std::string",
           /*default=*/"\"all\"",
           "Select which intrinsic family to fold: all, shape-only, or addr-only">
  ];
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOCanonicalizeIR : Pass<"pto-canonicalize-ir", "func::FuncOp"> {
  let summary = "Canonicalize rank-2 view descriptors into rank-5 form (VPTO-only)";
  let description = [{
    Rewrites rank-2 tensor_view / partition_tensor_view descriptors into the
    canonical right-aligned rank-5 form [1, 1, 1, R, C], matching the 5D
    descriptor layout expected by VPTO lowering. Stride expansion uses the
    same cumulative-product rule as rightAlignTo5D and
    buildGlobalTensorShapeAndStride: stride[i] = shape[i+1] * stride[i+1].

    Currently gated on --pto-backend=vpto to limit blast radius. A3/A5
    EmitC codegen already pads strides to rank-5 via InferPTOLayout and
    buildGlobalTensorShapeAndStride, so it does not need this pass at the
    IR level. The gate can be lifted once the pass is proven stable.

    A post-canonicalization verification detects any surviving rank-2 view
    types to prevent silent failures when new view-consuming ops are added.
  }];
  let constructor = "mlir::pto::createPTOCanonicalizeIRPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
}

def LowerPTOToUBufOps : Pass<"pto-lower-to-ubuf-ops", "mlir::func::FuncOp"> {
  let summary = "Lower eligible pto.tadd/tsub/tmul/tdiv to pto.ub.vadd/vsub/vmul/vdiv on a2a3 targets";
  let description = [{
    Converts pto.tadd/tsub/tmul/tdiv on UB tiles to pto.ub.vadd/vsub/vmul/vdiv
    when targeting a3 (dav-c220-vec).  Computes repeat count and repeat strides
    from the tile shape metadata and emits tile_buf_addr ops to extract UB
    pointers.  pto.ub.* binary ops are later lowered to llvm.hivm.VADD/VSUB/VMUL/VDIV.*
    by the VPTOLLVMEmitter.

    Ops that do not meet the eligibility criteria (wrong arch, non-UB space,
    dynamic shapes, unaligned repeats) pass through to the existing ExpandTileOp
    TileLang path unchanged.
  }];
  let constructor = "mlir::pto::createLowerPTOToUBufOpsPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOInlineLibCall : Pass<"pto-inline-libcall", "ModuleOp"> {
  let summary = "Materialize OP-Lib instance bodies and inline OP-Lib calls";
  let description = [{
    Resolves OP-Lib instance declarations generated by OP-Lib lowering,
    materializes instance bodies, and inlines OP-Lib calls into caller/fused
    helper functions. Function signatures stay in !pto.tile_buf form.
  }];
  let constructor = "mlir::pto::createPTOInlineLibCallPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for OP-Lib instantiation/inlining">];
}

def PTOInlineBackendHelpers
    : Pass<"pto-inline-backend-helpers", "ModuleOp"> {
  let summary = "Inline backend helper functions shared by VPTO and EmitC";
  let description = [{
    Force-inlines backend helper functions that should not survive to backend-
    specific lowering:
    - PTODSL TileOp helpers marked with `pto.tileop.helper`

    This pass runs on the shared mainline before backend-specific pipelines so
    both VPTO and EmitC consume the same helper-inlined IR.
  }];
  let constructor = "mlir::pto::createPTOInlineBackendHelpersPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for backend helper inlining">];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify explicit matching pto.tfree placement for pto.tpop";
  let description = [{
    For each `pto.tpop` in section.cube / section.vector, verify there is a
    matching later `pto.tfree(pipe_handle)` in the same block, the borrowed
    tile is not used after the free, and the same pipe does not accumulate
    multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOValidateIntToPtrUses : Pass<"pto-validate-inttoptr-uses", "func::FuncOp"> {
  let summary = "Validate restricted uses of PTO inttoptr results";
  let description = [{
    Verifies that `pto.inttoptr` results do not escape into general pointer IR.
    The result may only be used directly as the pointer operand of
    `pto.load_scalar` or `pto.store_scalar`.
  }];

  let constructor = "mlir::pto::createPTOValidateIntToPtrUsesPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOResolveBufferSelect : Pass<"pto-resolve-buffer-select", "ModuleOp"> {
  let summary = "Resolve tile views and multi-buffer selections to addressed handles";
  let description = [{
    Consumes tile-native `pto.subview` operations rooted at an addressed
    `pto.alloc_tile` or a runtime-bound `pto.declare_tile`, and
    `pto.multi_tile_get %src[%k]` operations. Subviews become zero-copy
    addressed `pto.alloc_tile` handles after folding their physical offset.

    The source `pto.alloc_multi_tile` carries either planner-assigned physical
    slot addresses or a level3 base address. Each selection is replaced by a
    `pto.alloc_tile` carrying the selected address; constant slots select one
    address directly, while dynamic slots use an N-way `arith.select` chain.

    This pass does not rewrite `%k` into `iv mod N`; the frontend expression
    remains the slot selector.
  }];

  let constructor = "mlir::pto::createPTOResolveBufferSelectPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOUnrollLoops : Pass<"pto-unroll-loops", "func::FuncOp"> {
  let summary =
      "Unroll explicitly annotated scf.for loops";
  let description = [{
    Fully or partially unrolls `scf.for` loops that carry an explicit unroll
    annotation, regardless of whether they are inside a SIMT context:

    - `{pto.unroll = "full"}`: fully unrolled via `loopUnrollByFactor` with
      the trip count as the factor (single-iteration promotion removes the
      resulting loop shell).  Only loops with a constant trip count are
      unrolled; other loops keep the loop and drop the hint with a remark.
    - `{pto.unroll_factor = N}`: unrolled by N via `loopUnrollByFactor`,
      which supports dynamic upper bounds and generates an epilogue
      (remainder) loop that threads live-out loop-carried values.  Only
      loops with a statically known positive step are unrolled; a no-op
      factor (1), a dynamic step, or a factor above `max-unroll-factor`
      keeps the loop and drops the hint with a remark.

    Malformed hints are hard errors: an unknown `pto.unroll` value, both
    attributes on one loop, or an out-of-contract factor (not a signless
    i32, or non-positive).

    The annotation is opt-in: loops without an attribute are left unchanged
    regardless of trip count.

    Full unrolling enables subsequent canonicalize/sccp/cse to constant-fold
    induction-variable-dependent conditionals (e.g. `scf.if i < 10`),
    eliminating divergent branches inside SIMTVF kernels.

    This pass must run before `createConvertSCFToCFPass` (i.e. before LLVM
    lowering) so that the unrolled structured control flow can be
    canonicalized.
  }];
  let constructor = "mlir::pto::createPTOUnrollLoopsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"maxFullUnrollTripCount", "max-full-unroll-trip-count", "int64_t",
           /*default=*/"1024",
           "Emit a warning when fully unrolling a loop whose static trip "
           "count exceeds this threshold.  The loop is still unrolled.">,
    Option<"maxUnrollFactor", "max-unroll-factor", "int64_t",
           /*default=*/"1024",
           "Do not natively unroll by a factor larger than this; the loop "
           "is kept and the hint is dropped with a remark instead.  "
           "A negative value disables the cap.">,
  ];
}

// Legacy alias for PTOUnrollLoops.  Kept so existing pipelines and tests
// that reference "pto-unroll-simt-for" keep working; the behavior is
// identical to "pto-unroll-loops".
def PTOUnrollSIMTFor : Pass<"pto-unroll-simt-for", "func::FuncOp"> {
  let summary =
      "Alias of pto-unroll-loops kept for backward compatibility";
  let description = [{
    Behaves exactly like `pto-unroll-loops`.  Historically this pass only
    unrolled `{pto.unroll = "full"}` loops inside SIMT contexts; the SIMT
    restriction has been lifted because the annotation is always explicit.
  }];
  let constructor = "mlir::pto::createPTOUnrollSIMTForPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"maxFullUnrollTripCount", "max-full-unroll-trip-count", "int64_t",
           /*default=*/"1024",
           "Emit a warning when fully unrolling a loop whose static trip "
           "count exceeds this threshold.  The loop is still unrolled.">,
    Option<"maxUnrollFactor", "max-unroll-factor", "int64_t",
           /*default=*/"1024",
           "Do not natively unroll by a factor larger than this; the loop "
           "is kept and the hint is dropped with a remark instead.  "
           "A negative value disables the cap.">,
  ];
}

def PTOConvertSCFToCFWithLoopHints : Pass<"pto-convert-scf-to-cf-with-loop-hints", "func::FuncOp"> {
  let summary =
      "Convert SCF to CF, preserving the enable loop hint as an LLVM loop "
      "annotation";
  let description = [{
    Translates `{pto.unroll = "enable"}` attributes on `scf.for` into
    `#llvm.loop_annotation<unroll = <disable = false>>` attributes so that
    the `!llvm.loop.unroll.enable` metadata reaches the emitted LLVM IR and
    the downstream compiler's cost model decides whether and how to unroll
    (LLVM's ForceEnable semantics).  All other unroll hints are owned by
    `pto-unroll-loops` and are left untouched.

    An existing `llvm.loop_annotation` attribute on the loop is merged rather
    than overwritten (a pre-existing unroll entry is replaced with a
    warning).

    The stock LLVM 19 `convert-scf-to-cf` does not propagate
    `llvm.loop_annotation` from `scf.for` to the loop latch, so this pass
    owns the SCF-to-CF conversion for the function: it runs the upstream
    conversion patterns together with a higher-benefit pattern that lowers
    annotated `scf.for` loops (mirroring `ForLowering`) and attaches the
    annotation to the latch `cf.br`.

    Converting the whole function is required for correctness - lowering an
    annotated loop in isolation would leave several blocks inside whatever
    enclosing single-block region held it (an outer `scf.for`, an `scf.if`,
    ...) and fail that op's verifier.

    Consequently this pass **replaces** `createConvertSCFToCFPass` in the
    pipelines that run it; it must sit at the same position, after every
    structured-loop transformation, so no later pass can clone a loop and
    lose its hint.
  }];
  let constructor = "mlir::pto::createPTOConvertSCFToCFWithLoopHintsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def PTOPromotePersistentFragmentLoops
    : Pass<"pto-promote-persistent-fragment-loops", "func::FuncOp"> {
  let summary =
      "Promote loops accessing persistent SIMT fragment buffers to forced "
      "full unrolling";
  let description = [{
    Persistent fragment materialization requires every loop touching a
    persistent buffer to be fully unrolled so that each access resolves to a
    stable resident slot.  This pass is the discovery half of that pipeline:
    it finds the loops and marks them for `pto-unroll-loops`; it does not do
    slot assignment, keep/resume generation, SIMT outlining, or LLVM
    metadata lowering.

    Every `llvm.alloca` carrying `{pto.persistent}` is an explicit entry
    point.  The pass walks the alloca's use graph and collects every
    enclosing `scf.for` of every related op (all nesting layers, including
    kernel-level loops wrapping `pto.section.simt`).  Each collected loop is
    promoted to `{pto.unroll = "full"}` (`"enable"` and no-hint are
    overridden, an existing `"full"` is kept) and additionally marked with
    `{pto.persistent_unroll}` so that `pto-unroll-loops` fails hard instead
    of dropping the hint when the loop cannot be unrolled natively - a
    surviving persistent loop would break fragment materialization.

    Fail-fast (hard errors): a fixed `pto.unroll_factor` on a persistent
    loop, a persistent access nested under `scf.while`, or a statically
    known trip count above `max-persistent-unroll-trip-count`.
  }];
  let constructor =
      "mlir::pto::createPTOPromotePersistentFragmentLoopsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::pto::PTODialect"
  ];
  let options = [
    Option<"maxPersistentUnrollTripCount",
           "max-persistent-unroll-trip-count", "int64_t",
           /*default=*/"128",
           "Fail when a persistent fragment loop with a statically known "
           "trip count exceeds this threshold.  A negative value disables "
           "the guardrail.">,
  ];
}

def PTONarrowVPTOLoopCounters
    : Pass<"pto-narrow-vpto-loop-counters", "func::FuncOp"> {
  let summary =
      "Narrow constant-bounded scf.for counters inside VPTO vecscope regions to i16";
  let description = [{
    Rewrites `scf.for` operations nested under `pto.vecscope` or
    `pto.strict_vecscope` when the lower bound, upper bound, positive step, and
    post-loop induction value are compile-time integer constants representable
    as signed i16 values.

    The rewritten loop uses i16 bounds and induction variable. The induction
    variable is cast back to its original index or wider integer type at the
    start of the body, preserving the authored body and result types while
    exposing the narrow counter form expected by the A5 hardware-loop backend.

    Loops with dynamic bounds, out-of-range constants, an overflowing final
    increment, non-positive steps, counters already 16 bits or narrower, or
    loops outside VPTO vecscope regions are left unchanged.
  }];
  let constructor =
      "mlir::pto::createPTONarrowVPTOLoopCountersPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::pto::PTODialect"
  ];
}

def PTOAnalyzeSIMTPersistentFragment
    : Pass<"pto-analyze-simt-persistent-fragment", "func::FuncOp"> {
  let summary = "Analyze SIMT persistent fragments without changing IR";
  let description = [{
    Validates persistent fragment definitions, access graphs, init/carry
    sections, resident elements, and keep/resume slot assignments. This pass
    is analysis-only; it does not insert or remove operations.
  }];
  let constructor =
      "mlir::pto::createPTOAnalyzeSIMTPersistentFragmentPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::pto::PTODialect"
  ];
}

def PTOMaterializeSIMTPersistentFragment
    : Pass<"pto-materialize-simt-persistent-fragment", "func::FuncOp"> {
  let summary =
      "Materialize SIMT persistent fragments as keep/resume slot traffic";
  let description = [{
    Consumes `llvm.alloca` values marked with `pto.persistent` in a kernel body
    before inline SIMT sections are outlined. The first implementation handles
    single-block `pto.section.simt` bodies with scalar and fixed-width vector
    LLVM load/store accesses through statically-resolved offsets. Vector
    accesses are split into scalar lanes before the section-local state is
    rewritten into `pto.resume` at entry and `pto.keep` at exit.

    Unsupported persistent fragment shapes fail fast because the allocation is
    only a lexical carrier for register-resident state, not real stack memory.
  }];
  let constructor =
      "mlir::pto::createPTOMaterializeSIMTPersistentFragmentPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOOutlineSIMTSections
    : Pass<"pto-outline-simt-sections", "ModuleOp"> {
  let summary =
      "Outline inline pto.section.simt regions into pto.simt_entry functions";
  let description = [{
    Converts each inline `pto.section.simt` region into a private helper
    function marked with `pto.simt_entry`, captures external SSA values as
    helper arguments, and replaces the original section with a
    `pto.simt_launch` using the section launch dimensions.

    This is the late outline step used by TileLang/PTODSL style frontends that
    keep SIMT code inline until PTOAS has had a chance to materialize
    persistent SIMT fragments.
  }];
  let constructor = "mlir::pto::createPTOOutlineSIMTSectionsPass()";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOInferVPTOVecScope
    : Pass<"pto-infer-vpto-vecscope", "func::FuncOp"> {
  let summary =
      "Infer missing pto.vecscope regions for VPTO vector operation clusters";
  let description = [{
    Runs near the VPTO emission boundary after VMI physicalization and the
    existing pre-emission canonicalization, pointer normalization, and
    wrapper-op expansion, but before VMI LICM and final cleanup. The pass
    greedily clusters contiguous VPTO vector operations into `pto.vecscope`
    regions while preserving explicit vector-scope carriers and treating
    DMA/copy/sync, unresolved calls, terminators, and forbidden operations as
    boundaries.

    The inferred `pto.vecscope` form remains resultless. Values whose type is
    `!pto.vreg`, `!pto.mask`, or `!pto.align` must not escape the inferred
    scope.
  }];
  let constructor = "mlir::pto::createPTOInferVPTOVecScopePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOIR : Pass<"pto-validate-vpto-ir", "ModuleOp"> {
  let summary =
      "Validate authoring-stage VPTO legality before ptr-boundary canonicalization";
  let description = [{
    Runs the authoring-stage VPTO legality verifier on post-mainline VPTO IR.
    This stage keeps the memref-first authoring surface legal, while checking
    the shared structural contracts that must hold before emission-boundary
    canonicalization.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVMIIR : Pass<"pto-validate-vmi-ir", "ModuleOp"> {
  let summary = "Validate VMI producer-boundary semantic IR";
  let description = [{
    Checks that VMI producer-boundary IR uses only surface VMI data/mask types,
    native pto.vmi semantic ops, and structural control-flow/function ops. This
    pass runs before layout assignment, so layout-assigned VMI types, VMI helper
    ops, and physical VPTO register types are rejected.
  }];
  let constructor = "mlir::pto::createPTOValidateVMIIRPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVMILayoutIR
    : Pass<"pto-validate-vmi-layout-ir", "ModuleOp"> {
  let summary = "Validate layout-assigned VMI IR";
  let description = [{
    Checks the post-layout-assignment VMI stage: every VMI data value must have
    a concrete VMI layout, every VMI mask must have concrete b8/b16/b32
    granularity and layout, physical VPTO register values must not appear yet,
    and VMI typed values must stay inside VMI semantic/helper or structural ops.
    vmi-to-vpto chooses deterministic lowerings from the current op's attrs,
    operand/result types, layouts, and operand values. Non-local choices must
    be represented as explicit attrs, helper ops, or diagnostics before this
    stage. Later VMI layout optimization passes may replace helpers with
    cloned/rematerialized producers, but the layout gate must not depend on
    hidden producer/user context.
  }];
  let constructor = "mlir::pto::createPTOValidateVMILayoutIRPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMIPreAssignmentCombine
    : Pass<"vmi-pre-assignment-combine", "ModuleOp"> {
  let summary = "Combine VMI operations before layout assignment";
  let description = [{
    Performs VMI-level structural combines before VMI layout assignment. This
    keeps layout assignment focused on choosing and materializing layouts while
    still exposing direct semantic operations to the later layout and lowering
    passes.

    The pass rewrites a `group_load` whose constant row stride equals its
    logical group size into an ordinary contiguous `load`. It also rewrites
    the semantic pattern `group_broadcast(group_slot_load(...))` into the
    equivalent `group_broadcast_load` operation.
  }];
  let constructor = "mlir::pto::createVMIPreAssignmentCombinePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect"];
}

def VMILayoutAssignment : Pass<"vmi-layout-assignment", "ModuleOp"> {
  let summary = "Assign concrete VMI layouts";
  let description = [{
    Solves VMI layout constraints and materializes the chosen layout into VMI
    types. Mask granularity is assigned by vmi-mask-granularity-assignment
    before this pass, so this pass only propagates and materializes mask
    layouts.
  }];
  let constructor = "mlir::pto::createVMILayoutAssignmentPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMILayoutRematerializeWeakProducers
    : Pass<"vmi-layout-rematerialize-weak-producers", "ModuleOp"> {
  let summary = "Rematerialize weak layout producers before layout assignment";
  let description = [{
    Clones pure layout-polymorphic producers at their multiple uses before
    layout assignment. This keeps independent consumer dataflows from being
    connected only by a shared rematerializable SSA value. Strong dataflow
    layout propagation remains the responsibility of vmi-layout-assignment.
  }];
  let constructor = "mlir::pto::createVMILayoutRematerializeWeakProducersPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMIMaskGranularityAssignment
    : Pass<"vmi-mask-granularity-assignment", "ModuleOp"> {
  let summary = "Assign concrete VMI mask granularities";
  let description = [{
    Assigns b8/b16/b32 granularity to VMI mask values before layout
    assignment. This pass follows VMI semantic op contracts where mask
    granularity is determined by the associated data element width. It does
    not choose layouts; conflicting use granularities are represented by
    rematerializing cheap mask producers when possible, otherwise with
    pto.vmi.ensure_mask_granularity.
  }];
  let constructor = "mlir::pto::createVMIMaskGranularityAssignmentPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMILayoutFold : Pass<"vmi-layout-fold", "ModuleOp"> {
  let summary = "Fold VMI layout materializations";
  let description = [{
    Optimizes legal layout-assigned VMI IR by folding selected ensure_layout
    helpers into layout-aware producers or consumers while preserving the same
    logical effect. The pass does not choose layouts by inspecting arbitrary
    producer/user context for vmi-to-vpto; it only rewrites explicit helper IR
    into equivalent local forms.
  }];
  let constructor = "mlir::pto::createVMILayoutFoldPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMILayoutRematerialize : Pass<"vmi-layout-rematerialize", "ModuleOp"> {
  let summary = "Rematerialize cheap VMI producers at layout helpers";
  let description = [{
    Optimizes legal layout-assigned VMI IR by replacing selected ensure_layout,
    ensure_mask_layout, and ensure_mask_granularity helpers with cloned
    producers that directly create the requested result type. The pass covers
    pure construction ops, selected layout-transparent data ops, and dense
    widening ext relation rematerialization. Memory, control-flow, and mask-tail
    proofs remain explicit in the IR.
  }];
  let constructor = "mlir::pto::createVMILayoutRematerializePass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMILayoutSinkMaterialization
    : Pass<"vmi-layout-sink-materialization", "ModuleOp"> {
  let summary = "Sink VMI layout materialization through transfer ops";
  let description = [{
    Optimizes legal layout-assigned VMI IR by moving matching operand
    ensure_layout helpers across pure layout-transparent elementwise operations.
    The rewritten IR keeps the layout conversion explicit as a result
    ensure_layout, so vmi-to-vpto still lowers from local op information only.
  }];
  let constructor = "mlir::pto::createVMILayoutSinkMaterializationPass()";
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMILegalizeArithSelect : Pass<"vmi-legalize-arith-select", "ModuleOp"> {
  let summary = "Legalize canonical arith.select over VMI values";
  let description = [{
    Rewrites scalar-condition arith.select operations that produce VMI values
    back to scf.if. MLIR canonicalization may fold simple scf.if regions into
    arith.select, but VMI values must not cross non-VMI semantic ops before
    vmi-to-vpto. This pass restores an explicit structural control-flow form
    that the VMI converter already handles.
  }];
  let constructor = "mlir::pto::createVMILegalizeArithSelectPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VMINormalizeSignlessIntToUnsigned :
    Pass<"pto-vmi-normalize-signless-int-to-unsigned", "func::FuncOp"> {
  let summary =
      "Normalize signless integer element types on whitelisted VMI ops to "
      "unsigned by wrapping their vreg operands/results in pto.vmi.bitcast";
  let description = [{
    Walks whitelisted VMI ops (currently pto.vmi.vdhist / pto.vmi.vchist)
    and rewrites any VMIVReg operand or result whose element type is a
    signless integer into the equivalent unsigned integer at the op's
    boundary, using pto.vmi.bitcast to preserve the surrounding IR's type
    contract. Downstream verifiers and lowerings therefore see only
    unsigned-typed whitelisted ops, matching the "signless == unsigned"
    convention at the VMI layer while leaving non-whitelisted IR untouched.
  }];
  let constructor = "pto::createVMINormalizeSignlessIntToUnsignedPass()";
  let dependentDialects = ["pto::PTODialect"];
}

def VMILowerUnifiedToLegacy : Pass<"vmi-lower-unified-to-legacy", "ModuleOp"> {
  let summary = "Lower unified VMI ops to legacy equivalents before layout assignment";
  let description = [{
    Expands most unified v-prefixed VMI ops into their legacy (mask-less)
    equivalents so downstream layout and VMIToVPTO passes only see legacy ops.

    Ops lowered (Category A–C6):
      A:  vci, vinterpret_cast, vsel, vbrc  →  iota, bitcast, select, broadcast/group_broadcast
      B:  vadd/vsub/vmul/vdiv/vmin/vmax/vand/vor/vxor/vshl/vshr (masked binary)
          vneg/vabs/vsqrt/vexp/vln/vrelu/vnot (masked unary)
          →  binary ops discard mask/pmode; unary ops preserve zero mode with select;
             vshr selects shrui for explicit unsigned elements and shrsi otherwise
      C1: vcmp/vcmps  →  legacy cmp + select
      C2: vcvt        →  legacy extf/truncf/fptosi/sitofp/extsi/extui/trunci
      C3: vload/vstore  →  legacy load/store variants
      C4: pset/pge    →  create_mask/create_group_mask
      C6: vcadd/vcmax/vcmin  →  legacy reduce variants

    Ops NOT lowered (no legacy equivalent — require direct VMIToVPTO 1:N patterns):
      plt, vadds, vmuls, vmaxs, vmins, vshls, vshrs, vhist, vintlv, vdintlv, vselr,
      vgather, vgatherb, vscatter,
      vexpdif, vaxpy, vlrelu, vprelu, vmull, vmula
  }];
  let constructor = "mlir::pto::createVMILowerUnifiedToLegacyPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VMIToVPTO : Pass<"vmi-to-vpto", "ModuleOp"> {
  let summary = "Convert layout-assigned VMI IR to physical VPTO IR";
  let description = [{
    Converts layout-assigned VMI aggregate data/mask types to ordered physical
    VPTO register and mask value lists using MLIR native 1:N dialect conversion
    APIs. This
    pass is responsible for VMI 1:N type conversion, structural control-flow
    and function/call signature conversion, and VMI semantic op physicalization.
  }];
  let constructor = "mlir::pto::createVMIToVPTOPass()";
  let options = [
    Option<"enableStableGatherMaskedLoad",
           "enable-stable-gather-masked-load", "bool",
           /*default=*/"false",
           "Reserve the stable VGATHER-based lowering path for VMI masked "
           "loads; currently emits a TODO diagnostic when used.">
  ];
  let dependentDialects = ["mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VPTOStatefulStreamFusion : Pass<"vpto-stateful-stream-fusion", "ModuleOp"> {
  let summary = "Fuse compatible VPTO stateful load and store streams";
  let description = [{
    Fuses adjacent VPTO streams that thread alignment state through SSA. Load
    streams use vldas/vldus and store streams use init_align/vstus with a
    matching flush. Fusion requires provably contiguous addresses and does not
    cross observable memory effects or control flow.
  }];
  let constructor = "mlir::pto::createVPTOStatefulStreamFusionPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOExpandSoftLib : Pass<"pto-expand-soft-lib", "ModuleOp"> {
  let summary = "Expand VPTO operations that lack native target support";
  let description = [{
    Materializes matching PTODSL SoftOps implementations for f32 SIMT
    `pto.sin`/`pto.cos` at the final VPTO emission boundary. The generated
    helpers are then inlined by PTOInlineLibCall. A5 `pto.vdiv`, including its
    supported integer forms, remains on the native emitter path. The A2/A3
    `pto.ub.vdiv` operation is a different operation and is intentionally not
    handled by this pass. Trigonometric helpers use the PTODSL software
    approximation authored in `lib/SoftOps`.
  }];
  let constructor = "mlir::pto::createPTOExpandSoftLibPass()";
  let dependentDialects = ["mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect"];
}

def PTOValidateVPTOEmissionIR
    : Pass<"pto-validate-vpto-emission-ir", "ModuleOp"> {
  let summary =
      "Validate emission-stage VPTO legality after ptr-boundary canonicalization";
  let description = [{
    Runs the emission-stage VPTO legality verifier on ptr-form VPTO IR after
    `PTOVPTOPtrBoundary`. This stage re-checks the shared authoring contracts
    and confirms the final emission surface no longer carries memref boundary
    state or residual wrapper scaffold.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOEmissionIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def VPTOScheduler : Pass<"pto-vpto-scheduler", "ModuleOp"> {
  let summary = "Analyze or schedule emission-ready VPTO regions";
  let description = [{
    Builds block-local scheduling regions and typed dependency DAGs for
    operations inside dedicated `pto.vecscope` and `pto.strict_vecscope`
    regions.
    The target model supports A5 vector kernel modules only:
    normalized cube kernel modules are skipped, while modules without an
    explicit kernel kind remain available for standalone pass testing. The
    module must declare `pto.target_arch = "a5"` directly or inherit it from
    an enclosing module; a missing or different target is rejected.
    `off` is an exact no-op. `analyze` and `on` share region construction,
    dependency DAG construction, coverage, and original-order pressure
    analysis. `analyze` stops after that common prefix without mutating IR.
    `on` additionally performs deterministic top-down list scheduling through
    an injectable pressure-aware candidate strategy. It preserves critical
    path priority at low pressure, prefers live-range-closing candidates near
    a modeled pressure limit within one modeled-latency urgency window, and
    retains verified pressure-driven advancement when all ready candidates
    exceed a limit. Validation and fresh replay precede application of each
    verified result.
  }];
  let constructor = "mlir::pto::createVPTOSchedulerPass()";
  let options = [
    Option<"mode", "mode", "std::string", "\"off\"",
           "Scheduler mode: off, analyze, or on">,
    Option<"trace", "trace", "bool", "false",
           "Print detailed on-mode schedule results">
  ];
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def VPTOExpandWrapperOps
    : Pass<"vpto-expand-wrapper-ops", "func::FuncOp"> {
  let summary =
      "Expand VPTO wrapper ops to emission-ready low-level VPTO IR";
  let description = [{
    Expand higher-level VPTO wrapper operations, including bridge, DMA, cube,
    accumulator-store, and MAD semantic forms, to the low-level pointer/raw VPTO
    operations consumed by backend emission.
  }];
  let constructor = "mlir::pto::createVPTOExpandWrapperOpsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::pto::PTODialect"];
}

def VPTOSoftPostUpdate
    : Pass<"vpto-soft-postupdate", "ModuleOp"> {
  let summary = "Convert fixed-stride VPTO memory ops to post-update form";
  let description = [{
    Uses VPTOAddressSemanticsOpInterface and the function-scoped
    PTOAddressAnalysis to identify eligible VPTO memory operations whose
    effective address advances by an exactly representable symbolic stride.
    Loop-carried accesses use scf.for iter_args. Typed i8, i16, i32, and index
    recurrences are accepted only when their real signed/unsigned cast and
    no-wrap semantics are proven; unsupported or inexact queries remain in
    normal form. Remaining arithmetic runs within each block are rewritten
    only when a conservative structural profitability check predicts address
    computation savings. For an accepted sequential run, all but the last
    operation form an updated_base pointer chain, while the last operation
    uses the current pointer in normal zero-offset form. Loop-aware liveness
    removes software accumulator recurrences made obsolete by new pointer
    chains while retaining accumulators with observable uses.
  }];
  let constructor = "mlir::pto::createVPTOSoftPostUpdatePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def VPTOGuardedLICM : Pass<"vpto-guarded-licm", "func::FuncOp"> {
  let summary = "Hoist loop-invariant guarded expressions out of scf.if regions";
  let description = [{
    The generic loop-invariant code motion pass only inspects the top-level
    operations of the scf.for body.  A guarded access pattern that computes a
    loop-invariant base address chain inside an scf.if whose condition depends
    on the induction variable is therefore never exposed to LICM, and the
    surrounding scf.if cannot itself be hoisted because its region contains
    side-effecting memory operations.

    This pass extracts the pure, speculatable scalar and address
    subexpressions (integer, index, and pto.ptr results) of such guarded
    regions and moves them in dependency (topological) order to just before
    the enclosing scf.for.  IV-dependent arithmetic, side-effecting
    operations, div/rem with unknown divisors, vector/mask/tile computations,
    and operations inside nested scf.for regions are left in place.  Loops
    are processed innermost-first so invariant chains can climb out through
    arbitrarily deep guards and nested loops.  When a property cannot be
    proven the pass conservatively keeps the expression in place.  The
    rewrite is enabled only for the A5 VPTO backend.
  }];
  let constructor = "mlir::pto::createVPTOGuardedLICMPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOPrintAddressAnalysis
    : Pass<"pto-print-address-analysis", "func::FuncOp"> {
  let summary = "Print cached PTO value-evolution and VPTO address facts";
  let description = [{
    Test/debug-only printer for PTOValueEvolutionAnalysis and
    PTOAddressAnalysis. It owns no analysis logic and is not required by any
    production pipeline.
  }];
  let constructor = "mlir::pto::createPTOPrintAddressAnalysisPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOVPTOPtrBoundary
    : Pass<"pto-vpto-ptr-boundary", "ModuleOp"> {
  let summary =
      "Canonicalize the final VPTO emission boundary from memref-first IR to ptr ABI";
  let description = [{
    Runs the final emission-boundary ptr canonicalization after the backend
    mainline has finished its memref-first optimization pipeline. This pass
    rewrites eligible memref function arguments to same-space `!pto.ptr`,
    rejects memref function results, canonicalizes supported body-level VPTO
    buffer-like ops to ptr-form, and drops dead boundary/view scaffold such as
    trivial `pto.castptr`, `memref.subview`,
    `memref.reinterpret_cast`, and `memref.memory_space_cast` once they become
    unused.
  }];
  let constructor = "mlir::pto::createPTOVPTOPtrBoundaryPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary =
      "Fuse adjacent VPTO post-lowering loop nests inside pto.fusion_region";
  let description = [{
    Conservatively fuses adjacent post-lowering loop stages for already-planned
    fusion groups after the tile-native frontend fusion core. The formal input
    contract for this pass is low-level `scf.for + pto.*` structure that
    remains inside `pto.fusion_region` until explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOLowLevelLoopFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::vector::VectorDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for low-level loop fusion">];
}

def PTOFusionPredicateElision
    : Pass<"pto-fusion-predicate-elision", "func::FuncOp"> {
  let summary =
      "Elide redundant fusion-local VPTO plt predicate materialization";
  let description = [{
    Scans `pto.fusion_region` bodies after VPTO post-lowering CSE and prepares
    fusion-local redundant `pto.plt_*` predicate elimination before
    load/store cleanup and explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOFusionPredicateElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFusionLoadStoreElision
    : Pass<"pto-fusion-load-store-elision", "func::FuncOp"> {
  let summary =
      "Elide fusion-local VPTO load/store round trips before flattening";
  let description = [{
    Removes local store materialization inside canonical fused loop bodies
    before explicit flatten. It also performs conservative frontier-aware
    cleanup for non-escaping fusion-region internal buffers.
  }];
  let constructor = "mlir::pto::createPTOFusionLoadStoreElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOVexpdifFusion : Pass<"pto-vexpdif-fusion", "func::FuncOp"> {
  let summary = "Fuse VPTO vsub and vexp into vexpdif";
  let description = [{
    Performs fusion after TileOp expansion and fusion-local load/store
    elimination. The pass rewrites an f32 `vsub` followed by an f32 `vexp`
    with the same mask into `vexpdif`.
  }];
  let constructor = "mlir::pto::createPTOVexpdifFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect"];
}

def PTOUnrollAfterLoopFusion
    : Pass<"pto-unroll-after-loop-fusion", "func::FuncOp"> {
  let summary = "Partial-unroll the innermost scf.for inside pto.fusion_region "
                "after PTOLowLevelLoopFusion (divisible trip count only)";
  let description = [{
    Partial-unrolls the loop nests inside `pto.fusion_region` according to the
    unroll factors supplied by the cost model.
  }];
  let constructor = "mlir::pto::createPTOUnrollAfterLoopFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFlattenFusionRegion : Pass<"pto-flatten-fusion-region", "func::FuncOp"> {
  let summary = "Flatten pto.fusion_region wrappers back into the parent block";
  let description = [{
    Eliminates residual `pto.fusion_region` wrappers after region-local
    lowering is complete by moving body operations back to the parent block,
    replacing region results with `pto.yield` operands, and erasing both
    `pto.yield` and the wrapper op.
  }];
  let constructor = "mlir::pto::createPTOFlattenFusionRegionPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VPTOPtrNormalize
    : Pass<"vpto-ptr-normalize", "ModuleOp"> {
  let summary =
      "Normalize VPTO ptr-like values and users into a uniform !pto.ptr form";
  let description = [{
    Uses MLIR's conversion framework to normalize VPTO ptr-related forms before
    the existing VPTO ptr-boundary canonicalization runs. This pass rewrites
    supported tile-buffer and memref view producers such as `pto.tile_buf_addr`
    and `memref.subview`, and updates VPTO memory ops to consume the normalized
    ptr-form consistently.
  }];
  let constructor = "mlir::pto::createVPTOPtrNormalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def VPTOPtrCastCleanup
    : Pass<"vpto-ptr-cast-cleanup", "ModuleOp"> {
  let summary = "Collapse transient ptr-memref-ptr bridge casts after VPTO ptr normalization";
  let description = [{
    Eliminates bridge chains such as
    `!pto.ptr -> builtin.unrealized_conversion_cast -> memref.cast ->
     builtin.unrealized_conversion_cast -> !pto.ptr`
    when the outer ptr types already match.
  }];
  let constructor = "mlir::pto::createVPTOPtrCastCleanupPass()";
  let dependentDialects = ["mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

def VPTOOptimizeVcvt
    : Pass<"vpto-optimize-vcvt", "ModuleOp"> {
  let summary = "Normalize or remove provably redundant VPTO vcvt operations";
  let description = [{
    Rewrites `pto.vcvt` operations whose `EVEN` and `ODD` part selections are
    provably equivalent into the canonical `EVEN` form. The pass currently
    recognizes all-true masked narrow-to-wide conversions from VPTO values with
    pair-wise equivalent input lanes, such as scalar/vector broadcasts and
    selected broadcast load distributions. It also replaces unsigned widening
    conversions with `pto.vbitcast` when a canonical zero-filling producer --
    an unpack load, `pto.vzunpack` chain, or aligned narrowing `pto.vcvt` --
    proves that every widened carrier's high bits are zero. A following CSE
    pass can then merge duplicate conversions and remove dead masks.
  }];
  let constructor = "mlir::pto::createVPTOOptimizeVcvtPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VPTOCombineReductions
    : Pass<"vpto-combine-reductions", "ModuleOp"> {
  let summary = "Combine VPTO reduction trees with equivalent masks";
  let description = [{
    Reassociates conservative physical reduction trees emitted by VMI-to-VPTO
    lowering. When every chunk reduction uses an equivalent mask, the pass
    combines the source chunks first and emits one vcadd/vcmax/vcmin or
    vcgadd/vcgmax/vcgmin operation. Run this after mask simplification, LICM,
    and CSE so equivalent predicates have stable VPTO producers.
  }];
  let constructor = "mlir::pto::createVPTOCombineReductionsPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VPTOMaskSimplify
    : Pass<"vpto-mask-simplify", "ModuleOp"> {
  let summary = "Simplify provably redundant VPTO mask transformations";
  let description = [{
    Replaces the two results of `pto.pintlv_b*` and `pto.pdintlv_b*` with
    their inputs when both inputs are masks produced with the `PAT_ALL`
    pattern. Interleaving or deinterleaving an all-true predicate cannot change
    its value, so this removes redundant predicate values before backend
    emission. Non-uniform masks are intentionally left unchanged.
  }];
  let constructor = "mlir::pto::createVPTOMaskSimplifyPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

#endif // MLIR_DIALECT_PTO_PASSES
