//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

def PTOConvertToDPS : Pass<"pto-convert-to-dps", "func::FuncOp"> {
  let summary = "Convert PTO value-semantic ops to DPS ops";
  let description = [{
    Rewrites High-Level PTO ops (load/store/matmul) to their DPS variants 
    by allocating empty tensors as destinations.
  }];
 
  let constructor = "mlir::pto::createPTOConvertToDPSPass()";
 
  let dependentDialects = [
    "tensor::TensorDialect",
    "pto::PTODialect"
  ];
}

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

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}


def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
           "pto::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
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

def PTOLowerTPop : Pass<"pto-lower-tpop", "func::FuncOp"> {
  let summary = "Lower slot-based tpop/get_fifo_tile/tfree to internal pipe consumer ops";
  let description = [{
    Rewrites frontend pipe-consumer IR into internal ops:
    - `pto.tpop + pto.get_fifo_tile` -> `pto.declare_tile + pto.tpop_internal`
    - `pto.tfree` -> `pto.tfree_internal`
    Before lowering, verifies that each `pto.tpop` result has exactly one
    matching `pto.get_fifo_tile` and exactly one explicit `pto.tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerTPopPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify lowered pto.tfree_internal placement for explicit tpop/tfree pairs";
  let description = [{
    For each lowered `pto.tpop_internal` in section.cube / section.vector, this
    pass verifies that there is a matching explicit
    `pto.tfree_internal(pipe_handle)` later in the same block, that the
    borrowed tile is not used after that free, and that the same pipe does not
    accumulate multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
