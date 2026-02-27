//===- PTOAttrs.td - PTO dialect attributes definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for PTO dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOATTRS
#define MLIR_DIALECT_PTO_IR_PTOATTRS

include "PTO/IR/PTODialect.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// PTO Enums and Attributes.
//===----------------------------------------------------------------------===//

class PTO_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::pto";
}

class PTO_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<PTO_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Address Space
//===----------------------------------------------------------------------===//

def PTO_AddressSpace_Default : I32EnumAttrCase<"Zero", 0, "zero">;
def PTO_AddressSpace_GM : I32EnumAttrCase<"GM", 1, "gm">;
def PTO_AddressSpace_MAT : I32EnumAttrCase<"MAT", 2, "mat">;
def PTO_AddressSpace_LEFT : I32EnumAttrCase<"LEFT", 3, "left">;
def PTO_AddressSpace_RIGHT : I32EnumAttrCase<"RIGHT", 4, "right">;
def PTO_AddressSpace_ACC : I32EnumAttrCase<"ACC", 5, "acc">;
def PTO_AddressSpace_VEC : I32EnumAttrCase<"VEC", 6, "vec">;
def PTO_AddressSpace_BIAS : I32EnumAttrCase<"BIAS", 7, "bias">;
def PTO_AddressSpace_SCALING : I32EnumAttrCase<"SCALING", 8, "scaling">;

def PTO_AddressSpaceEnum : PTO_I32Enum<
  "AddressSpace", "PTO Address Space", [
    PTO_AddressSpace_Default,
    PTO_AddressSpace_GM,
    PTO_AddressSpace_MAT,
    PTO_AddressSpace_LEFT,
    PTO_AddressSpace_RIGHT,
    PTO_AddressSpace_ACC,
    PTO_AddressSpace_VEC,
    PTO_AddressSpace_BIAS,
    PTO_AddressSpace_SCALING
  ]>;

def PTO_AddressSpaceAttr : PTO_Attr<"AddressSpace", "address_space"> {
  let parameters = (ins EnumParameter<PTO_AddressSpaceEnum>:$address_space);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO address space mapping attribute. Maps to GM, L1, L0A, L0B, L0C and UB.
  }];
}

//===----------------------------------------------------------------------===//
// Pipe
//===----------------------------------------------------------------------===//

def PTO_PIPE_S : I32EnumAttrCase<"PIPE_S", 0>;
def PTO_PIPE_V : I32EnumAttrCase<"PIPE_V", 1>;
def PTO_PIPE_M : I32EnumAttrCase<"PIPE_M", 2>;
def PTO_PIPE_MTE1 : I32EnumAttrCase<"PIPE_MTE1", 3>;
def PTO_PIPE_MTE2 : I32EnumAttrCase<"PIPE_MTE2", 4>;
def PTO_PIPE_MTE3 : I32EnumAttrCase<"PIPE_MTE3", 5>;
def PTO_PIPE_ALL : I32EnumAttrCase<"PIPE_ALL", 6>;
def PTO_PIPE_MTE4 : I32EnumAttrCase<"PIPE_MTE4", 7>;
def PTO_PIPE_MTE5 : I32EnumAttrCase<"PIPE_MTE5", 8>;
def PTO_PIPE_V2 : I32EnumAttrCase<"PIPE_V2", 9>;
def PTO_PIPE_FIX : I32EnumAttrCase<"PIPE_FIX", 10>;
def PTO_VIRTUAL_PIPE_MTE2_L1A : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1A", 11>;
def PTO_VIRTUAL_PIPE_MTE2_L1B : I32EnumAttrCase<"VIRTUAL_PIPE_MTE2_L1B", 12>;
def PTO_PIPE_NUM : I32EnumAttrCase<"PIPE_NUM", 13>;
def PTO_PIPE_UNASSIGNED : I32EnumAttrCase<"PIPE_UNASSIGNED", 99>;

def PTO_PipeEnum : PTO_I32Enum<
  "PIPE", "PTO Op Pipe", [
    PTO_PIPE_S,
    PTO_PIPE_V,
    PTO_PIPE_M,
    PTO_PIPE_MTE1,
    PTO_PIPE_MTE2,
    PTO_PIPE_MTE3,
    PTO_PIPE_ALL,
    PTO_PIPE_MTE4,
    PTO_PIPE_MTE5,
    PTO_PIPE_V2,
    PTO_PIPE_FIX,
    PTO_VIRTUAL_PIPE_MTE2_L1A,
    PTO_VIRTUAL_PIPE_MTE2_L1B,
    PTO_PIPE_NUM,
    PTO_PIPE_UNASSIGNED
  ]>;

def PTO_PipeAttr : PTO_Attr<"Pipe", "pipe"> {
  let parameters = (ins EnumParameter<PTO_PipeEnum>:$pipe);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO Op pipe attribute.
  }];
}

//===----------------------------------------------------------------------===//
// Sync Op Type (High Level Abstraction)
//===----------------------------------------------------------------------===//

def PTO_SYNC_TLOAD        : I32EnumAttrCase<"TLOAD", 0>;
def PTO_SYNC_TSTORE_ACC   : I32EnumAttrCase<"TSTORE_ACC", 1>;
def PTO_SYNC_TSTORE_VEC   : I32EnumAttrCase<"TSTORE_VEC", 2>;
def PTO_SYNC_TMOV_M2L     : I32EnumAttrCase<"TMOV_M2L", 3>;
def PTO_SYNC_TMOV_M2S     : I32EnumAttrCase<"TMOV_M2S", 4>;
def PTO_SYNC_TMOV_M2B     : I32EnumAttrCase<"TMOV_M2B", 5>;
def PTO_SYNC_TMOV_M2V     : I32EnumAttrCase<"TMOV_M2V", 6>;
def PTO_SYNC_TMOV_V2M     : I32EnumAttrCase<"TMOV_V2M", 7>;
def PTO_SYNC_TMATMUL      : I32EnumAttrCase<"TMATMUL", 8>;
def PTO_SYNC_TVEC         : I32EnumAttrCase<"TVEC", 9>;
def PTO_SYNC_TVECWAIT     : I32EnumAttrCase<"TVECWAIT_EVENT", 10>;

def PTO_SyncOpTypeEnum : PTO_I32Enum<
  "SyncOpType", "PTO High Level Sync Op Type", [
    PTO_SYNC_TLOAD,
    PTO_SYNC_TSTORE_ACC,
    PTO_SYNC_TSTORE_VEC,
    PTO_SYNC_TMOV_M2L,
    PTO_SYNC_TMOV_M2S,
    PTO_SYNC_TMOV_M2B,
    PTO_SYNC_TMOV_M2V,
    PTO_SYNC_TMOV_V2M,
    PTO_SYNC_TMATMUL,
    PTO_SYNC_TVEC,
    PTO_SYNC_TVECWAIT
  ]>;

// Canonical sync endpoint type attribute used by record_event/wait_event.
def PTO_PipeEventTypeAttr : PTO_Attr<"PipeEventType", "pipe_event_type"> {
  let parameters = (ins EnumParameter<PTO_SyncOpTypeEnum>:$op_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    High-level synchronization endpoint type. Lowering pass maps it to hardware pipes.
  }];
}

// Legacy alias kept for compatibility with older code.
def PTO_SyncOpTypeAttr : PTO_Attr<"SyncOpType", "sync_op_type"> {
  let parameters = (ins EnumParameter<PTO_SyncOpTypeEnum>:$op_type);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Deprecated alias of pipe_event_type. Prefer #pto.pipe_event_type<...>.
  }];
}

//===----------------------------------------------------------------------===//
// Layout (GlobalTensor layout inference)
//===----------------------------------------------------------------------===//

def PTO_Layout_ND : I32EnumAttrCase<"ND", 0, "nd">;
def PTO_Layout_DN : I32EnumAttrCase<"DN", 1, "dn">;
def PTO_Layout_NZ : I32EnumAttrCase<"NZ", 2, "nz">;

def PTO_LayoutEnum : PTO_I32Enum<
  "Layout", "Global tensor layout (row/col/fractal)", [
    PTO_Layout_ND,
    PTO_Layout_DN,
    PTO_Layout_NZ
  ]>;

def PTO_LayoutAttr : PTO_Attr<"Layout", "layout"> {
  let parameters = (ins EnumParameter<PTO_LayoutEnum>:$layout);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    Layout inferred from shape/stride for GlobalTensor:
      ND (row-major), DN (col-major), NZ (fractal).
  }];
}

//===----------------------------------------------------------------------===//
// Function and Module Core Type
//===----------------------------------------------------------------------===//

def PTO_AIC : I32EnumAttrCase<"AIC", 1>;
def PTO_AIV : I32EnumAttrCase<"AIV", 2>;
def PTO_MIX : I32EnumAttrCase<"MIX", 3>;
def PTO_AIC_OR_AIV : I32EnumAttrCase<"AIC_OR_AIV", 4>;

def PTO_TFuncCoreTypeEnum : PTO_I32Enum<
  "TFuncCoreType", "PTO Function Core Type", [
    PTO_AIC,
    PTO_AIV,
    PTO_MIX,
    PTO_AIC_OR_AIV
  ]>;

def PTO_TFuncCoreTypeAttr : PTO_Attr<"TFuncCoreType", "func_core_type"> {
  let parameters = (ins EnumParameter<PTO_TFuncCoreTypeEnum>:$funcCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO function core type attribute.
  }];
}

def PTO_TModuleCoreTypeEnum : PTO_I32Enum<
  "TModuleCoreType", "PTO Module Core Type", [
    PTO_AIC,
    PTO_AIV,
    PTO_MIX
  ]>;

def PTO_TModuleCoreTypeAttr : PTO_Attr<"TModuleCoreType", "module_core_type"> {
  let parameters = (ins EnumParameter<PTO_TModuleCoreTypeEnum>:$moduleCoreType);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO module core type attribute.

    If all of the functions within the module has `AIV` func core type , the
    module core type is `AIV`.

    If all of the functions within the module has `AIC` func core type , the
    module core type is `AIC`.

    Otherwise, the module core type is `MIX`.
  }];
}

//===----------------------------------------------------------------------===//
// Event ID
//===----------------------------------------------------------------------===//

def PTO_EVENT_ID0 : I32EnumAttrCase<"EVENT_ID0", 0>;
def PTO_EVENT_ID1 : I32EnumAttrCase<"EVENT_ID1", 1>;
def PTO_EVENT_ID2 : I32EnumAttrCase<"EVENT_ID2", 2>;
def PTO_EVENT_ID3 : I32EnumAttrCase<"EVENT_ID3", 3>;
def PTO_EVENT_ID4 : I32EnumAttrCase<"EVENT_ID4", 4>;
def PTO_EVENT_ID5 : I32EnumAttrCase<"EVENT_ID5", 5>;
def PTO_EVENT_ID6 : I32EnumAttrCase<"EVENT_ID6", 6>;
def PTO_EVENT_ID7 : I32EnumAttrCase<"EVENT_ID7", 7>;

def PTO_EventEnum : PTO_I32Enum<
  "EVENT", "Event ID for Synchronization", [
    PTO_EVENT_ID0,
    PTO_EVENT_ID1,
    PTO_EVENT_ID2,
    PTO_EVENT_ID3,
    PTO_EVENT_ID4,
    PTO_EVENT_ID5,
    PTO_EVENT_ID6,
    PTO_EVENT_ID7
  ]>;

def PTO_EventAttr : PTO_Attr<"Event", "event"> {
  let parameters = (ins EnumParameter<PTO_EventEnum>:$event);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO event attribute for synchronization.
  }];
}

//===----------------------------------------------------------------------===//
// Plan Memory Mode
//===----------------------------------------------------------------------===//

def PTO_LOCALMEMPLAN : I32EnumAttrCase<"LOCAL_MEM_PLAN", 0>;
def PTO_GLOBALWORKSPACEPLAN : I32EnumAttrCase<"GLOBAL_WORKSPACE_PLAN", 1>;

def PTO_MemPlanModeEnum : PTO_I32Enum<
  "MemPlanMode", "Mem Plan Mode", [
    PTO_LOCALMEMPLAN,
    PTO_GLOBALWORKSPACEPLAN
]>;

//===----------------------------------------------------------------------===//
// Pad Mode for LoadOp
//===----------------------------------------------------------------------===//

// No padding
def PTO_PadNull : I32EnumAttrCase<"PadNull", 0>;
// Pad using the first element
def PTO_PadFirstElem : I32EnumAttrCase<"PadFirstElem", 1>;
// Pad using specified pad value
def PTO_PadValue : I32EnumAttrCase<"PadValue", 2>;

def PTO_PadModeEnum : PTO_I32Enum<
  "PadMode", "Pad Mode for LoadOp", [
    PTO_PadNull,
    PTO_PadFirstElem,
    PTO_PadValue
  ]>;

def PTO_PadModeAttr : PTO_Attr<"PadMode", "padmode"> {
  let parameters = (ins EnumParameter<PTO_PadModeEnum>:$padmode);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    PTO pad mode attribute.
  }];
}

def PTO_VFAttr : PTO_Attr<"VectorFunction", "vector_function"> {
  let description = [{PTO vector function attribute}];
}
//===----------------------------------------------------------------------===//
// CmpMode
//===----------------------------------------------------------------------===//

def PTO_CmpModeEnum : PTO_I32Enum<
  "CmpMode", "PTO compare mode", [
    I32EnumAttrCase<"EQ", 0, "eq">,
    I32EnumAttrCase<"NE", 1, "ne">,
    I32EnumAttrCase<"LT", 2, "lt">,
    I32EnumAttrCase<"LE", 3, "le">,
    I32EnumAttrCase<"GT", 4, "gt">,
    I32EnumAttrCase<"GE", 5, "ge">
  ]>;

def PTO_CmpModeAttr : EnumAttr<PTO_Dialect, PTO_CmpModeEnum, "cmp"> {
  let summary = "cmp mode attribute";
}
//===----------------------------------------------------------------------===//
// RoundMode
//===----------------------------------------------------------------------===//

def PTO_RoundModeEnum : PTO_I32Enum<
  "RoundMode", "PTO rounding mode", [
    I32EnumAttrCase<"NONE",      0>,
    I32EnumAttrCase<"RINT",      1>,
    I32EnumAttrCase<"ROUND",     2>,
    I32EnumAttrCase<"FLOOR",     3>,
    I32EnumAttrCase<"CEIL",      4>,
    I32EnumAttrCase<"TRUNC",     5>,
    I32EnumAttrCase<"ODD",       6>,
    I32EnumAttrCase<"CAST_RINT", 7>
  ]>;

def PTO_RoundModeAttr : EnumAttr<PTO_Dialect, PTO_RoundModeEnum, "round_mode"> {
  let summary = "rounding mode attribute";
}
//===----------------------------------------------------------------------===//
// MaskPattern
//===----------------------------------------------------------------------===//

def PTO_MaskPatternEnum : PTO_I32Enum<
  "MaskPattern", "PTO mask pattern", [
    I32EnumAttrCase<"P0101", 0>,
    I32EnumAttrCase<"P0011", 1>,
    I32EnumAttrCase<"P0110", 2>,
    I32EnumAttrCase<"P0001", 3>,
    I32EnumAttrCase<"P1111", 4>,
    I32EnumAttrCase<"P1010", 5>
  ]
>;

def PTO_MaskPatternAttr : PTO_Attr<"MaskPattern", "mask_pattern"> {
  let summary = "mask pattern attribute";
  let parameters = (ins EnumParameter<PTO_MaskPatternEnum>:$value);
  let assemblyFormat = "`<` params `>`";
}

// ---------- BLayout / SLayout / PadValue (same style as AddressSpace via PTO_I32Enum) ----------
def PTO_BLayout_Enum : PTO_I32Enum<"BLayout", "Base layout", [
  I32EnumAttrCase<"RowMajor", 0, "row_major">,
  I32EnumAttrCase<"ColMajor", 1, "col_major">
  ]>;

def PTO_BLayoutAttr : PTO_Attr<"BLayout", "blayout"> {
  let parameters = (ins EnumParameter<PTO_BLayout_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

def PTO_SLayout_Enum : PTO_I32Enum<"SLayout", "Secondary layout", [
  I32EnumAttrCase<"NoneBox",  0, "none_box">,
  I32EnumAttrCase<"RowMajor", 1, "row_major">,
  I32EnumAttrCase<"ColMajor", 2, "col_major">
  ]>;

def PTO_SLayoutAttr : PTO_Attr<"SLayout", "slayout"> {
  let parameters = (ins EnumParameter<PTO_SLayout_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

def PTO_PadValue_Enum : PTO_I32Enum<"PadValue", "Pad value policy", [
  I32EnumAttrCase<"Null", 0, "null">,
  I32EnumAttrCase<"Zero", 1, "zero">,
  I32EnumAttrCase<"Max",  2, "max">,
  I32EnumAttrCase<"Min",  3, "min">
  ]>;

def PTO_PadValueAttr : PTO_Attr<"PadValue", "pad_value"> {
  let parameters = (ins EnumParameter<PTO_PadValue_Enum>:$value);
  let assemblyFormat = "`<` params `>`";
}

// ---------- tile_buf_config (NO b_fractal / s_fractal) ----------
def TileBufConfigAttr : AttrDef<PTO_Dialect, "TileBufConfig"> {
  let mnemonic = "tile_buf_config";
  let parameters = (ins
    "BLayoutAttr":$bLayout,
    "SLayoutAttr":$sLayout,
    "mlir::IntegerAttr":$sFractalSize, // i32
    "PadValueAttr":$pad
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilder<(ins
      "BLayoutAttr":$bLayout,
      "SLayoutAttr":$sLayout,
      "mlir::IntegerAttr":$sFractalSize,
      "PadValueAttr":$pad
    )>
  ];

  let extraClassDeclaration = [{
    static TileBufConfigAttr getDefault(MLIRContext *ctx);
    bool isDefault() const;

    static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                                mlir::Attribute bLayout,
                                mlir::Attribute sLayout,
                                mlir::IntegerAttr sFractalSize,
                                mlir::Attribute pad);
  }];
}

#endif // MLIR_DIALECT_PTO_IR_PTOATTRS
