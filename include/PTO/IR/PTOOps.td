// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOOps.td - Pattern descriptor operations -----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// This file declares the PTO dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_IR_PTOOPS
#define MLIR_DIALECT_PTO_IR_PTOOPS

include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"
include "PTO/IR/PTOTypeDefs.td"
include "PTO/IR/PTOInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

def PTODpsType :
  AnyTypeOf<[AnyRankedTensor, PartitionTensorViewType, TileBufType]>;

def PTOPipeEntryType :
  AnyTypeOf<[AnyRankedTensor, TensorViewType, TileBufType],
            "TensorView, TileBuf, or Tensor">;

def PTOCommType :
  AnyTypeOf<[AnyRankedTensor, TensorViewType, PartitionTensorViewType,
             TileBufType], "TensorView, PartitionTensorView, TileBuf, or Tensor">;

def PtrOrMemRef :
  AnyTypeOf<[PtrType, AnyMemRef], "Ptr or MemRef">;

def TensorViewOrPartitionView :
  AnyTypeOf<[TensorViewType, PartitionTensorViewType],
            "TensorView or PartitionTensorView">;

def ScalarType :
  AnyTypeOf<[AnySignlessInteger, AnyFloat], "numeric (integer/float)">;

def PrintScalarType :
  AnyTypeOf<[AnyInteger, AnyFloat], "integer/float printable scalar">;

//===----------------------------------------------------------------------===//
// Op Class
//===----------------------------------------------------------------------===//
class PTO_TOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic,
         !listconcat(traits, [TileOpInterface])>;

class PTO_DpsOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, traits # [DestinationStyleOpInterface]> {
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

class PTO_Op<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

include "PTO/IR/VMIOps.td"
include "PTO/IR/VPTOOps.td"
include "PTO/IR/VPTOUbOps.td"

//===----------------------------------------------------------------------===//
// Pointer/View Ops (for your front-end IR)
//===----------------------------------------------------------------------===//

def AddPtrOp : PTO_Op<"addptr", [
    Pure,
    AllTypesMatch<["ptr", "result"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Add an element offset to a !pto.ptr";
  let description = [{
    Computes a new pointer by adding an element offset to the base pointer.
    The offset is in elements (not bytes).
  }];

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset
  );

  let results = (outs PtrType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `,` $offset attr-dict `:` type($ptr) `->` type($result)
  }];
}

def PtrToIntOp : PTO_Op<"ptrtoint", [Pure]> {
  let summary = "Convert a global pointer to a byte address";
  let description = [{
    Converts a global pointer value to an i64 byte address. Pointer arithmetic
    such as `pto.addptr` remains explicit until backend lowering.
  }];

  let arguments = (ins
    PtrType:$ptr
  );

  let results = (outs
    I64:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr) `->` type($result)
  }];
}

def IntToPtrOp : PTO_Op<"inttoptr", [Pure]> {
  let summary = "Convert an i64 byte address to a global pointer";
  let description = [{
    Converts an i64 byte address to a global pointer of the requested element
    type. This is an escape hatch for explicit byte-address arithmetic and
    cross-element-type pointer reinterpretation. To keep byte-address-derived
    pointers from escaping into general pointer IR, the result may only be used
    directly as the pointer operand of `pto.load_scalar` or `pto.store_scalar`.
  }];

  let arguments = (ins
    AnyInteger:$addr
  );

  let results = (outs
    PtrType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $addr attr-dict `:` type($addr) `->` type($result)
  }];
}

def CastPtrOp : PTO_Op<"castptr", [Pure]> {
  let summary = "Cast between integer and !pto.ptr, or between !pto.ptr types";
  let description = [{
    Performs an explicit pointer-domain cast.

    Supported cases:
    - integer -> !pto.ptr<elem, space>
    - !pto.ptr<elem, space> -> integer
    - !pto.ptr<srcElem, space> -> !pto.ptr<dstElem, space>
    - memref<..., space> -> !pto.ptr<elem, space> (extract the aligned base ptr)

    Pointer-to-pointer casts must stay within the same PTO memory space. Cross
    space casts such as gm <-> ub are rejected by the verifier.
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Scalar pointer load/store
//===----------------------------------------------------------------------===//

def LoadScalarOp : PTO_Op<"load_scalar", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a single scalar element from a pointer at offset.";

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def StoreScalarOp : PTO_Op<"store_scalar", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a single scalar element to a pointer at offset.";

  let arguments = (ins
    PtrType:$ptr,
    Index:$offset,
    AnyType:$value
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `,` $ptr `[` $offset `]` attr-dict `:` type($ptr) `,` type($value)
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def MakeTensorViewOp : PTO_Op<"make_tensor_view", [AttrSizedOperandSegments,
                                                    Pure]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (no allocation, no copy).";

  let arguments = (ins
    PtrType:$ptr,
    Variadic<Index>:$shape,
    Variadic<Index>:$strides,
    OptionalAttr<PTO_LayoutAttr>:$layout
  );

  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// =============================================================================
// PartitionViewOp
// =============================================================================
def PartitionViewOp : PTO_Op<"partition_view", [AttrSizedOperandSegments,
                                                Pure]> {
  let summary = "Partition a tensor view into a smaller logical view (logical slicing).";
  let description = [{
    Captures a specific calculation region from a large view.
    It carries offsets (where to read) and sizes (how much to read).
    
    IR Example:
      %1 = pto.partition_view %0, offsets=[...], sizes=[...] 
           : !pto.tensor_view -> !pto.partition_tensor_view
  }];

  let arguments = (ins
    TensorViewOrPartitionView:$source,
    Variadic<Index>:$offsets,  // 动态 offsets
    Variadic<Index>:$sizes     // 动态 sizes
  );

  let results = (outs PartitionTensorViewType:$result); // 输出: 逻辑切片

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// Helper for either whole-tensor or partitioned GM views.
def TensorViewLike :
  AnyTypeOf<[TensorViewType, PartitionTensorViewType],
            "TensorView or PartitionTensorView">;

// Get the size of a dimension of a tensor_view.
// Result type: Index (use arith.index_cast if i32 is needed).
def GetTensorViewDimOp : PTO_Op<"get_tensor_view_dim", [Pure]> {
  let summary = "Get the size of a dimension of a tensor_view.";
  let description = [{
    Returns the size of the given dimension of a logical tensor view.
    This op accepts !pto.tensor_view or !pto.partition_tensor_view.
    IR:
      %dim_size = pto.get_tensor_view_dim %tv, %dim_index
        : !pto.tensor_view, index -> index
  }];
  let arguments = (ins
    TensorViewLike:$tensor_view,
    Index:$dim_index
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tensor_view `,` $dim_index `:` qualified(type($tensor_view)) `->` qualified(type($result))
    attr-dict
  }];
}

// Get the logical stride of a tensor_view dimension in elements.
// Result type: Index (use arith.index_cast if i32 is needed).
def GetTensorViewStrideOp : PTO_Op<"get_tensor_view_stride", [Pure]> {
  let summary = "Get the stride of a dimension of a tensor_view.";
  let description = [{
    Returns the stride, measured in elements, of the given dimension of a
    logical tensor view. This op accepts !pto.tensor_view or
    !pto.partition_tensor_view.
  }];
  let arguments = (ins
    TensorViewLike:$tensor_view,
    Index:$dim_index
  );
  let results = (outs Index:$result);
  let assemblyFormat = [{
    $tensor_view `,` $dim_index `:` qualified(type($tensor_view)) `->` qualified(type($result))
    attr-dict
  }];
}

def AllocTileOp : PTO_Op<"alloc_tile", [AttrSizedOperandSegments]> {
  let summary = "Allocates a tile buffer (logical buffer).";

  let arguments = (ins
    Optional<I64>:$addr,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    (`addr` `=` $addr^)?
    (`valid_row` `=` $valid_row^)?
    (`valid_col` `=` $valid_col^)?
     attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

//===----------------------------------------------------------------------===//
// Multi-buffer tile allocation and slot selection
//===----------------------------------------------------------------------===//
//
// `pto.alloc_multi_tile` declares an N-slot logical tile buffer. The actual
// physical allocation of N slots is performed directly by PTOPlanMemory.
// The op remains tile-native until PTOResolveBufferSelect materializes an
// addressed `pto.alloc_tile` for each selected slot.
//
// `pto.multi_tile_get` is the *only* way to consume a `multi_tile_buf`: it
// returns a regular `tile_buf` view onto the chosen slot. The slot index
// may be a constant or any SSA `index` value; PTOAS does not synthesize
// `iv mod N` automatically -- the user expression IS the slot selector.
//
def AllocMultiTileOp : PTO_Op<"alloc_multi_tile", [AttrSizedOperandSegments]> {
  let summary = "Allocate an N-slot multi-buffer tile";
  let description = [{
    Produces a `!pto.multi_tile_buf<S, count=N>` value owning N physical
    slots of the inner tile shape `S`.

    In the default pipeline (`--pto-level=level1|level2`) no `addr` operand
    is given and the physical addresses of the N slots are decided by
    PTOPlanMemory.

    Under `--pto-level=level3` the local memory is owned by the upper-level
    caller and PTOPlanMemory does not run, so an explicit base address
    operand `$addr` is required (mirroring `pto.alloc_tile`). The N slots
    are laid out contiguously as the region `[addr, addr + N * slotBytes)`
    with slot `k` at `addr + k * slotBytes`, where `slotBytes` is the byte
    footprint of one slot tile (`product(shape) * element_size`). The caller
    is responsible for having reserved `N * slotBytes` bytes at `addr`.

    `valid_row` / `valid_col` operands follow the same semantics as
    `pto.alloc_tile`: required when the inner tile's valid shape is `?`,
    absent when both valid dims are static.
  }];

  let arguments = (ins
    Optional<I64>:$addr,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs MultiTileBufType:$result);

  let assemblyFormat = [{
    (`addr` `=` $addr^)?
    (`valid_row` `=` $valid_row^)?
    (`valid_col` `=` $valid_col^)?
    attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

def MultiTileGetOp : PTO_Op<"multi_tile_get", [
    Pure,
    ViewLikeOpInterface
  ]> {
  let summary = "Pick one physical slot of a multi_tile_buf";
  let description = [{
    Returns a single-slot view of a `multi_tile_buf`. The result `tile_buf`
    must equal the source's per-slot type; the only thing this op selects
    is which of the N physical slots is referred to.

    The slot index is an `index` value in `[0, count)`. If it is a constant,
    the verifier checks the range. If it is a runtime SSA value, downstream
    sync analysis treats this use as a slot-indexed dynamic access; ptoas
    does NOT rewrite the user expression into `iv mod N`.

    This op is metadata-only (no data movement). PlanMemory preserves it,
    sync analysis consumes its slot index directly, and
    PTOResolveBufferSelect replaces it with an addressed `pto.alloc_tile`.
  }];

  let arguments = (ins
    MultiTileBufType:$source,
    Index:$slot
  );

  let results = (outs TileBufType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `[` $slot `]` attr-dict
    `:` qualified(type($source)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}


def SubViewOp : PTO_Op<"subview", [
    Pure,
    ViewLikeOpInterface,
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<InferTypeOpInterface> // 启用 C++ 推导
  ]> {
  
  let summary = "Create a tile subview from a parent tile (logical size + valid dims).";
  let description = [{
    Creates a view into the source tile.
    - Logical subview window: defined by `offsets` + static `sizes`.
    - Result tile type shape: equals subview `sizes` (logical shape).
    - Effective subview extent: represented by inferred `valid_row/valid_col`
      (i.e. `valid_shape`), clipped by subview `sizes`.
    - Lowering keeps parent physical stride/base semantics for non-compact
      access, while IR type exposes logical subview size for readability.
  }];

  let arguments = (ins
    TileBufType:$source,
    Variadic<Index>:$offsets, // 运行时动态偏移 [i, j]
    I64ArrayAttr:$sizes,      // 静态形状 [32, 32]
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs TileBufType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  // [新增] 显式实现 ViewLikeOpInterface 缺失的方法
  let extraClassDeclaration = [{
    // 接口要求 getViewSource，我们转发给自动生成的 getSource
    ::mlir::Value getViewSource() { return getSource(); }

    // ViewLikeOpInterface 可能还需要 getOffsets (如果 Variadic 不自动匹配)
    // 但通常 Variadic<Index>:$offsets 会生成 getOffsets()，这应该没问题。
    // 如果后续报 getOffsets 错，也可以在这里加。

    // Override the InferTypeOpInterface compatibility check so a declared empty
    // valid region (valid_row/valid_col = 0) is accepted in place of the size-
    // inferred valid extent on the operand-omitted path (PyPTO dual-AIV no-op
    // replay). inferReturnTypes' default impl dispatches to this static method.
    static bool isCompatibleReturnTypes(::mlir::TypeRange l, ::mlir::TypeRange r);
  }];
}

def SetValidShapeOp : PTO_Op<"set_validshape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Update valid row/col metadata in-place on a dynamic tile";
  let description = [{
    Updates the runtime valid_row/valid_col metadata on an existing dynamic
    tile_buf handle in place.

    This op does not allocate or alias a new tile handle. It is only valid for
    tile_buf types whose valid_row and valid_col are both dynamic (`?`), and
    the source must be a locally bound tile handle (for example `alloc_tile`
    or a view derived from it). Function arguments/results are not supported.
  }];

  let arguments = (ins
    TileBufType:$source,
    Index:$valid_row,
    Index:$valid_col
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $source `,` $valid_row `,` $valid_col attr-dict `:`
    qualified(type($source))
  }];
}

def GetValidShapeOp : PTO_Op<"get_validshape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Read runtime valid row/col metadata from a tile";
  let description = [{
    Reads the current runtime valid_row/valid_col metadata from a tile_buf
    handle. This op observes prior set_validshape updates and returns the
    current values as index-typed SSA values.
  }];

  let arguments = (ins
    TileBufType:$source
  );

  let results = (outs
    Index:$valid_row,
    Index:$valid_col
  );

  let hasVerifier = 1;

  let assemblyFormat = "$source attr-dict `:` qualified(type($source))";
}

// ============================================================================
// SSA TileBuf Config Ops (aliasing views)
// ============================================================================

def BitcastOp : PTO_Op<"bitcast", [
    Pure,
    ViewLikeOpInterface
  ]> {
  let summary = "SSA dtype reinterpretation of a tile buffer view (aliases src storage)";
  let description = [{
    Returns a tile buffer view with a different element type (dtype) while
    reusing the same underlying storage as the source.

    This op is a metadata/config rewrite only (no data movement).
  }];

  let arguments = (ins
    TileBufType:$src
  );

  let results = (outs
    TileBufType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSrc(); }
  }];
}

//===----------------------------------------------------------------------===//
// DMA Ops
//===----------------------------------------------------------------------===//
// -------------------------
// DPS versions in tile world
// -------------------------
def TLoadOp  : PTO_TOp<"tload", [
  PTO_DpsInitOpInterface,
  AttrSizedOperandSegments,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO data load operation (Dimension Collapse: PartitionView -> TileBuf).";
  let description = [{
    Loads data from a high-dimensional logical partition view into a 2D physical tile buffer.
    Constraint: The product of partition view sizes must match the product of tile buffer valid dims.
    DPS form:
      pto.tload ins(%partition_view) outs(%tile_buf)
  }];

  let arguments = (ins 
    PTODpsType:$src,
    PTODpsType:$dst,
    OptionalAttr<PTO_PadModeAttr>:$pad_mode,
    Optional<AnyType>:$pad_value,
    Optional<Index>:$left_padding_num,
    Optional<AnyType>:$right_padding_num,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
    Optional<AnyType>:$init_condition
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let builders = [
    // 1. Basic: (src, dst)
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst), [{
      build($_builder, $_state, res, src, dst, 
            /*pad_mode=*/nullptr, /*pad_value=*/nullptr, 
            /*left=*/nullptr, /*right=*/nullptr, 
            /*init_out=*/nullptr, /*init_cond=*/nullptr); 
    }]>,

    // 2. With left_padding_num
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "Value":$left_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            nullptr, nullptr, left_padding_num, nullptr, nullptr, nullptr);
    }]>,

    // 3. With pad_mode, pad_value
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, nullptr, nullptr, nullptr, nullptr);
    }]>,

    // 4. ... + left
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, nullptr, nullptr, nullptr);
    }]>,

    // 5. ... + left + right
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, right_padding_num, nullptr, nullptr);
    }]>,

    // 6. ... + left + right + bool
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst,
                   "pto::PadModeAttr":$pad_mode, "Value":$pad_value,
                   "Value":$left_padding_num, "Value":$right_padding_num,
                   "bool":$init_out_buffer), [{
      build($_builder, $_state, res, src, dst, 
            pad_mode, pad_value, left_padding_num, right_padding_num, 
            init_out_buffer, nullptr);
    }]>
  ];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`left_padding_num` `=` $left_padding_num^ `:` type($left_padding_num))?
    (`init_out_buffer` `=` $init_out_buffer^ )?
    (`right_padding_num` `=` $right_padding_num^ `:` type($right_padding_num))?
    (`init_condition` `=` $init_condition^ `:` type($init_condition))?
    (`->` qualified(type($result))^)?
  }];
  
  let extraClassDeclaration = [{
    static StringRef getOpName() { return "pto_load"; }
    ShapedType getSrcOperandType() {
       return cast<ShapedType>(getSrc().getType());
    }
    ShapedType getDstOperandType() {
       return cast<ShapedType>(getDst().getType());
    }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_MTE2;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TPrefetchOp : PTO_TOp<"tprefetch", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Prefetch global data into a local tile buffer.";
  let description = [{
    Prefetches data from a global partition view into a temporary local tile
    buffer. Unlike most PTO intrinsics, TPREFETCH does not carry implicit
    TSYNC(wait_events...) semantics in PTO-ISA.

    DPS form:
      pto.tprefetch ins(%src) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def MakePrefetchAsyncContextOp : PTO_Op<"make_prefetch_async_context"> {
  let summary = "Build a PrefetchAsyncContext from GM workspace.";
  let description = [{
    Materializes a first-class PTO IR `PrefetchAsyncContext` value from a GM
    workspace pointer.
  }];

  let arguments = (ins
    PtrType:$workspace
  );

  let results = (outs
    PrefetchAsyncContextType:$ctx
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $workspace `:` qualified(type($workspace)) `)`
    attr-dict `->` qualified(type($ctx))
  }];
}

def GetPrefetchAsyncSessionOp : PTO_Op<"get_prefetch_async_session", [Pure]> {
  let summary = "Project session from a PrefetchAsyncContext.";

  let arguments = (ins
    PrefetchAsyncContextType:$ctx
  );

  let results = (outs
    AsyncSessionType:$session
  );

  let assemblyFormat = [{
    $ctx attr-dict `:` qualified(type($ctx)) `->` qualified(type($session))
  }];
}

def TPrefetchAsyncOp : PTO_Op<"tprefetch_async"> {
  let summary = "Asynchronously prefetch global data into L2 cache via SDMA CMO.";
  let description = [{
    Starts one asynchronous prefetch operation.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PrefetchAsyncContextType:$ctx
  );

  let results = (outs
    AsyncEventType:$event
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $src `,` $ctx `:`
        qualified(type($src)) `,` qualified(type($ctx)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def TStoreOp: PTO_TOp<"tstore", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO data store operation (TileBuf -> PartitionView).";
  let description = [{
    Stores data from a 2D physical tile buffer back to a high-dimensional logical partition view.
    
    DPS form:
      pto.tstore ins(%tile_buf) outs(%partition_view)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    DefaultValuedAttr<PTO_STPhaseAttr, "::mlir::pto::STPhase::Unspecified">:$stPhase,
    DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
           (`fp` $fp^ `:` qualified(type($fp)))?
           (`,` $preQuantScalar^ `:` type($preQuantScalar))? `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // pto-isa lowers TSTORE differently depending on the source tile domain:
      //   - ACC (L0C) -> GM uses the FIX pipe (copy_matrix_cc_to_gm).
      //   - VEC/UB and MAT -> GM use the MTE3 pipe.
      //
      // Sync insertion must use the correct pipe, otherwise the generated
      // set_flag/wait_flag pairs won't order the actual hardware instructions.
      auto isAcc = [](Type ty) -> bool {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC;
          return false;
        }
        return false;
      };
      if (isAcc(getSrc().getType()))
        return ::mlir::pto::PIPE::PIPE_FIX;
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TTransOp : PTO_TOp<"ttrans", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix trans operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tile buf:  pto.ttrans ins(%src, %tmp) outs(%dst)

    The current PTO IR form always carries a tmp operand, but PTO IR does not
    require tmp to match the shape or valid shape of src/dst.
  }];
  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );
  let results = (outs);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V; // Vector Pipe
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}
def TMatmulBiasOp : PTO_TOp<"tmatmul.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$b,
    PTODpsType:$bias,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  //let builders = [
  //  OpBuilder<(ins "TypeRange":$resultTypes, "Value":$a, "Value":$b, "Value":$bias, "Value":$dst), [{
  //    build($_builder, $_state, resultTypes, a, b, bias, dst, ValueRange{});
  //  }]>
  //];

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx ins(%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxOp : PTO_TOp<"tmatmul.mx", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Matmul mx (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.acc ins(%c_in,%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxAccOp : PTO_TOp<"tmatmul.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx accumulate (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.tmatmul.mx.bias ins(%a,%a_scale,%b,%b_scale,%bias) outs(%dst)  -> tensor?
def TMatmulMxBiasOp : PTO_TOp<"tmatmul.mx.bias",[
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx with bias (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;


  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulOp : PTO_TOp<"tmatmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "matmul_dps"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMatmulAccOp : PTO_TOp<"tmatmul.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M; 
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvOp : PTO_TOp<"tgemv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "gemv"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvAccOp : PTO_TOp<"tgemv.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
      ::mlir::pto::PIPE getPipe() {
        return ::mlir::pto::PIPE::PIPE_M; 
      }
      ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvBiasOp : PTO_TOp<"tgemv.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "GEMV with bias add (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType :$a,
    PTODpsType :$b,
    PTODpsType :$bias,
    PTODpsType :$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_BIAS"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxOp : PTO_TOp<"tgemv.mx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxAccOp : PTO_TOp<"tgemv.mx.acc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV accumulate with scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$c_in,
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_AccPhaseAttr, "::mlir::pto::AccPhase::Unspecified">:$accPhase
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $c_in `,` $a `,` $a_scale `,` $b `,` $b_scale
          `:` type($c_in) `,` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGemvMxBiasOp : PTO_TOp<"tgemv.mx.bias", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Mixed-precision GEMV with bias and scale tiles (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$bias,
    PTODpsType:$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $a_scale `,` $b `,` $b_scale `,` $bias
          `:` type($a) `,` type($a_scale) `,` type($b) `,` type($b_scale) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_MX"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMovOp  : PTO_TOp<"tmov", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";
  
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode,
    OptionalAttr<PTO_MxGroupAxisAttr>:$grpAxis
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src))
    (`,` $fp^ `:` qualified(type($fp)))?
    (`,` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // TMOV spans multiple hardware pipelines depending on the source/dest
      // domains. Most tile-domain moves are executed by MTE1 (e.g. MAT->L0),
      // while UB->UB copies are vector-pipe operations.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      // UB -> UB copy is vector pipe.
      if (s == ::mlir::pto::AddressSpace::VEC && d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // MAT -> L0A/L0B/Bias moves run on MTE1.
      if ((s == ::mlir::pto::AddressSpace::MAT &&
           (d == ::mlir::pto::AddressSpace::LEFT || d == ::mlir::pto::AddressSpace::RIGHT ||
            d == ::mlir::pto::AddressSpace::BIAS))) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      if (s == ::mlir::pto::AddressSpace::MAT &&
          d == ::mlir::pto::AddressSpace::SCALING) {
        // PTOAS uses SCALING for two distinct PTO-ISA destinations. A5 MX
        // scale tiles are ScaleLeft/ScaleRight loads into L0A/L0B and run on
        // MTE1. Regular Scaling tiles are FBUF loads (TMOV_M2S) and run on
        // Distinguish the MX-only f8E8M0 type from regular scaling tiles.
        Type dstElemTy;
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(getDst().getType()))
          dstElemTy = tb.getElementType();
        if (dstElemTy && llvm::isa<::mlir::pto::F8E8M0Type>(dstElemTy))
          return ::mlir::pto::PIPE::PIPE_MTE1;
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      // ACC -> MAT / VEC use FIX in PTO-ISA sync mapping (TMOV_A2M/TMOV_A2V).
      if (s == ::mlir::pto::AddressSpace::ACC &&
          (d == ::mlir::pto::AddressSpace::MAT ||
           d == ::mlir::pto::AddressSpace::VEC)) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      // Fallback: treat as vector pipe (safe default for most intra-domain moves).
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


// =============================================================================
// System/Runtime Query Ops
// =============================================================================

def GetBlockIdxOp : PTO_Op<"get_block_idx", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the current block index (core ID).";
  let description = [{
    Returns the linear index of the current compute unit (CUBE Core) within the task.
    The return value is in the range [0, BlockNum - 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockIdxOp : PTO_Op<"get_subblock_idx", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the current vector core ID.";
  let description = [{
    Returns the ID of the current compute unit (Vector Core).
    The return value is in the range [0, 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetBlockNumOp : PTO_Op<"get_block_num", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the total number of blocks (cores).";
  let description = [{
    Returns the total number of compute units (Blocks) configured for the current task.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockNumOp : PTO_Op<"get_subblock_num", [
    Pure,
    MicroOpInterface,
    SimtOpInterface
  ]> {
  let summary = "Get the number of vector cores.";
  let description = [{
    Returns the total number of vector compute units.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// High-Level Synchronization Ops
//===----------------------------------------------------------------------===//

def PTO_PipeEventTypeLikeAttr : AnyAttrOf<[PTO_PipeEventTypeAttr, PTO_SyncOpTypeAttr]>;

def RecordEventOp : PTO_Op<"record_event"> {
  let summary = "Record an event for synchronization (High Level)";
  let description = [{
    Records an event from a source operation type to a destination operation type.
    Will be lowered to `pto.set_flag` based on op-to-pipe mapping.
  }];
  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$src_op,
    PTO_PipeEventTypeLikeAttr:$dst_op,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_op `,` $dst_op `,` $event_id `]` attr-dict
  }];
}

def WaitEventOp : PTO_Op<"wait_event"> {
  let summary = "Wait for an event (High Level)";
  let description = [{
    Waits for an event from a source operation type to a destination operation type.
    Will be lowered to `pto.wait_flag` based on op-to-pipe mapping.
  }];
  let arguments = (ins
    PTO_PipeEventTypeLikeAttr:$src_op,
    PTO_PipeEventTypeLikeAttr:$dst_op,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_op `,` $dst_op `,` $event_id `]` attr-dict
  }];
}

// High-Level Barrier (single pipe) with op type mapping
def BarrierSyncOp : PTO_Op<"barrier_sync"> {
  let summary = "High-level barrier mapped from SyncOpType to PIPE";
  let description = [{
    A convenience barrier that specifies a SyncOpType instead of PIPE. The lowering
    pass maps the op type to the corresponding hardware pipe and emits `pto.barrier`.
  }];
  let arguments = (ins
    PTO_SyncOpTypeAttr:$op_type
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $op_type `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Section Ops (Macros Containers)
//===----------------------------------------------------------------------===//

class PTO_SectionOp<string mnemonic>
    : PTO_Op<mnemonic, [SingleBlock, NoTerminator]> {
  let summary = "Container for core-specific code guarded by macros";
  let description = [{
    During conversion to EmitC, this op is lowered to:
    emitc.verbatim("#if defined(MACRO)")
    ... inlined body ...
    emitc.verbatim("#endif")
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def SectionCubeOp : PTO_SectionOp<"section.cube">;
def SectionVectorOp : PTO_SectionOp<"section.vector">;

def SectionSimtOp : PTO_Op<"section.simt", [SingleBlock, NoTerminator]> {
  let summary = "Container for inline SIMT code";
  let description = [{
    `pto.section.simt` keeps an inline SIMT body in the enclosing kernel
    function until downstream PTOAS lowering consumes it.  It is a
    source-level carrier for transformations that need to observe the lexical
    relationship between surrounding code and SIMT regions.

    Launch dimensions are explicit static i32 attributes.  The textual launch
    dimension order matches `pto.simt_launch`:
    `<<<dimX, dimY, dimZ>>>`.
  }];
  let arguments = (ins I32Attr:$dimX, I32Attr:$dimY, I32Attr:$dimZ);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `<` `<` `<` $dimX `,` $dimY `,` $dimZ `>` `>` `>` $body attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tile Fusion Region Ops
//===----------------------------------------------------------------------===//

def FusionRegionOp : PTO_Op<"fusion_region", [SingleBlock]> {
  let summary = "Structured region container for one scheduled tile fusion group";
  let description = [{
    `pto.fusion_region` wraps one already-scheduled fusion group inside the
    current function and exposes explicit escaping values as results.
    The body is a lightweight single-block container and may implicitly capture
    surrounding SSA values from the parent scope.
  }];

  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $body attr-dict `:` type(results)
  }];
}

def YieldOp : PTO_Op<"yield", [Terminator, ParentOneOf<["FusionRegionOp"]>]> {
  let summary = "Terminate a pto.fusion_region and return its visible outputs";
  let description = [{
    `pto.yield` is the dedicated terminator for `pto.fusion_region`. Its
    operands enumerate the tile/value set that remains externally visible after
    encapsulation. They define the region outputs in the same order as the
    parent `pto.fusion_region` results, giving downstream passes a compact
    summary of which internal tiles still escape the region boundary.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $values `)` attr-dict `:` functional-type($values, results)
  }];
}

//===----------------------------------------------------------------------===//
// Frontend TPUSH/TPOP Pipe Communication Ops
//===----------------------------------------------------------------------===//

def AicInitializePipeOp : PTO_Op<"aic_initialize_pipe",
                                 [AttrSizedOperandSegments]> {
  let summary = "Frontend Cube-kernel pipe initialization";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      OptionalAttr<I32Attr>:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      Optional<PtrType>:$gm_slot_buffer,
      Optional<TensorViewType>:$gm_slot_tensor,
      Optional<I32>:$c2v_consumer_buf,
      Optional<I32>:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def AivInitializePipeOp : PTO_Op<"aiv_initialize_pipe",
                                 [AttrSizedOperandSegments]> {
  let summary = "Frontend Vector-kernel pipe initialization";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      OptionalAttr<I32Attr>:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      Optional<PtrType>:$gm_slot_buffer,
      Optional<TensorViewType>:$gm_slot_tensor,
      Optional<I32>:$c2v_consumer_buf,
      Optional<I32>:$v2c_consumer_buf
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TAllocToAivOp : PTO_TOp<"talloc_to_aiv"> {
  let summary = "Frontend C2V producer alloc of a GlobalTensor FIFO entry";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs TensorViewType:$entry);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($entry))
  }];
}

def TAllocToAicOp : PTO_TOp<"talloc_to_aic"> {
  let summary = "Frontend V2C producer alloc of a GlobalTensor FIFO entry";

  let arguments = (ins
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs TensorViewType:$entry);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($entry))
  }];
}

def TPushToAivOp : PTO_TOp<"tpush_to_aiv", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Frontend C2V producer push from Cube to Vector";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

def TPushToAicOp : PTO_TOp<"tpush_to_aic"> {
  let summary = "Frontend V2C producer push from Vector to Cube";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      Optional<I64>:$aiv_subblockid,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `:` qualified(type($tile)) `)`
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
  }];
}

def TPopFromAicOp : PTO_TOp<"tpop_from_aic", [AttrSizedOperandSegments]> {
  let summary = "Frontend C2V consumer pop in Vector kernel";

  let arguments = (ins
      Optional<Index>:$valid_row,
      Optional<Index>:$valid_col,
      Optional<I64>:$aiv_subblockid,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs PTOPipeEntryType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $valid_row^ `,` $valid_col `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
    `->` qualified(type($tile))
  }];
}

def TPopFromAivOp : PTO_TOp<"tpop_from_aiv", [AttrSizedOperandSegments]> {
  let summary = "Frontend V2C consumer pop in Cube kernel";

  let arguments = (ins
      Optional<Index>:$valid_row,
      Optional<Index>:$valid_col,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs PTOPipeEntryType:$tile);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $valid_row^ `,` $valid_col `)`)? `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
    `->` qualified(type($tile))
  }];
}

def TFreeFromAicOp : PTO_TOp<"tfree_from_aic"> {
  let summary = "Frontend C2V consumer free in Vector kernel";

  let arguments = (ins
      Optional<PTOPipeEntryType>:$entry,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $entry^ `:` qualified(type($entry)) `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

def TFreeFromAivOp : PTO_TOp<"tfree_from_aiv"> {
  let summary = "Frontend V2C consumer free in Cube kernel";

  let arguments = (ins
      Optional<PTOPipeEntryType>:$entry,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`(` $entry^ `:` qualified(type($entry)) `)`)?
    `{` (`id` `=` $id^ `,`)? `split` `=` $split `}` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Fixpipe Quant Config Ops
//===----------------------------------------------------------------------===//

def SetQuantScalarOp : PTO_Op<"set_quant_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set scalar quantization parameters for fixpipe producer";

  let description = [{
    Producer-side scalar quantization configuration for fixpipe TPUSH.
    Binds an f32 scalar quant payload to a logical pipe id.
    This binding persists until overwritten by another set_quant_scalar with
    the same id. Used by scalar quant modes: deqf16_scalar, req8_scalar,
    qf322b8_pre_scalar, qf322f16_pre_scalar, qf322bf16_pre_scalar,
    qs322bf16_pre_scalar, qf322hif8_pre_scalar, qf322fp8_pre_scalar.
  }];

  let arguments = (ins
      F32:$scale,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scale `:` type($scale) `)` `{` `id` `=` $id `}` attr-dict
  }];
}

def SetQuantVectorOp : PTO_Op<"set_quant_vector", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Set vector quantization parameters for fixpipe producer";

  let description = [{
    Producer-side vector quantization configuration for fixpipe TPUSH.
    Binds a vector quant payload (scaling tile) to a logical pipe id.
    The input tile must have loc=scaling. A5 consumes a direct scaling
    payload with an element type in the f16/bf16/f32 family; A3 consumes
    a packed 64-bit integer payload (i64/ui64). This binding persists
    until overwritten by another set_quant_vector with the same id.
    Used by vector quant modes: deqf16_vec, req8_vec, qf322b8_pre_vec,
    qs322bf16_pre_vec.
  }];

  let arguments = (ins
      PTOPipeEntryType:$scaling_tile,
      DefaultValuedOptionalAttr<I32Attr, "0">:$id
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scaling_tile `:` qualified(type($scaling_tile)) `)`
    `{` `id` `=` $id `}` attr-dict
  }];
}

def ReserveBufferOp : PTO_Op<"reserve_buffer"> {
  let summary = "Reserve a local consumer slot buffer";

  let arguments = (ins
      StrAttr:$name,
      I32Attr:$size,
      PTO_AddressSpaceAttr:$location,
      BoolAttr:$autoAlloc,
      OptionalAttr<I32Attr>:$base
  );

  let results = (outs I32:$addr);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `name` `=` $name `,`
        `size` `=` $size `,`
        `location` `=` $location `,`
        `auto` `=` $autoAlloc
        (`,` `base` `=` $base^)?
        `}` attr-dict `->` type($addr)
  }];
}

def ImportReservedBufferOp : PTO_Op<"import_reserved_buffer"> {
  let summary = "Import a peer reserved local consumer slot buffer";

  let arguments = (ins
      StrAttr:$name,
      FlatSymbolRefAttr:$peer_func
  );

  let results = (outs I32:$addr);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `name` `=` $name `,`
        `peer_func` `=` $peer_func
        `}` attr-dict `->` type($addr)
  }];
}

//===----------------------------------------------------------------------===//
// TPUSH/TPOP Pipe Communication Ops
//===----------------------------------------------------------------------===//

def BuildAsyncSessionOp : PTO_Op<"comm.build_async_session", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Build an async DMA session handle for TPUT_ASYNC/TGET_ASYNC";

  let arguments = (ins
      TileBufType:$scratch,
      PtrType:$workspace,
      OptionalAttr<I32Attr>:$sync_id,
      OptionalAttr<I64Attr>:$block_bytes,
      OptionalAttr<I64Attr>:$comm_block_offset,
      OptionalAttr<I32Attr>:$queue_num,
      OptionalAttr<I64Attr>:$channel_group_idx
  );

  let results = (outs AsyncSessionType:$session);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $scratch `,` $workspace `:` qualified(type($scratch)) `,` qualified(type($workspace)) `)`
    attr-dict `->` qualified(type($session))
  }];
}

def TPutAsyncOp : PTO_Op<"comm.tput_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote write from local GM to remote GM";

  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      AsyncSessionType:$session
  );

  let results = (outs AsyncEventType:$event);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $dst `,` $src `,` $session `:`
        type($dst) `,` type($src) `,` qualified(type($session)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def TGetAsyncOp : PTO_Op<"comm.tget_async", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Asynchronous remote read from remote GM to local GM";

  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      AsyncSessionType:$session
  );

  let results = (outs AsyncEventType:$event);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $dst `,` $src `,` $session `:`
        type($dst) `,` type($src) `,` qualified(type($session)) `)`
    attr-dict `->` qualified(type($event))
  }];
}

def WaitAsyncEventOp : PTO_Op<"comm.wait_async_event", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until an async DMA event completes";

  let arguments = (ins
      AsyncEventType:$event,
      AsyncSessionType:$session
  );

  let results = (outs I1:$completed);

  let assemblyFormat = [{
    `(` $event `,` $session `:`
        qualified(type($event)) `,` qualified(type($session)) `)`
    attr-dict `->` type($completed)
  }];
}

def TestAsyncEventOp : PTO_Op<"comm.test_async_event", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-blocking completion test for an async DMA event";

  let arguments = (ins
      AsyncEventType:$event,
      AsyncSessionType:$session
  );

  let results = (outs I1:$completed);

  let assemblyFormat = [{
    `(` $event `,` $session `:`
        qualified(type($event)) `,` qualified(type($session)) `)`
    attr-dict `->` type($completed)
  }];
}

def TPutOp : PTO_Op<"comm.tput", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote write from local GM to remote GM";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      DefaultValuedAttr<PTO_AtomicTypeAttr, "::mlir::pto::AtomicType::AtomicNone">:$atomicType
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $dst `,` $src `,` `buf` `(` $ping (`,` $pong^)? `)`
    `:` type($dst) `,` type($src) `,` type($ping) (`,` type($pong)^)? `)`
    attr-dict
  }];
}

def TGetOp : PTO_Op<"comm.tget", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Synchronous remote read from remote GM to local GM";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $dst `,` $src `,` `buf` `(` $ping (`,` $pong^)? `)`
    `:` type($dst) `,` type($src) `,` type($ping) (`,` type($pong)^)? `)`
    attr-dict
  }];
}

def TNotifyOp : PTO_Op<"comm.tnotify", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Send a signal notification to remote GM";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$value,
      PTO_NotifyOpAttr:$notifyOp
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $value `:` type($signal) `,` type($value) `)`
    attr-dict
  }];
}

def TWaitOp : PTO_Op<"comm.twait", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Block until signal(s) satisfy a comparison";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $cmpValue `:` type($signal) `,` type($cmpValue) `)`
    attr-dict
  }];
}

def TTestOp : PTO_Op<"comm.ttest", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-blocking signal comparison test";
  let arguments = (ins
      PTOCommType:$signal,
      AnySignlessInteger:$cmpValue,
      PTO_WaitCmpAttr:$cmp
  );
  let results = (outs I1:$result);
  let hasVerifier = 1;
  let assemblyFormat = [{
    `(` $signal `,` $cmpValue `:` type($signal) `,` type($cmpValue) `)`
    attr-dict `->` type($result)
  }];
}

def TBroadcastOp : PTO_Op<"comm.tbroadcast", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast local GM data to all group members";
  let arguments = (ins
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CommTGatherOp : PTO_Op<"comm.tgather", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather remote GM data from a communication group";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def CommTScatterOp : PTO_Op<"comm.tscatter", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter local GM data to a communication group";
  let arguments = (ins
      PTOCommType:$src,
      PTOCommType:$ping,
      Optional<PTOCommType>:$pong,
      Variadic<PTOCommType>:$group,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def TReduceOp : PTO_Op<"comm.treduce", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce remote GM data from a communication group";
  let arguments = (ins
      PTOCommType:$dst,
      PTOCommType:$acc,
      PTOCommType:$recvPing,
      Optional<PTOCommType>:$recvPong,
      Variadic<PTOCommType>:$group,
      PTO_ReduceOpAttr:$reduceOp,
      I32Attr:$root
  );
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def InitializeL2G2LPipeOp : PTO_Op<"initialize_l2g2l_pipe", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Initialize a local-to-global-to-local pipe handle";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      I32Attr:$slot_num,
      OptionalAttr<I32Attr>:$local_slot_num,
      OptionalAttr<I32Attr>:$flag_base,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      AnyType:$gm_addr,
      Optional<AnyType>:$local_addr,
      Optional<AnyType>:$peer_local_addr
  );

  let results = (outs PipeType:$pipe);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `,`
        `slot_num` `=` $slot_num
        (`,` `local_slot_num` `=` $local_slot_num^)?
        (`,` `flag_base` `=` $flag_base^)?
        (`,` `nosplit` `=` $nosplit^)?
        `}`
    `(` $gm_addr `:` type($gm_addr)
        (`,` $local_addr^ `:` type($local_addr))?
        (`,` $peer_local_addr^ `:` type($peer_local_addr))? `)`
    attr-dict `->` qualified(type($pipe))
  }];
}

def InitializeL2LPipeOp : PTO_Op<"initialize_l2l_pipe", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Initialize a local-to-local pipe handle";

  let arguments = (ins
      I8Attr:$dir_mask,
      I32Attr:$slot_size,
      I32Attr:$slot_num,
      OptionalAttr<I32Attr>:$flag_base,
      OptionalAttr<BoolAttr>:$nosplit,
      OptionalAttr<PTO_AccPushEpilogueAttr>:$acc_push_epilogue,
      AnyType:$local_addr,
      Optional<AnyType>:$peer_local_addr
  );

  let results = (outs PipeType:$pipe);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `{` `dir_mask` `=` $dir_mask `,`
        `slot_size` `=` $slot_size `,`
        `slot_num` `=` $slot_num
        (`,` `flag_base` `=` $flag_base^)?
        (`,` `nosplit` `=` $nosplit^)?
        `}`
    `(` $local_addr `:` type($local_addr)
        (`,` $peer_local_addr^ `:` type($peer_local_addr))? `)`
    attr-dict `->` qualified(type($pipe))
  }];
}

def TPushOp : PTO_TOp<"tpush", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Push tile data through a unified pipe handle";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      PipeType:$pipe_handle,
      Optional<I64>:$aiv_subblockid,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAddressSpace = [](Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto as = getAddressSpace(getTile().getType());
      if (!as)
        return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
      if (*as == ::mlir::pto::AddressSpace::ACC)
        return ::mlir::pto::PIPE::PIPE_FIX;
      if (*as == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_MTE3;
      return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
    }
  }];
}

def DeclareTileOp : PTO_Op<"declare_tile"> {
  let summary = "Declare a tile value whose address will be assigned later";

  let results = (outs PTODpsType:$tile);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($tile))
  }];
}

def DeclareGlobalOp : PTO_Op<"declare_global"> {
  let summary = "Declare a GlobalTensor descriptor whose address is assigned later";

  let results = (outs TensorViewType:$entry);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($entry))
  }];
}

def TAllocOp : PTO_TOp<"talloc", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a declared GlobalTensor descriptor to the next producer FIFO slot";

  let arguments = (ins
      TensorViewType:$entry,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $entry `,` $pipe_handle `:` qualified(type($entry)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}` attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }
  }];
}

def TAssignOp : PTO_Op<"tassign", [Pure, AllTypesMatch<["tile", "result"]>]> {
  let summary = "Rebind a declared tile handle to a runtime address";
  let description = [{
    Rebinds an existing tile handle to a new address. This op is intended for
    manual synchronization workflows (for example, rebinding inside loops while
    keeping the tile declaration outside the loop).
  }];

  let arguments = (ins
    PTODpsType:$tile,
    I64:$addr
  );
  let results = (outs PTODpsType:$result);

  let assemblyFormat = [{
    $tile `,` $addr attr-dict `:` qualified(type($tile)) `->` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}

def DeclareEventIdArrayOp : PTO_Op<"declare_eventid_array"> {
  let summary = "Declare a manual-only local event-id array";
  let description = [{
    Declares a local array used to store dynamic event ids in manual sync code.
    This is a frontend/EmitC convenience feature and is not intended to
    participate in PlanMemory or automatic synchronization passes.
  }];

  let results = (outs EventIdArrayType:$array);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($array))
  }];
}

def EventIdArrayGetOp : PTO_Op<"eventid_array_get"> {
  let summary = "Read one element from a manual event-id array";

  let arguments = (ins
    EventIdArrayType:$array,
    Index:$index
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $array `[` $index `]` attr-dict `:` qualified(type($array))
  }];
}

def EventIdArraySetOp : PTO_Op<"eventid_array_set"> {
  let summary = "Write one element into a manual event-id array";

  let arguments = (ins
    EventIdArrayType:$array,
    Index:$index,
    Index:$value
  );
  let results = (outs);

  let assemblyFormat = [{
    $array `[` $index `]` `,` $value attr-dict `:` qualified(type($array))
  }];
}

//===----------------------------------------------------------------------===//
// Stack-local array (minimum set)
//===----------------------------------------------------------------------===//
//
// Mirrors the `eventid_array` triad (declare / get / set) for a generic
// statically-shaped scalar stack array. Disjoint from the tile-buf / scratch
// memory world: the array's address is decided by the host C++ compiler at
// codegen time, never by PTO memory planning.

def DeclareLocalArrayOp : PTO_Op<"declare_local_array"> {
  let summary = "Declare a stack-local statically-shaped array";
  let description = [{
    Lowers to a plain C++ stack variable like `T a[d1][d2]...;`. The element
    type is restricted to scalars (integer / float); no nested aggregates.

    This op does not participate in PTO memory planning. The resulting array's
    address is determined by the host C++ compiler.
  }];

  let results = (outs LocalArrayType:$array);

  let assemblyFormat = [{
    attr-dict `->` qualified(type($array))
  }];
}

def LocalArrayGetOp : PTO_Op<"local_array_get"> {
  let summary = "Read one element from a stack-local array";
  let description = [{
    Lowers to a scalar snapshot initialized from `a[i0][i1]...[iN-1]` in the
    emitted C++. This preserves SSA value semantics if the array is mutated
    later. The number of indices must equal the array's rank.
  }];

  let arguments = (ins
    LocalArrayType:$array,
    Variadic<Index>:$indices
  );
  let results = (outs ScalarType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $array `[` $indices `]` attr-dict
      `:` qualified(type($array)) `->` type($result)
  }];
}

def LocalArraySetOp : PTO_Op<"local_array_set"> {
  let summary = "Write one element into a stack-local array";
  let description = [{
    Lowers to `a[i0][i1]...[iN-1] = v;` in the emitted C++. The number of
    indices must equal the array's rank.
  }];

  let arguments = (ins
    LocalArrayType:$array,
    Variadic<Index>:$indices,
    ScalarType:$value
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $array `[` $indices `]` `,` $value attr-dict
      `:` qualified(type($array)) `,` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Stack-local heterogeneous aggregate (struct)
//===----------------------------------------------------------------------===//
//
// Mirrors the `local_array` triad (declare / get / set) for a generic
// heterogeneous scalar-unit aggregate. The struct's address is decided by the
// host C++ compiler at codegen time, never by PTO memory planning or
// `pto.pointer_cast`. Field positions are constant (struct semantics), so
// `get` / `set` carry a constant `path` of field indices that may descend
// through nested structs. The path always ends at a scalar; a nested struct is
// traversed by a longer path, never handed back as a value.

def DeclareStructOp : PTO_Op<"declare_struct"> {
  let summary = "Declare a stack-local heterogeneous aggregate";
  let description = [{
    Lowers to a plain C++ stack variable like `PtoStruct_X s;`. Field types are
    restricted to scalars (integers of width 8/16/32/64, f16/bf16/f32/f64) or a
    nested `!pto.struct`.

    This op does not participate in PTO memory planning. The resulting struct's
    address is determined by the host C++ compiler.

    The struct is stack storage owned by the scope that declares it, and a
    `!pto.struct` value lowers to a pointer to that storage. Handing the value
    to a terminator would publish the address outside the owning scope, so it is
    rejected: returning it would leave the caller holding a pointer to a dead
    frame. Module-level provenance validation also rejects function results and
    derived operation results of this type, so aliases cannot launder that
    address past the direct-use check. Struct storage originates from this op
    or is received through a function argument.
    Since `pto.struct_set` mutates in place, a struct never needs to be returned
    or yielded — declare it in the outer scope and pass it down as a function
    argument instead.
  }];

  let results = (outs StructType:$s);

  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `->` qualified(type($s))
  }];
}

def StructGetOp : PTO_Op<"struct_get"> {
  let summary = "Read a field from a stack-local struct by constant path";
  let description = [{
    Reads the field reached by following the constant `path` of field indices
    from the struct, descending through nested structs. Lowers to `s.fA.fB...`
    member access, read into a fresh variable so the SSA value keeps its value
    if the struct is mutated later (mirrors `pto.local_array_get`).

    The path must end at a scalar field, and the result type must equal that
    field's type. A path stopping on a nested `!pto.struct` is rejected —
    extend it to reach a scalar inside.
  }];

  let arguments = (ins
    StructType:$s,
    DenseI64ArrayAttr:$path
  );
  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $s custom<StructPath>($path) attr-dict
      `:` qualified(type($s)) `->` type($value)
  }];
}

def StructSetOp : PTO_Op<"struct_set"> {
  let summary = "Write a field into a stack-local struct by constant path";
  let description = [{
    Writes `value` into the field reached by following the constant `path` of
    field indices, descending through nested structs. Lowers to
    `s.fA.fB... = value;`.

    The path must end at a scalar field, and the value type must equal that
    field's type. A path stopping on a nested `!pto.struct` is rejected —
    extend it to reach a scalar inside.
  }];

  let arguments = (ins
    StructType:$s,
    DenseI64ArrayAttr:$path,
    AnyType:$value
  );
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $s custom<StructPath>($path) `,` $value attr-dict
      `:` qualified(type($s)) `,` type($value)
  }];
}

def TPopOp : PTO_TOp<"tpop", [
    PTO_DpsInitOpInterface,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a declared consumer tile to the next FIFO slot";

  let arguments = (ins
      PTOPipeEntryType:$tile,
      PipeType:$pipe_handle,
      Optional<I64>:$aiv_subblockid,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $tile `,` $pipe_handle `:` qualified(type($tile)) `,` qualified(type($pipe_handle)) `)`
    `{` `split` `=` $split `}`
    (`aiv_subblockid` `(` $aiv_subblockid^ `)`)?
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getAddressSpace = [](Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }
        return false;
      };

      auto as = getAddressSpace(getTile().getType());
      if (!as)
        return ::mlir::pto::PIPE::PIPE_UNASSIGNED;

      switch (*as) {
      case ::mlir::pto::AddressSpace::VEC:
      case ::mlir::pto::AddressSpace::MAT:
        return isA5Target() ? ::mlir::pto::PIPE::PIPE_S
                            : ::mlir::pto::PIPE::PIPE_MTE2;
      default:
        break;
      }
      return ::mlir::pto::PIPE::PIPE_UNASSIGNED;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable() { return getTileMutable(); }
  }];
}

def TFreeOp : PTO_TOp<"tfree", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Release the current consumer slot for a pipe";

  let arguments = (ins
      Optional<PTOPipeEntryType>:$entry,
      PipeType:$pipe_handle,
      I8Attr:$split
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization Ops
//===----------------------------------------------------------------------===//

def SetFlagOp : PTO_Op<"set_flag"> {
  let summary = "Set synchronization flag between pipes";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def WaitFlagOp : PTO_Op<"wait_flag"> {
  let summary = "Wait for synchronization flag";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def SetFlagDynOp : PTO_Op<"set_flag_dyn"> {
  let summary = "Set synchronization flag between pipes (dynamic event id)";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    Index:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def WaitFlagDynOp : PTO_Op<"wait_flag_dyn"> {
  let summary = "Wait for synchronization flag (dynamic event id)";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    Index:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Buffer-ID Synchronization (A5)
//===----------------------------------------------------------------------===//

def PTO_PipeLikeAttr
    : AnyAttrOf<[PTO_PipeEventTypeAttr, PTO_SyncOpTypeAttr, PTO_PipeAttr]>;

def GetBufOp : PTO_Op<"get_buf"> {
  let summary = "Acquire a buffer-id token for a sync op type (A5)";
  let description = [{
    `pto.get_buf` participates in a buffer-id based ordering model. Operations
    mapped to the same pipe and guarded by the same buffer-id are enforced to
    execute in program order relative to other mapped pipes using the same
    buffer-id.

    Like `pto.record_event` / `pto.wait_event`, this op accepts a high-level
    op type endpoint (`pipe_event_type` / `sync_op_type`) instead of a concrete
    pipe name.

    This op is intended to be lowered to the CCEC builtin intrinsic `get_buf`.
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def GetBufDynOp : PTO_Op<"get_buf_dyn"> {
  let summary = "Acquire a buffer-id token (dynamic buf-id)";
  let description = [{
    Dynamic variant of `pto.get_buf` where the buffer-id is provided as an
    SSA value (index) instead of a static integer attribute. This enables
    runtime-computed buf_id patterns such as SIMT ping-pong buffering
    (e.g. `iter & 1`).
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    Index:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def RlsBufOp : PTO_Op<"rls_buf"> {
  let summary = "Release a buffer-id token for a sync op type (A5)";
  let description = [{
    Releases the previously acquired buffer-id token for the given op type
    (mapped to a concrete pipe).

    This op is intended to be lowered to the CCEC builtin intrinsic `rls_buf`.
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    I32Attr:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result);
  }];
}

def RlsBufDynOp : PTO_Op<"rls_buf_dyn"> {
  let summary = "Release a buffer-id token (dynamic buf-id)";
  let description = [{
    Dynamic variant of `pto.rls_buf` where the buffer-id is provided as an
    SSA value (index) instead of a static integer attribute.
  }];

  let arguments = (ins
    PTO_PipeLikeAttr:$op_type,
    Index:$buf_id,
    DefaultValuedAttr<I32Attr, "0">:$mode
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SyncSetOp : PTO_Op<"sync.set"> {
  let summary = "Set a synchronization signal (trigger) between cube and vector.";
  let description = [{
    Sets a synchronization signal on the specified pipeline stage.
    Corresponds to `ffts_cross_core_sync` on both A3 and A5.  The optional
    `ffts_mode` selects FFTS mode 0 (inter), 1 (subblock), or 2 (block);
    if omitted, block mode (2) is used for backward compatibility.
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    OptionalAttr<I32Attr>:$event_id,
    OptionalAttr<I32Attr>:$ffts_mode,
    Optional<Index>:$event_id_dyn
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SyncWaitOp : PTO_Op<"sync.wait"> {
  let summary = "Wait for a synchronization signal (barrier) between cube and vector.";
  let description = [{
    Waits for a synchronization signal on the specified pipeline stage.
    Corresponds to `wait_flag_dev` on both A3 and A5.  `ffts_mode` records
    the matching set-side FFTS mode (0, 1, or 2).
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    OptionalAttr<I32Attr>:$event_id,
    OptionalAttr<I32Attr>:$ffts_mode,
    Optional<Index>:$event_id_dyn
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// Named cross-/intra-block synchronization operations.  Cross-core operations
// are canonicalized to sync.set/sync.wait with FFTS mode 0 in the VPTO path;
// intra-block operations lower directly to their CCE builtins.
def SetCrossBlockOp : PTO_Op<"set_cross_block"> {
  let summary = "Signal a cross-block synchronization flag";
  let description = [{
    Canonicalizes to `pto.sync.set` with FFTS mode 0.
  }];
  let arguments = (ins PTO_PipeAttr:$pipe, OptionalAttr<I32Attr>:$event_id,
                       DefaultValuedAttr<I32Attr, "0">:$ffts_mode,
                       Optional<Index>:$event_id_dyn);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def WaitCrossBlockOp : PTO_Op<"wait_cross_block"> {
  let summary = "Wait for a cross-block synchronization flag";
  let description = [{ Canonicalizes to `pto.sync.wait` with FFTS mode 0. }];
  let arguments = (ins PTO_PipeAttr:$pipe, OptionalAttr<I32Attr>:$event_id,
                       Optional<Index>:$event_id_dyn);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def SetIntraBlockOp : PTO_Op<"set_intra_block"> {
  let summary = "Signal an intra-block synchronization flag";
  let description = [{ Lowers to `__builtin_cce_set_intra_block`. }];
  let arguments = (ins PTO_PipeAttr:$pipe, OptionalAttr<I32Attr>:$event_id,
                       Optional<Index>:$event_id_dyn);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def WaitIntraBlockOp : PTO_Op<"wait_intra_block"> {
  let summary = "Wait for an intra-block synchronization flag";
  let description = [{ Lowers to `__builtin_cce_wait_intra_block`. }];
  let arguments = (ins PTO_PipeAttr:$pipe, OptionalAttr<I32Attr>:$event_id,
                       Optional<Index>:$event_id_dyn);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def BarrierOp : PTO_Op<"barrier"> {
  let summary = "Intra-pipeline memory barrier";
  let arguments = (ins PTO_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
}

def CmoCacheInvalidOp : PTO_Op<"cmo.cacheinvalid"> {
  let summary = "Perform cache maintenance for a memory space";
  let description = [{
    Cache maintenance operation for the specified memory space.

    The whole-cache form lowers to PTO-ISA's two-argument
    `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)`.

    The single-cache-line form carries the GM payload address selected by the
    producer DSL/user and lowers to
    `dcci(addr, cache_line_t::SINGLE_CACHE_LINE)`.

    This op is an explicit CMO boundary. PTOAS no longer runs the historical
    MemoryConsistency analysis pass that consumed marker-only CMO ops or inferred
    signal/payload pipe drains, so precise placement and coverage are the
    producer DSL/user's responsibility.
  }];

  let arguments = (ins
    Optional<AnyType>:$addr,
    PTO_AddressSpaceAttr:$space
  );

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

def FenceBarrierAllOp : PTO_Op<"fence.barrier_all"> {
  let summary = "All-direction memory fence for a visibility scope";
  let description = [{
    Memory fence for a visibility scope. Around signal/payload communication it
    can be used as the producer-side publish barrier or the consumer-side
    acquire barrier depending on placement. In EmitC, `scope = gm` and
    `scope = all` drain all vector-kernel pipes with `PIPE_ALL` before lowering
    to `dsb(DSB_DDR)`. Other kernel kinds conservatively drain MTE2, MTE3, and
    FIX before the DDR barrier. Users therefore do not need to separately model
    producer pipe classes before a GM visibility fence.
  }];

  let arguments = (ins PTO_FenceScopeAttr:$scope);
  let assemblyFormat = "$scope attr-dict";
}

def TSyncOp : PTO_TOp<"tsync"> {
  let summary = "Direct TSYNC mapping (variadic operands).";
  let description = [{
    Lowers directly to a `TSYNC(...)` intrinsic call with the same operand list.
    This op is intentionally a thin bridge for explicit synchronization usage.
  }];

  let arguments = (ins Variadic<PTODpsType>:$events);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $events `:` type($events) `)` attr-dict
  }];
}

def SyncAllOp : PTO_Op<"syncall", [AttrSizedOperandSegments]> {
  let summary = "Cross-core all-participant synchronization.";
  let description = [{
    Models `pto::SYNCALL`.

    Supported IR forms:
    - hard sync: no operands, `mode = hard`
    - soft sync: `gm_workspace`, optional `used_cores`

    The soft operand ABI is identical for AIV-only, AIC-only, and mixed
    synchronization. The GM workspace must reserve at least 16 `i32` elements
    (64 bytes), be zero-initialized before first use, and occupy an exclusive
    cache line.
  }];

  let arguments = (ins
    Optional<AnyTypeOf<[PtrType, TensorViewLike],
                       "Ptr, TensorView, or PartitionTensorView">>:$gm_workspace,
    Optional<I32>:$used_cores,
    PTO_SyncAllModeAttr:$mode,
    PTO_SyncCoreTypeAttr:$core_type
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// ---------------------------------------------------------------------------
// TileBuf intrinsics — used in TileLang DSL-generated template functions.
// These ops extract memref address and valid shape from tile_buf parameters.
// After inline, FoldTileBufIntrinsics resolves them to concrete values.
// ---------------------------------------------------------------------------

def TileValidRowsOp : PTO_Op<"tile_valid_rows", [Pure]> {
  let summary = "Extract valid row count from a tile_buf.";
  let description = [{
    Returns the valid row count (v_row) of a `tile_buf` as an `index` value.
    When the tile_buf has a static v_row, FoldTileBufIntrinsics folds this
    into `arith.constant`. When v_row is dynamic (`?`), the fold resolves
    it to the runtime index value carried by the tile_buf.
  }];

  let arguments = (ins TileBufType:$src);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($result)
  }];
}

def TileValidColsOp : PTO_Op<"tile_valid_cols", [Pure]> {
  let summary = "Extract valid column count from a tile_buf.";
  let description = [{
    Returns the valid column count (v_col) of a `tile_buf` as an `index` value.
    When the tile_buf has a static v_col, FoldTileBufIntrinsics folds this
    into `arith.constant`. When v_col is dynamic (`?`), the fold resolves
    it to the runtime index value carried by the tile_buf.
  }];

  let arguments = (ins TileBufType:$src);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $src attr-dict `:` qualified(type($src)) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// FFT Configuration Operation
//===----------------------------------------------------------------------===//

def SetFFTsOp : PTO_Op<"set_ffts", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Set FFTS/flags pointer for runtime (side-effecting).";
  let arguments = (ins PtrType:$ffts);
  let results = (outs);

  let assemblyFormat = "$ffts attr-dict `:` type($ffts)";

  let hasVerifier = 1;
}


def PrintOp : PTO_Op<"print", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Print debug: format string (attribute) and scalar value.";
  let description = [{
    Debug print op. First argument is the format string (string attribute), second is a scalar value.
    Format in IR: `pto.print ins("format", %scalar : type(%scalar))`
  }];
  let arguments = (ins StrAttr:$format, PrintScalarType:$scalar);
  let results = (outs);
  let assemblyFormat = [{
    `ins` `(` $format `,` $scalar `:` type($scalar) `)`
    attr-dict
  }];
}

def TrapOp : PTO_Op<"trap", [MicroOpInterface, SimtOpInterface]> {
  let summary = "Trap: abort execution (no operands).";
  let description = [{
    Inserts a trap to stop execution. No arguments, no results.
    Format in IR: pto.trap
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

// ---- tile-world TOp version (with 't') ----
// pto.tmgather ins(%mem, %idx) outs(%dst) [ ...] -> tensor?
def MGatherOp : PTO_TOp<"mgather", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  MteOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather-load elements from memory into a tile using per-element indices (tile world, ins/outs).";
  let description = [{
    Two destination forms are supported:

    - **GM -> UB (VEC)**: `mem` is a GM table, `idx` is a UB index tile, and
      `dst` is a UB (`loc=vec`) tile. This is the default path.
    - **GM -> L1 (MAT)**: `dst` is an L1 / cube (`loc=mat`) tile in NZ layout,
      `idx` is supplied as a **GM** tensor (the cube core cannot read UB on
      A5), and `Coalesce::Elem` additionally requires a contiguous **GM
      `scratch`** workspace used to stage the gathered elements into NZ layout
      before the bulk GM -> L1 copy. `Coalesce::Row` needs no scratch. The path
      is selected by `dst` being a `loc=mat` tile and mirrors the pto-isa
      `MGATHER` GM -> L1 overloads.
  }];

  let arguments = (ins
    PTODpsType:$mem,
    PTODpsType:$idx,
    PTODpsType:$dst,
    Optional<PTODpsType>:$scratch,
    PTO_CoalesceAttr:$coalesce,
    DefaultValuedAttr<PTO_GatherOOBAttr, "::mlir::pto::GatherOOB::Undefined">:$gatherOob);

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MGATHER"; }
    ::mlir::pto::PIPE getPipe() {
      // GM -> L1 gather: the destination is an L1 / cube (loc=mat) tile and the
      // gather is issued as a GM -> L1 DMA load (copy_gm_to_cbuf / nd2nz) on the
      // MTE2 pipe on every arch, mirroring pto.tload. The GM -> UB (VEC) path
      // below keeps its A5 SIMT (PIPE_V) / A2A3 MTE2 selection.
      //
      // Resolve the destination address space for InsertSync through the
      // OpPipeInterface.
      auto getAddressSpace =
          [](::mlir::Type ty) -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };
      if (auto as = getAddressSpace(getDst().getType());
          as && *as == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE2;

      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_MTE2;
    }
    ::mlir::Value getSource() { return getMem(); }
    ::mlir::Value getDestination() { return getDst(); }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}



// ---- tile-world TOp version ----
// pto.tsetval ins(%offset, %val) outs(%dst) : PTODpsType, index, T
def TSetValOp : PTO_TOp<"tsetval", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Write a scalar value into a single element of dst at offset (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$dst,
    Index:$offset,
    ScalarType:$val
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $offset `,` $val `:` type($offset) `,` type($val) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "SETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// ---- tile-world TOp version ----
// pto.tgetval ins(%src, %offset) outs(%dst) : PTODpsType, index -> T
def TGetValOp : PTO_TOp<"tgetval", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Read a single element from tile-like src at offset into a scalar (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    Index:$offset
  );

  let results = (outs ScalarType:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` qualified(type($dst) )
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "GETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
  }];
}

def THistogramOp : PTO_TOp<"thistogram", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "THISTOGRAM: accumulate per-row 256-bin histograms.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$dst,
    DefaultValuedOptionalAttr<I32Attr, "1">:$byte
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` qualified(type($idx)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGetScaleAddrOp : PTO_TOp<"tget_scale_addr", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Bind a scaling tile to the scaled address of a source tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


// ---- tile-world TOp version (with 't') ----
// pto.mscatter ins(%src, %idx) outs(%mem) [ ...]
def MScatterOp : PTO_TOp<"mscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  MteOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter-store elements from a tile into memory using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$mem,               // outs target
    OptionalAttr<PTO_CoalesceAttr>:$coalesce,
    DefaultValuedAttr<PTO_ScatterAtomicOpAttr, "::mlir::pto::ScatterAtomicOp::None">:$scatterAtomicOp,
    DefaultValuedAttr<PTO_ScatterOOBAttr, "::mlir::pto::ScatterOOB::Undefined">:$scatterOob,
    OptionalAttr<PTO_ScatterConflictAttr>:$scatterConflict
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MSCATTER"; }
    ::mlir::pto::PIPE getPipe() {
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::Value getSource() { return getSrc(); }
    ::mlir::Value getDestination() { return getMem(); }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getMemMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTO_TOPs(Tilebuffer, DPS)
//===----------------------------------------------------------------------===//

def TAbsOp : PTO_TOp<"tabs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise absolute value of a tile ";
  let description = [{
    For each element (i, j): dst[i,j] = |src[i,j]|.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs); 

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)` `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V; // Vector pipe for elementwise ops
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddOp : PTO_TOp<"tadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise add of two tiles ";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] + src1[i,j].
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def TAddReluOp : PTO_TOp<"taddrelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Fused elementwise add + ReLU of two tiles";
  let description = [{
    For each element (i, j): dst[i,j] = max(0, src0[i,j] + src1[i,j]).
    Maps to the fused VADDRELU intrinsic on C220.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_V;
    }

    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def TAddCOp : PTO_TOp<"taddc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise ternary add of tiles ";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] + src1[i,j] + src2[i,j].
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src2,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
          qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddSOp : PTO_TOp<"tadds", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise add a scalar to a tile ";
  let description = [{
    For each element (i, j): dst[i,j] = src[i,j] + scalar.
    Tile operands are tilebuffers; scalar is a builtin scalar type (e.g. f32).
  }];

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAxpyOp : PTO_TOp<"taxpy", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TAXPY: dst += src * scalar.";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAddSCOp : PTO_TOp<"taddsc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "dst = src0 + scalar + src1 ";
  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$src1,
    PTODpsType:$dst
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:` qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


def TAndOp : PTO_TOp<"tand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TAND(dst, src0, src1) bitwise-and on tiles";
  let arguments = (ins PTODpsType:$src0, PTODpsType:$src1, PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TConcatOp: PTO_TOp<"tconcat", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TCONCAT: Concatenate two tiles along the column dimension.";
  let description = [{
    Semantics (per row):
      dst[:, 0:validCol0] = src0
      dst[:, validCol0:validCol0+validCol1] = src1

    Constraints are implementation-defined; see verifier for current checks.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TConcatidxOp: PTO_TOp<"tconcatidx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TCONCAT (indexed): Concatenate two tiles along the column dimension with per-row index control.";
  let description = [{
    Indexed form of tconcat. Takes two additional index tiles (src0Idx, src1Idx)
    that specify the number of columns to copy from each source on a per-row basis.

    Semantics (per row):
      idx0_num = src0Idx[row, 0]
      idx1_num = src1Idx[row, 0]
      copy from src0: min(idx0_num, src0_valid_col, dst_valid_col) columns
      copy from src1: min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0) columns
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src0Idx,
    PTODpsType:$src1Idx,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TAndSOp : PTO_TOp<"tands", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TANDS(dst, src, scalar) bitwise-and tile with scalar";
  let arguments = (ins PTODpsType:$src, AnySignlessInteger:$scalar, PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


def TCIOp : PTO_TOp<"tci", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate contiguous integer sequence into dst tile.";

  let arguments = (ins
    AnyInteger:$S,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TTriOp : PTO_TOp<"ttri", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate a lower/upper triangular mask tile";

  let arguments = (ins
    AnyInteger:$diagonal,
    PTODpsType:$dst,
    DefaultValuedAttr<I32Attr, "0">:$upperOrLower
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $diagonal
    attr-dict
    `:` type($diagonal) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TCmpOp : PTO_TOp<"tcmp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare two tiles and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1
            attr-dict
          `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`   
  }];

   let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src0, "Value":$src1, "Value":$dst), [{
      build($_builder, $_state, res, src0, src1, dst, /*cmpMode=*/nullptr); 
    }]>];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TCmpSOp : PTO_TOp<"tcmps", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare scalar value against a tile and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    DefaultValuedAttr<PTO_CmpModeAttr, "::mlir::pto::CmpMode::EQ">:$cmpMode,
    PTODpsType:$dst
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar
          attr-dict
          `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    
  }];
}

def TColExpandOp : PTO_TOp<"tcolexpand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast src(0, j) to all rows in column j ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandAddOp : PTO_TOp<"tcolexpandadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast add: add a per-column scalar vector src1 to src0 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMulOp : PTO_TOp<"tcolexpandmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast multiply: multiply each element of src0 by a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandDivOp : PTO_TOp<"tcolexpanddiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast divide: divide each element of src0 by a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_DivPrecisionAttr, "::mlir::pto::DivPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandSubOp : PTO_TOp<"tcolexpandsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast subtract: subtract a per-column scalar vector src1 from src0 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandExpdifOp : PTO_TOp<"tcolexpandexpdif", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast expdif: compute exp(src0 - src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMaxOp : PTO_TOp<"tcolexpandmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast max: take max(src0, src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColExpandMinOp : PTO_TOp<"tcolexpandmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Column-wise broadcast min: take min(src0, src1) using a per-column scalar vector src1 ";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColMaxOp : PTO_TOp<"tcolmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by taking the maximum across rows ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColArgMaxOp : PTO_TOp<"tcolargmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column to the row index of the maximum element ";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def TColMinOp : PTO_TOp<"tcolmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by taking the minimum across rows ";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TColArgMinOp : PTO_TOp<"tcolargmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column to the row index of the minimum element ";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def TColSumOp : PTO_TOp<"tcolsum", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by summing across rows (tilebuf-based, explicit tmp)";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$isBinary
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(),
                                         getOperation()->getNumOperands() - 1,
                                         1);
    }
  }];
}

def TColProdOp : PTO_TOp<"tcolprod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Reduce each column by multiplying across rows";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TCvtOp : PTO_TOp<"tcvt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise type conversion with optional saturation mode (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RoundModeAttr, "::mlir::pto::RoundMode::CAST_RINT">:$rmode,
    DefaultValuedAttr<PTO_SaturationModeAttr, "::mlir::pto::SaturationMode::OFF">:$sat_mode
  );

  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRandomOp : PTO_TOp<"trandom", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate random values into dst tile using key/counter words.";

  let arguments = (ins
    AnySignlessInteger:$key0,
    AnySignlessInteger:$key1,
    AnySignlessInteger:$counter0,
    AnySignlessInteger:$counter1,
    AnySignlessInteger:$counter2,
    AnySignlessInteger:$counter3,
    PTODpsType:$dst,
    DefaultValuedAttr<I32Attr, "10">:$rounds
  );

  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $key0 `,` $key1 `,` $counter0 `,` $counter1 `,` $counter2 `,` $counter3
    attr-dict
    `:` type($key0) `,` type($key1) `,` type($counter0) `,` type($counter1) `,` type($counter2) `,` type($counter3) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TDivOp : PTO_TOp<"tdiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division of two tiles (tilebuf, DPS)";
  let description = [{
    For each element (i, j): dst[i,j] = src0[i,j] / src1[i,j].
    Division-by-zero behavior is target-defined.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_DivPrecisionAttr, "::mlir::pto::DivPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

}

def TDivSOp : PTO_TOp<"tdivs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division with a scalar (tilebuf, DPS)";

  let arguments = (ins
    AnyType:$src,
    AnyType:$scalar,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_DivPrecisionAttr, "::mlir::pto::DivPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

def TFModOp : PTO_TOp<"tfmod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise fmod/remainder of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_FmodPrecisionAttr, "::mlir::pto::FmodPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TFModSOp : PTO_TOp<"tfmods", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise fmod/remainder with a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (TPOW: elementwise tile-tile power, DPS, tmp scratch optional)
//===----------------------------------------------------------------------===//

def TPowOp : PTO_TOp<"tpow", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPOW: Elementwise power with both base and exponent as tiles.";
  let description = [{
    For each element (i, j): dst[i,j] = pow(base[i,j], exp[i,j]).
    The `tmp` tile is backend scratch only used by the floating-point lowering
    path: it is required when the element type is floating-point or when
    `precisionType = high_precision`, and must be omitted for the pure
    integer path.
  }];

  let arguments = (ins
    PTODpsType:$base,
    PTODpsType:$exp,
    PTODpsType:$dst,
    Optional<PTODpsType>:$tmp,
    DefaultValuedAttr<PTO_PowPrecisionAttr, "::mlir::pto::PowPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (TPOWS: elementwise tile-scalar power, DPS, tmp scratch optional)
//===----------------------------------------------------------------------===//

def TPowSOp : PTO_TOp<"tpows", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPOWS: Elementwise power with a scalar exponent.";
  let description = [{
    For each element (i, j): dst[i,j] = pow(src[i,j], scalar).
    The `tmp` tile is backend scratch only used by the floating-point lowering
    path: required when the element type is floating-point or when
    `precisionType = high_precision`, and must be omitted for the pure integer
    path.
  }];

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst,
    Optional<PTODpsType>:$tmp,
    DefaultValuedAttr<PTO_PowPrecisionAttr, "::mlir::pto::PowPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExpOp : PTO_TOp<"texp", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise exponential (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_ExpPrecisionAttr, "::mlir::pto::ExpPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExpandsOp : PTO_TOp<"texpands", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast scalar into dst (tilebuf, DPS)";

  let arguments = (ins
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $scalar `:` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TExtractOp : PTO_TOp<"textract", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract sub-tile window from src into dst (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol
    (`,` $preQuantScalar^)? `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol)
    (`,` type($preQuantScalar)^)?
    (`fp` $fp^ `:` qualified(type($fp)))? `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // Align with pto-isa TEXTRACT op classes:
      //   - TEXTRACT_M2LR : MAT -> LEFT/RIGHT (and scaling-like targets) (MTE1)
      //   - TEXTRACT_V2M  : VEC -> MAT       (FIX)
      //   - TEXTRACT_A2M  : ACC -> MAT       (FIX)
      // For pure UB slices (VEC -> VEC), treat as vector pipe.
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (!sOpt.has_value() || !dOpt.has_value())
        return ::mlir::pto::PIPE::PIPE_V;

      const auto s = sOpt.value();
      const auto d = dOpt.value();

      if (s == ::mlir::pto::AddressSpace::MAT &&
          (d == ::mlir::pto::AddressSpace::LEFT ||
           d == ::mlir::pto::AddressSpace::RIGHT ||
           d == ::mlir::pto::AddressSpace::BIAS ||
           d == ::mlir::pto::AddressSpace::SCALING)) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      if ((s == ::mlir::pto::AddressSpace::VEC &&
           d == ::mlir::pto::AddressSpace::MAT) ||
          (s == ::mlir::pto::AddressSpace::ACC &&
           (d == ::mlir::pto::AddressSpace::MAT ||
            d == ::mlir::pto::AddressSpace::VEC))) {
        return ::mlir::pto::PIPE::PIPE_FIX;
      }

      if (s == ::mlir::pto::AddressSpace::VEC &&
          d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // Default to vector pipe for unmatched combinations.
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TInsertOp : PTO_TOp<"tinsert", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Insert src sub-tile into dst at (indexRow, indexCol) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr, "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode,
    OptionalAttr<PTO_TInsertModeAttr>:$tinsertMode
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol)
    (`fp` $fp^ `:` qualified(type($fp)))?
    (`pre_quant` $preQuantScalar^ `:` type($preQuantScalar))?
    `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TINSERT runs on different pipes depending on target/path:
    // - Vec(UB)->Vec(UB): PIPE_V.
    // - A5 Vec(UB)->Mat(L1): PIPE_MTE3 (custom UB->L1 copy path).
    // - Acc->Mat / Acc->Vec (A2/A3/A5 regular path): PIPE_FIX.
    ::mlir::pto::PIPE getPipe() {
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }
        return false;
      };

      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::VEC)
        return ::mlir::pto::PIPE::PIPE_V;
      if (isA5Target() && sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
          dOpt.value() == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE3;

      // Default and Acc->Mat path.
      return ::mlir::pto::PIPE::PIPE_FIX;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TFillPadOp : PTO_TOp<"tfillpad", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Fill padding with compiler-inferred lowering semantics (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    OptionalAttr<PTO_PadValueAttr>:$padValue
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        return std::nullopt;
      };

      auto sOpt = getASFromType(getSrc().getType());
      auto dOpt = getASFromType(getDst().getType());
      if (sOpt.has_value() && dOpt.has_value() &&
          sOpt.value() == ::mlir::pto::AddressSpace::MAT &&
          dOpt.value() == ::mlir::pto::AddressSpace::MAT)
        return ::mlir::pto::PIPE::PIPE_MTE1;
      return ::mlir::pto::PIPE::PIPE_V;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGatherOp : PTO_TOp<"tgather", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather/select elements using index+tmp, scalar-compare+tmp, or a mask pattern (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$cdst,
    Optional<PTODpsType>:$indices,
    Optional<PTODpsType>:$tmp,
    Optional<ScalarType>:$kValue,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern,
    OptionalAttr<StrAttr>:$axis,
    OptionalAttr<PTO_CmpModeAttr>:$cmpMode,
    OptionalAttr<I32Attr>:$offset
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasCompareForm() { return static_cast<bool>(getCdst()) || static_cast<bool>(getKValue()); }
    bool hasIndexForm() { return static_cast<bool>(getIndices()); }
    bool hasMaskForm() { return static_cast<bool>(getMaskPatternAttr()); }
    bool hasAxis() { return static_cast<bool>(getAxisAttr()); }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, 1 + (getCdst() ? 1 : 0));
    }
  }];
}

def TGatherBOp : PTO_TOp<"tgatherb", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather 32-byte blocks using byte addresses (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$offsets,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offsets `:` qualified(type($src)) `,` type($offsets) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TLogOp : PTO_TOp<"tlog", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise natural logarithm (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_LogPrecisionAttr, "::mlir::pto::LogPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TLReluOp : PTO_TOp<"tlrelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Leaky ReLU with a scalar slope (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$slope,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $slope `:` qualified(type($src)) `,` type($slope) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMaxOp : PTO_TOp<"tmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise maximum of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMaxSOp : PTO_TOp<"tmaxs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise max of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMinOp : PTO_TOp<"tmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of two tiles (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMinSOp : PTO_TOp<"tmins", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TMrgSortOp: PTO_TOp<"tmrgsort", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMRGSORT: Merge sort (format1: ins(src, blockLen) outs(dst); format2: ins(2 to 4 srcs, tmp) outs(dst, executed)).";

  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Optional<AnyInteger>:$blockLen,
    Variadic<PTODpsType>:$dsts,
    Optional<PTODpsType>:$tmp,
    Optional<AnyType>:$excuted,
    DefaultValuedAttr<BoolAttr, "false">:$exhausted
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isFormat1() { return getSrcs().size() == 1u && getBlockLen() && getDsts().size() == 1u; }
    bool isFormat2() { return getSrcs().size() >= 2u && getSrcs().size() <= 4u && getTmp() && getDsts().size() == 1u && getExcuted(); }
    bool isFormat2WithoutTmp() { return getSrcs().size() >= 2u && getSrcs().size() <= 4u && !getTmp() && !getBlockLen() && getDsts().size() == 1u && getExcuted(); }
    Value getSrc() { return getSrcs().front(); }
    Value getDst() { return getDsts().front(); }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstsMutable(); }
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def TMulOp: PTO_TOp<"tmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMUL: Elementwise multiply of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TMULS DPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TMulSOp: PTO_TOp<"tmuls", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMULS: Elementwise multiply a tile by a scalar.";

  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `:` qualified(type($src0)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNEG TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TNegOp: PTO_TOp<"tneg", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNEG: Elementwise negation of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNOT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TNotOp: PTO_TOp<"tnot", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNOT: Elementwise bitwise NOT of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TOrOp: PTO_TOp<"tor", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TOR: Elementwise bitwise OR of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TOrSOp: PTO_TOp<"tors", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TORS: Elementwise bitwise OR of a tile and a scalar.";

  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTADD TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartAddOp: PTO_TOp<"tpartadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPARTADD: Partial elementwise add with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMaxOp: PTO_TOp<"tpartmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise max with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMinOp: PTO_TOp<"tpartmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise min with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TPartArgMaxOp: PTO_TOp<"tpartargmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise max with value and index tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src0Idx,
    PTODpsType:$src1Idx,
    PTODpsType:$dst,
    PTODpsType:$dstIdx
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    `outs` `(` $dst `,` $dstIdx `:`
      qualified(type($dst)) `,` qualified(type($dstIdx)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 4, 2);
    }
  }];
}

def TPartArgMinOp: PTO_TOp<"tpartargmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise min with value and index tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src0Idx,
    PTODpsType:$src1Idx,
    PTODpsType:$dst,
    PTODpsType:$dstIdx
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src0Idx `,` $src1Idx `:`
      qualified(type($src0)) `,` qualified(type($src1)) `,`
      qualified(type($src0Idx)) `,` qualified(type($src1Idx)) `)`
    `outs` `(` $dst `,` $dstIdx `:`
      qualified(type($dst)) `,` qualified(type($dstIdx)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 4, 2);
    }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPartMulOp: PTO_TOp<"tpartmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Partial elementwise mul with implementation-defined handling of mismatched valid regions.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPReluOp: PTO_TOp<"tprelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRELU: Elementwise PReLU (parametric ReLU) with a per-element slope tile.";

  let description = [{
    DPS form:
      tile buf:  pto.tprelu ins(%src0, %src1, %tmp) outs(%dst)
  }];
  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TQuantOp
//===----------------------------------------------------------------------===//

def TQuantOp : PTO_TOp<"tquant", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TQUANT: Quantize a tile using a scaling (fp) tile.";

  let description = [{
    Converts src tile elements to a lower-precision format using the scaling
    parameters held in the fp tile, writing results into dst.

    For INT8_ASYM an additional offset tile must be supplied in ins.

    The optional tmp tile is the scratch buffer used by the A2/A3
    TQUANT_IMPL row-wise broadcast (the fp->s32 conversion reuses it). On
    pto-isa builds where the 4-arg TQUANT overload routes through the fixed
    TMP_UB_OFFSET scratch area, supplying tmp selects the tmp-aware 5-arg
    overload and avoids the fixed-address collision. PTOAS preserves an omitted
    tmp and lowers it through the 4-arg overload. On A5 the tmp operand is
    accepted as a placeholder and does not add extra shape or memory-planning
    constraints.

    DPS form (INT8_SYM):
      pto.tquant ins(%src, %fp : <src_type>, <fp_type>)
                 outs(%dst : <dst_type>) {quant_type = #pto<quant_type INT8_SYM>}
    DPS form (INT8_SYM, with tmp):
      pto.tquant ins(%src, %fp : <src_type>, <fp_type>)
                 outs(%dst, %tmp : <dst_type>, <tmp_type>) {quant_type = #pto<quant_type INT8_SYM>}
    DPS form (INT8_ASYM):
      pto.tquant ins(%src, %fp, %offset : <src_type>, <fp_type>, <offset_type>)
                 outs(%dst : <dst_type>) {quant_type = #pto<quant_type INT8_ASYM>}
    DPS form (INT8_ASYM, with tmp):
      pto.tquant ins(%src, %fp, %offset : <src_type>, <fp_type>, <offset_type>)
                 outs(%dst, %tmp : <dst_type>, <tmp_type>) {quant_type = #pto<quant_type INT8_ASYM>}
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    Optional<PTODpsType>:$offset,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    PTO_QuantTypeAttr:$quant_type
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TQuantMxOp : PTO_TOp<"tquant.mx", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TQUANT MX: Quantize a vec tile into MXFP8/MXFP4 and materialize exp/max/scaling tiles.";

  let description = [{
    A5-only MX quantization form. Quantizes `src` into `dst` and additionally
    materializes the auxiliary `exp`, `max`, and `scaling` tiles required by
    PTO-ISA MX quantization.

    The legacy fused form with `exp_zz` and `storeMode` is deprecated. Use a
    separate `pto.tmov` with a non-scaling X-to-ZZ temporary tile when an
    exponent layout conversion is required.

    DPS form:
      pto.tquant.mx ins(%src : <src_type>)
                    outs(%dst, %exp, %max, %scaling : <dst_type>, <exp_type>, <max_type>, <scaling_type>)
                    {quant_type = #pto<quant_type MXFP8>}
      pto.tquant.mx ins(%src : <src_type>)
                    outs(%dst, %exp, %max, %scaling, %exp_zz : <dst_type>, <exp_type>, <max_type>, <scaling_type>, <exp_type>)
                    {quant_type = #pto<quant_type MXFP8>, storeMode = #pto<vec_store_mode nz>}
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    PTODpsType:$exp,
    PTODpsType:$max,
    PTODpsType:$scaling,
    Optional<PTODpsType>:$exp_zz,
    PTO_QuantTypeAttr:$quant_type,
    DefaultValuedAttr<PTO_QuantScaleAlgAttr, "::mlir::pto::QuantScaleAlg::OCP">:$quantScaleAlg,
    OptionalAttr<PTO_VecStoreModeAttr>:$storeMode,
    DefaultValuedAttr<PTO_MxGroupAxisAttr, "::mlir::pto::MxGroupAxis::Axis1">:$grpAxis,
    DefaultValuedAttr<BoolAttr, "false">:$interleave
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 1, getExpZz() ? 5 : 4);
    }
  }];
}
//===----------------------------------------------------------------------===//
// TDequantOp
//===----------------------------------------------------------------------===//

def TDequantOp : PTO_TOp<"tdequant", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TDEQUANT: Dequantize an integer tile using per-row scale and offset tiles.";

  let description = [{
    Converts an integer (int8 or int16) src tile to float32, applying per-row
    scale and offset parameters held in the scale and offset tiles.

    Formula: dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]

    The scale and offset tiles are row-vectors: one f32 value per row is
    broadcast across all columns of that row.

    DPS form:
      pto.tdequant ins(%src, %scale, %offset : <src_type>, <scale_type>, <offset_type>)
                   outs(%dst : <dst_type>)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$scale,
    PTODpsType:$offset,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scale `,` $offset `:` qualified(type($src)) `,` qualified(type($scale)) `,` qualified(type($offset)) `)`
    `outs` `(` $dst `:` qualified(type($dst)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRECIP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRecipOp: PTO_TOp<"trecip", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRECIP: Elementwise reciprocal of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RecipPrecisionAttr, "::mlir::pto::RecipPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TReluOp: PTO_TOp<"trelu", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRELU: Elementwise ReLU of a tile.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRemOp: PTO_TOp<"trem", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREM: Elementwise remainder of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RemPrecisionAttr, "::mlir::pto::RemPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREMS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRemSOp: PTO_TOp<"trems", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREMS: Elementwise remainder with a scalar";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (TRESHAPE: SSA view op; aliases src storage)
//===----------------------------------------------------------------------===//

def TReshapeOp: PTO_TOp<"treshape", [
  OpPipeInterface,
  Pure,
  ViewLikeOpInterface
]> {
  let summary = "TRESHAPE: Reinterpret a tile buffer view (SSA; aliases src storage)";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs
    PTODpsType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let extraClassDeclaration = [{
    // Keep consistent with pto-isa opPipeList: TRESHAPE is modeled on PIPE_S.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::Value getViewSource() { return getSrc(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPAND TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandOp: PTO_TOp<"trowexpand", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPAND: Broadcast the first element of each source row across the destination row.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDDIV TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandDivOp: PTO_TOp<"trowexpanddiv", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDDIV: Row-wise broadcast divide: divide each row of src0 by a per-row scalar vector src1.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_DivPrecisionAttr, "::mlir::pto::DivPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandMulOp: PTO_TOp<"trowexpandmul", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMUL: Row-wise broadcast multiply with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandSubOp: PTO_TOp<"trowexpandsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDSUB: Row-wise broadcast subtract: subtract a per-row scalar vector src1 from each row of src0.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandAddOp: PTO_TOp<"trowexpandadd", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDADD: Row-wise broadcast add: add a per-row scalar vector src1 to each row of src0.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandExpdifOp: PTO_TOp<"trowexpandexpdif", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDEXPDIF: Row-wise broadcast expdif with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandMaxOp: PTO_TOp<"trowexpandmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMAX: Row-wise broadcast max with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowExpandMinOp: PTO_TOp<"trowexpandmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMIN: Row-wise broadcast min with per-row scalar vector.";
  let description = [{
    pto-isa has overloads with/without tmp; on A5 the tmp operand is accepted as a placeholder and does not add extra shape constraints.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowMaxOp: PTO_TOp<"trowmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMAX: Reduce each row by taking the maximum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowArgMaxOp: PTO_TOp<"trowargmax", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWARGMAX: Reduce each row to the column index of the maximum element.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowMinOp: PTO_TOp<"trowmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMIN: Reduce each row by taking the minimum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TRowArgMinOp: PTO_TOp<"trowargmin", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWARGMIN: Reduce each row to the column index of the minimum element.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWSUM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowSumOp: PTO_TOp<"trowsum", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWSUM: Reduce each row by summing across columns.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TInterleaveOp: PTO_TOp<"tinterleave", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TINTERLEAVE: Interleave two source tiles into two destination tiles.";
  let description = [{
    Interleaves src0 and src1 and splits the result into dst0 and dst1.
    The operation uses the same valid shape for all four tiles.
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst0,
    PTODpsType:$dst1
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:`
      qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst0 `,` $dst1 `:`
      qualified(type($dst0)) `,` qualified(type($dst1)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), 2, 2);
    }
  }];
}

def TDeInterleaveOp: PTO_TOp<"tdeinterleave", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TDEINTERLEAVE: De-interleave one or two source tiles into two destination tiles.";
  let description = [{
    The two-source form de-interleaves two source tiles with matching valid
    shapes. The single-source form de-interleaves one source tile into two
    destination tiles whose valid column count is half the source count.
  }];

  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Variadic<PTODpsType>:$dsts
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstsMutable();
    }
    ::mlir::Value getSrc0() { return getSrcs().front(); }
    ::mlir::Value getSrc1() {
      return getSrcs().size() == 2 ? getSrcs()[1] : ::mlir::Value();
    }
    ::mlir::Value getDst0() { return getDsts().front(); }
    ::mlir::Value getDst1() { return getDsts()[1]; }
  }];
}

def TRowProdOp: PTO_TOp<"trowprod", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWPROD: Reduce each row by multiplying across columns.";

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRsqrtOp: PTO_TOp<"trsqrt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRSQRT: Elementwise reciprocal square root.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$tmp,
    DefaultValuedAttr<PTO_RsqrtPrecisionAttr, "::mlir::pto::RsqrtPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSCATTER TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TScatterOp: PTO_TOp<"tscatter", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSCATTER: Scatter elements of a source tile into a destination tile using indices or a mask pattern.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$indexes,
    OptionalAttr<StrAttr>:$axis,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    bool hasIndexForm() { return static_cast<bool>(getIndexes()); }
    bool hasMaskForm() { return static_cast<bool>(getMaskPatternAttr()); }
    bool hasAxis() { return static_cast<bool>(getAxisAttr()); }
    ::mlir::pto::PIPE getPipe() {
      if (hasMaskForm())
        return ::mlir::pto::PIPE::PIPE_V;

      // NOTE:
      // - On A2/A3, indexed TSCATTER lowers to a scalar UB loop (PIPE_S).
      // - On A5, pto-isa TSCATTER is vector scatter (PIPE_V).
      // We accept either `pto.target_arch=a5` or A5 `pto.device-spec`.
      auto isA5Target = [&]() -> bool {
        auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
        if (!moduleOp)
          return false;

        if (auto arch =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
          if (arch.getValue().equals_insensitive("a5"))
            return true;
        }

        if (auto spec =
                moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
          auto s = spec.getValue();
          if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
            return true;
        }

        return false;
      };

      return isA5Target() ? ::mlir::pto::PIPE::PIPE_V
                          : ::mlir::pto::PIPE::PIPE_S;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSEL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSelOp: PTO_TOp<"tsel", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSEL: Select between two tiles using a mask tile (per-element selection).";
  let description = [{
    DPS form (pto-isa TSEL requires tmp): ins(%mask, %src0, %src1, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSELS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSelSOp: PTO_TOp<"tsels", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSELS: Select one of two source tiles using a scalar selectMode (global select).";
  let description = [{
    DPS form (pto-isa TSELS requires tmp): ins(%mask, %src, %tmp, %scalar) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TShlOp: PTO_TOp<"tshl", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHL: Elementwise shift-left of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TShrOp: PTO_TOp<"tshr", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHR: Elementwise shift-right of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHLS/TSHRS TBDPS: shift-left/right by scalar)
//===----------------------------------------------------------------------===//

def TShlSOp : PTO_TOp<"tshls", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHLS: Elementwise shift-left of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def TShrSOp : PTO_TOp<"tshrs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHRS: Elementwise shift-right of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSORT32 TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSort32Op: PTO_TOp<"tsort32", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSORT32: Sort fixed-size 32-element blocks using explicit input indices.";
  let description = [{
    DPS form matching pto-isa overloads: (dst, src, idx) or (dst, src, idx, tmp).
    In PTO IR, `idx` is modeled as an input operand in `ins(...)`, while `dst`
    remains the sole DPS init/output buffer in `outs(...)`.
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSqrtOp: PTO_TOp<"tsqrt", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,  
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSQRT: Elementwise square root.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_SqrtPrecisionAttr, "::mlir::pto::SqrtPrecision::Default">:$precisionType
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TSubOp: PTO_TOp<"tsub", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUB: Elementwise subtract of two tiles.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubCOp: PTO_TOp<"tsubc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBC: Elementwise ternary op: src0 - src1 + src2.";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$src2,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $src2 `:`
            qualified(type($src0)) `,` qualified(type($src1)) `,` qualified(type($src2)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubSOp: PTO_TOp<"tsubs", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBS: Elementwise subtract a scalar from a tile.";

  let arguments = (ins
    PTODpsType:$src,
    ScalarType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBSC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSubSCOp: PTO_TOp<"tsubsc", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBSC: Elementwise fused op: src0 - scalar + src1.";

  let arguments = (ins
    PTODpsType:$src0,
    ScalarType:$scalar,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $scalar `,` $src1 `:`
            qualified(type($src0)) `,` type($scalar) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TXorSOp: PTO_TOp<"txors", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXORS: Elementwise bitwise XOR of a tile and a scalar.";
  let description = [{
    DPS form (pto-isa TXORS requires tmp): ins(%src, %scalar, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src,
    AnySignlessInteger:$scalar,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TXorOp: PTO_TOp<"txor", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXOR: Elementwise bitwise XOR of two tiles.";
  let description = [{
    DPS form:
      without tmp: pto.txor ins(%src0, %src1) outs(%dst)
      with tmp:    pto.txor ins(%src0, %src1, %tmp) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    Optional<PTODpsType>:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}

def TPrintOp: PTO_TOp<"tprint", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRINT: Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code.";
  let description = [{
    pto-isa overloads support `TPRINT(src)` and `TPRINT<Format>(src, tmp)`.
    The optional tmp operand is used when printing Mat/Acc tiles through a
    scratch GlobalTensor, while Vec tiles and GlobalTensors can be printed
    without tmp.
  }];

  let arguments = (ins
    PTODpsType:$src,
    Optional<PTODpsType>:$tmp,
    OptionalAttr<PTO_PrintFormatAttr>:$printFormat
  );

  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
 }

#endif // MLIR_DIALECT_PTO_IR_PTOOPS
