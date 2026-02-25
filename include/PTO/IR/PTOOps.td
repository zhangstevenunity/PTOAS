//===- PTOOps.td - Pattern descriptor operations -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "Tensor or Memref">;

def PTODpsType :
  AnyTypeOf<[AnyRankedTensor, AnyMemRef, PartitionTensorViewType, TileBufType]>;

def PtrOrMemRef :
  AnyTypeOf<[PtrType, AnyMemRef], "Ptr or MemRef">;

def ScalarPtrOrMemRef :
  TypeConstraint<
    CPred<"::mlir::pto::isScalarPtrOrMemRef($_self)">,
    "Ptr or MemRef in GM">;

//===----------------------------------------------------------------------===//
// Op Class
//===----------------------------------------------------------------------===//
class PTO_TOp<string mnemonic, list<Trait> traits = []>
    : Op<PTO_Dialect, mnemonic, !listconcat(traits, [ ])>;

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

//===----------------------------------------------------------------------===//
// Scalar pointer load/store
//===----------------------------------------------------------------------===//

def LoadScalarOp : PTO_Op<"load_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Load a single scalar element from a pointer at offset.";

  let arguments = (ins
    ScalarPtrOrMemRef:$ptr,
    Index:$offset
  );

  let results = (outs AnyType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $ptr `,` $offset `:` type($ptr) `,` type($offset) `)`
    `outs` `:` type($value)
    attr-dict
  }];
}

def StoreScalarOp : PTO_Op<"store_scalar", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Store a single scalar element to a pointer at offset.";

  let arguments = (ins
    ScalarPtrOrMemRef:$ptr,
    Index:$offset,
    AnyType:$value
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $ptr `,` $offset `,` $value `:` type($ptr) `,` type($offset) `,` type($value) `)`
    attr-dict
  }];
}

def MakeTensorViewOp : PTO_Op<"make_tensor_view", [AttrSizedOperandSegments]> {
  let summary = "Wrap a pointer as a tensor_view descriptor (no allocation, no copy).";

  let arguments = (ins
    AnyType:$ptr,
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
def PartitionViewOp : PTO_Op<"partition_view", [AttrSizedOperandSegments]> {
  let summary = "Partition a tensor view into a smaller logical view (logical slicing).";
  let description = [{
    Captures a specific calculation region from a large view.
    It carries offsets (where to read) and sizes (how much to read).
    
    IR Example:
      %1 = pto.partition_view %0, offsets=[...], sizes=[...] 
           : !pto.tensor_view -> !pto.partition_tensor_view
  }];

  let arguments = (ins
    TensorViewType:$source,    // 输入: 物理大底座 (MakeTensorViewOp 的结果)
    Variadic<Index>:$offsets,  // 动态 offsets
    Variadic<Index>:$sizes     // 动态 sizes
  );

  let results = (outs PartitionTensorViewType:$result); // 输出: 逻辑切片

  let assemblyFormat = [{
    $source `,` `offsets` `=` `[` $offsets `]` `,` `sizes` `=` `[` $sizes `]`
    attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

def AllocTileOp : PTO_Op<"alloc_tile", [AttrSizedOperandSegments]> {
  let summary = "Allocates a tile buffer (logical buffer).";

  let arguments = (ins
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col
  );

  let results = (outs TileBufType:$result);

  let assemblyFormat = [{
    (`valid_row` `=` $valid_row^)? 
    (`valid_col` `=` $valid_col^)?
     attr-dict `:` qualified(type($result))
  }];

  let extraClassDeclaration = [{
    ::mlir::LogicalResult verify();
  }];
}


// ============================================================================
// BindTileOp: 将 Config 和 Valid Dims 绑定到 MemRef 上
// ============================================================================
def BindTileOp : PTO_Op<"bind_tile", [
    Pure,
    AttrSizedOperandSegments
    // 允许输入 offset:0 -> 输出 offset:?
  ]> {
  let summary = "Binds metadata and implicitly casts layout";
  let description = [{
    Wraps a memref with PTO metadata (valid dimensions and config).
  }];

  // [修改] 以前是 Variadic<Index>:$valid_dims
  // 现在改为明确的 Optional Row/Col，与 PointerCastOp 保持一致
  let arguments = (ins
    AnyMemRef:$source,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col,
    TileBufConfigAttr:$config
  );

  let results = (outs AnyMemRef:$result);

  // [修改] assemblyFormat: 去掉 valid_dims，改为 ($row, $col)?
  let assemblyFormat = [{
    $source (`,` $valid_row^ `,` $valid_col)? attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];
}

def PTO_SubsetOp : PTO_Op<"subset", [
    Pure,
    ViewLikeOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface> // 启用 C++ 推导
  ]> {
  
  let summary = "Create a strided view (subset) from a parent tile.";
  let description = [{
    Creates a view into the source tile.
    - Result Shape: Defined by static `sizes`.
    - Result Strides: Inherited from `source`.
    - Result Offset: Represented as multi-dimensional symbols (s0, s1...) in the layout map.
  }];

  let arguments = (ins
    TileBufType:$source,
    Variadic<Index>:$offsets, // 运行时动态偏移 [i, j]
    I64ArrayAttr:$sizes       // 静态形状 [32, 32]
  );

  let results = (outs TileBufType:$result);
  let hasVerifier = 1;

  // 语法示例: %sub = pto.subset %src[%i, %j] sizes [32, 32] : !type
  // 注意：没有 -> qualified(type($result))
  let assemblyFormat = [{
    $source `[` $offsets `]` `sizes` $sizes attr-dict `:` qualified(type($source))
  }];

  // [新增] 显式实现 ViewLikeOpInterface 缺失的方法
  let extraClassDeclaration = [{
    // 接口要求 getViewSource，我们转发给自动生成的 getSource
    ::mlir::Value getViewSource() { return getSource(); }
    
    // ViewLikeOpInterface 可能还需要 getOffsets (如果 Variadic 不自动匹配)
    // 但通常 Variadic<Index>:$offsets 会生成 getOffsets()，这应该没问题。
    // 如果后续报 getOffsets 错，也可以在这里加。
  }];
}

//===----------------------------------------------------------------------===//
// DMA Ops
//===----------------------------------------------------------------------===//
// -------------------------
// DPS versions in tile world
// -------------------------
def TLoadOp  : PTO_TOp<"tload", [
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
    PartitionTensorViewType:$src,
    TileBufType:$dst,
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
  }];
}

def TStoreOp: PTO_TOp<"tstore", [
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
    TileBufType:$src,
    PartitionTensorViewType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
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
      if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(getSrc().getType())) {
        if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                tb.getMemorySpace())) {
          if (as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC)
            return ::mlir::pto::PIPE::PIPE_FIX;
        }
      }
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
  }];
}

// DPS addf for post-bufferization / EmitPTOManual lowering in tile world.
//   tensor:  %res = pto.addf_dps ins(%a,%b) outs(%dst) -> tensor<...>?
//   memref:  pto.addf_dps ins(%a,%b) outs(%dst)
class PTO_TBinaryFloatOp<string mnemonic, list<Trait> traits = []>
    : PTO_TOp<mnemonic, traits> {
  
  let arguments = (ins 
    PTODpsType:$lhs, 
    PTODpsType:$rhs, 
    PTODpsType:$dst
  );
  
  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];
}

def TTransOp : PTO_TOp<"ttrans", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix trans operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tile buf:  pto.ttrans ins(%src %tmp) outs(%dst)
  }];
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

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
//===----------------------------------------------------------------------===//
// TMATMUL_BIAS / TMATMUL_MX family
//   - DPS: AnyMemRef
//   - TOp: PTODpsType
//   - All in ins/outs form
//===----------------------------------------------------------------------===//

// -------------------------
// DPS versions (AnyMemRef)
// -------------------------

// pto.matmul.bias_dps ins(%a, %b, %bias) outs(%dst) -> tensor?
def MatmulBiasDpsOp : PTO_DpsOp<"matmul.bias_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul with bias add (DPS, ins/outs).";

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$b,
    AnyMemRef:$bias,
    AnyMemRef:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` type($a) `,` type($b) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TMATMUL_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.matmul.mx_dps ins(%a, %a_scale, %b, %b_scale) outs(%dst) -> tensor?
def MatmulMxDpsOp : PTO_DpsOp<"matmul.mx_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx (with scaling tiles) (DPS, ins/outs).";

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$a_scale,
    AnyMemRef:$b,
    AnyMemRef:$b_scale,
    AnyMemRef:$dst);

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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.matmul.mx.acc_dps ins(%c_in, %a, %a_scale, %b, %b_scale) outs(%dst) -> tensor?
def MatmulMxAccDpsOp : PTO_DpsOp<"matmul.mx.acc_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx accumulate (DPS, ins/outs).";

  let arguments = (ins
    AnyMemRef:$c_in,
    AnyMemRef:$a,
    AnyMemRef:$a_scale,
    AnyMemRef:$b,
    AnyMemRef:$b_scale,
    AnyMemRef:$dst);

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
    static StringRef getIntrinsicName() { return "TMATMUL_MX_ACC"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// pto.matmul.mx.bias_dps ins(%a,%a_scale,%b,%b_scale,%bias) outs(%dst) -> tensor?
def MatmulMxBiasDpsOp : PTO_DpsOp<"matmul.mx.bias_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul mx with bias (DPS, ins/outs).";

  let arguments = (ins
    AnyMemRef:$a,
    AnyMemRef:$a_scale,
    AnyMemRef:$b,
    AnyMemRef:$b_scale,
    AnyMemRef:$bias,
    AnyMemRef:$dst);

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
    static StringRef getIntrinsicName() { return "TMATMUL_MX_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}


// -------------------------
// Tile-world TOp versions (PTODpsType)
// -------------------------

def TMatmulBiasOp : PTO_TOp<"tmatmul.bias", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Matmul with bias add (tile world, ins/outs).";

  let arguments = (ins
    TileBufType :$a,
    TileBufType :$b,
    TileBufType :$bias,
    TileBufType :$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);
  //let hasVerifier = 1;

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
  }];
}

// pto.tmatmul.mx ins(%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxOp : PTO_TOp<"tmatmul.mx", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Matmul mx (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$a_scale,
    PTODpsType:$b,
    PTODpsType:$b_scale,
    PTODpsType:$dst);

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
  }];
}

// pto.tmatmul.mx.acc ins(%c_in,%a,%a_scale,%b,%b_scale) outs(%dst) -> tensor?
def TMatmulMxAccOp : PTO_TOp<"tmatmul.mx.acc", [
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
    PTODpsType:$dst);

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
    static StringRef getIntrinsicName() { return "TMATMUL_MX_ACC"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M;
    }
  }];
}

// pto.tmatmul.mx.bias ins(%a,%a_scale,%b,%b_scale,%bias) outs(%dst)  -> tensor?
def TMatmulMxBiasOp : PTO_TOp<"tmatmul.mx.bias",[
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
    static StringRef getIntrinsicName() { return "TMATMUL_MX_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
  }];
}

def TMatmulOp : PTO_TOp<"tmatmul", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    TileBufType:$lhs,
    TileBufType:$rhs,
    Optional<TileBufType>:$bias,
    TileBufType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    ( `bias` `=` $bias^ `:` qualified(type($bias)) )?
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
    static StringRef getOpName() { return "matmul_dps"; }
    ::mlir::pto::PIPE getPipe() {
      return ::mlir::pto::PIPE::PIPE_M; 
    }
  }];
}

def TMatmulAccOp : PTO_TOp<"tmatmul.acc", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    TileBufType:$acc_in,
    TileBufType:$lhs,
    TileBufType:$rhs,
    TileBufType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

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
  }];
}

def TGemvOp : PTO_TOp<"tgemv", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication operation (optional bias), destination-style (DPS) in tile world.";

  let arguments = (ins
    TileBufType:$lhs,
    TileBufType:$rhs,
    TileBufType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

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
  }];
}

def TGemvAccOp : PTO_TOp<"tgemv.acc", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix-vector multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    TileBufType:$acc_in,
    TileBufType:$lhs,
    TileBufType:$rhs,
    TileBufType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

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
  }];
}

def TGemvBiasOp : PTO_TOp<"tgemv.bias", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "GEMV with bias add (tile world, ins/outs).";

  let arguments = (ins
    TileBufType :$a,
    TileBufType :$b,
    TileBufType :$bias,
    TileBufType :$dst
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

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
  }];
}

def TMovOp  : PTO_TOp<"tmov", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";
  
  let arguments = (ins
    TileBufType:$src,
    TileBufType:$dst
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
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
      auto srcTy = llvm::dyn_cast<::mlir::pto::TileBufType>(getSrc().getType());
      auto dstTy = llvm::dyn_cast<::mlir::pto::TileBufType>(getDst().getType());
      if (!srcTy || !dstTy) {
        return ::mlir::pto::PIPE::PIPE_V;
      }
      auto srcAS = llvm::dyn_cast<::mlir::pto::AddressSpaceAttr>(srcTy.getMemorySpace());
      auto dstAS = llvm::dyn_cast<::mlir::pto::AddressSpaceAttr>(dstTy.getMemorySpace());
      if (!srcAS || !dstAS) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      const auto s = srcAS.getAddressSpace();
      const auto d = dstAS.getAddressSpace();

      // UB -> UB copy is vector pipe.
      if (s == ::mlir::pto::AddressSpace::VEC && d == ::mlir::pto::AddressSpace::VEC) {
        return ::mlir::pto::PIPE::PIPE_V;
      }

      // MAT -> L0 (Left/Right/Bias/Scaling) and ACC -> MAT are MTE1 moves.
      if ((s == ::mlir::pto::AddressSpace::MAT &&
           (d == ::mlir::pto::AddressSpace::LEFT || d == ::mlir::pto::AddressSpace::RIGHT ||
            d == ::mlir::pto::AddressSpace::BIAS || d == ::mlir::pto::AddressSpace::SCALING)) ||
          (s == ::mlir::pto::AddressSpace::ACC && d == ::mlir::pto::AddressSpace::MAT)) {
        return ::mlir::pto::PIPE::PIPE_MTE1;
      }

      // Fallback: treat as vector pipe (safe default for most intra-domain moves).
      return ::mlir::pto::PIPE::PIPE_V;
    }
  }];
}

// -------------------------
// DPS versions (with suffix)
// -------------------------
def LoadDpsOp : PTO_DpsOp<"load_dps", [
  AttrSizedOperandSegments,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO data load operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tensor:  %res = pto.load_dps ins(%src) outs(%dst) -> tensor<...>  (optional result)
      memref:  pto.load_dps ins(%src) outs(%dst)
  }];

  let arguments = (ins 
    PTODpsType:$src,
    PTODpsType:$dst,
    OptionalAttr<PTO_PadModeAttr>:$pad_mode,
    Optional<AnyType>:$pad_value,
    Optional<Index>:$left_padding_num,
    Optional<AnyType>:$right_padding_num,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$init_out_buffer,
    Optional<AnyType>:$init_condition);

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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    static StringRef getOpName() { return "pto_load"; }
    ShapedType getSrcOperandType() {
      return cast<ShapedType>(getSrc().getType());
    }
    ShapedType getDstOperandType() {
      return cast<ShapedType>(getDst().getType());
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def CopyOp : PTO_DpsOp<"copy",[
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO data copy operation";
  let description = [{
    Copy the data between local memory hierarchies.
    Currently support:
      - UB to UB
      - UB to L1 (for Ascend910_95 series)

    Examples:
    ```mlir
    pto.hir.copy ins(%src : memref<16x16xf16, #pto.address_space<ub>>) outs(%dst : memref<16x16xf16, #pto.address_space<ub>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `pad_mode` is not set, `src` and `dst` shape should be the same.
    - Only support left padding.
    - `pad_value` should have the same element type as `src` and `dst`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<PTO_PadModeAttr>:$pad_mode,
                       Optional<AnyType>:$pad_value
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$result_tensor, "Value":$src, "Value":$dst)>,
    OpBuilder<(ins "TypeRange":$result_tensor, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`pad_mode` `=` $pad_mode^)?
    (`pad_value` `=` $pad_value^ `:` type($pad_value))?
    (`->` qualified(type($result_tensor) )^)?
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe();
    static StringRef getOpName() { return "pto_copy"; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

def TransDpsOp : PTO_DpsOp<"trans_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO matrix transpose operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tile buf:  pto.trans_dps ins(%src) outs(%dst)
  }];
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TTRANS is a vector compute op (V pipe). The inserted sync pass relies on
    // this pipe classification to build correct event dependencies.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def StoreDpsOp : PTO_DpsOp<"store_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "PTO data store operation (destination-style, DPS).";
  let description = [{
    DPS form:
      tensor:  %res = pto.store_dps ins(%src) outs(%dst) -> tensor<...>  (optional result)
      memref:  pto.store_dps ins(%src) outs(%dst)
  }];

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst);

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];
  
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() {
      // ACC (L0C) -> GM stores are implemented via FIX-pipe instructions in
      // pto-isa; other domains use MTE3.
      if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(getSrc().getType())) {
        if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                tb.getMemorySpace())) {
          if (as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC)
            return ::mlir::pto::PIPE::PIPE_FIX;
        }
      } else if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(getSrc().getType())) {
        if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                mr.getMemorySpace())) {
          if (as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC)
            return ::mlir::pto::PIPE::PIPE_FIX;
        }
      }
      return ::mlir::pto::PIPE::PIPE_MTE3;
    }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];
}

// -------------------------
// non-DPS versions (no suffix)
//   NOTE: here we keep them permissive so that:
//     - old IR: tile_view -> tile
//     - new IR: memref -> tensor
//   can BOTH parse. Strict checks are done in PTO.cpp only for memref/tensor case.
// -------------------------

def LoadOp : PTO_Op<"load", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PTO load (polymorphic): tile_view->tile OR memref->tensor.";
  
  let arguments = (ins 
    AnyType:$src);
  
  let results   = (outs AnyType:$result);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

}

def StoreOp : PTO_Op<"store"> {
  let summary = "PTO store (polymorphic): (tile_view,tile)->() OR (tensor,memref)->().";
  
  let arguments = (ins 
    AnyType:$src, // Value to store
    AnyType:$dst // Destination memory
  );
  
  let results = (outs);
  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Arith Ops
//===----------------------------------------------------------------------===//

def AddFOp : PTO_Op<"addf", [
    AllTypesMatch<["lhs", "rhs", "result"]>, 
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PTO addf (non-DPS).";

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs 
    attr-dict 
    `:` `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

// Non-DPS transpose (your desired IR uses this):
//   %tb = pto.trans %ta : tensor<32x32xf32>
def TransOp : PTO_Op<"trans", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PTO transpose (non-DPS).";
  let arguments = (ins
    AnyType:$src,
    AnyType:$tmp);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $src `,` $tmp attr-dict `:` qualified(type($src)) `,` qualified(type($tmp)) `->` qualified(type($result))
  }];


  let hasVerifier = 1;
}

// DPS addf for post-bufferization / EmitPTOManual lowering.
//    tensor:  %res = pto.addf_dps ins(%a,%b) outs(%dst) -> tensor<...>?
//    memref:  pto.addf_dps ins(%a,%b) outs(%dst)
class PTO_BinaryFloatDpsOp<string mnemonic, list<Trait> traits = []>
    : PTO_DpsOp<mnemonic, traits> {
  
  let arguments = (ins 
    PTODpsType:$lhs, 
    PTODpsType:$rhs, 
    PTODpsType:$dst);
  
  let results = (outs Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)` 
    attr-dict
    (`->` qualified(type($result))^)?
  }];
}

def AddFDpsOp : PTO_BinaryFloatDpsOp<"addf_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]>{
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    static StringRef getOpName() { return "pto_add"; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
        return getDstMutable();
    }
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$dst), [{
      build($_builder, $_state, TypeRange{}, lhs, rhs, dst);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Pointer Cast Op (existing)
//===----------------------------------------------------------------------===//

def PointerCastOp : PTO_Op<"pointer_cast", [AttrSizedOperandSegments, Pure]> {
  let summary = "Casts an integer address to a MemRef with optional valid dims";
  
  // 参数定义 (保持 Optional)
  let arguments = (ins 
    Variadic<I64>:$addrs,
    Optional<Index>:$valid_row,
    Optional<Index>:$valid_col,
    OptionalAttr<TileBufConfigAttr>:$config
  );
  
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$result);

  // Assembly Format (去掉了 [])
  let assemblyFormat = [{
    `(` $addrs `)` ($valid_row^ `,` $valid_col)? attr-dict `:` qualified(type($result))
  }];

  // Builder (修复 addOperand -> addOperands)
  let builders = [
    OpBuilder<(ins "Type":$result, "ValueRange":$addrs, "Value":$vRow, "Value":$vCol, "Attribute":$config), [{
      $_state.addTypes(result);
      $_state.addOperands(addrs);
      // [关键修复] addOperand -> addOperands
      if (vRow) $_state.addOperands(vRow);
      if (vCol) $_state.addOperands(vCol);
      if (config) $_state.addAttribute("config", config);
      
      int32_t addrsSize = addrs.size();
      int32_t vRowSize = vRow ? 1 : 0;
      int32_t vColSize = vCol ? 1 : 0;
      $_state.addAttribute("operandSegmentSizes", 
        $_builder.getDenseI32ArrayAttr({addrsSize, vRowSize, vColSize}));
    }]>
  ];
}

// =============================================================================
// System/Runtime Query Ops
// =============================================================================

def GetBlockIdxOp : PTO_Op<"get_block_idx", [Pure]> {
  let summary = "Get the current block index (core ID).";
  let description = [{
    Returns the linear index of the current compute unit (CUBE Core) within the task.
    The return value is in the range [0, BlockNum - 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockIdxOp : PTO_Op<"get_subblock_idx", [Pure]> {
  let summary = "Get the current vector core ID.";
  let description = [{
    Returns the ID of the current compute unit (Vector Core).
    The return value is in the range [0, 1].
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetBlockNumOp : PTO_Op<"get_block_num", [Pure]> {
  let summary = "Get the total number of blocks (cores).";
  let description = [{
    Returns the total number of compute units (Blocks) configured for the current task.
  }];

  let arguments = (ins);
  let results = (outs I64:$result);

  let assemblyFormat = "attr-dict";
}

def GetSubBlockNumOp : PTO_Op<"get_subblock_num", [Pure]> {
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

def PTO_RecordEventOp : PTO_Op<"record_event"> {
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

def PTO_WaitEventOp : PTO_Op<"wait_event"> {
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
def PTO_BarrierSyncOp : PTO_Op<"barrier_sync"> {
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

//===----------------------------------------------------------------------===//
// Synchronization Ops
//===----------------------------------------------------------------------===//

def PTO_SetFlagOp : PTO_Op<"set_flag"> {
  let summary = "Set synchronization flag between pipes";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def PTO_WaitFlagOp : PTO_Op<"wait_flag"> {
  let summary = "Wait for synchronization flag";
  let arguments = (ins
    PTO_PipeAttr:$src_pipe,
    PTO_PipeAttr:$dst_pipe,
    PTO_EventAttr:$event_id
  );
  let results = (outs);
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def PTO_SyncSetOp : PTO_Op<"sync.set"> {
  let summary = "Set a synchronization signal (trigger) between cube and vector.";
  let description = [{
    Sets a synchronization signal on the specified pipeline stage.
    Corresponds to `ffts_cross_core_sync` (A3) or `set_intra_block` (A5).
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    I32Attr:$event_id
  );

  let assemblyFormat = "$pipe `,` $event_id attr-dict";
}

def PTO_SyncWaitOp : PTO_Op<"sync.wait"> {
  let summary = "Wait for a synchronization signal (barrier) between cube and vector.";
  let description = [{
    Waits for a synchronization signal on the specified pipeline stage.
    Corresponds to `wait_flag_dev` (A3) or `wait_intra_block` (A5).
  }];

  let arguments = (ins
    PTO_PipeAttr:$pipe,
    I32Attr:$event_id
  );

  let assemblyFormat = "$pipe `,` $event_id attr-dict";
}

def BarrierOp : PTO_Op<"barrier"> {
  let summary = "Intra-pipeline memory barrier";
  let arguments = (ins PTO_PipeAttr:$pipe);
  let assemblyFormat = "$pipe attr-dict";
}

//===----------------------------------------------------------------------===//
// Matmul Ops
//===----------------------------------------------------------------------===//

def MatmulDpsOp : PTO_DpsOp<"matmul_dps", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO matrix multiplication operation (optional bias), destination-style (DPS).";

  let arguments = (ins
    PTODpsType:$lhs, // L0A Buffer
    PTODpsType:$rhs, // L0B Buffer
    Optional<PTODpsType>:$bias, // L1 Bias (Optional)
    PTODpsType:$dst , // L0C Buffer (Accumulator)
 
    // 硬件关键属性
    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b
  );

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    ( `bias` `=` $bias^ `:` qualified(type($bias)) )?
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let extraClassDeclaration = [{
   ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    static StringRef getOpName() { return "matmul_dps"; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

def MatmulOp : PTO_Op<"matmul", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PTO matrix multiplication operation (optional bias), non-destination-style.";

  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    Optional<AnyType>:$bias
  );

  let results = (outs
    AnyType:$result
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    ( `bias` `=` $bias^ `:` qualified(type($bias)) )?
    attr-dict
    `->` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// Matmul Acc Ops
//===----------------------------------------------------------------------===//

def MatmulAccOp : PTO_Op<"matmul_acc", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "PTO matrix multiplication accumulate operation, non-destination-style (no bias).";

  let arguments = (ins
    AnyType:$acc_in,
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs AnyType:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $acc_in `,` $lhs `,` $rhs `:` qualified(type($acc_in) ) `,` qualified(type($lhs)) `,` qualified(type($rhs)) `)`
    attr-dict
    `->` qualified(type($result))
  }];
}

def MatmulAccDpsOp : PTO_DpsOp<"matmul_acc_dps", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO matrix multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// Gemv Ops (Matrix-Vector Multiplication)
//===----------------------------------------------------------------------===//

def GemvDpsOp : PTO_DpsOp<"gemv_dps", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO matrix-vector multiplication operation (optional bias), destination-style (DPS).";

  let arguments = (ins
    PTODpsType:$lhs, // L0A Buffer
    PTODpsType:$rhs, // L0B Buffer (vector)
    PTODpsType:$dst // L0C Buffer (Accumulator)
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
   ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    static StringRef getOpName() { return "gemv_dps"; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

def GemvAccDpsOp : PTO_DpsOp<"gemv.acc_dps", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "PTO matrix-vector multiplication accumulate operation, destination-style (DPS, no bias).";

  let arguments = (ins
    PTODpsType:$acc_in,
    PTODpsType:$lhs,
    PTODpsType:$rhs,
    PTODpsType:$dst
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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def GemvBiasDpsOp : PTO_DpsOp<"gemv.bias_dps", [
    OpPipeInterface,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "GEMV with bias add (DPS, ins/outs).";

  let arguments = (ins
    PTODpsType:$a,
    PTODpsType:$b,
    PTODpsType:$bias,
    PTODpsType:$dst);

  let results = (outs Optional<AnyRankedTensor>:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $a `,` $b `,` $bias `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($bias)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    (`->` qualified(type($result))^)?
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "TGEMV_BIAS"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_M; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// ===----------------------------------------------------------------------===//
// Move Ops (For CV Separation)
// ===----------------------------------------------------------------------===//

// High-Level Mov: 值语义
def MovOp : PTO_Op<"mov", [
    SameOperandsAndResultType, // 输入输出类型相同 (!pto.tile -> !pto.tile)
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Move data between Cube and Vector domains.";
  
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
}

def MovDpsOp : PTO_DpsOp<"mov_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";
  
  let arguments = (ins 
    PTODpsType:$src,
    PTODpsType:$dst);

  let results = (outs
    Optional<AnyRankedTensor>:$result
  );

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
    ( `->` qualified(type($result))^ )?
  }];

  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
    ::mlir::pto::PIPE getPipe();
  }];
}
//===----------------------------------------------------------------------===//
// FFT Configuration Operation
//===----------------------------------------------------------------------===//

def SetFFTsOp : PTO_Op<"set_ffts", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Set FFTS/flags pointer for runtime (side-effecting).";
  let arguments = (ins AnyMemRef:$ffts);
  let results = (outs);

  let assemblyFormat = "$ffts attr-dict `:` type($ffts)";

  let hasVerifier = 1;
}
// ---- DPS version (no 't' prefix) ----
// pto.mgather_dps ins(%mem, %idx) outs(%dst) [...] -> tensor?
def MGatherDpsOp : PTO_DpsOp<"mgather_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather-load elements from global memory into a tile using per-element indices (DPS).";

  let arguments = (ins
    AnyMemRef:$mem,
    PTODpsType:$idx,
    PTODpsType:$dst);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mem `,` $idx `:` type($mem) `,` type($idx) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MGATHER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

// ---- tile-world TOp version (with 't') ----
// pto.tmgather ins(%mem, %idx) outs(%dst) [ ...] -> tensor?
def TMGatherOp : PTO_TOp<"mgather", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather-load elements from memory into a tile using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$mem,
    PTODpsType:$idx,
    PTODpsType:$dst);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mem `,` $idx `:` type($mem) `,` type($idx) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MGATHER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE2; }
  }];
}

//===----------------------------------------------------------------------===//
// MSCATTER (DPS)  +  TMSCATTER (tile world TOp)
//===----------------------------------------------------------------------===//

// ---- DPS version (no 't' prefix) ----
// pto.mscatter_dps ins(%src, %idx) outs(%mem) [ ...]
def MScatterDpsOp : PTO_DpsOp<"mscatter_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter-store elements from a tile into global memory using per-element indices (DPS).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    AnyMemRef:$mem);

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` type($idx) `)`
    `outs` `(` $mem `:` type($mem) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MSCATTER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }

    // DestinationStyleOpInterface hook: declare explicitly for THIS op.
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return getMemMutable();
    }
  }];
}
//===----------------------------------------------------------------------===//
// GETVAL (DPS)  +  TGETVAL (tile world TOp)
//   Read a single element into a scalar.
//===----------------------------------------------------------------------===//

// ---- DPS version ----
// pto.getval_dps ins(%src, %offset) outs(%dst) : memref<...>, index -> T
def GetValDpsOp : PTO_Op<"getval_dps", []> {
  let summary = "Read a single element from src at offset into a scalar (ins/outs).";

  let arguments = (ins
    AnyMemRef:$src,
    Index:$offset
  );

  let results = (outs AnyType:$dst);

  let hasVerifier = 1;


  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` qualified(type($dst) )
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "GETVAL"; }
  }];
}
//===----------------------------------------------------------------------===//
// SETVAL (DPS) + TSETVAL (tile world TOp)
//   Write a scalar into a single element at offset.
//===----------------------------------------------------------------------===//

// ---- DPS version ----
// pto.setval_dps ins(%offset, %val) outs(%dst) : memref<...>, index, T
def SetValDpsOp : PTO_DpsOp<"setval_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Write a scalar value into a single element of dst at offset (DPS, ins/outs).";

  let arguments = (ins
    AnyMemRef:$dst,      // written memory (outs)
    Index:$offset,
    AnyType:$val
  );

  let results = (outs);

  let hasVerifier = 1;

  // Keep ins/outs style: put offset+val in ins, dst in outs.
  let assemblyFormat = [{
    `ins` `(` $offset `,` $val `:` type($offset) `,` type($val) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "SETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }

    // DestinationStyleOpInterface hook (avoid relying on base class injection)
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

// ---- tile-world TOp version ----
// pto.tsetval ins(%offset, %val) outs(%dst) : PTODpsType, index, T
def TSetValOp : PTO_TOp<"tsetval", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Write a scalar value into a single element of dst at offset (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$dst,
    Index:$offset,
    AnyType:$val
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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
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

  let results = (outs AnyType:$dst);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $offset `:` qualified(type($src)) `,` type($offset) `)`
    `outs` `:` qualified(type($dst) )
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "GETVAL"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}


// ---- tile-world TOp version (with 't') ----
// pto.tmscatter ins(%src, %idx) outs(%mem) [ ...]
def TMScatterOp : PTO_TOp<"mscatter", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Scatter-store elements from a tile into memory using per-element indices (tile world, ins/outs).";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$idx,
    PTODpsType:$mem               // outs target
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $idx `:` qualified(type($src)) `,` type($idx) `)`
    `outs` `(` $mem `:` type($mem) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    static StringRef getIntrinsicName() { return "MSCATTER"; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }
  }];
}

//===----------------------------------------------------------------------===//
// TABS (sync, dps style)
// dst = abs(src) elementwise
//===----------------------------------------------------------------------===//

def PTO_AbsOp_DPS : PTO_DpsOp<"abs_dps", [
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
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// TADD (sync, tilebuf-based tile add)
//===----------------------------------------------------------------------===//

def PTO_AddOp_DPS : PTO_DpsOp<"add_dps", [
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TADDC (sync, tilebuf-based tile ternary add)
// dst = src0 + src1 + src2
//===----------------------------------------------------------------------===//

def PTO_AddCOp_DPS : PTO_DpsOp<"addc_dps", [
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
//===----------------------------------------------------------------------===//
// TADDS (sync, tilebuf-based tile + scalar)
// dst = src + scalar
//===----------------------------------------------------------------------===//

def PTO_AddSOp_DPS : PTO_DpsOp<"adds_dps", [
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
    AnyType:$scalar,
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
//===----------------------------------------------------------------------===//
// TADDSC (sync, tilebuf-based tile + scalar + tile)
// dst = src0 + scalar + src1
//===----------------------------------------------------------------------===//

def PTO_AddSCOp_DPS : PTO_DpsOp<"addsc_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "dst = src0 + scalar + src1 ";
  let arguments = (ins
    PTODpsType:$src0,
    AnyType:$scalar,
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


def PTO_AndOp_DPS : PTO_DpsOp<"and_dps", [
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

def PTO_AndSOp_DPS : PTO_DpsOp<"ands_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TANDS(dst, src, scalar) bitwise-and tile with scalar";
  let arguments = (ins PTODpsType:$src, AnyType:$scalar, PTODpsType:$dst);
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



def PTO_CIOp_DPS : PTO_DpsOp<"ci_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate contiguous integer sequence into dst tile.";

  let arguments = (ins
    AnyInteger:$S,
    PTODpsType:$dst,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $S `:` type($S) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def PTO_CmpOp_DPS : PTO_DpsOp<"cmp_dps", [
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
          (`{` `cmpMode` `=` $cmpMode^ `}`)?
          `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
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

// ---- TCMPS ----
def PTO_CmpSOp_DPS : PTO_DpsOp<"cmps_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare scalar value against a tile and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_CmpModeAttr, "::mlir::pto::CmpMode::EQ">:$cmpMode
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar
          (`{` `cmpMode` `=` $cmpMode^ `}`)?
          `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// TCOLEXPAND (tilebuf, DPS style: ins/outs, 0 result)
//===----------------------------------------------------------------------===//

def PTO_ColExpandOp_DPS : PTO_DpsOp<"colexpand_dps", [
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
// TCOLMAX (tilebuf, DPS style: ins/outs, 0 result)
// dst[0, j] = max_i src[i, j]
//===----------------------------------------------------------------------===//

def PTO_ColMaxOp_DPS : PTO_DpsOp<"colmax_dps", [
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
// TCOLMIN (tilebuf, DPS style: ins/outs, 0 result)
// dst[0, j] = min_i src[i, j]
//===----------------------------------------------------------------------===//

def PTO_ColMinOp_DPS : PTO_DpsOp<"colmin_dps", [
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
// TCOLSUM (tilebuf, DPS style: ins/outs, explicit tmp)
// dst[0, j] = sum_i src[i, j]
// intrinsic: TCOLSUM(dst, src, tmp, isBinary)
//===----------------------------------------------------------------------===//

def PTO_ColSumOp_DPS : PTO_DpsOp<"colsum_dps", [
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TCVT (tilebuf, DPS)
//   dst = cast_rmode(src)
//===----------------------------------------------------------------------===//

def PTO_CvtOp_DPS : PTO_DpsOp<"cvt_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise type conversion with rounding mode (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RoundModeAttr, "::mlir::pto::RoundMode::CAST_RINT">:$rmode
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

//===----------------------------------------------------------------------===//
// TDIV (sync, tilebuf-based tile div)  DPS style
// dst = src0 / src1 elementwise
//===----------------------------------------------------------------------===//

def PTO_DivOp_DPS : PTO_DpsOp<"div_dps", [
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
// TDIVS (tilebuf, DPS)
// pto.tdivs ins(%src, %scalar) outs(%dst) [scalar_lhs = true|false]
//===----------------------------------------------------------------------===//

def PTO_DivSOp_DPS : PTO_DpsOp<"divs_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division with a scalar (tilebuf, DPS)";

  let arguments = (ins
    AnyType:$src,
    AnyType:$scalar,
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
// TEXP (tilebuf, DPS)
// pto.texp ins(%src) outs(%dst)
//===----------------------------------------------------------------------===//

def PTO_ExpOp_DPS : PTO_DpsOp<"exp_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise exponential (tilebuf, DPS)";

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
// TEXPANDS (tilebuf, DPS)
// pto.texpands ins(%scalar) outs(%dst)
//===----------------------------------------------------------------------===//

def PTO_ExpandsOp_DPS : PTO_DpsOp<"expands_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast scalar into dst (tilebuf, DPS)";

  let arguments = (ins
    AnyTypeOf<[F16, F32, I16, I32, I8, UI8, UI16, UI32]>:$scalar,
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

//===----------------------------------------------------------------------===//
// TEXTRACT (tilebuf, DPS)
// Copy a sub-window from src into dst, starting at (indexRow, indexCol).
//===----------------------------------------------------------------------===//

def PTO_ExtractOp_DPS : PTO_DpsOp<"extract_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract sub-tile window from src into dst (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TEXTRACT is an L1->L0 move (MTE1), not a vector compute op.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TFILLPAD (tilebuf, DPS)
// Copy src into dst and fill padded region with dst's compile-time PadVal.
//===----------------------------------------------------------------------===//

def PTO_FillPadOp_DPS : PTO_DpsOp<"fillpad_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy src into dst and fill padded elements using dst PadVal (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// TGATHER (tilebuf, DPS)
// - Index gather:   TGATHER(dst, src0, indices)
// - Mask gather:    TGATHER<DstTile, SrcTile, MaskPattern::Pxxxx>(dst, src)
//===----------------------------------------------------------------------===//

def PTO_GatherOp_DPS : PTO_DpsOp<"gather_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather/select elements using an index tile or a mask pattern (tilebuf, DPS)";

  // --- operands (DPS): src0 + optional indices + outs(dst) ---
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$indices,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern
  );

  // --- DPS op: no SSA results ---
  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) ( `,` $indices^ `:` qualified(type($indices)) )? `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    (`maskPattern` `=` $maskPattern^)?
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst), [{
      build($_builder, $_state, res, src, dst,  /*indices=*/nullptr, /*maskPattern=*/nullptr); 
    }]>,
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst, "Value":$indices), [{
      build($_builder, $_state, res, src, dst,  indices, /*maskPattern=*/nullptr); 
    }]>];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
    
}

def PTO_GatherbOp_DPS : PTO_DpsOp<"gatherb_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather elements using byte offsets (tilebuf, DPS)";

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


//===----------------------------------------------------------------------===//
// TLOG (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_LogOp_DPS : PTO_DpsOp<"log_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise natural logarithm (tilebuf, DPS)";

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
// TLRELU (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_LReluOp_DPS : PTO_DpsOp<"lrelu_dps", [
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
//===----------------------------------------------------------------------===//
// TMAX (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_MaxOp_DPS : PTO_DpsOp<"max_dps", [
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
//===----------------------------------------------------------------------===//
// TMAXS (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_MaxSOp_DPS : PTO_DpsOp<"maxs_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise max of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    F32:$scalar,
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
// TMIN (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_MinOp_DPS : PTO_DpsOp<"min_dps", [
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

//===----------------------------------------------------------------------===//
// TMINS (tilebuf, DPS)
//===----------------------------------------------------------------------===//

def PTO_MinsOp_DPS : PTO_DpsOp<"mins_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
// PTOOps.td  (add TMOV_FP DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_MovFPOp_DPS : PTO_DpsOp<"mov.fp_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMOV_FP: move/convert using fp (scaling) tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TMOV_FP is an ACC->MAT move (Cc->Cb) with vector quant parameters in
    // SCALING (fbuf). Treat it as a data-movement op for sync insertion.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TMRGSORT DPS/tilebuf op - two formats: 1 tile+blockLen+dst or 4 tiles)
//===----------------------------------------------------------------------===//

def PTO_MrgSortOp_DPS : PTO_DpsOp<"mrgsort_dps", [
  AttrSizedOperandSegments,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMRGSORT: merge sort (format1: ins(src,blockLen) out(dst); format2: ins(src0..src3) outs(dst,tmp,executed))";

  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Optional<AnyInteger>:$blockLen,
    Variadic<PTODpsType>:$dsts,
    Optional<AnyType>:$excuted,
    DefaultValuedAttr<BoolAttr, "false">:$exhausted
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isFormat1() { return getSrcs().size() == 1u && getBlockLen() && getDsts().size() == 1u; }
    bool isFormat2() { return getSrcs().size() == 4u && getDsts().size() == 2u && getExcuted(); }
    Value getSrc() { return getSrcs().front(); }
    Value getDst() { return getDsts().front(); }
    Value getTmp() { return getDsts()[1]; }
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstsMutable(); }
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TMUL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_MulOp_DPS : PTO_DpsOp<"mul_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMUL: elementwise multiply (tilebuf, DPS)";

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
// PTOOps.td  (add TMULS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_MulsOp_DPS : PTO_DpsOp<"muls_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMULS: elementwise multiply by scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    F32:$scalar,
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
// PTOOps.td  (add TNEG DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_NegOp_DPS : PTO_DpsOp<"neg_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNEG: elementwise negation (tilebuf, DPS)";

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
// PTOOps.td  (add TNOT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_NotOp_DPS : PTO_DpsOp<"not_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TNOT: elementwise bitwise NOT (tilebuf, DPS)";

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
// PTOOps.td  (add TOR DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_OrOp_DPS : PTO_DpsOp<"or_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TOR: elementwise bitwise OR (tilebuf, DPS)";

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
// PTOOps.td  (add TORS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_OrsOp_DPS : PTO_DpsOp<"ors_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TORS: elementwise bitwise OR with scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    AnySignlessInteger:$scalar,
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
// PTOOps.td  (add TPARTADD DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_PartAddOp_DPS : PTO_DpsOp<"partadd_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPARTADD: partial elementwise add (tilebuf, DPS)";

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
// PTOOps.td  (add TPARTMAX DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_PartMaxOp_DPS : PTO_DpsOp<"partmax_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPARTMAX: partial elementwise max (tilebuf, DPS)";

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
// PTOOps.td  (add TPARTMIN DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_PartMinOp_DPS : PTO_DpsOp<"partmin_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPARTMIN: partial elementwise min (tilebuf, DPS)";

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
// PTOOps.td  (add TPRELU DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_PreluOp_DPS : PTO_DpsOp<"prelu_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRELU: elementwise PReLU with per-element slope tile (tilebuf, DPS)";

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
// PTOOps.td  (add TRECIP DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RecipOp_DPS : PTO_DpsOp<"recip_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRECIP: elementwise reciprocal (tilebuf, DPS)";

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
// PTOOps.td  (add TRELU DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_ReluOp_DPS : PTO_DpsOp<"relu_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRELU: elementwise ReLU (tilebuf, DPS)";

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
// PTOOps.td  (add TREM DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RemOp_DPS : PTO_DpsOp<"rem_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREM: elementwise remainder/fmod (tilebuf, DPS)";

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
// PTOOps.td  (add TREMS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RemSOp_DPS : PTO_DpsOp<"rems_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREMS: elementwise remainder/fmod with scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
// PTOOps.td  (add TRESHAPE DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_ReshapeOp_DPS : PTO_DpsOp<"reshape_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRESHAPE: reinterpret tile layout/shape (tilebuf, DPS)";

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
// PTOOps.td  (add TROWEXPAND DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowExpandOp_DPS : PTO_DpsOp<"rowexpand_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPAND: broadcast src(i,0) across each dst row (tilebuf, DPS)";

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
// PTOOps.td  (add TROWEXPANDDIV DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowExpandDivOp_DPS : PTO_DpsOp<"rowexpanddiv_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDDIV: dst(i,j)=src0(i,j)/src1(0,i) (tilebuf, DPS)";

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
// PTOOps.td  (add TROWEXPANDMUL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowExpandMulOp_DPS : PTO_DpsOp<"rowexpandmul_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMUL: dst(i,j)=src0(i,j)*src1(0,i) (tilebuf, DPS)";

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
// PTOOps.td  (add TROWEXPANDSUB DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowExpandSubOp_DPS : PTO_DpsOp<"rowexpandsub_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDSUB: dst(i,j)=src0(i,j)-src1(0,i) (tilebuf, DPS)";

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
// PTOOps.td  (add TROWMAX DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowMaxOp_DPS : PTO_DpsOp<"rowmax_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMAX: row-reduce max across cols into dst(i,0) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMIN DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowMinOp_DPS : PTO_DpsOp<"rowmin_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMIN: row-reduce min across cols into dst(i,0) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWSUM DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RowSumOp_DPS : PTO_DpsOp<"rowsum_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWSUM: row-reduce sum across cols into dst(i,0) (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRSQRT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_RsqrtOp_DPS : PTO_DpsOp<"rsqrt_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRSQRT: elementwise reciprocal sqrt (tilebuf, DPS)";

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
// PTOOps.td  (add TSCATTER DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_ScatterOp_DPS : PTO_DpsOp<"scatter_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSCATTER: scatter rows using per-element row indices (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$indexes,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexes `:` qualified(type($src)) `,` qualified(type($indexes) ) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_S; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSEL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SelOp_DPS : PTO_DpsOp<"sel_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSEL: select between two tiles using a mask tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mask `:` qualified(type($mask) ) `,` $src0 `:` qualified(type($src0)) `,` $src1 `:` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSELS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SelsOp_DPS : PTO_DpsOp<"sels_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSELS: select one of two source tiles using a scalar selectMode (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    AnyInteger:$selectMode,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `:` qualified(type($src0)) `,` $src1 `:` qualified(type($src1)) `,` $selectMode `:` type($selectMode) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHL DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_ShlOp_DPS : PTO_DpsOp<"shl_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHL: elementwise shift-left of two tiles (tilebuf, DPS)";

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
// PTOOps.td  (add TSHLS/TSHRS DPS: shift-left/right by scalar)
//===----------------------------------------------------------------------===//

def PTO_ShlSOp_DPS : PTO_DpsOp<"shls_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHLS: elementwise shift-left by scalar (tilebuf, DPS)";
  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
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

def PTO_ShrSOp_DPS : PTO_DpsOp<"shrs_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHRS: elementwise shift-right by scalar (tilebuf, DPS)";
  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
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
// PTOOps.td  (add TSHR DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_ShrOp_DPS : PTO_DpsOp<"shr_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHR: elementwise shift-right of two tiles (tilebuf, DPS)";

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
// PTOOps.td  (add TSORT32 DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_Sort32Op_DPS : PTO_DpsOp<"sort32_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSORT32: sort a fixed-size 32-element block and produce an index mapping (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    PTODpsType:$idx
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `,` $idx `:` qualified(type($dst) ) `,` qualified(type($idx)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
    ::mlir::MutableOperandRange getDpsInitsMutable() {
      return ::mlir::MutableOperandRange(getOperation(), /*start=*/1, /*length=*/2);
    }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSQRT DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SqrtOp_DPS : PTO_DpsOp<"sqrt_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSQRT: elementwise square root (tilebuf, DPS)";

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
// PTOOps.td  (add TSTORE_FP DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_StoreFPOp_DPS : PTO_DpsOp<"store_fp_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSTORE_FP: store an accumulator tile into global memory with scaling (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE3; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUB DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SubOp_DPS : PTO_DpsOp<"sub_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUB: elementwise subtract two tiles (tilebuf, DPS)";

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
// PTOOps.td  (add TSUBC DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SubCOp_DPS : PTO_DpsOp<"subc_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBC: elementwise ternary operation (src0 - src1 + src2, tilebuf, DPS)";

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
// PTOOps.td  (add TSUBS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SubSOp_DPS : PTO_DpsOp<"subs_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBS: elementwise subtract scalar from tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
// PTOOps.td  (add TSUBSC DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SubSCOp_DPS : PTO_DpsOp<"subsc_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBSC: elementwise fused subtract scalar and add tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    F32:$scalar,
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
// PTOOps.td  (add TXORS DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_XORSOp_DPS : PTO_DpsOp<"xors_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXORS: elementwise bitwise XOR of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src0,
    AnyInteger:$scalar,
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

def PTO_PrintOp_DPS: PTO_TOp<"print_dps", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "TPRINT: Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code..";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs);

  let hasVerifier = 1;
  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSYNC DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_SYNCOp_DPS : PTO_DpsOp<"sync_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSYNC: Synchronize PTO execution with event tokens or single-op barrier.";

  let arguments = (ins
    PTODpsType:$events,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $events `:` type($events) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_ALL; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXOR DPS/tilebuf op)
//===----------------------------------------------------------------------===//

def PTO_XOROp_DPS : PTO_DpsOp<"xor_dps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXOR: elementwise bitwise XOR of two tiles (tilebuf, DPS)";

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
// PTO_TOPs(Tilebuffer, DPS)
//===----------------------------------------------------------------------===//

def PTO_TAbsOp : PTO_TOp<"tabs", [
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
  }];
}

def PTO_TAddOp : PTO_TOp<"tadd", [
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

def PTO_TAddCOp : PTO_TOp<"taddc", [
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
  }];
}

def PTO_TAddSOp : PTO_TOp<"tadds", [
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
    AnyType:$scalar,
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
  }];
}

def PTO_TAddSCOp : PTO_TOp<"taddsc", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "dst = src0 + scalar + src1 ";
  let arguments = (ins
    PTODpsType:$src0,
    AnyType:$scalar,
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
  }];
}


def PTO_TAndOp : PTO_TOp<"tand", [
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
  }];
}

def PTO_TAndSOp : PTO_TOp<"tands", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TANDS(dst, src, scalar) bitwise-and tile with scalar";
  let arguments = (ins PTODpsType:$src, AnyType:$scalar, PTODpsType:$dst);
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}


def PTO_TCIOp : PTO_TOp<"tci", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Generate contiguous integer sequence into dst tile.";

  let arguments = (ins
    AnyInteger:$S,
    PTODpsType:$dst,
    DefaultValuedAttr<BoolAttr, "false">:$descending
  );
  let results = (outs);

  let assemblyFormat = [{
    `ins` `(` $S 
    attr-dict
    `:` type($S) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def PTO_TCmpOp : PTO_TOp<"tcmp", [
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
  }];
}

def PTO_TCmpSOp : PTO_TOp<"tcmps", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Compare scalar value against a tile and write a packed predicate mask";

  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
    DefaultValuedAttr<PTO_CmpModeAttr, "::mlir::pto::CmpMode::EQ">:$cmpMode,
    PTODpsType:$dst
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];

  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar
          attr-dict
          `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    
  }];
}

def PTO_TColExpandOp : PTO_TOp<"tcolexpand", [
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
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def PTO_TColMaxOp : PTO_TOp<"tcolmax", [
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
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def PTO_TColMinOp : PTO_TOp<"tcolmin", [
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
  }];

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def PTO_TColSumOp : PTO_TOp<"tcolsum", [
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
  }];
}

def PTO_TCvtOp : PTO_TOp<"tcvt", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise type conversion with rounding mode (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    DefaultValuedAttr<PTO_RoundModeAttr, "::mlir::pto::RoundMode::CAST_RINT">:$rmode
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];

  let assemblyFormat = [{
    `ins` `(` $src 
    attr-dict
    `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
  }];
}

def PTO_TDivOp : PTO_TOp<"tdiv", [
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
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `:` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

}

def PTO_TDivSOp : PTO_TOp<"tdivs", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise division with a scalar (tilebuf, DPS)";

  let arguments = (ins
    AnyType:$src,
    AnyType:$scalar,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];

}

def PTO_TExpOp : PTO_TOp<"texp", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise exponential (tilebuf, DPS)";

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
  }];
}

def PTO_TExpandsOp : PTO_TOp<"texpands", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Broadcast scalar into dst (tilebuf, DPS)";

  let arguments = (ins
    AnyTypeOf<[F16, F32, I16, I32, I8, UI8, UI16, UI32]>:$scalar,
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
  }];
}

def PTO_TExtractOp : PTO_TOp<"textract", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract sub-tile window from src into dst (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    Index:$indexRow,
    Index:$indexCol,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexRow `,` $indexCol `:` qualified(type($src)) `,` type($indexRow) `,` type($indexCol) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TEXTRACT moves data between memory domains (L1/cbuf -> L0A/L0B/L0C),
    // which is executed by the MTE1 pipeline.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
  }];
}

def PTO_TFillPadOp : PTO_TOp<"tfillpad", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Copy src into dst and fill padded elements using dst PadVal (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins`  `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def PTO_TGatherOp : PTO_TOp<"tgather", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather/select elements using an index tile or a mask pattern (tilebuf, DPS)";

  // --- operands (DPS): src0 + optional indices + outs(dst) ---
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    Optional<PTODpsType>:$indices,
    OptionalAttr<PTO_MaskPatternAttr>:$maskPattern
  );

  // --- DPS op: no SSA results ---
  let results = (outs);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def PTO_TGatherbOp : PTO_TOp<"tgatherb", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Gather elements using byte offsets (tilebuf, DPS)";

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
  }];
}

def PTO_TLogOp : PTO_TOp<"tlog", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise natural logarithm (tilebuf, DPS)";

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
  }];
}

def PTO_TLReluOp : PTO_TOp<"tlrelu", [
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
  }];
}

def PTO_TMaxOp : PTO_TOp<"tmax", [
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
  }];
}

def PTO_TMaxSOp : PTO_TOp<"tmaxs", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise max of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
  }];
}

def PTO_TMinOp : PTO_TOp<"tmin", [
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
  }];
}

def PTO_TMinsOp : PTO_TOp<"tmins", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Elementwise minimum of a tile and a scalar (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
  }];
}

def PTO_TMovFPOp : PTO_TOp<"tmov.fp", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMOV_FP: move/convert using fp (scaling) tile (tilebuf, DPS)";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    // TMOV_FP is an ACC->MAT move (Cc->Cb) with vector quant parameters in
    // SCALING (fbuf). Treat it as a data-movement op for sync insertion.
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_MTE1; }
  }];
}

def PTO_TMrgSortOp: PTO_TOp<"tmrgsort", [
  AttrSizedOperandSegments,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMRGSORT: Merge sort (format1: ins(src,blockLen) out(dst); format2: ins(src0..src3) outs(dst,tmp,executed)).";

  let arguments = (ins
    Variadic<PTODpsType>:$srcs,
    Optional<AnyInteger>:$blockLen,
    Variadic<PTODpsType>:$dsts,
    Optional<AnyType>:$excuted,
    DefaultValuedAttr<BoolAttr, "false">:$exhausted
  );

  let results = (outs);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isFormat1() { return getSrcs().size() == 1u && getBlockLen() && getDsts().size() == 1u; }
    bool isFormat2() { return getSrcs().size() == 4u && getDsts().size() == 2u && getExcuted(); }
    Value getSrc() { return getSrcs().front(); }
    Value getDst() { return getDsts().front(); }
    Value getTmp() { return getDsts()[1]; }
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstsMutable(); }
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}

def PTO_TMulOp: PTO_TOp<"tmul", [
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
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TMULS DPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TMulsOp: PTO_TOp<"tmuls", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TMULS: Elementwise multiply a tile by a scalar.";

  let arguments = (ins
    PTODpsType:$src0,
    F32:$scalar,
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNEG TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TNegOp: PTO_TOp<"tneg", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TNOT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TNotOp: PTO_TOp<"tnot", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TOrOp: PTO_TOp<"tor", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TOrsOp: PTO_TOp<"tors", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTADD TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TPartAddOp: PTO_TOp<"tpartadd", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TPartMaxOp: PTO_TOp<"tpartmax", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPARTMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TPartMinOp: PTO_TOp<"tpartmin", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TPreluOp: PTO_TOp<"tprelu", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRELU: Elementwise PReLU (parametric ReLU) with a per-element slope tile.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRECIP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRecipOp: PTO_TOp<"trecip", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRECIP: Elementwise reciprocal of a tile.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TReluOp: PTO_TOp<"trelu", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRemOp: PTO_TOp<"trem", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREM: Elementwise remainder of two tiles.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TREMS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRemSOp: PTO_TOp<"trems", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TREMS: Elementwise remainder with a scalar";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRESHAPE TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TReshapeOp: PTO_TOp<"treshape", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRESHAPE: Elementwise remainder with a scalar";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPAND TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowExpandOp: PTO_TOp<"trowexpand", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDDIV TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowExpandDivOp: PTO_TOp<"trowexpanddiv", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDDIV: Row-wise broadcast divide: divide each row of src0 by a per-row scalar vector src1.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowExpandMulOp: PTO_TOp<"trowexpandmul", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDMUL: Row-wise broadcast divide: divide each row of src0 by a per-row scalar vector src1.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWEXPANDSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowExpandSubOp: PTO_TOp<"trowexpandsub", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWEXPANDSUB: Row-wise broadcast subtract: subtract a per-row scalar vector src1 from each row of src0.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMAX TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowMaxOp: PTO_TOp<"trowmax", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMAX: Reduce each row by taking the maximum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWMIN TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowMinOp: PTO_TOp<"trowmin", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWMIN: Reduce each row by taking the minimum across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TROWSUM TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRowSumOp: PTO_TOp<"trowsum", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TROWSUM: Reduce each row by summing across columns.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$tmp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $tmp `:` qualified(type($src)) `,` qualified(type($tmp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TRSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TRsqrtOp: PTO_TOp<"trsqrt", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TRSQRT: Elementwise reciprocal square root.";

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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSCATTER TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TScatterOp: PTO_TOp<"tscatter", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSCATTER: Scatter rows of a source tile into a destination tile using per-element row indices.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$indexes,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $indexes `:` qualified(type($src)) `,` qualified(type($indexes) ) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSEL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSelOp: PTO_TOp<"tsel", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSEL: Select between two tiles using a mask tile (per-element selection).";

  let arguments = (ins
    PTODpsType:$mask,
    PTODpsType:$src0,
    PTODpsType:$src1,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $mask `,` $src0 `,` $src1 `:` qualified(type($mask)) `,` qualified(type($src0)) `,` qualified(type($src1)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSELS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSelsOp: PTO_TOp<"tsels", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSELS: Select one of two source tiles using a scalar selectMode (global select).";

  let arguments = (ins
    PTODpsType:$src0,
    PTODpsType:$src1,
    AnyInteger:$selectMode,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src0 `,` $src1 `,` $selectMode `:` qualified(type($src0)) `,` qualified(type($src1)) `,` type($selectMode) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TShlOp: PTO_TOp<"tshl", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TShrOp: PTO_TOp<"tshr", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSHLS/TSHRS TBDPS: shift-left/right by scalar)
//===----------------------------------------------------------------------===//

def PTO_TShlSOp : PTO_TOp<"tshls", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHLS: Elementwise shift-left of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
  let assemblyFormat = [{
    `ins` `(` $src `,` $scalar `:` qualified(type($src)) `,` type($scalar) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}

def PTO_TShrSOp : PTO_TOp<"tshrs", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSHRS: Elementwise shift-right of a tile by a scalar (shift count).";
  let arguments = (ins
    PTODpsType:$src,
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
    PTODpsType:$dst
  );
  let results = (outs);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
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

def PTO_TSort32Op: PTO_TOp<"tsort32", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSORT32: Sort a fixed-size 32-element block and produce an index mapping.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst,
    PTODpsType:$idx
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    `outs` `(` $dst `,` $idx `:` qualified(type($dst) ) `,` qualified(type($idx)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSQRT TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSqrtOp: PTO_TOp<"tsqrt", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSQRT: Elementwise square root.";

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
  }];
}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSTORE_FP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TStoreFPOp: PTO_TOp<"tstore_fp", []> {
  let summary = "TSTORE_FP: Store an accumulator tile into global memory using a scaling (fp) tile for vector quantization parameters.";

  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$fp,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `,` $fp `:` qualified(type($src)) `,` qualified(type($fp)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSubOp: PTO_TOp<"tsub", [
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
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSubCOp: PTO_TOp<"tsubc", [
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
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSubSOp: PTO_TOp<"tsubs", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBS: Elementwise subtract a scalar from a tile.";

  let arguments = (ins
    PTODpsType:$src,
    F32:$scalar,
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
  }];


}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUBSC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSubSCOp: PTO_TOp<"tsubsc", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TSUBSC: Elementwise fused op: src0 - scalar + src1.";

  let arguments = (ins
    PTODpsType:$src0,
    F32:$scalar,
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
  }];

}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TXORSOp: PTO_TOp<"txors", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXORS: Elementwise bitwise XOR of a tile and a scalar.";

  let arguments = (ins
    PTODpsType:$src,
    AnyInteger:$scalar,
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
  }];

}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSYNC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TSYNCOp: PTO_TOp<"tsync", []> {
  let summary = "TSYNC: Synchronize PTO execution with event tokens or single-op barrier.";

  let arguments = (ins
    PTODpsType:$events,
    PTODpsType:$dst
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $events `:` type($events) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
    attr-dict
  }];

}

//===----------------------------------------------------------------------===//
// PTOOps.td  (add TXOR TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def PTO_TXOROp: PTO_TOp<"txor", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TXOR: Elementwise bitwise XOR of two tiles.";

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
  }];

}

def PTO_TPrintOp: PTO_TOp<"tprint", [
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "TPRINT: Print the contents of a Tile or GlobalTensor for debugging purposes directly from device code..";

  let arguments = (ins
    PTODpsType:$src
  );

  let results = (outs);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `ins` `(` $src `:` qualified(type($src)) `)`
    attr-dict
  }];

  let extraClassDeclaration = [{
    ::mlir::pto::PIPE getPipe() { return ::mlir::pto::PIPE::PIPE_V; }
  }];
 }

#endif // MLIR_DIALECT_PTO_IR_PTOOPS
