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

def PrintScalarType :
  AnyTypeOf<[Index, AnySignlessInteger, AnyFloat], "numeric (index/integer/float)">;

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
    $ptr `[` $offset `]` attr-dict `:` type($ptr) `->` type($value)
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
    $value `,` $ptr `[` $offset `]` attr-dict `:` type($ptr) `,` type($value)
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

// Helper: tensor_view or memref (after lowering tensor_view to memref).
def TensorViewOrMemRef :
  AnyTypeOf<[TensorViewType, AnyMemRef], "TensorView or MemRef">;

// Get the size of a dimension of a tensor_view or its lowered memref view.
// Result type: Index (use arith.index_cast if i32 is needed).
def GetTensorViewDimOp : PTO_Op<"get_tensor_view_dim", [Pure]> {
  let summary = "Get the size of a dimension of a tensor_view.";
  let description = [{
    Returns the size of the given dimension of a logical tensor view.
    This op accepts either !pto.tensor_view or the memref it is lowered to.
    IR (tensor_view form):
      %dim_size = pto.get_tensor_view_dim %tv, %dim_index
        : !pto.tensor_view, index -> index
    IR (memref form, after lowering):
      %dim_size = pto.get_tensor_view_dim %mr, %dim_index
        : memref<...>, index -> index
  }];
  let arguments = (ins
    TensorViewOrMemRef:$tensor_view,
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

def SubsetOp : PTO_Op<"subset", [
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

def TStoreOp: PTO_TOp<"tstore", [
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
    PTODpsType:$dst
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
      auto isAcc = [](Type ty) -> bool {
        if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC;
          return false;
        }
        if (auto mr = ::mlir::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as = ::mlir::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace() == ::mlir::pto::AddressSpace::ACC;
          }
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

// -------------------------
// Tile-world TOp versions (PTODpsType)
// -------------------------

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
    PTODpsType:$dst
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
    static StringRef getIntrinsicName() { return "TMATMUL_MX_BIAS"; }
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
    PTODpsType:$dst
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
    PTODpsType:$dst
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
    PTODpsType :$dst
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

def TMovOp  : PTO_TOp<"tmov", [
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Move data between domains (DPS version).";
  
  let arguments = (ins
    PTODpsType:$src,
    PTODpsType:$dst
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
      auto getASFromType = [](Type ty)
          -> std::optional<::mlir::pto::AddressSpace> {
        // Pre-lowering: tile_buf carries the address space in its memorySpace.
        if (auto tb = llvm::dyn_cast<::mlir::pto::TileBufType>(ty)) {
          if (auto as = llvm::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
                  tb.getMemorySpace()))
            return as.getAddressSpace();
          return std::nullopt;
        }
        // Post PTOViewToMemref: tile_buf is erased to memref but memorySpace is
        // preserved in memref's memorySpace attribute.
        if (auto mr = llvm::dyn_cast<::mlir::MemRefType>(ty)) {
          if (auto ms = mr.getMemorySpace()) {
            if (auto as =
                    llvm::dyn_cast<::mlir::pto::AddressSpaceAttr>(ms))
              return as.getAddressSpace();
          }
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
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
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
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
  let assemblyFormat = [{
    `[` $src_pipe `,` $dst_pipe `,` $event_id `]` attr-dict
  }];
}

def SyncSetOp : PTO_Op<"sync.set"> {
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

def SyncWaitOp : PTO_Op<"sync.wait"> {
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

def TrapOp : PTO_Op<"trap"> {
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// MSCATTER (DPS)  +  TMSCATTER (tile world TOp)
//===----------------------------------------------------------------------===//

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
// pto.mscatter ins(%src, %idx) outs(%mem) [ ...]
def MScatterOp : PTO_TOp<"mscatter", [
  PTO_DpsInitOpInterface,
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

def TAddSCOp : PTO_TOp<"taddsc", [
  PTO_DpsInitOpInterface,
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

def TAndSOp : PTO_TOp<"tands", [
  PTO_DpsInitOpInterface,
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


def TCIOp : PTO_TOp<"tci", [
  PTO_DpsInitOpInterface,
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
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
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
      return ::mlir::MutableOperandRange(getOperation(), 1, getOperation()->getNumOperands());
    }
  }];
}

def TCvtOp : PTO_TOp<"tcvt", [
  PTO_DpsInitOpInterface,
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
    `ins` `(` $src 
    attr-dict
    `:` qualified(type($src)) `)`
    `outs` `(` $dst `:` qualified(type($dst) ) `)`
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

def TDivSOp : PTO_TOp<"tdivs", [
  PTO_DpsInitOpInterface,
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

def TExpOp : PTO_TOp<"texp", [
  PTO_DpsInitOpInterface,
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

def TExpandsOp : PTO_TOp<"texpands", [
  PTO_DpsInitOpInterface,
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

def TExtractOp : PTO_TOp<"textract", [
  PTO_DpsInitOpInterface,
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TFillPadOp : PTO_TOp<"tfillpad", [
  PTO_DpsInitOpInterface,
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

def TGatherOp : PTO_TOp<"tgather", [
  PTO_DpsInitOpInterface,
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def TGatherBOp : PTO_TOp<"tgatherb", [
  PTO_DpsInitOpInterface,
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

def TLogOp : PTO_TOp<"tlog", [
  PTO_DpsInitOpInterface,
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

def TMovFPOp : PTO_TOp<"tmov.fp", [
  PTO_DpsInitOpInterface,
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

def TMrgSortOp: PTO_TOp<"tmrgsort", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
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
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TPRELU TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TPReluOp: PTO_TOp<"tprelu", [
  PTO_DpsInitOpInterface,
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
// PTOOps.td  (add TRESHAPE TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TReshapeOp: PTO_TOp<"treshape", [
  PTO_DpsInitOpInterface,
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
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
// PTOOps.td  (add TROWEXPANDMUL TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TRowExpandMulOp: PTO_TOp<"trowexpandmul", [
  PTO_DpsInitOpInterface,
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
// PTOOps.td  (add TSCATTER TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TScatterOp: PTO_TOp<"tscatter", [
  PTO_DpsInitOpInterface,
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
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
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
    AnyTypeOf<[AnyFloat, AnySignlessInteger, Index]>:$scalar,
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
// PTOOps.td  (add TSTORE_FP TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TStoreFPOp: PTO_TOp<"tstore_fp", [
  PTO_DpsInitOpInterface,
]> {
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
  let extraClassDeclaration = [{
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSUB TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

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
// PTOOps.td  (add TXORS TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TXorSOp: PTO_TOp<"txors", [
  PTO_DpsInitOpInterface,
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
    ::mlir::MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];

}
//===----------------------------------------------------------------------===//
// PTOOps.td  (add TSYNC TBDPS/tile buffer op)
//===----------------------------------------------------------------------===//

def TSyncOp: PTO_TOp<"tsync", [
  PTO_DpsInitOpInterface,
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

def TPrintOp: PTO_TOp<"tprint", [
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
