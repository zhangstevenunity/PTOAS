// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOTypeDefs.td --------------------------------------------------*- tablegen -*-===//
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"

// ---- !pto.ptr<elem, gm|ub> ----
def PtrType : TypeDef<PTO_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let parameters = (ins
    "mlir::Type":$elementType,
    "mlir::pto::AddressSpaceAttr":$memorySpace
  );
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return Base::get($_ctxt, elementType,
                       mlir::pto::AddressSpaceAttr::get($_ctxt,
                                                        mlir::pto::AddressSpace::GM));
    }]>,
    TypeBuilder<(ins "Type":$elementType,
                     "mlir::pto::AddressSpaceAttr":$memorySpace), [{
      return Base::get($_ctxt, elementType, memorySpace);
    }]>
  ];
}

// ---- !pto.tensor_view<d0xd1x...xelem> ----
def TensorViewType : TypeDef<PTO_Dialect, "TensorView"> {
  let mnemonic = "tensor_view";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "mlir::Attribute":$layout
  );
  let hasCustomAssemblyFormat = 1;

  // A null layout preserves the historical spelling and means the default ND
  // layout. Non-ND layouts are materialized in the type when they cross an SSA
  // merge or a function boundary.
  let builders = [
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "Type":$elementType), [{
      return Base::get($_ctxt, shape, elementType, mlir::Attribute{});
    }]>,
    TypeBuilder<(ins "int64_t":$rank,
                     "Type":$elementType), [{
      SmallVector<int64_t, 4> shp(rank, mlir::ShapedType::kDynamic);
      return Base::get($_ctxt, shp, elementType, mlir::Attribute{});
    }]>
  ];

  let extraClassDeclaration = [{
    LayoutAttr getLayoutAttr() const {
      return ::llvm::dyn_cast_or_null<LayoutAttr>(getLayout());
    }
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) {
        if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
        num *= dim;
      }
      return num;
    }
  }];
}

// =============================================================================
// PartitionTensorViewType (原 TileViewType)
// =============================================================================
// 对应 IR: !pto.partition_tensor_view<d0xd1x...xelem>
// =============================================================================
def PartitionTensorViewType : TypeDef<PTO_Dialect, "PartitionTensorView"> {
  let mnemonic = "partition_tensor_view";
  
  let summary = "A logical partition of a tensor view (high-dimensional)";
  let description = [{
    Represents a logical slice of a global tensor. It holds shape and stride 
    information but does not own the data.
    Corresponds to the 'partition_view' concept in the design spec.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "mlir::Attribute":$layout
  );

  let hasCustomAssemblyFormat = 1;

  let builders = [
    TypeBuilder<(ins "ArrayRef<int64_t>":$shape,
                     "Type":$elementType), [{
      return Base::get($_ctxt, shape, elementType, mlir::Attribute{});
    }]>,
    TypeBuilder<(ins "int64_t":$rank,
                     "Type":$elementType), [{
      SmallVector<int64_t, 4> shp(rank, mlir::ShapedType::kDynamic);
      return Base::get($_ctxt, shp, elementType, mlir::Attribute{});
    }]>
  ];

  let extraClassDeclaration = [{
    LayoutAttr getLayoutAttr() const {
      return ::llvm::dyn_cast_or_null<LayoutAttr>(getLayout());
    }
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) {
        if (dim == mlir::ShapedType::kDynamic) return mlir::ShapedType::kDynamic;
        num *= dim;
      }
      return num;
    }
  }];
}

def TileType : TypeDef<PTO_Dialect, "Tile"> {
  let mnemonic = "tile";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getDimSize(unsigned idx) const { return getShape()[idx]; }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) num *= dim;
      return num;
    }
  }];
}

def TileBufType : TypeDef<PTO_Dialect, "TileBuf"> {
  let mnemonic = "tile_buf";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    "mlir::Attribute":$memorySpace,
    ArrayRefParameter<"int64_t">:$validShape,
    "mlir::pto::TileBufConfigAttr":$config // TileBufConfigAttr (or null -> default)
  );

  let hasCustomAssemblyFormat = 1;

  // 兼容性 builder：老代码只传 shape/elem/memorySpace
  let builders = [
    // (1) 旧接口：shape/elem/memorySpace
    TypeBuilder<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      "Attribute":$memorySpace
    ), [{
      auto cfg = mlir::pto::TileBufConfigAttr::getDefault($_ctxt);

      // ✅ 默认 validShape = shape（你要求的语义）
      SmallVector<int64_t, 2> vs(shape.begin(), shape.end());
      for (int64_t &dim : vs)
        if (dim < 0)
          dim = mlir::ShapedType::kDynamic;

      return Base::get($_ctxt, shape, elementType, memorySpace, vs, cfg);
    }]>,

    // (2) 兼容接口：shape/elem/memorySpace/config
    TypeBuilder<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      "Attribute":$memorySpace,
      "mlir::pto::TileBufConfigAttr":$config
    ), [{
      // ✅ 默认 validShape = shape
      SmallVector<int64_t, 2> vs(shape.begin(), shape.end());
      for (int64_t &dim : vs)
        if (dim < 0)
          dim = mlir::ShapedType::kDynamic;
      return Base::get($_ctxt, shape, elementType, memorySpace, vs, config);
    }]>,

    // (3) 便利接口：shape/elem/memorySpace/validShape（config 默认）
    TypeBuilder<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType,
      "Attribute":$memorySpace,
      "ArrayRef<int64_t>":$validShape
    ), [{
      auto cfg = mlir::pto::TileBufConfigAttr::getDefault($_ctxt);
      SmallVector<int64_t, 4> vs(validShape.begin(), validShape.end());
      for (int64_t &dim : vs)
        if (dim < 0)
          dim = mlir::ShapedType::kDynamic;
      return Base::get($_ctxt, shape, elementType, memorySpace, vs, cfg);
    }]>
  ];

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }

    bool hasValidShape() const { return !getValidShape().empty(); }
    bool hasDynamicValid() const {
      auto vs = getValidShape();
      return vs.size()==2 && vs[0] < 0 && vs[1] < 0;
    }

    mlir::pto::TileBufConfigAttr getConfigAttr() const;
    bool hasNonDefaultConfig() const;

    // ✅ 返回强类型 Attr，而不是 enum
    mlir::Attribute getBLayoutAttr() const;
    mlir::Attribute getSLayoutAttr() const;
    int32_t getSFractalSizeI32() const;
    mlir::Attribute getPadValueAttr() const;
    mlir::Attribute getCompactModeAttr() const;

    // 如果你仍然想要“数值枚举”，就提供 int getter（不会依赖 enum 类型）
    int32_t getBLayoutValueI32() const;   // 0 row_major, 1 col_major
    int32_t getSLayoutValueI32() const;   // 0 none_box, 1 row_major, 2 col_major
    int32_t getPadValueI32() const;       // 0 null, 1 zero, 2 max, 3 min
    int32_t getCompactModeI32() const;    // 0 null, 1 normal, 2 row_plus_one
  }];
}

// =============================================================================
// MultiTileBufType
// =============================================================================
// 对应 IR: !pto.multi_tile_buf<<slotType>, count=N>
//
// 表达一个"由 N 个物理 slot 组成"的逻辑 tile buffer。每个 slot 与
// `slotType` 完全同构（同 shape / dtype / loc / valid / config）。该类型只
// 能通过 `pto.alloc_multi_tile` 产生，通过 `pto.multi_tile_get [%k]` 取出
// 第 k 个 slot 后转回普通 `tile_buf`，下游所有 op 仍然只见到 `tile_buf`。
// =============================================================================
def MultiTileBufType : TypeDef<PTO_Dialect, "MultiTileBuf"> {
  let mnemonic = "multi_tile_buf";
  let summary = "An array of N physical slots sharing one tile_buf shape";
  let description = [{
    A `multi_tile_buf` value owns `count` physically-distinct slots of an
    inner `tile_buf` shape. The slot type is identical across all N slots;
    only the underlying physical address differs.

    Multi-buffer is opaque to all existing `tile_buf` ops: extracting a slot
    via `pto.multi_tile_get` yields a regular `tile_buf` that can be passed
    to any DMA / compute / view op unchanged. The slot index expression is
    user-provided (constant or any SSA `index`); ptoas does not synthesize
    `iv mod N` for users.
  }];

  let parameters = (ins
    "mlir::pto::TileBufType":$slotType,
    "uint32_t":$count
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Return the inner per-slot tile_buf type.
    mlir::pto::TileBufType getTileBufType() const { return getSlotType(); }
  }];
}

def EventIdArrayType : TypeDef<PTO_Dialect, "EventIdArray"> {
  let mnemonic = "eventid_array";
  let summary = "Manual-only local array for dynamic event ids";
  let parameters = (ins
    "int64_t":$size
  );
  let assemblyFormat = "`<` $size `>`";
}

// ---- !pto.local_array<D1 x D2 x ... x Dk x T> ----
// Statically-shaped C++ stack-local array. Lowers to a plain `T a[D1][D2]...;`
// declaration in the emitted C++. Element type is restricted to scalars
// (integer / float). This type does NOT participate in PTO memory planning
// lowering; the underlying address is decided by the
// host C++ compiler, not by PTOAS.
def LocalArrayType : TypeDef<PTO_Dialect, "LocalArray"> {
  let mnemonic = "local_array";
  let summary = "C++ stack-local statically-sized array";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    int64_t getRank() const { return getShape().size(); }
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t d : getShape()) num *= d;
      return num;
    }
  }];
}

// ---- !pto.struct<T0, T1, ..., Tn-1> ----
// Scalar-unit heterogeneous aggregate. The heterogeneous dual of
// !pto.local_array: it lowers to a plain C++ `struct { ... };` whose address is
// decided by the host C++ compiler, never by PTO memory planning or
// pto.pointer_cast. Fields are positionally numbered (f0, f1, ...), matching an
// LLVM literal struct.
//
// A field type is either a scalar the backend can name exactly (an integer of
// width 8/16/32/64, or f16/bf16/f32/f64) or a nested !pto.struct — that is how
// nested aggregates are expressed. Three things are rejected by the type
// verifier:
//   - vec/cube types (tile_buf / tensor_view), keeping the scalar world
//     disjoint from the fractal/layout world;
//   - scalars with no exact C++ spelling (i1, i24, the packed f8/f4 formats),
//     which would otherwise be emitted as `float`;
//   - !pto.local_array, because a field is reached with emitc.member, whose
//     result must be an !emitc.lvalue, and !emitc.lvalue cannot wrap the
//     !emitc.array an array field lowers to.
//
// A value of this type lowers to a *pointer* to the struct's storage. To keep
// the pointed-to lifetime visible, storage originates only from
// pto.declare_struct in the owning function; function arguments and other
// operations may not produce values of this type; see PTO.cpp.
def StructType : TypeDef<PTO_Dialect, "Struct"> {
  let mnemonic = "struct";
  let summary = "Scalar-unit heterogeneous aggregate (C++ struct / llvm.struct)";
  let parameters = (ins ArrayRefParameter<"mlir::Type">:$fieldTypes);

  let hasCustomAssemblyFormat = 1;   // form: !pto.struct<f16, i8, !pto.struct<f32, i8>>
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    size_t getNumFields() const { return getFieldTypes().size(); }
    mlir::Type getFieldType(unsigned i) const { return getFieldTypes()[i]; }
  }];
}

def PipeType : TypeDef<PTO_Dialect, "Pipe"> {
  let mnemonic = "pipe";
  let summary = "Opaque pipe handle type for TPUSH/TPOP communication";
}

def AsyncSessionType : TypeDef<PTO_Dialect, "AsyncSession"> {
  let mnemonic = "async_session";
  let summary = "Opaque async DMA session handle type";
}

def AsyncEventType : TypeDef<PTO_Dialect, "AsyncEvent"> {
  let mnemonic = "async_event";
  let summary = "Opaque async DMA event handle type";
}

def PrefetchAsyncContextType
    : TypeDef<PTO_Dialect, "PrefetchAsyncContext"> {
  let mnemonic = "prefetch_async_context";
  let summary = "Opaque PTO prefetch async context handle type";
}

// ---- !pto.hif8 ---- Corresponding type name: hifloat8_t (1 byte)
def HiF8Type : TypeDef<PTO_Dialect, "HiF8", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "hif8";
  let summary = "Ascend A5 HiFloat8 (8-bit, 1 byte per element).";
}

// ---- !pto.f8E8M0 ---- Corresponding type name: float8_e8m0_t (1 byte)
def F8E8M0Type : TypeDef<PTO_Dialect, "F8E8M0", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "f8E8M0";
  let summary = "Ascend A5 MX exponent/scale payload (8-bit E8M0, 1 byte per element).";
}

// ---- !pto.hif8x2 ---- Corresponding type name: hifloat8x2_t (2 bytes)
def HiF8x2Type : TypeDef<PTO_Dialect, "HiF8x2", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "hif8x2";
  let summary = "Packed pair of Ascend A5 HiFloat8 values (2 bytes).";
}

// ---- !pto.f4E1M2x2 ---- Corresponding type name: float4_e1m2x2_t (1 byte = 2 FP4 packed)
def F4E1M2x2Type : TypeDef<PTO_Dialect, "F4E1M2x2", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "f4E1M2x2";
  let summary =
      "Packed pair of FP4 values (E1M2 format, 2-per-byte). "
      "tile_buf shape counts pairs; each pair occupies 1 byte.";
}

// ---- !pto.f4E2M1x2 ---- Corresponding type name: float4_e2m1x2_t (1 byte = 2 FP4 packed)
def F4E2M1x2Type : TypeDef<PTO_Dialect, "F4E2M1x2", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "f4E2M1x2";
  let summary =
      "Packed pair of FP4 values (E2M1 format, 2-per-byte). "
      "tile_buf shape counts pairs; each pair occupies 1 byte.";
}

// ---- !pto.bf16x2 ---- Corresponding type name: bf16x2_t (4 bytes)
def BF16x2Type : TypeDef<PTO_Dialect, "BF16x2", [
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface,
      ["getTypeSizeInBits", "getABIAlignment", "getPreferredAlignment"]>
]> {
  let mnemonic = "bf16x2";
  let summary = "Packed pair of BF16 values (4 bytes).";
}

include "PTO/IR/VMITypeDefs.td"
include "PTO/IR/VPTOTypeDefs.td"
