//===- PTOTypeDefs.td --------------------------------------------------*- tablegen -*-===//
include "mlir/IR/AttrTypeBase.td"
include "PTO/IR/PTODialect.td"
include "PTO/IR/PTOAttrs.td"

// ---- !pto.ptr<elem> ----
def PtrType : TypeDef<PTO_Dialect, "Ptr"> {
  let mnemonic = "ptr";
  let parameters = (ins
    "mlir::Type":$elementType
  );
  let assemblyFormat = "`<` $elementType `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "Type":$elementType), [{
      return Base::get($_ctxt, elementType);
    }]>
  ];
}

// ---- !pto.tensor_view<d0xd1x...xelem> ----
def TensorViewType : TypeDef<PTO_Dialect, "TensorView"> {
  let mnemonic = "tensor_view";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );
  let hasCustomAssemblyFormat = 1;

  // Convenience builders: explicit shape, or rank -> all dynamic ("?")
  let builders = [
    TypeBuilder<(ins "int64_t":$rank,
                     "Type":$elementType), [{
      SmallVector<int64_t, 4> shp(rank, mlir::ShapedType::kDynamic);
      return Base::get($_ctxt, shp, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
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
    "mlir::Type":$elementType
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
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
      return Base::get($_ctxt, shape, elementType, memorySpace, validShape, cfg);
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

    // 如果你仍然想要“数值枚举”，就提供 int getter（不会依赖 enum 类型）
    int32_t getBLayoutValueI32() const;   // 0 row_major, 1 col_major
    int32_t getSLayoutValueI32() const;   // 0 none_box, 1 row_major, 2 col_major
    int32_t getPadValueI32() const;       // 0 null, 1 zero, 2 max, 3 min
  }];
}
