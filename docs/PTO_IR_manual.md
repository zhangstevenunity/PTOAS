# PTO IR Reference

- **Version:** `v 0.1`
- **Date:** `2026-02-14`
- **Author:** `Wenbo Sun`

## 1. Overview

The **PTO Dialect** (`pto`) is an MLIR dialect for expressing tile-based computations targeting Ascend NPU hardware. It is part of the PTOAS (PTO Assembler & Optimizer) compiler toolchain.

- **Dialect name:** `pto`
- **Source:** `include/PTO/IR/`

### PTO IR Level Model

PTO IR is organized as a hierarchical, multi-level IR stack and intentionally exposes multiple abstraction levels to external users and frameworks.

- **Level-1 (SSA-centric IR):** `pto.tile` is an SSA value; PTO-AS is responsible for buffer allocation and storage planning during lowering.
- **Level-2 (DPS tile-buffer IR):** `pto.tile_buf` is represented in destination-passing style (DPS), i.e., as explicit buffer objects rather than SSA value semantics.
- **Level-3 (Low-level scheduling IR):** pipeline/event synchronization is explicit and user-managed, enabling direct control over execution ordering and inter-op dependencies.

These levels are lowered progressively from Level-1 to Level-3, serving distinct optimization and control requirements across different users and integrations. **This PTO IR API document focuses on Level-2 and Level-3 interfaces.** *The Level-1 public interface is still under active design and will be specified in a future revision.*

### Hardware Memory Hierarchy

```
GM (Global Memory)
|- MAT (L1 Cache)
|  |- LEFT  (L0A - left matrix buffer)
|  |- RIGHT (L0B - right matrix buffer)
|  |- ACC   (L0C - accumulator)
|  `- BIAS  (bias buffer)
`- VEC (UB  - unified buffer)
```

## 1.1 Rationale

For the Level-2/Level-3 profiles documented here, PTO IR models tiles as buffers rather than SSA values. A `pto.tile_buf` denotes a storage object with an explicit lifetime, not a pure value. This design intentionally decouples allocation/tiling from pipeline scheduling: buffer allocation is NP-hard, and pipeline scheduling is also NP-hard. Coupling both problems in a single compiler pass is intractable in practice. Therefore, PTO IR requires users or higher-level frameworks to manage buffer reuse explicitly via `pto.alloc_tile`, while PTO AS passes focus on scheduling and pipeline orchestration.

**Example (explicit buffer lifetime):**

```mlir
%a0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%a1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
pto.tload ins(%pv0 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
pto.tload ins(%pv1 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

## 2. Type System

### 2.1 Element Types

Element types describe the primitive scalar values that can be stored in tensors/tiles; by themselves they do not form a value. They define how a sequence of bits is interpreted and the number of bits required to represent the value. This is distinct from any storage size implied by tensor layout.

Common element categories include:

- **Integers**: signless integers such as `i1/i8/i16/i32`. Signedness is not encoded in the type; it is selected by operation semantics or attributes where required.
- **Floating-point**: IEEE floating-point types such as `f16/f32`. Some targets may also support additional formats (e.g., `bf16` or low-precision exponent/mantissa formats) with stricter constraints.
- **Index-like**: index values may appear as scalar operands in certain operations (e.g., offsets, sizes, or scalar comparisons).

Element type constraints are operation-specific:

- **Shape/type consistency**: most elementwise ops require all operands and results to have the same element type.
- **Numeric domain**: reductions, math ops, and division typically restrict element types to floating-point or a subset of integer types.
- **Bitwise ops**: require integer element types.
- **Conversions**: `pto.tcvt` defines explicit element type changes and is controlled by `RoundMode` when converting between numeric domains.

In addition, memory layout and address space do not change the element type semantics; they only affect placement and access patterns.

### 2.2 `!pto.ptr<elementType>`

A pointer to global memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32...)` | Element type pointed to |

**Syntax:** `!pto.ptr<f16>`

---

### 2.3 `!pto.tensor_view<d0 x d1 x elementType>`

A descriptor for a global memory tensor. Does not own data - represents a view with shape and stride information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<int64_t>` | Tensor shape `[d0, d1]` (each dim may be `?` for dynamic) |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32...)` | Element data type |

**Syntax:** `!pto.tensor_view<1024x512xf16>`

---

### 2.4 `!pto.partition_tensor_view<d0 x d1 x elementType>`

A logical partition (slice) of a `tensor_view`. Holds shape and stride information for a tile-sized region but does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<int64_t>` | Partition shape `[d0, d1]` |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32...)` | Element data type |

**Syntax:** `!pto.partition_tensor_view<16x16xf16>`

---

### 2.5 `!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`

`pto.tile_buf` represents a local scratchpad memory tile buffer with explicit placement, shape, valid region, and layout/fractal metadata. Based on formats used in `PTOAS/test`, the canonical textual form is a key-value list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `loc` | keyword (`vec/mat/left/right/acc/bias`) | Local memory domain (`vec` maps to UB; use `vec` in textual IR) |
| `dtype` | `element-type(i1/i8/i16/i32/f16/f32...)` | Element data type |
| `rows` | `int64` | Physical row count |
| `cols` | `int64` | Physical column count |
| `v_row` | `int64` or `?` | Valid row count |
| `v_col` | `int64` or `?` | Valid column count |
| `blayout` | `BLayout` mnemonic | Base layout (`row_major` / `col_major`) |
| `slayout` | `SLayout` mnemonic | Secondary layout (`none_box` / `row_major` / `col_major`) |
| `fractal` | `int32` | Fractal size |
| `pad` | `PadValue` mnemonic or integer literal | Padding policy/value selector (tests commonly use `pad=0`) |

Here, `?` denotes a dynamic symbol resolved at runtime.

**Syntax:**
```mlir
// canonical form used in current tests
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>

// legacy form still seen in old samples
!pto.tile_buf<loc=mat, dtype=f32, rows=32, cols=32, blayout=col_major, valid=32x32, slayout=row_major, fractal=512, pad=0>
```

---

## 3. Enums & Attributes

### 3.1 AddressSpace

Defines the physical storage location of a buffer in the Ascend NPU memory hierarchy. This affects which operations are legal on the buffer and how data movement is scheduled (e.g., GM <-> UB, L1 <-> L0).

| Value | Int | Mnemonic | Hardware Mapping |
|-------|-----|----------|-----------------|
| `Zero` | 0 | `zero` | Default (unspecified) |
| `GM` | 1 | `gm` | Global Memory |
| `MAT` | 2 | `mat` | L1 Cache |
| `LEFT` | 3 | `left` | L0A (left matrix buffer) |
| `RIGHT` | 4 | `right` | L0B (right matrix buffer) |
| `ACC` | 5 | `acc` | L0C (accumulator) |
| `VEC` | 6 | `vec` | UB (unified buffer) |
| `BIAS` | 7 | `bias` | Bias buffer |

**Attribute syntax:** `loc=<mnemonic>` (for example, `loc=vec`)

---

### 3.2 PipeEventKind

Defines intra-core pipeline synchronization event kinds in PTO IR, used to express dependencies between pipelines (for example, in [`pto.record_event`](#ptorecord_event) and [`pto.wait_event`](#ptowait_event)).

| Value | Int | Description |
|-------|-----|-------------|
| `EVENT_LOAD_FROM_GM` | 0 | Load from GM |
| `EVENT_STORE_FROM_ACC` | 1 | Store from accumulator |
| `EVENT_STORE_FROM_VEC` | 2 | Store from vector/UB |
| `EVENT_MOVE_MAT_TO_LEFT` | 3 | Move: MAT -> LEFT |
| `EVENT_MOVE_MAT_TO_SCALAR` | 4 | Move: MAT -> scalar |
| `EVENT_MOVE_MAT_TO_BIAS` | 5 | Move: MAT -> BIAS |
| `EVENT_MOVE_MAT_TO_VEC` | 6 | Move: MAT -> VEC |
| `EVENT_MOVE_VEC_TO_MAT` | 7 | Move: VEC -> MAT |
| `EVENT_COMPUTE_MATMUL` | 8 | Matrix multiplication |
| `EVENT_COMPUTE_VEC` | 9 | Vector operation |
| `EVENT_VEC_WAITPOINT` | 10 | Vector wait event |

**Attribute syntax:** `#pto.pipe_event_type<EVENT_LOAD_FROM_GM>`

---

### 3.3 EVENT (Hardware Event IDs)

8 hardware event IDs for synchronization primitives.

| Value | Int |
|-------|-----|
| `EVENT_ID0` - `EVENT_ID7` | 0 - 7 |

**Attribute syntax:** `#pto.event<EVENT_ID0>`

---

### 3.4 Tile Buf config

Composite attribute and component enums for tile buffer configuration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bLayout` | `BLayoutAttr` | Base layout (RowMajor / ColMajor) |
| `sLayout` | `SLayoutAttr` | Secondary layout (NoneBox / RowMajor / ColMajor) |
| `sFractalSize` | `IntegerAttr (i32)` | Secondary fractal size |
| `pad` | `PadValueAttr` | Pad value policy |

**Syntax:** `#pto.tile_buf_config<row_major, none_box, 16, zero>`

**BLayout** (Base layout):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `RowMajor` | 0 | `row_major` |
| `ColMajor` | 1 | `col_major` |

**SLayout** (Secondary layout):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `NoneBox` | 0 | `none_box` |
| `RowMajor` | 1 | `row_major` |
| `ColMajor` | 2 | `col_major` |

**PadValue** (Pad value policy):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `Null` | 0 | `null` |
| `Zero` | 1 | `zero` |
| `Max` | 2 | `max` |
| `Min` | 3 | `min` |

---

### 3.5 Layout

Global tensor layout inference for [`tensor_view` (Section 2.3)](#23-ptotensor_viewd0-x-d1-x-elementtype)/[`partition_tensor_view` (Section 2.4)](#24-ptopartition_tensor_viewd0-x-d1-x-elementtype). Tile buffers additionally use **Tile Buf config** (see 3.4) to describe physical/fractal layout.

| Value | Int | Mnemonic | Description |
|-------|-----|----------|-------------|
| `ND` | 0 | `nd` | Row-major (Normal-Dimension) |
| `DN` | 1 | `dn` | Column-major (Dimension-Normal) |
| `NZ` | 2 | `nz` | Fractal/blocked layout |

**Attribute syntax:** `#pto.layout<nd>`

---

## 4. Operations Reference

### 4.1 Pointer & View Operations

##### `pto.make_tensor_view` - Create Tensor View

**Summary:** Constructs a global tensor view from a pointer, declaring the physical base and strides (no allocation, no data movement).

**Semantics:**

```
result = tensor_view(ptr, shape, strides, layout)
```

This operation defines the physical "base" and stride rules for global memory. It is the reference view for all subsequent partitioning, and it does not move any data.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `AnyType` | Source pointer |
| `shape` | `Variadic<Index>` | Dynamic shape dimensions |
| `strides` | `Variadic<Index>` | Dynamic strides |
| `layout` | `LayoutAttr` (optional) | ND/DN/NZ layout hint |

**Results:** `!pto.tensor_view<...>`

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `ptr` must be `!pto.ptr<...>` and its element type must match the result element type
  - `shape` and `strides` operand counts must match the tensor_view rank
  - If `layout` is provided with static shapes/strides, it must be consistent with inferred layout

**Notes:**

- Stride patterns may allow the compiler to infer hardware layout hints (e.g., `layout = nz`) to guide later DMA operations.

**Hardware Mapping:**

- No hardware pipeline (metadata/view construction only)

**Basic Example:**

```mlir
%tv = pto.make_tensor_view %ptr, [%m, %n], [%s0, %s1] : !pto.ptr<f16> -> !pto.tensor_view<? x ? x f16>
```

---

##### `pto.partition_view` - Partition Tensor View

**Summary:** Creates a logical window on a tensor_view using offsets and sizes, producing a `partition_tensor_view`.

**Semantics:**

```
result = source[offsets, sizes]
```

This op captures both static and dynamic shapes. It represents a logical slice without moving data.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `TensorViewType` | Input tensor view |
| `offsets` | `Variadic<Index>` | Dynamic offsets |
| `sizes` | `Variadic<Index>` | Dynamic sizes |

**Results:** `!pto.partition_tensor_view<...>`

**Constraints & Verification:**

- No custom verifier beyond type consistency
- `offsets`/`sizes` counts must match the rank of `source`

**Notes:**

- Pointer arithmetic is modeled as `BasePtr + Offset`, and the logical shape is determined by `sizes`.

**Hardware Mapping:**

- No hardware pipeline (metadata/view construction only)

**Basic Example:**

```mlir
%pv = pto.partition_view %tv, offsets=[%off0, %off1], sizes=[%s0, %s1]
       : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

##### `pto.alloc_tile` - Allocate Tile Buffer

**Summary:** Declares the lifetime of a tile buffer. Each `alloc_tile` produces an independent tile buffer instance.

**Semantics:**

```
result = alloc_tile(base_addr, valid_row, valid_col)   // operands are optional
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `base_addr` | `Optional<I64>` | Optional start address for the tile buffer |
| `valid_row` | `Optional<Index>` | Dynamic valid row count |
| `valid_col` | `Optional<Index>` | Dynamic valid column count |

**Results:** `!pto.tile_buf<...>`

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - If result `v_row`/`v_col` are dynamic (`?`), the corresponding operands must be present
  - If result `v_row`/`v_col` are static, the corresponding operands must be absent
- If `base_addr` is omitted, the address is assigned by the compiler

**Hardware Mapping:**

- No hardware pipeline (allocation/metadata op)

**Basic Example:**

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

##### `pto.tile_subview` - Subview Tile View

**Summary:** Creates a subview from a parent tile. The result tile buffer is a logical subset of the input tile buffer.

**Semantics:**

```
result = source[offsets] with static sizes
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `pto.tile_buf` | Parent tile buffer |
| `offsets` | `Variadic<Index>` | Runtime dynamic offsets [i, j] |
| `sizes` | `I64ArrayAttr` | Static shape [rows, cols] |

**Results:** `pto.tile_buf`

**Constraints & Verification:**

- No custom verifier beyond type consistency
- Result shape is defined by `sizes` and must be rank-2

**Hardware Mapping:**

- No hardware pipeline (view construction only)

**Basic Example:**

```mlir
%sub = pto.tile_subview %src[%i, %j] sizes [32, 32] : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=64, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

### 4.2 DMA Data Movement Operations

#### PadMode

Padding mode for load operations.

| Value | Int | Description |
|-------|-----|-------------|
| `PadNull` | 0 | No padding |
| `PadFirstElem` | 1 | Pad using the first element |
| `PadValue` | 2 | Pad using a specified value |

---

##### `pto.tload` - Load Partition View to Tile

**Summary:** Physical DMA transfer from a global partition view into a local tile buffer.

**Semantics:**

```
For each element (i, j) in the tile valid region:
    dst[i, j] = src[i, j]
```

`partition_tensor_view` and `tile_buf` are both 2-D in this IR profile. `pto.tload` moves data from the global logical view into the local physical tile buffer.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `PartitionTensorViewType` | Source partition view |
| `dst` | `pto.tile_buf` | Destination tile buffer |
| `pad_mode` | `PadModeAttr` (optional) | Padding mode |
| `pad_value` | `AnyType` (optional) | Padding value |
| `left_padding_num` | `Index` (optional) | Left padding count |
| `right_padding_num` | `Index` (optional) | Right padding count |
| `init_out_buffer` | `BoolAttr` (default: false) | Initialize output buffer |
| `init_condition` | `AnyType` (optional) | Init condition |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` is `!pto.partition_tensor_view`, `dst` is `!pto.tile_buf`
  - `dst` is rank-2 and valid shape is rank-2
  - The product of `sizes` in `partition_view` equals `dst.v_row * dst.v_col` (when statically known)

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`, GM -> UB)

**Basic Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.tstore` - Store Tile to Partition View

**Summary:** Stores a 2-D tile buffer back to a 2-D partition view.

**Semantics:**

```
For each element (i, j) in the tile valid region:
    dst[i, j] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `PartitionTensorViewType` | Destination partition view |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` is `!pto.tile_buf`, `dst` is `!pto.partition_tensor_view`
  - `src` is rank-2 and valid shape is rank-2
  - The product of `dst` partition sizes equals `src.v_row * src.v_col` (when statically known)

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`, UB -> GM)

**Basic Example:**

```mlir
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
```

---

##### `pto.tmov` - Tile Move Between Local Domains

**Summary:** Moves data between local memory domains (e.g., ACC <-> VEC) using tile buffers.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` are tile_buf types
  - `src`/`dst` ranks and element types match

**Notes:**

- `pto.tmov` targets tile_buf-to-tile_buf local domain moves.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.ttrans` - Transpose Tile

**Summary:** Transposes a tile buffer, using a temporary buffer (tmp is required, TBD).

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[j, i]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier that checks tile_buf types and element type consistency
- `tmp` is currently required (TBD)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

### 4.3 Matrix Compute Operations

##### `pto.tmatmul` - Matrix Multiply (Tile World)

**Summary:** Matrix multiplication producing an accumulator tile.

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix (L0A) |
| `rhs` | `pto.tile_buf` | Right matrix (L0B) |
| `dst` | `pto.tile_buf` | Destination (L0C accumulator) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier (shape/layout/element type legality is target-defined)

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul ins(%a, %b : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                          !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
            outs(%c : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.acc` - Matrix Multiply with Accumulation

**Summary:** Matrix multiplication with accumulation (`C = C_in + A * B`).

**Semantics:**

```
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Previous accumulator value |
| `lhs` | `pto.tile_buf` | Left matrix |
| `rhs` | `pto.tile_buf` | Right matrix |
| `dst` | `pto.tile_buf` | Destination accumulator |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
                               !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                               !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
               outs(%c_out : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.bias` - Matrix Multiply with Bias

**Summary:** Matrix multiplication with bias addition (`C = A * B + bias`).

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `rhs` | `pto.tile_buf` | Right matrix |
| `bias` | `pto.tile_buf` | Bias tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                                   !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>,
                                   !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%c : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.mx` - Mixed-Precision Matrix Multiply

**Summary:** Matrix multiplication with additional scaling tiles for mixed-precision/quantized matmul.

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
// scaling tiles configure target-defined quantization behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier (scaling tile legality is target-defined)

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                               !pto.tile_buf<...>, !pto.tile_buf<...>)
               outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tmatmul.mx.acc` - Mixed-Precision Matmul with Accumulation

**Summary:** Mixed-precision matrix multiplication with accumulation.

**Semantics:**

```
dst = acc_in + (lhs * rhs)   // scaling tiles configure target-defined behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Accumulator input |
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                      !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                   outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tmatmul.mx.bias` - Mixed-Precision Matmul with Bias

**Summary:** Mixed-precision matrix multiplication with bias addition.

**Semantics:**

```
dst = (lhs * rhs) + bias   // scaling tiles configure target-defined behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `bias` | `pto.tile_buf` | Bias tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                            !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                    outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv` - Matrix-Vector Multiply

**Summary:** General matrix-vector multiplication.

**Semantics:**

```
For each row i:
    dst[i, 0] = sum_j lhs[i, j] * rhs[j, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.acc` - Matrix-Vector Multiply with Accumulation

**Summary:** Matrix-vector multiplication with accumulation.

**Semantics:**

```
dst = acc_in + (lhs * rhs)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Accumulator input |
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.bias` - Matrix-Vector Multiply with Bias

**Summary:** Matrix-vector multiplication with bias addition.

**Semantics:**

```
dst = (lhs * rhs) + bias
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `bias` | `pto.tile_buf` | Bias vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%c : !pto.tile_buf<...>)
```

---

### 4.4 Vector Arithmetic Operations

All vector arithmetic operations execute on the **Vector pipeline** (`PIPE_V`) and use `ins`/`outs` with tile buffers in the **VEC (UB)** memory space.

#### Binary Tile-Tile Operations

| Op | Semantics |
|----|----------|
| `pto.tadd` | `dst[i,j] = src0[i,j] + src1[i,j]` |
| `pto.tsub` | `dst[i,j] = src0[i,j] - src1[i,j]` |
| `pto.tmul` | `dst[i,j] = src0[i,j] * src1[i,j]` |
| `pto.tdiv` | `dst[i,j] = src0[i,j] / src1[i,j]` |
| `pto.tmax` | `dst[i,j] = max(src0[i,j], src1[i,j])` |
| `pto.tmin` | `dst[i,j] = min(src0[i,j], src1[i,j])` |
| `pto.trem` | `dst[i,j] = fmod(src0[i,j], src1[i,j])` |
| `pto.tpartadd` | Partial elementwise add |
| `pto.tpartmax` | Partial elementwise max |
| `pto.tpartmin` | Partial elementwise min |
| `pto.tprelu` | `dst[i,j] = src0[i,j] > 0 ? src0[i,j] : src1[i,j] * src0[i,j]` |

---

##### `pto.tadd` - Elementwise Add of Two Tiles

**Summary:** Adds two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- Implements `OpPipeInterface`

**Basic Example:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tsub` - Elementwise Subtract of Two Tiles

**Summary:** Subtracts two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Minuend tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsub ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmul` - Elementwise Multiply of Two Tiles

**Summary:** Multiplies two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tdiv` - Elementwise Division of Two Tiles

**Summary:** Divides two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, j]
```

Division-by-zero behavior is target-defined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tdiv ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmax` - Elementwise Maximum of Two Tiles

**Summary:** Computes the element-wise maximum of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmin` - Elementwise Minimum of Two Tiles

**Summary:** Computes the element-wise minimum of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.trem` - Elementwise Remainder of Two Tiles

**Summary:** Computes the element-wise floating-point remainder of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trem ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src0`, `src1`, and `dst` must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trem ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tpartadd` - Partial Elementwise Add

**Summary:** Partial elementwise add with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] + src1[i, j]
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

When `src0` and `src1` have different valid regions, the behavior in non-overlapping areas is implementation-defined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tpartadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tpartmax` - Partial Elementwise Max

**Summary:** Partial elementwise max with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = max(src0[i, j], src1[i, j])
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tpartmin` - Partial Elementwise Min

**Summary:** Partial elementwise min with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = min(src0[i, j], src1[i, j])
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tprelu` - Parametric ReLU with Per-Element Slope

**Summary:** Applies the Parametric ReLU activation function with a per-element slope tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src1[i, j] * src0[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (input activations) |
| `src1` | `pto.tile_buf` | Slope tile buffer (per-element negative slopes) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tprelu ins(<src0>, <src1> : <src0_type>, <src1_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - All operands must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tprelu ins(%a, %slopes : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

#### Tile-Scalar Operations

| Op | Semantics |
|----|----------|
| `pto.tadds` | `dst[i,j] = src[i,j] + scalar` |
| `pto.tsubs` | `dst[i,j] = src[i,j] - scalar` |
| `pto.tmuls` | `dst[i,j] = src[i,j] * scalar` |
| `pto.tdivs` | `dst[i,j] = src[i,j] / scalar` (or `scalar / src[i,j]`) |
| `pto.tmaxs` | `dst[i,j] = max(src[i,j], scalar)` |
| `pto.tmins` | `dst[i,j] = min(src[i,j], scalar)` |
| `pto.trems` | `dst[i,j] = fmod(src[i,j], scalar)` |

---

##### `pto.tadds` - Elementwise Add Scalar to Tile

**Summary:** Adds a scalar value to every element of a tile buffer.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] + scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer containing the input data |
| `scalar` | `AnyType` (e.g. `f32`, `f16`) | Scalar value to add to each element |
| `dst` | `pto.tile_buf` | Destination tile buffer for the result |

**Results:** None. The operation writes results into `dst` following the Destination-Passing Style (DPS) pattern.

**Assembly Format:**

```
pto.tadds ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - `scalar` type must be compatible with the element type of the tile buffers
- Tile operands are `tile_buf` types; scalar is a builtin scalar type (e.g. `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (Unified Buffer / UB)** memory space (`AddressSpace::VEC`)
- The source and destination tile buffers should reside in `VEC` memory (loaded via `tload` from Global Memory)

**Basic Example:**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubs` - Elementwise Subtract Scalar from Tile

**Summary:** Subtracts a scalar value from every element of a tile buffer.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] - scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `F32` | Scalar value to subtract |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - Tile operands are `tile_buf` types; scalar is a builtin scalar type (currently `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmuls` - Elementwise Multiply Tile by Scalar

**Summary:** Multiplies every element of a tile buffer by a scalar value.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] * scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `F32` | Scalar multiplier |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmuls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - Tile operands are rank-2 `tile_buf` types
  - Tile operands are `tile_buf` types; scalar is a builtin scalar type (currently `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmuls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tdivs` - Elementwise Division with Scalar

**Summary:** Divides every element of a tile buffer by a scalar, or divides a scalar by every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] / scalar    (default)
    dst[i, j] = scalar / src[i, j]    (reverse mode)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `AnyType` | Source tile buffer |
| `scalar` | `AnyType` | Scalar divisor (or dividend in reverse mode) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
// Tile / scalar
pto.tdivs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)

// Scalar / tile (reverse mode)
pto.tdivs ins(<scalar>, <src> : <scalar_type>, <src_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - Exactly one operand is a `tile_buf`, the other is a scalar
  - `dst` must be a `tile_buf` type
  - Tile operand and `dst` must have the same shape and element type
  - Scalar type must be integer, float, or index, and must match the tile element type

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// tile / scalar
pto.tdivs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)

// scalar / tile (reverse mode)
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmaxs` - Elementwise Max of Tile and Scalar

**Summary:** Computes the element-wise maximum between a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `F32` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmaxs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - Tile operands are rank-2 `tile_buf` types
  - Tile operands are `tile_buf` types; scalar is a builtin scalar type (currently `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmaxs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmins` - Elementwise Min of Tile and Scalar

**Summary:** Computes the element-wise minimum between a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `F32` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmins ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - Tile operands are rank-2 `tile_buf` types
  - Tile operands are `tile_buf` types; scalar is a builtin scalar type (currently `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmins ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.trems` - Elementwise Remainder with Scalar

**Summary:** Computes the element-wise floating-point remainder of a tile divided by a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `F32` | Scalar divisor |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trems ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types
  - Tile operands are rank-2 `tile_buf` types
  - Scalar must be a floating-point type (textual form currently uses `f32`)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trems ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### Ternary Operations

| Op | Semantics |
|----|----------|
| `pto.taddc` | `dst = src0 + src1 + src2` |
| `pto.tsubc` | `dst = src0 - src1 + src2` |
| `pto.taddsc` | `dst = src0 + scalar + src1` |
| `pto.tsubsc` | `dst = src0 - scalar + src1` |

---

##### `pto.taddc` - Elementwise Ternary Add of Tiles

**Summary:** Adds three tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j] + src2[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `src2` | `pto.tile_buf` | Third source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taddc ins(<src0>, <src1>, <src2> : <type0>, <type1>, <type2>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - All operands must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.taddc ins(%a, %b, %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubc` - Elementwise Ternary Subtract-Add

**Summary:** Computes `src0 - src1 + src2` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j] + src2[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `src2` | `pto.tile_buf` | Addend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubc ins(<src0>, <src1>, <src2> : <type0>, <type1>, <type2>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - All operands must have same shapes and compatible element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubc ins(%a, %b, %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.taddsc` - Fused Add-Scalar-Add

**Summary:** Computes `src0 + scalar + src1` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + scalar + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `scalar` | `AnyType` | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taddsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.taddsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubsc` - Fused Subtract-Scalar-Add

**Summary:** Computes `src0 - scalar + src1` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - scalar + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `scalar` | `F32` | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### Unary Operations

| Op | Semantics |
|----|----------|
| `pto.tabs` | `dst[i,j] = abs(src[i,j])` |
| `pto.tneg` | `dst[i,j] = -src[i,j]` |
| `pto.texp` | `dst[i,j] = exp(src[i,j])` |
| `pto.tlog` | `dst[i,j] = ln(src[i,j])` |
| `pto.tsqrt` | `dst[i,j] = sqrt(src[i,j])` |
| `pto.trsqrt` | `dst[i,j] = 1/sqrt(src[i,j])` |
| `pto.trecip` | `dst[i,j] = 1/src[i,j]` |
| `pto.trelu` | `dst[i,j] = max(0, src[i,j])` |
| `pto.tlrelu` | `dst[i,j] = src[i,j] > 0 ? src[i,j] : slope * src[i,j]` |

---

##### `pto.tabs` - Elementwise Absolute Value

**Summary:** Computes the absolute value of every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = |src[i, j]|
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tabs ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Implements `OpPipeInterface`
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tneg` - Elementwise Negation

**Summary:** Negates every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = -src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tneg ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tneg ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.texp` - Elementwise Exponential

**Summary:** Computes the exponential function for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = exp(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.texp ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.texp ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tlog` - Elementwise Natural Logarithm

**Summary:** Computes the natural logarithm for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = ln(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tlog ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tlog ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tsqrt` - Elementwise Square Root

**Summary:** Computes the square root for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = sqrt(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsqrt ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.trsqrt` - Elementwise Reciprocal Square Root

**Summary:** Computes the reciprocal square root for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trsqrt ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trecip` - Elementwise Reciprocal

**Summary:** Computes the reciprocal for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = 1.0 / src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trecip ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trecip ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trelu` - Elementwise ReLU

**Summary:** Applies the Rectified Linear Unit activation function to every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(0, src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trelu ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trelu ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tlrelu` - Leaky ReLU with Scalar Slope

**Summary:** Applies the Leaky ReLU activation function with a scalar slope parameter.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `slope` | `F32` | Negative slope coefficient |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tlrelu ins(<src>, <slope> : <src_type>, <slope_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `src` and `dst` must have same shapes and element types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tlrelu ins(%a, %slope : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>, f32)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### 4.5 Reduction Operations

Reduce along rows or columns of a tile. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowsum` | `dst[i,0] = sum_j src[i,j]` |
| `pto.trowmax` | `dst[i,0] = max_j src[i,j]` |
| `pto.trowmin` | `dst[i,0] = min_j src[i,j]` (requires tmp) |
| `pto.tcolsum` | `dst[0,j] = sum_i src[i,j]` (requires tmp, optional isBinary) |
| `pto.tcolmax` | `dst[0,j] = max_i src[i,j]` |
| `pto.tcolmin` | `dst[0,j] = min_i src[i,j]` |

---

##### `pto.trowsum` - Row-wise Sum Reduction

**Summary:** Reduces each row by summing across columns.

**Semantics:**

```
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier
- `dst` should have a single column (or compatible shape for row reduction output)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowmax` - Row-wise Max Reduction

**Summary:** Reduces each row by taking the maximum across columns.

**Semantics:**

```
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowmin` - Row-wise Min Reduction

**Summary:** Reduces each row by taking the minimum across columns. Requires a temporary buffer.

**Semantics:**

```
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Temporary buffer (required for intermediate computation) |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.tcolsum` - Column-wise Sum Reduction

**Summary:** Reduces each column by summing across rows. Requires a temporary buffer.

**Semantics:**

```
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Temporary buffer (required for intermediate computation) |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |
| `isBinary` | `BoolAttr` (default: `false`) | Use binary reduction tree |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolsum ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>) isBinary = false
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolsum ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>) isBinary = false
```

---

##### `pto.tcolmax` - Column-wise Max Reduction

**Summary:** Reduces each column by taking the maximum across rows.

**Semantics:**

```
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.tcolmin` - Column-wise Min Reduction

**Summary:** Reduces each column by taking the minimum across rows.

**Semantics:**

```
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### 4.6 Broadcast Operations

Broadcast values across rows or columns. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowexpand` | Broadcast `src[i,0]` across row `i` |
| `pto.tcolexpand` | Broadcast `src[0,j]` across column `j` |
| `pto.trowexpandmul` | `dst[i,j] = src0[i,j] * src1[i,0]` |
| `pto.trowexpanddiv` | `dst[i,j] = src0[i,j] / src1[i,0]` |
| `pto.trowexpandsub` | `dst[i,j] = src0[i,j] - src1[i,0]` |
| `pto.texpands` | Broadcast scalar to all elements of dst |

---

##### `pto.trowexpand` - Row-wise Broadcast

**Summary:** Broadcasts the first element of each source row across the entire destination row.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer (column vector) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

##### `pto.tcolexpand` - Column-wise Broadcast

**Summary:** Broadcasts the first element of each source column across the entire destination column.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer (row vector) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

##### `pto.trowexpandmul` - Row-wise Broadcast Multiply

**Summary:** Multiplies each row of `src0` by a per-row scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpanddiv` - Row-wise Broadcast Divide

**Summary:** Divides each row of `src0` by a per-row scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector (divisor) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandsub` - Row-wise Broadcast Subtract

**Summary:** Subtracts a per-row scalar from `src1` from each row of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector (subtrahend) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.texpands` - Broadcast Scalar to Tile

**Summary:** Broadcasts a scalar value to all elements of a destination tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `scalar` | `AnyTypeOf<[F16, F32, I16, I32, I8, UI8, UI16, UI32]>` | Scalar value to broadcast |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.texpands ins(<scalar> : <scalar_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier
- Supported scalar types: `f16`, `f32`, `i16`, `i32`, `i8`, `ui8`, `ui16`, `ui32`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- Has `MemWrite` memory effect

**Basic Example:**

```mlir
pto.texpands ins(%scalar : f32)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### 4.7 Compare & Select Operations

#### CmpMode

Comparison modes for `pto.tcmp` / `pto.tcmps`.

| Value | Int | Mnemonic |
|-------|-----|----------|
| `EQ` | 0 | `equal` |
| `NE` | 1 | `not_equal` |
| `LT` | 2 | `less_than` |
| `LE` | 3 | `less_equal` |
| `GT` | 4 | `greater_than` |
| `GE` | 5 | `greater_equal` |

**Attribute syntax:** `#pto<cmp less_than>`

---

#### `pto.tcmp`

**Summary:** Compares two tiles element-wise and writes a packed predicate mask.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = (src0[i, j] <cmpMode> src1[i, j]) ? 1 : 0
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First operand |
| `src1` | `pto.tile_buf` | Second operand |
| `dst` | `pto.tile_buf` | Destination mask |
| `cmpMode` | `CmpModeAttr` (optional) | Comparison mode (EQ/NE/LT/LE/GT/GE) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcmp ins(<src0>, <src1> {cmpMode = <mode>} : <type0>, <type1>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcmp ins(%a, %b {cmpMode = #pto<cmp less_than>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### `pto.tcmps`

**Summary:** Compares a tile against a scalar value element-wise.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = (src[i, j] <cmpMode> scalar) ? 1 : 0
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Tile operand |
| `scalar` | `AnyFloat/AnyInteger/Index` | Scalar value to compare against |
| `cmpMode` | `CmpModeAttr` (default: EQ) | Comparison mode |
| `dst` | `pto.tile_buf` | Destination mask |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcmps ins(%a, %s {cmpMode = #pto<cmp less_than>} :
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f16)
          outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### `pto.tsel`

**Summary:** Selects between two tiles using a mask tile (per-element selection).

**Semantics:**

```
For each element (i, j):
    dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mask` | `pto.tile_buf` | Predicate mask |
| `src0` | `pto.tile_buf` | Value when mask is true |
| `src1` | `pto.tile_buf` | Value when mask is false |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsel ins(<mask>, <src0>, <src1> : <mask_type>, <type0>, <type1>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsel ins(%mask, %a, %b : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### `pto.tsels`

**Summary:** Selects one of two source tiles using a scalar selection mode (global select).

**Semantics:**

```
dst = (selectMode != 0) ? src0 : src1
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile |
| `src1` | `pto.tile_buf` | Second source tile |
| `selectMode` | `AnyInteger` | Selection mode (scalar) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsels ins(%a, %b, %mode : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### 4.8 Bitwise Operations

All bitwise operations execute on the **Vector pipeline** (`PIPE_V`) and operate on data in the **VEC (UB)** memory space.

#### Binary Tile-Tile Bitwise

| Op | Semantics |
|----|----------|
| `pto.tand` | `dst = src0 & src1` |
| `pto.tor` | `dst = or(src0, src1)` |
| `pto.txor` | `dst = src0 ^ src1` |
| `pto.tshl` | `dst = src0 << src1` |
| `pto.tshr` | `dst = src0 >> src1` |

---

##### `pto.tand` - Elementwise Bitwise AND

**Summary:** Computes the bitwise AND of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] & src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tand ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tor` - Elementwise Bitwise OR

**Summary:** Computes the bitwise OR of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] | src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>,
            !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

##### `pto.txor` - Elementwise Bitwise XOR

**Summary:** Computes the bitwise XOR of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] ^ src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.txor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tshl` - Elementwise Shift Left

**Summary:** Shifts each element of `src0` left by the corresponding element of `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] << src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (values to shift) |
| `src1` | `pto.tile_buf` | Shift amount tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshl ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tshr` - Elementwise Shift Right

**Summary:** Shifts each element of `src0` right by the corresponding element of `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] >> src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (values to shift) |
| `src1` | `pto.tile_buf` | Shift amount tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshr ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### Unary Bitwise

##### `pto.tnot` - Elementwise Bitwise NOT

**Summary:** Computes the bitwise NOT of every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = ~src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tnot ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### Tile-Scalar Bitwise

| Op | Semantics |
|----|----------|
| `pto.tands` | `dst = src & scalar` |
| `pto.tors` | `dst = or(src, scalar)` |
| `pto.txors` | `dst = src ^ scalar` |
| `pto.tshls` | `dst = src << scalar` |
| `pto.tshrs` | `dst = src >> scalar` |

---

##### `pto.tands` - Bitwise AND with Scalar

**Summary:** Computes the bitwise AND of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] & scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnyType` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tors` - Bitwise OR with Scalar

**Summary:** Computes the bitwise OR of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] | scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.txors` - Bitwise XOR with Scalar

**Summary:** Computes the bitwise XOR of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] ^ scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnyInteger` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.txors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tshls` - Shift Left by Scalar

**Summary:** Shifts each element of a tile left by a scalar amount.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] << scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Shift amount |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tshrs` - Shift Right by Scalar

**Summary:** Shifts each element of a tile right by a scalar amount.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] >> scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Shift amount |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshrs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### 4.9 Data Rearrangement Operations

#### MaskPattern

Predefined mask patterns for gather operations.

| Value | Int | Pattern |
|-------|-----|---------|
| `P0101` | 0 | Alternating 0-1-0-1 |
| `P0011` | 1 | 0-0-1-1 |
| `P0110` | 2 | 0-1-1-0 |
| `P0001` | 3 | 0-0-0-1 |
| `P1111` | 4 | All ones |

---

##### `pto.tgather` - Gather/Select Elements

**Summary:** Gathers elements from a source tile using indices or a mask pattern.

**Semantics:**

```
If indices are provided:
    dst[i, j] = src[indices[i, j]]
Else (mask pattern):
    dst[i, j] = src[...] according to mask pattern
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indices` | `Optional<pto.tile_buf>` | Index tile (index gather) |
| `maskPattern` | `MaskPatternAttr` (optional) | Mask pattern (mask gather) |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier (exact legality of indices/masks is implementation-defined)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tgather ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tgatherb` - Gather by Byte Offsets

**Summary:** Gathers elements using per-element byte offsets.

**Semantics:**

```
dst[i, j] = src[byte_offsets[i, j]]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `offsets` | `pto.tile_buf` | Byte offset tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tgatherb ins(%src, %offs : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tscatter` - Scatter Rows

**Summary:** Scatters rows from a source tile into a destination tile using per-row indices.

**Semantics:**

```
dst[row_index[i], j] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexes` | `pto.tile_buf` | Row index tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.mgather` - Gather-Load from Global Memory

**Summary:** Loads elements from global memory into a tile using per-element indices.

**Semantics:**

```
dst[i, j] = mem[idx[i, j]]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mem` | `AnyMemRef/pto.tile_buf` | Source memory |
| `idx` | `pto.tile_buf` | Index tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`)

**Basic Example:**

```mlir
pto.mgather ins(%mem, %idx : memref<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.mscatter` - Scatter-Store to Global Memory

**Summary:** Stores elements from a tile into global memory using per-element indices.

**Semantics:**

```
mem[idx[i, j]] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `idx` | `pto.tile_buf` | Index tile |
| `mem` | `AnyMemRef/pto.tile_buf` | Destination memory |

**Results:** None. Writes into `mem` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`)

**Basic Example:**

```mlir
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : memref<...>)
```

---

##### `pto.treshape` - Reinterpret Tile Shape/Layout

**Summary:** Reinterprets a tile buffer with a new shape/layout (no data movement).

**Semantics:**

```
dst = reinterpret(src)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (different shape) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier that requires tile_buf types

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.textract` - Extract Sub-Tile Window

**Summary:** Extracts a sub-tile window from a source tile into a destination tile.

**Semantics:**

```
dst[i, j] = src[i + indexRow, j + indexCol]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexRow` | `Index` | Starting row |
| `indexCol` | `Index` | Starting column |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.textract ins(%src[%row, %col] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tfillpad` - Fill Padding Region

**Summary:** Copies `src` into `dst` and fills padded elements using `dst`'s PadVal.

**Semantics:**

```
For valid elements: dst = src
For padded elements: dst = PadVal(dst)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (with pad config) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### 4.10 Sorting Operations

##### `pto.tsort32` - Sort Fixed 32-Element Blocks

**Summary:** Sorts fixed-size 32-element blocks and produces an index mapping.

**Semantics:**

```
dst = sort(src)
idx = permutation indices for the sort
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Input tile |
| `dst` | `pto.tile_buf` | Sorted output |
| `idx` | `pto.tile_buf` | Index mapping output |

**Results:** None. Writes into `dst`/`idx` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier
- Sorting granularity and layout constraints are target-defined

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tsort32 ins(%src : !pto.tile_buf<...>)
           outs(%dst, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
```

---

##### `pto.tmrgsort` - Merge Sort

**Summary:** Performs merge sort on one or more sorted lists (implementation-defined layout).

**Semantics:**

```
dst = merge_sort(src, blockLen)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Input tile |
| `dst` | `pto.tile_buf` | Output tile |
| `blockLen` | `I32Attr` | Block length for merge |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tmrgsort ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>) blockLen = 32
```

---

### 4.11 Type Conversion

#### RoundMode

Rounding modes for type conversion (`pto.tcvt`) operations.

| Value | Int | Description |
|-------|-----|-------------|
| `NONE` | 0 | No rounding |
| `RINT` | 1 | Round to nearest integer |
| `ROUND` | 2 | Round half away from zero |
| `FLOOR` | 3 | Round toward negative infinity |
| `CEIL` | 4 | Round toward positive infinity |
| `TRUNC` | 5 | Truncate toward zero |
| `ODD` | 6 | Round to odd |
| `CAST_RINT` | 7 | Cast with round-to-nearest (default) |

**Attribute syntax:** `#pto<round_mode FLOOR>`

---

##### `pto.tcvt` - Elementwise Type Conversion

**Summary:** Converts each element to a new type with a specified rounding mode.

**Semantics:**

```
dst[i, j] = cast(src[i, j], rmode)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (different element type) |
| `rmode` | `RoundModeAttr` (default: `CAST_RINT`) | Rounding mode |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.12 Integer Sequence Generation Operations

##### `pto.tci` - Contiguous Integer Sequence

**Summary:** Generates a contiguous integer sequence into a destination tile.

**Semantics:**

```
dst[i, j] = S + linear_index(i, j)   // or descending if requested
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `S` | `AnyInteger` | Starting value |
| `dst` | `pto.tile_buf` | Destination tile |
| `descending` | `BoolAttr` (default: false) | Generate descending sequence |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tci ins(%start : i32) outs(%dst : !pto.tile_buf<...>)
```

---

### 4.13 Scalar Element Access

##### `pto.tgetval` - Read Single Element

**Summary:** Reads a single element from a tile or memref at a linear offset.

**Semantics:**

```
result = src[offset]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf / AnyMemRef` | Source tile/memref |
| `offset` | `Index` | Linear element offset |

**Results:** Scalar value (`AnyType`)

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) when operating on tile_buf

**Basic Example:**

```mlir
%val = pto.tgetval ins(%src, %off : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : f16
```

---

##### `pto.tsetval` - Write Single Element

**Summary:** Writes a scalar value into a tile or memref at a linear offset.

**Semantics:**

```
dst[offset] = val
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tile_buf / AnyMemRef` | Destination tile/memref |
| `offset` | `Index` | Linear element offset |
| `val` | `AnyType` | Scalar value to write |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) when operating on tile_buf

**Basic Example:**

```mlir
pto.tsetval ins(%off, %val : index, f16) outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.14 MX Quantized Operations

##### `pto.tmov_fp` - Move/Convert with Scaling Tile

**Summary:** Moves/converts from an accumulator tile using a scaling (`fp`) tile for quantization.

**Semantics:**

```
dst[i, j] = Convert(src[i, j]; fp)   // target-defined quantization/dequantization
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `fp` | `pto.tile_buf` | Scaling (fp) tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier (fp tile legality is target-defined)

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) for accumulator conversion

**Basic Example:**

```mlir
pto.tmov_fp ins(%acc, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tstore_fp` - Store Accumulator with Scaling

**Summary:** Stores an accumulator tile into global memory using a scaling (`fp`) tile.

**Semantics:**

```
dst[...] = Convert(src[i, j]; fp)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source accumulator tile |
| `fp` | `pto.tile_buf` | Scaling tile |
| `dst` | `AnyMemRef` | Destination memory |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier (quantized accumulator store legality is target-defined)

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`)

**Basic Example:**

```mlir
pto.tstore_fp ins(%acc, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst : memref<...>)
```

---

### 4.15 Synchronization Operations

##### `pto.barrier`

**Summary:** Inserts an intra-pipeline memory barrier.

**Semantics:**

```
barrier(pipe)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline to barrier |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Pipeline barrier for the specified pipe

**Basic Example:**

```mlir
pto.barrier #pto.pipe<PIPE_V>
```

---

##### `pto.record_event`

**Summary:** Records an event for synchronization between producer and consumer operation classes.

**Semantics:**

```
record_event(src_op, dst_op, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src_op` | `PipeEventKindAttr` | Source operation type |
| `dst_op` | `PipeEventKindAttr` | Destination operation type |
| `event_id` | `EventAttr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Lowered to pipe/event synchronization primitives

**Basic Example:**

```mlir
pto.record_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

##### `pto.wait_event`

**Summary:** Waits for a recorded event between producer and consumer operation classes.

**Semantics:**

```
wait_event(src_op, dst_op, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src_op` | `PipeEventKindAttr` | Source operation type |
| `dst_op` | `PipeEventKindAttr` | Destination operation type |
| `event_id` | `EventAttr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Lowered to pipe/event synchronization primitives

**Basic Example:**

```mlir
pto.wait_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

#### Cross-Core Synchronization

##### `pto.sync.set`

**Summary:** Sets a synchronization signal between cube and vector cores.

**Semantics:**

```
sync.set(pipe, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline stage |
| `event_id` | `I32Attr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Cross-core synchronization signal

**Basic Example:**

```mlir
pto.sync.set #pto.pipe<PIPE_M>, 0
```

---

##### `pto.sync.wait`

**Summary:** Waits for a synchronization signal between cube and vector cores.

**Semantics:**

```
sync.wait(pipe, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline stage |
| `event_id` | `I32Attr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Cross-core synchronization signal

**Basic Example:**

```mlir
pto.sync.wait #pto.pipe<PIPE_V>, 0
```

---

### 4.16 CV-Related Operations

##### `pto.section.cube` - Core-Specific Section (Cube)

**Summary:** Marks a region of code that should be emitted only for cube cores.

**Semantics:**

```
section.cube { ... }  // lowered to #if defined(CUBE) ... #endif
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- The operation has `SingleBlock` and `NoTerminator` traits

**Hardware Mapping:**

- Compile-time control (lowered to preprocessor guards)

**Basic Example:**

```mlir
pto.section.cube {
  // Cube-core-only operations
  pto.tmatmul ins(...) outs(...)
}
```

---

##### `pto.section.vector` - Core-Specific Section (Vector)

**Summary:** Marks a region of code that should be emitted only for vector cores.

**Semantics:**

```
section.vector { ... }  // lowered to #if defined(VECTOR) ... #endif
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- The operation has `SingleBlock` and `NoTerminator` traits

**Hardware Mapping:**

- Compile-time control (lowered to preprocessor guards)

**Basic Example:**

```mlir
pto.section.vector {
  // Vector-core-only operations
  pto.tadd ins(...) outs(...)
}
```

---

### 4.17 Runtime Intrinsics

##### `pto.get_block_idx`

**Summary:** Returns the current block (core) index.

**Semantics:**

```
result = block_idx()
```

**Arguments:** None.

**Results:** `i64` block index in `[0, BlockNum-1]`.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%idx = pto.get_block_idx
```

---

##### `pto.get_subblock_idx`

**Summary:** Returns the current sub-block (vector core) index.

**Semantics:**

```
result = subblock_idx()
```

**Arguments:** None.

**Results:** `i64` sub-block index.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%idx = pto.get_subblock_idx
```

---

##### `pto.get_block_num`

**Summary:** Returns the total number of blocks (cores).

**Semantics:**

```
result = block_num()
```

**Arguments:** None.

**Results:** `i64` total block count.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%num = pto.get_block_num
```

---

##### `pto.get_subblock_num`

**Summary:** Returns the total number of vector cores (sub-blocks).

**Semantics:**

```
result = subblock_num()
```

**Arguments:** None.

**Results:** `i64` total sub-block count.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%num = pto.get_subblock_num
```

---

### 4.18 Debug Operations

##### `pto.tprint` - Print Tile

**Summary:** Prints the contents of a tile for debugging.

**Semantics:**

```
print(src)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Tile to print |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond type consistency

**Hardware Mapping:**

- Debug/diagnostic intrinsic (implementation-defined)

**Basic Example:**

```mlir
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

## 5. Operation Summary Table

| Category | Count | Pipeline |
|----------|-------|----------|
| Pointer/View | 4 | - |
| DMA Data Movement | 4 | MTE2/MTE3/V |
| Matrix Compute | 9 | M (Cube) |
| Vector Arithmetic & Math | 31 | V (Vector) |
| Reduction | 6 | V |
| Broadcast | 6 | V |
| Compare & Select | 4 | V |
| Bitwise | 11 | V |
| Data Rearrangement | 8 | V |
| Sorting | 2 | V |
| Type Conversion | 1 | V |
| Integer Sequence Generation | 1 | V |
| Scalar Element Access | 2 | V |
| MX Quantized | 2 | M/V |
| Synchronization | 5 | - |
| CV-Related | 2 | - |
| Runtime Intrinsics | 4 | - (Pure) |
| Debug | 1 | - |

**Total: 103 operations**
