# 2. Types & Attributes

> **Category:** Type system and attribute vocabulary

This chapter defines the types and attributes used across the Tile Instruction chapters.

---

## 2.1 Element Types

Element types describe the primitive scalar values stored in tiles; by themselves they do not form a value. Common element categories:

- **Integers:** signless — `i1`, `i8`, `i16`, `i32`, `i64`. Signedness is not encoded in the type; it is selected by operation semantics or attributes.
- **Floating-point:** `f16`, `bf16`, `f32`.
- **Index-like:** `index` values appear as scalar operands (offsets, sizes, scalar compares).

Operation-specific constraints:

- Elementwise ops typically require operand and result element types to match.
- Reductions, math ops, and division typically restrict to floating-point or a subset of integer types.
- Bitwise ops require integer element types.
- `pto.tcvt` defines explicit element-type changes under an explicit rounding mode.

Memory layout and address space do not change element-type semantics; they only affect placement and access patterns.

## 2.2 `!pto.ptr<elementType[, memorySpace]>`

A typed pointer. `memorySpace` is optional and defaults to `gm`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `elementType` | element type | Element type pointed to. |
| `memorySpace` | `gm` \| `vec` | Pointer address space (`gm` → global memory, `vec` → UB / vector memory). |

**Syntax:** `!pto.ptr<f16>` or `!pto.ptr<f16, vec>`

Pointer conversions are modeled explicitly with `pto.castptr`. Between two `!pto.ptr` types, casts are only legal when both pointers stay in the same PTO memory space.

## 2.3 `!pto.tensor_view<d0 x d1 x elementType[, layout]>`

A descriptor for a global-memory tensor. Holds shape information; strides are supplied at `pto.make_tensor_view` construction time. Does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Tensor shape `[d0, d1]` (each dim may be `?`). |
| `elementType` | element type | Element data type. |
| `layout` | `#pto.layout<...>` | Optional resolved GM layout. Omission leaves the layout to construction-time resolution and defaults to ND when no stronger fact exists. |

**Syntax:** `!pto.tensor_view<1024x512xf16>` or
`!pto.tensor_view<1x8x8x16x8xf32, #pto.layout<nz>>`

## 2.4 `!pto.partition_tensor_view<d0 x d1 x elementType[, layout]>`

A logical partition (slice) of a `tensor_view`. Holds shape information for a tile-sized region; strides are inherited from the parent `tensor_view`. Does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Partition shape `[d0, d1]`. |
| `elementType` | element type | Element data type. |
| `layout` | `#pto.layout<...>` | Optional resolved GM layout inherited from the source view. |

**Syntax:** `!pto.partition_tensor_view<16x16xf16>` or
`!pto.partition_tensor_view<1x4x8x16x8xf32, #pto.layout<nz>>`

## 2.5 `!pto.tile_buf<loc, RxCxdtype[, valid=v_rxv_c][, blayout=..., slayout=..., fractal=..., pad=...]>`

`pto.tile_buf` represents a local on-chip tile buffer with explicit placement, shape, valid region, and layout/fractal metadata. The textual form is **compact**: only the leading `<loc, RxCxdtype>` triple is mandatory; everything else is omitted when it equals its default.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loc` | keyword | — | Local memory domain (`vec` / `mat` / `left` / `right` / `acc` / `bias` / `scaling`). |
| `R` × `C` × `dtype` | shape × element type | — | Physical row/column count and element type. |
| `valid` | `v_row x v_col` (each `int64` or `?`) | `R x C` | Valid region. Omitted when equal to physical shape. |
| `blayout` | `BLayout` | `row_major` | Base layout. |
| `slayout` | `SLayout` | `none_box` | Secondary layout. |
| `fractal` | `int32` | `512` | Fractal size. |
| `pad` | `PadValue` enum int | `0` (`null`) | Padding policy/value selector. |

**Examples:**

```mlir
// Default config, valid == physical
!pto.tile_buf<vec, 16x16xf16>

// Dynamic valid region
!pto.tile_buf<vec, 16x16xf16, valid=?x?>

// Non-default config
!pto.tile_buf<vec, 8x8xf32, blayout=col_major, slayout=row_major, fractal=1024, pad=1>
```

`?` denotes a dynamic symbol resolved at runtime (via `pto.alloc_tile` operands or `pto.set_validshape`).

## 2.6 AddressSpace

Defines the physical storage location of a buffer in the Ascend NPU memory hierarchy.

| Value | Int | Mnemonic | Hardware Mapping |
|-------|-----|----------|------------------|
| `Zero` | 0 | `zero` | Default (unspecified). |
| `GM` | 1 | `gm` | Global Memory. |
| `MAT` | 2 | `mat` | L1 Cache. |
| `LEFT` | 3 | `left` | L0A (left matrix buffer). |
| `RIGHT` | 4 | `right` | L0B (right matrix buffer). |
| `ACC` | 5 | `acc` | L0C (accumulator). |
| `VEC` | 6 | `vec` | UB (unified buffer). |
| `BIAS` | 7 | `bias` | Bias buffer. |
| `SCALING` | 8 | `scaling` | Scaling buffer. |

**Attribute syntax:** `loc=<mnemonic>` (for example `loc=vec`).

## 2.7 Tile Buf Config

Composite attribute for tile-buffer layout/fractal/pad.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bLayout` | `BLayoutAttr` | Base layout (RowMajor / ColMajor). |
| `sLayout` | `SLayoutAttr` | Secondary layout (NoneBox / RowMajor / ColMajor). |
| `sFractalSize` | `IntegerAttr (i32)` | Secondary fractal size. |
| `pad` | `PadValueAttr` | Pad value policy. |

**Syntax:** `#pto.tile_buf_config<row_major, none_box, 16, zero>`

**BLayout:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `RowMajor` | 0 | `row_major` |
| `ColMajor` | 1 | `col_major` |

**SLayout:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `NoneBox` | 0 | `none_box` |
| `RowMajor` | 1 | `row_major` |
| `ColMajor` | 2 | `col_major` |

**PadValue:**

| Value | Int | Mnemonic |
|-------|-----|----------|
| `Null` | 0 | `null` |
| `Zero` | 1 | `zero` |
| `Max` | 2 | `max` |
| `Min` | 3 | `min` |

## 2.8 Layout

Global tensor layout attribute for `tensor_view` and `partition_tensor_view`. Tile buffers additionally use **Tile Buf Config** (§2.7) to describe physical/fractal layout.

| Value | Int | Mnemonic | Description |
|-------|-----|----------|-------------|
| `ND` | 0 | `nd` | Row-major (Normal-Dimension). |
| `DN` | 1 | `dn` | Column-major (Dimension-Normal). |
| `NZ` | 2 | `nz` | Fractal / blocked layout. |

**Attribute syntax:** `#pto.layout<nd>`

Layout is preserved when a view crosses `scf.if`, `scf.for`, `scf.while`, a
control-flow block argument, or a direct internal function argument/return.
Every incoming view at a merge must resolve to the same layout. For example,
merging NZ with NZ produces NZ, while merging NZ with ND is invalid. A
view-typed indirect call is invalid because it has no statically resolvable
layout contract.

## 2.9 PadMode (for loads)

Padding mode for `pto.tload`.

| Value | Int | Description |
|-------|-----|-------------|
| `PadNull` | 0 | No padding. |
| `PadFirstElem` | 1 | Pad using the first element. |
| `PadValue` | 2 | Pad using a specified value. |

## 2.10 Shared Scalar and Control-Flow Ops

Tile programs commonly interleave `pto` instructions with a small set of supporting ops:

- **`func`** — `func.func`, `func.return`, `func.call`.
- **`arith`** — scalar constants/casts (`arith.constant`, `arith.index_cast`, `arith.bitcast`, `arith.extf`/`truncf`/…), integer/float arithmetic, bitwise/shift, compares/select, extended and min/max ops.
- **`scf`** — `scf.for`, `scf.if`, `scf.yield`; several other structured control-flow forms are lowered through `cf`.

These supporting ops are included here only insofar as tile programs need function structure, scalar computation, and structured control flow; full coverage of those surfaces is out of scope for this reference.
