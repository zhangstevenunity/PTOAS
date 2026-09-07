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

- **Integers**: signless integers such as `i1/i8/i16/i32`, and, where an op
  surface requires it, explicitly signed/unsigned integers such as
  `si8/ui8/si16/ui16/si32/ui32`. Whether an operation accepts only signless
  integers or also accepts explicit signedness is operation-specific and
  documented per op family.
- **Floating-point**: IEEE floating-point types such as `f16/f32`. Some targets may also support additional formats (e.g., `bf16` or low-precision exponent/mantissa formats) with stricter constraints.
- **Index-like**: index values may appear as scalar operands in certain operations (e.g., offsets, sizes, or scalar comparisons).

Element type constraints are operation-specific:

- **Shape/type consistency**: most elementwise ops require all operands and results to have the same element type.
- **Numeric domain**: reductions, math ops, and division typically restrict element types to floating-point or a limited set of integer types.
- **Bitwise ops**: require integer element types.
- **Conversions**: `pto.tcvt` defines explicit element type changes and is controlled by `RoundMode` when converting between numeric domains.

In addition, memory layout and address space do not change the element type semantics; they only affect placement and access patterns.

#### Low-Precision Types

PTO IR currently recognizes the following low-precision element types:

- `f8E4M3FN` (corresponding C++ type name: `float8_e4m3_t`)
- `f8E5M2` (corresponding C++ type name: `float8_e5m2_t`)

- `!pto.hif8`
- `!pto.f4E1M2x2`
- `!pto.f4E2M1x2`

These types are recognized by the parser/printer, CAPI/Python bindings, and
basic storage-size plumbing. Their storage size is currently modeled as:

- `f8E4M3FN`: 1 byte per element
- `f8E5M2`: 1 byte per element
- `!pto.hif8`: 1 byte per element
- `!pto.f4E1M2x2`: 1 byte per packed pair of FP4 values
- `!pto.f4E2M1x2`: 1 byte per packed pair of FP4 values

For the packed FP4 PTO dialect types, `tile_buf` shape/valid-shape dimensions
describe the physical packed extent, i.e. the number of packed FP4 pairs
(equivalently the number of bytes in the packed dimension), not the logical
number of scalar FP4 elements.

Operation support is still opt-in. Defining the type in PTO IR does not by
itself imply that any particular operation accepts it.

### 2.2 `!pto.ptr<elementType[, memorySpace]>`

A typed pointer. `memorySpace` is optional and defaults to `gm`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element type pointed to |
| `memorySpace` | `gm` or `ub` | Pointer address space alias (`gm` -> global memory, `ub` -> vector/UB memory) |

**Syntax:** `!pto.ptr<f16>` or `!pto.ptr<f16, ub>`

Pointer conversions are modeled explicitly with [`pto.castptr`](#ptocastptr).
Between two `!pto.ptr` types, casts are only legal when both pointers stay in
the same PTO memory space.

---

### 2.3 `!pto.tensor_view<d0 x d1 x elementType[, layout]>`

A descriptor for a global memory tensor. Does not own data - represents a view with shape and stride information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Tensor shape `[d0, d1]` (each dim may be `?` for dynamic) |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element data type |
| `layout` | `#pto.layout<...>` | Optional resolved GM layout; omission allows construction-time resolution and defaults to ND when no stronger fact exists |

**Syntax:** `!pto.tensor_view<1024x512xf16>` or
`!pto.tensor_view<1x8x8x16x8xf32, #pto.layout<nz>>`

---

### 2.4 `!pto.partition_tensor_view<d0 x d1 x elementType[, layout]>`

A logical partition (slice) of a `tensor_view`. Holds shape and stride information for a tile-sized region but does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Partition shape `[d0, d1]` |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element data type |
| `layout` | `#pto.layout<...>` | Optional resolved GM layout inherited from the source view |

**Syntax:** `!pto.partition_tensor_view<16x16xf16>` or
`!pto.partition_tensor_view<1x4x8x16x8xf32, #pto.layout<nz>>`

---

### 2.5 `!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`

`pto.tile_buf` represents a local scratchpad memory tile buffer with explicit placement, shape, valid region, and layout/fractal metadata. Based on formats used in `PTOAS/test`, the canonical textual form is a key-value list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `loc` | keyword (`vec/mat/left/right/acc/bias`) | Local memory domain (`vec` maps to UB; use `vec` in textual IR) |
| `dtype` | `element-type(i1/i8/i16/i32/f16/f32/bf16/!pto.hif8/!pto.f4E1M2x2/!pto.f4E2M1x2...)` | Element data type |
| `rows` | `int64` | Physical row count |
| `cols` | `int64` | Physical column count |
| `v_row` | `int64` or `?` | Valid row count |
| `v_col` | `int64` or `?` | Valid column count |
| `blayout` | `BLayout` mnemonic | Base layout (`row_major` / `col_major`) |
| `slayout` | `SLayout` mnemonic | Secondary layout (`none_box` / `row_major` / `col_major`) |
| `fractal` | `int32` | Fractal size |
| `pad` | `PadValue` mnemonic or integer literal | Padding policy/value selector (tests commonly use `pad=0`) |

Here, `?` denotes a dynamic symbol resolved at runtime. Static and dynamic
valid dimensions are non-negative counts; a static `v_row=0` or `v_col=0`
represents an empty valid region. Physical `rows`/`cols` may still describe the
storage extent, for example an even physical row count with an odd or zero
`v_row` on a tail tile.

For `dtype=!pto.f4E1M2x2` and `dtype=!pto.f4E2M1x2`, the `rows`/`cols` and
`v_row`/`v_col` values are physical packed extents. In other words, the packed
dimension counts FP4 pairs stored per byte, not logical scalar FP4 elements.

**Syntax:**
```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

### 2.6 `!pto.multi_tile_buf<slotType, count=N>`

A **multi-buffer tile** representing N physically-distinct slots that share
one `tile_buf` shape. Only the underlying physical address differs across
slots; rank, valid shape, dtype, memory space, and config are identical.

| Parameter | Type | Description |
|-----------|------|-------------|
| `slotType` | `!pto.tile_buf<...>` | The per-slot tile_buf type |
| `count` | unsigned `[2, 16]` | Number of physical slots N |

**Constraints (enforced by the type verifier):**
- `2 <= count <= 16` (`kPtoMultiBufferMaxNum`)
- `slotType` follows all the same constraints as a single-slot `tile_buf`
- `multi_tile_buf` does not appear on function arguments or returns in the
  initial release; the design's multi-buffer ownership stays inside PTOAS

**Two compatible spellings:**

```mlir
// Compact (preferred): the slot tile_buf is described inline.
!pto.multi_tile_buf<vec, 16x16xf16, count=2>

// Verbose: spell out the slot tile_buf explicitly.
!pto.multi_tile_buf<!pto.tile_buf<vec, 16x16xf16>, count=2>
```

**Associated ops** (see Section 4 -- multi-buffer expression and slot
selection):

- `pto.alloc_multi_tile` -- allocate an N-slot multi-buffer tile
- `pto.multi_tile_get`   -- pick one slot of a multi_tile_buf, yielding a
  regular `tile_buf` that flows through every existing DMA / compute / view
  op unchanged

The slot count is part of `!pto.multi_tile_buf`. In level1/level2,
`PTOPlanMemory` plans all slots and records their physical addresses in the
internal `pto.multi_buffer_addrs` attribute. `PTOResolveBufferSelect` and
InsertSync consume the tile-native allocation and slot selection directly; no
memref allocation is introduced.

See `docs/designs/ptoas-multi-buffer-explicit-design.md` for the full
design.

---

### 2.7 `!pto.local_array<D1 x D2 x ... x Dk x T>`

A **C++ stack-local statically-shaped array**. Lowers to a plain `T a[D1][D2]...;`
declaration in the emitted C++ — the array's address is decided by the host C++
compiler, not by PTO memory planning.

| Parameter | Type | Description |
|-----------|------|-------------|
| shape (`D1..Dk`) | static positive `int64` per dim, `k ≥ 1` | Static dimensions; `?` is **not** allowed |
| `T` | scalar `int` / `float` family | Element type; aggregates / nested `local_array` are not allowed |

**Constraints (enforced by the type verifier):**
- rank ≥ 1
- every `Di > 0` (no dynamic shape, no zero-sized dims)
- `T` is a scalar integer or float

**Disjoint from tile-buf world.** Values of `!pto.local_array<...>` never
participate in `pto-plan-memory`; the planner only collects tile allocation
roots and does not see this type.

**Syntax:**
```mlir
!pto.local_array<16xi32>      // int32_t a[16];
!pto.local_array<4x8xf32>     // float   m[4][8];
```

**Associated ops** (see Section 4 — mirrors the `eventid_array` triad):
- `pto.declare_local_array` — declare
- `pto.local_array_get`     — `a[i0][i1]...` rvalue
- `pto.local_array_set`     — `a[i0][i1]... = v;`

The number of indices on `get` / `set` must equal the array's rank
(verifier-checked).

---

### 2.8 `!pto.struct<T0, T1, ..., Tn-1>`

A **C++ stack-local heterogeneous aggregate** — the heterogeneous dual of
`!pto.local_array`. Lowers to a plain `struct PtoStruct_X { ... };` definition
plus a stack variable. The struct's address is decided by the host C++
compiler, never by PTO memory planning or `pto.pointer_cast`. Fields are
positionally numbered `f0, f1, ...`, matching an LLVM literal struct.

| Parameter | Type | Constraints |
|-----------|------|-------------|
| `fieldTypes` (`T0..Tn-1`) | `i8`/`i16`/`i32`/`i64` (signed, unsigned or signless), `f16`/`bf16`/`f32`/`f64`, or a nested `!pto.struct` | At least one field; recursively scalar-storable |

**Constraints (enforced by the type verifier):**

- At least one field
- Each field is *scalar-storable*: an integer of width 8/16/32/64, one of
  `f16`/`bf16`/`f32`/`f64`, or a nested `!pto.struct`

A scalar field must map onto a C++ scalar the backend can name exactly. Integer
widths with no C++ spelling (`i1`, `i24`, ...) and the packed low-precision
vec/cube formats (`f8`/`f4` variants) are therefore **rejected**: emitting them
would silently widen or narrow the field in the generated C++.

**`!pto.local_array` is not a valid field type.** A field is reached with
`emitc.member`, whose result must be an `!emitc.lvalue`, and `!emitc.lvalue`
cannot wrap the `!emitc.array` that an array field lowers to. There is no way to
spell the access, so the type verifier rejects it rather than letting the
backend fail later. Use a nested `!pto.struct` of scalars, or keep the array as
a separate `!pto.declare_local_array` value alongside the struct.

**Disjoint from tile-buf world.** Vec/cube types (`tile_buf`, `tensor_view`,
partition view) and other handle types are **rejected** as fields. This keeps
the scalar struct world separate from the fractal/layout world, exactly like
`!pto.local_array`.

**Syntax:**
```mlir
!pto.struct<f16, i8>                          // { half f0; int8_t f1; };
!pto.struct<!pto.struct<f32, i8>, i16>
  // { PtoStruct_f32_i8 f0; int16_t f1; };
```

**Passing a struct to a function.** A `!pto.struct` value lowers to a **pointer**
to its storage, so it can be passed to a helper `func.func` and mutated there:

```mlir
func.func private @helper(%s : !pto.struct<f32, i8>, %v : f32) {
  pto.struct_set %s[0], %v : !pto.struct<f32, i8>, f32   // emits `p->f0 = v;`
  return
}
```

The callee reaches fields through `->`; the caller declares real storage, passes
its address, and still uses `.` on its own copy. Carrying the struct by value
would make the callee's writes land in a copy, and an `!emitc.lvalue` is
rejected outright as a function argument type by both `emitc.func` and the C++
emitter — the pointer is what makes helper functions expressible at all.

**A struct must not escape the scope that declares it.** Because the value is a
pointer to stack storage, passing it to a terminator would publish that address
outside the scope that owns it, so `pto.declare_struct` rejects it:

```mlir
// Rejected: hands the caller a pointer into a dead frame.
func.func @make() -> !pto.struct<f32, i8> {
  %s = pto.declare_struct -> !pto.struct<f32, i8>
  return %s : !pto.struct<f32, i8>
}

// Rejected: carries the address out of the region that owns the storage.
%r = scf.if %c -> (!pto.struct<f32, i8>) {
  %a = pto.declare_struct -> !pto.struct<f32, i8>
  scf.yield %a : !pto.struct<f32, i8>
} else { ... }
```

**A function may not return a `!pto.struct` either**, even one that just passes
an argument back:

```mlir
// Rejected: laundering. %r is no longer a pto.declare_struct result, so the
// per-op check above cannot see that `return %r` publishes the caller's local.
func.func private @passthrough(%s : !pto.struct<f32, i8>) -> !pto.struct<f32, i8> {
  return %s : !pto.struct<f32, i8>
}
```

Operation and region results can launder provenance in the same way. Therefore,
struct storage must originate from `pto.declare_struct` or a function argument.
Other operation results of this type are rejected; this includes aliases
produced by `arith.select`, `scf.if`, calls, casts, and loop-carried values.

This costs nothing in expressiveness: `pto.struct_set` mutates in place, so a
struct never needs to be returned or yielded. Declare it in the outer scope and
pass it **down** as a function argument — the callee writes through the pointer
and the caller sees the result.

**Generated type name.** The C++ name is derived from the MLIR type spelling of
the fields (`!pto.struct<f16, i8>` → `PtoStruct_f16_i8`), not from their C++
spellings. That mapping is injective, so two distinct `!pto.struct` types never
collide on one name — mangling from the C++ token instead would give `i32` and
`si32` (both `int32_t`) the same name and emit a duplicate `struct` definition.

**Associated ops** (see Section 4 — mirrors the `local_array` triad):
- `pto.declare_struct` — declare
- `pto.struct_get`     — `s.fA.fB...` field read by constant path
- `pto.struct_set`     — `s.fA.fB... = v;`

Field positions in `get` / `set` are a constant `path` of indices that may
descend through nested structs (like LLVM `extractvalue` / `insertvalue`).

The path must **end at a scalar**. A path that stops on a nested `!pto.struct`
is rejected: the member chain lowers to `emitc.member`, which yields an lvalue,
and returning a whole aggregate as an SSA value would copy it out of the parent
struct — so a field inside a nested struct is reached with a longer path
(`%s[0, 1]`) rather than by first getting the inner struct.

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

Global tensor layout inference for
[`tensor_view` (Section 2.3)](#23-ptotensor_viewd0-x-d1-x-elementtype-layout) /
[`partition_tensor_view` (Section 2.4)](#24-ptopartition_tensor_viewd0-x-d1-x-elementtype-layout).
Tile buffers additionally use **Tile Buf config** (see 3.4) to describe
physical/fractal layout.

| Value | Int | Mnemonic | Description |
|-------|-----|----------|-------------|
| `ND` | 0 | `nd` | Row-major (Normal-Dimension) |
| `DN` | 1 | `dn` | Column-major (Dimension-Normal) |
| `NZ` | 2 | `nz` | Fractal/blocked layout |

**Attribute syntax:** `#pto.layout<nd>`

Resolution follows one order: an explicit `layout` is authoritative; a
partition inherits its source layout; only an unannotated root view is
pattern-inferred. For a storage element of `sizeof(T)` bytes, canonical NZ uses
`C0 = 32 / sizeof(T)` and:

```text
shape  = [1, n1, m1, 16, C0]
stride = [n1*m1*16*C0, m1*16*C0, 16*C0, C0, 1]
```

An unannotated single-fractal `[1,1,1,16,C0]` view remains ND. Use an explicit
`#pto.layout<nz>` when its logical meaning is NZ.

A resolved layout remains part of the view contract across structured control
flow, control-flow block arguments, and direct internal function calls and
returns. All incoming values at a merge must agree: NZ merged with NZ remains
NZ, while NZ merged with ND is rejected. View-typed indirect calls are not
supported because their layout contract cannot be resolved statically.

---

## 4. Operations Reference

In addition to the `pto.*` operations documented below, PTOAS also accepts a limited set of commonly used third-party MLIR dialect operations as part of the input IR and lowering pipeline.

- **`func`**
  - `func.func`
  - `func.return`
  - `func.call`
- **`arith`**
  - constants and casts such as `arith.constant`, `arith.constant_index`, `arith.index_cast`, `arith.index_castui`, `arith.bitcast`, `arith.extf`, `arith.truncf`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, and `arith.fptoui`
  - integer and floating-point arithmetic such as `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.divui`, `arith.remsi`, `arith.remui`, `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.remf`, and `arith.negf`
  - bitwise and shift operations such as `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, and `arith.shrui`
  - comparisons and selection such as `arith.cmpi`, `arith.cmpf`, and `arith.select`
  - selected extended and min/max operations such as `arith.addui_extended`, `arith.mulsi_extended`, `arith.mului_extended`, `arith.ceildivsi`, `arith.ceildivui`, `arith.floordivsi`, `arith.maxsi`, `arith.minsi`, `arith.maxui`, `arith.minui`, `arith.maximumf`, `arith.minimumf`, `arith.maxnumf`, and `arith.minnumf`
- **`scf`**
  - `scf.for`
  - `scf.if`
  - `scf.yield`
  - PTOAS also handles several structured-control-flow forms by lowering them through `cf`, including `scf.execute_region`, `scf.while`, `scf.index_switch`, and `scf.condition`

NOTE: These third-party ops are supported only to the extent required by PTOAS front-end construction, analysis, and lowering. PTOAS does not imply full support for every operation in these dialects.

### 4.1 Pointer & View Operations

##### `pto.ptrtoint` - Convert Pointer to Byte Address

**Summary:** Converts a global pointer to an `i64` byte address.

**Semantics:**

```
result = reinterpret_cast<i64>(ptr)
```

If the source is produced by `pto.addptr`, the addptr offset is materialized as an explicit byte offset:

```
pto.ptrtoint(pto.addptr %p, %idx) == pto.ptrtoint(%p) + idx * sizeof(elementType)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `!pto.ptr<elementType>` | Source global pointer |

**Results:** `i64`

**Lowering Notes:**

- PTO view lowering accepts either PTO pointer form or the lowered rank-1 GM memref form.
- `pto.addptr` sources are folded into explicit byte-address arithmetic before EmitC lowering.
- EmitC lowering emits a C++ `reinterpret_cast<int64_t>`.

##### `pto.inttoptr` - Convert Byte Address to Pointer

**Summary:** Converts an `i64` byte address to a global pointer of the requested element type.

**Semantics:**

```
result = reinterpret_cast<result-element-type *>(addr)
```

This op is an escape hatch for explicit byte-address arithmetic and
cross-element-type pointer reinterpretation.

To limit provenance loss from integer-derived pointers, the result is
restricted to scalar memory access: every direct use must be the pointer operand
of `pto.load_scalar` or `pto.store_scalar`. The result cannot feed
`pto.addptr`, `pto.make_tensor_view`, returns, or other general pointer users.
Use the offset operand on `pto.load_scalar` / `pto.store_scalar` for element
offsets from an `inttoptr` pointer.

The result element type must be representable by EmitC scalar pointer lowering:
floating-point element types (`f16`, `bf16`, `f32`, `f64`), 8/16/32/64-bit
integer element types, and PTO low-precision floating-point element types are
accepted. Non-scalar element types such as `index` are rejected by the verifier.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `addr` | `i64` | Source byte address |

**Results:** `!pto.ptr<resultElementType>`

**Lowering Notes:**

- PTO view lowering rewrites the result to an equivalent rank-1 GM memref form.
- EmitC lowering emits a C++ `reinterpret_cast<__gm__ T*>`.

**Basic Example:**

```mlir
%p64_off = pto.addptr %p64, %idx : !pto.ptr<ui64> -> !pto.ptr<ui64>
%addr = pto.ptrtoint %p64_off : !pto.ptr<ui64> -> i64
%p32 = pto.inttoptr %addr : i64 -> !pto.ptr<ui32>
%val = pto.load_scalar %p32[%c0] : !pto.ptr<ui32> -> ui32
```

##### `pto.addptr` - Add Element Offset to Pointer

**Summary:** Computes a new pointer by adding an element offset to the base pointer.

**Semantics:**

```
result = ptr + offset   // offset is in elements, not bytes
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `!pto.ptr<elementType>` | Base pointer |
| `offset` | `index` | Element offset (not byte offset) |

**Results:** `!pto.ptr<elementType>` — the same pointer type as the input.

**Constraints & Verification:**

- result type must match the input pointer type
- The operation is pure (no side effects)

**Hardware Mapping:**

- No hardware pipeline (pointer arithmetic only)

**Basic Example:**

```mlir
%ptr_off = pto.addptr %base, %offset : !pto.ptr<f32> -> !pto.ptr<f32>
```

##### `pto.castptr` - Explicit Pointer Cast

**Summary:** Performs an explicit cast between integer addresses and `!pto.ptr`,
or between two `!pto.ptr` types.

**Semantics:**

```mlir
%p0 = pto.castptr %addr : i64 -> !pto.ptr<f32, ub>
%p1 = pto.castptr %p0 : !pto.ptr<f32, ub> -> !pto.ptr<i8, ub>
%addr2 = pto.castptr %p1 : !pto.ptr<i8, ub> -> i64
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `input` | `integer` or `!pto.ptr<...>` | Source value to cast |

**Results:** `integer` or `!pto.ptr<...>`

**Constraints & Verification:**

- Integer-to-integer casts are rejected; use normal integer cast ops instead
- Descriptor values such as `!pto.tensor_view<...>` and `!pto.partition_tensor_view<...>` are not legal direct inputs; extract a memref address first
- Pointer-to-pointer casts are only legal when source and destination stay in
  the same PTO memory space (`gm` or `ub`)
- The operation is pure (no side effects)

**Hardware Mapping:**

- No hardware pipeline (representation conversion only)

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
| `layout` | `LayoutAttr` (optional) | Authoritative ND/DN/NZ logical layout |

**Results:** `!pto.tensor_view<...>`

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `ptr` must be `!pto.ptr<...>` and its element type must match the result element type
  - `shape` and `strides` operand counts must match the tensor_view rank
  - Explicit ND/DN are preserved and are not replaced by shape/stride pattern
    inference
  - Explicit NZ must keep rank 5, `shape[0] == 1`, the complete
    `[16,C0]` inner fractal, and compatible C0-aligned outer strides
- `pto.inttoptr` results cannot feed `pto.make_tensor_view`. Tensor views must
  be constructed from a source pointer that already carries the desired element
  type.

**Notes:**

- An unannotated non-degenerate canonical NZ five-dimensional shape is inferred
  as NZ. Other unannotated views use the existing ND/DN fallback.

**Hardware Mapping:**

- No hardware pipeline (metadata/view construction only)

**Basic Example:**

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1] : !pto.tensor_view<?x?xf32>
```

For an explicit fp32 NZ view (`C0 = 8`):

```mlir
%tv = pto.make_tensor_view %ptr,
  shape = [%c1, %c8, %c8, %c16, %c8],
  strides = [%c8192, %c1024, %c128, %c8, %c1]
  {layout = #pto.layout<nz>}
  : !pto.tensor_view<1x8x8x16x8xf32>
```

---

##### `pto.get_tensor_view_dim` - Get Tensor View Dimension Size

**Summary:** Returns the size of a given dimension of a logical tensor view.

**Semantics:**

```mlir
dim = get_tensor_view_dim(tv_or_mr, dim_index)
```

This op is primarily defined on `!pto.tensor_view`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `tensor_view` | `!pto.tensor_view<...>` | Logical tensor view |
| `dim_index` | `index` | Dimension index (0-based) |

**Results:** `index` — the runtime size of the requested dimension.

**Notes:**

- Commonly used to drive `partition_view` sizes when the tensor_view shape is dynamic.

**Basic Example:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
%pv = pto.partition_view %tv,
       offsets = [%c0, %c0], sizes = [%h, %w]
       : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
```

---

##### `pto.get_tensor_view_stride` - Get Tensor View Dimension Stride

**Summary:** Returns the logical stride of a given dimension of a tensor view.

**Semantics:**

```mlir
stride = get_tensor_view_stride(tv_or_mr, dim_index)
```

This op is defined on `!pto.tensor_view`. During internal lowering, the same
query may temporarily appear on the memref form lowered from the tensor view.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `tensor_view` | `!pto.tensor_view<...>` or `memref<...>` | Logical tensor view or its lowered memref form |
| `dim_index` | `index` | Dimension index (0-based) |

**Results:** `index` — the logical stride of the requested dimension, measured
in elements rather than bytes.

**Notes:**

- This op is the IR counterpart of the DSL-side `TensorView.strides` metadata access.
- After lowering to memref, static strides may be folded into constants, while dynamic strides are derived from memref metadata.

**Basic Example:**

```mlir
%s0 = pto.get_tensor_view_stride %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%s1 = pto.get_tensor_view_stride %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

---

##### `pto.partition_view` - Partition Tensor View

**Summary:** Creates a logical window on a tensor_view or partition_tensor_view
using offsets and sizes, producing a `partition_tensor_view`.

**Semantics:**

```
result = source[offsets, sizes]
```

This op captures both static and dynamic shapes. It represents a logical slice without moving data.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `TensorViewType` or `PartitionTensorViewType` | Input root or partition view |
| `offsets` | `Variadic<Index>` | Dynamic offsets |
| `sizes` | `Variadic<Index>` | Dynamic sizes |

**Results:** `!pto.partition_tensor_view<...>`

**Constraints & Verification:**

- `offsets`/`sizes` counts must match the rank of `source`
- An NZ partition may shrink or dynamically offset/size only `d1` and `d2`
- NZ `d0`, `d3`, and `d4` must be statically proven complete; slicing inside
  `d3`/`d4` is rejected instead of falling back to ND

**Notes:**

- Pointer arithmetic is modeled as `BasePtr + Offset`, and the logical shape is determined by `sizes`.
- Nested partitions inherit the already-resolved source layout and preserve the
  root strides.

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
  - The result tile type may use standard or PTO low-precision element types.
  - If result `v_row`/`v_col` are dynamic (`?`), the corresponding operands must be present
  - If result `v_row`/`v_col` are static, the corresponding operands must be absent
- If `base_addr` is omitted, the address is assigned by the compiler

**Hardware Mapping:**

- No hardware pipeline (allocation/metadata op)

**Basic Example:**

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

##### `pto.alloc_multi_tile` - Allocate N-Slot Multi-Buffer Tile

**Summary:** Declares the lifetime of an N-slot multi-buffer tile. Each slot has the same `tile_buf` shape; only the underlying physical address differs. In the default pipeline (`--pto-level=level1|level2`), `PTOPlanMemory` plans the tile-native allocation directly and records all slot addresses in `pto.multi_buffer_addrs`. Under `--pto-level=level3`, the caller owns local memory and must provide an explicit base `addr`; `PTOResolveBufferSelect` derives each selected slot address from that base.

**Semantics:**

```
result = alloc_multi_tile(addr, valid_row, valid_col)   // operands are optional
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `addr` | `Optional<I64>` | Base physical byte address of the contiguous N-slot region. Required under `--pto-level=level3`, rejected otherwise. Slot `k` lives at `addr + k * slotBytes` where `slotBytes = product(shape) * element_size`; the caller must reserve `N * slotBytes` bytes at `addr`. |
| `valid_row` | `Optional<Index>` | Dynamic valid row count (required when slot `v_row` is `?`) |
| `valid_col` | `Optional<Index>` | Dynamic valid column count (required when slot `v_col` is `?`) |

**Results:** `!pto.multi_tile_buf<...>`

**Constraints & Verification:**

- The result type must have `count` in `[2, 16]`.
- The slot tile type (rank, valid shape, dtype, memory space, config) is verified the same way as `pto.alloc_tile` for a single slot.
- `addr` is gated by build level: required under `--pto-level=level3`, rejected under level1/level2 (the same rule `pto.alloc_tile` follows). With an explicit `addr` the slot shape and element size must be static so `slotBytes` is known.
- **Storage-compact slot layout required.** The N physical slots are placed at `product(shape) * element_size` byte intervals (both the level3 lowering and PlanMemory size them this way). A `row_plus_one` compact mode inflates the major stride by one element per row, so the slot's physical strided footprint exceeds `product(shape)` and adjacent slots would silently overlap. `pto.alloc_multi_tile` therefore rejects `compact = row_plus_one` slot layouts. Non-`row_plus_one` compact layouts and boxed fractal `slayout`s pack densely (footprint equals `product(shape)`) and are supported. (This limit can be lifted once the slot stride is derived from the true strided footprint instead of `product(shape)`.)

**Hardware Mapping:**

- No hardware pipeline (allocation/metadata op). In the default pipeline the N-way physical fan-out is realized by PlanMemory; under level3 it is realized directly from the base `addr` during view lowering.

**Basic Example:**

```mlir
// level1/level2: addresses chosen by PlanMemory.
%mb = pto.alloc_multi_tile : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
%mb2 = pto.alloc_multi_tile : !pto.multi_tile_buf<!pto.tile_buf<vec, 16x16xf16>, count=3>

// level3: caller-owned memory; slot 0 at %base, slot 1 at %base + 512 bytes.
%base = arith.constant 0 : i64
%mb3 = pto.alloc_multi_tile addr = %base : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
```

##### `pto.multi_tile_get` - Select One Slot Of A Multi-Buffer Tile

**Summary:** Returns a single-slot view of a `multi_tile_buf`. The frontend is the source of truth for which slot a given use refers to; the slot index `%k` is an `index` value (constant or any SSA expression) in `[0, count)`. PTOAS does NOT synthesize `iv mod N` for users -- the user expression IS the slot selector. Downstream sync and event-id allocation analyze the slot expressions and emit static `set_flag` / `wait_flag` for constant slots or `set_flag_dyn` / `wait_flag_dyn` for runtime slots.

**Semantics:**

```
result = multi_tile_get(source, slot)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `MultiTileBufType` | The N-slot multi-buffer tile |
| `slot` | `Index` | Slot index in `[0, count)` |

**Results:** `!pto.tile_buf<...>` (must equal `source.slotType`)

**Constraints & Verification:**

- Result `tile_buf` must equal `source.slotType` (rank, valid shape, dtype, memory space, config all identical).
- If `slot` is a constant, the verifier checks `0 <= slot < count`.
- Pure view op -- no data movement, no extra address arithmetic.
- `multi_tile_buf` is not allowed on function arguments or results in the initial release.

**Hardware Mapping:**

- No hardware pipeline (metadata-only view).

**Basic Example:**

```mlir
%mb = pto.alloc_multi_tile : !pto.multi_tile_buf<vec, 16x16xf16, count=2>

// constant-slot selection
%c0 = arith.constant 0 : index
%s0 = pto.multi_tile_get %mb[%c0]
    : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
   -> !pto.tile_buf<vec, 16x16xf16>

// dynamic-slot selection (e.g. prefetch with %k from the loop body)
%s_k = pto.multi_tile_get %mb[%k]
    : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
   -> !pto.tile_buf<vec, 16x16xf16>
```

See `docs/designs/ptoas-multi-buffer-explicit-design.md` for the full design (sync/event-id derivation, downstream pass interplay, and end-to-end usage examples).

##### `pto.subview` - Tile SubView

**Summary:** Create a logical subview from a parent tile. The subview window is expressed by `offsets + sizes`, and the result tile type shape equals `sizes`.

**Semantics:**

```
result = source[offsets] with static sizes
result.shape = sizes
result.valid = clip(explicit_valid_or_sizes, sizes)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `pto.tile_buf` | Parent tile buffer |
| `offsets` | `Variadic<Index>` | Runtime dynamic offsets [i, j] |
| `sizes` | `I64ArrayAttr` | Static shape [rows, cols] |
| `valid_row` | `Optional<Index>` | Optional explicit valid row |
| `valid_col` | `Optional<Index>` | Optional explicit valid col |

**Results:** `pto.tile_buf`

**Constraints & Verification:**

- The verifier derives boxed-vs-non-boxed behavior from `source`'s tile config (`blayout`, `slayout`, `fractal`) and element type.
- For non-boxed layouts (`slayout=none_box`), no additional subview-specific structural checks are enforced.
- For boxed layouts (`slayout != none_box`):
  - The tile layout must be one of the subview layouts supported by the current implementation; otherwise verification fails.
  - `sizes` must be present, must have length 2, and both subview sizes must be positive.
  - The subview sizes must be multiples of the inferred inner boxed shape.
  - `offsets` must have length 2.
  - If an offset is compile-time constant, it must be non-negative and must be a multiple of the inferred inner boxed shape in that dimension.
  - The source tile shape must be statically known.
  - For boxed row-major tiles, the subview must keep the full source column extent, and the column offset must be the constant `0`.
  - For boxed col-major tiles, the subview must keep the full source row extent, and the row offset must be the constant `0`.
- `valid_row` and `valid_col` must be both present or both absent.
- If `valid_row/valid_col` are omitted, the result type is authoritative for the
  valid extent: any static `valid_shape` in `[0, sizes]` is accepted (this covers
  the full-size default and a `v_row=0`/`v_col=0` no-op-replay empty marker), and
  lowering takes the valid extent from the result type rather than `sizes`. A
  dynamic result valid (`?`) still requires an explicit `valid_row`/`valid_col`
  operand to supply the runtime extent.
- If `valid_row/valid_col` are provided:
  - constant values must be non-negative and `<= sizes` in each dimension
  - non-constant values are represented as dynamic valid dims in the result type
- The inferred result type uses:
  - `shape = sizes` (logical subview size)
  - the same element type and address space as `source`
  - the same tile config as `source`
  - `valid_shape` defaults to `sizes`
  - if explicit `valid_row/valid_col` are provided, `valid_shape` is clipped by `sizes`
- Lowering keeps parent physical stride/base semantics for non-compact access,
  so EmitC behavior remains unchanged from the previous implementation.
- If an explicit valid dimension is zero, the subview still has the requested
  physical `sizes`, but its valid region is empty in that dimension.

**Hardware Mapping:**

- No hardware pipeline (view construction only)

**Basic Example:**

```mlir
%sub = pto.subview %src[%i, %j] sizes [32, 32] :
  !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=64, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%sub2 = pto.subview %src[%i, %j] sizes [32, 32] valid [%vr, %vc] :
  !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=64, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
  -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

##### `pto.set_validshape` - Update Dynamic Tile Valid Row/Col In Place

**Summary:** Updates runtime valid row/col metadata directly on an existing dynamic `pto.tile_buf`.

**Semantics:**

```
set_validshape(source, valid_row, valid_col)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `pto.tile_buf` | Dynamic tile buffer whose runtime valid shape will be updated |
| `valid_row` | `Index` | Runtime valid row count |
| `valid_col` | `Index` | Runtime valid column count |

**Results:** None

**Constraints & Verification:**

- `source` must be a rank-2 `pto.tile_buf`
- `source` must have dynamic valid shape:
  - `v_row = ?`
  - `v_col = ?`
- User-authored PTO IR must use the `pto.tile_buf` form; any memref form seen
  later in the pipeline is compiler-internal lowering state only
- If `valid_row` / `valid_col` are compile-time constants, they must be non-negative and not exceed the tile's static shape bounds

**Hardware Mapping:**

- No hardware pipeline (metadata update op only)
- Lowers in `PTOToEmitC` to updates of the tile's runtime valid-shape fields

**Basic Example:**

```mlir
%src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
pto.set_validshape %src, %vr, %vc
  : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

### 4.2 Buffer-ID Token Operations (A5)

The following operations implement a **buffer-id based ordering model** for the A5 architecture: acquire and release a buffer-id token by high-level sync op type (the op type is mapped to a concrete pipe internally), so that operations guarded by the same buffer-id execute in program order across mapped pipes. They lower to the CCEC builtins `get_buf` and `rls_buf`.

##### `pto.get_buf` - Acquire Buffer-ID Token (A5)

**Summary:** Acquires a buffer-id token for a sync op type (`pipe_event_type` / `sync_op_type`). Used in a buffer-id based ordering model: operations on the mapped pipe that share the same buffer-id are enforced to execute in program order relative to other mapped pipes using the same buffer-id.

**Semantics:**

```
get_buf(op_type, buf_id [, mode])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `PipeEventTypeAttr` / `SyncOpTypeAttr` | High-level sync op type (mapped to concrete pipe) |
| `buf_id` | `I32Attr` | Buffer ID (token identifier) |
| `mode` | `I32Attr` (default: 0) | Optional mode (attribute) |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Intended for **A5**; lowered to the CCEC builtin intrinsic `get_buf`

**Basic Example:**

```mlir
pto.get_buf [#pto.pipe_event_type<TVEC>, 0]
pto.get_buf [#pto.pipe_event_type<TMATMUL>, 1] { mode = 0 }
```

---

##### `pto.rls_buf` - Release Buffer-ID Token (A5)

**Summary:** Releases a previously acquired buffer-id token for a sync op type. Used in conjunction with `pto.get_buf`: after operations that were ordered under the same buffer-id complete, `rls_buf` releases the token for that mapped pipe and buffer-id.

**Semantics:**

```
rls_buf(op_type, buf_id [, mode])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `PipeEventTypeAttr` / `SyncOpTypeAttr` | High-level sync op type (mapped to concrete pipe) |
| `buf_id` | `I32Attr` | Buffer ID (must match a prior `pto.get_buf`) |
| `mode` | `I32Attr` (default: 0) | Optional mode (attribute) |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Intended for **A5**; lowered to the CCEC builtin intrinsic `rls_buf`

**Basic Example:**

```mlir
pto.get_buf [#pto.pipe_event_type<TVEC>, 0]
// ... operations under buffer-id 0 ...
pto.rls_buf [#pto.pipe_event_type<TVEC>, 0]
pto.rls_buf [#pto.pipe_event_type<TMATMUL>, 1] { mode = 0 }
```

---

### 4.3 DMA Data Movement Operations

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`.
  - The destination tile must use `loc=vec` or `loc=mat`.
  - The destination tile element type and source partition element type must have the same bitwidth.
  - Runtime: all source partition extents must be positive; the destination valid region must be non-negative.
- **Implementation checks (A5)**
  - The source partition and destination tile element types must be one of `i8/i16/i32/i64/f16/bf16/f32/f8E4M3*/f8E5M2*/!pto.hif8/!pto.f4E1M2x2/!pto.f4E2M1x2`.
  - The destination tile element size must be `1`, `2`, `4`, or `8` bytes, and must match the source partition element size.
  - For `i64`, the destination tile `pad` must be `null` or `zero`.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`, GM -> UB)

**Basic Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.tprefetch` - Prefetch Partition View into Tile

**Summary:** Prefetches a GM-backed partition view into a temporary local tile buffer. This maps to PTO-ISA `TPREFETCH(dst, src)` and, unlike most PTO intrinsics, does not add implicit wait-event synchronization in the C++ wrapper.

**Semantics:**

```
TPREFETCH(dst, src)
```

The detailed caching / hint behavior is target-defined by PTO-ISA. In PTOAS the
op is modeled as writing the prefetched data into `dst`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.partition_tensor_view` | Source global view |
| `dst` | `pto.tile_buf` | Destination local tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `src` must be a partition tensor view.
- `dst` must be a tile buffer.
- `dst` must use `loc=vec` or `loc=mat`.
- Static source extents must be positive when known; static destination valid extents must be non-negative when known.
- `src` and `dst` element types must have the same element size in bytes.
- Low-precision element types (`f8E4M3*`, `f8E5M2*`, `!pto.hif8`, `!pto.f4E1M2x2`, `!pto.f4E2M1x2`) are only accepted on A5.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`, GM -> local tile)

**Basic Example:**

```mlir
pto.tprefetch ins(%pv : !pto.partition_tensor_view<16x16xf16>)
              outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

##### `pto.tprefetch_async` - Asynchronous GM Prefetch into L2

**Summary:** Starts an SDMA-backed asynchronous prefetch from GM into cache and returns the async event. The associated async session is obtained separately from an explicit prefetch context SSA value.

**Semantics:**

```
%event = pto.tprefetch_async(%src, %ctx)
%session = pto.get_prefetch_async_session %ctx
```

Lowering maps `%ctx` to `pto::PrefetchAsyncContext`, emits
`TPREFETCH_ASYNC(src, ctx)`, and projects `ctx.session` through
`pto.get_prefetch_async_session`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Source GM region to prefetch |
| `ctx` | `!pto.prefetch_async_context` | Explicit PTO prefetch async context |

**Results:**

| Name | Type | Description |
|------|------|-------------|
| `event` | `!pto.async_event` | Async completion event returned by PTO-ISA |

**Constraints & Verification:**

- `src` must be a flat contiguous logical-1D GM view-like value.
- `ctx` must be a valid `!pto.prefetch_async_context`.
- This op intentionally mirrors the PTO-ISA API input checks in `verify()`, so shape/address-space mismatches fail at PTO IR verification time.

**Basic Example:**

```mlir
%ctx = pto.make_prefetch_async_context(%workspace : !pto.ptr<i8>)
    -> !pto.prefetch_async_context
%event = pto.tprefetch_async(
    %src, %ctx
    : !pto.partition_tensor_view<128xf32>,
      !pto.prefetch_async_context)
    -> !pto.async_event
%session = pto.get_prefetch_async_session %ctx
    : !pto.prefetch_async_context -> !pto.async_session
%done = pto.comm.wait_async_event(%event, %session : !pto.async_event, !pto.async_session) -> i1
```

---

##### `pto.tstore` - Store Tile to Partition View

**Summary:** Stores a 2-D tile buffer back to a 2-D partition view. Supports phase/atomic/relu/pre-quant controls and an optional scaling tile. The scaling-tile form lowers to `TSTORE_FP`.

**Semantics:**

```
For each element (i, j) in the tile valid region:
    dst[i, j] = src[i, j]
```

**Arguments:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `src` | `pto.tile_buf` | `NA` |Source tile buffer |
| `dst` | `PartitionTensorViewType` | `NA` | Destination partition view |
| `fp` | `pto.tile_buf` (optional) | `NA` | Scaling tile (`loc=scaling`) for accumulator conversion |
| `preQuantScalar` | `i64` (optional) | `NA` |Optional scalar used by pre-quantized `acc` store forms |
| `stPhase` | `#pto<st_phase ...>` | `unspecified` | Store phase selector (`unspecified/partial/final`) |
| `atomicType` | `#pto<atomic_type ...>` | `atomic_none` | Atomic mode (`atomic_none/atomic_add`) |
| `reluPreMode` | `#pto<relu_pre_mode ...>` | `no_relu` | ReLU pre-processing mode (`no_relu/normal_relu`) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- Common checks:
  - `src` must be `!pto.tile_buf`, `dst` must be `!pto.partition_tensor_view`.
  - Static `dst` shape dims must be positive, and static `src` valid-shape dims
    must be non-negative.
  - `fp` and `preQuantScalar` are mutually exclusive.
  - If `fp` or `preQuantScalar` is present, `src` must be `loc=acc`.
  - `fp` must use `loc=scaling`; the fp form uses the default `stPhase`.
  - If `reluPreMode != no_relu`, `src` must be `loc=acc`.
- A2/A3 checks:
  - `src.loc` must be one of `vec/mat/acc`.
  - For `loc=vec` or `loc=mat`:
    - `preQuantScalar` is not allowed.
    - `src` element type must be one of `i8/i16/i32/i64/f16/bf16/f32`.
    - `src`/`dst` element bitwidth must match.
  - For `loc=acc`:
    - `src` element type must be `i32` or `f32`.
    - Without `preQuantScalar`: `dst` element type must be `i32/f32/f16/bf16`.
    - With `preQuantScalar`:
      - `src=i32` -> `dst=i8(ui8)/f16`
      - `src=f32` -> `dst=i8(ui8)`
    - Static/runtime column bound checks on `src`: `1 <= cols <= 4095` and `1 <= valid_shape[1] <= 4095` (when static).
- A5 checks:
  - `src.loc` must be `vec` or `acc` (A5 does not support `mat` here).
  - For `loc=vec`:
    - `preQuantScalar` is not allowed.
    - `src` element type must be one of `i8/i16/i32/i64/f16/bf16/f32/f8E4M3*/f8E5M2*/!pto.hif8/!pto.f4E1M2x2/!pto.f4E2M1x2`.
    - `src`/`dst` element bitwidth must match.
  - For `loc=acc`:
    - `src` element type must be `i32` or `f32`.
    - Without `preQuantScalar`: `dst` element type must be `i32/f32/f16/bf16`.
    - With `preQuantScalar`:
      - `src=i32` -> `dst=i8(ui8)/f16/bf16`
      - `src=f32` -> `dst=i8(ui8)/f16/bf16/f32/!pto.hif8/f8E4M3*`

**Type Note (PTO IR):**

- PTO IR is not limited to signless integers. In particular, several current
  surfaces already use distinct signed/unsigned integer element types such as
  `ui8`, `si8`, `ui16`, and `ui32`, and verifier rules may depend on that
  signedness.
- For this `preQuantScalar` contract, destination element types written below as
  `i8(ui8)` mean:
  - signless `i8` remains accepted on legacy surfaces that only constrain width
    and numeric family
  - explicit `ui8` is also a first-class PTO IR dtype on surfaces whose
    semantics depend on unsignedness
- When a newer contract needs signedness to be user-visible, the manual will
  call that out explicitly instead of relying on signless `i8` as a synonym for
  `ui8`.

**Hardware Mapping:**

- `src=loc=acc`: uses **PIPE_FIX** path.
- `src=loc=vec` or `src=loc=mat`: uses **PIPE_MTE3** path.

**Basic Example:**

```mlir
// 1) TSTORE(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)

// 2) TSTORE<STPhase::Final>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>}

// 3) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {atomicType = #pto<atomic_type atomic_add>}

// 4) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>}

// 5) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 6) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 7) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src, preQuantScalar)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, %pq : i64)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 8) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src, preQuantScalar)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, %pq : i64)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}
```

---

##### `pto.load_scalar` - Load Single Scalar Element

**Summary:** Loads a single scalar element from a pointer at the given offset.

**Semantics:**

```
value = ptr[offset]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `!pto.ptr<...>` | Source pointer |
| `offset` | `index` | Element offset |

**Results:** `AnyType` — the element type of the pointed-to memory.

**Constraints & Verification:**

- The operation has a custom verifier
- `ptr` element type must match the result type

**Hardware Mapping:**

- Scalar load from global

**Basic Example:**

```mlir
%val = pto.load_scalar %ptr[%offset] : !pto.ptr<f32> -> f32
```

---

##### `pto.store_scalar` - Store Single Scalar Element

**Summary:** Stores a single scalar element to a pointer at the given offset.

**Semantics:**

```
ptr[offset] = value
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `AnyType` | Value to store |
| `ptr` | `!pto.ptr<...>` | Destination pointer |
| `offset` | `index` | Element offset |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier
- `value` type must match the element type of `ptr`

**Hardware Mapping:**

- Scalar store to global memory space.

**Basic Example:**

```mlir
pto.store_scalar %val, %ptr[%offset] : !pto.ptr<f32>, f32
```

---

##### `pto.tmov` - Tile Move Between Local Domains

**Summary:** Moves data between local memory domains. On A5, a three-tile
`vec` form also converts an MX exponent tile from ND/DN order into the ZZ box
order consumed by Cube operations.

**Semantics:**

```
plain move:
    dst = Move(src)

scaling-assisted move:
    dst = MoveWithScaling(src, fp)

MX exponent X-to-ZZ move:
    dst = PackExponentToZZ(src, grpAxis)
```

The optional third tile keeps the public operand name `fp`. Its address space
selects the form: `loc=scaling` retains the existing FP move semantics, while a
non-scaling third tile is the temporary workspace for X-to-ZZ. Omitting
`grpAxis` on X-to-ZZ means `axis1`.

For `grpAxis=axis0`, an exponent tile with valid shape `[2P, 16C]` is packed as:

```text
dst[c, p, q, delta] = src[2*p + delta, 16*c + q]
```

For `grpAxis=axis1`, source rows are padded with zeros to a multiple of 16 and
each adjacent exponent-column pair is packed together:

```text
R = alignTo(dst.valid_rows, 16)
dst[row_block, col_pair, row_in_block, pair_lane] =
    src[16*row_block + row_in_block, 2*col_pair + pair_lane]
```

Reads outside the valid source rows in this pseudocode produce zero padding.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |
| `fp` | `pto.tile_buf` | Optional third tile. `loc=scaling` means the existing FP parameter; a non-scaling tile is X-to-ZZ workspace. The public API name remains `fp`. |
| `preQuantScalar` | `i64` | Optional scalar pre-quant parameter used by scalar-quant TMOV forms |
| `accToVecMode` | `pto.acc_to_vec_mode` | Optional acc-to-vec mode template parameter |
| `reluPreMode` | `pto.relu_pre_mode` | Optional relu mode template parameter, default is `NoRelu` |
| `grpAxis` | `#pto.mx_group_axis` | Optional X-to-ZZ grouping axis: `axis0` (DN) or `axis1` (ND). It is invalid on plain and scaling-FP forms. |

**Results:** None. Writes into `dst` via DPS pattern. ND X-to-ZZ also writes
the source padding and its temporary workspace; DN X-to-ZZ does not access the
temporary tile.

**Supported PTO IR Forms:**

- `pto.tmov ins(%src) outs(%dst)`
  - maps to `TMOV(dst, src)`
- `pto.tmov ins(%src) outs(%dst) {reluPreMode = ...}`
  - maps to `TMOV<..., ReluPreMode>(dst, src)`
- `pto.tmov ins(%src) outs(%dst) {accToVecMode = ..., reluPreMode = ...}`
  - maps to `TMOV<..., AccToVecMode, ReluPreMode>(dst, src)`
- `pto.tmov ins(%src, %fp) outs(%dst) {reluPreMode = ...}`
  - maps to `TMOV_FP<..., FpTileData, ReluPreMode>(dst, src, fp)`
- `pto.tmov ins(%src, %fp) outs(%dst) {accToVecMode = ..., reluPreMode = ...}`
  - maps to `TMOV<..., FpTileData, AccToVecMode, ReluPreMode>(dst, src, fp)`
- `pto.tmov` with `preQuantScalar`
  - maps to `TMOV<..., ReluPreMode>(dst, src, preQuantScalar)`
  - or `TMOV<..., AccToVecMode, ReluPreMode>(dst, src, preQuantScalar)`
- `pto.tmov ins(%src, %tmp) outs(%dst)` with `%tmp.loc != scaling`
  - `grpAxis` omitted or `axis1`: maps to `TMOV(dst, src, tmp)`
  - `grpAxis=axis0`: maps to `TMOV<0>(dst, src, tmp)`

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Static tile shapes must match: `src.rows == dst.rows` and `src.cols == dst.cols`.
  - Supported location pairs (compile-time checked):
    - `loc=mat -> loc=left/right/bias/scaling`
    - `loc=vec -> loc=vec`
    - `loc=acc -> loc=mat`
    - `loc=acc -> loc=vec`
  - `accToVecMode` is only valid for `loc=acc -> loc=vec`.
  - When `fp` or `preQuantScalar` is present, only single-mode acc-to-vec forms are legal.
  - `reluPreMode` / `fp` / `preQuantScalar` forms require `loc=acc` source.
  - For `loc=acc -> loc=mat`, additional fractal and dtype constraints apply (for example `acc` uses accumulator-style layout, `mat` uses `fractal=512`, and only selected dtype conversions are legal).
- **Implementation checks (A5)**
  - For `loc=mat -> *`, static tile shapes must match; for some `loc=vec` moves, the effective copy size is the min of the source and destination valid regions.
  - Supported location pairs include (target-dependent):
    - `loc=mat -> loc=left/right/bias/scaling/scale`
    - `loc=vec -> loc=vec` and `loc=vec -> loc=mat`
    - `loc=acc -> loc=vec` and `loc=acc -> loc=mat`
  - `accToVecMode` is only valid for `loc=acc -> loc=vec`.
  - When `fp` or `preQuantScalar` is present, only single-mode acc-to-vec forms are legal.
  - `reluPreMode` / `fp` / `preQuantScalar` forms require `loc=acc` source.
  - `loc=mat -> loc=left/right` has additional target-specific fractal and dtype constraints.
  - `loc=acc -> loc=vec/mat` has additional target-specific fractal, dtype, and alignment constraints.
  - `loc=mat -> loc=scale` has additional target-specific fractal and dtype constraints.
  - A5 accepts PTO low-precision element types (`f8E4M3*`, `f8E5M2*`, `!pto.hif8`, `!pto.f4E2M1x2`) for `src`/`dst` in supported location pairs.

**A5 X-to-ZZ constraints:**

- `src`, `dst`, and the non-scaling `fp`/workspace operand must be rank-2
  `loc=vec` tiles with the same element type. Legal element types are `ui8`,
  `!pto.hif8`, and `!pto.f8E8M0`; signed `i8` is not legal.
- `src` uses `blayout=row_major, slayout=none_box`; `dst` uses
  `blayout=row_major, slayout=row_major` to denote the ZZ box.
- The source and destination valid element counts must match. Their valid and
  physical shapes, plus the workspace physical shape, must be static.
- `preQuantScalar`, `accToVecMode`, and non-default `reluPreMode` are mutually
  exclusive with X-to-ZZ. `grpAxis` is legal only on X-to-ZZ.
- `src` and `dst` must refer to non-overlapping byte ranges, including when
  either operand is a view of another allocation.
- ND (`axis1`): `dst.valid_cols` is even. Source valid elements form a compact
  prefix (`src.valid_rows == 1` or `src.cols == src.valid_cols`). Both source
  and destination have at least
  `alignTo(dst.valid_rows, 16) * dst.valid_cols` bytes of capacity. Workspace
  capacity is at least
  `64 + ceil(dst.valid_rows / 16) * dst.valid_cols` bytes.
- DN (`axis0`): `src.valid_rows` is even and at least 2;
  `src.valid_cols` is a multiple of 16; `src.cols == src.valid_cols`; and both
  source and destination have at least
  `src.valid_rows * src.valid_cols` bytes of capacity. The workspace requires
  a static physical shape but has no minimum capacity beyond a valid tile.

ND X-to-ZZ may zero source row padding in place and therefore has
`src: Read+Write`, `tmp: Read+Write`, `dst: Write` effects. DN has
`src: Read`, `tmp: none`, `dst: Write` effects. Existing scaling-FP moves remain
`src/fp: Read`, `dst: Write`.

**Hardware Mapping:**

- `vec -> vec` executes on **PIPE_V**
- `mat -> left/right/bias/scaling` executes on **PIPE_MTE1**
- `acc -> mat/vec` executes on **PIPE_FIX**
- X-to-ZZ executes on **PIPE_V**

**Basic Example:**

```mlir
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

// DN exponent [2, 16] -> ZZ. The optional operand is still spelled `fp`
// by generated Python/C++ builders, although this non-scaling tile is tmp.
pto.tmov ins(%exp : !pto.tile_buf<vec, 2x16xui8>,
             %tmp : !pto.tile_buf<vec, 1x1xui8>)
         outs(%exp_zz : !pto.tile_buf<vec, 2x16xui8, slayout=row_major>)
         {grpAxis = #pto<mx_group_axis axis0>}
```

---

##### `pto.ttrans` - Transpose Tile

**Summary:** Transposes a tile buffer. The current DPS form always carries a required `tmp` operand, but PTO IR does not encode a same-shape tmp contract.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[j, i]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary workspace operand required by the current DPS form |
| `dst` | `pto.tile_buf` | Destination tile |
| `fp` | `pto.tile_buf` | Optional scaling tile (`loc=scaling`) for accumulator conversion |
| `preQuantScalar` | `i64` | Optional scalar pre-quant parameter |
| `accToVecMode` | `pto.acc_to_vec_mode` | Optional A5 acc-to-vec mode |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src`, `tmp`, and `dst` must have the same element type.
  - The source tile must use `blayout=row_major`.
  - Element size must be `1`, `2`, or `4` bytes.
  - Supported element types are restricted per element width:
    - 4 bytes: `i32`, `f32`
    - 2 bytes: `i16`, `f16`, `bf16`
    - 1 byte: `i8`
  - The transpose domain is taken from the source tile valid region.
- **Implementation checks (A5)**
  - `src`, `tmp`, and `dst` must have the same element type.
  - 32-byte alignment constraints are enforced on the major dimension of both input and output (for `blayout=row_major`, check `cols * sizeof(T) % 32 == 0`; for `blayout=col_major`, check `rows * sizeof(T) % 32 == 0`).
  - Supported element types are restricted per element width:
    - 4 bytes: `i32`, `f32`
    - 2 bytes: `i16`, `f16`, `bf16`
    - 1 byte: `i8`
  - The implementation operates over the static tile shape (`rows/cols`) and does not consult the valid region.
- **Temporary tile**:
  - The current PTO IR surface requires `tmp`.
  - PTO IR does not require `tmp` to match the shape or valid shape of `src` / `dst`.
  - The exact workspace size remains target-defined and may depend on transpose mode and stride alignment.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=16,
               v_row=32, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
               v_row=1, v_col=8, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
               v_row=16, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

### 4.4 Matrix Compute Operations

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

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m/k/n` (taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in `[0, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, supported lhs/rhs element type pairs are:
      - `(f16, f16)`, `(bf16, bf16)`, `(f32, f32)`
      - any pair from the fp8 e4m3/e5m2 families, which lower to `float8_e4m3_t` or `float8_e5m2_t`
      - `(!pto.hif8, !pto.hif8)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - Runtime: `m/k/n` (taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in `[0, 4095]`.

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

- All constraints from `pto.tmatmul` apply to the `(Destination accumulator, Left matrix, Right matrix)` triple.
- **A2/A3 and A5 notes:**
  - `lhs valid row`, `lhs valid column`, and `rhs valid column` for `m/k/n`.
  - `acc_in Matrix` is not validated by explicit assertions in the current implementations (target-defined behavior).

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

- All constraints from `pto.tmatmul` apply to the `(Destination accumulator, Left matrix, Right matrix)` triple.
- **A2/A3 bias constraints:**
  - `bias` element type must match `dst` element type.
  - `bias` must use `loc=bias` and `rows=1`.
- **A5 bias constraints:**
  - `bias` element type must match `dst` element type.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.

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

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The destination element type must be `f32`.
  - Supported lhs/rhs element type pairs are:
    - any pair from the fp8 e4m3/e5m2 families, which lower to `float8_e4m3_t` or `float8_e5m2_t`
    - any pair from `!pto.f4E1M2x2` and `!pto.f4E2M1x2`
  - `lhs`/`rhs`/`dst` follow the same A5 tile location and layout constraints as `pto.tmatmul`.
  - `lhs_scale` and `rhs_scale` must use `loc=scaling` and compact MX block-scale shapes:
    - `lhs_scale` shape and valid shape are `[M, ceil(K/32)]`
    - `rhs_scale` shape and valid shape are `[ceil(K/32), N]`
  - `lhs_scale` must use `blayout=row_major`, `slayout=row_major`, `fractal=32`.
  - `rhs_scale` must use `blayout=col_major`, `slayout=col_major`, `fractal=32`.
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.
  - Runtime: `m/k/n` must be in `[1, 4095]`.

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

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The `lhs`/`rhs`/`dst` and scaling tile constraints are the same as `pto.tmatmul.mx`.
  - `acc_in` must be an accumulator tile.
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.
  - Runtime: `m/k/n` must be in `[1, 4095]`.

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

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The `lhs`/`rhs`/`dst` and scaling tile constraints are the same as `pto.tmatmul.mx`.
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.
  - Runtime: `m/k/n` must be in `[1, 4095]`.
- **Bias form**:
  - `bias` must use element type `f32`, `loc=bias`, `rows=1`, and `blayout=row_major`.

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

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.

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


- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.

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

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.
  - Bias checks:
    - The bias tile element type must exactly match the result tile element type.
    - The bias tile must be configured as a single row.
    - The bias tile must use `loc=bias`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[0, 4095]`.
  - Bias checks:
    - The bias tile element type must exactly match the result tile element type.
    - The bias tile must be configured as a single row.
    - The bias tile must use `loc=bias`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx` - Mixed-Precision Matrix-Vector Multiply

**Summary:** Mixed-precision GEMV with explicit A/B scaling tiles.

**Semantics:**

```
dst = gemv(a, b)   // quantization/mixed-precision behavior is target-defined
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `pto.tile_buf` | Matrix tile (`loc=left`) |
| `a_scale` | `pto.tile_buf` | Scale tile associated with `a` |
| `b` | `pto.tile_buf` | Vector tile (`loc=right`) |
| `b_scale` | `pto.tile_buf` | Scale tile associated with `b` |
| `dst` | `pto.tile_buf` | Destination accumulator tile (`loc=acc`) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The destination element type must be `f32`.
  - Supported `a`/`b` element type pairs are:
    - `f8E4M3FN` / `f8E4M3FN`
    - `f8E4M3FN` / `f8E5M2`
    - `f8E5M2` / `f8E4M3FN`
    - `f8E5M2` / `f8E5M2`
    - any pair from `!pto.f4E1M2x2` and `!pto.f4E2M1x2`
  - `a`/`b`/`dst` follow the same A5 tile location and layout constraints as `pto.tmatmul`.
  - `a_scale` and `b_scale` must be tile buffers in `loc=scaling`.
  - `a_scale` shape and valid shape must be `[M, ceil(K/32)]`.
  - `b_scale` shape and valid shape must be `[ceil(K/32), N]`.
  - `a_scale` must use `blayout=row_major`, `slayout=row_major`, `fractal=32`.
  - `b_scale` must use `blayout=col_major`, `slayout=col_major`, `fractal=32`.
  - Runtime: `m/k/n` are taken from `a valid row`, `a valid column`, and `b valid column`, and must be in `[1, 4095]`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                            !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx.acc` - Mixed-Precision GEMV with Accumulation

**Summary:** Mixed-precision GEMV accumulation form using scale tiles.

**Semantics:**

```
dst = c_in + gemv(a, b)
```

**Arguments:** `c_in, a, a_scale, b, b_scale, dst`

**Constraints & Verification:**

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The `a`/`b`/`dst` and scaling tile constraints are the same as `pto.tgemv.mx`.
  - `c_in` must be an accumulator tile with the same element type and valid shape as `dst`.
  - Runtime: `m/k/n` are taken from `a valid row`, `a valid column`, and `b valid column`, and must be in `[1, 4095]`.

**Hardware Mapping:** Matrix pipeline (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                        !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                 outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx.bias` - Mixed-Precision GEMV with Bias

**Summary:** Mixed-precision GEMV bias form using scale tiles.

**Semantics:**

```
dst = gemv(a, b) + bias
```

**Arguments:** `a, a_scale, b, b_scale, bias, dst`

**Constraints & Verification:**

- **Implementation checks (A5)**
  - Supported only on A5 targets.
  - The `a`/`b`/`dst` and scaling tile constraints are the same as `pto.tgemv.mx`.
  - `bias` must use element type `f32`, `loc=bias`, `rows=1`, and `blayout=row_major`.
  - `bias` and `dst` must have the same valid shape.
  - Runtime: `m/k/n` are taken from `a valid row`, `a valid column`, and `b valid column`, and must be in `[1, 4095]`.

**Hardware Mapping:** Matrix pipeline (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                            !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                  outs(%c : !pto.tile_buf<...>)
```

---

### 4.5 Vector Arithmetic Operations

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
| `pto.tpartargmax` | Partial elementwise max with index propagation |
| `pto.tpartargmin` | Partial elementwise min with index propagation |
| `pto.tpartmul` | Partial elementwise mul |
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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`, `bf16`, `i8`.
  - Tile must use row-major layout (`blayout=row_major`).

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

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
| `tmp` | `pto.tile_buf` | Temporary tile buffer required by the ISA API |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Division-by-zero**:
  - Behavior is target-defined.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

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
| `tmp` | `pto.tile_buf` | A2/A3 workspace tile. On A5 this operand is kept for ABI compatibility and is not used by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trem ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.
- **Implementation checks (A2A3)**
  - `src0/src1/dst` element type must match, and must be `i32` or `f32`.
  - `tmp` element type must match `dst`.
  - `src0/src1/dst` must use row-major layout (`blayout=row_major`).
  - `tmp` must be a row-major `loc=vec` tile.
  - `src0/src1/dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `2` valid rows and `tmp.validCol >= dst.validCol`.
- **Implementation checks (A5)**
  - `src0/src1/dst` element type must match, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - `src0/src1/dst` must use row-major layout (`blayout=row_major`).
  - `src0/src1/dst` must have the same `validRow/validCol`.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no element-type, shape, or valid-shape relation with `src0/src1/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trem ins(%a, %b, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=2, cols=16,
             v_row=2, v_col=16, blayout=row_major, slayout=none_box,
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

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Only certain partial-validity patterns are handled (e.g., one source equal to `dst` while the other is smaller by valid-rows or valid-cols); other patterns are not supported (target-defined behavior).

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

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

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

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

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

##### `pto.tpartargmax` - Partial Elementwise ArgMax

**Summary:** Partial elementwise max that also propagates the selected element index.

**Semantics:**

```
For each element (i, j) in the valid region:
    if src0[i, j] is selected:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

The selection is the target-defined partial max comparison between `src0` and `src1`. Equal-value tie behavior follows the underlying PTO ISA implementation.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source value tile buffer |
| `src1` | `pto.tile_buf` | Second source value tile buffer |
| `src0Idx` | `pto.tile_buf` | Index tile paired with `src0` |
| `src1Idx` | `pto.tile_buf` | Index tile paired with `src1` |
| `dst` | `pto.tile_buf` | Destination value tile buffer |
| `dstIdx` | `pto.tile_buf` | Destination index tile buffer |

**Results:** None. Writes into `dst` and `dstIdx` via DPS pattern.

**Assembly Format:**

```
pto.tpartargmax ins(<src0>, <src1>, <src0Idx>, <src1Idx> :
                    <src0_type>, <src1_type>, <src0Idx_type>, <src1Idx_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` element types must be identical and have the same shape.
- `src0Idx`, `src1Idx`, and `dstIdx` element types must be identical and must be `i32`.
- Data tiles and index tiles must have the same shape.
- Each data tile and its paired index tile must have the same valid shape.
- Uses the same partial valid-region constraints as `pto.tpartmax`.
- **Implementation checks (A2A3)**: data element type must be one of `i32`, `i16`, `f16`, `f32`.
- **Implementation checks (A5)**: data element type must be one of `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- EmitC lowers to `TPARTARGMAX(dst, src0, src1, dstIdx, src0Idx, src1Idx)`

**Basic Example:**

```mlir
pto.tpartargmax ins(%a, %b, %a_idx, %b_idx :
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
                outs(%c, %c_idx :
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

##### `pto.tpartargmin` - Partial Elementwise ArgMin

**Summary:** Partial elementwise min that also propagates the selected element index.

**Semantics:**

```
For each element (i, j) in the valid region:
    if src0[i, j] is selected:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

The selection is the target-defined partial min comparison between `src0` and `src1`. Equal-value tie behavior follows the underlying PTO ISA implementation.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source value tile buffer |
| `src1` | `pto.tile_buf` | Second source value tile buffer |
| `src0Idx` | `pto.tile_buf` | Index tile paired with `src0` |
| `src1Idx` | `pto.tile_buf` | Index tile paired with `src1` |
| `dst` | `pto.tile_buf` | Destination value tile buffer |
| `dstIdx` | `pto.tile_buf` | Destination index tile buffer |

**Results:** None. Writes into `dst` and `dstIdx` via DPS pattern.

**Assembly Format:**

```
pto.tpartargmin ins(<src0>, <src1>, <src0Idx>, <src1Idx> :
                    <src0_type>, <src1_type>, <src0Idx_type>, <src1Idx_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` element types must be identical and have the same shape.
- `src0Idx`, `src1Idx`, and `dstIdx` element types must be identical and must be `i32`.
- Data tiles and index tiles must have the same shape.
- Each data tile and its paired index tile must have the same valid shape.
- Uses the same partial valid-region constraints as `pto.tpartmin`.
- **Implementation checks (A2A3)**: data element type must be one of `i32`, `i16`, `f16`, `f32`.
- **Implementation checks (A5)**: data element type must be one of `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- EmitC lowers to `TPARTARGMIN(dst, src0, src1, dstIdx, src0Idx, src1Idx)`

**Basic Example:**

```mlir
pto.tpartargmin ins(%a, %b, %a_idx, %b_idx :
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
                outs(%c, %c_idx :
                    !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                    !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

##### `pto.tpartmul` - Partial Elementwise Mul

**Summary:** Partial elementwise mul with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] * src1[i, j]
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

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
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
| `tmp` | `pto.tile_buf` | A2/A3 workspace tile. On A5 this operand is kept for ABI compatibility and is not read by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tprelu ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `f16`, `f32`.
  - `tmp` element type must be an 8-bit integer tile type.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`) and have the same shape.
  - `tmp` must be a row-major `loc=vec` tile.
  - `src0` and `src1` must have the same `validRow/validCol` as `dst`.
  - `tmp.shape[0] >= dst.validRow + 1`. The A2/A3 `TPRELU` implementation uses one extra physical tmp row as scratch when materializing row cmp-mask addresses for `TSEL`.
  - `tmp.validCol >= ceil(dst.validCol / 8)`. The tmp valid region stores one packed predicate bit per destination element.
  - `tmp.validRow` does not need to cover the extra scratch row. PTO IR follows the official A2/A3 runtime contract: the extra row is a physical workspace row addressed through `TSUBVIEW`, not part of the tmp valid region.
  - On A3, `src0`, `src1`, `tmp`, and `dst` must use different storage ranges without overlap.
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `f16`, `f32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`) and have the same shape.
  - `src0` and `src1` must have the same `validRow/validCol` as `dst`.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no shape or valid-shape relation with `src0/src1/dst` is required.


**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// A2/A3
pto.tprelu ins(%a, %slopes, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=ui8, rows=17, cols=32,
               v_row=16, v_col=2, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
// A5 tmp reused out %c
pto.tprelu ins(%a, %slopes, %c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
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
| `pto.taxpy` | `dst[i,j] = dst[i,j] + src[i,j] * scalar` |
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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to add to each element |
| `dst` | `pto.tile_buf` | Destination tile buffer for the result |

**Results:** None. The operation writes results into `dst` following the Destination-Passing Style (DPS) pattern.

**Assembly Format:**

```
pto.tadds ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
  
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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to subtract |
| `dst` | `pto.tile_buf` | Destination tile buffer for the result |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar multiplier |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmuls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.

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

##### `pto.taxpy` - Multiply-Add Into Destination Tile

**Summary:** Updates the destination tile in place with `dst += src * scalar`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = dst[i, j] + src[i, j] * scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar multiplier |
| `dst` | `pto.tile_buf` | Destination tile buffer, updated in place |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taxpy ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src` and `dst` must use `loc=vec`.
  - `src` and `dst` must have the same shape and valid shape.
  - `scalar` type must exactly match the element type of `src`.
  - `src` element type must be `f16` or `f32`.
  - `dst` element type must be `f16` or `f32`.
  - Element types must either match, or use the widening form `src=f16`, `dst=f32`.
- **Implementation checks (A5)**
  - `src` and `dst` must use `loc=vec`.
  - `src` and `dst` must have the same shape and valid shape.
  - `scalar` type must exactly match the element type of `src`.
  - `src` element type must be `f16`, `bf16`, or `f32`.
  - `dst` element type must be `f16`, `bf16`, or `f32`.
  - Element types must either match, or use the widening form `src=f16`, `dst=f32`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
%alpha = arith.constant 2.0 : f16
pto.taxpy ins(%src, %alpha : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f16)
          outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
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
| `src/scalar` | `pto.tile_buf/scalar` | Source tile buffer or Scalar divisor (or dividend in reverse mode)|
| `src/scalar` | `pto.tile_buf/scalar` | Source tile buffer or Scalar divisor (or dividend in reverse mode)|
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

- **A2/A3 constraints (both overloads):**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **A5 constraints (both overloads):**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Division-by-zero**:
  - Behavior is target-defined; on A5 the tile/scalar form maps to multiply-by-reciprocal and uses `1/0 -> +inf` for `scalar == 0`.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmaxs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **A2/A3 constraints (both overloads):**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.
- **A5 constraints (both overloads):**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmins ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar divisor |
| `tmp` | `pto.tile_buf` | Temporary tile buffer required by the ISA API |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trems ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- **Implementation checks (A2A3)**
  - `src/dst` element type must match, and must be `i32` or `f32`.
  - `scalar` type must match the tile element type.
  - `tmp` element type must match `dst`.
  - `src/dst` must use row-major layout (`blayout=row_major`).
  - `tmp` must be a row-major `loc=vec` tile.
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row and `tmp.validCol >= dst.validCol`.
- **Implementation checks (A5)**
  - `src/dst` element type must match, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - `scalar` type must match the tile element type.
  - `src/dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no element-type, shape, or valid-shape relation with `src/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trems ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32,
              v_row=1, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
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

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

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

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taddsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

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

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.

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

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.

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

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.
- **Domain / NaN**:
  - Domain behavior (e.g., `log(<=0)`) is target-defined.

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

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.
- **Domain / NaN**:
  - Behavior is target-defined (e.g., for negative inputs).

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

##### `pto.ttri` - Fill Triangular Tile Region

**Summary:** Fills a VEC tile using the PTO-ISA `TTRI` triangular pattern.

**Semantics:**

```
TTRI(dst, diagonal)
```

`upperOrLower=0` selects the lower-triangular form and `upperOrLower=1`
selects the upper-triangular form. The exact per-element fill pattern follows
the target PTO-ISA implementation.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `diagonal` | integer SSA value | Runtime diagonal selector |
| `dst` | `pto.tile_buf` | Destination vector tile |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `upperOrLower` | `I32Attr` (default: `0`) | `0` for lower triangular, `1` for upper triangular |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```mlir
pto.ttri ins(%diag {upperOrLower = 1 : i32} : i32)
         outs(%dst : !pto.tile_buf<...>)
```

**Constraints & Verification:**

- `dst` must be a VEC tile (`loc=vec`) whose valid shape stays within the static tile shape.
- `dst` must use `blayout=row_major`.
- `diagonal` must have an integer type.
- `upperOrLower` must be either `0` or `1`.
- Supported element types:
  - A2/A3: `f16`, `f32`, `i16`, `i32`, `u16`, `u32`
  - A5: `f16`, `f32`, `bf16`, `i8`, `i16`, `i32`, `u8`, `u16`, `u32`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.ttri ins(%diag : i32)
         outs(%lower : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
               v_row=32, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

pto.ttri ins(%diag {upperOrLower = 1 : i32} : i32)
         outs(%upper : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
               v_row=32, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trsqrt` - Elementwise Reciprocal Square Root

**Summary:** Computes the reciprocal square root for every element. The tmp-taking form is optional and is not shape-coupled to `src` or `dst`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile |
| `dst` | `pto.tile_buf` | Destination tile buffer |
| `precisionType` | `#pto<rsqrt_precision ...>` | Precision selector. `high_precision` requires `tmp` |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trsqrt ins(<src> : <src_type>) outs(<dst> : <dst_type>)
pto.trsqrt ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>) {precisionType = #pto<rsqrt_precision high_precision>}
```

**Constraints & Verification:**

- Tile element type must be `f16` or `f32`.
- `src` and `dst` must use `loc=vec` and `blayout=row_major`.
- `src` and `dst` must have the same valid region.
- `precisionType = #pto<rsqrt_precision high_precision>` requires `tmp`.
- When `tmp` is provided, it must be a `loc=vec` tile with at least `32` bytes of static storage.
- PTO IR does not require `tmp` to match the shape or valid shape of `src` / `dst`.
- **Domain / NaN**:
  - Behavior is target-defined (e.g., for `src == 0` or negative inputs).

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

pto.trsqrt ins(%a, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
               v_row=1, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
               v_row=1, v_col=8, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
               v_row=1, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>) {precisionType = #pto<rsqrt_precision high_precision>}
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

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - Tile must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`;
  - Tile must use row-major layout (`blayout=row_major`).
  - A3's TRECIP instruction does not support setting the source Tile and destination Tile to the same memory.
- **Domain / NaN**:
  - Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`, `i32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `f16`, `f32`, `i32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.

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

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `0 <= valid row <= rows` and `0 <= valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.

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

### 4.6 Reduction Operations

Reduce along rows or columns of a tile. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowsum` | `dst[i,0] = sum_j src[i,j]` (requires tmp) |
| `pto.trowprod` | `dst[i,0] = prod_j src[i,j]` (requires tmp) |
| `pto.trowmax` | `dst[i,0] = max_j src[i,j]` (requires tmp) |
| `pto.trowargmax` | `dst[i,0] = argmax_j src[i,j]` (requires tmp) |
| `pto.trowmin` | `dst[i,0] = min_j src[i,j]` (requires tmp) |
| `pto.trowargmin` | `dst[i,0] = argmin_j src[i,j]` (requires tmp) |
| `pto.tcolsum` | `dst[0,j] = sum_i src[i,j]` (requires tmp, optional isBinary) |
| `pto.tcolmax` | `dst[0,j] = max_i src[i,j]` |
| `pto.tcolargmax` | `dst[0,j] = argmax_i src[i,j]` (requires tmp) |
| `pto.tcolmin` | `dst[0,j] = min_i src[i,j]` |
| `pto.tcolargmin` | `dst[0,j] = argmin_i src[i,j]` (requires tmp) |
| `pto.thistogram` | `dst[i, idx[i,0]] = histogram_update(dst[i, idx[i,0]], src[i,:])` (A5 only) |

---

##### `pto.trowsum` - Row-wise Sum Reduction

**Summary:** Reduces each row by summing across columns. The current DPS form requires a `tmp` operand.

**Semantics:**

```
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowsum ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- `tmp` must be a `loc=vec` tile.
- `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
- `dst` must use `slayout=none_box` and either:
  - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
  - a legacy ND-style tile with `valid column == 1`.
- `src` and `dst` must have the same element type.
- Supported element types are `i16`, `i32`, `f16`, and `f32`.
- Runtime valid checks:
  - `src valid_shape[0]` and `src valid_shape[1]` must be non-zero.
  - `src valid_shape[0] == dst valid_shape[0]`.
  - `dst valid_shape[1] == 1`.
- PTO IR does not impose extra `tmp` element-type, shape, or layout constraints for this op.
- On A2/A3, `tmp` is consumed as reduction workspace.
- On A5, `tmp` is kept for ABI compatibility and the current implementation does not consume it.
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowsum ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowmax` - Row-wise Max Reduction

**Summary:** Reduces each row by taking the maximum across columns. The current DPS form requires a `tmp` operand.

**Semantics:**

```
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- `tmp` must be a `loc=vec` tile and have the same element type as `src`.
- `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
- `dst` must use `slayout=none_box` and either:
  - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
  - a legacy ND-style tile with `valid column == 1`.
- `src` and `dst` must have the same element type.
- Supported element types are `i16`, `i32`, `f16`, and `f32`.
- Runtime valid checks:
  - `src valid_shape[0]` and `src valid_shape[1]` must be non-zero.
  - `src valid_shape[0] == dst valid_shape[0]`.
  - `dst valid_shape[1] == 1`.
- PTO IR does not impose extra `tmp` shape or layout constraints for this op.
- On A2/A3, `tmp` is consumed as reduction workspace.
- On A5, `tmp` is kept for ABI compatibility and the current implementation does not consume it.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowargmax` - Row-wise ArgMax Reduction

**Summary:** Reduces each row to the column index of its maximum element. Requires a temporary buffer.

**Semantics:**

```
For each row i:
    dst[i, 0] = argmax over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | A2/A3 reduction workspace tile. On A5 this operand is kept for ABI compatibility and is not used by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer containing row-wise indices |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src`, `tmp`, and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - `tmp` must have the same element type as `src`.
  - `elementPerRepeat = 2048 / bitwidth(src element type)`.
  - `elementPerBlock = 256 / bitwidth(src element type)`.
  - PTO IR accepts either a legacy `tmp` whose known `valid_shape` exactly matches `src`, or a smaller workspace tile.
  - For `src.validCol <= elementPerRepeat`, `tmp` may be either:
    - a DN single-column tile with `tmp.validCol == 1` and `tmp.validRow >= 2 * src.validRow`, or
    - an ND tile with `tmp.validRow >= src.validRow` and `tmp.validCol >= 2`.
  - For `src.validCol > elementPerRepeat`, `tmp.validRow >= src.validRow`, and if the physical row count is statically known it must match `src.rows`.
  - In the large-column path, the minimum required `tmp.validCol` is `stride`, where `repeats = ceil(src.validCol / elementPerRepeat)` and `stride = (ceil(repeats * 2 / elementPerBlock) + ceil(repeats / elementPerBlock)) * elementPerBlock`.
  - `dst` must use `slayout=none_box` and either:
    - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
    - a legacy ND-style tile with `valid column == 1`.
  - `src` element type must be `i16`, `i32`, `f16`, or `f32`.
  - `dst` element type must be `i32` or `ui32`.
  - Runtime valid checks:
    - `src valid row != 0` and `src valid column != 0`
    - `src valid row == dst valid row`
    - `dst valid column == 1`
- **Implementation checks (A5)**
  - `src` and `dst` follow the same layout, element-type, and valid-region rules as A2/A3.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no shape or valid-shape relation with `src/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=2,
                   v_row=16, v_col=2, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
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
| `tmp` | `pto.tile_buf` | Workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- `tmp` must be a `loc=vec` tile and have the same element type as `src`.
- `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
- `dst` must use `slayout=none_box` and either:
  - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
  - a legacy ND-style tile with `valid column == 1`.
- `src` and `dst` must have the same element type.
- Supported element types are `i16`, `i32`, `f16`, and `f32`.
- Runtime valid checks:
  - `src valid_shape[0]` and `src valid_shape[1]` must be non-zero.
  - `src valid_shape[0] == dst valid_shape[0]`.
  - `dst valid_shape[1] == 1`.
- PTO IR does not impose extra `tmp` shape or layout constraints for this op.
- On A2/A3, `tmp` is consumed as reduction workspace.
- On A5, `tmp` is kept for ABI compatibility and the current implementation does not consume it.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowargmin` - Row-wise ArgMin Reduction

**Summary:** Reduces each row to the column index of its minimum element. Requires a temporary buffer.

**Semantics:**

```
For each row i:
    dst[i, 0] = argmin over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | A2/A3 reduction workspace tile. On A5 this operand is kept for ABI compatibility and is not used by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer containing row-wise indices |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src`, `tmp`, and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - `tmp` must have the same element type as `src`.
  - `elementPerRepeat = 2048 / bitwidth(src element type)`.
  - `elementPerBlock = 256 / bitwidth(src element type)`.
  - PTO IR accepts either a legacy `tmp` whose known `valid_shape` exactly matches `src`, or a smaller workspace tile.
  - For `src.validCol <= elementPerRepeat`, `tmp` may be either:
    - a DN single-column tile with `tmp.validCol == 1` and `tmp.validRow >= 2 * src.validRow`, or
    - an ND tile with `tmp.validRow >= src.validRow` and `tmp.validCol >= 2`.
  - For `src.validCol > elementPerRepeat`, `tmp.validRow >= src.validRow`, and if the physical row count is statically known it must match `src.rows`.
  - In the large-column path, the minimum required `tmp.validCol` is `stride`, where `repeats = ceil(src.validCol / elementPerRepeat)` and `stride = (ceil(repeats * 2 / elementPerBlock) + ceil(repeats / elementPerBlock)) * elementPerBlock`.
  - `dst` must use `slayout=none_box` and either:
    - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
    - a legacy ND-style tile with `valid column == 1`.
  - `src` element type must be `i16`, `i32`, `f16`, or `f32`.
  - `dst` element type must be `i32` or `ui32`.
  - Runtime valid checks:
    - `src valid row != 0` and `src valid column != 0`
    - `src valid row == dst valid row`
    - `dst valid column == 1`
- **Implementation checks (A5)**
  - `src` and `dst` follow the same layout, element-type, and valid-region rules as A2/A3.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no shape or valid-shape relation with `src/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=2,
                   v_row=16, v_col=2, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

##### `pto.trowprod` - Row-wise Product Reduction

**Summary:** Reduces each row by multiplying across columns. Requires a temporary buffer.

**Semantics:**

```
For each row i:
    dst[i, 0] = product over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowprod ins(<src>, <tmp> : <src_type>, <tmp_type>)
             outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- `tmp` must be a `loc=vec` tile and have the same element type as `src`.
- `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
- `dst` must use `slayout=none_box` and either:
  - a DN-style column vector tile (`blayout=col_major`, `cols=1`), or
  - a legacy ND-style tile with `valid column == 1`.
- `src` and `dst` must have the same element type.
- Supported element types are `i16`, `i32`, `f16`, and `f32`.
- Runtime valid checks:
  - `src valid_shape[0]` and `src valid_shape[1]` must be non-zero.
  - `src valid_shape[0] == dst valid_shape[0]`.
  - `dst valid_shape[1] == 1`.
- PTO IR does not impose extra `tmp` shape or layout constraints for this op.
- On A2/A3, `tmp` is consumed as reduction workspace.
- On A5, `tmp` is kept for ABI compatibility and the current implementation does not consume it.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
                 v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=1,
                 v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.thistogram` - Row-wise Histogram Accumulation

**Summary:** Updates a 256-bin histogram row in `dst` using A5 `THISTOGRAM` modes selected by the `byte` attribute.

**Semantics:**

```
For `src : ui16`, each row updates one histogram bin selected by `idx[i, 0]`.

For `src : ui32`, the exact filtering behavior is defined by the backend `THISTOGRAM<HistByte::BYTE_n>` intrinsic:
- `byte = 3`: histogram byte 3 directly
- `byte = 2`: histogram byte 2, filtered by byte 3
- `byte = 1`: histogram byte 1, filtered by bytes 3 and 2
- `byte = 0`: histogram byte 0, filtered by bytes 3, 2, and 1
```

The exact accumulation performed inside the selected bin is target-defined by the hardware `THISTOGRAM` intrinsic. The `byte` attribute maps directly to `HistByte::BYTE_<byte>`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer (`ui16` or `ui32`) |
| `idx` | `pto.tile_buf` | Selector/filter tile interpreted according to `src` dtype and `byte` |
| `dst` | `pto.tile_buf` | Destination histogram tile |
| `byte` | `I32Attr` (default: `1`) | Selects `THISTOGRAM<HistByte::BYTE_<byte>>`, valid range `[0, 3]` |

Legacy inputs using `isMSB` are accepted for compatibility: `isMSB = false`
maps to `byte = 0`, and `isMSB = true` maps to `byte = 1`. Specifying both
`byte` and a conflicting legacy `isMSB` is rejected.

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.thistogram ins(<src>, <idx> : <src_type>, <idx_type>)
               outs(<dst> : <dst_type>)
               {byte = 1 : i32}
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Not supported.
- **Implementation checks (A5)**
  - `src`, `idx`, and `dst` must all be `tile_buf` values in `loc=vec`.
  - `src` and `dst` must use `row_major + none_box` layout.
  - `src` element type must be `ui16` or `ui32`.
  - `idx` element type must be `ui8`.
  - `dst` element type must be `ui32`.
  - `byte` must be in `[0, 3]`.
  - `src`, `idx`, and `dst` must all be rank-2 tiles.
  - `dst` rows and valid rows must match `src`.
  - `dst` shape[1] and valid_shape[1] must be at least `256`.
  - When `src` is `ui16`:
    - `byte` must be `0` or `1`.
    - `idx` must use DN-style layout (`col_major + none_box`).
    - `idx` rows and valid rows must match `src`.
    - `idx` must have exactly one column.
  - When `src` is `ui32`:
    - When `byte = 3`, `idx` is accepted but not semantically used by the A5 backend intrinsic; no additional layout or shape constraints are imposed beyond the generic `tile_buf`, `loc=vec`, `dtype=ui8`, and rank-2 requirements.
    - When `byte = 2`, `1`, or `0`, `idx` must use `row_major + none_box` layout. `idx` cols and valid cols must match `src`.
    - When `byte = 2`, `idx` rows / valid rows must be `1`.
    - When `byte = 1`, `idx` rows / valid rows must be `2`.
    - When `byte = 0`, `idx` rows / valid rows must be `3`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.thistogram ins(%src, %idx : !pto.tile_buf<loc=vec, dtype=ui16, rows=8, cols=32,
                   v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=ui8, rows=8, cols=1,
                   v_row=8, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=8, cols=256,
                   v_row=8, v_col=256, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>) {byte = 0 : i32}
```

---

##### `pto.tcolsum` - Column-wise Sum Reduction

**Summary:** Reduces each column by summing across rows. The tmp-taking form is optional; when present, it models a workspace tile instead of a same-shape scratch tile.

**Semantics:**

```
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile used by the tmp-taking form |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |
| `isBinary` | `BoolAttr` (default: `false`) | Use binary reduction tree |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolsum ins(<src> : <src_type>)
            outs(<dst> : <dst_type>)
pto.tcolsum ins(<src>, <tmp> {isBinary = true} : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- `src` and `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src` and `dst` must have the same element type.
- `src valid_shape[1]` must equal `dst valid_shape[1]`.
- **Implementation checks (A2/A3)**:
  - Supported element types: `f16`, `f32`, `i16`, `i32`.
- **Implementation checks (A5)**:
  - Supported element types: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - `src valid_shape[0]` and `src valid_shape[1]` must both be non-zero.
- When `tmp` is present:
  - `tmp` must also use `loc=vec` and ND-style layout.
  - `tmp` must have the same element type as `src` / `dst`.
  - In the current PTO IR surface, `tmp` and `isBinary` must appear together.
  - When `isBinary = true`, `tmp shape[1]` must be at least `src valid_shape[1]`.
- PTO IR does not require `tmp` to match the full shape or valid shape of `src`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolsum ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>) {isBinary = true}
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

- **Implementation checks (A2A3):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i16`, `i32`, and `dst_type == tmp_type == src_type`.
- `src valid column == dst valid column`;
- **Implementation checks (A5):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i8`, `i16`, `i32`,`bf16`, and `dst_type == tmp_type == src_type`.
- `src valid row` and `src valid column` must be non-zero; `src valid column == dst valid column` is required.

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

##### `pto.tcolargmax` - Column-wise ArgMax Reduction

**Summary:** Reduces each column to the row index of its maximum element. Requires a temporary buffer.

**Semantics:**

```
For each column j:
    dst[0, j] = argmax over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | A2/A3 reduction workspace tile. On A5 this operand is kept for ABI compatibility and is not used by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer containing column-wise indices |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src`, `tmp`, and `dst` must use `loc=vec`.
  - `src` and `dst` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - `tmp` must be a row-major `loc=vec` tile and must have the same element type as `src`.
  - `elementPerRepeat = 2048 / bitwidth(src element type)`.
  - `elementPerBlock = 256 / bitwidth(src element type)`.
  - PTO IR accepts either a legacy `tmp` whose known `valid_shape` exactly matches `src`, or a smaller workspace tile.
  - For the workspace form, `tmp.validRow >= 1`.
  - The minimum required `tmp.validCol` is `stride`, where `repeats = ceil(src.validCol / elementPerRepeat)` and `stride = (ceil(repeats * 2 / elementPerBlock) + ceil(repeats / elementPerBlock)) * elementPerBlock`.
  - `src` element type must be `f16` or `f32`.
  - `dst` element type must be `i32` or `ui32`.
  - Runtime valid checks:
    - `src valid row != 0` and `src valid column != 0`
    - `dst valid row == 1`
    - `src valid column == dst valid column`
- **Implementation checks (A5)**
  - `src` and `dst` follow the same layout, element-type, and valid-region rules as A2/A3.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no shape or valid-shape relation with `src/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
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

- **Implementation checks (A2A3):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i16`, `i32`, and `dst_type == tmp_type == src_type`.
- `src valid column == dst valid column`;
- **Implementation checks (A5):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i8`, `i16`, `i32`,`bf16`, and `dst_type == tmp_type == src_type`.
- `src valid row` and `src valid column` must be non-zero; `src valid column == dst valid column` is required.

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

##### `pto.tcolargmin` - Column-wise ArgMin Reduction

**Summary:** Reduces each column to the row index of its minimum element. Requires a temporary buffer.

**Semantics:**

```
For each column j:
    dst[0, j] = argmin over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | A2/A3 reduction workspace tile. On A5 this operand is kept for ABI compatibility and is not used by the instruction. |
| `dst` | `pto.tile_buf` | Destination tile buffer containing column-wise indices |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `src`, `tmp`, and `dst` must use `loc=vec`.
  - `src` and `dst` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - `tmp` must be a row-major `loc=vec` tile and must have the same element type as `src`.
  - `elementPerRepeat = 2048 / bitwidth(src element type)`.
  - `elementPerBlock = 256 / bitwidth(src element type)`.
  - PTO IR accepts either a legacy `tmp` whose known `valid_shape` exactly matches `src`, or a smaller workspace tile.
  - For the workspace form, `tmp.validRow >= 1`.
  - The minimum required `tmp.validCol` is `stride`, where `repeats = ceil(src.validCol / elementPerRepeat)` and `stride = (ceil(repeats * 2 / elementPerBlock) + ceil(repeats / elementPerBlock)) * elementPerBlock`.
  - `src` element type must be `f16` or `f32`.
  - `dst` element type must be `i32` or `ui32`.
  - Runtime valid checks:
    - `src valid row != 0` and `src valid column != 0`
    - `dst valid row == 1`
    - `src valid column == dst valid column`
- **Implementation checks (A5)**
  - `src` and `dst` follow the same layout, element-type, and valid-region rules as A2/A3.
  - `tmp` is not used by the A5 implementation. PTO IR still requires it to be a row-major `loc=vec` tile, but no shape or valid-shape relation with `src/dst` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### 4.7 Broadcast Operations

Broadcast values across rows or columns. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowexpand` | Broadcast `src[i,0]` across row `i` |
| `pto.tcolexpand` | Broadcast `src[0,j]` across column `j` |
| `pto.tcolexpandmul` | `dst[i,j] = src0[i,j] * src1[0,j]` |
| `pto.tcolexpandadd` | `dst[i,j] = src0[i,j] + src1[0,j]` |
| `pto.tcolexpanddiv` | `dst[i,j] = src0[i,j] / src1[0,j]` |
| `pto.tcolexpandsub` | `dst[i,j] = src0[i,j] - src1[0,j]` |
| `pto.tcolexpandexpdif` | `dst[i,j] = exp(src0[i,j] - src1[0,j])` |
| `pto.tcolexpandmax` | `dst[i,j] = max(src0[i,j], src1[0,j])` |
| `pto.tcolexpandmin` | `dst[i,j] = min(src0[i,j], src1[0,j])` |
| `pto.trowexpandmul` | `dst[i,j] = src0[i,j] * src1[i,0]` |
| `pto.trowexpanddiv` | `dst[i,j] = src0[i,j] / src1[i,0]` |
| `pto.trowexpandsub` | `dst[i,j] = src0[i,j] - src1[i,0]` |
| `pto.trowexpandadd` | `dst[i,j] = src0[i,j] + src1[i,0]` |
| `pto.trowexpandexpdif` | `dst[i,j] = exp(src0[i,j] - src1[i,0])` |
| `pto.trowexpandmax` | `dst[i,j] = max(src0[i,j], src1[i,0])` |
| `pto.trowexpandmin` | `dst[i,j] = min(src0[i,j], src1[i,0])` |
| `pto.texpands` | Broadcast scalar to all elements of dst |

For `pto.trowexpandadd/trowexpandsub/trowexpandmul/trowexpandmax/trowexpandmin`
and `pto.tcolexpandadd/tcolexpandsub/tcolexpandmul/tcolexpandmax/tcolexpandmin`,
element-type constraints are:
- A2/A3: `i16`, `i32`, `f16`, `f32`
- A5: `i8`, `i16`, `i32`, `f16`, `f32`

`pto.trowexpandexpdif` and `pto.tcolexpandexpdif` remain floating-point only (`f16`/`f32`).

All `pto.trowexpandmul/div/sub/add/expdif/max/min` ops accept both the plain
two-source form and the tmp-taking form. PTO IR does not require `tmp` to
match the shape or valid shape of `src0` / `src1` / `dst`. On A5, `tmp` is
accepted as a compatibility placeholder and may be smaller than the data tiles.

For `pto.trowexpandexpdif/max/min`, PTO IR requires one source to match
`dst valid_shape` and the other to be a per-row scalar vector. When the A3
tmp-taking form is used, that scalar-vector operand must use a non-row-major
layout. When `tmp` is present, it must have the same element type as `dst`.

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

- **NPU constraints:**

- `src` and `dst` must use `loc=vec`.
- `src` must `slayout=none_box`.
- `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `dst_type == src_type`
- Data type: A2/A3/A5 element types must be one of: `i8` or `i16` or `i32` or `f16` or `bf16` or `f32`.
- requires `src valid row == dst valid row` and requires `src valid row != 0 && src valid column != 0 && dst valid row != 0 && dst valid column != 0`.

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

- `src` and `dst` must use `loc=vec`.
- Both `src` and `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `dst_type == src_type`
- Data type: A2/A3/A5 element types must be one of: `i8` or `i16` or `i32` or `f16` or `bf16` or `f32`.
- requires `src valid column == dst valid column`

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

##### `pto.tcolexpandmul` - Column-wise Broadcast Multiply

**Summary:** Multiplies each element of `src0` by a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpanddiv` - Column-wise Broadcast Divide

**Summary:** Divides each element of `src0` by a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector (divisor) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type.
  - The shared element type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]` (one scalar per destination column).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandsub` - Column-wise Broadcast Subtract

**Summary:** Subtracts a per-column scalar from `src1` from each element of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector (subtrahend) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]` (one scalar per destination column).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandmax` - Column-wise Broadcast Max

**Summary:** Takes the elementwise maximum of `src0` and a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandmin` - Column-wise Broadcast Min

**Summary:** Takes the elementwise minimum of `src0` and a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
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
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be `f16` or `f32`.
  - `dst` must use row-major layout (`blayout=row_major`).
- When `tmp` is present:
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.
 
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                      v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
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
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpanddiv ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be `f16` or `f32`.
  - `dst` must use row-major layout (`blayout=row_major`).
- `precisionType = #pto<div_precision high_precision>` requires `tmp`.
- When `tmp` is present:
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                      v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
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
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `dst` must use row-major layout (`blayout=row_major`).
- When `tmp` is present:
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                      v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandadd` - Row-wise Broadcast Add

**Summary:** Adds a per-row scalar from `src1` to each row of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector |
| `tmp` | `pto.tile_buf` (optional) | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandadd ins(<src0>, <src1> [, <tmp>] : <src0_type>, <src1_type> [, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**
- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - Element type:
    - A2/A3: `i16`, `i32`, `f16`, `f32`
    - A5: `i8`, `i16`, `i32`, `f16`, `f32`
  - `src0` and `dst` must have the same `valid_shape`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[0]` must equal `dst valid_shape[0]`.
  - If `src1` is row-major: `src1 valid_shape[1] == 32 / sizeof(dtype)` (`16` for `f16`, `8` for `f32`).
  - If `src1` is not row-major: `src1 valid_shape[1] == 1`.
- When `tmp` is present:
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandexpdif` - Row-wise Broadcast Exp-Difference

**Summary:** Computes `exp(src0[i, j] - src1[i, 0])` using a per-row scalar vector.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[i, 0])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | One input tile; one of `src0` / `src1` must match `dst` valid shape |
| `src1` | `pto.tile_buf` | The other input; interpreted as a per-row scalar vector |
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandexpdif ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                     outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` must have the same element type.
- The shared element type must be `f16` or `f32`.
- `dst` must use row-major layout (`blayout=row_major`).
- One of `src0` / `src1` must match `dst valid_shape`; the other must be a per-row scalar vector.
- The per-row scalar-vector operand must satisfy:
  - `valid_shape[0] == dst valid_shape[0]`
  - if it is row-major, `valid_shape[1] == 32 / sizeof(dtype)` (`16` for `f16`, `8` for `f32`)
  - otherwise, `valid_shape[1] == 1`
- When the A3 tmp-taking form is used, the scalar-vector operand must use a non-row-major layout.
- When `tmp` is present:
  - `tmp` must have the same element type as `dst`.
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandexpdif ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                         v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                         v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                         v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                         v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

##### `pto.trowexpandmax` - Row-wise Broadcast Max

**Summary:** Takes the elementwise maximum of a full tile and a per-row scalar vector.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, 0])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | One input tile; one of `src0` / `src1` must match `dst` valid shape |
| `src1` | `pto.tile_buf` | The other input; interpreted as a per-row scalar vector |
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandmax ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` must have the same element type.
- Supported element types are:
  - A2/A3: `i16`, `i32`, `f16`, `f32`
  - A5: `i8`, `i16`, `i32`, `f16`, `f32`
- `dst` must use row-major layout (`blayout=row_major`).
- One of `src0` / `src1` must match `dst valid_shape`; the other must be a per-row scalar vector.
- The per-row scalar-vector operand must satisfy:
  - `valid_shape[0] == dst valid_shape[0]`
  - if it is row-major, `valid_shape[1] == 32 / sizeof(dtype)` (`32` for `i8`, `16` for `i16` / `f16`, `8` for `i32` / `f32`)
  - otherwise, `valid_shape[1] == 1`
- When the A3 tmp-taking form is used, the scalar-vector operand must use a non-row-major layout.
- When `tmp` is present:
  - `tmp` must have the same element type as `dst`.
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandmax ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                      v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandmin` - Row-wise Broadcast Min

**Summary:** Takes the elementwise minimum of a full tile and a per-row scalar vector.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, 0])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | One input tile; one of `src0` / `src1` must match `dst` valid shape |
| `src1` | `pto.tile_buf` | The other input; interpreted as a per-row scalar vector |
| `tmp` | `Optional<pto.tile_buf>` | Optional workspace tile on A2/A3; compatibility placeholder on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandmin ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` must have the same element type.
- Supported element types are:
  - A2/A3: `i16`, `i32`, `f16`, `f32`
  - A5: `i8`, `i16`, `i32`, `f16`, `f32`
- `dst` must use row-major layout (`blayout=row_major`).
- One of `src0` / `src1` must match `dst valid_shape`; the other must be a per-row scalar vector.
- The per-row scalar-vector operand must satisfy:
  - `valid_shape[0] == dst valid_shape[0]`
  - if it is row-major, `valid_shape[1] == 32 / sizeof(dtype)` (`32` for `i8`, `16` for `i16` / `f16`, `8` for `i32` / `f32`)
  - otherwise, `valid_shape[1] == 1`
- When the A3 tmp-taking form is used, the scalar-vector operand must use a non-row-major layout.
- When `tmp` is present:
  - `tmp` must have the same element type as `dst`.
  - PTO IR does not require `tmp` to match the shape or valid shape of `src0` / `src1` / `dst`.
  - On A5, `tmp` is accepted as a compatibility placeholder and is not consumed by the current implementation.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandmin ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8,
                      v_row=1, v_col=8, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                      v_row=16, v_col=64, blayout=row_major, slayout=none_box,
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
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to broadcast |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.texpands ins(<scalar> : <scalar_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`:
   - Tile must use row-major layout (`blayout=row_major`).
   - Valid bounds: `valid row <= rows` and `valid column <= cols`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`:
   - Valid bounds: `valid row <= rows` and `valid column <= cols`.

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

### 4.8 Compare & Select Operations

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

- **Implementation checks (A2A3)**:
  - Input type must be one of: `i32`, `f16`, `f32`.
  - Output type must be `i8`.
  - `src0/src1/dst` must use `loc=vec`.
  - Valid bounds: `src valid row <= src.rows` and `src valid column <= src.cols`.
  - `src0` and `dst` must have the same valid region: 
      `src0 valid row == src1 valid row == dst valid row` and `src0 valid column == src1 valid column`.
- **Implementation checks (A5)**:
  - Input type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.

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
| `scalar` | `ScalarType` (signless integer / float)| Scalar value to compare against |
| `cmpMode` | `CmpModeAttr` (default: EQ) | Comparison mode |
| `dst` | `pto.tile_buf` | Destination mask |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - Input type must be one of: `i32`, `f16`, `f32`, `i16`.
  - `src` and `dst` must use `loc=vec`.
  - Static valid bounds: `src valid row <= src.rows` and `src valid column <= src.cols`.
  - `src` and `dst` must have the same valid row.
- **Implementation checks (A5)**:
  - Input type must be one of: `i32`, `f16`, `f32`, `i16`, `i8`.
  - `src` and `dst` must use `loc=vec`.
  - Static valid bounds: `src valid row <= src.rows`, `src valid column <= src.cols`, `dst valid row <= dst.rows`, and `dst valid column <= dst.cols`.
  - `src` and `dst` must have the same valid row.

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
| `tmp` | `pto.tile_buf` | Placeholder / workspace operand kept by the current DPS form |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsel ins(<mask>, <src0>, <src1>, <tmp> : <mask_type>, <type0>, <type1>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src0`, `src1`, and `dst` must have the same element type.
- `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
- **Implementation checks (A2/A3)**:
  - The shared element type must be `i16`, `i32`, `f16`, `bf16`, or `f32`.
- **Implementation checks (A5)**:
  - The shared element type must be `i8`, `i16`, `i32`, `f16`, `bf16`, or `f32`.
- The DPS form keeps a `tmp` operand, but PTO IR does not impose an extra tmp shape, valid-shape, or layout relation.
- On A5, `tmp` may be a placeholder tile that is smaller than `src0` / `src1` / `dst`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsel ins(%mask, %a, %b, %tmp : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=8,
             v_row=1, v_col=8, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### `pto.tsels`

**Summary:** Selects between a source tile and a scalar using a mask tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = mask[i, j] ? src[i, j] : scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mask` | `pto.tile_buf` | Mask tile (select predicate carrier) |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Placeholder / workspace operand kept by the current DPS form |
| `scalar` | `ScalarType` | Scalar value selected when the mask bit is false |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsels ins(<mask>, <src>, <tmp>, <scalar> : <mask_type>, <src_type>, <tmp_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must have the same element type.
- `src` and `dst` must have the same valid region.
- **Implementation checks (A2/A3)**:
  - The shared element type must be `i16`, `i32`, `f16`, or `f32`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
- **Implementation checks (A5)**:
  - The shared element type must be `i8`, `i16`, `i32`, `f16`, or `f32`.
  - `mask`, `src`, and `dst` must use row-major layout (`blayout=row_major`).
- The DPS form keeps a `tmp` operand, but PTO IR does not impose an extra tmp shape, valid-shape, or layout relation.
- On A5, `tmp` may be a placeholder tile that is smaller than `src` / `dst`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
              v_row=1, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### 4.9 Bitwise Operations

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

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

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

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

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
| `tmp` | `pto.tile_buf` | Workspace operand on A2/A3; placeholder operand on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `src0`, `src1`, and `dst` must have the same element type.
- `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
- `src0` and `dst` must have the same valid region.
- `src1` and `dst` must have the same valid region.
- **Implementation checks (A2/A3)**:
  - The shared element type must be `i8` or `i16`.
  - `tmp` must have the same element type as `src0` / `src1` / `dst`.
  - `tmp` must use row-major layout and have the same valid region as `dst`.
- **Implementation checks (A5)**:
  - The shared element type must be `i8`, `i16`, or `i32`.
  - The with-`tmp` DPS form is retained, but PTO IR does not impose an extra tmp shape, valid-shape, or layout relation on A5.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// A2/A3
pto.txor ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
// A5: %tmp_small is a placeholder operand and may be smaller than src/dst.
pto.txor ins(%src0, %src1, %tmp_small : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8,
             v_row=1, v_col=8, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
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

- **Implementation checks (A2A3)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

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

- **Implementation checks (A2A3)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
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

## Constraints

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i16`.
  - `src.Dtype == dst.Dtype`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`.
  - `src.Dtype == dst.Dtype`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` tiles should have the same `validRow/validCol`.

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
| `scalar` | `AnySignlessInteger` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- Setting the source Tile and destination Tile to the same memory is **Unsupported**.
- **Implementation checks (A2A3)**:
  - `src` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0` and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

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

  - Setting the source Tile and destination Tile to the same memory is **Unsupported**.
- **Implementation checks (A2A3)**:
  - `src0` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  
**Unsupported**.

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
| `scalar` | `AnySignlessInteger` | Scalar value |
| `tmp` | `pto.tile_buf` | Workspace operand on A2/A3; placeholder operand on A5 |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.txors ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- Setting the source tile and destination tile to the same memory is **Unsupported**.
- `src` and `dst` must have the same element type.
- `src` and `dst` must use row-major layout (`blayout=row_major`).
- `src` and `dst` must have the same valid region.
- **Implementation checks (A2/A3)**:
  - The shared element type must be `i8` or `i16`.
  - `tmp` must have the same element type as `src` / `dst`.
  - `tmp` must use row-major layout.
- **Implementation checks (A5)**:
  - The shared element type must be `i8`, `i16`, or `i32`.
  - The current DPS form still carries `tmp`, but PTO IR does not impose an extra tmp shape, valid-shape, or layout relation on A5.
  
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32,
              !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=8,
              v_row=1, v_col=8, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
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

- **Implementation checks (A2A3)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

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

- **Implementation checks (A2A3)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

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

### 4.10 Data Rearrangement Operations

#### MaskPattern

Predefined mask patterns for gather operations.

| Value | Int | Pattern |
|-------|-----|---------|
| `P0101` | 1 | Alternating 0-1-0-1 |
| `P1010` | 2 | Alternating 1-0-1-0 |
| `P0001` | 3 | 0-0-0-1 |
| `P0010` | 4 | 0-0-1-0 |
| `P0100` | 5 | 0-1-0-0 |
| `P1000` | 6 | 1-0-0-0 |
| `P1111` | 7 | All ones |

---

##### `pto.tconcat` - Concatenate Tiles (Column-wise)

**Summary:** Concatenates two source tiles along the column dimension into a destination tile.

**Semantics:**

Let \(R\) be `dst` valid rows, \(C_0\) be `src0` valid columns, and \(C_1\) be `src1` valid columns. For each row \(i\):

\[
dst[i, 0:C_0) = src0[i, 0:C_0)
\]
\[
dst[i, C_0:C_0+C_1) = src1[i, 0:C_1)
\]

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile (first segment) |
| `src1` | `pto.tile_buf` | Second tile (second segment) |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tconcat ins(<src0>, <src1> : <src0_type>, <src1_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
- `src0`, `src1`, and `dst` must have the same element type, and must be one of : `i8/i16/i32/f16/f32/bf16`
- TileType of src and dst tiles must be `loc=vec`
- The total concatenated valid columns must fit in `dst` capacity:
  - `src0.valid_col + src1.valid_col <= dst.cols` (checked when these values are statically known).
- `dst valid row = src0/src1 valid row`, 
- **Implementation checks (A5)**:
- `src0`, `src1`, and `dst` must have the same element type, and must be one of : `i8/i16/i32/f16/f32/bf16`.
- All tiles must `blayout=row_major`
- TileType of src and dst tiles must be `loc=vec`
- The total concatenated valid columns must fit in `dst` capacity:
  - `src0.valid_col + src1.valid_col <= dst.cols` (checked when these values are statically known).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tconcat ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tconcatidx` - Indexed Tile Concatenation

**Summary:** Concatenates two source tiles along the column dimension with per-row index control, where two additional index tiles specify the number of columns to copy from each source on a per-row basis.

**Semantics:**

For each row \(i\):
- Read `idx0_num = src0Idx[i, 0]` and `idx1_num = src1Idx[i, 0]` as element counts
- Copy the first `min(idx0_num, src0_valid_col, dst_valid_col)` columns from `src0` to `dst`
- Copy the first `min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0)` columns from `src1` to `dst`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile |
| `src1` | `pto.tile_buf` | Second source tile |
| `src0Idx` | `pto.tile_buf` | Per-row index for `src0` (column count to copy) |
| `src1Idx` | `pto.tile_buf` | Per-row index for `src1` (column count to copy) |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tconcatidx ins(<src0>, <src1>, <src0Idx>, <src1Idx> :
                   <src0_type>, <src1_type>, <src0Idx_type>, <src1Idx_type>)
               outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `dst`, `src0`, `src1` must have the same data type, and must be one of: `i8/i16/i32/f16/f32/bf16`
- `src0Idx`, `src1Idx` must have the same index type, and must be one of: `i8/i16/i32`
- All tiles must use `loc=vec` and row-major layout
- `validRow(src0) == validRow(src1) == validRow(dst)`
- `validRow(src0Idx) == validRow(src1Idx) == validRow(dst)`
- Index tile must have `valid_col >= 1`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tconcatidx ins(%src0, %src1, %idx0, %idx1 :
  !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
  outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tgather` - Gather/Select Elements

**Summary:** Gathers elements from a source tile using one of three PTO-ISA-compatible forms:

- index gather: `src + indices[+ tmp] -> dst`
- compare gather: `src + kValue + tmp -> dst + cdst`
- mask-pattern gather: `src + maskPattern -> dst`

**Semantics:**

```
Index form:
    dst[i, j] = src[indices[i, j]]

Compare form:
    dst stores gathered indices that satisfy the scalar compare
    cdst stores the per-row selected-count / compact-count result

Mask form:
    dst[i, j] = src[...] according to mask pattern
```

**Arguments / Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Main destination tile |
| `cdst` | `Optional<pto.tile_buf>` | Secondary destination tile used only by compare form |
| `indices` | `Optional<pto.tile_buf>` | Index tile used only by index form |
| `tmp` | `Optional<pto.tile_buf>` | Temporary tile used by compare form; optionally used by index form on A2/A3 (required), A5 (optional) |
| `kValue` | `Optional<scalar>` | Scalar compare value used only by compare form |
| `maskPattern` | `Optional<MaskPatternAttr>` | Mask pattern used only by mask form |
| `cmpMode` | `Optional<CmpModeAttr>` | Compare mode used only by compare form; defaults to `eq` when omitted |
| `offset` | `Optional<i32>` | Compare-form gather base index offset; defaults to `0` when omitted |

**Results:** None. Writes into DPS destinations.

- Index form writes `dst`
- Compare form writes both `dst` and `cdst`
- Mask form writes `dst`

Note: the compare-form C++ API is spelled as `TGATHER(dst, src, k_value, cdst, tmp)`, but in PTO IR the writable operands are grouped under `outs(...)`, so `cdst` appears in `outs(...)` rather than `ins(...)`.

**Assembly Format:**

```mlir
// index + tmp
pto.tgather ins(%src, %indices, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

// compare + tmp
pto.tgather ins(%src, %kValue, %tmp : !pto.tile_buf<...>, <scalar_type>, !pto.tile_buf<...>)
           outs(%dst, %cdst : !pto.tile_buf<...>, !pto.tile_buf<...>)
           {cmpMode = #pto<cmp eq|gt>, offset = <i32>}

// mask pattern
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>} : !pto.tile_buf<...>, "row")
           outs(%dst : !pto.tile_buf<...>)
```

**Constraints & Verification:**

- Exactly one of the following forms must be used:
  - index form: `indices` (A5: `tmp` is optional; A2/A3: `tmp` is required)
  - compare form: `kValue`, `tmp`, and `cdst`
  - mask form: `maskPattern`
- **Index gather: implementation checks (A2/A3)**:
  - `src` and `dst` element types must match and be one of `i16/i32/f16/f32`.
  - `indices` element type must be `i32`.
  - `tmp` is required; `tmp` element type must match `indices`.
  - `dst`, `indices`, and `tmp` must use row-major layout.
  - `dst` and `indices` must have the same valid shape.
  - `indices` and `tmp` must have the same valid shape; their allocated row
    widths may differ and are used as independent physical strides.
- **Index gather: implementation checks (A5)**:
  - `src` and `dst` element types must match and be one of `i8/i16/i32/f16/f32`, or a target-supported fp8 type (`f8E4M3*`/`f8E5M2*`).
  - `indices` element type must be `i16` or `i32`.
  - `tmp` is optional; PTO IR does not impose an extra tmp shape or valid-shape relation in the A5 index form.
- **Compare gather: implementation checks (A2/A3)**:
  - `dst` element type must be `i32`.
  - `src` element type must be `f16/f32`, or `i32` when `cmpMode=eq`.
  - `cmpMode` must be `eq` or `gt`.
  - `src`, `dst`, `cdst`, and `tmp` must all be `loc=vec` row-major tiles.
- **Compare gather: implementation checks (A5)**:
  - `dst` element type must be `i32`.
  - `src` element type must be one of `i16/i32/f16/f32`.
  - `cmpMode` must be `eq` or `gt`.
  - `src`, `dst`, `cdst`, and `tmp` must all be `loc=vec` row-major tiles.
- **Mask-pattern gather: implementation checks (A2/A3)**:
  - Source element size must be `2` or `4` bytes.
  - `src` and `dst` must both use `loc=vec` and `blayout=row_major`.
  - `src` and `dst` element sizes must match.
- **Mask-pattern gather: implementation checks (A5)**:
  - Source element size must be `1`, `2`, or `4` bytes.
  - `src` and `dst` must both use `loc=vec` and `blayout=row_major`.
  - `src`/`dst` element type must be `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`, `!pto.hif8`, or a target-supported fp8 type (`f8E4M3*`/`f8E5M2*`).
  - `src` and `dst` element sizes must match.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Examples:**

```mlir
// index + tmp
pto.tgather ins(%src, %idx, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

// compare + tmp
pto.tgather ins(%src, %k, %tmp : !pto.tile_buf<...>, f16, !pto.tile_buf<...>)
           outs(%dst, %cdst : !pto.tile_buf<...>, !pto.tile_buf<...>)
           {offset = 7 : i32}

// mask pattern
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P1111>} : !pto.tile_buf<...>, "row")
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tgatherb` - Gather 32-Byte Blocks

**Summary:** Gathers 32-byte source blocks using byte addresses.

**Semantics:**

Each `offsets[i, k]` is a 32-byte-aligned byte address relative to the source
UB base. It selects one complete 32-byte source block, which is copied to the
corresponding output block. This is not a scalar per-element gather; use the
index form of `pto.tgather` for arbitrary element indices.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `offsets` | `pto.tile_buf` | Compact 32-bit source block-address tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2/A3)**
  - `src` and `dst` must have the same valid shape.
  - `dst` and `offsets` must use row-major layout (`blayout=row_major`).
  - `dst` element size must be `2` or `4` bytes.
  - `offsets` must use a 32-bit integer element type.
  - `offsets.v_row` must equal `dst.v_row`.
  - For destination element size `E`, each row needs
    `ceil(dst.v_col / (32 / E))` block addresses. `offsets.v_col` is this
    count rounded up to a multiple of eight because each hardware repeat
    consumes eight addresses.
  - Allocated `dst.cols` and `offsets.cols` are their independent physical row
    strides and may exceed their valid widths.
- **Implementation checks (A5)**
  - Destination element size must be `1`, `2`, or `4` bytes.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tgatherb ins(%src, %offs : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tscatter` - Tile Scatter

**Summary:** Scatters elements from a source tile into a destination tile. Two modes are supported: **index mode** (per-element column offsets) and **mask-pattern mode** (regular spacing controlled by a mask pattern and axis).

**Semantics:**

Index mode:

```
dst[i, indexes[i, j]] = src[i, j]
```

Mask-pattern mode (axis = `"row"`):

```
dst[i, j * times + offset] = src[i, j]   // elements interleaved with zeros along columns
```

Mask-pattern mode (axis = `"col"`):

```
dst[i * stride + start, j] = src[i, j]   // elements placed at strided row positions
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile |
| `indexes` | `pto.tile_buf` (optional) | Index tile (index mode only) |
| `axis` | `str` attr (optional) | Scatter direction: `"row"` or `"col"` (mask-pattern mode only) |
| `maskPattern` | `pto.mask_pattern` attr (optional) | Spacing pattern: `P0101`, `P1010`, `P0001`, `P0010`, `P0100`, `P1000`, `P1111` (mask-pattern mode only) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Common**
  - Exactly one of `indexes` operand or `maskPattern` attribute must be provided.
  - `axis` attribute must not be provided together with `indexes`.
- **Index mode**
  - `dst`, `src`, and `indexes` must all use `loc=vec`.
  - `dst`/`src` element type must be one of: `i32`, `i16`, `i8`, `f16`, `f32`, `bf16`.
  - `indexes` element type must be one of: `i16`, `i32`.
  - No bounds checks are enforced on `indexes` values.
  - Valid bounds: `dst.valid_shape[i] <= dst.shape[i]`, `src.valid_shape[i] <= src.shape[i]`, and `indexes.valid_shape[i] <= indexes.shape[i]` for each dimension `i`.
  - `dst` and `src` must have the same element type.
  - `dst.valid_shape[d] >= src.valid_shape[d]` for each dimension `d`.
  - When `dst` element size is 4 bytes, `indexes` element size must also be 4 bytes.
  - When `dst` element size is 2 bytes, `indexes` element size must also be 2 bytes.
  - When `dst` element size is 1 byte, `indexes` element size must be 2 bytes.
- **Mask-pattern mode**
  - `dst` and `src` must use `loc=vec` and `row_major` blayout.
  - `dst`/`src` element type must be one of: `i32`, `i16`, `i8`, `f16`, `f32`, `bf16`.
  - `dst` and `src` must have the same element type.
  - `axis` must be `"row"` or `"col"`.
  - When `axis = "row"`: `dst.valid_rows == src.valid_rows` and `dst.valid_cols == src.valid_cols * times`.
  - When `axis = "col"`: `dst.valid_cols == src.valid_cols` and `dst.valid_rows == src.valid_rows * times`.
  - `times` is the mask expansion factor: `P1111`→1, `P0101`/`P1010`→2, `P0001`/`P0010`/`P0100`/`P1000`→4.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Index mode example:**

```mlir
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

**Mask-pattern mode example:**

```mlir
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} : !pto.tile_buf<...>, "row")
            outs(%dst : !pto.tile_buf<...>)
```

`maskPattern` form lowers to the `pto-isa` `TSCATTER<MaskPattern, ScatterType>` overload on
backends that provide it, including A2/A3 and A5.

---

##### `pto.mgather` - Gather-Load from Global Memory

**Summary:** Loads elements from a global table into a VEC tile using per-element indices. Requires an explicit `coalesce` attribute and supports an optional `gatherOob` attribute; these lower to the corresponding `MGATHER<...>` template overload.

**Semantics:**

```
row mode:  dst[r, j] = mem[idx[r], j]
elem mode: dst[i, j] = mem[idx[i, j]]
```

**Arguments:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `mem` | `!pto.partition_tensor_view<...>` | `NA` | Global source table |
| `idx` | `pto.tile_buf` | `NA` | Index tile |
| `dst` | `pto.tile_buf` | `NA` | Destination VEC tile |
| `coalesce` | `#pto<coalesce ...>` | required | Explicit coalesce mode (`row` / `elem`) |
| `gatherOob` | `#pto<gather_oob ...>` | `undefined` | Out-of-bounds mode (`undefined/clamp/wrap/zero`) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Types (data and indices)**  
  - `mem` and `dst` must have the **same element type**. Supported element types: `i8`/`i16`/`i32`/`f16`/`bf16`/`f32`. On **A5** targets, `float8_e4m3` / `float8_e5m2` family and `!pto.hif8` element types are also supported.
  - `idx` element type must be signless `i32`.

- **Tile / memory roles**  
  - `dst` must be `loc=vec`, `blayout=row_major`, `slayout=none_box`.
  - `idx` must be `loc=vec`, `slayout=none_box`. `row_major` and `col_major` are both accepted for row mode.
  - `mem` must denote a GlobalTensor in GM memory.
  - `mem` must use `ND` layout when layout can be inferred.

- **Shape**  
  - Element mode: `idx valid_shape == dst valid_shape`.
  - Row mode: `idx valid_shape` may be `[1, dst.valid_row]` or `[dst.valid_row, 1]`.
  - The `[1, R]` row-mode variant uses `row_major`; the `[R, 1]` row-mode variant uses `col_major`.

- **Out-of-bounds mode**
  - Default `gatherOob = undefined` lowers to the default `MGATHER(dst, mem, idx)` overload.
  - Non-default `gatherOob` values lower to `MGATHER<Coalesce, GatherOOB::...>(dst, mem, idx)`.
- **Coalesce mode**
  - `coalesce` must be specified explicitly; PTOAS does not infer Row or Elem from the `idx` tile shape/layout.
  - The `idx` tile shape/layout must match the specified mode.
  - `coalesce = #pto<coalesce row>` lowers to `MGATHER<pto::Coalesce::Row, ...>`.
  - `coalesce = #pto<coalesce elem>` lowers to `MGATHER<pto::Coalesce::Elem, ...>`.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`)

**Basic Example:**

```mlir
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
           {coalesce = #pto<coalesce row>}

pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
           {coalesce = #pto<coalesce elem>}

pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
           {coalesce = #pto<coalesce row>, gatherOob = #pto<gather_oob zero>}
```

**GM → L1 (cube `loc=mat`) destination:**

The destination may also be an **L1 / cube `loc=mat` tile in NZ layout**, so a
gathered table can be fed straight into a matmul without a UB round-trip. This
mirrors the pto-isa `MGATHER` GM → L1 overloads and is selected automatically
when `dst` is a `loc=mat` tile.

- **`dst`** — `loc=mat`, `blayout=col_major`, `slayout=row_major`, `fractal=512`
  (NZ). Padded `cols` must be a multiple of `C0 = 32 / sizeof(elem)` and padded
  `rows` a multiple of `16` (`FRACTAL_NZ_ROW`).
- **`idx`** — supplied as a GM `partition_tensor_view`
  of `i32`, **not** a UB tile: on A5 the cube core that issues the L1 transfer
  cannot read AIV's UB.
- **`coalesce`** — must be **explicit** (`row` or `elem`); there is no UB index
  tile to infer the mode from.
- **`scratch`** — `coalesce=elem` requires a contiguous **GM `scratch`** operand
  (`>= dst.rows * dst.cols` elements, element type matching `dst`) used to stage
  the discrete elements into NZ layout before the bulk GM → L1 copy. `coalesce=row`
  takes no scratch.
- Executes on the GM → L1 DMA pipeline (`PIPE_MTE2`) on every arch, like `pto.tload`.
- Lowering: `row` → `MGATHER<pto::Coalesce::Row[, GatherOOB]>(dst, table, idx)`;
  `elem` → `MGATHER<pto::Coalesce::Elem[, GatherOOB]>(dst, table, idx, scratch)`.

```mlir
// Row: no scratch, idx is a GM [1, R] tensor.
pto.mgather ins(%mem, %idx : !pto.partition_tensor_view<1x1x1x64x32xf16>,
                             !pto.partition_tensor_view<1x1x1x1x32xi32>)
           outs(%dst : !pto.tile_buf<loc=mat, dtype=f16, rows=32, cols=32, v_row=32,
                                     v_col=32, blayout=col_major, slayout=row_major,
                                     fractal=512, pad=0>)
           {coalesce = #pto<coalesce row>}

// Elem: GM scratch workspace staged in NZ layout before the bulk copy.
pto.mgather ins(%mem, %idx, %scratch
                : !pto.partition_tensor_view<1x1x1x64x32xf16>,
                  !pto.partition_tensor_view<1x1x1x32x32xi32>,
                  !pto.partition_tensor_view<1x1x1x32x32xf16>)
           outs(%dst : !pto.tile_buf<loc=mat, dtype=f16, rows=32, cols=32, v_row=32,
                                     v_col=32, blayout=col_major, slayout=row_major,
                                     fractal=512, pad=0>)
           {coalesce = #pto<coalesce elem>}
```

---

##### `pto.mscatter` - Scatter-Store to Global Memory

**Summary:** Stores elements from a VEC tile into a global table using per-element indices. Supports optional `coalesce`, atomic, out-of-bounds, and A5-only conflict-mode attributes that lower to the corresponding `MSCATTER<...>` template overload family.

**Semantics:**

```
row mode (default): mem[idx[r], j] = src[r, j]
elem mode:          mem[idx[i, j]] = src[i, j]
```

**Arguments:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `src` | `pto.tile_buf` | `NA` | Source VEC tile |
| `idx` | `pto.tile_buf` | `NA` | Index tile |
| `mem` | `!pto.partition_tensor_view<...>` | `NA` | Global destination table |
| `coalesce` | `#pto<coalesce ...>` | inferred | Explicit coalesce mode (`row` / `elem`) |
| `scatterAtomicOp` | `#pto<scatter_atomic_op ...>` | `none` | Atomic mode (`none/add/max/min`) |
| `scatterOob` | `#pto<scatter_oob ...>` | `undefined` | Out-of-bounds mode (`undefined/skip/clamp/wrap`) |
| `scatterConflict` | `#pto<scatter_conflict ...>` | omitted | Optional A5 conflict mode (`default` / `last`) |

**Results:** None. Writes into `mem` via DPS pattern.

**Constraints & Verification:**

- **Types (data and indices)**  
  - `src` and `mem` must have the **same element type**. Supported element types: `i8`/`i16`/`i32`/`f16`/`bf16`/`f32`. On **A5** targets, `float8_e4m3` / `float8_e5m2` family and `!pto.hif8` element types are also supported.
  - `idx` element type must be signless `i32`.

- **Tile / memory roles**  
  - `src` must be `loc=vec`, `blayout=row_major`, `slayout=none_box`.
  - `idx` must be `loc=vec`, `slayout=none_box`. `row_major` and `col_major` are both accepted for row mode.
  - `mem` must denote a GlobalTensor in GM memory.
  - `mem` must use `ND` layout when layout can be inferred.

- **Shape**  
  - Element mode: `idx valid_shape == src valid_shape`.
  - Row mode: `idx valid_shape` may be `[1, src.valid_row]` or `[src.valid_row, 1]`.
  - The `[1, R]` row-mode variant uses `row_major`; the `[R, 1]` row-mode variant uses `col_major`.

- **Atomic modes**  
  - Default `scatterAtomicOp = none` lowers to the default `MSCATTER(mem, src, idx)` overload.
  - Non-default `scatterAtomicOp` values lower to `MSCATTER<Coalesce, ScatterAtomicOp::...>(mem, src, idx)`.
  - `add` requires `i32`/`f16`/`f32`.
  - `max`/`min` require signless `i32` or `f32`.

- **Out-of-bounds modes**
  - Default `scatterOob = undefined` lowers to the `MSCATTER<Coalesce, ScatterAtomicOp::...>(mem, src, idx)` form when only atomic is specified, or to the default overload when both attrs are default.
  - Non-default `scatterOob` values lower to `MSCATTER<Coalesce, ScatterAtomicOp::..., ScatterOOB::...>(mem, src, idx)`.
- **Coalesce and conflict modes**
  - If `coalesce` is omitted, PTOAS preserves the existing inference from the `idx` tile shape/layout.
  - If `coalesce` is specified, the `idx` tile shape/layout must match that mode.
  - `scatterConflict` is only meaningful on A5 and lowers by filling the full `MSCATTER<Coalesce, Atomic, Oob, Conflict>` template parameter list.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`)

**Basic Example:**

```mlir
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : !pto.partition_tensor_view<...>)

pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : !pto.partition_tensor_view<...>)
            {scatterAtomicOp = #pto<scatter_atomic_op add>}

pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : !pto.partition_tensor_view<...>)
            {scatterAtomicOp = #pto<scatter_atomic_op add>,
             scatterOob = #pto<scatter_oob skip>}

pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : !pto.partition_tensor_view<...>)
            {coalesce = #pto<coalesce elem>,
             scatterConflict = #pto<scatter_conflict last>}
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

- **Tile location must match**: `src.loc == dst.loc`.
- **Total byte size must match**: `sizeof(srcElem) * drcNumel == sizeof(dstElem) * dstNumel`.
- **No boxed/non-boxed conversion**:
  - cannot reshape between `SLayout::NoneBox` and boxed layouts.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tinsert` - Insert Sub-Tile Window

**Summary:** Inserts a source tile into a destination tile at a given row/col offset.

**Semantics:**

```
dst[i + indexRow, j + indexCol] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexRow` | `Index` | Destination row offset |
| `indexCol` | `Index` | Destination column offset |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Without `fp`, lowers to **`TINSERT(dst, src, indexRow, indexCol)`**.
- With `fp`, lowers to **`TINSERT_FP(dst, src, fp, indexRow, indexCol)`**;
  an explicit A5 `accToVecMode` selects the fp-parameterized `TINSERT` overload.
- Uses the target data-movement pipeline: `Vec -> Vec` uses `PIPE_V`, A5
  `Vec -> Mat` uses `PIPE_MTE3`, and regular `Acc -> Mat` uses `PIPE_FIX`.

**Basic Example:**

```mlir
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<...>, index, index) outs(%dst : !pto.tile_buf<...>)
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<...>, index, index fp %fp : !pto.tile_buf<scaling, ...>) outs(%dst : !pto.tile_buf<...>)
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
| `fp` | `pto.tile_buf` | Optional scaling tile (`loc=scaling`) for accumulator conversion |
| `preQuantScalar` | `i64` | Optional scalar pre-quant parameter |
| `accToVecMode` | `pto.acc_to_vec_mode` | Optional A5 acc-to-vec mode |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst` element type must match `src` element type and must be one of: `i8`, `f16`, `bf16`, `f32`.
  - `Vec -> Vec` extraction is supported for matching element types.
  - Source layout/fractal must satisfy one of the target-supported combinations: `slayout=col_major` with `blayout=row_major`, or `slayout=row_major`.
  - Runtime bounds checks:
    - `indexRow + dst.rows <= src.rows`
    - `indexCol + dst.cols <= src.cols`
  - `dst` must use `loc=left` or `loc=right` with a target-supported fractal configuration.
- **Implementation checks (A5)**
  - `dst` element type must match `src` element type and must be one of the target-supported low-precision/fp16/bf16/f32 families (`f8E4M3*`, `f8E5M2*`, `!pto.hif8`, `!pto.f4E2M1x2`, `f16`, `bf16`, `f32`, `i8`).
  - Source layout/fractal must satisfy the target-supported combinations for `left`/`right`/scaling destinations; in PTO IR terms this is expressed through the `blayout`/`slayout`/`fractal` tuple.
  - Destination supports `Mat -> Left/Right/Scale` and also supports `Vec -> Mat` for specific tile locations.

**Hardware Mapping:**

- Base forms lower to `TEXTRACT`; an `fp` form without mode lowers to
  `TEXTRACT_FP`, while an explicit A5 `accToVecMode` selects the
  fp-parameterized `TEXTRACT` overload.

**Basic Example:**

```mlir
pto.textract ins(%src, %row, %col : !pto.tile_buf<...>, index, index) outs(%dst : !pto.tile_buf<...>)
pto.textract ins(%src, %row, %col : !pto.tile_buf<...>, index, index fp %fp : !pto.tile_buf<scaling, ...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tfillpad` - Fill Padding Region

**Summary:** Unified padding operation. PTOAS infers normal, in-place, or expand lowering from physical shapes and post-PlanMemory addresses.

**Semantics:**

```
normal:   copy valid src elements, then fill dst padding
in_place: keep valid data already in shared storage, then fill dst padding
expand:   copy src into a possibly larger dst, then fill the expanded region
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (with pad config) |
| `padValue` | `#pto.pad_value<...>` (optional) | Explicit `TFILLPAD<PadValue>` template argument for `loc=mat`. When present, it must match `dst`'s tile pad configuration. |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `dst.pad` must not be `null`.
- `src` and `dst` element sizes must match, and the element size must be `1`, `2`, or `4` bytes.
- If source and destination physical shapes differ, every destination dimension must be at least the corresponding source dimension; PTOAS then infers expand lowering.
- If physical shapes are equal, exact starting-address equality after PlanMemory selects in-place lowering; otherwise PTOAS selects normal lowering.
- `valid_shape` does not participate in expand inference.
- In-place and expand lowering require both operands to use `loc=vec`.
- If `padValue` is present, `dst` must be `loc=mat`, and `padValue` must equal the tile type's `pad`.
- MAT always uses Normal lowering, including when source and destination share the same starting address.
- For `loc=mat`, `src` and `dst` must be lowerable to the same `TFILLPAD` tile specialization, i.e. `validShape` and `pad` must be identical.

**Hardware Mapping:**

- VEC forms execute on the **Vector pipeline** (`PIPE_V`).
- The normal homogeneous MAT form executes on `PIPE_MTE1`.
- Normal lowers to `TFILLPAD(dst, src)`; compiler-inferred in-place and expand forms lower to `TFILLPAD<pto::TFillPadMode::InPlace/Expand>(dst, src)`.

**Basic Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)

pto.tfillpad ins(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
             outs(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)

pto.tfillpad ins(%src_small : !pto.tile_buf<vec, 16x16xf32>)
             outs(%dst_large : !pto.tile_buf<vec, 32x32xf32, pad=1>)
```

---

### 4.11 Sorting Operations

##### `pto.tsort32` - Sort Fixed 32-Element Blocks

**Summary:** Sorts fixed-size 32-element blocks using an explicit input index tile.

**Semantics:**

```
dst = sort(src, idx)
idx = permutation indices for the sort
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Input value tile |
| `idx` | `pto.tile_buf` | Input index tile permuted together with `src` |
| `tmp` | `Optional<pto.tile_buf>` | Optional scratch tile for the tmp-taking DPS overload |
| `dst` | `pto.tile_buf` | Output tile storing sorted value-index pairs |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsort32 ins(<src>, <idx>[, <tmp>] : <src_type>, <idx_type>[, <tmp_type>])
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `dst` element type must be `f16` or `f32`.
- `src` element type must match `dst` element type.
- `idx` element type must be `u32`.
- `src`, `dst`, and `idx` must all use `loc=vec` and `blayout=row_major`.
- If the 4-operand form is used, `tmp` only needs to be a `loc=vec` tile.
- PTO IR does not impose a same-shape or rounded-tail tmp formula in the verifier.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

# Optional scratch form:
pto.tsort32 ins(%src, %idx, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
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
| `src` / `src0..src3` | PTO shaped-like type | Input tile(s); format2 supports 2 to 4 sources |
| `blockLen` | `AnyInteger` operand | Block length for format1 |
| `dst` | PTO shaped-like type | Output tile |
| `tmp` | PTO shaped-like type | Temporary workspace input tile for format2 |
| `excuted` | `vector<4xi16>` | Output vector written by format2 |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmrgsort ins(<src>, <blockLen> : <src_type>, <blockLen_type>)
             outs(<dst> : <dst_type>)

pto.tmrgsort ins(<src0>, <src1>[, <src2>[, <src3>]], <tmp> {exhausted = <bool>} :
                 <src0_type>, <src1_type>[, <src2_type>[, <src3_type>]], <tmp_type>)
             outs(<dst>, <excuted> : <dst_type>, vector<4xi16>)
```


**Constraints & Verification:**

- **Format1 (`ins(src, blockLen) outs(dst)`)**
  - `src` and `dst` must have the same element type.
  - The element type must be `f16` or `f32`.
  - `src` and `dst` must both be rank-2 single-row tiles (`rows == 1` when statically known).
- **Single-list variant (`TMRGSORT(dst, src, blockLen)`)**:
  - `blockLen` must be a multiple of 64 (as checked by the implementation).
  - PTO IR currently does not enforce the `src valid column` multiple or `repeatTimes` range in the verifier.
- **Multi-list variants**:
  - Accepts 2-way, 3-way, and 4-way merge forms.
  - `dst`, `tmp`, and every `src` must have the same element type.
  - The shared element type must be `f16` or `f32`.
  - `dst` and `tmp` must both be rank-2 single-row tiles (`rows == 1` when statically known).
  - Every `src` must also be a rank-2 single-row tile.
  - `tmp.cols >= dst.cols`.
  - `tmp.cols` must be at least the sum of the sources' effective valid column
    extents. For a subview, this uses the subview window's valid columns, not
    the capacity of its backing tile.
  - `excuted` must be `vector<4xi16>`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
// format1
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, i32)
             outs(%dst : !pto.tile_buf<...>)

// format2
pto.tmrgsort ins(%src0, %src1, %tmp2 {exhausted = false} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst2, %excuted :
                 !pto.tile_buf<...>, vector<4xi16>)

pto.tmrgsort ins(%src0, %src1, %src2, %tmp3 {exhausted = true} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>,
                 !pto.tile_buf<...>)
             outs(%dst3, %excuted :
                 !pto.tile_buf<...>, vector<4xi16>)

pto.tmrgsort ins(%src0, %src1, %src2, %src3, %tmp4 {exhausted = false} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>,
                 !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst4, %excuted :
                 !pto.tile_buf<...>, vector<4xi16>)
```

---

### 4.12 Type Conversion

#### RoundMode

Rounding modes for type conversion (`pto.tcvt`) operations.

| Value | Int | Description |
|-------|-----|-------------|
| `NONE` | 0 | No rounding |
| `RINT` | 1 | Round to nearest integer |
| `ROUND` | 2 | Round f16 away from zero |
| `FLOOR` | 3 | Round toward negative infinity |
| `CEIL` | 4 | Round toward positive infinity |
| `TRUNC` | 5 | Truncate toward zero |
| `ODD` | 6 | Round to odd |
| `CAST_RINT` | 7 | Cast with round-to-nearest (default) |

**Attribute syntax:** `#pto<round_mode FLOOR>`

---

##### `pto.tcvt` - Elementwise Type Conversion

**Summary:** Converts each element to a new type with a specified rounding mode and optional saturation mode.

**Semantics:**

```
dst[i, j] = saturate(cast(src[i, j], rmode), satmode)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (different element type) |
| `rmode` | `RoundModeAttr` (default: `CAST_RINT`) | Rounding mode |
| `satmode` | `SaturationModeAttr` (default: `OFF`) | Saturation mode |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- `satmode = ON` requests destination-range clamping after rounding; `OFF` preserves the target's non-saturating conversion path.
- **A2/A3 and A5 notes:**
  - A2/A3 reject all low-precision `tcvt` operands.
  - A5 only accepts the following low-precision pairs: 
    `f32 -> f8E4M3*`, 
    `f32 -> f8E5M2*`, 
    `f32 -> !pto.hif8`, 
    `f16 -> !pto.hif8`, 
    `bf16 -> !pto.f4E1M2x2`, 
    `bf16 -> !pto.f4E2M1x2`, 
    `!pto.f4E1M2x2 -> bf16`, 
    `!pto.f4E2M1x2 -> bf16`, 
    `f8E4M3* -> f32`, 
    `f8E5M2* -> f32`, 
    `!pto.hif8 -> f32`.
  - Non-low-precision pairs continue to use the existing target-defined behavior.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.13 Integer Sequence Generation Operations

##### `pto.tci` - Contiguous Integer Sequence

**Summary:** Generates a contiguous integer sequence into a destination tile.

**Semantics:**

```
dst[i, j] = S + linear_index(i, j)   // or descending if requested
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `S` | `Integer` | Starting value |
| `tmp` | `pto.tile_buf` (optional) | Optional scratch tile forwarded to the `pto-isa` tmp-buffer overload |
| `dst` | `pto.tile_buf` | Destination tile |
| `descending` | `BoolAttr` (default: false) | Generate descending sequence |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2/A3/A5)**
  - Tile element type must be exactly the same type as the `S`.
  - `dst/scalar` element types must be identical, and must be one of: `i16`, `ui16`, `i32`, `ui32`.
  - `dst.cols != 1`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tci ins(%start : i32) outs(%dst : !pto.tile_buf<...>)
pto.tci ins(%start, %tmp : i32, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### 4.14 Scalar Element Access

##### `pto.tgetval` - Read Single Element

**Summary:** Reads a single element from a tile at a linear offset.

**Semantics:**

```
result = src[offset]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `offset` | `Index` | Linear element offset |

**Results:** Scalar value (`ScalarType`)

**Constraints & Verification:**

- `src` must be a `!pto.tile_buf` or a `memref`.
- `src` must use `loc=vec`.
- If `src` uses `loc=mat`, the current verifier rejects it explicitly because scalar reads from mat tiles are not supported.
- Result type must exactly match the element type of `src`.

**Hardware Mapping:**

- Executes on the **Scalar pipeline** (`PIPE_S`) when operating on tile_buf

**Basic Example:**

```mlir
%val = pto.tgetval ins(%src, %off : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : f16
```

---

##### `pto.tsetval` - Write Single Element

**Summary:** Writes a scalar value into a tile at a linear offset.

**Semantics:**

```
dst[offset] = val
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tile_buf` | Destination tile |
| `offset` | `Index` | Linear element offset |
| `val` | `ScalarType` (signless integer / float) | Scalar value to write |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

If `dst` is a shaped type, `val` must have exactly the same type as `dst`'s element type.
- The current verifier does not add extra checks on `offset`.

**Hardware Mapping:**

- Executes on the **Scalar pipeline** (`PIPE_S`) when operating on tile_buf

**Basic Example:**

```mlir
pto.tsetval ins(%off, %val : index, f16) outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.15 MX Quantized Operations

##### `pto.tget_scale_addr` - Bind Scaling Tile View

**Summary:** Binds a scaling tile to the scale-address associated with a source tile. No elementwise computation or data movement is performed.

**Semantics:**

```
dst = scale_view_of(src)
```

`dst` becomes a scaling-tile view compatible with `src`, so it can be consumed by later MX / scaling-aware ops.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile that owns or references scale storage |
| `dst` | `pto.tile_buf` | Destination scaling tile view |

**Results:** None. Initializes or rebinds `dst` via DPS pattern.

**Assembly Format:**

```
pto.tget_scale_addr ins(<src> : <src_type>)
                    outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Not supported.
- **Implementation checks (A5)**
  - `src` must be a valid `pto.tile_buf`.
  - `dst` must be a valid `pto.tile_buf` in `loc=scaling`.
  - `dst` must have the same rank, shape, and valid shape as `src`.
  - `dst` must satisfy the target-specific scaling-tile compatibility rules for `src`.

**Hardware Mapping:**

- Executes on the **Scalar pipeline** (`PIPE_S`)

**Basic Example:**

```mlir
pto.tget_scale_addr ins(%src : !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
                        v_row=1, v_col=128, blayout=col_major, slayout=row_major,
                        fractal=512, pad=0>)
                    outs(%scale : !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=128,
                        v_row=1, v_col=128, blayout=row_major, slayout=row_major,
                        fractal=512, pad=0>)
```

---

##### `pto.tquant` - Quantize Tile with Scaling Tile

**Summary:** Quantizes `f32` source tile elements into a lower-precision integer format using a scaling (`fp`) tile. The quantization mode is controlled by the `quant_type` attribute.

**Semantics:**

```
dst[i, j] = Quantize(src[i, j]; fp, quant_type)
```

- `INT8_SYM`: symmetric quantization; `dst` element type must be `i8`; no `offset` operand is allowed.
- `INT8_ASYM`: asymmetric quantization; `dst` element type must be `ui8`; an `offset` operand is required.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile (`f32`) |
| `fp` | `pto.tile_buf` | Scaling parameter tile (`f32`) |
| `offset` | `pto.tile_buf` | Optional asymmetric offset tile (`f32`, required for `INT8_ASYM`) |
| `dst` | `pto.tile_buf` | Destination tile (`i8` for SYM, `ui8` for ASYM) |
| `tmp` | `pto.tile_buf` | Optional scratch tile. A2/A3 uses it for row-broadcast and fp32-to-s32 conversion scratch. When omitted, PTOAS preserves the no-tmp form and selects the 4-argument backend overload. A5 accepts it as a placeholder but does not require it. |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `quant_type` | `#pto.quant_type` | `INT8_SYM` or `INT8_ASYM` |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `src` element type must be `f32`.
- `src` and `dst` must have the same valid shape.
- `dst` element type must be `i8` (`INT8_SYM`) or `ui8` (`INT8_ASYM`).
- `pto.tquant` only models the plain INT8 quantization forms. MX quantization uses `pto.tquant.mx`.
- A2/A3: `src`, `dst`, and explicit `tmp` must use row-major layout.
- A2/A3: `fp` and `offset` must be per-row broadcast tiles: non-row-major layout, `valid_shape[0] == dst.valid_shape[0]`, and `valid_shape[1] == 1`.
- A2/A3: explicit `tmp` must have the same element type, shape, and valid shape as `src`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tquant ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
           {quant_type = #pto<quant_type INT8_SYM>}

pto.tquant ins(%src, %fp, %offset : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>)
           {quant_type = #pto<quant_type INT8_ASYM>}
```

---

##### `pto.tquant.mx` - MX Quantize Tile

**Summary:** A5-only grouped MX quantization. It groups source values along
axis 0 or axis 1, quantizes them to MXFP8 or packed MXFP4 E2M1, and
materializes the shared exponent, per-group maximum, and reciprocal scaling
outputs.

**Semantics:**

```
groupSize = 32
for each group g selected by grpAxis:
    max[g] = max(abs(src[x])) for x in g
    exp[g], scaling[g] = SharedScale(max[g], quant_type, quantScaleAlg)
    dst[x] = Quantize(src[x] * scaling[g], quant_type)
```

- `MXFP8`: source elements are `f32`, `f16`, or `bf16`; destination elements are `i8/ui8`.
- `MXFP4_E2M1`: source elements are `f16` or `bf16`; destination elements are `!pto.f4E2M1x2`.
- `grpAxis=axis1` groups consecutive columns in each row and is the default.
- `grpAxis=axis0` groups consecutive rows in each column.
- `quantScaleAlg=ocp` and `quantScaleAlg=nv` select the OCP and NV shared-scale rules.
- `interleave=true` changes only the axis0 exponent layout. `max` and
  `scaling` retain the ordinary grouped shape.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source vec tile |
| `dst` | `pto.tile_buf` | Quantized output tile |
| `exp` | `pto.tile_buf` | Exponent output tile |
| `max` | `pto.tile_buf` | Per-group max output tile |
| `scaling` | `pto.tile_buf` | Scaling output tile |
| `exp_zz` | `pto.tile_buf` | Optional legacy fused exponent-layout output. This form is deprecated; use a separate X-to-ZZ `pto.tmov`. |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `quant_type` | `#pto.quant_type` | `MXFP8` or `MXFP4_E2M1` |
| `quantScaleAlg` | `#pto.quant_scale_alg` | `ocp` (default) or `nv` |
| `grpAxis` | `#pto.mx_group_axis` | `axis1` (default, group columns) or `axis0` (group rows) |
| `interleave` | `bool` | `false` by default. When true, requires `axis0` and interleaves the exponent output. |
| `storeMode` | `#pto.vec_store_mode` | Legacy fused `exp_zz` store mode. Deprecated together with `exp_zz`. |

**Results:** None. Writes into `dst`, `exp`, `max`, and `scaling` via DPS pattern.
The optional legacy fused form also writes `exp_zz`.

**Constraints & Verification:**

- Supported only on A5 targets.
- `src`, `dst`, `exp`, `max`, and `scaling` must all be rank-2 vec tiles
  with static valid and physical shapes.
- `scaling` and `max` have the same element type as `src`.
- `exp` uses `i8` or `ui8`. The quantized destination uses `i8/ui8` for
  MXFP8 or packed `!pto.f4E2M1x2` for MXFP4.
- The physical source stride determines several output strides. Declaring only
  enough storage for each valid rectangle is insufficient; the physical rules
  below are part of the operation contract.

Let `M=src.valid_rows`, `N=src.valid_cols`, `Ps=src.cols`, and
`G=M*N/32`. For MXFP8, `pack=1`; for MXFP4, `pack=2` and `Ps` must be even.

**Valid shapes:**

| Form | Required valid shapes |
|------|-----------------------|
| `axis0`, `interleave=false` | `M % 32 == 0`; `exp/max/scaling = [M/32, N]` |
| `axis0`, `interleave=true` | `M % 64 == 0`; `exp = [M/64, 2N]`; `max/scaling = [M/32, N]` |
| `axis1`, canonical | `N % 32 == 0`; `exp/max/scaling = [M, N/32]` |
| `axis1`, legacy flat | `N % 32 == 0`; `exp/max/scaling = [1, G]`; source must be tight (`Ps == N`) |

`dst.valid_shape` is `[M, N]` for MXFP8 and `[M, N/2]` packed bytes for
MXFP4. For axis1, physical row count `1` on `exp` selects the legacy-flat
branch; a legacy-flat valid shape with more than one physical row is invalid.

**Physical stride and capacity:**

| Form | Required physical contract |
|------|----------------------------|
| axis0 MXFP8 `dst` | Ordinary 2-D destination footprint |
| axis0 MXFP4 `dst` | `dst.cols == Ps/2`; capacity at least `(M-1)*(Ps/2) + N/2` packed bytes |
| axis0 `max/scaling` | physical columns equal `Ps`; capacity at least `(M/32)*Ps` elements |
| axis0 non-interleaved `exp` | physical columns equal `Ps`; capacity at least `(M/32)*Ps` elements |
| axis0 interleaved `exp` | exact physical shape `[src.rows/64, alignTo(2*Ps, 32)]`; `src.rows` is a multiple of 64 |
| axis1 MXFP8 `dst` | `dst.cols == Ps`; capacity at least `(M-1)*Ps + N` bytes |
| axis1 MXFP4 `dst` | `dst.cols == Ps/2`; capacity at least `(M-1)*(Ps/2) + N/2` packed bytes |
| axis1 `max` | valid elements form a compact allocation prefix; capacity at least `G` elements |
| axis1 canonical `exp` | physical rows cover `M`, physical columns cover `N/32`, and capacity covers `(M-1)*exp.cols + N/32` elements |
| axis1 canonical `scaling` | compact prefix and capacity at least `G` elements |
| axis1 flat `exp` | physical rows equal `1`; capacity at least `G` elements |

Axis1 flat scaling uses full-vector stores and therefore has byte-capacity
requirements beyond its valid shape:

| Source/algorithm | Required flat scaling capacity |
|------------------|--------------------------------|
| f32, non-unrolled | `alignTo(G, 64) * 4` bytes, at least 256 bytes |
| f32, unrolled | `alignTo(2*G, 64) * 4` bytes, at least 512 bytes |
| f16/bf16, OCP | `alignTo(G, 128) * 2` bytes, at least 256 bytes |
| f16/bf16, NV | Unsupported on the pinned A5 revisions |

The f32 flat form is unrolled when all of these hold:

```text
src.rows * src.cols > 1024
(src.rows * src.cols) % 256 == 0
(src.valid_rows * src.cols) % 256 == 0
```

**Pinned A5 support boundary:**

- axis1 canonical 2-D quantization with f16/bf16 source is rejected;
- axis0 FP16 MXFP4 is rejected when the packed source stride `Ps/2` is not a
  multiple of 32 bytes;
- axis0 FP32 MXFP8 with `interleave=true` is rejected;
- padded f16/bf16 source is rejected when its final vector-aligned padding
  store would be incomplete;
- axis1 legacy-flat f16/bf16 with `quantScaleAlg=nv` is rejected.

For f16/bf16 source with `src.valid_cols < src.cols`, the operation may zero
row padding in place, so the source has `Read+Write` effects. Tight B16 and all
f32 sources remain `Read`. The source byte range must not overlap `dst`, `exp`,
`max`, `scaling`, or optional `exp_zz`, including through views.

The legacy `exp_zz + storeMode` form is retained for compatibility but is
deprecated and restricted to `grpAxis=axis1`. New code should emit the ordinary
four-output `pto.tquant.mx` followed by X-to-ZZ `pto.tmov`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Ordinary four-output forms map uniformly to
  `TQUANT<grp_axis, MxQuantAlg[, interleave]>`.
- `quant_type` and `quantScaleAlg` select one of
  `OcpMxFp8E4M3`, `NvMxFp8E4M3`, `OcpMxFp4E2M1`, or `NvMxFp4E2M1`.

**Basic Example:**

```mlir
pto.tquant.mx ins(%src : !pto.tile_buf<...>)
              outs(%dst, %exp, %max, %scaling : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                  !pto.tile_buf<...>, !pto.tile_buf<...>)
              {quant_type = #pto<quant_type MXFP8>}

// DN grouping: src [64,64], grouped auxiliaries [2,64].
pto.tquant.mx ins(%src_dn : !pto.tile_buf<vec, 64x64xf32>)
              outs(%dst_dn, %exp_dn, %max_dn, %scaling_dn :
                    !pto.tile_buf<vec, 64x64xi8>,
                    !pto.tile_buf<vec, 2x64xui8>,
                    !pto.tile_buf<vec, 2x64xf32>,
                    !pto.tile_buf<vec, 2x64xf32>)
              {quant_type = #pto<quant_type MXFP8>,
               grpAxis = #pto<mx_group_axis axis0>}
```

---

### 4.16 Synchronization Operations

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

##### `pto.barrier_sync`

**Summary:** High-level barrier that specifies a `SyncOpType` instead of a concrete PIPE. The lowering pass maps the op type to the corresponding hardware pipe and emits `pto.barrier`.

**Semantics:**

```
barrier_sync(op_type)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `SyncOpTypeAttr` | High-level sync endpoint (e.g. `TLOAD`, `TSTORE_ACC`, `TMATMUL`, `TVEC`) |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond type consistency

**Hardware Mapping:**

- Pipeline barrier for the specified operation

**Basic Example:**

```mlir
pto.barrier_sync [<TMATMUL>]
pto.barrier_sync [<TVEC>]
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

##### `pto.syncall`

**Summary:** Models the PTO-ISA `SYNCALL` family for all-participant synchronization across AIC/AIV cores.

**Forms:**

- Hard sync: no workspace operands
- Soft sync: `gm_workspace [+ used_cores]`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `gm_workspace` | optional GM pointer/tensor/partition view of `i32` | Global shared workspace used by soft mode |
| `used_cores` | optional `i32` | Explicit participant count for soft mode |
| `mode` | `#pto.sync_all_mode<...>` | `hard` or `soft` |
| `core_type` | `#pto.sync_core_type<...>` | `aiv_only`, `aic_only`, or `mix` |

**Results:** None.

**Constraints & Verification:**

- Hard mode requires no workspace operands and no `used_cores`.
- `core_type` remains required in Hard mode because lowering selects
  `SYNCALL<SyncCoreType::...>()` from it.
- Soft mode always requires `gm_workspace`.
- Soft `aiv_only`, `aic_only`, and `mix` use the same operand ABI.
- `gm_workspace` must be a GM `!pto.ptr`, `!pto.tensor_view`, or
  `!pto.partition_tensor_view` with `i32` elements.
- A pointer workspace must reference at least 16 contiguous `i32` elements;
  pointer capacity is a runtime responsibility because it is not encoded in
  the pointer type.
- When all dimensions are static, `gm_workspace` must contain at least 16
  elements (64 bytes). This verifier check applies to tensor/partition views;
  dynamic views must provide at least that capacity at runtime.
- The workspace must occupy an exclusive 64-byte cache line and be
  zero-initialized before its first `SYNCALL`. Alignment, aliasing, and
  initialization are runtime responsibilities and cannot be proven by the
  verifier.
- Omitting `used_cores`, or passing zero, asks PTO-ISA to derive the participant
  count from the launch configuration.

**Basic Example:**

```mlir
pto.syncall(%gm, %used : !pto.ptr<i32, gm>, i32)
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<aiv_only>

"pto.syncall"(%gm_view, %used) {
  operandSegmentSizes = array<i32: 1, 1>,
  mode = #pto.sync_all_mode<soft>,
  core_type = #pto.sync_core_type<aiv_only>
} : (!pto.partition_tensor_view<16xi32>, i32) -> ()

"pto.syncall"() {
  operandSegmentSizes = array<i32: 0, 0>,
  mode = #pto.sync_all_mode<hard>,
  core_type = #pto.sync_core_type<mix>
} : () -> ()
```

**Compatibility:** This is a breaking operand-schema change. Existing
three/four-operand `.pto` files and `.ptobc` files carrying the former
four-entry `operandSegmentSizes` must be regenerated. PTOAS and IR producers
such as PyPTO should be upgraded together.

---

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

### 4.17 CV-Related Operations

##### `#pto.kernel_kind<cube>` - Cube Kernel Function Attribute

**Summary:** Marks a `func.func` as a Cube-side kernel function.

**Semantics:**

```mlir
func.func @cube_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<cube>}
```

The attribute declares that the function is executed in Cube kernel context.
PTOAS uses this attribute to validate Cube-only frontend operations and to
recognize the function as a Cube participant in Cube/Vector communication.

**Attachment Site:** `func.func` attribute.

**Constraints & Verification:**

- Applies to kernel functions only
- Must not conflict with Vector-only frontend operations

**Basic Example:**

```mlir
func.func @cube_kernel()
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  // Cube-only operation
  pto.tmatmul ins(...) outs(...)
  return
}
```

---

##### `#pto.kernel_kind<vector>` - Vector Kernel Function Attribute

**Summary:** Marks a `func.func` as a Vector-side kernel function.

**Semantics:**

```mlir
func.func @vector_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<vector>}
```

The attribute declares that the function is executed in Vector kernel context.
PTOAS uses this attribute to validate Vector-only frontend operations and to
recognize the function as a Vector participant in Cube/Vector communication.

**Attachment Site:** `func.func` attribute.

**Constraints & Verification:**

- Applies to kernel functions only
- Must not conflict with Cube-only frontend operations

**Basic Example:**

```mlir
func.func @vector_kernel()
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  // Vector-only operation
  pto.tadd ins(...) outs(...)
  return
}
```

---

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

### 4.18 Frontend Pipe Communication Operations

PTOAS exposes a frontend-facing pipe communication interface for Cube/Vector
FIFO-style exchange. A pipe entry can be either a local tile buffer or a
GlobalTensor-like GM view descriptor. These operations are intended for
frontend/framework generated IR. The detailed design document is:

- `docs/designs/ptoas-tpush-tpop-design.md`

#### Common Notes

- `dir_mask` uses the current directional encoding:
  - `1`: C2V
  - `2`: V2C
  - `3`: both directions at frontend level
- `id` is a compile-time integer attribute used to bind
  `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` with the matching
  `pto.talloc_*` / `pto.tpush_*` / `pto.tpop_*` / `pto.tfree_*` ops in the same
  function.
- `slot_size` is expressed in bytes and uses the pre-split logical pipe-entry
  size.
- `slot_num` is an optional compile-time integer attribute on
  `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`. It controls the pipe
  FIFO depth. The `effective_slot_num` is the explicit value when present, or
  the default value: `8` for `dir_mask = 1/2` or `4` for `dir_mask = 3`.
- `local_slot_num` is an optional compile-time integer attribute on
  `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`.
  On A2/A3 it overrides the default consumer-side local FIFO slot count only
  when the pipe uses a local consumer FIFO buffer. Global-only GM FIFO pipes
  omit it.
- `pto.reserve_buffer.size` is the byte size of the consumer-side local FIFO
  buffer. For A2/A3 local FIFO pipes, it should be
  `slot_size * effective_local_slot_num`, where `effective_local_slot_num` is
  the explicit `local_slot_num` when present or the effective `slot_num`
  otherwise. For A5 local FIFO pipes, `local_slot_num` is not configurable and
  the reserved byte size should be `slot_size * effective_slot_num`.
- `nosplit` is an optional compile-time boolean attribute on
  `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`.
- `split` is a compile-time attribute, not a runtime SSA operand.
- `split = 0/1/2/3/4` corresponds to `TILE_NO_SPLIT`, `TILE_UP_DOWN`,
  `TILE_LEFT_RIGHT`, `TILE_UP_DOWN_ODD`, and `TILE_LEFT_RIGHT_ODD`.
- `TILE_UP_DOWN_ODD` splits an odd valid-row count so AIV0 receives
  `ceil(valid_rows / 2)` rows and AIV1 receives `floor(valid_rows / 2)` rows.
  `TILE_LEFT_RIGHT_ODD` applies the same rule to valid columns.
- Odd split modes are supported by GM-backed tile pipes in the C2V direction
  on A2/A3/A5 and in the V2C direction on A2/A3. A2/A3 also supports odd V2C
  on the enabled direction of a bidirectional pipe. The initialize op must
  provide GM slot backing and the matching direction's consumer buffer.
  Local-only pipes, A5 odd V2C, and GlobalTensor-entry odd splits are rejected.
- For an odd C2V tile pop, AIV0 and AIV1 must pass different
  `valid_row` / `valid_col` operands. The frontend must derive these operands
  from `pto.get_subblock_idx`: AIV0 supplies the `ceil` half and AIV1 supplies
  the `floor` half. PTO-ISA does not derive the two valid shapes automatically.
- For an odd V2C tile push, the two AIV producers must likewise set their tile
  valid shapes to the `ceil` and `floor` halves before `pto.tpush_to_aic`. The
  Cube-side `pto.tpop_from_aiv` describes the complete unsplit tile and, when
  its valid shape is dynamic, receives the full `valid_row` / `valid_col`.
- `pto.tpop_from_aic` and `pto.tpop_from_aiv` are result-valued frontend ops.
- Pipe entries support two forms:
  - tile entry: `!pto.tile_buf<...>` or the equivalent local memref after view
    lowering.
  - global entry: `!pto.tensor_view<...>` or the equivalent GM descriptor after
    lowering. This maps to pto-isa `GlobalTensor` overloads and only manages
    FIFO synchronization plus GM slot address assignment. Use
    `pto.partition_view` to derive a `!pto.partition_tensor_view<...>` window
    when a `pto.tload` / `pto.tstore` needs a sub-view of the entry.
- Global-entry pipe communication currently applies to the A2/A3 GM FIFO path
  (`pto.initialize_l2g2l_pipe`). It does not implicitly execute `pto.tstore` or
  `pto.tload`; callers move data explicitly before `tpush` or after `tpop`.
- When every transfer op bound to one pipe id uses a global entry, the pipe is
  a global-only GM FIFO. Its frontend initialize op carries `gm_slot_tensor`
  and may carry `slot_num`; `gm_slot_buffer`, `c2v_consumer_buf`,
  `v2c_consumer_buf`, `local_slot_num`, `pto.reserve_buffer`, and
  `pto.import_reserved_buffer` are not used.
- A GM-backed tile pipe carries GM slot backing plus the consumer buffer for
  each enabled direction: `c2v_consumer_buf` for C2V and
  `v2c_consumer_buf` for V2C. The GM descriptor describes the complete FIFO
  slot, while each consumer buffer supplies its direction's local staging
  area. `dir_mask = 3` requires both consumer buffers.
- For global entries, the matched initialize op's `gm_slot_tensor` describes
  one FIFO slot entry, not the full multi-slot FIFO buffer. Its dtype, shape,
  stride, and layout must match the `tensor_view` returned by `talloc` /
  `tpop` and form the pto-isa `GlobalData` template argument. Global entries
  support `TILE_NO_SPLIT`, `TILE_UP_DOWN`, and `TILE_LEFT_RIGHT`; odd split
  modes are limited to tile entries.
- If a global-entry result op does not carry explicit stride/layout metadata,
  PTOAS treats it as a row-major contiguous GM view. Non-contiguous cases must
  preserve stride/layout through the producing op metadata, the source view, or
  the lowered GM memref layout.
- A single logical pipe cannot mix `split = 0` with `split = 1` / `2` / `3` /
  `4`.
  `nosplit = true` requires all bound data-transfer ops to use `split = 0`;
  `nosplit = false` requires all bound data-transfer ops to use `split = 1`,
  `2`, `3`, or `4`.
- Multiple logical pipes are allowed in one function.
- A frontend logical pipe is uniquely identified by `function + id + direction`.
- When `dir_mask = 1` or `2`, one `id` denotes one single-direction logical
  pipe.
- When `dir_mask = 3`, one `id` denotes one DIR_BOTH physical pipe covering
  both logical directions.
- Lowered pipe components consume hardware flag ids per function:
  one single-direction pipe uses 2 ids, and one `dir_mask = 3` pipe uses 4 ids.
  The total usage in one function must fit within 16 hardware flag ids.

`nosplit` platform restrictions:

- On A5, `nosplit` supports a `1C:1V` pipe communication mode. The vector side
  may execute the pipe sequence on a single vector core.
- On A2/A3, `nosplit` follows the hardware `1C:2V` synchronization
  configuration. The two vector cores must run the same code for the same
  logical pipe, and the `talloc` / `tpush` / `tpop` / `tfree` sequence for that
  pipe must be identical in order on both vector cores. They do not need to
  reach each operation at the same time; only the relative order must remain
  consistent.

##### `pto.reserve_buffer` - Reserve Local Consumer FIFO Buffer

**Summary:** Declares a local reserved FIFO buffer region for the consumer side
of one frontend logical pipe when that pipe uses a local consumer FIFO buffer.
Global-only GM FIFO pipes do not use this op. The valid way to write this op
depends on whether the active PTOAS compilation flow enables local address
planning.

**Syntax:**

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = true
} -> i32
```

When the address is already fixed in the input IR:

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = false,
  base = 4096
} -> i32
```

**Arguments:**

- `name`: string attribute identifying the logical reserved buffer
- `size`: reserved buffer size in bytes. For A2/A3 local FIFO pipes this is
  `slot_size * effective_local_slot_num`; for A5 local FIFO pipes this is
  `slot_size * effective_slot_num`. Global-only GM FIFO pipes do not use
  `pto.reserve_buffer`.
- `location`: local address-space attribute, typically `vec` or `mat`
- `auto`: boolean allocation-mode flag in textual IR
- `base`: optional explicit local base address

**Results:** `i32` local base address value.

**Constraints & Verification:**

- Multiple `pto.reserve_buffer` ops are allowed in one function, but `name`
  must be unique within that function
- `size` must be greater than `0`; PTOAS allocates exactly the requested byte
  size, so it should match the local FIFO sizing rule of the pipe that consumes
  this buffer
- `location` must be a supported local address space
- Op-level verification requires:
  - `auto = false` must provide `base`
  - `auto = true` must not provide `base`
- Pipeline compatibility requires:
  - if the active compilation flow enables local address planning: write
    `auto = true` and omit `base`; `PlanMemory` assigns the address and
    `pto-resolve-reserved-buffers` materializes it
  - if the active compilation flow skips local address planning: write
    `auto = false` with explicit `base`; `pto-resolve-reserved-buffers` only
    propagates the pre-resolved address

##### `pto.import_reserved_buffer` - Import Peer Reserved FIFO Buffer

**Summary:** Imports the resolved local FIFO base address from the peer
function's reserved buffer declaration. Global-only GM FIFO pipes do not use
this op.

**Syntax:**

```mlir
%buf = pto.import_reserved_buffer {
  name = "c2v_fifo",
  peer_func = @vector_kernel
} -> i32
```

**Arguments:**

- `name`: reserved-buffer name in the peer function
- `peer_func`: peer `func.func` symbol

**Results:** `i32` imported local base address value.

**Constraints & Verification:**

- Multiple `pto.import_reserved_buffer` ops are allowed in one function, but
  the `(name, peer_func)` pair must be unique within that function
- `peer_func` must contain a matching `pto.reserve_buffer`
- The imported address is resolved by `pto-resolve-reserved-buffers`
  - from the peer `reserve_buffer.base` filled by `PlanMemory` when the active
    compilation flow enables local address planning
  - from the peer's explicit `base` when the active compilation flow skips
    local address planning

##### `pto.aic_initialize_pipe` - Frontend Cube Pipe Initialization

**Summary:** Frontend pipe initialization op used in Cube kernels.

**Syntax:**

```mlir
// A2/A3 (with GM slot buffer):
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_import : i32,
   v2c_consumer_buf = %c0_i32 : i32)

// A2/A3 global-only GM FIFO (GlobalTensor pipe entry):
%gm_slots = pto.make_tensor_view %gm_slot_buffer,
  shape = [%c16, %c16], strides = [%c16, %c1]
  : !pto.tensor_view<16x16xf32>
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
  (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

// A5 (without GM slot buffer):
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
  (c2v_consumer_buf = %c2v_import : i32,
   v2c_consumer_buf = %c0_i32 : i32)
```

**Arguments:**

- `id`: compile-time pipe identifier, unique among frontend initialize ops in
  the same function
- `dir_mask`: communication direction encoding
- `slot_size`: logical slot size in bytes
- `slot_num`: optional GM ring FIFO slot count; omitted defaults to `8` for
  `dir_mask = 1/2` or `4` for `dir_mask = 3`
- `local_slot_num`: optional A2/A3-only local FIFO slot count override for the
  lowered `pto.initialize_l2g2l_pipe`; omitted for global-only GM FIFO
- `nosplit`: optional compile-time boolean controlling no-split pipe mode
- `gm_slot_buffer`: optional GM pointer (`!pto.ptr<T>`), used by A2/A3 GM FIFO
  paths that also use a local consumer FIFO buffer
- `gm_slot_tensor`: optional single-slot entry descriptor
  (`!pto.tensor_view<...>`), required by global-only GM FIFO and by the A5
  GM-backed tile paths. For a global entry, its type describes the
  `tensor_view` returned by `talloc` / `tpop`. This descriptor is retained in
  IR for entry type validation; EmitC lowers the `TPipe` constructor argument
  to only the GM FIFO start address
- `c2v_consumer_buf`: optional C2V consumer local base address; omitted for
  global-only GM FIFO, but required when `gm_slot_tensor` backs a C2V tile pipe
- `v2c_consumer_buf`: optional V2C consumer local base address; omitted for
  global-only GM FIFO, but required when `gm_slot_tensor` backs a V2C tile pipe

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- Multiple `pto.aic_initialize_pipe` ops are allowed in one Cube function, but
  `id` must be unique among frontend initialize ops in that function
- If `slot_num` is present, it must be greater than `0`
- If `local_slot_num` is present, it must be greater than `0` and no greater
  than the effective `slot_num`
- On A5, `local_slot_num` must be omitted. A local-only frontend pipe lowers to
  `pto.initialize_l2l_pipe`; a C2V pipe carrying both `gm_slot_tensor` and
  `c2v_consumer_buf` lowers to `pto.initialize_l2g2l_pipe`
- A global-only GM FIFO initialize carries only `gm_slot_tensor`; it must not
  carry `gm_slot_buffer`, `local_slot_num`, `c2v_consumer_buf`, or
  `v2c_consumer_buf`; it may carry `slot_num`
- A GM-backed tile initialize may combine `gm_slot_tensor` only with
  `c2v_consumer_buf`, and currently requires `dir_mask = 1`
- For global-only GM FIFO, `slot_size` must match the byte size of
  `gm_slot_tensor`
- Global-entry `talloc` / `tpush` / `tpop` / `tfree` entry types must match the
  `gm_slot_tensor` descriptor in element type, rank, static shape, and byte size
- The lowered pipes for one function must fit within 16 hardware flag ids in
  total
- If `nosplit = true`, all frontend data-transfer ops bound to the same logical
  pipe must use `split = 0`
- If `nosplit = false`, all frontend data-transfer ops bound to the same
  logical pipe must use `split = 1`, `2`, `3`, or `4`

##### `pto.aiv_initialize_pipe` - Frontend Vector Pipe Initialization

**Summary:** Frontend pipe initialization op used in Vector kernels.

**Syntax:**

```mlir
// A2/A3 (with GM slot buffer):
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, slot_num = 2, local_slot_num = 1}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_local : i32,
   v2c_consumer_buf = %c0_i32 : i32)

// A2/A3 global-only GM FIFO (GlobalTensor pipe entry):
%gm_slots = pto.make_tensor_view %gm_slot_buffer,
  shape = [%c16, %c16], strides = [%c16, %c1]
  : !pto.tensor_view<16x16xf32>
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
  (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

// A5 (without GM slot buffer):
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024, nosplit = true}
  (c2v_consumer_buf = %c2v_local : i32,
   v2c_consumer_buf = %c0_i32 : i32)
```

**Arguments:** Same operand and attribute structure as
`pto.aic_initialize_pipe`.

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- Multiple `pto.aiv_initialize_pipe` ops are allowed in one Vector function,
  but `id` must be unique among frontend initialize ops in that function
- The lowered pipes for one function must fit within 16 hardware flag ids in
  total
- If `nosplit = true`, all frontend data-transfer ops bound to the same logical
  pipe must use `split = 0`
- If `nosplit = false`, all frontend data-transfer ops bound to the same
  logical pipe must use `split = 1`, `2`, `3`, or `4`

**Basic Example: GlobalTensor Pipe Entry Without Reserve/Import**

This C2V global-only GM FIFO example intentionally has no
`pto.reserve_buffer` and no `pto.import_reserved_buffer`. Both function
signatures keep the FIFO buffer as `!pto.ptr<f32>`, then use
`pto.make_tensor_view` before initialize_pipe to build the `gm_slot_tensor`
descriptor for one FIFO slot entry.

```mlir
func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                       %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>
  pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
             outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
  pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}

func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>
  pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
  pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}
```

##### `pto.talloc_to_aiv` - Frontend C2V Producer Global Entry Allocate

**Summary:** Allocates the next C2V GM FIFO slot for a GlobalTensor-like entry
in a Cube kernel.

**Syntax:**

```mlir
%entry = pto.talloc_to_aiv {id = 0, split = 1}
  -> !pto.tensor_view<128x512xf32>
```

**Arguments:**

- compile-time `id` attribute
- compile-time `split` attribute

**Results:** one `!pto.tensor_view<...>` global entry describing the
currently allocated FIFO slot.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents producer-side allocation for a C2V global-entry transaction
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 1` or `dir_mask = 3`
- Requires the matched pipe to lower to the A2/A3 GM FIFO path
- The result type must match the single-slot descriptor derived from the matched
  initialize op's `gm_slot_tensor`; element type, rank, static shape, and byte
  size are checked against the initialize op
- Does not write data and does not notify the consumer; callers must write the
  returned GM entry explicitly before the matching `pto.tpush_to_aiv`

##### `pto.talloc_to_aic` - Frontend V2C Producer Global Entry Allocate

**Summary:** Allocates the next V2C GM FIFO slot for a GlobalTensor-like entry
in a Vector kernel.

**Syntax:**

```mlir
%entry = pto.talloc_to_aic {id = 0, split = 1}
  -> !pto.tensor_view<128x512xf32>
```

**Arguments:**

- compile-time `id` attribute
- compile-time `split` attribute

**Results:** one `!pto.tensor_view<...>` global entry describing the
currently allocated FIFO slot.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents producer-side allocation for a V2C global-entry transaction
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 2` or `dir_mask = 3`
- Requires the matched pipe to lower to the A2/A3 GM FIFO path
- The result type must match the single-slot descriptor derived from the matched
  initialize op's `gm_slot_tensor`; element type, rank, static shape, and byte
  size are checked against the initialize op
- Does not write data and does not notify the consumer; callers must write the
  returned GM entry explicitly before the matching `pto.tpush_to_aic`

##### `pto.tpush_to_aiv` - Frontend C2V Producer Push

**Summary:** Pushes one C2V pipe entry from a Cube kernel. For tile entries this
keeps the existing tile-transfer behavior; for global entries this commits a GM
FIFO slot previously allocated by `pto.talloc_to_aiv`.

**Syntax:**

```mlir
pto.tpush_to_aiv(%tile : !pto.tile_buf<...>) {id = 0, split = 1}
pto.tpush_to_aiv(%entry : !pto.tensor_view<...>) {id = 0, split = 1}
```

**Arguments:**

- one pipe-entry operand: either a tile entry or a global entry
- compile-time `id` attribute
- compile-time `split` attribute

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the producer side of a C2V transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 1` or `dir_mask = 3`
- A global-entry operand requires a dominating matching `pto.talloc_to_aiv`
- A global-entry operand type must match the single-slot descriptor derived from
  the matched initialize op's `gm_slot_tensor`
- A global-entry push does not perform `TSTORE`; it only notifies the consumer
  that the GM FIFO slot is ready

##### `pto.tpush_to_aic` - Frontend V2C Producer Push

**Summary:** Pushes one V2C pipe entry from a Vector kernel. For tile entries
this keeps the existing tile-transfer behavior; for global entries this commits
a GM FIFO slot previously allocated by `pto.talloc_to_aic`.

**Syntax:**

```mlir
pto.tpush_to_aic(%tile : !pto.tile_buf<...>) {id = 0, split = 1}
pto.tpush_to_aic(%entry : !pto.tensor_view<...>) {id = 0, split = 1}
```

**Arguments:**

- one pipe-entry operand: either a tile entry or a global entry
- compile-time `id` attribute
- compile-time `split` attribute

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the producer side of a V2C transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 2` or `dir_mask = 3`
- A global-entry operand requires a dominating matching `pto.talloc_to_aic`
- A global-entry operand type must match the single-slot descriptor derived from
  the matched initialize op's `gm_slot_tensor`
- A global-entry push does not perform `TSTORE`; it only notifies the consumer
  that the GM FIFO slot is ready

##### `pto.tpop_from_aic` - Frontend C2V Consumer Pop

**Summary:** Pops one C2V pipe entry in a Vector kernel. For tile entries this
keeps the existing tile-pop behavior; for global entries this waits for producer
ready and assigns the current GM FIFO slot address into the returned
GlobalTensor-like descriptor.

**Syntax:**

```mlir
%tile = pto.tpop_from_aic {id = 0, split = 1} -> !pto.tile_buf<...>
%entry = pto.tpop_from_aic {id = 0, split = 1}
  -> !pto.tensor_view<128x512xf32>
```

**Arguments:** compile-time `id` and `split` attributes.

**Results:** one pipe entry. The result may be a `!pto.tile_buf<...>` tile entry
or a GlobalTensor-like `!pto.tensor_view<...>` GM entry.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the consumer side of a C2V transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 1` or `dir_mask = 3`
- A global-entry result requires the matched pipe to lower to the A2/A3 GM FIFO
  path
- A global-entry result type must match the single-slot descriptor derived from
  the matched initialize op's `gm_slot_tensor`
- A global-entry pop does not perform `TLOAD`; callers explicitly load from the
  returned GM entry or from views derived from it

##### `pto.tpop_from_aiv` - Frontend V2C Consumer Pop

**Summary:** Pops one V2C pipe entry in a Cube kernel. For tile entries this
keeps the existing tile-pop behavior; for global entries this waits for producer
ready and assigns the current GM FIFO slot address into the returned
GlobalTensor-like descriptor.

**Syntax:**

```mlir
%tile = pto.tpop_from_aiv {id = 0, split = 1} -> !pto.tile_buf<...>
%entry = pto.tpop_from_aiv {id = 0, split = 1}
  -> !pto.tensor_view<128x512xf32>
```

**Arguments:** compile-time `id` and `split` attributes.

**Results:** one pipe entry. The result may be a `!pto.tile_buf<...>` tile entry
or a GlobalTensor-like `!pto.tensor_view<...>` GM entry.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the consumer side of a V2C transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 2` or `dir_mask = 3`
- A global-entry result requires the matched pipe to lower to the A2/A3 GM FIFO
  path
- A global-entry result type must match the single-slot descriptor derived from
  the matched initialize op's `gm_slot_tensor`
- A global-entry pop does not perform `TLOAD`; callers explicitly load from the
  returned GM entry or from views derived from it

##### `pto.tfree_from_aic` - Frontend C2V Consumer Free

**Summary:** Releases the current C2V consumer slot in a Vector kernel.

**Syntax:**

```mlir
pto.tfree_from_aic {id = 0, split = 1}
pto.tfree_from_aic(%entry : !pto.tensor_view<...>) {id = 0, split = 1}
```

**Arguments:** compile-time `id` and `split` attributes. A global-entry free
also carries the entry descriptor returned by the matching `pto.tpop_from_aic`.

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the consumer free side of a C2V transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 1` or `dir_mask = 3`
- Tile-entry frees use the no-operand form
- Global-entry frees use the entry operand and must run after all explicit reads
  from that GM FIFO slot are complete; the entry operand type must match the
  single-slot descriptor derived from the matched initialize op's
  `gm_slot_tensor`

##### `pto.tfree_from_aiv` - Frontend V2C Consumer Free

**Summary:** Releases the current V2C consumer slot in a Cube kernel.

**Syntax:**

```mlir
pto.tfree_from_aiv {id = 0, split = 1}
pto.tfree_from_aiv(%entry : !pto.tensor_view<...>) {id = 0, split = 1}
```

**Arguments:** compile-time `id` and `split` attributes. A global-entry free
also carries the entry descriptor returned by the matching `pto.tpop_from_aiv`.

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the consumer free side of a V2C transfer
- `id` must match exactly one frontend initialize_pipe op in the same function
  with `dir_mask = 2` or `dir_mask = 3`
- Tile-entry frees use the no-operand form
- Global-entry frees use the entry operand and must run after all explicit reads
  from that GM FIFO slot are complete; the entry operand type must match the
  single-slot descriptor derived from the matched initialize op's
  `gm_slot_tensor`

---

### 4.19 Runtime Intrinsics

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

### 4.20 Debug Operations

##### `pto.tprint` - Print Tile

**Summary:** Prints the contents of a tile for debugging.

**Semantics:**

```
print(src)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` / global-memory view | Tile or global-memory view to print |
| `printFormat` | `i32` (optional, default: `0`) | Print format selector: `0=Width8_Precision4`, `1=Width8_Precision2`, `2=Width10_Precision6` |

**Results:** None.

**Constraints & Verification:**

- **Supported element type**:
  - Floating-point: `f32`, `f16`
  - Signless integers (by bitwidth): `i8`, `i16`, `i32`
- **For Tiles**: only `loc=vec` tiles are printable.
- **For GlobalTensor**: Layout must be one of `Layout::ND`, `Layout::DN`, or `Layout::NZ`.

## Behavior
- **Mandatory Compilation Flag**:

  On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the device-to-host debug channel. **You must enable the CCE option `-D_DEBUG --cce-enable-print`**.

- **Buffer Limitation:**

  The internal print buffer of `cce::printf` is limited in size. If the output exceeds this buffer, a warning message such as `"Warning: out of bound! try best to print"` may appear, and **only partial data will be printed**.

- **Synchronization**:

  Automatically inserts a `pipe_barrier(PIPE_ALL)` before printing to ensure all prior operations complete and data is consistent.

- **Formatting**:

  - `printFormat = 0`: `Width8_Precision4`
  - `printFormat = 1`: `Width8_Precision2`
  - `printFormat = 2`: `Width10_Precision6`
  - For `GlobalTensor`, due to data size and buffer limitations, only elements within its logical shape (defined by `Shape`) are printed.
  - For `tile_buf`, elements outside `valid_shape` are still printed and are marked with a `|` separator when partial validity is specified.

**Hardware Mapping:**

- Debug/diagnostic intrinsic (implementation-defined)

**Basic Example:**

```mlir
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {printFormat = 1 : i32}
```

---

##### `pto.print` - Print Scalar with Format String

**Summary:** Prints a scalar value using a compile-time format string (host-visible debug output).

**Semantics:**

```c
printf(format, scalar);
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `format` | `StrAttr` | Compile-time format string (e.g. `"%+08.3f"`); must be a literal attribute |
| `ScalarType` (signless integer / float) | Numeric value to print |

**Results:** None.

**Constraints & Verification:**

- `format` is a string attribute; it is not a pointer operand.
- `scalar` must be a numeric type (index / signless integer / f32).
- The op is side-effecting (marked with `MemWrite`) to prevent CSE from removing it.

**Hardware Mapping:**

- Lowered to a call to a debug printing routine (e.g. `cce::printf`) in the generated C++.

**Basic Example:**

```mlir
// Print a single f32 with fixed width/precision.
pto.print ins("%+08.3f", %v : f32)
```

---

##### `pto.trap` - Trap / Abort Execution

**Summary:** Unconditionally aborts execution at runtime. Intended for assertions and debug-only fail-fast paths.

**Semantics:**

```c
trap(); // does not return
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- May be used anywhere; terminates the current kernel or program as implementation-defined.
- Typically combined with `pto.print` or higher-level assertions for diagnostics.

**Hardware Mapping:**

- Lowered to a device-specific trap/abort intrinsic in the generated C++ (e.g. `TRAP()` or equivalent).

**Basic Example:**

```mlir
// Debug-only guard, e.g. in a lowered assertion.
pto.trap
```

---

### 4.21 Communication Operations

This section documents PTO communication primitives. PTOAS currently exposes:

- Synchronous point-to-point ops: `pto.comm.tput`, `pto.comm.tget`
- Synchronous signal ops: `pto.comm.tnotify`, `pto.comm.twait`, `pto.comm.ttest`
- Synchronous collective ops: `pto.comm.tbroadcast`, `pto.comm.tgather`, `pto.comm.tscatter`, `pto.comm.treduce`
- Asynchronous communication/session ops: `pto.comm.build_async_session`, `pto.comm.tput_async`, `pto.comm.tget_async`, `pto.comm.wait_async_event`, `pto.comm.test_async_event`

##### `pto.comm.build_async_session` - Create Async DMA Session

**Summary:** Creates an async DMA session handle used by `pto.comm.tput_async` and `pto.comm.tget_async`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `scratch` | `pto.tile_buf` | Local scratch/staging buffer used by the async runtime |
| `workspace` | `!pto.ptr<...>` | Global workspace backing the async session |
| `sync_id` | optional `i32` attr | Session synchronization ID |
| `block_bytes` | optional `i64` attr | Communication block size in bytes |
| `comm_block_offset` | optional `i64` attr | Per-block GM offset in bytes |
| `queue_num` | optional `i32` attr | Queue count hint |
| `channel_group_idx` | optional `i64` attr | Communication channel-group selector |

**Results:** `!pto.async_session`

**Constraints & Verification:**

- `scratch` must be a local tile buffer.
- `workspace` must be a GM pointer.
- Optional attrs are forwarded as session configuration and must use the declared integer types.

**Basic Example:**

```mlir
%session = pto.comm.build_async_session(%scratch, %workspace : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=256, v_row=1, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.ptr<i8>) {sync_id = 0 : i32} -> !pto.async_session
```

---

##### `pto.comm.tput_async` - Asynchronous Remote Write

**Summary:** Starts an asynchronous remote write from local GM to remote GM and returns an async event handle.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tensor_view` / `pto.partition_tensor_view` | Remote destination buffer |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Local source buffer |
| `session` | `!pto.async_session` | Async DMA session |

**Results:** `!pto.async_event`

**Constraints & Verification:**

- `dst` / `src` must be GM-shaped values with identical element type and static shape.
- Current lowering only supports flat contiguous logical-1D transfers for async GM operands.
- `session` must come from `pto.comm.build_async_session`.

**Basic Example:**

```mlir
%event = pto.comm.tput_async(%dst, %src, %session : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.async_session) -> !pto.async_event
```

---

##### `pto.comm.tget_async` - Asynchronous Remote Read

**Summary:** Starts an asynchronous remote read from remote GM to local GM and returns an async event handle.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tensor_view` / `pto.partition_tensor_view` | Local destination buffer |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Remote source buffer |
| `session` | `!pto.async_session` | Async DMA session |

**Results:** `!pto.async_event`

**Constraints & Verification:**

- Same operand constraints as `pto.comm.tput_async`.
- `session` must be compatible with the transfer workspace and staging configuration.

**Basic Example:**

```mlir
%event = pto.comm.tget_async(%dst, %src, %session : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.async_session) -> !pto.async_event
```

---

##### `pto.comm.wait_async_event` / `pto.comm.test_async_event` - Async Event Completion

**Summary:** Consume an async event produced by `pto.comm.tput_async` / `pto.comm.tget_async`.

**Arguments:**

| Op | Operands | Result | Description |
|----|----------|--------|-------------|
| `pto.comm.wait_async_event` | `event`, `session` | `i1` | Blocking wait for completion |
| `pto.comm.test_async_event` | `event`, `session` | `i1` | Non-blocking completion test |

**Constraints & Verification:**

- `event` must have type `!pto.async_event`.
- `session` must have type `!pto.async_session`.
- The event/session pair is expected to come from the same async communication flow.

**Basic Example:**

```mlir
%done0 = pto.comm.wait_async_event(%event0, %session : !pto.async_event, !pto.async_session) -> i1
%done1 = pto.comm.test_async_event(%event1, %session : !pto.async_event, !pto.async_session) -> i1
```

---

##### `pto.comm.tput` - Synchronous Remote Write

**Summary:** Copies data synchronously from local GM to remote GM through a VEC staging tile.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tensor_view` / `pto.partition_tensor_view` | Remote destination buffer |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Local source buffer |
| `buf` | `buf(%ping)` or `buf(%ping, %pong)` | Staging bundle: one or two local VEC tiles |
| `atomicType` | `#pto<atomic_type ...>` | Atomic mode, e.g. `atomic_none` or `atomic_add` |

**Constraints & Verification:**

- `dst` / `src` must be GM-shaped values. Static dimensions must be positive.
- Dynamic dimensions are supported only when both operands are
  `pto.partition_tensor_view` values.
- `dst` and `src` must have the same element type and identical static/dynamic
  shape signatures. Corresponding dynamic extents must be equal at runtime.
- Runtime extents must be nonnegative, and each partition range must remain
  within its backing tensor view. A zero extent denotes an empty transfer.
- `ping` / `pong` must be local VEC tile-like values whose element type matches
  `src`. Their physical `rows` / `cols` must be positive static values.
- Staging `v_row` / `v_col` values must be positive and may be static or
  dynamic. If a transfer enters the chunked path, a static `v_row` / `v_col`
  must exactly divide the corresponding logical row / column extent. Use
  dynamic valid dimensions when a partial final chunk is possible.
- When `pong` is present, it must have the same type as `ping`.

**Semantics:**

For every logical index in the common `src` / `dst` shape, the operation reads
the local `src` element and writes the corresponding remote `dst` element.
`atomic_none` performs a normal write; an atomic mode such as `atomic_add`
combines the source value with the destination according to that mode. The
staging bundle may divide the logical range into chunks but does not change the
logical transfer extent.

**Examples:**

Staging operands use the `buf(...)` bundle: one tile `buf(%ping)`, or ping–pong `buf(%ping, %pong)` for overlapping transfers.

```mlir
pto.comm.tput(%dst, %src, buf(%ping) : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {atomicType = #pto<atomic_type atomic_none>}

pto.comm.tput(%dst, %src, buf(%ping, %pong) : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) {atomicType = #pto<atomic_type atomic_add>}
```

For a variable-size transfer, construct both partitions with the same runtime
extent and use dynamic staging valid dimensions when the extent may require a
partial final chunk:

```mlir
%dst_part = pto.partition_view %dst_view,
  offsets = [%c0, %c0], sizes = [%rows, %c4096]
  : !pto.tensor_view<?x?xi8> -> !pto.partition_tensor_view<?x4096xi8>
%src_part = pto.partition_view %src_view,
  offsets = [%c0, %c0], sizes = [%rows, %c4096]
  : !pto.tensor_view<?x?xi8> -> !pto.partition_tensor_view<?x4096xi8>
%stage = pto.alloc_tile addr = %c0_i64
  valid_row = %c1 valid_col = %c4096
  : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=4096,
                  v_row=?, v_col=?, blayout=row_major,
                  slayout=none_box, fractal=512, pad=0>
pto.comm.tput(%dst_part, %src_part, buf(%stage)
  : !pto.partition_tensor_view<?x4096xi8>,
    !pto.partition_tensor_view<?x4096xi8>,
    !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=4096,
                  v_row=?, v_col=?, blayout=row_major,
                  slayout=none_box, fractal=512, pad=0>)
  {atomicType = #pto<atomic_type atomic_none>}
```

---

##### `pto.comm.tget` - Synchronous Remote Read

**Summary:** Copies data synchronously from remote GM to local GM through a VEC staging tile.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tensor_view` / `pto.partition_tensor_view` | Local destination buffer |
| `src` | `pto.tensor_view` / `pto.partition_tensor_view` | Remote source buffer |
| `ping` | `pto.tile_buf` | Required staging tile (wrapped in `buf(%ping)`) |
| `pong` | `pto.tile_buf` | Optional second staging tile (`buf(%ping, %pong)`) |

**Constraints & Verification:**

- The GM/global-like, dynamic partition, runtime extent, zero-length, and
  staging constraints are the same as for `pto.comm.tput`.
- `dst` and `src` must have the same element type and identical static/dynamic
  shape signatures.

**Semantics:**

For every logical index in the common `src` / `dst` shape, the operation reads
the remote `src` element and writes the corresponding local `dst` element. The
staging bundle may divide the logical range into chunks but does not change the
logical transfer extent.

**Examples:**

```mlir
pto.comm.tget(%dst, %src, buf(%ping) : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

pto.comm.tget(%dst, %src, buf(%ping, %pong) : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.comm.tnotify` / `pto.comm.twait` / `pto.comm.ttest` - Communication Signal Ops

**Summary:** Lower to `pto::comm::TNOTIFY/TWAIT/TTEST` for GM `i32` signal buffers.

**Arguments:**

| Op | Operands | Attributes | Result |
|----|----------|------------|--------|
| `pto.comm.tnotify` | `signal`, `value` | `notifyOp = #pto<notify_op atomic_add>` or `#pto<notify_op set>` | none |
| `pto.comm.twait` | `signal`, `cmpValue` | `cmp = #pto<wait_cmp eq/ne/gt/ge/lt/le>` | none |
| `pto.comm.ttest` | `signal`, `cmpValue` | `cmp = #pto<wait_cmp eq/ne/gt/ge/lt/le>` | `i1` |

**Constraints & Verification:**

- `signal` must be a GM-shaped value with element type `i32`.
- `value` / `cmpValue` must be signless integer scalars.

**Lowering ordering guarantee:**

- `pto.comm.tnotify` is lowered with a `pipe_barrier(PIPE_ALL)` emitted
  immediately before the `pto::comm::TNOTIFY(...)` call. `TNOTIFY_IMPL` writes
  the signal on the scalar pipe and only issues its trailing barrier *after*
  the store, so this preceding drain is what makes the
  `peer_TWAIT_returns ⇒ everything I issued before my TNOTIFY is visible`
  contract hold across `pto.tload` / `pto.tstore` (local or peer-addressed).
  Callers do not need to insert manual sync.

**Examples:**

```mlir
pto.comm.tnotify(%sig, %v : !pto.partition_tensor_view<1xi32>, i32) {notifyOp = #pto<notify_op set>}
pto.comm.tnotify(%sig, %v : !pto.partition_tensor_view<1xi32>, i32) {notifyOp = #pto<notify_op atomic_add>}
pto.comm.twait(%sig, %v : !pto.partition_tensor_view<1xi32>, i32) {cmp = #pto<wait_cmp ge>}
%ok = pto.comm.ttest(%sig, %v : !pto.partition_tensor_view<1xi32>, i32) {cmp = #pto<wait_cmp eq>} -> i1
```

---

##### `pto.comm.tbroadcast` - Collective Broadcast

**Summary:** Lowers to `pto::comm::TBROADCAST(...)`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | GM-shaped value | Root source buffer |
| `recv` | `recv(%ping)` or `recv(%ping, %pong)` | One or two local VEC staging tiles |
| `group` | variadic GM-shaped values | Parallel group members |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `src` must have the same type as each `group` member.
- `root` must be in range `[0, group.size)`.

**Examples:**

Single receive buffer:

```mlir
pto.comm.tbroadcast(%src, recv(%ping), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}
```

Optional ping–pong (`recv(%ping, %pong)` adds a second tile type in the operand-type list):

```mlir
pto.comm.tbroadcast(%src, recv(%ping, %pong), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}
```

---

##### `pto.comm.tgather` - Collective Gather

**Summary:** Communication collective that lowers to `pto::comm::TGATHER(...)`. This op is distinct from tile-level `pto.tgather`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM-shaped value | Destination buffer (gather target) |
| `recv` | `recv(%ping)` or `recv(%ping, %pong)` | Staging tile(s) |
| `group` | variadic GM-shaped values | Parallel group members |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `dst` element type must match the group element type.
- `ping` / `pong` must be local VEC tile-like values with element type matching `dst`.
- `root` must be a valid index into `group` (i.e. `root < group.size()`).

**Examples:**

```mlir
pto.comm.tgather(%dst, recv(%ping), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}

pto.comm.tgather(%dst, recv(%ping, %pong), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}
```

---

##### `pto.comm.tscatter` - Collective Scatter

**Summary:** Communication collective that lowers to `pto::comm::TSCATTER(...)`. This op is distinct from tile-level `pto.tscatter`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | GM-shaped value | Source buffer (scatter root) |
| `recv` | `recv(%ping)` or `recv(%ping, %pong)` | Staging tile(s) |
| `group` | variadic GM-shaped values | Parallel group members |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `src` element type must match the group element type.
- `ping` / `pong` must be local VEC tile-like values with matching element type.

**Examples:**

```mlir
pto.comm.tscatter(%src, recv(%ping), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}

pto.comm.tscatter(%src, recv(%ping, %pong), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {root = 1 : i32}
```

---

##### `pto.comm.treduce` - Collective Reduce

**Summary:** Lowers to `pto::comm::TREDUCE(...)`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM-shaped value | Reduced output buffer |
| `acc` | local VEC tile-like value | Accumulation tile |
| `recv` | `recv(%ping)` or `recv(%ping, %pong)` | One or two receive staging tiles |
| `group` | variadic GM-shaped values | Parallel group members |
| `reduceOp` | `#pto<reduce_op sum>` / `#pto<reduce_op max>` / `#pto<reduce_op min>` | Reduction mode |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `dst` element type must match the group element type.
- `acc` and `recv(%ping)` / `recv(%ping, %pong)` operands must be local VEC tile-like values whose element type matches `dst`.

**Examples:**

Sum with a single receive tile:

```mlir
pto.comm.treduce(%dst, %acc, recv(%ping), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {reduceOp = #pto<reduce_op sum>, root = 1 : i32}
```

Max with ping–pong receive buffers (two staging tiles — operand-type list includes three `tile_buf` entries: `acc`, `ping`, `pong`):

```mlir
pto.comm.treduce(%dst, %acc, recv(%ping, %pong), group(%g0, %g1, %g2) :
  !pto.partition_tensor_view<128xf32>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>,
  !pto.partition_tensor_view<128xf32>) {reduceOp = #pto<reduce_op max>, root = 1 : i32}
```

---

### 4.22 Stack-Local Array Operations

Minimum support for **C++ stack-local statically-shaped arrays of scalars** —
suitable for small auxiliary buffers in host scalar code. Disjoint from the
tile-buf world: these values do not participate in PTO memory planning or
tile address materialization, and their underlying address is decided by the host C++
compiler. Naming and asm style mirror the `eventid_array` triad.

Operates on the [`!pto.local_array<...>`](#26-ptolocal_arrayd1-x-d2-x--x-dk-x-t) type. See Section 2.6 for type-level constraints.

##### `pto.declare_local_array` - Declare a Stack-Local Array

**Summary:** Declare a statically-shaped scalar array on the C++ stack.

**Semantics:** Lowers to `T a[D1][D2]...;` in the emitted C++.

**Arguments:** None.

**Results:** `!pto.local_array<D1 x D2 x ... x T>`

**Basic Example:**

```mlir
%a = pto.declare_local_array -> !pto.local_array<16xi32>     // int32_t a[16];
%m = pto.declare_local_array -> !pto.local_array<4x8xf32>    // float   m[4][8];
```

---

##### `pto.local_array_get` - Read One Element by Index

**Summary:** Read a single element by full-rank indexing.

**Semantics:** `result = a[i0][i1]...[iN-1]` (rvalue).

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `array` | `!pto.local_array<D1 x ... x Dk x T>` | The local array |
| `indices` | variadic `Index` | Exactly `k` indices, one per dim |

**Results:** Scalar matching the array's element type.

**Constraints & Verification:**

- Number of indices must equal the array's rank (verifier error otherwise).
- Result type must equal the array's element type.

**Basic Example:**

```mlir
%v = pto.local_array_get %m[%i, %j]
   : !pto.local_array<4x8xf32> -> f32                      // v = m[i][j];
```

---

##### `pto.local_array_set` - Write One Element by Index

**Summary:** Write a scalar into one element by full-rank indexing.

**Semantics:** `a[i0][i1]...[iN-1] = value;`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `array` | `!pto.local_array<D1 x ... x Dk x T>` | The local array |
| `indices` | variadic `Index` | Exactly `k` indices, one per dim |
| `value` | `ScalarType` | Scalar value to write |

**Results:** None.

**Constraints & Verification:**

- Number of indices must equal the array's rank.
- `value` type must equal the array's element type.

**Basic Example:**

```mlir
pto.local_array_set %m[%i, %j], %v
   : !pto.local_array<4x8xf32>, f32                        // m[i][j] = v;
```

---

#### Struct Ops (`pto.declare_struct` / `pto.struct_get` / `pto.struct_set`)

C++ stack-local heterogeneous aggregate ops. Disjoint from the tile-buf /
scratch memory world: the struct's address is decided by the host C++
compiler. Naming and asm style mirror the `local_array` triad.

Operates on the [`!pto.struct<...>`](#28-ptostructt0-t1--tn-1) type. See
Section 2.8 for type-level constraints.

##### `pto.declare_struct` - Declare a Stack-Local Struct

**Summary:** Declare a heterogeneous aggregate on the C++ stack.

**Semantics:** `PtoStruct_X s;`

**Arguments:** None.

**Results:** `!pto.struct<T0, ..., Tn-1>`

**Constraints & Verification:**

- The result must not be passed to a terminator (`func.return`, `scf.yield`,
  ...). The value is a pointer to stack storage owned by the declaring scope, so
  letting it out would expose the address of storage that is about to die.
- Relatedly, no function may **return** a `!pto.struct` (checked module-wide),
  which is what stops a pass-through helper from laundering that address past
  the per-op check.
- No other operation may produce a `!pto.struct` result. This prevents
  `arith.select`, `scf.if`, calls, casts, or loop-carried values from hiding the
  storage origin.

  See [Section 2.8](#28-ptostructt0-t1--tn-1); mutation is in place, so declare
  the struct in the outer scope and pass it down instead.

**Basic Example:**

```mlir
%s = pto.declare_struct -> !pto.struct<f16, i8>   // PtoStruct_f16_i8 s;
```

---

##### `pto.struct_get` - Read a Field by Constant Path

**Summary:** Read the field reached by a constant `path` of field indices,
descending through nested structs.

**Semantics:** `s.fA.fB...`, read into a fresh variable so the SSA value keeps
its value if the struct is mutated later (value semantics, like
`pto.local_array_get`).

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `s` | `!pto.struct<...>` | The struct |
| `path` | `DenseI64ArrayAttr` | Constant field indices, e.g. `[0, 2]` |

**Results:** `value` — the scalar field type at the end of `path`.

**Constraints & Verification:**

- `path` is non-empty; every index is in range at its level.
- An intermediate index may only descend into a nested struct.
- `path` must **end at a scalar field**. A path stopping on a nested
  `!pto.struct` is rejected — returning the aggregate would copy it out of the
  parent struct; extend the path to reach a scalar inside it.
- Result type must equal the field type at the end of `path`.

**Basic Example:**

```mlir
%r = pto.struct_get %s[0]    : !pto.struct<f16, i8> -> f16      // r = s.f0;
%v = pto.struct_get %s[0, 1]                                    // v = s.f0.f1;
   : !pto.struct<!pto.struct<f32, i8>, i16> -> i8
```

---

##### `pto.struct_set` - Write a Field by Constant Path

**Summary:** Write a value into the field reached by a constant `path` of field
indices, descending through nested structs.

**Semantics:** `s.fA.fB... = value;`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `s` | `!pto.struct<...>` | The struct |
| `path` | `DenseI64ArrayAttr` | Constant field indices, e.g. `[0, 2]` |
| `value` | `AnyType` | Value to write (must match the field type) |

**Results:** None.

**Constraints & Verification:**

- `path` is non-empty; every index is in range at its level.
- An intermediate index may only descend into a nested struct.
- `path` must **end at a scalar field**, same as `pto.struct_get`. A path
  stopping on a nested `!pto.struct` is rejected — extend it to reach a scalar.
- Value type must equal the field type at the end of `path`.

**Basic Example:**

```mlir
pto.struct_set %s[1], %v : !pto.struct<f16, i8>, i8            // s.f1 = v;
pto.struct_set %s[0, 0], %v                                    // s.f0.f0 = v;
   : !pto.struct<!pto.struct<f32, i8>, i16>, f32
```

When the struct is a function argument the same op emits `s->f0 = v;` — see
[Section 2.8](#28-ptostructt0-t1--tn-1) for why a struct is carried as a
pointer.

---

## 5. Operation Summary Table

| Category | Count | Pipeline |
|----------|-------|----------|
| Pointer/View | 5 | - |
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
| MX Quantized | 3 | M/V |
| Synchronization | 5 | - |
| CV-Related | 2 | - |
| Runtime Intrinsics | 4 | - (Pure) |
| Debug | 3 | - |
| Stack-Local Array | 3 | - (Scalar / Host) |
| Stack-Local Struct | 3 | - (Scalar / Host) |

**Total: 113 operations**
