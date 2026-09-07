# 1. Load / Store

> **Category:** A (+B on `dintlv`/`intlv`). **Mask:** load none (A5 loads are unpredicated), store `Pg`.
>
> `vload`/`vstore` are logical memory ops. **`[dist_mode]` explicitly declares
> the access pattern**, defaulting to `continuous` (contiguous); the optional
> modes are `dintlv` (deinterleaved dual load), `brc` (broadcast) and `intlv` (interleaved dual store).


---

## `pto.vmi.vload`

- **semantics:** Load elements of type `T` from UB into a logical vector
  register starting at `%source + %offset` (element offset). The default
  (`continuous`) is a contiguous stride-1 read:

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ub[base + offset + i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the load may instead read in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode), deinterleave the source, or broadcast. The exact pattern is
  determined by these mutually exclusive attributes (see attributes and
  lowering below).

- **syntax:**
  ```mlir
  %result = pto.vmi.vload %source[%offset] : !pto.ptr<T, ub> -> !pto.vmi.vreg<L×T>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group load: C rows of L/C elements, row g at base + g*stride
  %result = pto.vmi.vload %source[%offset], %stride {group = C}
      : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<L×T>
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided load: %block_stride is a dynamic i16 operand (no mask)
  %result = pto.vmi.vload %source[%offset], %block_stride
      : !pto.ptr<T, ub>, i16 -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Loaded logical vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"dintlv"`, `"brc"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group load arity; mutually exclusive with `dist_mode`; requires `stride` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior (applied at consumer, not on load) |

- **lowering to `pto.mi`:**
  - **dist-mode** `vload` and `vstore` accept an optional `{dist_mode = "..."}` attribute
declaring the memory access pattern. Default is `"continuous"`.

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | Known 32B-aligned addresses use the aligned fast path; other effective UB addresses use an unaligned sequence with lowering-managed alignment state |
  | `"dintlv"` | `K × pto.vldsx2 {dist="DINTLV_B*"}` (deinterleaved dual load; suffix from `Ptr<T>`) |
  | `"brc"` | `1 × pto.vlds {dist="BRC_B*"}` or `BRC_BLK`; broadcast-axis (1-reg backing, replicate-read) |

  **Group mode** (`{group = C}` + `stride`) has two sub-cases, decided by the
  relation between `result.L` and `C`:
  - **Full-group load** (`result.L > C`): each group loads `L/C` elements,
    row-strided tile load: `C·(L/C) = L` elements across `C` rows,
    each row `g` at offset `base + g·stride`.
  - **Slot load** (`result.L == C`): each group loads **1 scalar** into the
    corresponding slot, producing a compact `V<C×T>`. This is the
    dual of group reduce — reduce
    folds lanes into slots, slot load reads those slots back into a vreg.
    `C ∈ {1, 2, 4, 8}`. Not combinable with `dist_mode`.

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided load.
  Memory is read in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. A5 loads are
  unpredicated, so an implicit all-active mask is applied. Not combinable
  with `dist_mode` or `group`.

  `B*` suffix is derived from `Ptr<T>` element width: `f32/i32 → B32`, `f16/bf16/i16 → B16`, `i8/fp8 → B8`.

- **examples:**

  ```mlir
  // Continuous load (default dist_mode): UB → vreg
  %v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // → pto.as: Ptr<f32> → B32, dist_mode=continuous → pto.mi.vlds {dist="NORM"}
  // Slot load: 1 scalar per group → compact V<8×f32> (reads back reduce output)
  %s = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<8×f32>
  // → each of 8 groups loads 1 scalar into its slot
  // Full-group load: 8 rows × 8 elements, stride 64
  %t = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<64×f32>
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided load: block_stride = 8 (dynamic i16 operand, no mask)
  %vb = pto.vmi.vload %ub[%off], %c8_i16
      : !pto.ptr<f32, ub>, i16 -> !pto.vmi.vreg<64×f32>
  // → block-strided load (block=8), all lanes active

  // Broadcast load: scalar/block replicate into vreg
  %vb = pto.vmi.vload %ub[%offset] {dist_mode = "brc"} : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // → pto.as: Ptr<f32> → B32, dist_mode=brc → pto.mi.vlds {dist="BRC_B32"}
  ```

- **notes:**
  - **A5 loads are unpredicated.** A tail mask associated with a `vload` is
    never lowered as a masked load. It migrates to the consuming compute op or
    to a `vstore`.
  - Continuous loads support effective UB addresses that are not 32B-aligned.
    Alignment state is managed by the lowering and is not part of the VMI
    programming model.
  - `dist_mode` and layout inference are orthogonal: `pto.as` may still
    rewrite the physical layout of a `continuous` load to serve a downstream
    consumer (e.g. a grouped reduce).
  - The `pmode` attribute on `vload` governs the result lane behavior at the
    *consumer*, not on the load itself.

- **attention:**
  - **`{group}`, `%block_stride`, and `{dist_mode}` are mutually exclusive.**
    Specifying more than one at once is rejected by `pto.as`.
  - **`stride` operand is bound to `{group}`.** It is required with
    `{group = C}` and invalid otherwise; `block_stride` is bound to the
    block-stride mode and invalid otherwise. `vload` has no mask operand in
    any mode (A5 loads are unpredicated).


---

## `pto.vmi.vstore`

- **semantics:** Store elements from a vector register to UB starting at
  `%dest + %offset` (element offset). The default (`continuous`) is a
  contiguous stride-1 write; only lanes where `mask[i] != 0` are written
  (A5 stores are predicated):

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offset + i] = src[i];
  ```

  The access pattern is not always contiguous: depending on the attributes
  (`{dist_mode}`, `{group = C}` with a `stride` operand, or
  `%block_stride`), the store may instead write in a strided/scattered
  fashion (e.g. per-row stride for group mode, 32B-block stride for
  block-stride mode). The exact pattern is determined by these mutually
  exclusive attributes (see attributes and lowering below).

- **syntax:**
  ```mlir
  pto.vmi.vstore %value, %dest[%offset], %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.mask<L>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group store: C rows of L/C elements, row g at base + g*stride (no mask)
  pto.vmi.vstore %value, %dest[%offset], %stride {group = C}
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, index
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided store: %block_stride is a dynamic i16 operand (mask required)
  pto.vmi.vstore %value, %dest[%offset], %block_stride, %mask
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, i16, !pto.vmi.mask<L>
  ```
- **operands:**

 | Operand | Type | Description |
 |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Vector value to store |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (variadic: 0 or 1) |

- **results:** *(none)*

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `dist_mode` | `"continuous"`, `"intlv"` | `"continuous"` | Memory access pattern |
  | `group` | positive integer | *(none)* | Strided group store arity; mutually exclusive with `dist_mode`; requires `stride`; forbids `mask` |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior: `"zero"` (default) stores 0; `"merge"` skips write on inactive lanes |

- **lowering to `pto.mi`:**
  - **dist-mode** `vload` and `vstore` accept an optional `{dist_mode = "..."}` attribute
declaring the memory access pattern. Default is `"continuous"`.

  | `dist_mode` | Physical lowering |
  |---|---|
  | `"continuous"` | Known 32B-aligned addresses use the aligned fast path; unmasked accesses to other effective UB addresses use an unaligned sequence with lowering-managed alignment state |
  | `"intlv"` | `K × pto.vstsx2 {dist="INTLV_B*"}` (interleaved dual store; suffix from `Ptr<T>`) |

  **Group mode** (`{group = C}` + `stride`): row-strided tile store. Not combinable with
  `dist_mode` or `mask` (group stores are unpredicated).

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided store.
  Memory is written in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. An explicit
  `mask` is applied; if absent an implicit all-active mask is used. Not
  combinable with `dist_mode` or `group`.

  Continuous unmasked stores support effective UB addresses that are not
  32B-aligned, including a final prefix tail. Alignment state and the required
  final flush are managed by lowering and are not part of the VMI programming
  model. Explicit sparse-mask stores still use the predicated store path and
  require their target alignment constraints to be statically provable;
  otherwise the access is unsupported.

- **examples:**

  ```mlir
  // Continuous store (default): vreg → UB, masked
  pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
  // → pto.as: Ptr<f32> → B32, dist_mode=continuous → pto.mi.vsts {dist="NORM_B32"}
  ```

  ```mlir
  // Group (strided) store: 8 rows × 8 elements, stride 64 (no mask)
  pto.vmi.vstore %tile, %ub_out[%off], %stride {group = 8}
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, index
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided store: block_stride = 8 (dynamic i16 operand + mask)
  pto.vmi.vstore %v, %ub_out[%off], %c8_i16, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, i16, !pto.vmi.mask<64>
  // → block-strided store (block=8), governed by mask
  ```


---

## `pto.vmi.vsstb`

- **semantics:** Block-strided store — a dedicated form of `vstore`'s
  block-stride mode, where the store is read in 32B blocks with block `blk`
  written to `base + blk * block_stride` (scattered access). Only the
  `%block_stride` operand is exposed; the internal repeat stride is fixed to 0,
  so this is the zero-repeat-stride alias of the `pto.mi.vsstb`
  hardware op. Stores are predicated, so only lanes where `mask[i] != 0`
  are written:

  ```c
  // repeat_stride fixed to 0; block = 32B / sizeof(T) lanes
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offset + block_index(i) * block_stride] = src[i];
  ```

  Equivalently, at the 32B-block granularity exposed by the hardware
  (`E_v = 32 / sizeof(T)` lanes per block, i.e. one VLane per block):

  ```c
  for (int blk = 0; blk < L / E_v; ++blk) {
      if (pg[blk])
          UB_block[base + offset + blk * block_stride] = src_block[blk];
  }
  ```

- **syntax:**
  ```mlir
  // block-strided store: %block_stride is a dynamic i16 operand (mask required)
  pto.vmi.vsstb %value, %dest[%offset], %block_stride, %mask
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, i16, !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Vector value to store |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offset` | `index` | Element offset from base |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate; `E_v` lanes per 32B block share one mask bit at block granularity |

- **results:** *(none)*

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"` | `"zero"` | Inactive-lane behavior: `"zero"` skips the write on inactive blocks |

- **lowering to `pto.mi`:**
  ```
  1 × pto.mi.vsstb {repeat_stride = 0}  (per physical reg)
  ```
  `#mi = K`, `dep = 1`. Structurally 1:1 with `pto.mi.vsstb` at `repeat_stride = 0`
  — one store op per physical register, each governed by the per-reg block mask.

- **examples:**
  ```mlir
  // Block-strided store: block_stride = 8 (dynamic i16 operand + mask)
  pto.vmi.vsstb %v, %ub_out[%off], %c8_i16, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, i16, !pto.vmi.mask<64>
  // → block-strided store (block=8), governed by mask, repeat_stride = 0
  ```

- **notes:**
  - `vsstb` is the **alias** of `vstore`'s block-stride mode:
    `vstore ... %block_stride, %mask` and `vsstb ... %block_stride, %mask` denote
    the same access pattern. `vsstb` exists so block-strided tile stores are
    spelled by intent rather than overloaded onto `vstore`'s operand list.
  - The `repeat_stride` field of the underlying `pto.mi.vsstb` hardware control
    word is **not exposed** here; it is fixed to 0. A nonzero repeat stride is not
    expressible on the `pto.vmi` surface (no op carries it).
  - **Stores are predicated.** The mask gates 32B-block participation; an
    implicit all-active mask is applied when the operand is omitted.
  - Not combinable with `{dist_mode}`, `{group}`, or the `%block_stride` form of
    `vstore` — `vsstb` is mutually exclusive with all other `vstore` access modes.
