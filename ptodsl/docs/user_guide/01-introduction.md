# 1. Introduction

**PTO** is a virtual instruction set designed for the Ascend NPU — a hardware-abstracted programming model that exposes the full capability of the Cube, Vector, and Scalar compute units through a unified operation set. **PTODSL** is the Python frontend for PTO. It wraps the PTO instruction set in a Python-embedded DSL with tracing-based compilation, so you can write PTO programs using familiar Python syntax. Under the hood, PTODSL traces your kernel function into PTO IR, which the PTOAS compiler then lowers, optimizes, and emits as NPU executables. In short: PTO defines the *what* (the instruction set), PTODSL provides the *how* (the authoring experience), and together they give you direct access to all three NPU compute units without leaving Python.

## 1.1 Target hardware

The Ascend NPU is organized around three compute units and a shared on-chip buffer, connected through the Memory Transfer Engine (MTE):

```
                          ┌─────────────────────────┐
                          │     Global Memory (GM)    │
                          │       (off-chip HBM)      │
                          └────────────┬──────────────┘
                                       │
                            ┌──────────┴──────────┐
                            │    MTE (DMA engine)  │
                            └──────────┬──────────┘
                                       │
                          ┌────────────┴──────────────┐
                          │   Unified Buffer (UB)      │
                          │     (on-chip scratchpad)   │
                          └──┬───────────┬──────────┬──┘
                             │           │          │
                    ┌────────┴──┐  ┌─────┴──────┐  │
                    │ LEFT/RIGHT│  │            │  │
                    │  /ACC/BIAS│  │  Vector    │  │
                    │           │  │  (SIMD)    │  │
                    │   Cube    │  │            │  │
                    │           │  └────────────┘  │
                    └───────────┘                  │
                                        ┌──────────┴──┐
                                        │   SIMT      │
                                        │ (scalar PG)  │
                                        └─────────────┘
```

| Unit | Role | Typical workload |
|------|------|------------------|
| **Cube** | Matrix multiplication | GEMM, convolution |
| **SIMD** | Row-wise vector math | activation, normalization, reduction |
| **SIMT** | Scalar-programmable unit | pointwise tile walks, metadata |

- **Global Memory (GM)** is off-chip HBM. All input and output tensors reside here.
- **Unified Buffer (UB)** is the on-chip scratchpad shared by all three compute units. Tile buffers and intermediate results live here during kernel execution.
- **MTE** (Memory Transfer Engine) handles DMA transfers between GM and UB, and between UB regions.
- **Cube** has its own private on-chip buffers — LEFT, RIGHT, ACC, and BIAS — for staging matrix operands and accumulators.
- **SIMD** executes row-wise vector instructions directly on UB-resident data.
- **SIMT** is a scalar-programmable processor group that executes scalar instructions across many work-items in parallel. It is well-suited for per-element control logic, tile boundary metadata, and pointwise blends.

PTODSL gives you direct access to all three units and the data-movement
surfaces around them, without abstracting away the hardware boundaries.

## 1.2 Authoring model

PTODSL's public kernel model is **one decorator, two roles, two backends**:

```
Python Wrapper              L0  user-facing wrapper (NumPy, torch-npu, pure Python)
  └─ @pto.jit(entry=True)         host-launchable kernel entry
  │    ├─ backend="vpto"          VPTO backend (default), mode="auto" or "explicit"
  │    └─ backend="emitc"         EmitC backend, mode="auto" only
  ├─ @pto.jit(entry=False)        kernel module, callable from entries and other modules
  │    ├─ backend="vpto"          VPTO backend, mode="auto" or "explicit"
  │    └─ backend="emitc"         EmitC backend, mode="auto" only
  ├─ Tile Ops                     tile.load, tile.store, tile.add, ...
  ├─ MTE Ops                      mte_load / mte_store / mte_gm_ub / ...
  ├─ @pto.tileop                    matrix products (mad, mte_l1_l0a, mte_l0c_ub, ...)
  ├─ @pto.tileop                    row-wise vector math (vlds, vadd, vexp, vsts, ...)
  └─ @pto.simt                    scalar-like compute (lds, sts, pointwise blends, ...)
```

### Python wrapper

The outermost layer is plain Python. It handles ergonomic runtime concerns: allocating output tensors, extracting shapes and strides from framework tensors, compiling the JIT kernel, and launching it. Because the wrapper is just Python, you can freely mix in NumPy, torch-npu, or any other Python framework for pre- and post-processing, data preparation, or composing multiple kernel launches. It knows nothing about NPU internals — it is just a convenience function that most end users will call.

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.flash_attention_wrapper","symbol":"flash_attention"} -->
```python
def flash_attention(Q, K, V, *, O=None, causal=False):
    if O is None:
        O = pto.empty_like(Q)
    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape
    compiled = flash_attention_kernel.compile(
        BLOCK_Q=128, BLOCK_KV=128, CAUSAL=causal
    )
    compiled[batch * heads, stream](
        Q.data_ptr(),
        K.data_ptr(),
        V.data_ptr(),
        O.data_ptr(),
        batch,
        seq_q,
        seq_k,
        heads,
        dim,
    )
    return O
```

### `@pto.jit` — kernel entries and kernel modules

Decorating a function with `@pto.jit` marks it as a PTO kernel. The decorator
has two roles controlled by `entry`:

- **`entry=True`** (the default): a host-launchable kernel entry. The function
  is traced, compiled, and can be invoked with `compiled[grid, stream](args...)`.
- **`entry=False`**: a kernel module — a device-side function that entries and
  other modules can call from their traced bodies. Modules are not
  host-launchable — calling `.compile()` or `[grid, stream]` on them raises an
  error. They are compiled and linked together with their callers automatically.

In both roles, compilation, caching, and tracing work the same way: the body is
traced once, lowered through the PTOAS compiler pipeline, and cached by
specialization key (function identity + entry annotation signature + const_expr
parameter values).

#### `entry=True` — host-launchable kernel

The host-entry contract is pointer-first. Device buffers are explicit GM
pointers (`pto.ptr(..., "gm")`), launch-varying shape/stride metadata travels as
runtime scalars, and the kernel body materializes `TensorView` descriptors with
`make_tensor_view(ptr, shape=..., strides=...)`. Compile-time constants remain
keyword-only `pto.const_expr` parameters:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"flash_attention_kernel","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def flash_attention_kernel(
    Q_ptr: pto.ptr(pto.f32, "gm"),
    K_ptr: pto.ptr(pto.f32, "gm"),
    V_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    batch: pto.i32,
    seq_q: pto.i32,
    seq_k: pto.i32,
    heads: pto.i32,
    dim: pto.i32,
    *,
    BLOCK_Q: pto.const_expr = 128,
    BLOCK_KV: pto.const_expr = 128,
    CAUSAL: pto.const_expr = False,
):
    q_view = pto.make_tensor_view(
        Q_ptr,
        shape=[batch, seq_q, heads, dim],
        strides=[seq_q * heads * dim, heads * dim, dim, 1],
    )
    # ... tile allocation, block partitioning, and sub-kernel dispatch ...
    return
```

`@pto.jit(entry=True)` is the host-visible kernel entry. The SPMD launch
contract is also owned here: the runtime grid (e.g., `batch * heads` blocks) is
declared at the call site, and block/subblock indices are queried via
`pto.get_block_idx()` and friends.

`make_tensor_view(..., layout=...)` accepts `"ND"`, `"DN"`, and `"NZ"`
(case-insensitive). Use it when the logical layout cannot be distinguished from
shape/stride alone. For example, fp32 canonical NZ has `C0 = 32 / 4 = 8`:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"explicit_nz_view","compile":{}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def explicit_nz_view(ptr: pto.ptr(pto.f32, "gm")):
    view = pto.make_tensor_view(
        ptr,
        shape=[1, 8, 8, 16, 8],
        strides=[8192, 1024, 128, 8, 1],
        layout="NZ",
    )
    _ = view
    return
```

An explicit layout is preserved by PTOAS. Without one, only a non-degenerate
canonical NZ five-dimensional root is inferred as NZ; a single-fractal
`[1,1,1,16,C0]` view remains ND.

#### `entry=False` — kernel modules

A kernel module is a device-side function that entries and other modules call
directly from their traced bodies. You cannot call `.compile()` on a module or
invoke it with `[grid, stream]`. Instead, the module is compiled and linked
together with its callers automatically.

The module uses a **C ABI** — only `pto.ptr(...)` and PTO scalars can cross
the function boundary. `Tile`, `TensorView`, and `PartitionTensorView` stay
inside the caller; the module receives raw pointers and scalar metadata.

```python
@pto.jit(entry=False)
def process_tile(
    a_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    b_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    o_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    rows: pto.i32,
    cols: pto.i32,
):
    # ... SIMD/SIMT/Cube operations on the local tiles ...
    return

@pto.jit(target="a5", entry=True)
def my_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    # ... allocate tiles, load data ...
    process_tile(a_tile.as_ptr(), b_tile.as_ptr(), o_tile.as_ptr(), 1, cols)
    # ... store results ...
```

#### `mode`: auto vs explicit

`mode` chooses how you write the kernel body:

- `mode="auto"` (the default) keeps you at **tile level**. You write
  `tile.load`, `tile.store`, `tile.add` — the compiler handles everything
  below the tile abstraction: staging, instruction ordering, and
  synchronization. This is the right choice for most kernels.
- `mode="explicit"` gives you **micro-instruction control** on top of
  everything `auto` offers. You keep tiles and Tile Ops, but you can also
  write MTE DMA instructions (`mte_load`, `mte_store`), place synchronization
  barriers by hand, and work with raw pointers — useful when you need to
  hand-tune instruction schedules or overlap DMA with compute.

For native launch builds, `mode` also selects the default PTOAS build policy:
`mode="auto"` keeps the PTOAS default build level and enables sync insertion,
while `mode="explicit"` uses `--pto-level=level3` and leaves synchronization
under user control by default.

#### `backend`: VPTO vs EmitC

`backend` chooses which compiler pipeline builds your kernel:

- `backend="vpto"` (the default) is the native VPTO path. It works with both
  `mode="auto"` and `mode="explicit"`. Use this unless you specifically need
  C++ codegen.
- `backend="emitc"` generates C++ through the EmitC pipeline. It only works
  with `mode="auto"` — if you try `mode="explicit"` with it, PTODSL raises
  an error at decoration time.

You can freely mix backends across modules: an `emitc` entry can call a
`vpto` module, and vice versa. The compiler handles the linking automatically.

#### How kernels and modules are compiled together

When you compile an `entry=True` kernel that calls modules, PTODSL bundles
everything into a single compilation unit. Each `@pto.jit` function is
compiled independently with its own `backend` and `mode`, then linked
together. You don't need to manage this yourself — calling a module from a
traced body is all it takes. If you need to combine kernels that were
compiled separately, use `pto.merge_jit_modules()`.

This means you can decompose a large kernel into modules without worrying
about how they'll be wired up. A pure VPTO kernel calling VPTO modules, an
EmitC entry calling VPTO modules, a VPTO entry with EmitC modules — all of
these work through the same mechanism.

In both modes, `@pto.jit` is where you allocate tiles (`alloc_tile`) and use
Tile Ops. The difference is that `explicit` additionally opens up the
micro-instruction surface — MTE ops, explicit sync, and pointer-level
control — so you can mix tile operations with hand-authored instructions in
the same kernel.

### Compute helpers — `@pto.tileop` / `@pto.simt`

These are reusable hardware-bound compute helpers:

- **Cube-kind `@pto.tileop`** consumes prepared cube-local scratch (LEFT,
  RIGHT, ACC, BIAS). It contains Cube compute such as `mad`; staging and
  writeback remain in the caller.

- **Vector-kind `@pto.tileop`** operates on vector registers (`vreg`). Typical
  operations include `vlds`, `vadd`, `vexp`, `vcmax`, and `vsts`. Vector
  registers never cross the helper boundary.

- **`@pto.simt`** is a scalar-programmable processor group that executes scalar instructions across many work-items in parallel. Typical operations: `lds`, `sts`, scalar arithmetic and comparison. Well-suited for per-element tile walks, boundary metadata, and pointwise blends.

Named helpers use `@pto.tileop` or `@pto.simt`. One-off unit scopes use
`with pto.tileop():`, `with pto.simt():`, or
`with pto.simt(dim_x, dim_y, dim_z):`; inline SIMT dimensions must be Python
integers known at trace time.

The boundary contract is strict: transient compute values do not escape, and
data crosses helper boundaries only through Tiles, permitted scalars, or the
explicit pointer ABI of `@pto.simt`.

## 1.3 Tracing execution model

PTODSL uses a **tracing** compilation model. When you call `kernel.compile(...)`, PTODSL executes your Python function body once to record every PTO instruction into an intermediate representation — this pass is called *tracing*. The traced IR is then lowered and optimized into device code. Once compiled, invoking `compiled[grid, stream](args...)` launches the already-built device code directly on the NPU.

This has one critical implication for how you write control flow and scalar logic:

- **Python native control flow** (`for`, `if`) is rewritten to device-side control flow by default. A `for i in range(rows)` loop records a device loop, and a runtime `if` records both branches.

- **`pto.for_` / `pto.if_`** are recorded as structured control-flow IR. They preserve loop and branch semantics into the compiler pipeline, where the PTOAS compiler may further optimize them — unrolling, folding, or keeping them as runtime control flow depending on what is known at compile time.

- **`pto.const_expr` / `pto.static_range`** keep control flow at trace time for compile-time specialization and deliberate unrolling.

- **Python scalar expressions** (`alpha * x`, `1.0 / sqrt(d)`) are evaluated at trace time and their results are baked into the IR as constants — the compiler never sees the original expression.

- **PTO scalar instructions** (`scalar.load(...)`, `scalar.max(...)`, `scalar.exp(...)`) are recorded as scalar IR and enter the compiler pipeline, where they may be constant-folded or lowered to runtime scalar operations depending on whether their inputs are compile-time known.

A simple rule of thumb: **Python control flow becomes IR by default; use explicit compile-time helpers when you want trace-time behavior. PTO scalar constructs are recorded into IR and the compiler decides.**

Chapter 5 (Control Flow) and Chapter 6 (Scalar & Pointer Operations) cover this in detail.

## 1.4 A worked example

The flash attention kernel from Section 1.2 is not just an architectural diagram — it is a complete, runnable design sketch distributed with PTODSL (`examples/flash_attention_sketch.py`). Here is how the layers map to actual code:

**Top-level `@pto.jit(entry=True, mode="explicit")` schedule** allocates tiles
for the Q block, KV block, online-softmax state (m/l/o ping-pong tiles), and
cube-local scratch. It loops over Q blocks (outer `pto.for_`) and KV blocks
(inner `pto.for_` with carry state), and uses `tile.load`/`tile.store` at the
GM boundary.

**Explicit orchestration path** stages the current K and V blocks with
`mte_load`, issues `pipe_barrier(Pipe.ALL)` at phase boundaries, then
sequences four sub-kernel calls: `qk_matmul` (cube),
`online_softmax_rows` (simd), `pv_matmul` (cube), `blend_output_rows` (simt).

**Explicit kernel orchestration** performs `mte_l1_l0a` / `mte_l1_l0b` and
`mte_l0c_ub` around the Cube helpers. The `@pto.tileop` functions themselves
contain only the `mad` compute operation.

**`@pto.tileop`** implements the online softmax update: per-row max, exp, sum,
and alpha/beta computation using vector ops (`vlds`, `vcmax`, `vexp`,
`vcadd`, `vsts`).

**`@pto.simt`** blends the old and new output accumulators with per-element
`lds`/`sts` and scalar arithmetic.

Chapter 11 walks through this example in full detail.

## 1.5 Reading guide

| If you are... | Start with... |
|---------------|---------------|
| New to PTODSL | Chapter 2 (Quick Start), then Chapter 3 (Kernel Entries & Modules) |
| Writing your first kernel | Chapter 2 → Chapter 4 (Type System) → Chapter 5 (Control Flow) |
| Looking up a specific operation | Chapters 6–10, Chapters 13–15 (organized by topic) |
| Understanding the flash attention reference | Chapter 11 |

**Chapter overview:**

| Chapter | Topic |
|---------|-------|
| 1 | Introduction (this chapter) |
| 2 | Quick Start — a minimal working kernel |
| 3 | Kernel entries, kernel modules, and helpers: `@pto.jit(entry=True/False, backend=...)`, `@pto.tileop`, `@pto.simt` |
| 4 | Type system and buffer management: scalars, tiles, views, allocation |
| 5 | Control flow: trace-time Python vs device-side `pto.for_` / `pto.if_` |
| 6 | Scalar and pointer operations |
| 7 | Data movement: tile loads/stores, DMA, vector loads/stores, cube data movement |
| 8 | Compute operations: tile-level, vector, and cube arithmetic |
| 9 | Predicate and mask operations |
| 10 | Synchronization: barriers, flags, memory fences |
| 11 | Flash attention walkthrough |
| 12 | Additional examples |
| 13 | SIMT micro-ops |
| 14 | The Virtual Micro-instruction Set (VMI): logical vector instructions, masks, and type-directed authoring |
| 15 | Special scalar memory access: scalar-pipeline loads/stores, L1-bypassing GM access, and SIMT scalar access |
