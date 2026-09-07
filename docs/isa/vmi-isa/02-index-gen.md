# 2. Index-gen

> **Category:** A. **Mask:** none.
>
> Index materialization. Produces an index vector; the single physical reg
> backing is replicate-read until a Category B/C edge needs the expanded form.


---

## `pto.vmi.vci`

- **semantics:** Generate a per-lane index/counter vector from a single scalar base such as `[base, base±1, base±2, ...]`, lane `i` gets `base + i` (ASC) or `base - i` (DESC). It is the index source for `vgather`/`vscatter` offsets.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = base + (order == "ASC" ? i : -i);
  ```

  With `group=C>1`, each group of `S=L/C` lanes restarts the ramp:

  ```c
  dst[g*S + j] = base + (order == "ASC" ? j : -j);
  ```

  `group=1` is normalized to ordinary continuous `iota`, so it has exactly the
  same semantics and tail support as omitting `group`. Group-periodic iota is
  an internal contiguous-only producer; layout assignment inserts
  `ensure_layout` when a consumer requests a deinterleaved layout.

- **syntax:**
  ```mlir
  %result = pto.vmi.vci %base {order = "ASC", group = 2} : T -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `base` | scalar (`i8`/`i16`/`i32`, `f16`/`f32`) | Starting value |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Index vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `order` | `"ASC"`, `"DESC"` | `"ASC"` | Index generation direction |
  | `group` | positive integer | omitted | Number of equal groups. `1` is equivalent to omitted; values greater than one restart the ramp per group. |

- **lowering to `pto.mi`:**
  ```
  1 × pto.vci {ASC/DESC} per chunk
  ```
  `#mi = 1/chunk`, `dep = 1`.

- **datatypes:** `i8`/`i16`/`i32`, `f16`, `f32`. For every element type,
  the legal lane counts `L` of the result are `1, 2, 4, 8, 64, 128, 256`.

- **example:**
  ```mlir
  // Ascending i32 indices for a gather base
  %idx = pto.vmi.vci %c0 {order = "ASC"} : i32 -> !pto.vmi.vreg<64×i32>
  // Descending f32 ramp
  %ramp = pto.vmi.vci %c10 {order = "DESC"} : f32 -> !pto.vmi.vreg<64×f32>
  ```

- **example:**
  ```mlir
  %idx = pto.vmi.vci %base {order = "ASC"} : i32 -> !pto.vmi.vreg<64×i32>
  // → pto.as: pto.vci {order="ASC"}, one op per physical chunk
  ```
