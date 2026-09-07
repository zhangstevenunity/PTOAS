# 5. Reduce

> **Category:** B (VLane-aligned), C (unaligned sub-VLane).
> **Mask:** `Pg req` (governing mask is a required operand).
>
> Reduction ops collapse lanes into compact scalars, governed by a mask.
> `{group=C}` controls the number of sub-groups. Inactive lane behavior:
> `vcadd` treats inactive as 0; `vcmax`/`vcmin` treat inactive as `-∞`/`+∞`
> (fp) or type min/max (int).


---

## `pto.vmi.vcadd`

- **semantics:** Masked add-reduction. When `{group=C}` is absent, reduces all
  `L` active lanes to a single scalar (`V<1×T>`).

  ```c
  // Without group: full reduction to scalar
  T sum = 0;
  for (int i = 0; i < L; i++)
      if (mask[i]) sum += src[i];
  dst[0] = sum;

  // With {group=C}: per-group reduction
  int gs = L / C;  // lanes per group
  for (int g = 0; g < C; g++) {
      T sum = 0;
      for (int i = 0; i < gs; i++)
          if (mask[g*gs + i]) sum += src[g*gs + i];
      dst[g] = sum;
  }
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcadd %src, %mask {group = C, reassoc} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Source vector |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<C×T>` | Compact scalar vector (`C = 1` if no group) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | `1`, `2`, `4`, `8` | `1` (full reduce) | Number of sub-groups |
  | `reassoc` | *(unit attr)* | *(absent)* | Permit reassociation (**required** for fp sources) |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-result behavior |

- **datatypes:** full reduce — `i32`, `f16`/`f32`; grouped reduce — `i8`/`i16`/`i32`, `f16`/`f32`
- **lowering to `pto.mi`:**

  | Group / W | Category | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|---|
  | No group (`C=1`), `K=1` | B | `1 × pto.vcadd` | `1` | `1` |
  | No group, `K>1` (fold) | B | `(K-1) × vadd` + `1 × vcadd` | `K` | `K` |
  | No group, `K>1` (partial) | B | `K × vcadd` + combine | `K` | `1+⌈log₂K⌉` |
  | `group=8` (W=32B, VLane-aligned) | B | `K × pto.vcgadd` | `K` | `1` |
  | `group=2/4` (W=64B/128B aligned) | B | `(k-1) × vadd` fold + `vcgadd` | `K+k-1` | `k` |

- **example:**
  ```mlir
  // Full sum reduction (to scalar)
  %sum = pto.vmi.vcadd %x, %mask {reassoc}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 256-lane → 8 groups of 32, each VLane-aligned (W=32B)
  %sums = pto.vmi.vcadd %x, %mask {group = 8}
      : !pto.vmi.vreg<256×f16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×f16>
  ```


---

## `pto.vmi.vcmax` / `pto.vmi.vcmin`

- **semantics:** Masked max/min reduction.

  ```c
  // vcmax: inactive lanes treated as -∞
  T best = -INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = max(best, src[i]);
  dst[0] = best;

  // vcmin: inactive lanes treated as +∞
  T best = +INF;
  for (int i = 0; i < L; i++)
      if (mask[i]) best = min(best, src[i]);
  dst[0] = best;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmax %src, %mask {group = C} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<C×T>
  ```
- **operands:** Same as `vcadd` (without `reassoc`).
- **results:** Same as `vcadd`.
- **attributes:** `group`, `pmode` (same as `vcadd`, no `reassoc`).
- **datatypes:** `i8`/`i16`/`i32`, `f16`/`f32`.
- **lowering to `pto.mi`:**

  | Group / W | Physical lowering |
  |---|---|
  | No group, fold | `(K-1) × vmax` + `1 × vcmax` |
  | VLane-aligned | `K × pto.vcgmax` / `K × pto.vcgmin` |

- **example:**
  ```mlir
  // Full max reduction
  %mx = pto.vmi.vcmax %x, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<1×f32>

  // Grouped: 8-sub-group max (MX block-scale exponent pattern)
  %maxe = pto.vmi.vcmax %exp, %mask {group = 8}
      : !pto.vmi.vreg<256×ui16>, !pto.vmi.mask<256> -> !pto.vmi.vreg<8×ui16>
  ```
