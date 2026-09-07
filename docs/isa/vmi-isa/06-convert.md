# 6. Convert

> **Category:** B (`vcvt`), A (`vinterpret_cast`).
> **Mask:** `Pg` (`vcvt`), none (`vinterpret_cast`).
>
> One logical `vcvt` whose target dtype IS the layout. `pto.as` expands it into
> the dtype-specific cast chain + part/width staging + matching store
> distribution. The author never spells `EVEN`/`ODD`, `P0`–`P3`, `PK`/`UNPK`,
> or `VL/2` addresses.


---

## `pto.vmi.vcvt`

- **semantics:** Unified elementwise type conversion. The conversion direction
  is derived from the source and destination element types; the verifier
  dispatches to one of seven kinds:

  1. **FpWiden** — `fp → fp`, `|dst| > |src|` (e.g. `f16 → f32`,
     `bf16 → f32`, `fp8_e4m3 → f16`, `f4x2 → bf16x2`).

  2. **FpNarrow** — `fp → fp`, `|dst| < |src|` (e.g. `f32 → f16`,
     `f32 → bf16`, `f32 → fp8_e4m3`, `bf16x2 → f4x2`). Same-width `fp → fp`
     (`|dst| == |src|`, e.g. `bf16 → f16`, `f16 → bf16`).

  3. **FpToSi** — `fp → signed int`. Supported pairs follow the contract
     table `lookupVMIFpToSiContract`: `f32→si32`, `f16→si16`, `f32→si16`,
     `f16→si8`, `f16→si32` (nosat), `bf16→si32`.

  4. **FpToUi** — `fp → unsigned int`. Supported pairs follow the contract
     table `lookupVMIFpToUIContract`: currently `f16→u8`.

  5. **SiToFp** — `signed int → fp` (e.g. `si32 → f32`, `si8 → f16`).

  6. **IntWiden** — `int → int`, `|dst| > |src|`.

  7. **IntNarrow** — `int → int`, `|dst| < |src|`.

- **syntax:**
  ```mlir
  %r = pto.vmi.vcvt %src {rounding = "H", saturate = "SAT"} : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Converted vector (same `L`, different `T`) |

- **attributes:**

  | Attribute | Values | Valid for | Description |
  |---|---|---|---|
  | `rounding` | `"R"` (nearest-even), `"A"` (away-from-zero), `"H"` (half-up), `"Z"` (toward-zero); for the `bf16x2→f4x2` contract pair the allowed set is `"R"`,`"A"`,`"F"` (floor), `"C"` (ceil), `"Z"` (toward-zero) — `"H"` is **rejected** | fp narrowing | Rounding mode |
  | `saturate` | `"SAT"`, `"NOSAT"` | required for fp-narrow / int-narrow; for fp→si / fp→ui the requirement follows the vcvt contract's `requiresSat` (e.g. `f16→si8` required, `f16→si32` **forbidden** — no overflow possible; same-width `bf16→f16` required, same-width `f16→bf16` **forbidden**); the `bf16x2→f4x2` narrow has `requiresSat=false` — any `saturate` is **forbidden**; `si32→si8` int-narrow accepts only `"NOSAT"` | For signed destinations, `SAT` clamps to `[min, max]`; for unsigned or signless destinations, it clamps to `[0, max]`. `NOSAT` performs a direct bit truncation of the result representation. |

- **datatypes:** Source and destination from `{f32, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i16, i8, si32, si16, si8, ui32, ui16, ui8}`; packed carrier types `{!pto.bf16x2, !pto.f4E1M2x2, !pto.f4E2M1x2}` for the bf16x2↔f4x2 fp-to-fp pair (see contract `lookupVMIFpToFpContract`). `bf16x2` is **conversion-only** — it may not appear as a compute element type (`vfadd`/`vfmul`/`vcmp`/...). Signless `iN` is treated as unsigned; use `siN` for signed conversion semantics.
- **lowering to `pto.mi`:**

  | Conversion | Physical lowering | `#mi` | `dep` |
  |---|---|---|---|
  | 16↔32 (radix-2) | `2K × vcvt EVEN/ODD` + predicate `ppack`/`punpack` companion | `2K` | `2` |
  | 8↔32 (radix-4) | widen: `UNPK_B8` + `vintlv` + `vcvt P0` + `punpack`; narrow: `PK4_B32` store (or `vselr` gather) + `ppack` | `2–3` | `2–3` |
  | f32→fp8 quant | `1 cast` + `PK4_B32` | `K` | `1` |
  | f32→int8 quant | 3-stage cast + `PK4_B32` | `~3K` | `3` |
  | fp↔fp same-width (`bf16→f16`, `f16→bf16`) | `K × vcvt` (1:1, no part) | `K` | `1` |
  | fp→si / fp→ui | per contract pair: same-width 1:1, widen EVEN/ODD, narrow EVEN/ODD+Vor | `K`–`~3K` | `2`–`3` |
  | int↔int (same width) | `K × vtrc` or `K × vcvt` | `K` | `1` |
  | `bf16x2→f4x2` narrow (32→8) | source viewed as raw `bf16` lanes (2 bf16/bf16x2); `vcvt{P0}` 1:1, `rnd` set, **no sat**; reuse prior pairing `vbitcast` when present | `K` | `1` |
  | `f4x2→bf16x2` widen (8→32) | `vcvt{P0}` produces `bf16` lanes; result-side `vbitcast` reinterprets them as `bf16x2`; no rnd, no sat | `K` | `1` |

- **example:**
  ```mlir
  // fp16 → fp32 widen (radix-2, produces parity EVEN/ODD)
  %w = pto.vmi.vcvt %a
      : !pto.vmi.vreg<128×f16>
      -> !pto.vmi.vreg<128×f32>
  // → pto.as: 2 × pto.vcvt EVEN/ODD + ppack (parity companion)

  // fp32 → fp16 narrow with half-up rounding
  %n = pto.vmi.vcvt %y {rounding = "H", saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×f16>

  // ui8 -> i16 unsigned extension
  %z = pto.vmi.vcvt %a
      : !pto.vmi.vreg<256×ui8> -> !pto.vmi.vreg<256×i16>

  // f32 → fp8 quantized narrow (saturate required)
  %q = pto.vmi.vcvt %s {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×fp8_e4m3>

  // i32 → i8 int-narrow without saturation (wrap on overflow)
  %t = pto.vmi.vcvt %v {saturate = "NOSAT"}
      : !pto.vmi.vreg<64×i32> -> !pto.vmi.vreg<64×i8>

  // f32 → si32 fp-to-si (saturate required)
  %r = pto.vmi.vcvt %x {saturate = "SAT"}
      : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×si32>

  // bf16 → f16 same-width fp-to-fp (VPTO contract pair, routed via FpNarrow;
  // saturate required)
  %h = pto.vmi.vcvt %g {saturate = "SAT"}
      : !pto.vmi.vreg<128×bf16> -> !pto.vmi.vreg<128×f16>

  // f16 → u8 fp-to-ui (unsigned; contract pair, saturate required)
  %u = pto.vmi.vcvt %x {saturate = "SAT"}
      : !pto.vmi.vreg<128×f16> -> !pto.vmi.vreg<128×ui8>

  // bf16x2 → f4x2 quantized narrow (rounding required; saturate forbidden;
  // bf16x2 arrives via a physical-noop vinterpret_cast pairing of 2 bf16 lanes)
  %pair = pto.vmi.vinterpret_cast %b
      : !pto.vmi.vreg<128×bf16> -> !pto.vmi.vreg<64×!pto.bf16x2>
  %q4 = pto.vmi.vcvt %pair {rounding = "R"}
      : !pto.vmi.vreg<64×!pto.bf16x2> -> !pto.vmi.vreg<64×!pto.f4E1M2x2>

  // f4x2 → bf16x2 dequant widen (no rounding, no saturate; bf16x2 is the
  // only legal bf16 carrier for f4 dequant; bare f4x2→bf16 is rejected)
  %d = pto.vmi.vcvt %f4
      : !pto.vmi.vreg<64×!pto.f4E1M2x2> -> !pto.vmi.vreg<64×!pto.bf16x2>
  ```

- **notes:**
  - `vcvt` **does not change lane count** — `src.L == dst.L` always. The
    physical register count `K` changes because `bitwidth(T)` changes.
  - The `part`/`parity`/`width` axes are lowering-only; the user never writes
    `EVEN`/`ODD`/`P0..P3`.
  - Radix-4 (8↔32) is **not** a stacked predicate chain and **not** a UB
    roundtrip; the 1↔4 lane spread rides data load/store distribution
    (`UNPK_B*`/`PK4_B32`) or a `vselr` byte-gather.
  - `bf16x2` is **conversion-only**: it is rejected as an element type by all
    compute verifiers (`vfadd`/`vfmul`/`vfma`/`vcmp`/`vcmps`/...). The only
    way to produce/consume `bf16x2` is via `vcvt` against `f4x2`, or a
    bit-conserving `vinterpret_cast` against `bf16` lanes.
  - The `bf16x2↔f4x2` pair is the only f4 conversion path exposed at VMI. The
    physical `pto.vcvt` consumes/produces raw `bf16` lanes; the `bf16x2`
    packaging is a `vbitcast` view inserted by lowering (`vinterpret_cast`
    from `128×bf16` to `64×!pto.bf16x2` is a physical no-op pairing).


---

## `pto.vmi.vinterpret_cast`

- **semantics:** Bitwise reinterpretation of a vector register — same bits,
  different element type. No data movement. The lane count may change so long
  as the total number of bits is conserved.

  ```c
  // Same bits, reinterpreted element-by-element
  memcpy(&dst, &src, L * sizeof(T_src));
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %src : !pto.vmi.vreg<L×T_src> -> !pto.vmi.vreg<L×T_dst>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T_src>` | Source vector |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T_dst>` | Bit-reinterpreted vector |

- **attributes:** *(none)*
- **datatypes:** Any `T_src`, `T_dst` (including packed PTO types `!pto.bf16x2`, `!pto.f4E1M2x2`, `!pto.f4E2M1x2`) with `L · bitwidth(T_src) == L · bitwidth(T_dst)`
- **lowering to `pto.mi`:**
  ```
  K × pto.vbitcast (or no-op if same physical layout)
  ```
  `#mi = 0` or `K`, `dep = 0` or `1`.

- **notes:**
  - **Category A** — layout-transparent, no new axis produced.
  - This is **not** `vcvt` — no dtype cast chain, no `part`/`parity`/`width`
    axis, no `[pmode]`.
  - The user must ensure semantic legality (e.g., `f32` → `i32` bitcast is
    valid; `f32` → `f16` is not — use `vcvt` for that).

- **example:**
  ```mlir
  %r = pto.vmi.vinterpret_cast %a : !pto.vmi.vreg<64×f32> -> !pto.vmi.vreg<64×i32>
  ```
