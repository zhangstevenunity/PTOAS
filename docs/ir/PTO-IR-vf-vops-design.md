# PTO IR: VF/VOPS (vtile) design notes

This document is **restored from OpenClaw session logs** (jsonl) and recent chat decisions.
It records the intended IR surface syntax and canonicalization expectations for the VF/VOPS layer.

## Types

- `!pto.vtile<LANESxELEM>`
  - example: `!pto.vtile<64xf32>`
- `!pto.uscalar<ELEM>`
  - example: `!pto.uscalar<f32>`
- `!pto.preg`

## Target config

Attach `pto.target_config` on module or function:

```mlir
module attributes {
  pto.target_config = #pto.target_config<arch=a3, isa="kirin9030", repeat_bytes=256, block_bytes=32, caps={}>
} {
  func.func @k() { return }
}
```

## Core ops

- `pto.vf.scope { ... }`
- Predication:
  - `pto.vpred.all`
  - `pto.vpred.tail %count`
- Loads/stores:
  - `pto.vload %tile, %row, %col, %pred`
  - `pto.vstore %tile, %row, %col, %value, %pred`
  - `pto.vload_tail %tile, %row, %col, %count`
  - `pto.vstore_tail %tile, %row, %col, %count, %value`

## Canonicalization (pass: `-pto-canonicalize-vops`)

- `vload/vstore` with `vpred.tail(count)` should be rewritten to `vload_tail/vstore_tail`.
- If an operand is produced by `vload_tail(count)`, downstream binops/stores should use `vpred.tail(count)` / `vstore_tail`.
- If `count == lanes` (constant), tail ops may be simplified to non-tail ops.
- Conservative loop-invariant hoisting may move pure pto ops that do not depend on the induction variable out of a `scf.for`.
