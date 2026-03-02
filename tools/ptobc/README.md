# ptobc — PTO Bytecode encoder/decoder (C++ / MLIR)

This tool implements **PTO-BC v0** encoding/decoding as specified in:

- `docs/bytecode/pto-bc.md`

Design goals:
- Reuse MLIR infrastructure for parsing `.pto` and (optionally) printing.
- Provide a single standalone binary.

## Build

`ptobc` is built **in-tree** as part of PTOAS.

```bash
# Configure PTOAS normally (LLVM_DIR / MLIR_DIR as you already do for PTOAS)
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=... -DMLIR_DIR=... \
  -DCMAKE_BUILD_TYPE=Release

ninja -C build ptobc ptoas
```

## Usage

Encode:

```bash
build/bin/ptobc encode input.pto -o out.ptobc
```

Decode:

```bash
build/bin/ptobc decode out.ptobc -o out.pto
```

## DebugInfo

`DEBUGINFO` is optional in PTO-BC v0.

- Emit DebugInfo during **encode** (ValueNames + OpLocations when source IR has `FileLineColLoc`):

```bash
PTOBC_EMIT_DEBUGINFO=1 build/bin/ptobc encode input.pto -o out.ptobc
```

- Print `loc(...)` during **decode** (parseable form):

```bash
PTOBC_PRINT_LOC=1 build/bin/ptobc decode out.ptobc -o out.pto
```

Notes:
- The canonical printer strips `loc(unknown)` to avoid noise.
- Float constants are printed in hex bitpattern form (`0x... : f32/f16/f64`) to guarantee lossless round-trip.
