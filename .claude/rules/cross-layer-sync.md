# Cross-Layer Synchronization (PTOAS)

## Overview

PTOAS changes often span multiple layers. **All relevant layers must be updated together** when changing public APIs or semantics.

## Layers to keep in sync

1. **ODS / Dialect** (`include/PTO/IR/*.td`)
   - Op definitions, assembly formats, traits/interfaces, verifiers (decls)
2. **C++ IR & verifiers** (`lib/PTO/IR`)
   - `Op::verify()` logic and semantic constraints
3. **Lowering / transforms** (`lib/PTO/Transforms`)
   - Type conversions, legality, conversion patterns
   - Any new op attribute/operand must be handled through lowering
4. **CLI tool** (`tools/ptoas/ptoas.cpp`)
   - Flag parsing, validation gating (`--pto-level`, `--pto-arch`, etc.)
5. **Python bindings** (`python/`)
   - Op bindings generation inputs (TableGen wrapper `.td`), python API, samples
6. **Docs & specs** (`README.md`, `docs/`)
7. **Tests** (`test/`)

## Typical examples

### Adding/changing an op operand

- Update ODS (`*.td`): operand definition + assembly format
- Update C++ verifier (`Op::verify`) if constraints change
- Update lowering patterns (PTO → memref / EmitC) to consume/forward the operand
- Update python bindings if the op is exposed and used from Python
- Add or adjust a regression test
- Update docs/spec if user-visible

### Adding a CLI flag that changes lowering/codegen

- Implement flag parsing in `tools/ptoas/ptoas.cpp`
- Thread the flag into pass options or pass construction (avoid hidden global mutable state)
- Update docs and add a test demonstrating behavior

## Checklist

Before finishing a change:

- [ ] ODS matches actual operands/attrs and prints/parses correctly
- [ ] C++ verifier errors are actionable
- [ ] Lowering/codegen handles all new cases (including view ops like `memref.subview`)
- [ ] Python side still builds/imports (if impacted)
- [ ] Docs/specs updated
- [ ] Tests cover regression

