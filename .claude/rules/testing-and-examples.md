# Testing and Examples Policy (PTOAS)

## Core Principle

Prefer **regression tests** over ad-hoc scripts. Keep examples user-facing.

## Where tests belong

- **IR / lowering / codegen tests**: `test/`
- **Python sample-based tests**: `test/samples/` (when the project already uses this flow)

## What to include in bug reports / tests

For pass bugs, prefer the **Before / Expected / Actual** pattern:

- Minimal input IR (or Python that prints IR)
- The exact `ptoas` invocation
- Expected property (verifier success, type preservation, addr propagation, emitted C++ compiles, etc.)

## Avoid

- Temporary scripts committed outside `test/` or `test/samples/`
- Large, non-minimized reproductions when a smaller IR can isolate the issue

