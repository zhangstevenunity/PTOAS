# ptobc maintenance notes (PTOAS)

This tool encodes/decodes PTO-BC v0.

## When you change the PTO dialect / IR
If you change any of the following:
- `include/PTO/IR/PTOOps.td` (add/remove ops, rename mnemonics)
- operand counts / region structure / immediates semantics

…then you **must** update the PTO-BC v0 opcode/schema tables (regenerate `tools/ptobc/generated/ptobc_opcodes_v0.h`) and ensure tests pass.

## Required gates
Run (or rely on CI):
- `ctest -R ptobc_stage9_e2e`
- `ctest -R ptobc_to_ptoas_smoke`
- `ctest -R ptobc_opcode_coverage_check`

## Notes
- `ptobc_opcode_coverage_check` is a heuristic based on `mnemonic = "..."` occurrences.
  If PTOOps.td patterns change, update `tools/ptobc/tests/opcode_coverage_check.py`.
