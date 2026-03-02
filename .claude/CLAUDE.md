# AI Assistant Rules for PTOAS

This directory contains project-specific rules for AI-assisted development.

## Scope

When you change any user-visible behavior, keep these layers synchronized:

1. **ODS / Dialect definitions**: `include/PTO/IR/*.td`
2. **C++ implementation & verifiers**: `lib/PTO/IR`, `lib/PTO/Transforms`
3. **CLI / tool behavior**: `tools/ptoas`
4. **Python bindings / samples** (if affected): `python/`, `test/samples`
5. **Docs**: `README.md`, `docs/`
6. **Tests**: `test/`

## Rules

See `.claude/rules/` for specific guidance:

- `cross-layer-sync.md`
- `testing-and-examples.md`

