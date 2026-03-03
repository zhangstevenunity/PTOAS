# AI Instructions for PTOAS

All AI instructions for the PTOAS project are centralized in the `.claude/` directory.

Please refer to:

- **`.claude/CLAUDE.md`** - Main AI assistant rules and entry point
- **`.claude/rules/`** - Development principles and conventions

When changing public-facing behavior (dialect ops, passes, CLI flags, codegen),
ensure cross-layer consistency (ODS/IR verifiers, lowering/codegen, Python bindings, docs, tests).

