#!/usr/bin/env python3
import re
import sys
from pathlib import Path


def parse_td_mnemonics(td_path: Path):
    td = td_path.read_text(encoding="utf-8", errors="ignore")
    # Heuristic: collect all `let mnemonic = "..."` occurrences.
    mns = set(re.findall(r"\bmnemonic\s*=\s*\"([^\"]+)\"", td))
    # Prefix with dialect.
    return {f"pto.{m}" for m in mns}


def parse_header_names(h_path: Path):
    txt = h_path.read_text(encoding="utf-8", errors="ignore")
    # Matches: {0x0000, "pto.get_block_idx", ...}
    return set(re.findall(r"\{0x[0-9A-Fa-f]+,\s*\"([^\"]+)\"", txt))


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <PTOOps.td> <ptobc_opcodes_v0.h>", file=sys.stderr)
        return 2

    td_path = Path(sys.argv[1])
    h_path = Path(sys.argv[2])

    td_ops = parse_td_mnemonics(td_path)
    hdr_ops = parse_header_names(h_path)

    missing = sorted(op for op in td_ops if op not in hdr_ops)
    if missing:
        print("ERROR: ptobc v0 opcode table is missing ops present in PTOOps.td:")
        for op in missing:
            print(f"  - {op}")
        print("\nFix: extend docs/bytecode/tools/gen_v0_tables.py (or table source) and regenerate ptobc_opcodes_v0.h")
        return 1

    print(f"OK: opcode coverage check passed (PTOOps.td ops={len(td_ops)}, table ops={len(hdr_ops)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
