#!/usr/bin/env bash
set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

# You can pass multiple roots separated by ':'
# Default: ptobc testdata + PTOAS test/samples
DEFAULT_A="$(cd "$(dirname "${BASH_SOURCE[0]}")/../testdata" && pwd)"
DEFAULT_B="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/test/samples"
TESTDATA_DIRS=${TESTDATA_DIRS:-"${DEFAULT_A}:${DEFAULT_B}"}

OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_stage9_out"}
mkdir -p "${OUT_DIR}"

failed=0
IFS=':' read -r -a roots <<< "${TESTDATA_DIRS}"
for root in "${roots[@]}"; do
  [[ -d "${root}" ]] || continue
  while IFS= read -r -d '' f; do
    base=$(basename "$f" .pto)
    # avoid name collisions across directories
    hash=$(python3 - <<PY
import hashlib,sys
print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])
PY
"$f")
    base="${base}.${hash}"

  bc1="${OUT_DIR}/${base}.ptobc"
  pto2="${OUT_DIR}/${base}.roundtrip.pto"
  bc2="${OUT_DIR}/${base}.roundtrip.ptobc"

    "${PTOBC_BIN}" encode "$f" -o "$bc1" || { echo "encode failed: $f"; failed=1; continue; }
    "${PTOBC_BIN}" decode "$bc1" -o "$pto2" || { echo "decode failed: $f"; failed=1; continue; }
    "${PTOBC_BIN}" encode "$pto2" -o "$bc2" || { echo "re-encode failed: $f"; failed=1; continue; }

    if command -v cmp >/dev/null 2>&1; then
      cmp "$bc1" "$bc2" || { echo "mismatch: $f"; failed=1; }
    fi
  done < <(find "${root}" -type f -name '*.pto' -print0)
done

exit $failed
