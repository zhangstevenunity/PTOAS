#!/usr/bin/env bash
set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
PTOAS_BIN=${PTOAS_BIN:-}

if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi
if [[ -z "${PTOAS_BIN}" ]]; then
  echo "error: PTOAS_BIN not set" >&2
  exit 2
fi

TESTDATA_DIR=${TESTDATA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../testdata" && pwd)"}
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_ptoas_smoke_out"}
mkdir -p "${OUT_DIR}"

in_pto="${TESTDATA_DIR}/add_static_multicore.pto"
if [[ ! -f "${in_pto}" ]]; then
  echo "error: missing test input: ${in_pto}" >&2
  exit 2
fi

bc="${OUT_DIR}/in.ptobc"
cpp="${OUT_DIR}/out.cpp"

"${PTOBC_BIN}" encode "${in_pto}" -o "${bc}"
"${PTOAS_BIN}" "${bc}" -o "${cpp}"

test -s "${cpp}"
