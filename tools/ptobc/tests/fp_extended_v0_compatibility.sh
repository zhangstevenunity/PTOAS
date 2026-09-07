#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
PTOAS_BIN=${PTOAS_BIN:-}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-}
TESTDATA_DIR=${TESTDATA_DIR:-}
LEGACY_PTOBC_BIN=${LEGACY_PTOBC_BIN:-}
LEGACY_PTOAS_BIN=${LEGACY_PTOAS_BIN:-}
if [[ -z "${PTOBC_BIN}" || -z "${PTOAS_BIN}" || -z "${PYTHON_EXECUTABLE}" || -z "${TESTDATA_DIR}" ]]; then
  echo "error: PTOBC_BIN, PTOAS_BIN, PYTHON_EXECUTABLE, and TESTDATA_DIR must be set" >&2
  exit 2
fi
if [[ -n "${LEGACY_PTOBC_BIN}" && -z "${LEGACY_PTOAS_BIN}" ]] ||
   [[ -z "${LEGACY_PTOBC_BIN}" && -n "${LEGACY_PTOAS_BIN}" ]]; then
  echo "error: LEGACY_PTOBC_BIN and LEGACY_PTOAS_BIN must be set together" >&2
  exit 2
fi

OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_fp_extended_v0_out"}
mkdir -p "${OUT_DIR}"

TMOV_IN="${TESTDATA_DIR}/tmov_fp_extended_v0_roundtrip.pto"
TMOV_BC="${OUT_DIR}/tmov_fp_extended_v0_roundtrip.ptobc"
TMOV_CURRENT_IR="${OUT_DIR}/tmov_fp_extended_v0.current.pto"
TMOV_CURRENT_CPP="${OUT_DIR}/tmov_fp_extended_v0.current.cpp"

"${PTOBC_BIN}" encode "${TMOV_IN}" -o "${TMOV_BC}"
"${PYTHON_EXECUTABLE}" - <<'PY' "${TMOV_BC}"
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()
if b"\x39\x10" in data:
    raise SystemExit("extended tmov fp form reused legacy opcode 0x1039")
if b"\xff\xff" not in data:
    raise SystemExit("extended tmov fp form did not use generic v0 encoding")
PY

"${PTOBC_BIN}" decode "${TMOV_BC}" -o "${TMOV_CURRENT_IR}"
grep -F "pto.tmov ins(" "${TMOV_CURRENT_IR}" >/dev/null
grep -F "accToVecMode = #pto.acc_to_vec_mode<single_mode_vec0>" "${TMOV_CURRENT_IR}" >/dev/null
grep -F "reluPreMode = #pto<relu_pre_mode normal_relu>" "${TMOV_CURRENT_IR}" >/dev/null
"${PTOAS_BIN}" --pto-arch=a5 "${TMOV_CURRENT_IR}" -o "${TMOV_CURRENT_CPP}"
grep -F "TMOV<" "${TMOV_CURRENT_CPP}" \
  | grep -F "AccToVecMode::SingleModeVec0" \
  | grep -F "ReluPreMode::NormalRelu" >/dev/null

TMOV_RESULT_IN="${TESTDATA_DIR}/tmov_fp_result_v0_reject.pto"
TMOV_RESULT_ERROR="${OUT_DIR}/tmov_fp_result_v0.stderr"
if "${PTOBC_BIN}" encode "${TMOV_RESULT_IN}" \
    -o "${OUT_DIR}/tmov_fp_result_v0.ptobc" 2>"${TMOV_RESULT_ERROR}"; then
  echo "error: result-bearing tmov fp unexpectedly encoded as PTO-BC v0" >&2
  exit 1
fi
grep -F "pto.tmov fp with a result" "${TMOV_RESULT_ERROR}" >/dev/null
grep -F "PTOAS backends do not lower the generic result-bearing form" \
  "${TMOV_RESULT_ERROR}" >/dev/null

TSTORE_EXTENDED_IN="${TESTDATA_DIR}/tstore_fp_extended_v0_reject.pto"
TSTORE_ERROR="${OUT_DIR}/tstore_fp_extended_v0.stderr"
if "${PTOBC_BIN}" encode "${TSTORE_EXTENDED_IN}" \
    -o "${OUT_DIR}/tstore_fp_extended_v0.ptobc" 2>"${TSTORE_ERROR}"; then
  echo "error: extended tstore fp form unexpectedly encoded as PTO-BC v0" >&2
  exit 1
fi
grep -F "cannot be represented safely in PTO-BC v0" "${TSTORE_ERROR}" >/dev/null
grep -F "legacy opcode 0x1066 would silently drop those semantics" "${TSTORE_ERROR}" >/dev/null

TSTORE_RESULT_IN="${TESTDATA_DIR}/tstore_fp_result_v0_reject.pto"
TSTORE_RESULT_ERROR="${OUT_DIR}/tstore_fp_result_v0.stderr"
if "${PTOBC_BIN}" encode "${TSTORE_RESULT_IN}" \
    -o "${OUT_DIR}/tstore_fp_result_v0.ptobc" 2>"${TSTORE_RESULT_ERROR}"; then
  echo "error: result-bearing tstore fp unexpectedly encoded as PTO-BC v0" >&2
  exit 1
fi
grep -F "pto.tstore fp with a result" "${TSTORE_RESULT_ERROR}" >/dev/null
grep -F "cannot be represented safely in PTO-BC v0" "${TSTORE_RESULT_ERROR}" >/dev/null

if [[ -z "${LEGACY_PTOBC_BIN}" ]]; then
  exit 0
fi

# Decode and lower with the last pre-unification PTOAS reader. This proves the
# generic tmov record retains mode/relu semantics across the version boundary.
TMOV_LEGACY_IR="${OUT_DIR}/tmov_fp_extended_v0.legacy.pto"
TMOV_LEGACY_CPP="${OUT_DIR}/tmov_fp_extended_v0.legacy.cpp"
"${LEGACY_PTOBC_BIN}" decode "${TMOV_BC}" -o "${TMOV_LEGACY_IR}"
grep -F "pto.tmov ins(" "${TMOV_LEGACY_IR}" >/dev/null
grep -F "accToVecMode = #pto.acc_to_vec_mode<single_mode_vec0>" "${TMOV_LEGACY_IR}" >/dev/null
grep -F "reluPreMode = #pto<relu_pre_mode normal_relu>" "${TMOV_LEGACY_IR}" >/dev/null
"${LEGACY_PTOAS_BIN}" --pto-arch=a5 "${TMOV_LEGACY_IR}" -o "${TMOV_LEGACY_CPP}"
grep -F "TMOV<" "${TMOV_LEGACY_CPP}" \
  | grep -F "AccToVecMode::SingleModeVec0" \
  | grep -F "ReluPreMode::NormalRelu" >/dev/null

# The simple form remains on 0x1066 and must still lower through the removed
# legacy pto.tstore_fp operation.
TSTORE_SIMPLE_BC="${OUT_DIR}/tstore_fp_simple_v0.ptobc"
TSTORE_LEGACY_IR="${OUT_DIR}/tstore_fp_simple_v0.legacy.pto"
TSTORE_LEGACY_CPP="${OUT_DIR}/tstore_fp_simple_v0.legacy.cpp"
"${PTOBC_BIN}" encode "${TESTDATA_DIR}/tstore_fp_v0_legacy_roundtrip.pto" \
  -o "${TSTORE_SIMPLE_BC}"
"${LEGACY_PTOBC_BIN}" decode "${TSTORE_SIMPLE_BC}" -o "${TSTORE_LEGACY_IR}"
grep -F "pto.tstore_fp" "${TSTORE_LEGACY_IR}" >/dev/null
"${LEGACY_PTOAS_BIN}" --pto-arch=a3 "${TSTORE_LEGACY_IR}" -o "${TSTORE_LEGACY_CPP}"
grep -F "TSTORE_FP" "${TSTORE_LEGACY_CPP}" >/dev/null
