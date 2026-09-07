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
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

PTOAS_BIN=${PTOAS_BIN:-}
if [[ -z "${PTOAS_BIN}" ]]; then
  echo "error: PTOAS_BIN not set" >&2
  exit 2
fi

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-}
if [[ -z "${PYTHON_EXECUTABLE}" ]]; then
  echo "error: PYTHON_EXECUTABLE not set" >&2
  exit 2
fi

TESTDATA_DIR=${TESTDATA_DIR:-}
if [[ -z "${TESTDATA_DIR}" ]]; then
  echo "error: TESTDATA_DIR not set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/tstore_fp_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_tstore_fp_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/tstore_fp_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/tstore_fp_v0_roundtrip.roundtrip.pto"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"

"${PYTHON_EXECUTABLE}" - <<'PY' "${BC}"
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()
if b"\x66\x10" not in data:
    raise SystemExit("missing legacy tstore_fp opcode encoding")
PY

grep -F "pto.tstore ins(" "${ROUNDTRIP}" >/dev/null
grep -F " fp " "${ROUNDTRIP}" >/dev/null
grep -F "pto.partition_view" "${ROUNDTRIP}" |
  grep -F ": !pto.partition_tensor_view<32x32xi8>" >/dev/null
grep -F "!pto.partition_tensor_view<16x32xi8>" "${ROUNDTRIP}" >/dev/null
grep -F "!pto.partition_tensor_view<32x32xi8>" "${ROUNDTRIP}" >/dev/null

"${PTOAS_BIN}" --pto-arch=a3 --emit-pto-ir "${ROUNDTRIP}" -o /dev/null
