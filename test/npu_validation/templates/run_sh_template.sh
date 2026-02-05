#!/usr/bin/env bash
set -euo pipefail

RUN_MODE="@RUN_MODE@"
SOC_VERSION="@SOC_VERSION@"
BUILD_DIR="${BUILD_DIR:-build}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
make -j

cd "${ROOT_DIR}"
"${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"

python3 "${ROOT_DIR}/compare.py"
