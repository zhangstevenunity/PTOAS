#!/usr/bin/env bash
set -euo pipefail

RUN_MODE="@RUN_MODE@"
SOC_VERSION="@SOC_VERSION@"
GOLDEN_MODE="${GOLDEN_MODE:-npu}"  # sim|npu|skip
BUILD_DIR="${BUILD_DIR:-build}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

# Best-effort resolve PTO_ISA_ROOT for generated CMakeLists.txt.
if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
  search_dir="${ROOT_DIR}"
  for _ in {1..8}; do
    if [[ -d "${search_dir}/pto-isa/include" && -d "${search_dir}/pto-isa/tests/common" ]]; then
      PTO_ISA_ROOT="${search_dir}/pto-isa"
      break
    fi
    if [[ "${search_dir}" == "/" ]]; then
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  export PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
fi

# Best-effort load Ascend/CANN environment (toolchains + runtime). Be careful with set -euo pipefail.
if [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  echo "[INFO] Sourcing /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  set +e
  set +u
  set +o pipefail
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -o pipefail
  set -u
  set -e
fi

# Improve runtime linking robustness.
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
fi

LD_LIBRARY_PATH_NPU="${LD_LIBRARY_PATH:-}"
LD_LIBRARY_PATH_SIM="${LD_LIBRARY_PATH_NPU}"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  SIM_SOC_VERSION="${SOC_VERSION}"
  if [[ "${SOC_VERSION}" == "Ascend910" ]]; then
    if [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910A/lib" ]]; then
      SIM_SOC_VERSION="Ascend910A"
    elif [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910ProA/lib" ]]; then
      SIM_SOC_VERSION="Ascend910ProA"
    fi
  fi

  for d in \
    "${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SIM_SOC_VERSION}/lib" \
    "${ASCEND_HOME_PATH}/simulator/${SIM_SOC_VERSION}/lib" \
    "${ASCEND_HOME_PATH}/tools/simulator/${SIM_SOC_VERSION}/lib"; do
    [[ -d "$d" ]] && LD_LIBRARY_PATH_SIM="$d:${LD_LIBRARY_PATH_SIM}"
  done
fi

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
ENABLE_SIM_GOLDEN="OFF"
[[ "${GOLDEN_MODE}" == "sim" ]] && ENABLE_SIM_GOLDEN="ON"
if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
  cmake -DSOC_VERSION="${SIM_SOC_VERSION:-${SOC_VERSION}}" -DENABLE_SIM_GOLDEN="${ENABLE_SIM_GOLDEN}" -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" ..
else
  cmake -DSOC_VERSION="${SIM_SOC_VERSION:-${SOC_VERSION}}" -DENABLE_SIM_GOLDEN="${ENABLE_SIM_GOLDEN}" ..
fi
make -j

cd "${ROOT_DIR}"

copy_outputs_as_golden() {
  if [[ -f "${ROOT_DIR}/outputs.txt" ]]; then
    while IFS= read -r name; do
      [[ -n "${name}" ]] || continue
      cp -f "${ROOT_DIR}/${name}.bin" "${ROOT_DIR}/golden_${name}.bin"
    done < "${ROOT_DIR}/outputs.txt"
    return 0
  fi
  # Fallback: copy every .bin (best-effort).
  for f in "${ROOT_DIR}"/*.bin; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    cp -f "$f" "${ROOT_DIR}/golden_${base}"
  done
}

case "${GOLDEN_MODE}" in
  sim)
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SIM}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@_sim"
    copy_outputs_as_golden
    if [[ "${RUN_MODE}" == "npu" ]]; then
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    fi
    COMPARE_STRICT=1 python3 "${ROOT_DIR}/compare.py"
    ;;
  npu)
    if [[ "${RUN_MODE}" != "npu" ]]; then
      echo "[ERROR] GOLDEN_MODE=npu requires RUN_MODE=npu" >&2
      exit 2
    fi
    python3 "${ROOT_DIR}/golden.py"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    copy_outputs_as_golden
    python3 "${ROOT_DIR}/golden.py"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    COMPARE_STRICT=1 python3 "${ROOT_DIR}/compare.py"
    ;;
  skip)
    if [[ "${RUN_MODE}" == "npu" ]]; then
      python3 "${ROOT_DIR}/golden.py"
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    fi
    echo "[WARN] compare skipped (GOLDEN_MODE=skip)"
    ;;
  *)
    echo "[ERROR] Unknown GOLDEN_MODE=${GOLDEN_MODE} (expected: sim|npu|skip)" >&2
    exit 2
    ;;
esac
