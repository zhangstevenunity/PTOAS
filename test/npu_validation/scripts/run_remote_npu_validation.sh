#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-run}"         # build|run
RUN_MODE="${RUN_MODE:-npu}"   # npu|sim
SOC_VERSION="${SOC_VERSION:-Ascend910}"
GOLDEN_MODE="${GOLDEN_MODE:-npu}"  # sim|npu|skip
PTO_ISA_REPO="${PTO_ISA_REPO:-https://gitcode.com/cann/pto-isa.git}"
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-}"
DEVICE_ID="${DEVICE_ID:-0}"
SKIP_CASES="${SKIP_CASES:-}"          # comma/space separated testcase names
RUN_ONLY_CASES="${RUN_ONLY_CASES:-}"  # comma/space separated testcase names
# Run specific flaky/stress-sensitive cases multiple times (run stage only).
REPEAT_CASES="${REPEAT_CASES:-insertSync,injectSync}"  # comma/space separated testcase names
REPEAT_ITERS="${REPEAT_ITERS:-10}"                     # iteration count for REPEAT_CASES
DEFAULT_ITERS="${DEFAULT_ITERS:-1}"                    # iteration count for all other cases

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/test/npu_validation/scripts/generate_testcase.py" ]]; then
  ROOT_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../../../test/npu_validation/scripts/generate_testcase.py" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
else
  echo "ERROR: cannot locate repo root from SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi

log() { echo "[$(date +'%F %T')] $*"; }

log "=== Remote NPU Validation ==="
log "STAGE=${STAGE} RUN_MODE=${RUN_MODE} SOC_VERSION=${SOC_VERSION}"
log "GOLDEN_MODE=${GOLDEN_MODE}"
log "DEVICE_ID=${DEVICE_ID}"
log "PTO_ISA_REPO=${PTO_ISA_REPO}"
log "PTO_ISA_COMMIT=${PTO_ISA_COMMIT}"
log "REPEAT_CASES=${REPEAT_CASES} REPEAT_ITERS=${REPEAT_ITERS} DEFAULT_ITERS=${DEFAULT_ITERS}"
log "ROOT_DIR=${ROOT_DIR}"

RESULTS_TSV="${RESULTS_TSV:-${ROOT_DIR}/remote_npu_validation_results.tsv}"
# Put all generated validation projects under a single root to avoid sprinkling
# `npu_validation/` folders under every sample directory.
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/npu_validation}"

normalize_list() {
  local s="$1"
  s="${s//$'\n'/,}"
  s="${s//$'\t'/,}"
  s="${s// /,}"
  while [[ "$s" == *",,"* ]]; do
    s="${s//,,/,}"
  done
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

list_contains() {
  local list="$1"
  local item="$2"
  [[ -n "${item}" ]] || return 1
  [[ ",${list}," == *",${item},"* ]]
}

SKIP_CASES_NORM="$(normalize_list "${SKIP_CASES}")"
RUN_ONLY_CASES_NORM="$(normalize_list "${RUN_ONLY_CASES}")"
REPEAT_CASES_NORM="$(normalize_list "${REPEAT_CASES}")"

source_rc() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  log "Sourcing ${f}"
  set +e +u +o pipefail
  # shellcheck disable=SC1090
  source "$f" || true
  set -euo pipefail
  set -o pipefail
}

for f in "$HOME/.bash_profile" "$HOME/.bashrc"; do
  source_rc "$f"
done

if [[ -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  log "Sourcing /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  set +e +u +o pipefail
  # shellcheck disable=SC1091
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -euo pipefail
  set -o pipefail
fi

log "=== Tool Versions ==="
whoami || true
hostname || true
uname -a || true
python3 --version || true
cmake --version || true
make --version || true
command -v bisheng || true
bisheng --version || true

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-*; do
    [[ -d "$d" ]] || continue
    export ASCEND_HOME_PATH="$d"
    break
  done
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  log "ERROR: ASCEND_HOME_PATH is not set and cannot be auto-detected."
  exit 1
fi
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if ! command -v bisheng >/dev/null 2>&1; then
  if [[ -x "${ASCEND_HOME_PATH}/bin/bisheng" ]]; then
    export PATH="${ASCEND_HOME_PATH}/bin:${PATH}"
  fi
fi

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# Some CANN installs do not provide a simulator directory named exactly
# "Ascend910". Map it to a real directory so we can link/run camodel.
SIM_SOC_VERSION="${SOC_VERSION}"
if [[ "${SOC_VERSION}" == "Ascend910" ]]; then
  if [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910A/lib" ]]; then
    SIM_SOC_VERSION="Ascend910A"
  elif [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910ProA/lib" ]]; then
    SIM_SOC_VERSION="Ascend910ProA"
  fi
fi
log "SIM_SOC_VERSION=${SIM_SOC_VERSION}"

LD_LIBRARY_PATH_NPU="${LD_LIBRARY_PATH}"
LD_LIBRARY_PATH_SIM="${LD_LIBRARY_PATH}"
for d in \
  "${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SIM_SOC_VERSION}/lib" \
  "${ASCEND_HOME_PATH}/simulator/${SIM_SOC_VERSION}/lib" \
  "${ASCEND_HOME_PATH}/tools/simulator/${SIM_SOC_VERSION}/lib"; do
  [[ -d "$d" ]] && LD_LIBRARY_PATH_SIM="$d:${LD_LIBRARY_PATH_SIM}"
done

if [[ "${STAGE}" == "run" ]]; then
  log "=== NPU Device Check ==="
  id || true
  ls -l /dev/davinci* 2>/dev/null || true
  devnode="/dev/davinci${DEVICE_ID}"
  [[ -e "${devnode}" ]] || { log "ERROR: ${devnode} not found"; exit 1; }
  [[ -r "${devnode}" && -w "${devnode}" ]] || {
    log "ERROR: no access to ${devnode} (need HwHiAiUser group)";
    exit 1;
  }
  python3 -c "import numpy as np; print('numpy', np.__version__)" >/dev/null
fi

PTO_ISA_ROOT="${ROOT_DIR}/pto-isa"
if [[ ! -d "${PTO_ISA_ROOT}/.git" ]]; then
  log "Cloning pto-isa into ${PTO_ISA_ROOT} ..."
  git clone "${PTO_ISA_REPO}" "${PTO_ISA_ROOT}"
fi
if [[ -n "${PTO_ISA_COMMIT}" ]]; then
  log "Checking out pto-isa ${PTO_ISA_COMMIT} ..."
  git -C "${PTO_ISA_ROOT}" fetch --all --prune
  git -C "${PTO_ISA_ROOT}" checkout "${PTO_ISA_COMMIT}"
fi

status=0
ok_count=0
fail_count=0
skip_count=0
printf "testcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"
while IFS= read -r -d '' cpp; do
  # macOS tarballs may contain AppleDouble metadata files like `._foo-pto.cpp`.
  # They are not valid C++ sources; skip them.
  if [[ "$(basename "${cpp}")" == ._* ]]; then
    continue
  fi

  base="$(basename "${cpp}" .cpp)"
  testcase="${base}"
  testcase="${testcase%-pto}"
  testcase="${testcase%_pto}"

  if [[ -n "${RUN_ONLY_CASES_NORM}" ]] && ! list_contains "${RUN_ONLY_CASES_NORM}" "${testcase}"; then
    continue
  fi
  if [[ -n "${SKIP_CASES_NORM}" ]] && list_contains "${SKIP_CASES_NORM}" "${testcase}"; then
    skip_count=$((skip_count + 1))
    printf "%s\tSKIP\t%s\tlisted in SKIP_CASES\n" "${testcase}" "${STAGE}" >> "${RESULTS_TSV}"
    log "SKIP: ${testcase} (SKIP_CASES)"
    continue
  fi

  echo
  log "=== CASE: ${cpp} ==="

  case_dir="$(cd "$(dirname "${cpp}")" && pwd)"
  sample_name="$(basename "${case_dir}")"
  nv_dir="${OUTPUT_ROOT}/${sample_name}/${testcase}"
  iters="${DEFAULT_ITERS}"
  if [[ "${STAGE}" == "run" ]]; then
    if [[ -n "${REPEAT_CASES_NORM}" ]] && list_contains "${REPEAT_CASES_NORM}" "${testcase}"; then
      iters="${REPEAT_ITERS}"
    fi
  else
    iters=1
  fi

  set +e
  python3 "${ROOT_DIR}/test/npu_validation/scripts/generate_testcase.py" \
    --input "${cpp}" \
    --testcase "${testcase}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-mode "${RUN_MODE}" \
    --soc-version "${SIM_SOC_VERSION}"
  gen_rc=$?
  set -euo pipefail
  if [[ $gen_rc -ne 0 ]]; then
    status=1
    fail_count=$((fail_count + 1))
    printf "%s\tFAIL\tgen\texit=%s\n" "${testcase}" "${gen_rc}" >> "${RESULTS_TSV}"
    log "ERROR: generate_testcase failed (exit ${gen_rc}): ${testcase}"
    continue
  fi

  set +e
  (
    set -euo pipefail
    cd "${nv_dir}"
    export ACL_DEVICE_ID="${DEVICE_ID}"

    enable_sim_golden="OFF"
    [[ "${GOLDEN_MODE}" == "sim" ]] && enable_sim_golden="ON"
    cmake -S . -B ./build \
      -DSOC_VERSION="${SIM_SOC_VERSION}" \
      -DENABLE_SIM_GOLDEN="${enable_sim_golden}" \
      -DPTO_ISA_ROOT="${PTO_ISA_ROOT}"
    cmake --build ./build --parallel

    if [[ "${STAGE}" != "run" ]]; then
      log "BUILD OK: ${testcase}"
      exit 0
    fi

    # Stress loops (run stage only). Keep default to 1 for most cases.
    if ! [[ "${iters}" =~ ^[0-9]+$ ]] || [[ "${iters}" -lt 1 ]]; then
      iters=1
    fi
    if [[ "${iters}" -gt 1 ]]; then
      log "STRESS: ${testcase} iters=${iters}"
    fi

    copy_outputs_as_golden() {
      if [[ -f "./outputs.txt" ]]; then
        while IFS= read -r name; do
          [[ -n "${name}" ]] || continue
          cp -f "./${name}.bin" "./golden_${name}.bin"
        done < "./outputs.txt"
        return 0
      fi
      for f in ./*.bin; do
        [[ -f "$f" ]] || continue
        base="$(basename "$f")"
        cp -f "$f" "./golden_${base}"
      done
    }

    case "${GOLDEN_MODE}" in
      sim)
        for ((i=1; i<=iters; i++)); do
          if [[ "${iters}" -gt 1 ]]; then
            log "RUN(sim) ${testcase} iter ${i}/${iters}"
          fi
          python3 ./golden.py
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SIM}" ./build/${testcase}_sim
          copy_outputs_as_golden
          if [[ "${RUN_MODE}" == "npu" ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase}
          fi
          COMPARE_STRICT=1 python3 ./compare.py
        done
        ;;
      npu)
        if [[ "${RUN_MODE}" != "npu" ]]; then
          log "ERROR: GOLDEN_MODE=npu requires RUN_MODE=npu"
          exit 2
        fi
        for ((i=1; i<=iters; i++)); do
          if [[ "${iters}" -gt 1 ]]; then
            log "RUN(npu) ${testcase} iter ${i}/${iters}"
          fi
          python3 ./golden.py
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase}
          copy_outputs_as_golden
          python3 ./golden.py
          LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase}
          COMPARE_STRICT=1 python3 ./compare.py
        done
        ;;
      skip)
        for ((i=1; i<=iters; i++)); do
          if [[ "${iters}" -gt 1 ]]; then
            log "RUN(skip) ${testcase} iter ${i}/${iters}"
          fi
          python3 ./golden.py
          if [[ "${RUN_MODE}" == "npu" ]]; then
            LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" ./build/${testcase}
          fi
          log "WARN: compare skipped (GOLDEN_MODE=skip)"
        done
        ;;
      *)
        log "ERROR: unknown GOLDEN_MODE=${GOLDEN_MODE} (expected: sim|npu|skip)"
        exit 2
        ;;
    esac
    log "OK: ${testcase}"
  )
  case_rc=$?
  set -euo pipefail
  if ! [[ "${iters}" =~ ^[0-9]+$ ]] || [[ "${iters}" -lt 1 ]]; then
    iters=1
  fi
  info="-"
  if [[ "${STAGE}" == "run" && "${iters}" -gt 1 ]]; then
    info="repeat=${iters}"
  fi
  if [[ $case_rc -ne 0 ]]; then
    status=1
    fail_count=$((fail_count + 1))
    if [[ "${STAGE}" == "run" && "${iters}" -gt 1 ]]; then
      info="repeat=${iters} exit=${case_rc}"
    else
      info="exit=${case_rc}"
    fi
    printf "%s\tFAIL\t%s\t%s\n" "${testcase}" "${STAGE}" "${info}" >> "${RESULTS_TSV}"
    log "ERROR: testcase failed (exit ${case_rc}): ${testcase}"
  else
    ok_count=$((ok_count + 1))
    printf "%s\tOK\t%s\t%s\n" "${testcase}" "${STAGE}" "${info}" >> "${RESULTS_TSV}"
  fi
done < <(find "${ROOT_DIR}/test/samples" -type f -name '*-pto.cpp' -print0)

log "=== SUMMARY ==="
log "OK=${ok_count} FAIL=${fail_count} SKIP=${skip_count}"
log "RESULTS_TSV=${RESULTS_TSV}"

exit "${status}"
