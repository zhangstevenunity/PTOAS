#!/usr/bin/env bash
set -uo pipefail   # 注意：去掉 -e，避免失败直接退出整个脚本

BASE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"

# Allow overriding tool/python explicitly:
#   PTOAS_BIN=/path/to/ptoas PYTHON_BIN=/path/to/python ./runop.sh all
PTOAS_BIN="${PTOAS_BIN:-}"
PTOBC_BIN="${PTOBC_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
PTOAS_OUT_DIR="${PTOAS_OUT_DIR:-}"
PTOAS_ENABLE_INSERT_SYNC="${PTOAS_ENABLE_INSERT_SYNC:-1}"
PTOAS_FLAGS="${PTOAS_FLAGS:-}"
PTO_PTO_DIRS="${PTO_PTO_DIRS:-InjectSync}"
ENABLE_BC=0

usage() {
  cat <<EOF
Usage:
  $0 [--enablebc] -t <name>   # e.g. -t Shls  -> run all .py in folder Shls
  $0 [--enablebc] all         # traverse every subfolder, run all .py under each
  $0 --enablebc               # alias for: $0 --enablebc all

Env:
  PTOAS_BIN   # path to ptoas executable (optional)
  PTOBC_BIN   # path to ptobc executable (optional)
  PYTHON_BIN  # python executable to run samples (optional)
  PTOAS_OUT_DIR  # where generated *.mlir/*.cpp go (optional; defaults to a temp dir)
  PTOAS_FLAGS  # extra flags passed to ptoas (e.g. --enable-insert-sync)
  PTOAS_ENABLE_INSERT_SYNC  # 1 to append --enable-insert-sync to PTOAS_FLAGS (default: 1)
  PTO_PTO_DIRS  # space-separated dirs to run .pto directly (default: InjectSync)

Flags:
  --enablebc  # enable: python -> .pto -> ptobc -> .pto -> ptoas
EOF
  exit 1
}

ucfirst() {
  local s="$1"
  local first="${s:0:1}"
  local rest="${s:1}"
  printf '%s%s\n' "$(printf '%s' "$first" | tr '[:lower:]' '[:upper:]')" "$rest"
}

lcfirst() {
  local s="$1"
  local first="${s:0:1}"
  local rest="${s:1}"
  printf '%s%s\n' "$(printf '%s' "$first" | tr '[:upper:]' '[:lower:]')" "$rest"
}

resolve_ptoas_bin() {
  if [[ -n "${PTOAS_BIN}" ]]; then
    echo "${PTOAS_BIN}"
    return 0
  fi

  # Common locations:
  # - out-of-tree build in repo: PTOAS/build/tools/ptoas/ptoas
  # - legacy layout: build/bin/ptoas
  local cand
  cand="${BASE_DIR}/../../build/tools/ptoas/ptoas"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../../../build/bin/ptoas"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v ptoas 2>/dev/null || true)"
  [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return 0; }

  echo ""
  return 1
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  local cand
  cand="$(command -v python 2>/dev/null || true)"
  [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v python3 2>/dev/null || true)"
  [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  echo ""
  return 1
}

resolve_ptobc_bin() {
  if [[ -n "${PTOBC_BIN}" ]]; then
    echo "${PTOBC_BIN}"
    return 0
  fi

  local cand
  cand="${BASE_DIR}/../../build/tools/ptobc/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../build/bin/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../../../build/bin/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v ptobc 2>/dev/null || true)"
  [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return 0; }

  echo ""
  return 1
}

process_one_dir() {
  local A="$1" # folder name (e.g. Abs)
  local out_dir="$2"
  local dir ptoas ptobc python out_subdir
  dir="${BASE_DIR}/${A}"
  out_subdir="${out_dir}/${A}"
  mkdir -p "${out_subdir}"

  ptoas="$(resolve_ptoas_bin)"
  ptobc="$(resolve_ptobc_bin)"
  python="$(resolve_python_bin)"
  local use_ptobc_roundtrip=0
  if [[ "${ENABLE_BC}" == "1" ]]; then
    use_ptobc_roundtrip=1
  fi
  local -a ptoas_flags=()
  if [[ -n "${PTOAS_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    ptoas_flags=(${PTOAS_FLAGS})
  fi
  if [[ "${PTOAS_ENABLE_INSERT_SYNC}" == "1" ]]; then
    local has_insync=0
    if ((${#ptoas_flags[@]})); then
      for f in "${ptoas_flags[@]}"; do
        if [[ "$f" == "--enable-insert-sync" ]]; then
          has_insync=1
          break
        fi
      done
    fi
    [[ $has_insync -eq 1 ]] || ptoas_flags+=(--enable-insert-sync)
  fi
  local -a ptoas_cmd_base=("$ptoas")
  if (( ${#ptoas_flags[@]} )); then
    ptoas_cmd_base+=("${ptoas_flags[@]}")
  fi

  if [[ -z "$ptoas" || ! -x "$ptoas" ]]; then
    echo -e "${A}\tFAIL\tMissing executable: PTOAS_BIN (searched common paths)"
    return 0
  fi
  if [[ -z "$python" || ! -x "$python" ]]; then
    echo -e "${A}\tFAIL\tMissing python: PYTHON_BIN (python/python3 not found)"
    return 0
  fi
  if [[ $use_ptobc_roundtrip -eq 1 ]] && [[ -z "$ptobc" || ! -x "$ptobc" ]]; then
    echo -e "${A}\tFAIL\tMissing executable: PTOBC_BIN (searched common paths)"
    return 0
  fi
  if [[ ! -d "$dir" ]]; then
    echo -e "${A}\tSKIP\tMissing dir: $dir"
    return 0
  fi

  # Run every .py file in this directory (no requirement that name matches folder).
  local f mlir ptobc_file decoded_pto cpp base overall=0
  for f in "$dir"/*.py; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f" .py)"
    local expect_fail=0
    case "$base" in
      *_invalid|*_xfail) expect_fail=1 ;;
    esac
    mlir="${out_subdir}/${base}-pto-ir.pto"
    cpp="${out_subdir}/${base}-pto.cpp"

    if ! "$python" "$f" > "$mlir"; then
      if [[ $expect_fail -eq 1 ]]; then
        echo -e "${A}(${base}.py)\tXFAIL\tpython failed as expected"
        continue
      fi
      echo -e "${A}(${base}.py)\tFAIL\tpython failed: ${base}.py"
      overall=1
      continue
    fi

    local pto_input="$mlir"
    ptobc_file="${out_subdir}/${base}.ptobc"
    decoded_pto="${out_subdir}/${base}-roundtrip.pto"
    if [[ $use_ptobc_roundtrip -eq 1 ]]; then
      if ! "$ptobc" encode "$mlir" -o "$ptobc_file" >/dev/null 2>&1; then
        if [[ $expect_fail -eq 1 ]]; then
          echo -e "${A}(${base}.py)\tXFAIL\tptobc encode failed as expected"
          continue
        fi
        echo -e "${A}(${base}.py)\tFAIL\tptobc encode failed: $(basename "$mlir")"
        overall=1
        continue
      fi
      if ! "$ptobc" decode "$ptobc_file" -o "$decoded_pto" >/dev/null 2>&1; then
        if [[ $expect_fail -eq 1 ]]; then
          echo -e "${A}(${base}.py)\tXFAIL\tptobc decode failed as expected"
          continue
        fi
        echo -e "${A}(${base}.py)\tFAIL\tptobc decode failed: $(basename "$ptobc_file")"
        overall=1
        continue
      fi
      pto_input="$decoded_pto"
    fi

    # Write output via -o to avoid mixing debug prints with generated C++.
    local -a ptoas_cmd=("${ptoas_cmd_base[@]}" "$pto_input" -o "$cpp")
    if ! "${ptoas_cmd[@]}" >/dev/null 2>&1; then
      if [[ $expect_fail -eq 1 ]]; then
        echo -e "${A}(${base}.py)\tXFAIL\tptoas failed as expected"
        continue
      fi
      echo -e "${A}(${base}.py)\tFAIL\tptoas failed: $(basename "$mlir")"
      overall=1
      continue
    fi

    if [[ $expect_fail -eq 1 ]]; then
      echo -e "${A}(${base}.py)\tFAIL\texpected failure but succeeded"
      overall=1
      continue
    fi

    # Regression guard: SubsetOp valid-shape inference must not produce 0.
    # This breaks downstream NPU compilation (e.g. vadd_pto_pingpong workspace ping/pong).
    if [[ "$base" == "vadd_pto_pingpong" ]]; then
      if grep -Fq ", 0, SLayout" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tgenerated tile has valid dim 0 (subset valid-shape bug)"
        overall=1
        continue
      fi
    fi

    # Regression guard for Issue #112:
    # `--enable-insert-sync` must not push PIPE_M -> PIPE_FIX into high event IDs
    # for the autosync tmatmulk sample, otherwise it may deadlock on Ascend NPU.
    if [[ "$base" == "tmatmulk_autosync" ]]; then
      if grep -Eq "set_flag\\(PIPE_M,[[:space:]]*PIPE_FIX,[[:space:]]*EVENT_ID[3-7]\\)" "$cpp" || \
         grep -Eq "wait_flag\\(PIPE_M,[[:space:]]*PIPE_FIX,[[:space:]]*EVENT_ID[3-7]\\)" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tdeadlock signature: PIPE_M->PIPE_FIX uses EVENT_ID[3-7]"
        overall=1
        continue
      fi
    fi

    # Regression guard for issue #117: vector mask must be reset for each
    # `pto.section.vector` region to avoid cross-kernel state leakage.
    # Use an existing sample (Complex/cv_region.py) that contains a vector section.
    if [[ "$base" == "cv_region" ]]; then
      if ! grep -Fq "#if defined(__DAV_VEC__)" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing __DAV_VEC__ guard"
        overall=1
        continue
      fi
      if ! grep -Fq "set_mask_norm();" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing set_mask_norm() reset"
        overall=1
        continue
      fi
      if ! grep -Fq "set_vector_mask(-1, -1);" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing set_vector_mask(-1, -1) reset"
        overall=1
        continue
      fi
    fi

    # Regression guard: bf16 tiles must lower to `__bf16` in Tile<> / GlobalTensor<> templates.
    if [[ "$base" == "bf16_tile" ]]; then
      if ! grep -Fq "GlobalTensor<__bf16" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tbf16 GlobalTensor element type is not __bf16"
        overall=1
        continue
      fi
      if ! grep -Eq "Tile<[^>]*, __bf16," "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tbf16 Tile element type is not __bf16"
        overall=1
        continue
      fi
    fi

    echo -e "${A}(${base}.py)\tOK\tgenerated: $(basename "$cpp")"
  done

  # Run .pto files only for allowed dirs (default: InjectSync) to avoid legacy IR.
  local allow_pto=0
  for d in ${PTO_PTO_DIRS}; do
    if [[ "$A" == "$d" ]]; then
      allow_pto=1
      break
    fi
  done

  if [[ $allow_pto -eq 1 ]]; then
    for f in "$dir"/*.pto; do
      [[ -f "$f" ]] || continue
      case "$f" in
        *-pto-ir.pto) continue ;;
      esac
      base="$(basename "$f" .pto)"
      local pto_input="$f"
      ptobc_file="${out_subdir}/${base}.ptobc"
      decoded_pto="${out_subdir}/${base}-roundtrip.pto"
      cpp="${out_subdir}/${base}.cpp"

      if [[ $use_ptobc_roundtrip -eq 1 ]]; then
        if ! "$ptobc" encode "$f" -o "$ptobc_file" >/dev/null 2>&1; then
          echo -e "${A}(${base}.pto)\tFAIL\tptobc encode failed: $(basename "$f")"
          overall=1
          continue
        fi
        if ! "$ptobc" decode "$ptobc_file" -o "$decoded_pto" >/dev/null 2>&1; then
          echo -e "${A}(${base}.pto)\tFAIL\tptobc decode failed: $(basename "$ptobc_file")"
          overall=1
          continue
        fi
        pto_input="$decoded_pto"
      fi

      local -a ptoas_cmd=("${ptoas_cmd_base[@]}" "$pto_input" -o "$cpp")
      if ! "${ptoas_cmd[@]}" >/dev/null 2>&1; then
        echo -e "${A}(${base}.pto)\tFAIL\tptoas failed: $(basename "$f")"
        overall=1
        continue
      fi

      # Regression guard: dynamic valid_shape must be preserved through lowering.
      # If `valid_col` is dynamic, PTOToEmitC must construct the Tile with a
      # runtime argument (i.e. emit `= Tile<...>(...)` instead of `Tile<...>;`).
      if [[ "$base" == "test_dynamic_valid_shape" ]]; then
        if ! grep -Fq "= Tile<TileType::Vec, float" "$cpp"; then
          echo -e "${A}(${base}.pto)\tFAIL\tmissing dynamic Tile constructor (valid_col likely dropped)"
          overall=1
          continue
        fi
      fi

      echo -e "${A}(${base}.pto)\tOK\tgenerated: $(basename "$cpp")"
    done
  fi

  return $overall
}

run_all() {
  local results tmp out_dir
  out_dir="${PTOAS_OUT_DIR}"
  if [[ -z "${out_dir}" ]]; then
    out_dir="$(mktemp -d -t ptoas.samples.XXXXXX)"
  else
    mkdir -p "${out_dir}"
  fi

  echo "PTOAS_OUT_DIR=${out_dir}"

  tmp="$(mktemp -t ptoas.runop.XXXXXX)"
  for d in "${BASE_DIR}"/*/; do
    [[ -d "$d" ]] || continue
    process_one_dir "$(basename "$d")" "$out_dir" >>"$tmp"
  done

  echo "========== SUMMARY =========="
  sort "$tmp" | awk -F'\t' '
    BEGIN { ok=0; fail=0; skip=0; }
    {
      printf "%-12s %-4s %s\n", $1, $2, $3;
      if ($2=="OK") ok++;
      else if ($2=="FAIL") fail++;
      else if ($2=="SKIP") skip++;
    }
    END {
      print "-----------------------------";
      printf "OK=%d  FAIL=%d  SKIP=%d\n", ok, fail, skip;
      print "=============================";
      exit (fail==0 ? 0 : 1);
    }'
}

# -----------------------------------------------------------------------------
# CLI flags
# -----------------------------------------------------------------------------
positional_args=()
for arg in "$@"; do
  case "$arg" in
    --enablebc) ENABLE_BC=1 ;;
    -h|--help) usage ;;
    *) positional_args+=("$arg") ;;
  esac
done
set -- "${positional_args[@]}"

if [[ "${ENABLE_BC}" == "1" ]] && [[ $# -eq 0 ]]; then
  set -- all
fi

if [[ $# -eq 1 && "$1" == "all" ]]; then
  run_all
elif [[ $# -eq 2 && "$1" == "-t" ]]; then
  A="$(ucfirst "$2")"
  out_dir="${PTOAS_OUT_DIR}"
  if [[ -z "${out_dir}" ]]; then
    out_dir="$(mktemp -d -t ptoas.samples.XXXXXX)"
  else
    mkdir -p "${out_dir}"
  fi
  echo "PTOAS_OUT_DIR=${out_dir}"
  echo "========== SUMMARY =========="
  process_one_dir "$A" "$out_dir" | awk -F'\t' '{ printf "%-12s %-4s %s\n", $1, $2, $3 }'
else
  usage
fi
