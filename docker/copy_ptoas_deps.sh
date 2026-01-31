#!/usr/bin/env bash
# Collect only *.so actually needed by ptoas (transitive closure under /llvm-workspace).
# Expects: LLVM_BUILD_DIR, PTO_INSTALL_DIR, PTOAS_DEPS_DIR, PTO_SOURCE_DIR

set -e

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"
PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"

copy_so() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  local name
  name=$(basename "$f")
  [[ -f "${PTOAS_DEPS_DIR}/${name}" ]] && return 0
  cp -n "$f" "${PTOAS_DEPS_DIR}/" 2>/dev/null || true
  while read -r res; do
    copy_so "$res"
  done < <(ldd "$f" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
}

mkdir -p "$PTOAS_DEPS_DIR"
while read -r res; do
  copy_so "$res"
done < <(ldd "$PTOAS_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')
