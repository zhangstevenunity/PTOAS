#!/usr/bin/env bash
# Collect ptoas binary and macOS dylib dependencies into a self-contained distribution.
#
# Usage: ./collect_ptoas_dist_mac.sh <output_directory>
#
# Required environment variables:
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   PTO_SOURCE_DIR  - Path to PTO source directory
#
# Output structure:
#   <output_directory>/
#     ptoas           - Wrapper script that sets up DYLD_LIBRARY_PATH
#     bin/ptoas       - The actual ptoas binary
#     lib/*.dylib     - Required shared library dependencies

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_directory>" >&2
  exit 1
fi

PTOAS_DIST_DIR="$1"

# Validate required environment variables
for var in LLVM_BUILD_DIR PTO_INSTALL_DIR PTO_SOURCE_DIR; do
  if [ -z "${!var:-}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"
PTOAS_DEPS_DIR="${PTOAS_DIST_DIR}/lib"

if [ ! -f "$PTOAS_BIN" ]; then
  echo "Error: ptoas binary not found at $PTOAS_BIN" >&2
  exit 1
fi

mkdir -p "${PTOAS_DIST_DIR}/bin" "${PTOAS_DEPS_DIR}"
cp "$PTOAS_BIN" "${PTOAS_DIST_DIR}/bin/"

# Resolve @rpath / @loader_path / @executable_path / absolute install names.
resolve_dep_path() {
  local owner="$1"
  local dep="$2"
  local owner_dir
  owner_dir="$(dirname "$owner")"

  # macOS GitHub runners use bash 3.2; avoid mapfile for compatibility.
  local owner_rpaths=()
  local rp_line
  while IFS= read -r rp_line; do
    [ -n "$rp_line" ] && owner_rpaths+=("$rp_line")
  done < <(
    otool -l "$owner" | awk '
      $1=="cmd" && $2=="LC_RPATH" {want=1; next}
      want && $1=="path" {print $2; want=0}
    '
  )

  local dep_tail="$dep"
  if [[ "$dep" == @rpath/* ]]; then
    dep_tail="${dep#@rpath/}"
  fi

  local candidates=()
  if [[ "$dep" = /* ]]; then
    candidates+=("$dep")
  fi
  if [[ "$dep" == @loader_path/* ]]; then
    candidates+=("${owner_dir}/${dep#@loader_path/}")
  fi
  if [[ "$dep" == @executable_path/* ]]; then
    candidates+=("${PTOAS_DIST_DIR}/bin/${dep#@executable_path/}")
  fi
  if [[ "$dep" == @rpath/* ]]; then
    for rp in "${owner_rpaths[@]:-}"; do
      case "$rp" in
        @loader_path/*) rp="${owner_dir}/${rp#@loader_path/}" ;;
        @executable_path/*) rp="${PTOAS_DIST_DIR}/bin/${rp#@executable_path/}" ;;
      esac
      candidates+=("${rp}/${dep_tail}")
    done
    candidates+=(
      "${LLVM_BUILD_DIR}/lib/${dep_tail}"
      "${PTO_INSTALL_DIR}/lib/${dep_tail}"
      "${owner_dir}/${dep_tail}"
    )
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

collect_dylibs() {
  local bin="$1"
  while read -r dep; do
    [ -n "$dep" ] || continue
    local resolved
    resolved="$(resolve_dep_path "$bin" "$dep" || true)"
    if [ -z "$resolved" ]; then
      echo "WARN: unresolved dep for $bin -> $dep"
      continue
    fi

    local base
    base="$(basename "$resolved")"
    if [ ! -f "${PTOAS_DEPS_DIR}/${base}" ]; then
      cp "$resolved" "${PTOAS_DEPS_DIR}/${base}"
      install_name_tool -id "@loader_path/${base}" "${PTOAS_DEPS_DIR}/${base}" || true
      collect_dylibs "${PTOAS_DEPS_DIR}/${base}"
    fi
    install_name_tool -change "$dep" "@loader_path/../lib/${base}" "$bin" || true
  done < <(otool -L "$bin" | awk 'NR>1 {print $1}')
}

echo "Collecting dylib dependencies..."
collect_dylibs "${PTOAS_DIST_DIR}/bin/ptoas"

echo "Creating wrapper script..."
cat > "${PTOAS_DIST_DIR}/ptoas" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${DYLD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/bin/ptoas" "$@"
WRAPPER_EOF
chmod +x "${PTOAS_DIST_DIR}/ptoas"

echo ""
echo "=== ptoas distribution contents ==="
ls -la "${PTOAS_DIST_DIR}/"
ls -la "${PTOAS_DIST_DIR}/bin/"
DYLIB_COUNT=$(find "${PTOAS_DEPS_DIR}" -name "*.dylib" 2>/dev/null | wc -l)
echo "=== Collected .dylib dependencies (${DYLIB_COUNT} files) ==="
du -sh "${PTOAS_DEPS_DIR}/"
echo ""
echo "Distribution created at: ${PTOAS_DIST_DIR}"
