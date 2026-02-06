#!/usr/bin/env bash
# Collect ptoas binary and its dependencies into a self-contained distribution.
#
# Usage: ./collect_ptoas_dist.sh <output_directory>
#
# Required environment variables:
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   PTO_SOURCE_DIR  - Path to PTO source directory
#
# Output structure:
#   <output_directory>/
#     ptoas           - Wrapper script that sets up LD_LIBRARY_PATH
#     bin/ptoas       - The actual ptoas binary
#     lib/*.so*       - Required shared library dependencies

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_directory>" >&2
  exit 1
fi

PTOAS_DIST_DIR="$1"

# Validate required environment variables
for var in LLVM_BUILD_DIR PTO_INSTALL_DIR PTO_SOURCE_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"
PTOAS_DEPS_DIR="${PTOAS_DIST_DIR}/lib"

if [ ! -f "$PTOAS_BIN" ]; then
  echo "Error: ptoas binary not found at $PTOAS_BIN" >&2
  exit 1
fi

# Create output directories
mkdir -p "${PTOAS_DIST_DIR}/bin" "${PTOAS_DEPS_DIR}"

# Copy ptoas binary
echo "Copying ptoas binary..."
cp "$PTOAS_BIN" "${PTOAS_DIST_DIR}/bin/"

# Collect *.so dependencies (transitive closure under /llvm-workspace)
echo "Collecting shared library dependencies..."
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

while read -r res; do
  copy_so "$res"
done < <(ldd "$PTOAS_BIN" 2>/dev/null | awk '/=> \/llvm-workspace\// {print $3}')

# Create wrapper script
echo "Creating wrapper script..."
cat > "${PTOAS_DIST_DIR}/ptoas" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${LD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/bin/ptoas" "$@"
WRAPPER_EOF
chmod +x "${PTOAS_DIST_DIR}/ptoas"

# Show collected files
echo ""
echo "=== ptoas distribution contents ==="
ls -la "${PTOAS_DIST_DIR}/"
ls -la "${PTOAS_DIST_DIR}/bin/"
SO_COUNT=$(find "${PTOAS_DEPS_DIR}" -name "*.so*" 2>/dev/null | wc -l)
echo "=== Collected .so dependencies (${SO_COUNT} files) ==="
du -sh "${PTOAS_DEPS_DIR}/"
echo ""
echo "Distribution created at: ${PTOAS_DIST_DIR}"
