#!/usr/bin/env bash
# Test ptoas CLI with sample files.
#
# Usage: ./test_ptoas_cli.sh
#
# Required environment variables:
#   PTO_SOURCE_DIR  - Path to PTO source directory
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory

set -e

# Validate required environment variables
for var in PTO_SOURCE_DIR LLVM_BUILD_DIR PTO_INSTALL_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

# Setup environment
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export PYTHONPATH="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core:${PTO_INSTALL_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}"

echo "Testing ptoas CLI..."
which ptoas

# Test MatMul sample
echo "Testing MatMul sample..."
cd "${PTO_SOURCE_DIR}/test/samples/MatMul/"
python ./tmatmulk.py > ./tmatmulk.pto
ptoas ./tmatmulk.pto -o ./tmatmulk.cpp
echo "MatMul test passed"

# Test Abs sample
echo "Testing Abs sample..."
cd "${PTO_SOURCE_DIR}/test/samples/Abs/"
python ./abs.py > ./abs.pto
ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp
echo "Abs test passed"

echo "All ptoas CLI tests passed!"
