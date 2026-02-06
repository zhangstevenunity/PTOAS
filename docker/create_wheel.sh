#!/usr/bin/env bash
# Create Python wheel for ptoas.
#
# Usage: ./create_wheel.sh
#
# Required environment variables:
#   PTO_SOURCE_DIR  - Path to PTO source directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   LLVM_BUILD_DIR  - Path to LLVM build directory (for python packages location)

set -e

# Validate required environment variables
for var in PTO_SOURCE_DIR PTO_INSTALL_DIR LLVM_BUILD_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

PY_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"

echo "Creating Python wheel..."

# Copy PTO dialect files to MLIR package directory
echo "Copying PTO dialect files..."
cp "${PTO_INSTALL_DIR}/mlir/dialects/"*.py "${PY_PACKAGE_DIR}/mlir/dialects/"

# Copy setup.py to package directory
echo "Copying setup.py..."
cp "${PTO_SOURCE_DIR}/docker/setup.py" "${PY_PACKAGE_DIR}/"

# Determine Python version tag (e.g., cp311, cp312)
PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Python version tag: ${PY_VERSION}"

# Build the wheel with version-specific tag
echo "Building wheel..."
cd "${PY_PACKAGE_DIR}"
python setup.py bdist_wheel --python-tag "${PY_VERSION}"

echo "Wheel created at ${PY_PACKAGE_DIR}/dist/"
ls -la "${PY_PACKAGE_DIR}/dist/"*.whl
