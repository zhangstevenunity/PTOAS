#!/usr/bin/env bash
# Create Python wheel for ptoas.
#
# Usage: ./create_wheel.sh
#
# Required environment variables:
#   PTO_SOURCE_DIR  - Path to PTO source directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   LLVM_BUILD_DIR  - Path to LLVM build directory (for python packages location)
#
# Optional environment variables:
#   WHEEL_PLAT_NAME - Explicit wheel platform tag (for bdist_wheel --plat-name)
#   PTOAS_PYTHON_PACKAGE_VERSION - Wheel package version override

set -e

# Validate required environment variables
for var in PTO_SOURCE_DIR PTO_INSTALL_DIR LLVM_BUILD_DIR; do
  if [ -z "${!var}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

PY_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
PTOAS_PYTHON_PACKAGE_VERSION="${PTOAS_PYTHON_PACKAGE_VERSION:-${PTOAS_VERSION:-}}"
if [ -z "${PTOAS_PYTHON_PACKAGE_VERSION}" ]; then
  PTOAS_PYTHON_PACKAGE_VERSION="$(python "${PTO_SOURCE_DIR}/.github/scripts/compute_ptoas_version.py" --cmake-file "${PTO_SOURCE_DIR}/CMakeLists.txt" --mode dev)"
fi
export PTOAS_PYTHON_PACKAGE_VERSION

echo "Creating Python wheel..."
echo "Wheel package version: ${PTOAS_PYTHON_PACKAGE_VERSION}"

# Copy PTO dialect files to MLIR package directory
echo "Copying PTO dialect files..."
cp "${PTO_INSTALL_DIR}/mlir/dialects/"*.py "${PY_PACKAGE_DIR}/mlir/dialects/"

# Copy platform-specific setup.py to package directory.
# On macOS, use setup_mac.py and rename it to setup.py in the build dir.
SETUP_TEMPLATE="${PTO_SOURCE_DIR}/docker/setup.py"
if [ "$(uname -s)" = "Darwin" ] && [ -f "${PTO_SOURCE_DIR}/docker/setup_mac.py" ]; then
  SETUP_TEMPLATE="${PTO_SOURCE_DIR}/docker/setup_mac.py"
fi
echo "Copying $(basename "${SETUP_TEMPLATE}") as setup.py..."
cp "${SETUP_TEMPLATE}" "${PY_PACKAGE_DIR}/setup.py"

# Determine Python version tag (e.g., cp311, cp312)
PY_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
echo "Python version tag: ${PY_VERSION}"

# Build the wheel with version-specific tag
echo "Building wheel..."
cd "${PY_PACKAGE_DIR}"
if [ -n "${WHEEL_PLAT_NAME:-}" ]; then
  echo "Using wheel platform tag: ${WHEEL_PLAT_NAME}"
  python setup.py bdist_wheel --python-tag "${PY_VERSION}" --plat-name "${WHEEL_PLAT_NAME}"
else
  python setup.py bdist_wheel --python-tag "${PY_VERSION}"
fi

echo "Wheel created at ${PY_PACKAGE_DIR}/dist/"
ls -la "${PY_PACKAGE_DIR}/dist/"*.whl

EXPECTED_WHEEL_GLOB="${PY_PACKAGE_DIR}/dist/ptoas-${PTOAS_PYTHON_PACKAGE_VERSION}-"*.whl
if ! compgen -G "${EXPECTED_WHEEL_GLOB}" >/dev/null 2>&1; then
  echo "Error: expected wheel matching ${EXPECTED_WHEEL_GLOB}" >&2
  exit 1
fi
