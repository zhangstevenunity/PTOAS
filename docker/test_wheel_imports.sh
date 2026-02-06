#!/usr/bin/env bash
# Test wheel installation by verifying imports work correctly.
#
# Usage: ./test_wheel_imports.sh
#
# This script tests that the installed wheel can import:
#   - mlir.ir
#   - mlir.dialects.pto

set -e

echo "Testing wheel imports..."

# Test in a clean directory to avoid local imports
cd /tmp

echo "Testing mlir.ir import..."
python -c "import mlir.ir; print('mlir.ir imported successfully')"

echo "Testing pto dialect import..."
python -c "from mlir.dialects import pto; print('pto dialect imported successfully')"

echo "All wheel import tests passed!"
