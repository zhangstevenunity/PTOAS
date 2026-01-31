#!/bin/bash

# Usage: ./bundle_bin.sh <your_binary>
RAW_BINARY=$1

if [ -z "$RAW_BINARY" ] || [ ! -f "$RAW_BINARY" ]; then
    echo "Usage: $0 <binary_path>"
    exit 1
fi

# 1. Setup workspace
BINARY_NAME=$(basename "$RAW_BINARY")
BUNDLE_DIR="bundle_$BINARY_NAME"
LIB_DIR="$BUNDLE_DIR/libs"
# We hide the real binary in the libs folder to avoid name collision
APP_REAL="$LIB_DIR/$BINARY_NAME.real"

mkdir -p "$LIB_DIR"

echo "--- Step 1: Copying binary and dependencies ---"
cp "$RAW_BINARY" "$APP_REAL"

# Get absolute paths of all dependencies and copy them
ldd "$RAW_BINARY" | grep "=> /" | awk '{print $3}' | while read -r lib; do
    cp -vL "$lib" "$LIB_DIR/"
done

# 2. Copy the dynamic loader (The Interpreter)
INTERPRETER=$(ldd "$RAW_BINARY" | grep "ld-linux" | awk '{print $1}')
if [ -f "$INTERPRETER" ]; then
    cp -vL "$INTERPRETER" "$LIB_DIR/"
    LOADER_NAME=$(basename "$INTERPRETER")
else
    echo "Warning: Could not find dynamic loader. Portability might be limited."
fi

echo "--- Step 2: Creating the Launcher Script ---"
# This script becomes the new "entry point"
cat << 'EOF' > "$BUNDLE_DIR/$BINARY_NAME"
#!/bin/bash
# Get the absolute path of the directory containing this script
HERE=$(dirname "$(readlink -f "$0")")
LIB_PATH="$HERE/libs"
BINARY_NAME=$(basename "$0")

# Use the bundled loader to run the bundled binary
# We pass --library-path to ensure it looks in our local folder first
exec "$LIB_PATH/$(ls "$LIB_PATH" | grep ld-linux)" --library-path "$LIB_PATH" "$LIB_PATH/$BINARY_NAME.real" "$@"
EOF

chmod +x "$BUNDLE_DIR/$BINARY_NAME"

echo "--- Step 3: Zipping Bundle ---"
zip -r "${BUNDLE_DIR}.zip" "$BUNDLE_DIR"

echo "--- Done! ---"
echo "Package created: ${BUNDLE_DIR}.zip"
echo "To run on any server: unzip and run ./${BUNDLE_DIR}/$BINARY_NAME"