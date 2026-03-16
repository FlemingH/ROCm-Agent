#!/bin/bash
ARCH=${PYTORCH_ROCM_ARCH:-gfx1201}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

START_TIME=$(date +%s.%N)

cd "$PROJECT_ROOT" && python3 -m tools.compile --arch "$ARCH"
EXIT_CODE=$?

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")
echo "[TIME] Compilation took ${ELAPSED}s (arch=$ARCH)"

exit $EXIT_CODE
