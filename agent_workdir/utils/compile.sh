#!/bin/bash
ARCH=${PYTORCH_ROCM_ARCH:-gfx1100}

START_TIME=$(date +%s.%N)

python3 -m utils.compile --arch "$ARCH"
EXIT_CODE=$?

END_TIME=$(date +%s.%N)
ELAPSED=$(awk "BEGIN {printf \"%.2f\", $END_TIME - $START_TIME}")
echo "[TIME] Compilation took ${ELAPSED}s (arch=$ARCH)"

exit $EXIT_CODE
