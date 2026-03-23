#!/usr/bin/env bash
# Start vLLM generation server in Docker on GPU 0+1 (Tensor Parallel = 2).
#
# Usage:
#   bash scripts/start-vllm.sh [model_path] [tp_size]
#
# Defaults:
#   model_path = models/Qwen3-8B
#   tp_size    = 2

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-models/Qwen3-8B}"
TP="${2:-2}"
IMAGE="rocm-agent-vllm"
CONTAINER="rocm-agent-vllm"
PORT=8000

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building Docker image '$IMAGE' ..."
    docker build -f docker/Dockerfile.vllm -t "$IMAGE" .
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Stopping existing container '$CONTAINER' ..."
    docker rm -f "$CONTAINER" >/dev/null
fi

echo "Starting vLLM server: model=$MODEL, TP=$TP, port=$PORT"
docker run -d --name "$CONTAINER" \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --shm-size 16g \
    -e ROCR_VISIBLE_DEVICES=0,1 \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -v "$(pwd)/$MODEL:/model" \
    -p "${PORT}:${PORT}" \
    "$IMAGE" \
    trl vllm-serve \
        --model /model \
        --tensor-parallel-size "$TP" \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 2048 \
        --enforce-eager \
        --port "$PORT"

echo "Waiting for vLLM server to be ready ..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "vLLM server ready on port $PORT"
        exit 0
    fi
    sleep 2
done
echo "ERROR: vLLM server did not start within 240s"
docker logs "$CONTAINER" --tail 30
exit 1
