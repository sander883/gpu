#!/usr/bin/env bash
set -e

# Auto-detect GPU compute capability
COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$COMPUTE" ]; then
    COMPUTE="86"
    echo "GPU tidak terdeteksi, pakai default sm_${COMPUTE} (Ampere)."
else
    echo "GPU compute capability: ${COMPUTE} → sm_${COMPUTE}"
fi

echo "Compiling miner_gpu.cu..."
nvcc -O3 -arch=sm_${COMPUTE} --ptxas-options=-v miner_gpu.cu -o miner_gpu

echo ""
echo "Build selesai! Jalankan miner:"
echo "  npm start"
