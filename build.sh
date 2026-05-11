#!/usr/bin/env bash
set -e

# ── Cari nvcc ─────────────────────────────────────────────────────────────────
find_nvcc() {
    # 1. PATH
    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc"
        return
    fi
    # 2. Lokasi umum
    for p in /usr/local/cuda/bin/nvcc \
             /usr/local/cuda-*/bin/nvcc \
             /opt/cuda/bin/nvcc; do
        if [ -x "$p" ]; then
            echo "$p"
            return
        fi
    done
    echo ""
}

NVCC=$(find_nvcc)

if [ -z "$NVCC" ]; then
    echo "nvcc tidak ditemukan. Install CUDA toolkit dulu..."
    echo ""

    # Deteksi CUDA runtime version
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | awk -F. '{print $1}')
    echo "Driver version: $CUDA_VER"

    if command -v apt-get >/dev/null 2>&1; then
        echo "Mencoba install via apt..."
        apt-get update -qq
        # Coba install cuda-nvcc package (lebih kecil dari full toolkit)
        apt-get install -y cuda-nvcc-12-9 2>/dev/null \
            || apt-get install -y cuda-nvcc-12-8 2>/dev/null \
            || apt-get install -y cuda-toolkit 2>/dev/null \
            || apt-get install -y nvidia-cuda-toolkit 2>/dev/null \
            || {
                echo ""
                echo "Auto-install gagal. Install manual:"
                echo "  apt-get update && apt-get install -y cuda-toolkit"
                echo "atau download dari https://developer.nvidia.com/cuda-downloads"
                exit 1
            }
        NVCC=$(find_nvcc)
    fi

    if [ -z "$NVCC" ]; then
        echo "Tetap tidak ketemu nvcc setelah install."
        exit 1
    fi
fi

echo "Pakai nvcc: $NVCC"

# Tambahkan ke PATH untuk shared libs
CUDA_DIR=$(dirname "$(dirname "$NVCC")")
export PATH="$CUDA_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:$LD_LIBRARY_PATH"

# ── Deteksi compute capability ────────────────────────────────────────────────
COMPUTE=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
if [ -z "$COMPUTE" ]; then
    COMPUTE="86"
    echo "GPU tidak terdeteksi, pakai default sm_${COMPUTE} (Ampere)."
else
    echo "GPU compute capability: ${COMPUTE} → sm_${COMPUTE}"
fi

# Cek apakah nvcc support arch ini
NVCC_VER=$("$NVCC" --version | grep -oP 'release \K[0-9.]+' | head -1)
echo "nvcc version: $NVCC_VER"

# Fallback ke sm_89 (Ada Lovelace) kalau sm_120 (Blackwell) gak didukung
COMPILE_ARCH="sm_${COMPUTE}"
echo "Compiling miner_gpu.cu untuk ${COMPILE_ARCH}..."

if ! "$NVCC" -O3 -arch=${COMPILE_ARCH} miner_gpu.cu -o miner_gpu 2>&1; then
    echo ""
    echo "Compile dengan ${COMPILE_ARCH} gagal, coba fallback sm_89..."
    "$NVCC" -O3 -arch=sm_89 miner_gpu.cu -o miner_gpu
fi

echo ""
echo "Build selesai! ./miner_gpu"
echo "Jalankan miner: npm start"
