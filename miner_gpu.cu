/*
 * GPU Keccak-256 miner for HASH256 token
 * Usage: ./miner_gpu <challenge_hex> <difficulty_hex> [start_nonce]
 * Output: "FOUND:<nonce>" on stdout when solved
 * Hashrate info on stderr
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define ROTL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))
#define BSWAP64(x) ( \
    ((x) << 56) | (((x) & 0xFF00ULL) << 40) | (((x) & 0xFF0000ULL) << 24) | \
    (((x) & 0xFF000000ULL) << 8) | (((x) >> 8) & 0xFF000000ULL) | \
    (((x) >> 24) & 0xFF0000ULL) | (((x) >> 40) & 0xFF00ULL) | ((x) >> 56))

/* Keccak-f[1600] round constants */
__device__ __constant__ uint64_t RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

/* Rho offsets and Pi lane indices */
__device__ __constant__ int ROTC[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};
__device__ __constant__ int PILN[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

__device__ __forceinline__ void keccak_f1600(uint64_t st[25]) {
    uint64_t bc[5], t;

    #pragma unroll
    for (int round = 0; round < 24; round++) {
        /* Theta */
        #pragma unroll
        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i+5] ^ st[i+10] ^ st[i+15] ^ st[i+20];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            t = bc[(i+4)%5] ^ ROTL64(bc[(i+1)%5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5)
                st[j+i] ^= t;
        }
        /* Rho + Pi */
        t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; i++) {
            int j = PILN[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, ROTC[i]);
            t = bc[0];
        }
        /* Chi */
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            uint64_t row[5];
            #pragma unroll
            for (int i = 0; i < 5; i++) row[i] = st[j+i];
            #pragma unroll
            for (int i = 0; i < 5; i++)
                st[j+i] ^= (~row[(i+1)%5]) & row[(i+2)%5];
        }
        /* Iota */
        st[0] ^= RC[round];
    }
}

/*
 * Compute keccak256 of abi.encodePacked(bytes32 challenge, uint256 nonce).
 * Input = 64 bytes: challenge (32 bytes) || nonce_big_endian (32 bytes).
 * Rate = 136 bytes → lanes 0-16. Input occupies lanes 0-7 (64 bytes).
 * challenge_lanes: lanes 0-3 from challenge (little-endian lane encoding).
 * nonce: uint64 → encoded as big-endian uint256 → lane[7] = bswap64(nonce).
 */
__device__ __forceinline__ void keccak256_mine(
    const uint64_t *challenge_lanes, uint64_t nonce, uint8_t *output)
{
    uint64_t st[25];
    /* Zero state */
    #pragma unroll
    for (int i = 0; i < 25; i++) st[i] = 0;

    /* XOR challenge (lanes 0-3) */
    #pragma unroll
    for (int i = 0; i < 4; i++) st[i] = challenge_lanes[i];

    /* lanes 4,5,6 = 0 (nonce bytes 32-55 = zero) */

    /* lane 7 = bswap64(nonce) (nonce bytes 56-63 big-endian) */
    st[7] = BSWAP64(nonce);

    /* Keccak padding: 0x01 at byte 64 → lane[8] |= 0x01 */
    st[8] ^= 0x0000000000000001ULL;
    /* 0x80 at byte 135 (last byte of rate) → lane[16] |= 0x8000...0 */
    st[16] ^= 0x8000000000000000ULL;

    keccak_f1600(st);

    /* Squeeze first 32 bytes (4 lanes) into output */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lane = st[i];
        #pragma unroll
        for (int b = 0; b < 8; b++)
            output[i*8 + b] = (lane >> (b*8)) & 0xFF;
    }
}

/* True if a < b (32-byte big-endian numbers) */
__device__ __forceinline__ bool bytes32_lt(const uint8_t *a, const uint8_t *b) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

__global__ void mine_kernel(
    const uint64_t *challenge_lanes,
    const uint8_t  *difficulty,
    uint64_t        start_nonce,
    uint64_t       *result_nonce,
    int            *found)
{
    if (*found) return;

    uint64_t nonce = start_nonce + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t hash[32];
    keccak256_mine(challenge_lanes, nonce, hash);

    if (bytes32_lt(hash, difficulty)) {
        if (atomicCAS(found, 0, 1) == 0)
            *result_nonce = nonce;
    }
}

/* ── Host helpers ─────────────────────────────────────────────────────────── */

static void hex_to_bytes(const char *hex, uint8_t *out, int out_len) {
    if (hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) hex += 2;
    int hlen = (int)strlen(hex);
    memset(out, 0, out_len);
    /* right-align into out_len bytes */
    int off = out_len * 2 - hlen;
    for (int i = 0; i < hlen; i++) {
        char c = hex[i];
        uint8_t v = (c >= '0' && c <= '9') ? c - '0'
                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
        int pos = off + i;
        if (pos < 0) continue;
        out[pos/2] |= (pos % 2 == 0) ? (v << 4) : v;
    }
}

static void bytes_to_lanes_le(const uint8_t *bytes, uint64_t *lanes, int n_lanes) {
    for (int i = 0; i < n_lanes; i++) {
        lanes[i] = 0;
        for (int b = 0; b < 8; b++)
            lanes[i] |= ((uint64_t)bytes[i*8 + b]) << (b * 8);
    }
}

/* ── main ─────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <challenge_hex> <difficulty_hex> [start_nonce]\n", argv[0]);
        return 1;
    }

    uint8_t challenge[32], difficulty[32];
    hex_to_bytes(argv[1], challenge, 32);
    hex_to_bytes(argv[2], difficulty, 32);

    uint64_t start_nonce;
    if (argc >= 4) {
        start_nonce = strtoull(argv[3], NULL, 10);
    } else {
        srand((unsigned)time(NULL) ^ (unsigned)getpid());
        start_nonce = ((uint64_t)rand() << 33) ^ ((uint64_t)rand() << 17) ^ (uint64_t)rand();
    }

    uint64_t challenge_lanes[4];
    bytes_to_lanes_le(challenge, challenge_lanes, 4);

    /* Detect GPU and pick block count */
    int device_id = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    fprintf(stderr, "GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    fflush(stderr);

    int threads = 256;
    int blocks  = prop.multiProcessorCount * 32; /* ~32 waves per SM */
    uint64_t batch = (uint64_t)blocks * threads;

    /* Allocate device memory */
    uint64_t *d_challenge_lanes, *d_result;
    uint8_t  *d_difficulty;
    int      *d_found;

    cudaMalloc(&d_challenge_lanes, 4 * sizeof(uint64_t));
    cudaMalloc(&d_difficulty,      32);
    cudaMalloc(&d_result,          sizeof(uint64_t));
    cudaMalloc(&d_found,           sizeof(int));

    cudaMemcpy(d_challenge_lanes, challenge_lanes, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_difficulty,      difficulty,      32,                    cudaMemcpyHostToDevice);

    int h_found = 0;
    uint64_t h_result = 0;
    cudaMemcpy(d_found,  &h_found,  sizeof(int),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(uint64_t), cudaMemcpyHostToDevice);

    uint64_t nonce        = start_nonce;
    uint64_t total_hashes = 0;
    double   elapsed_sec  = 0.0;

    fprintf(stderr, "Mining: blocks=%d threads=%d batch=%llu\n",
            blocks, threads, (unsigned long long)batch);
    fflush(stderr);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    time_t t0 = time(NULL);

    while (!h_found) {
        cudaEventRecord(ev0);
        mine_kernel<<<blocks, threads>>>(d_challenge_lanes, d_difficulty,
                                         nonce, d_result, d_found);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);

        float ms = 0;
        cudaEventElapsedTime(&ms, ev0, ev1);
        elapsed_sec += ms / 1000.0;

        cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

        nonce        += batch;
        total_hashes += batch;

        if (elapsed_sec >= 5.0) {
            double mhs = (double)total_hashes / elapsed_sec / 1e6;
            fprintf(stderr, "Hashrate: %.1f MH/s | Nonces tried: %llu M\n",
                    mhs, (unsigned long long)(total_hashes / 1000000));
            fflush(stderr);
            elapsed_sec  = 0.0;
            total_hashes = 0;
        }
    }

    cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("FOUND:%llu\n", (unsigned long long)h_result);
    fflush(stdout);

    fprintf(stderr, "Done in %ld seconds.\n", (long)(time(NULL) - t0));

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d_challenge_lanes);
    cudaFree(d_difficulty);
    cudaFree(d_result);
    cudaFree(d_found);

    return 0;
}
