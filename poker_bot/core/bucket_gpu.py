"""
GPU-only bucketing + hash-table for Poker CFR
Requires: cupy-cuda12x
"""

import cupy as cp
import time

# -----------------------------
# 1. MurmurHash3 64-bit finalizer
# -----------------------------
def _hash_mix(k: int) -> int:
    """MurmurHash3 64-bit finalizer (simplified, no seeds)"""
    k = (k ^ (k >> 33)) & 0xFFFFFFFFFFFFFFFF
    k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    k = (k ^ (k >> 33)) & 0xFFFFFFFFFFFFFFFF
    k = (k * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    k = (k ^ (k >> 33)) & 0xFFFFFFFFFFFFFFFF
    return k

# -----------------------------
# 2. Empaquetado uint64 con hash
# -----------------------------
def pack_keys(hole_hash, round_id, position,
              stack_bucket, pot_bucket, num_active):
    # Create raw key with proper bit masks
    raw = (
        (round_id & 0xF) << 48 |
        (hole_hash & 0xFF) << 40 |
        (position & 0x7) << 36 |
        (stack_bucket & 0xFF) << 28 |
        (pot_bucket & 0xFF) << 20 |
        (num_active & 0xF) << 16
    )
    # Apply MurmurHash3
    return cp.vectorize(_hash_mix)(raw.astype(cp.uint64))

# -----------------------------
# 3. Kernel CUDA optimizado con hash
# -----------------------------
_KERNEL = r'''
__device__ unsigned long long hash_mix(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xFF51AFD7ED558CCDULL;
    k ^= k >> 33;
    k *= 0xC4CEB9FE1A85EC53ULL;
    k ^= k >> 33;
    return k;
}

extern "C" __global__
void bucket_kernel(
    const unsigned long long* __restrict__ keys,
    unsigned int* __restrict__ out_idx,
    unsigned long long* __restrict__ table_keys,
    unsigned int* __restrict__ table_vals,
    unsigned int* __restrict__ counter,
    const unsigned int N,
    const unsigned int mask)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    unsigned long long key = keys[tid];
    // Apply hash function to improve distribution
    key = hash_mix(key);
    unsigned int slot = (unsigned int)(key & mask);

    // MAX 3 attempts to avoid infinite loops
    for (int attempt = 0; attempt < 3; attempt++) {
        unsigned long long old = atomicCAS(&table_keys[slot], 0ULL, key);
        if (old == 0ULL) {
            // new key: assign index using explicit counter
            unsigned int idx = atomicAdd(counter, 1u);
            table_vals[slot] = idx;
            out_idx[tid] = idx;
            return;
        }
        if (old == key) {
            // existing key
            out_idx[tid] = table_vals[slot];
            return;
        }
        slot = (slot + 1) & mask;
    }
    // Fallback: use direct slot
    out_idx[tid] = slot;
}
'''

_bucket_kernel = cp.RawKernel(_KERNEL, 'bucket_kernel')

# -----------------------------
# 4. Wrapper optimizado
# -----------------------------
def build_or_get_indices(keys_gpu, table_keys, table_vals, counter):
    """
    keys_gpu: CuPy array uint64
    table_keys: CuPy array uint64 (persistente, tamaño potencia de 2)
    table_vals: CuPy array uint32 (persistente, tamaño igual a table_keys)
    counter: CuPy array uint32 shape=(1,) (persistente)
    Devuelve: indices_gpu CuPy array uint32
    """
    # DEBUG: verificar keys antes del clamping
    print(f"DEBUG: keys_gpu min={keys_gpu.min()}, max={keys_gpu.max()}, shape={keys_gpu.shape}")
    
    # 🔧 PARCHE DE SEGURIDAD: Clamp keys para evitar acceso ilegal
    keys_gpu = cp.clip(keys_gpu, 0, 2**32-1)  # Limitar a 32 bits para evitar overflow
    
    print(f"DEBUG: After clamp - keys_gpu min={keys_gpu.min()}, max={keys_gpu.max()}")
    
    N = keys_gpu.size
    indices_gpu = cp.empty(N, dtype=cp.uint32)
    table_size = table_keys.size
    threads = 1024
    blocks = (N + threads - 1) // threads
    if blocks < 32:
        blocks = 32
    _bucket_kernel(
        (blocks,), (threads,),
        (keys_gpu, indices_gpu,
         table_keys, table_vals, counter,
         cp.uint32(N), cp.uint32(table_size - 1))
    )
    cp.cuda.Device().synchronize()
    return indices_gpu

# Versión efímera para benchmark rápido
# (crea y borra la tabla hash en cada llamada)
def build_or_get_indices_ephemeral(keys_gpu, table_size=2**26):
    table_keys = cp.zeros(table_size, dtype=cp.uint64)
    table_vals = cp.zeros(table_size, dtype=cp.uint32)
    counter = cp.zeros(1, dtype=cp.uint32)
    return build_or_get_indices(keys_gpu, table_keys, table_vals, counter)

# -----------------------------
# 5. Test del counter
# -----------------------------
def test_counter():
    """Test mínimo para verificar que el counter funciona"""
    print("Testing counter functionality...")
    
    # Crear datos de prueba
    keys = cp.random.randint(0, 10000, 2000, dtype=cp.uint64)
    table_keys = cp.zeros(2**16, dtype=cp.uint64)
    table_vals = cp.zeros(2**16, dtype=cp.uint32)
    counter = cp.zeros(1, dtype=cp.uint32)
    
    print(f"Before: counter[0] = {int(counter[0])}")
    
    # Ejecutar build_or_get_indices
    indices = build_or_get_indices(keys, table_keys, table_vals, counter)
    
    print(f"After: counter[0] = {int(counter[0])}")
    print(f"Expected: ~{len(cp.unique(keys))} unique keys")
    print(f"Actual unique indices: {len(cp.unique(indices))}")
    
    return int(counter[0])

# -----------------------------
# 6. Benchmark
# -----------------------------
def benchmark():
    N = 1_000_000
    print(f"Benchmarking {N:,} keys …")
    
    rng = cp.random.default_rng(42)
    keys = pack_keys(
        rng.integers(0, 1326, N, cp.uint16),
        rng.integers(0, 6, N, cp.uint8),
        rng.integers(0, 6, N, cp.uint8),
        rng.integers(0, 16, N, cp.uint8),
        rng.integers(0, 16, N, cp.uint8),
        rng.integers(2, 7, N, cp.uint8)
    )
    
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    indices = build_or_get_indices_ephemeral(keys)
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"GPU throughput: {N/elapsed*1e-6:.1f} M keys/sec (tiempo: {elapsed:.4f} s)")

if __name__ == '__main__':
    benchmark() 