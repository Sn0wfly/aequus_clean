import cupy as cp
import numpy as np
from typing import Tuple, Optional
import time

# CUDA kernel for Monte-Carlo CFR rollouts
ROLLOUT_KERNEL = """
extern "C" __global__
void rollout_kernel(
    const unsigned long long* __restrict__ keys,
    float* __restrict__ cf_values,
    const unsigned long long seed,
    const int batch_size,
    const int N_rollouts,
    const int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (num_actions * N_rollouts);
    int action_idx = (idx % (num_actions * N_rollouts)) / N_rollouts;
    int rollout_idx = idx % N_rollouts;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize random state - use 1-D keys array
    unsigned long long state = keys[batch_idx] + seed + rollout_idx;
    
    // Simple poker simulation parameters
    const int stack_size = 10000;  // 100 BB
    const int pot_size = 150;      // 1.5 BB
    const int max_depth = 4;       // Preflop -> River
    
    // Simulate poker hand
    float payoff = 0.0f;
    
    // Use the key to seed the simulation
    for (int depth = 0; depth < max_depth; depth++) {
        // Simple action selection based on key
        int action = (state >> (depth * 8)) % num_actions;  // Dynamic action range
        
        // Update state for next iteration
        state = state * 1103515245ULL + 12345ULL;
        
        // Simple payoff calculation (normalized to reasonable range)
        if (action == 0) {  // fold
            payoff = -0.5f;  // Small loss
            break;
        } else if (action < num_actions / 3) {  // call-like actions
            payoff = (float)(state % 200 - 100) / 100.0f;  // -1.0 to 1.0
        } else if (action < 2 * num_actions / 3) {  // bet-like actions
            payoff = (float)(state % 300 - 150) / 150.0f;  // -1.0 to 1.0
        } else {  // raise-like actions
            payoff = (float)(state % 400 - 200) / 200.0f;  // -1.0 to 1.0
        }
    }
    
    // Store result - access 2-D cf_values array correctly
    if (rollout_idx < N_rollouts) {
        atomicAdd(&cf_values[batch_idx * num_actions + action_idx], payoff);
    }
}

extern "C" __global__
void normalize_rollouts_kernel(
    float* cf_values,
    const int batch_size,
    const int N_rollouts,
    const int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_actions;
    int action_idx = idx % num_actions;
    
    if (batch_idx >= batch_size) return;
    
    // Normalize by number of rollouts
    int value_idx = batch_idx * num_actions + action_idx;
    cf_values[value_idx] = cf_values[value_idx] / N_rollouts;
}
"""

# Compile the kernel
ROLLOUT_MODULE = cp.RawModule(code=ROLLOUT_KERNEL)
rollout_kernel = ROLLOUT_MODULE.get_function("rollout_kernel")
normalize_kernel = ROLLOUT_MODULE.get_function("normalize_rollouts_kernel")

def mccfr_rollout_gpu(keys_gpu: cp.ndarray, N_rollouts: int = 100, num_actions: int = 4) -> cp.ndarray:
    """
    Monte-Carlo CFR rollout on GPU.
    
    Args:
        keys_gpu: (B,) uint64 keys (1-D array)
        N_rollouts: Number of rollouts per action (default: 100)
        num_actions: Number of actions (default: 4)
    
    Returns:
        cf_values: (B, num_actions) counterfactual values as float32 ready for scatter_update
    """
    # 🔧 PARCHE DE SEGURIDAD: Clamp keys para evitar acceso ilegal
    keys_gpu = cp.clip(keys_gpu, 0, 25000)
    
    batch_size = keys_gpu.size
    
    # DEBUG: verificar límites
    print(f"DEBUG: keys_gpu min={keys_gpu.min()}, max={keys_gpu.max()}, shape={keys_gpu.shape}")
    print(f"DEBUG: N_rollouts={N_rollouts}, num_actions={num_actions}")
    print(f"DEBUG: batch_size={batch_size}")
    
    # Allocate output array
    cf_values = cp.zeros((batch_size, num_actions), dtype=cp.float32)
    print(f"DEBUG: cf_values shape={cf_values.shape}")
    
    # Calculate grid and block dimensions
    total_threads = batch_size * num_actions * N_rollouts
    threads_per_block = 256
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
    
    print(f"DEBUG: total_threads={total_threads}, blocks_per_grid={blocks_per_grid}")
    print(f"DEBUG: About to launch kernel with {len(keys_gpu)} keys")
    
    # Launch rollout kernel
    rollout_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            keys_gpu,
            cf_values,
            cp.uint64(int(time.time() * 1000)),  # Seed
            batch_size,
            N_rollouts,
            num_actions
        )
    )
    
    # Synchronize to ensure kernel completes
    cp.cuda.Stream.null.synchronize()
    
    # Normalize results
    normalize_blocks = (batch_size * num_actions + 255) // 256
    normalize_kernel(
        (normalize_blocks,),
        (256,),
        (cf_values, batch_size, N_rollouts, num_actions)
    )
    
    # Synchronize again
    cp.cuda.Stream.null.synchronize()
    
    return cf_values

def benchmark_mccfr_rollout():
    """Benchmark the MCCFR rollout performance."""
    print("Benchmarking MCCFR rollout...")
    
    # Test parameters
    batch_size = 1024
    num_actions = 6
    N_rollouts = 100
    
    # Create dummy keys
    keys_gpu = cp.random.randint(0, 2**32, (batch_size, num_actions), dtype=cp.uint64)
    
    # Create dummy hash table arrays
    table_keys = cp.zeros(2**24, dtype=cp.uint64)
    table_vals = cp.zeros(2**24, dtype=cp.uint64)
    counter = cp.zeros(1, dtype=cp.uint64)
    
    # Warm up
    for _ in range(3):
        _ = mccfr_rollout_gpu(keys_gpu, table_keys, table_vals, counter, N_rollouts)
    
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start_time = time.time()
    cf_values = mccfr_rollout_gpu(keys_gpu, table_keys, table_vals, counter, N_rollouts)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    total_rollouts = batch_size * num_actions * N_rollouts
    throughput = total_rollouts / (end_time - start_time)
    
    print(f"Batch size: {batch_size}")
    print(f"Actions per batch: {num_actions}")
    print(f"Rollouts per action: {N_rollouts}")
    print(f"Total rollouts: {total_rollouts:,}")
    print(f"Time: {end_time - start_time:.3f}s")
    print(f"Throughput: {throughput:,.0f} rollouts/sec")
    print(f"Memory usage: ~{batch_size * num_actions * 8 / 1024 / 1024:.1f} MB")
    
    return cf_values

if __name__ == "__main__":
    # Run benchmark
    cf_values = benchmark_mccfr_rollout()
    print(f"Sample cf_values shape: {cf_values.shape}")
    print(f"Sample values: {cf_values[:5, :5]}") 