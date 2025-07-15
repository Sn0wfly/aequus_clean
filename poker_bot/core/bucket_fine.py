import cupy as cp
import numpy as np
from typing import Tuple

# CUDA kernel for fine-grained bucketing
FINE_BUCKET_KERNEL = """
extern "C" __global__
void fine_bucket_kernel(
    const int* hole_cards,      // (B*6*2) flat
    const int* community_cards, // (B*6*5) flat
    const int* positions,       // (B*6) flat
    const float* stack_sizes,   // (B*6) flat
    const float* pot_sizes,     // (B*6) flat
    const int* num_active,      // (B*6) flat
    unsigned long long* keys,    // (B*6) output
    const int batch_size,
    const int num_players
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_players;
    int player_idx = idx % num_players;
    
    if (batch_idx >= batch_size) return;
    
    // Calculate flat indices
    int hole_offset = (batch_idx * num_players + player_idx) * 2;
    int comm_offset = (batch_idx * num_players + player_idx) * 5;
    int data_offset = batch_idx * num_players + player_idx;
    
    // Get hole cards
    int card1 = hole_cards[hole_offset];
    int card2 = hole_cards[hole_offset + 1];
    
    // Skip if player not active
    if (card1 == -1 || card2 == -1) {
        keys[data_offset] = 0;
        return;
    }
    
    // Get community cards
    int comm_cards[5];
    for (int i = 0; i < 5; i++) {
        comm_cards[i] = community_cards[comm_offset + i];
    }
    
    // Count dealt community cards
    int dealt_comm = 0;
    for (int i = 0; i < 5; i++) {
        if (comm_cards[i] >= 0) dealt_comm++;
    }
    
    // Determine round
    int round_bucket;
    if (dealt_comm == 0) round_bucket = 0;      // Preflop
    else if (dealt_comm == 3) round_bucket = 1; // Flop
    else if (dealt_comm == 4) round_bucket = 2; // Turn
    else round_bucket = 3;                       // River
    
    // Hand bucket calculation
    int hand_bucket;
    if (round_bucket == 0) {
        // Preflop: use card ranks and suitedness
        int rank1 = card1 % 13;
        int rank2 = card2 % 13;
        int suit1 = card1 / 13;
        int suit2 = card2 / 13;
        bool suited = (suit1 == suit2);
        
        // Simple preflop bucket (0-168)
        if (rank1 == rank2) {
            hand_bucket = rank1; // Pairs: 0-12
        } else {
            int high = max(rank1, rank2);
            int low = min(rank1, rank2);
            if (suited) {
                hand_bucket = 13 + high * 12 + low; // Suited: 13-168
            } else {
                hand_bucket = 169 + high * 12 + low; // Offsuit: 169-324
            }
        }
    } else {
        // Post-flop: simple strength bucket (0-9)
        // This is a simplified version - in practice would use proper evaluator
        int total_cards = 0;
        int all_cards[7];
        all_cards[0] = card1;
        all_cards[1] = card2;
        for (int i = 0; i < dealt_comm; i++) {
            all_cards[2 + i] = comm_cards[i];
            total_cards++;
        }
        
        // Simple strength calculation (placeholder)
        hand_bucket = (card1 + card2) % 10; // 0-9 strength buckets
    }
    
    // Position bucket (0-5)
    int pos_bucket = positions[data_offset];
    
    // Stack bucket: intervals of 5 BB (0-39)
    float stack = stack_sizes[data_offset];
    int stack_bucket = min(39, (int)(stack / 5.0f));
    
    // Pot bucket: intervals of 1 BB (0-199)
    float pot = pot_sizes[data_offset];
    int pot_bucket = min(199, (int)pot);
    
    // Active players bucket (2-6)
    int active_bucket = num_active[data_offset] - 2; // 0-4
    
    // History bucket (simplified - would need action history)
    int hist_bucket = 0; // Placeholder
    
    // Pack into 64-bit key
    unsigned long long key = 0;
    key |= (unsigned long long)round_bucket << 60;    // 4 bits
    key |= (unsigned long long)hand_bucket << 50;     // 10 bits
    key |= (unsigned long long)pos_bucket << 47;      // 3 bits
    key |= (unsigned long long)stack_bucket << 42;    // 5 bits
    key |= (unsigned long long)pot_bucket << 33;      // 9 bits
    key |= (unsigned long long)active_bucket << 30;   // 3 bits
    key |= (unsigned long long)hist_bucket << 18;     // 12 bits
    
    keys[data_offset] = key;
}
"""

# Compile the kernel
FINE_BUCKET_MODULE = cp.RawModule(code=FINE_BUCKET_KERNEL)
fine_bucket_kernel = FINE_BUCKET_MODULE.get_function("fine_bucket_kernel")

def fine_bucket_kernel_wrapper(
    hole_cards: cp.ndarray,
    community_cards: cp.ndarray,
    positions: cp.ndarray,
    stack_sizes: cp.ndarray,
    pot_sizes: cp.ndarray,
    num_active: cp.ndarray
) -> cp.ndarray:
    """
    Fine-grained bucketing kernel wrapper.
    
    Args:
        hole_cards: (B, 6, 2) int32
        community_cards: (B, 6, 5) int32
        positions: (B, 6) int32
        stack_sizes: (B, 6) float32
        pot_sizes: (B, 6) float32
        num_active: (B, 6) int32
    
    Returns:
        keys: (B, 6) uint64 - fine-grained bucket keys
    """
    batch_size, num_players = hole_cards.shape[:2]
    
    # Flatten arrays for kernel
    hole_flat = hole_cards.reshape(-1)
    comm_flat = community_cards.reshape(-1)
    pos_flat = positions.reshape(-1)
    stack_flat = stack_sizes.reshape(-1)
    pot_flat = pot_sizes.reshape(-1)
    active_flat = num_active.reshape(-1)
    
    # Allocate output
    keys = cp.zeros((batch_size * num_players,), dtype=cp.uint64)
    
    # Calculate grid and block dimensions
    total_threads = batch_size * num_players
    threads_per_block = 256
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    fine_bucket_kernel(
        (blocks_per_grid,),
        (threads_per_block,),
        (
            hole_flat,
            comm_flat,
            pos_flat,
            stack_flat,
            pot_flat,
            active_flat,
            keys,
            batch_size,
            num_players
        )
    )
    
    # Synchronize
    cp.cuda.Stream.null.synchronize()
    
    # Reshape back to (B, 6)
    return keys.reshape(batch_size, num_players)

def benchmark_fine_bucketing():
    """Benchmark the fine bucketing performance."""
    print("Benchmarking fine bucketing...")
    
    # Test parameters
    batch_size = 1024
    num_players = 6
    
    # Create test data
    hole_cards = cp.random.randint(0, 52, (batch_size, num_players, 2), dtype=cp.int32)
    community_cards = cp.random.randint(-1, 52, (batch_size, num_players, 5), dtype=cp.int32)
    positions = cp.random.randint(0, 6, (batch_size, num_players), dtype=cp.int32)
    stack_sizes = cp.random.uniform(10.0, 200.0, (batch_size, num_players), dtype=cp.float32)
    pot_sizes = cp.random.uniform(1.5, 100.0, (batch_size, num_players), dtype=cp.float32)
    num_active = cp.random.randint(2, 7, (batch_size, num_players), dtype=cp.int32)
    
    # Warm up
    for _ in range(3):
        _ = fine_bucket_kernel_wrapper(hole_cards, community_cards, positions, stack_sizes, pot_sizes, num_active)
    
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    import time
    start_time = time.time()
    keys = fine_bucket_kernel_wrapper(hole_cards, community_cards, positions, stack_sizes, pot_sizes, num_active)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    
    # Count unique keys
    unique_keys = cp.unique(keys)
    unique_count = len(unique_keys)
    
    print(f"Batch size: {batch_size}")
    print(f"Players per batch: {num_players}")
    print(f"Total info sets: {batch_size * num_players}")
    print(f"Unique info sets: {unique_count}")
    print(f"Time: {end_time - start_time:.3f}s")
    print(f"Throughput: {batch_size * num_players / (end_time - start_time):,.0f} info-sets/sec")
    print(f"Uniqueness ratio: {unique_count / (batch_size * num_players) * 100:.1f}%")
    
    return keys

if __name__ == "__main__":
    # Run benchmark
    keys = benchmark_fine_bucketing()
    print(f"Sample keys shape: {keys.shape}")
    print(f"Sample keys: {keys[:5, :5]}") 