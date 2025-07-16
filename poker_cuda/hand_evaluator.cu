/*
🚀 POKER HAND EVALUATOR - CUDA Native
=====================================
Custom CUDA implementation para máxima GPU performance
Reemplaza phevaluator (CPU-only) con solución 100% GPU

PERFORMANCE TARGET: >10M hands/second
MEMORY USAGE: <1GB lookup tables
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// ============================================================================
// 🃏 POKER CONSTANTS & LOOKUP TABLES
// ============================================================================

#define NUM_CARDS 52
#define NUM_RANKS 13
#define NUM_SUITS 4
#define MAX_HAND_SIZE 7
#define ROYAL_FLUSH 8
#define STRAIGHT_FLUSH 7
#define FOUR_KIND 6
#define FULL_HOUSE 5
#define FLUSH 4
#define STRAIGHT 3
#define THREE_KIND 2
#define TWO_PAIR 1
#define PAIR 0
#define HIGH_CARD -1

// Lookup table for fast rank/suit extraction
__constant__ int RANK_TABLE[52] = {
    0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,
    6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10,11,11,11,11,12,12,12,12
};

__constant__ int SUIT_TABLE[52] = {
    0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,
    0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3
};

// Pre-computed prime numbers for hand hashing
__constant__ int PRIME_RANKS[13] = {2,3,5,7,11,13,17,19,23,29,31,37,41};

// ============================================================================
// 🚀 CUDA DEVICE FUNCTIONS - Ultra Fast
// ============================================================================

__device__ inline int get_rank(int card) {
    return card >> 2;  // card / 4
}

__device__ inline int get_suit(int card) {
    return card & 3;   // card % 4
}

__device__ inline bool is_valid_card(int card) {
    return card >= 0 && card < 52;
}

// Fast bit manipulation for rank counting
__device__ inline void count_ranks_fast(int* cards, int num_cards, int* rank_counts) {
    // Initialize
    for (int i = 0; i < 13; i++) {
        rank_counts[i] = 0;
    }
    
    // Count using unrolled loop for speed
    for (int i = 0; i < num_cards; i++) {
        if (is_valid_card(cards[i])) {
            int rank = get_rank(cards[i]);
            rank_counts[rank]++;
        }
    }
}

__device__ inline void count_suits_fast(int* cards, int num_cards, int* suit_counts) {
    // Initialize
    for (int i = 0; i < 4; i++) {
        suit_counts[i] = 0;
    }
    
    // Count suits
    for (int i = 0; i < num_cards; i++) {
        if (is_valid_card(cards[i])) {
            int suit = get_suit(cards[i]);
            suit_counts[suit]++;
        }
    }
}

// Ultra-fast straight detection using bit manipulation
__device__ inline bool is_straight_fast(int* rank_counts) {
    // Convert rank counts to bitmask
    int rank_mask = 0;
    for (int i = 0; i < 13; i++) {
        if (rank_counts[i] > 0) {
            rank_mask |= (1 << i);
        }
    }
    
    // Check for 5 consecutive bits
    // Standard straights
    int straight_masks[10] = {
        0b0000000011111,  // A-2-3-4-5 (wheel)
        0b0000000111110,  // 2-3-4-5-6
        0b0000001111100,  // 3-4-5-6-7
        0b0000011111000,  // 4-5-6-7-8
        0b0000111110000,  // 5-6-7-8-9
        0b0001111100000,  // 6-7-8-9-10
        0b0011111000000,  // 7-8-9-10-J
        0b0111110000000,  // 8-9-10-J-Q
        0b1111100000000,  // 9-10-J-Q-K
        0b1111000000001   // 10-J-Q-K-A (broadway)
    };
    
    for (int i = 0; i < 10; i++) {
        if ((rank_mask & straight_masks[i]) == straight_masks[i]) {
            return true;
        }
    }
    
    return false;
}

// ============================================================================
// 🎯 MAIN HAND EVALUATION KERNEL
// ============================================================================

__device__ int evaluate_hand_strength(int* cards, int num_cards) {
    /*
    ULTRA-FAST hand evaluation optimized for GPU
    Returns hand strength: higher = better
    Range: 0-10000 (roughly)
    */
    
    if (num_cards < 5) return 0;  // Invalid hand
    
    int rank_counts[13];
    int suit_counts[4];
    
    count_ranks_fast(cards, num_cards, rank_counts);
    count_suits_fast(cards, num_cards, suit_counts);
    
    // Find max counts
    int max_rank_count = 0;
    int max_suit_count = 0;
    int pairs = 0, trips = 0, quads = 0;
    int high_rank = -1;
    
    for (int i = 0; i < 13; i++) {
        if (rank_counts[i] > max_rank_count) {
            max_rank_count = rank_counts[i];
        }
        if (rank_counts[i] >= 2) {
            if (rank_counts[i] == 2) pairs++;
            else if (rank_counts[i] == 3) trips++;
            else if (rank_counts[i] == 4) quads++;
        }
        if (rank_counts[i] > 0 && i > high_rank) {
            high_rank = i;
        }
    }
    
    for (int i = 0; i < 4; i++) {
        if (suit_counts[i] > max_suit_count) {
            max_suit_count = suit_counts[i];
        }
    }
    
    bool is_flush = (max_suit_count >= 5);
    bool is_straight = is_straight_fast(rank_counts);
    
    // Hand type evaluation with precise scoring
    int base_score = 0;
    int kicker_score = high_rank;  // High card kicker
    
    if (is_flush && is_straight) {
        // Straight flush or royal flush
        if (high_rank == 12) {  // Ace high straight flush (royal)
            base_score = 9000;
        } else {
            base_score = 8000 + high_rank * 10;
        }
    }
    else if (quads > 0) {
        // Four of a kind
        base_score = 7000;
        // Find the quad rank
        for (int i = 12; i >= 0; i--) {
            if (rank_counts[i] == 4) {
                base_score += i * 20;
                break;
            }
        }
    }
    else if (trips > 0 && pairs > 0) {
        // Full house
        base_score = 6000;
        // Find trip rank and pair rank
        for (int i = 12; i >= 0; i--) {
            if (rank_counts[i] == 3) {
                base_score += i * 15;
                break;
            }
        }
        for (int i = 12; i >= 0; i--) {
            if (rank_counts[i] == 2) {
                base_score += i * 3;
                break;
            }
        }
    }
    else if (is_flush) {
        // Flush
        base_score = 5000 + high_rank * 8;
    }
    else if (is_straight) {
        // Straight
        base_score = 4000 + high_rank * 6;
    }
    else if (trips > 0) {
        // Three of a kind
        base_score = 3000;
        for (int i = 12; i >= 0; i--) {
            if (rank_counts[i] == 3) {
                base_score += i * 12;
                break;
            }
        }
    }
    else if (pairs >= 2) {
        // Two pair
        base_score = 2000;
        int pair_ranks[2] = {-1, -1};
        int pair_idx = 0;
        for (int i = 12; i >= 0 && pair_idx < 2; i--) {
            if (rank_counts[i] == 2) {
                pair_ranks[pair_idx++] = i;
            }
        }
        if (pair_ranks[0] >= 0) base_score += pair_ranks[0] * 10;
        if (pair_ranks[1] >= 0) base_score += pair_ranks[1] * 5;
    }
    else if (pairs == 1) {
        // One pair
        base_score = 1000;
        for (int i = 12; i >= 0; i--) {
            if (rank_counts[i] == 2) {
                base_score += i * 8;
                break;
            }
        }
    }
    else {
        // High card
        base_score = high_rank * 4;
    }
    
    return base_score + kicker_score;
}

// ============================================================================
// 🚀 BATCH EVALUATION KERNELS - Maximum Throughput
// ============================================================================

__global__ void evaluate_hands_batch(
    int* cards_batch,      // [batch_size, max_cards_per_hand]
    int* num_cards_batch,  // [batch_size]
    int* results,          // [batch_size] - output
    int batch_size,
    int max_cards_per_hand
) {
    /*
    Kernel para evaluar thousands de manos en paralelo
    Cada thread evalúa una mano completa
    */
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Get this hand's cards
        int* hand_cards = &cards_batch[idx * max_cards_per_hand];
        int num_cards = num_cards_batch[idx];
        
        // Evaluate hand strength
        int strength = evaluate_hand_strength(hand_cards, num_cards);
        
        // Store result
        results[idx] = strength;
    }
}

__global__ void generate_and_evaluate_random_hands(
    unsigned long long* seeds,  // [batch_size] - random seeds
    int* results,              // [batch_size] - output hand strengths
    int batch_size,
    int cards_per_hand
) {
    /*
    Kernel que GENERA y EVALÚA manos random en GPU
    Útil para training donde necesitamos muchas manos sintéticas
    */
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Initialize random state
        unsigned long long seed = seeds[idx];
        
        // Simple LCG random number generator (GPU-friendly)
        auto next_random = [&]() -> int {
            seed = seed * 1103515245ULL + 12345ULL;
            return (seed >> 16) & 0x7fff;
        };
        
        // Generate random hand (simple approach)
        int hand_cards[7] = {-1, -1, -1, -1, -1, -1, -1};
        bool used_cards[52] = {false};
        
        // Generate unique cards
        for (int i = 0; i < cards_per_hand && i < 7; i++) {
            int card;
            int attempts = 0;
            do {
                card = next_random() % 52;
                attempts++;
            } while (used_cards[card] && attempts < 100);
            
            if (!used_cards[card]) {
                hand_cards[i] = card;
                used_cards[card] = true;
            }
        }
        
        // Evaluate the generated hand
        int strength = evaluate_hand_strength(hand_cards, cards_per_hand);
        results[idx] = strength;
        
        // Update seed for next time
        seeds[idx] = seed;
    }
}

// ============================================================================
// 🎯 SPECIALIZED POKER TRAINING KERNELS
// ============================================================================

__global__ void evaluate_poker_scenarios(
    int* hole_cards_batch,     // [batch_size, 2] - player hole cards
    int* community_cards,      // [5] - shared community cards
    int* player_positions,     // [batch_size] - positions (0-5)
    float* hand_strengths,     // [batch_size] - output normalized strengths
    int* hand_types,          // [batch_size] - output hand type classifications
    int batch_size
) {
    /*
    Specialized kernel para poker training scenarios
    Evalúa hole cards + community cards y normaliza results
    */
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Combine hole cards + community cards
        int full_hand[7];
        full_hand[0] = hole_cards_batch[idx * 2];
        full_hand[1] = hole_cards_batch[idx * 2 + 1];
        
        // Add community cards
        for (int i = 0; i < 5; i++) {
            full_hand[i + 2] = community_cards[i];
        }
        
        // Evaluate hand
        int raw_strength = evaluate_hand_strength(full_hand, 7);
        
        // Normalize to 0-1 range for neural networks
        float normalized_strength = raw_strength / 10000.0f;
        hand_strengths[idx] = fminf(1.0f, fmaxf(0.0f, normalized_strength));
        
        // Classify hand type for strategic decisions
        int hand_type = HIGH_CARD;
        if (raw_strength >= 9000) hand_type = ROYAL_FLUSH;
        else if (raw_strength >= 8000) hand_type = STRAIGHT_FLUSH;
        else if (raw_strength >= 7000) hand_type = FOUR_KIND;
        else if (raw_strength >= 6000) hand_type = FULL_HOUSE;
        else if (raw_strength >= 5000) hand_type = FLUSH;
        else if (raw_strength >= 4000) hand_type = STRAIGHT;
        else if (raw_strength >= 3000) hand_type = THREE_KIND;
        else if (raw_strength >= 2000) hand_type = TWO_PAIR;
        else if (raw_strength >= 1000) hand_type = PAIR;
        
        hand_types[idx] = hand_type;
    }
}

// ============================================================================
// 🚀 HOST INTERFACE FUNCTIONS
// ============================================================================

extern "C" {

int cuda_evaluate_single_hand(int* cards, int num_cards) {
    /*
    CPU interface para evaluar una sola mano
    Útil para debugging y tests
    */
    
    int* d_cards;
    int* d_result;
    
    // Allocate GPU memory
    cudaMalloc(&d_cards, num_cards * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy to GPU
    cudaMemcpy(d_cards, cards, num_cards * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel (single thread)
    evaluate_hands_batch<<<1, 1>>>(d_cards, &num_cards, d_result, 1, num_cards);
    
    // Copy result back
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_cards);
    cudaFree(d_result);
    
    return result;
}

void cuda_evaluate_hands_batch_wrapper(
    int* cards_batch,
    int* num_cards_batch,
    int* results,
    int batch_size,
    int max_cards_per_hand
) {
    /*
    Wrapper para evaluar batch de manos desde Python
    */
    
    // Calculate grid dimensions
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    evaluate_hands_batch<<<blocks, threads_per_block>>>(
        cards_batch, num_cards_batch, results, batch_size, max_cards_per_hand
    );
    
    // Synchronize
    cudaDeviceSynchronize();
}

void cuda_generate_training_batch(
    unsigned long long* seeds,
    int* hand_strengths,
    int batch_size,
    int cards_per_hand
) {
    /*
    Genera batch de training data completamente en GPU
    */
    
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    generate_and_evaluate_random_hands<<<blocks, threads_per_block>>>(
        seeds, hand_strengths, batch_size, cards_per_hand
    );
    
    cudaDeviceSynchronize();
}

} // extern "C" 