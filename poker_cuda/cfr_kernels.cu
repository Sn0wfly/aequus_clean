/*
🚀 POKER CFR KERNELS - CUDA Implementation
==========================================
Complete CFR training pipeline en GPU puro
- Regret accumulation
- Strategy computation
- Game simulation
- Info set processing

PERFORMANCE TARGET: >100 it/s training speed
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>

// ============================================================================
// 🎯 CFR CONSTANTS
// ============================================================================

#define MAX_INFO_SETS 50000
#define NUM_ACTIONS 6
#define MAX_PLAYERS 6
#define MAX_CARDS_PER_HAND 7
#define MAX_ACTIONS_PER_GAME 48

// Action definitions
#define ACTION_FOLD 0
#define ACTION_CHECK 1
#define ACTION_CALL 2
#define ACTION_BET 3
#define ACTION_RAISE 4
#define ACTION_ALL_IN 5

// External hand evaluator
extern "C" int cuda_evaluate_single_hand(int* cards, int num_cards);

// ============================================================================
// 🚀 CUDA DEVICE FUNCTIONS FOR CFR
// ============================================================================

__device__ inline float compute_regret_matching_strategy(
    float* regrets,
    float* strategy_out,
    int info_set_idx,
    int num_actions
) {
    /*
    Compute strategy from regrets using regret matching
    Returns normalization factor
    */
    
    float positive_regret_sum = 0.0f;
    
    // Sum positive regrets
    for (int a = 0; a < num_actions; a++) {
        float regret = regrets[info_set_idx * num_actions + a];
        if (regret > 0.0f) {
            positive_regret_sum += regret;
        }
    }
    
    // Compute strategy
    if (positive_regret_sum > 1e-6f) {
        // Use regret matching
        for (int a = 0; a < num_actions; a++) {
            float regret = regrets[info_set_idx * num_actions + a];
            strategy_out[a] = fmaxf(0.0f, regret) / positive_regret_sum;
        }
    } else {
        // Uniform strategy
        float uniform_prob = 1.0f / num_actions;
        for (int a = 0; a < num_actions; a++) {
            strategy_out[a] = uniform_prob;
        }
    }
    
    return positive_regret_sum;
}

__device__ inline int compute_info_set_advanced(
    int* hole_cards,      // [2] player hole cards
    int* community_cards, // [5] board cards
    int player_position,  // 0-5
    float pot_size,
    int num_community_cards
) {
    /*
    Advanced info set computation matching the CPU version
    */
    
    // Extract hole card features
    int hole_ranks[2] = {hole_cards[0] >> 2, hole_cards[1] >> 2};
    int hole_suits[2] = {hole_cards[0] & 3, hole_cards[1] & 3};
    
    int high_rank = max(hole_ranks[0], hole_ranks[1]);
    int low_rank = min(hole_ranks[0], hole_ranks[1]);
    bool is_suited = (hole_suits[0] == hole_suits[1]);
    bool is_pair = (hole_ranks[0] == hole_ranks[1]);
    
    // Street bucketing (4 buckets: preflop, flop, turn, river)
    int street_bucket = 0;
    if (num_community_cards == 0) street_bucket = 0;      // Preflop
    else if (num_community_cards == 3) street_bucket = 1; // Flop
    else if (num_community_cards == 4) street_bucket = 2; // Turn
    else street_bucket = 3;                               // River
    
    // Hand bucketing (169 preflop buckets like Pluribus)
    int preflop_bucket;
    if (is_pair) {
        preflop_bucket = high_rank; // Pairs: 0-12
    } else if (is_suited) {
        preflop_bucket = 13 + high_rank * 12 + low_rank; // Suited: 13-168
    } else {
        preflop_bucket = 169 + high_rank * 12 + low_rank; // Offsuit: 169+
    }
    
    int hand_bucket = preflop_bucket % 169; // Normalize to 0-168
    
    // Position bucketing (6 buckets: 0-5)
    int position_bucket = player_position;
    
    // Stack/pot bucketing
    int stack_bucket = min(19, (int)(pot_size / 5.0f));
    int pot_bucket = min(9, (int)(pot_size / 10.0f));
    int active_bucket = min(4, player_position);
    
    // Combine all factors
    int info_set_id = (
        street_bucket * 10000 +
        hand_bucket * 50 +
        position_bucket * 8 +
        stack_bucket * 2 +
        pot_bucket * 1 +
        active_bucket
    );
    
    return info_set_id % MAX_INFO_SETS;
}

__device__ inline float evaluate_action_value(
    int action,
    float player_payoff,
    float hand_strength_normalized,
    int player_position
) {
    /*
    Evaluate the value of taking a specific action
    Used for regret calculation
    */
    
    float base_value = player_payoff;
    
    // Hand strength factor
    float hand_factor = hand_strength_normalized;
    
    // Position factor (later positions can be more aggressive)
    float position_factor = (player_position + 1) / 6.0f;
    
    // Action-specific evaluation
    float action_multiplier = 1.0f;
    
    switch (action) {
        case ACTION_FOLD:
            // Fold value depends on avoiding losses
            if (player_payoff < 0) {
                action_multiplier = 0.8f; // Good to avoid losses
            } else {
                action_multiplier = 0.1f; // Bad to fold when winning
            }
            break;
            
        case ACTION_CHECK:
        case ACTION_CALL:
            // Passive actions - neutral
            action_multiplier = 0.6f + hand_factor * 0.4f;
            break;
            
        case ACTION_BET:
        case ACTION_RAISE:
            // Aggressive actions - reward strong hands
            action_multiplier = 0.4f + hand_factor * 0.8f + position_factor * 0.2f;
            break;
            
        case ACTION_ALL_IN:
            // Very aggressive - only with very strong hands
            action_multiplier = 0.2f + hand_factor * 1.0f + position_factor * 0.1f;
            break;
    }
    
    return base_value * action_multiplier;
}

// ============================================================================
// 🚀 GAME SIMULATION KERNELS
// ============================================================================

__global__ void simulate_poker_games_gpu(
    curandState* rand_states,    // [batch_size] random states
    int* hole_cards_out,         // [batch_size, MAX_PLAYERS, 2] output
    int* community_cards_out,    // [batch_size, 5] output
    float* payoffs_out,          // [batch_size, MAX_PLAYERS] output
    int* action_histories_out,   // [batch_size, MAX_ACTIONS_PER_GAME] output
    float* pot_sizes_out,        // [batch_size] output
    int batch_size
) {
    /*
    Simulate complete poker games entirely on GPU
    Each thread simulates one game
    */
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        curandState* state = &rand_states[idx];
        
        // Generate deck and deal cards
        bool used_cards[52] = {false};
        int cards_dealt = 0;
        
        // Deal hole cards to all players
        for (int player = 0; player < MAX_PLAYERS; player++) {
            for (int card = 0; card < 2; card++) {
                int dealt_card;
                int attempts = 0;
                do {
                    dealt_card = curand(state) % 52;
                    attempts++;
                } while (used_cards[dealt_card] && attempts < 100);
                
                hole_cards_out[idx * MAX_PLAYERS * 2 + player * 2 + card] = dealt_card;
                used_cards[dealt_card] = true;
                cards_dealt++;
            }
        }
        
        // Deal community cards
        for (int card = 0; card < 5; card++) {
            int dealt_card;
            int attempts = 0;
            do {
                dealt_card = curand(state) % 52;
                attempts++;
            } while (used_cards[dealt_card] && attempts < 100);
            
            community_cards_out[idx * 5 + card] = dealt_card;
            used_cards[dealt_card] = true;
        }
        
        // Evaluate hand strengths for all players
        float hand_strengths[MAX_PLAYERS];
        for (int player = 0; player < MAX_PLAYERS; player++) {
            int full_hand[7];
            // Copy hole cards
            full_hand[0] = hole_cards_out[idx * MAX_PLAYERS * 2 + player * 2];
            full_hand[1] = hole_cards_out[idx * MAX_PLAYERS * 2 + player * 2 + 1];
            // Copy community cards
            for (int i = 0; i < 5; i++) {
                full_hand[i + 2] = community_cards_out[idx * 5 + i];
            }
            
            // Simplified hand evaluation for GPU
            // (We could call the full evaluator but this is faster)
            int strength = 0;
            for (int i = 0; i < 7; i++) {
                if (full_hand[i] >= 0) {
                    strength += (full_hand[i] >> 2) * (i + 1); // Simplified scoring
                }
            }
            hand_strengths[player] = strength / 1000.0f; // Normalize
        }
        
        // Simulate action sequence
        float pot_size = 15.0f; // Initial pot
        int num_actions = 0;
        
        for (int round = 0; round < 4 && num_actions < MAX_ACTIONS_PER_GAME - 6; round++) {
            for (int player = 0; player < MAX_PLAYERS && num_actions < MAX_ACTIONS_PER_GAME; player++) {
                // Probabilistic action generation based on hand strength and position
                float action_prob = 0.3f + hand_strengths[player] * 0.4f + curand_uniform(state) * 0.3f;
                
                int action;
                if (action_prob < 0.2f) action = ACTION_FOLD;
                else if (action_prob < 0.4f) action = ACTION_CHECK;
                else if (action_prob < 0.6f) action = ACTION_CALL;
                else if (action_prob < 0.8f) action = ACTION_BET;
                else if (action_prob < 0.95f) action = ACTION_RAISE;
                else action = ACTION_ALL_IN;
                
                action_histories_out[idx * MAX_ACTIONS_PER_GAME + num_actions] = action;
                num_actions++;
                
                // Update pot size based on action
                if (action >= ACTION_BET) {
                    pot_size += 2.0f + curand_uniform(state) * 8.0f; // Random bet size
                }
            }
        }
        
        // Fill remaining action history with -1
        for (int i = num_actions; i < MAX_ACTIONS_PER_GAME; i++) {
            action_histories_out[idx * MAX_ACTIONS_PER_GAME + i] = -1;
        }
        
        // Determine winner (player with best hand strength)
        int winner = 0;
        float best_strength = hand_strengths[0];
        for (int player = 1; player < MAX_PLAYERS; player++) {
            if (hand_strengths[player] > best_strength) {
                best_strength = hand_strengths[player];
                winner = player;
            }
        }
        
        // Calculate payoffs
        float contribution_per_player = pot_size / MAX_PLAYERS;
        for (int player = 0; player < MAX_PLAYERS; player++) {
            if (player == winner) {
                payoffs_out[idx * MAX_PLAYERS + player] = pot_size - contribution_per_player;
            } else {
                payoffs_out[idx * MAX_PLAYERS + player] = -contribution_per_player;
            }
        }
        
        pot_sizes_out[idx] = pot_size;
    }
}

// ============================================================================
// 🚀 CFR UPDATE KERNELS
// ============================================================================

__global__ void cfr_regret_update_kernel(
    float* regrets,              // [MAX_INFO_SETS, NUM_ACTIONS] - input/output
    int* hole_cards_batch,       // [batch_size, MAX_PLAYERS, 2]
    int* community_cards_batch,  // [batch_size, 5]
    float* payoffs_batch,        // [batch_size, MAX_PLAYERS]
    float* pot_sizes_batch,      // [batch_size]
    int batch_size
) {
    /*
    Update regrets for all info sets based on game outcomes
    Each thread processes multiple games for one info set
    */
    
    int info_set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (info_set_idx < MAX_INFO_SETS) {
        float regret_updates[NUM_ACTIONS] = {0.0f};
        int updates_count = 0;
        
        // Process all games in batch
        for (int game_idx = 0; game_idx < batch_size; game_idx++) {
            for (int player = 0; player < MAX_PLAYERS; player++) {
                // Get player's hole cards
                int hole_cards[2];
                hole_cards[0] = hole_cards_batch[game_idx * MAX_PLAYERS * 2 + player * 2];
                hole_cards[1] = hole_cards_batch[game_idx * MAX_PLAYERS * 2 + player * 2 + 1];
                
                // Get community cards
                int community_cards[5];
                for (int i = 0; i < 5; i++) {
                    community_cards[i] = community_cards_batch[game_idx * 5 + i];
                }
                
                // Compute info set for this player
                float pot_size = pot_sizes_batch[game_idx];
                int computed_info_set = compute_info_set_advanced(
                    hole_cards, community_cards, player, pot_size, 5
                );
                
                // Only update if this matches our info set
                if (computed_info_set == info_set_idx) {
                    float player_payoff = payoffs_batch[game_idx * MAX_PLAYERS + player];
                    
                    // Evaluate hand strength for this player
                    int full_hand[7] = {
                        hole_cards[0], hole_cards[1],
                        community_cards[0], community_cards[1], community_cards[2],
                        community_cards[3], community_cards[4]
                    };
                    
                    // Simplified hand strength calculation
                    float hand_strength = 0.0f;
                    for (int i = 0; i < 7; i++) {
                        if (full_hand[i] >= 0) {
                            hand_strength += (full_hand[i] >> 2) / 100.0f;
                        }
                    }
                    hand_strength = fminf(1.0f, fmaxf(0.0f, hand_strength));
                    
                    // Calculate regret for each action
                    for (int action = 0; action < NUM_ACTIONS; action++) {
                        float action_value = evaluate_action_value(
                            action, player_payoff, hand_strength, player
                        );
                        
                        float regret = action_value - player_payoff;
                        regret_updates[action] += regret;
                    }
                    
                    updates_count++;
                }
            }
        }
        
        // Apply regret updates if we had any
        if (updates_count > 0) {
            for (int action = 0; action < NUM_ACTIONS; action++) {
                float avg_regret = regret_updates[action] / updates_count;
                regrets[info_set_idx * NUM_ACTIONS + action] += avg_regret;
                
                // Clamp regrets to reasonable range
                regrets[info_set_idx * NUM_ACTIONS + action] = fmaxf(
                    -1000.0f, fminf(1000.0f, regrets[info_set_idx * NUM_ACTIONS + action])
                );
            }
        }
    }
}

__global__ void cfr_strategy_update_kernel(
    float* regrets,   // [MAX_INFO_SETS, NUM_ACTIONS] - input
    float* strategy,  // [MAX_INFO_SETS, NUM_ACTIONS] - output
    int num_info_sets
) {
    /*
    Compute strategy from regrets using regret matching
    Each thread handles one info set
    */
    
    int info_set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (info_set_idx < num_info_sets) {
        float strategy_probs[NUM_ACTIONS];
        
        // Compute strategy using regret matching
        compute_regret_matching_strategy(
            regrets,
            strategy_probs,
            info_set_idx,
            NUM_ACTIONS
        );
        
        // Store strategy
        for (int action = 0; action < NUM_ACTIONS; action++) {
            strategy[info_set_idx * NUM_ACTIONS + action] = strategy_probs[action];
        }
    }
}

// ============================================================================
// 🚀 INITIALIZATION KERNELS
// ============================================================================

__global__ void init_random_states(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed + idx, idx, 0, &states[idx]);
    }
}

__global__ void init_regrets_and_strategy(
    float* regrets,
    float* strategy,
    int num_info_sets,
    int num_actions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_info_sets * num_actions;
    
    if (idx < total_elements) {
        regrets[idx] = 0.0f;
        strategy[idx] = 1.0f / num_actions; // Uniform initial strategy
    }
}

// ============================================================================
// 🚀 HOST INTERFACE FUNCTIONS
// ============================================================================

extern "C" {

void cuda_cfr_train_step(
    float* d_regrets,              // Device pointer
    float* d_strategy,             // Device pointer
    curandState* d_rand_states,    // Device pointer
    int* d_hole_cards,             // Device pointer - workspace
    int* d_community_cards,        // Device pointer - workspace
    float* d_payoffs,              // Device pointer - workspace
    int* d_action_histories,       // Device pointer - workspace
    float* d_pot_sizes,            // Device pointer - workspace
    int batch_size
) {
    /*
    Complete CFR training step on GPU
    */
    
    // Step 1: Simulate games
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    simulate_poker_games_gpu<<<blocks, threads_per_block>>>(
        d_rand_states,
        d_hole_cards,
        d_community_cards,
        d_payoffs,
        d_action_histories,
        d_pot_sizes,
        batch_size
    );
    
    // Step 2: Update regrets
    blocks = (MAX_INFO_SETS + threads_per_block - 1) / threads_per_block;
    
    cfr_regret_update_kernel<<<blocks, threads_per_block>>>(
        d_regrets,
        d_hole_cards,
        d_community_cards,
        d_payoffs,
        d_pot_sizes,
        batch_size
    );
    
    // Step 3: Update strategy
    cfr_strategy_update_kernel<<<blocks, threads_per_block>>>(
        d_regrets,
        d_strategy,
        MAX_INFO_SETS
    );
    
    // Synchronize
    cudaDeviceSynchronize();
}

void cuda_init_cfr_trainer(
    float** d_regrets,
    float** d_strategy,
    curandState** d_rand_states,
    int** d_hole_cards,
    int** d_community_cards,
    float** d_payoffs,
    int** d_action_histories,
    float** d_pot_sizes,
    int batch_size,
    unsigned long long seed
) {
    /*
    Initialize all GPU memory and data structures for CFR training
    */
    
    // Allocate main CFR data
    cudaMalloc(d_regrets, MAX_INFO_SETS * NUM_ACTIONS * sizeof(float));
    cudaMalloc(d_strategy, MAX_INFO_SETS * NUM_ACTIONS * sizeof(float));
    
    // Allocate random states
    cudaMalloc(d_rand_states, batch_size * sizeof(curandState));
    
    // Allocate game simulation workspace
    cudaMalloc(d_hole_cards, batch_size * MAX_PLAYERS * 2 * sizeof(int));
    cudaMalloc(d_community_cards, batch_size * 5 * sizeof(int));
    cudaMalloc(d_payoffs, batch_size * MAX_PLAYERS * sizeof(float));
    cudaMalloc(d_action_histories, batch_size * MAX_ACTIONS_PER_GAME * sizeof(int));
    cudaMalloc(d_pot_sizes, batch_size * sizeof(float));
    
    // Initialize data
    int threads_per_block = 256;
    int blocks;
    
    // Initialize random states
    blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    init_random_states<<<blocks, threads_per_block>>>(*d_rand_states, seed, batch_size);
    
    // Initialize regrets and strategy
    int total_elements = MAX_INFO_SETS * NUM_ACTIONS;
    blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    init_regrets_and_strategy<<<blocks, threads_per_block>>>(
        *d_regrets, *d_strategy, MAX_INFO_SETS, NUM_ACTIONS
    );
    
    cudaDeviceSynchronize();
}

void cuda_cleanup_cfr_trainer(
    float* d_regrets,
    float* d_strategy,
    curandState* d_rand_states,
    int* d_hole_cards,
    int* d_community_cards,
    float* d_payoffs,
    int* d_action_histories,
    float* d_pot_sizes
) {
    /*
    Clean up all GPU memory
    */
    
    cudaFree(d_regrets);
    cudaFree(d_strategy);
    cudaFree(d_rand_states);
    cudaFree(d_hole_cards);
    cudaFree(d_community_cards);
    cudaFree(d_payoffs);
    cudaFree(d_action_histories);
    cudaFree(d_pot_sizes);
}

} // extern "C" 