"""
🚀 GPU Trainer V4 - VECTORIZED OPERATIONS
==========================================
SOLUCIÓN FINAL: 
- ❌ Zero pure_callback
- ❌ Zero lax.cond anidadas (causan CPU fallback)
- ✅ Operaciones vectorizadas GPU-friendly
- ✅ jnp.where en lugar de lax.cond anidadas
- ✅ Máxima GPU utilization

PROBLEMA V3: lax.cond muy anidadas → CPU fallback
SOLUCIÓN V4: Operaciones vectorizadas puras
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import logging
from functools import partial

logger = logging.getLogger(__name__)

# ============================================================================
# 🚀 GPU HAND STRENGTH V4 - VECTORIZED OPERATIONS
# ============================================================================

@jax.jit
def gpu_hand_strength_v4(cards):
    """
    ✅ SOLUCIÓN V4: Hand strength 100% vectorizado
    - Zero pure_callback
    - Zero lax.cond anidadas
    - Pure jnp.where operations (GPU-friendly)
    """
    batch_size = cards.shape[0]
    
    # Convert cards to ranks and suits
    ranks = cards // 4  # 0-12 (2-A)
    suits = cards % 4   # 0-3 
    
    # Mask invalid cards
    valid_mask = cards >= 0
    ranks = jnp.where(valid_mask, ranks, -1)
    suits = jnp.where(valid_mask, suits, -1)
    
    def evaluate_single_hand_vectorized(hand_ranks, hand_suits, hand_valid):
        """Evaluate single hand using vectorized operations"""
        # Only consider valid cards
        valid_ranks = jnp.where(hand_valid, hand_ranks, -1)
        valid_suits = jnp.where(hand_valid, hand_suits, -1)
        
        # Count each rank using vectorized operations
        rank_counts = jnp.array([
            jnp.sum((valid_ranks == rank) & hand_valid) 
            for rank in range(13)
        ])
        
        # Count each suit using vectorized operations
        suit_counts = jnp.array([
            jnp.sum((valid_suits == suit) & hand_valid) 
            for suit in range(4)
        ])
        
        # Hand type detection (vectorized)
        max_rank_count = jnp.max(rank_counts)
        pairs = jnp.sum(rank_counts == 2)
        trips = jnp.sum(rank_counts == 3)  
        quads = jnp.sum(rank_counts == 4)
        is_flush = jnp.max(suit_counts) >= 5
        
        # Simplified straight detection
        has_ace = rank_counts[12] > 0
        has_king = rank_counts[11] > 0
        has_ten = rank_counts[8] > 0
        has_five = rank_counts[3] > 0
        has_two = rank_counts[0] > 0
        
        is_straight = (has_ace & has_king & has_ten) | (has_ace & has_five & has_two)
        
        # High card value
        high_card_value = jnp.max(jnp.where(hand_valid, valid_ranks, -1))
        
        # VECTORIZED HAND RANKING (NO lax.cond anidadas)
        # Use jnp.where cascading instead of nested lax.cond
        
        # Create condition masks
        is_straight_flush = is_flush & is_straight
        is_four_kind = quads > 0
        is_full_house = (trips > 0) & (pairs > 0)
        is_flush_only = is_flush & (~is_straight)
        is_straight_only = is_straight & (~is_flush)
        is_three_kind = (trips > 0) & (pairs == 0)
        is_two_pair = pairs >= 2
        is_one_pair = pairs == 1
        
        # Vectorized strength calculation (GPU-friendly)
        strength = jnp.where(
            is_straight_flush, 8000 + high_card_value * 10,
            jnp.where(
                is_four_kind, 7000 + high_card_value * 10,
                jnp.where(
                    is_full_house, 6000 + high_card_value * 10,
                    jnp.where(
                        is_flush_only, 5000 + high_card_value * 10,
                        jnp.where(
                            is_straight_only, 4000 + high_card_value * 10,
                            jnp.where(
                                is_three_kind, 3000 + high_card_value * 10,
                                jnp.where(
                                    is_two_pair, 2000 + high_card_value * 10,
                                    jnp.where(
                                        is_one_pair, 1000 + high_card_value * 10,
                                        high_card_value * 50  # High card
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        
        return jnp.clip(strength, 0, 9000).astype(jnp.int32)
    
    # Vectorize evaluation across batch
    return jax.vmap(evaluate_single_hand_vectorized)(ranks, suits, valid_mask)

# ============================================================================
# 🚀 VECTORIZED GAME SIMULATION V4
# ============================================================================

@jax.jit
def gpu_game_simulation_v4(keys):
    """
    ✅ GAME SIMULATION V4: Vectorized operations
    Replace nested lax.cond with vectorized alternatives
    """
    batch_size = len(keys)
    
    def simulate_single_game_v4(key):
        """Single game simulation V4 - Pure vectorized"""
        # Generate cards
        deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
        hole_cards = deck[:12].reshape((6, 2))
        community_cards = deck[12:17]
        
        # Create 7-card hands
        def make_7_card_hand(player_idx):
            player_hole = hole_cards[player_idx]
            full_hand = jnp.concatenate([player_hole, community_cards])
            padded = jnp.pad(full_hand, (0, max(0, 7 - len(full_hand))), constant_values=-1)
            return padded[:7]
        
        player_hands = jax.vmap(make_7_card_hand)(jnp.arange(6))
        
        # ✅ VECTORIZED HAND EVALUATION V4
        hand_strengths = gpu_hand_strength_v4(player_hands)
        
        # Generate action sequence with simplified logic
        key1, key2, key3 = jax.random.split(key, 3)
        max_actions = 48
        action_sequence = jnp.full(max_actions, -1, dtype=jnp.int32)
        
        def add_action_vectorized(carry, i):
            """Add actions with vectorized decision making"""
            action_seq, action_count = carry
            
            # Vectorized probability calculation
            cycle_length = 8
            street = (i // cycle_length) % 4
            player = i % 6
            
            # Decision factors (vectorized)
            hand_strength_norm = hand_strengths[player] / 9000.0
            position_factor = (player + 1) / 6.0
            street_factor = (street + 1) / 4.0
            
            # Random factor
            chaos_key = jax.random.fold_in(key1, i * 71 + player * 37)
            rand_factor = jax.random.uniform(chaos_key)
            
            # Composite decision score
            decision_score = (
                hand_strength_norm * 0.4 +
                position_factor * 0.2 +
                street_factor * 0.1 +
                rand_factor * 0.3
            )
            
            # VECTORIZED ACTION SELECTION (no nested lax.cond)
            action = jnp.where(
                decision_score < 0.15, 0,  # FOLD
                jnp.where(
                    decision_score < 0.35, 1,  # CHECK
                    jnp.where(
                        decision_score < 0.55, 2,  # CALL
                        jnp.where(
                            decision_score < 0.75, 3,  # BET
                            jnp.where(
                                decision_score < 0.9, 4,  # RAISE
                                5  # ALL_IN
                            )
                        )
                    )
                )
            )
            
            # Termination probability (vectorized)
            game_progress = i / (max_actions * 2)
            termination_chance = game_progress * 0.3
            
            early_term_key = jax.random.fold_in(key2, i)
            early_termination = jax.random.uniform(early_term_key) < termination_chance
            
            should_add = (action_count < max_actions) & (~early_termination)
            
            # Update sequence (vectorized)
            new_action_seq = jnp.where(
                should_add,
                action_seq.at[action_count].set(action),
                action_seq
            )
            
            new_count = jnp.where(should_add, action_count + 1, action_count)
            
            return (new_action_seq, new_count), None
        
        # Generate action sequence
        total_iterations = max_actions * 2
        (final_action_seq, final_count), _ = jax.lax.scan(
            add_action_vectorized,
            (action_sequence, 0),
            jnp.arange(total_iterations)
        )
        
        # Calculate payoffs (vectorized)
        winner_key = jax.random.fold_in(key1, 999)
        
        # Winner selection (vectorized)
        deterministic_winner = jnp.argmax(hand_strengths)
        random_winner = jax.random.randint(winner_key, (), 0, 6)
        
        should_be_deterministic = jax.random.uniform(winner_key) < 0.8
        winner = jnp.where(should_be_deterministic, deterministic_winner, random_winner)
        
        # Pot calculation
        valid_actions = jnp.sum(final_action_seq >= 0)
        pot_size = 15.0 + valid_actions * 5.0
        
        # Payoffs (vectorized)
        base_contribution = pot_size / 8.0
        payoffs = jnp.full(6, -base_contribution)
        payoffs = payoffs.at[winner].set(pot_size - base_contribution)
        
        return {
            'payoffs': payoffs,
            'action_hist': final_action_seq,
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'hand_strengths': hand_strengths,
            'final_pot': pot_size,
        }
    
    # Vectorize across batch
    game_results = jax.vmap(simulate_single_game_v4)(keys)
    
    return game_results['payoffs'], game_results['action_hist'], game_results

# ============================================================================
# 🚀 VECTORIZED INFO SET V4
# ============================================================================

@jax.jit
def compute_info_set_v4(game_results, player_idx, game_idx):
    """
    ✅ INFO SET V4: Vectorized computation
    """
    hole_cards = game_results['hole_cards'][game_idx, player_idx]
    community_cards = game_results['community_cards'][game_idx]
    hand_strength = game_results['hand_strengths'][game_idx, player_idx]
    
    # Extract features (vectorized)
    hole_ranks = hole_cards // 4
    hole_suits = hole_cards % 4
    
    # Hand classification (vectorized)
    high_rank = jnp.maximum(hole_ranks[0], hole_ranks[1])
    low_rank = jnp.minimum(hole_ranks[0], hole_ranks[1])
    is_suited = (hole_suits[0] == hole_suits[1]).astype(jnp.int32)
    is_pair = (hole_ranks[0] == hole_ranks[1]).astype(jnp.int32)
    
    # Hand bucket (vectorized - no nested lax.cond)
    hand_bucket = jnp.where(
        is_pair == 1,
        high_rank,  # Pairs: 0-12
        jnp.where(
            is_suited == 1,
            13 + high_rank * 12 + low_rank,  # Suited
            169 + high_rank * 12 + low_rank  # Offsuit
        )
    )
    hand_bucket = jnp.mod(hand_bucket, 169)
    
    # Street detection (vectorized)
    num_community = jnp.sum(community_cards >= 0)
    street_bucket = jnp.where(
        num_community == 0, 0,  # Preflop
        jnp.where(
            num_community == 3, 1,  # Flop
            jnp.where(num_community == 4, 2, 3)  # Turn/River
        )
    )
    
    # Other buckets
    position_bucket = player_idx
    strength_bucket = jnp.clip(hand_strength // 500, 0, 17).astype(jnp.int32)
    pot_bucket = jnp.clip(game_results['final_pot'][game_idx] // 10, 0, 9).astype(jnp.int32)
    
    # Combine info set ID
    info_set_id = (
        street_bucket * 10000 +
        hand_bucket * 50 +
        position_bucket * 8 +
        strength_bucket * 2 +
        pot_bucket
    )
    
    return jnp.mod(info_set_id, 50000).astype(jnp.int32)

# ============================================================================
# 🚀 VECTORIZED CFR STEP V4
# ============================================================================

@jax.jit
def cfr_training_step_v4(regrets, strategy, key):
    """
    ✅ CFR STEP V4: Vectorized CFR with no nested conditionals
    """
    batch_size = 128
    num_actions = 6
    max_info_sets = 50000
    
    keys = jax.random.split(key, batch_size)
    
    # Get game data V4 (vectorized)
    payoffs, histories, game_results = gpu_game_simulation_v4(keys)
    
    def process_single_game_v4(game_idx):
        """Process single game V4 - Vectorized CFR"""
        game_payoff = payoffs[game_idx]
        game_regrets = jnp.zeros_like(regrets)
        
        def update_regrets_for_player_vectorized(current_regrets, player_idx):
            """Update regrets - vectorized version"""
            info_set_idx = compute_info_set_v4(game_results, player_idx, game_idx)
            player_payoff = game_payoff[player_idx]
            
            def calculate_action_regret_vectorized(action):
                """Vectorized action regret calculation"""
                expected_value = player_payoff
                
                hand_strength = game_results['hand_strengths'][game_idx, player_idx]
                normalized_hand_strength = hand_strength / 9000.0
                
                # VECTORIZED synergy calculation (no nested lax.cond)
                hand_action_synergy = jnp.where(
                    action == 0, 0.1,  # FOLD
                    jnp.where(
                        action <= 2, 0.3 + normalized_hand_strength * 0.4,  # CHECK/CALL
                        0.5 + normalized_hand_strength * 0.5  # BET/RAISE/ALL_IN
                    )
                )
                
                # VECTORIZED outcome factor (no nested lax.cond)
                outcome_factor = jnp.where(
                    player_payoff > 0,  # Won
                    jnp.where(
                        action >= 3, 1.5,  # Aggressive when winning
                        jnp.where(action == 0, 0.2, 1.0)  # Fold when winning vs other
                    ),
                    jnp.where(  # Lost
                        action == 0, 0.8,  # Fold when losing
                        jnp.where(action >= 3, 0.3, 0.6)  # Aggressive vs passive when losing
                    )
                )
                
                # Action value calculation
                action_value = player_payoff * hand_action_synergy * outcome_factor
                
                # Special fold handling (vectorized)
                adjusted_action_value = jnp.where(
                    action == 0,  # FOLD
                    jnp.where(
                        player_payoff < 0, -player_payoff * 0.1,  # Save 90% when losing
                        -2.0  # Penalty for folding winner
                    ),
                    action_value
                )
                
                regret = adjusted_action_value - expected_value
                return jnp.clip(regret, -100.0, 100.0)
            
            # Calculate regrets for all actions (vectorized)
            action_regrets = jax.vmap(calculate_action_regret_vectorized)(jnp.arange(num_actions))
            
            # Update regrets
            return current_regrets.at[info_set_idx].add(action_regrets)
        
        # Update regrets for all players
        final_regrets = game_regrets
        for player_idx in range(6):
            final_regrets = update_regrets_for_player_vectorized(final_regrets, player_idx)
        
        return final_regrets
    
    # Process all games in batch
    batch_regrets = jax.vmap(process_single_game_v4)(jnp.arange(batch_size))
    
    # Accumulate regrets
    accumulated_regrets = regrets + jnp.sum(batch_regrets, axis=0)
    
    # Strategy update (vectorized)
    positive_regrets = jnp.maximum(accumulated_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    
    new_strategy = jnp.where(
        regret_sums > 1e-6,
        positive_regrets / regret_sums,
        jnp.ones((max_info_sets, num_actions)) / num_actions
    )
    
    return accumulated_regrets, new_strategy

# ============================================================================
# 🚀 GPU TRAINER V4 - VECTORIZED
# ============================================================================

class GPUTrainerV4:
    """
    ✅ GPU TRAINER V4: Maximum vectorization
    - Zero pure_callback
    - Zero nested lax.cond 
    - Pure vectorized operations
    - Maximum GPU utilization expected
    """
    
    def __init__(self):
        self.max_info_sets = 50000
        self.num_actions = 6
        self.regrets = jnp.zeros((self.max_info_sets, self.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones((self.max_info_sets, self.num_actions), dtype=jnp.float32) / self.num_actions
    
    def train(self, num_iterations, verbose=True):
        """Train V4 - Pure vectorized training"""
        if verbose:
            print("\n🚀 INICIANDO GPU TRAINING V4 - VECTORIZED")
            print("   ❌ Zero callbacks")
            print("   ❌ Zero nested lax.cond")
            print("   ✅ Pure vectorized operations") 
            print("   ✅ Maximum GPU utilization expected")
            print(f"   Total iteraciones: {num_iterations}")
        
        key = jax.random.PRNGKey(42)
        start_time = time.time()
        
        for i in range(1, num_iterations + 1):
            iter_key = jax.random.fold_in(key, i)
            
            iter_start = time.time()
            
            # ✅ V4 VECTORIZED TRAINING STEP
            self.regrets, self.strategy = cfr_training_step_v4(
                self.regrets,
                self.strategy, 
                iter_key
            )
            
            # Wait for completion
            self.regrets.block_until_ready()
            
            iter_time = time.time() - iter_start
            
            # Progress logging
            if verbose and (i % max(1, num_iterations // 10) == 0):
                progress = 100 * i / num_iterations
                print(f"🚀 V4 Progress: {progress:.0f}% ({i}/{num_iterations}) - {iter_time:.2f}s")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n🎉 GPU TRAINING V4 COMPLETADO!")
            print(f"   ⏰ Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"   🚀 Velocidad final: {num_iterations/total_time:.2f} it/s")
            print(f"   💎 Pure vectorized operations!")
        
        return {
            'total_time': total_time,
            'speed': num_iterations / total_time,
            'regrets': self.regrets,
            'strategy': self.strategy
        }

# ============================================================================
# 🚀 TESTING FUNCTIONS V4
# ============================================================================

def test_gpu_v4_speed(iterations=20):
    """Test GPU V4 speed - vectorized operations"""
    print(f"\n🚀 TEST GPU V4 SPEED ({iterations} iterations)")
    print("="*50)
    print("Vectorized operations = Maximum GPU expected")
    
    trainer = GPUTrainerV4()
    results = trainer.train(iterations, verbose=True)
    
    print(f"\n⚡ RESULTADO V4:")
    print(f"   - Tiempo: {results['total_time']:.2f}s")
    print(f"   - Velocidad: {results['speed']:.1f} it/s") 
    print(f"   - Throughput: ~{results['speed'] * 128 * 50:.0f} hands/s")
    print(f"   - ¿Ultra fast? {'✅ SÍ' if results['speed'] > 100 else '❌ NO'} (>100 it/s)")
    
    return results

def test_gpu_v4_learning(iterations=50):
    """Test GPU V4 learning capability"""
    print(f"\n🧠 TEST GPU V4 LEARNING ({iterations} iterations)")
    print("="*50)
    
    trainer = GPUTrainerV4()
    
    # Initial state
    initial_std = float(jnp.std(trainer.strategy))
    print(f"📊 ESTADO INICIAL:")
    print(f"   - Strategy STD: {initial_std:.6f}")
    
    # Train
    results = trainer.train(iterations, verbose=True)
    
    # Final analysis  
    final_std = float(jnp.std(trainer.strategy))
    strategy_change = final_std - initial_std
    regrets_nonzero = int(jnp.sum(trainer.regrets != 0))
    
    print(f"\n🧠 ANÁLISIS DE APRENDIZAJE:")
    print(f"   - Strategy STD final: {final_std:.6f}")
    print(f"   - Cambio total: {strategy_change:.6f}")
    print(f"   - Regrets no-cero: {regrets_nonzero:,}")
    
    # Learning criteria
    change_detected = strategy_change > 1e-4
    significant_change = strategy_change > 1e-3
    diversification = final_std > 1e-5
    substantial_regrets = regrets_nonzero > 1000
    
    print(f"\n📈 CRITERIOS DE APRENDIZAJE:")
    print(f"   - Cambio detectado: {'✅ SÍ' if change_detected else '❌ NO'} (>1e-04)")
    print(f"   - Cambio significativo: {'✅ SÍ' if significant_change else '❌ NO'} (>1e-03)")
    print(f"   - Diversificación: {'✅ SÍ' if diversification else '❌ NO'} (>1e-5)")
    print(f"   - Regrets sustanciales: {'✅ SÍ' if substantial_regrets else '❌ NO'} (>1000)")
    
    # Overall verdict
    learning_success = change_detected and diversification and substantial_regrets
    
    print(f"\n🏆 VEREDICTO V4:")
    if learning_success:
        print("   ✅ APRENDIZAJE EXITOSO")
        print("   ✅ Vectorized operations working")
    else:
        print("   ❌ Problemas de aprendizaje detectados")
    
    return results, learning_success

def main():
    """Main testing function V4"""
    print("🚀 GPU TRAINER V4 - VECTORIZED OPERATIONS TESTING")
    print("="*60)
    print("OBJETIVO: Pure vectorized = Maximum GPU utilization")
    
    # Test 1: Speed
    print("\n" + "="*60)
    print("📊 TEST 1: VELOCIDAD V4")
    speed_results = test_gpu_v4_speed(20)
    
    # Test 2: Learning
    print("\n" + "="*60) 
    print("📊 TEST 2: APRENDIZAJE V4")
    learning_results, learning_ok = test_gpu_v4_learning(50)
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESUMEN V4")
    print("="*60)
    print(f"🚀 Velocidad: {speed_results['speed']:.1f} it/s")
    print(f"🧠 Aprendizaje: {'✅ EXITOSO' if learning_ok else '❌ FALLO'}")
    
    if speed_results['speed'] > 100 and learning_ok:
        print("\n🏆 GPU TRAINER V4: ✅ ÉXITO TOTAL")
        print("   - Velocidad ultra alta (>100 it/s)")
        print("   - Aprendizaje funcionando")
        print("   - Pure vectorized operations")
        print("   - ¡CPU fallback ELIMINADO!")
    else:
        print("\n🔧 GPU TRAINER V4: Partial success")
        print(f"   - Velocidad: {'✅' if speed_results['speed'] > 50 else '❌'}")
        print(f"   - Aprendizaje: {'✅' if learning_ok else '❌'}")

if __name__ == "__main__":
    main() 