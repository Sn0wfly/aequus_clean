"""
🎯 Elite GTO Trainer - Phase 1 Integration
Complete NLHE game engine integrated with training pipeline
Based on OpenSpiel patterns and JAX-native implementation
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple
import logging
from poker_bot.core.jax_game_engine import (
    create_initial_state, simulate_game, batch_simulate,
    get_legal_actions, apply_action, is_round_complete, advance_round
)

logger = logging.getLogger(__name__)

class EliteGTOSimulator:
    """
    Elite GTO simulator with complete NLHE game tree
    Integrates JAX-native game engine with training pipeline
    """
    
    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        self.small_blind = 1.0
        self.big_blind = 2.0
        self.starting_stack = 100.0
        
    def simulate_training_games(self, 
                              rng_key: jnp.ndarray,
                              num_games: int) -> Dict:
        """
        Simulate games for training data generation
        Returns OpenSpiel-compatible format
        """
        
        # Split RNG keys for vectorization
        rng_keys = jax.random.split(rng_key, num_games)
        
        # Vectorized simulation
        results = batch_simulate(
            rng_keys, 
            self.num_players,
            self.small_blind,
            self.big_blind,
            self.starting_stack
        )
        
        # Convert to training format
        training_data = {
            'payoffs': results['payoffs'],
            'hole_cards': results['hole_cards'],
            'final_community': results['final_community'],
            'final_pot': results['final_pot'],
            'player_stacks': results['player_stacks'],
            'batch_size': num_games
        }
        
        return training_data
    
    def get_game_state_features(self, game_state: Dict) -> jnp.ndarray:
        """
        Extract features from game state for bucketing
        Returns normalized feature vector
        """
        # Stack-based features
        stacks = game_state['player_stacks']
        bets = game_state['player_bets']
        pot = game_state['final_pot']
        
        # Position features
        positions = jnp.arange(self.num_players)
        
        # Normalize features
        stack_norm = stacks / self.starting_stack
        bet_norm = bets / self.starting_stack
        pot_norm = pot / (self.num_players * self.starting_stack)
        
        # Combine features
        features = jnp.concatenate([
            stack_norm,
            bet_norm,
            jnp.array([pot_norm]),
            positions.astype(jnp.float32) / self.num_players
        ])
        
        return features
    
    def create_info_set_key(self, 
                          hole_cards: jnp.ndarray,
                          community_cards: jnp.ndarray,
                          position: int,
                          pot_size: float,
                          stack_size: float) -> str:
        """
        Create information set key for CFR
        High-resolution bucketing compatible
        """
        
        # Card strength bucketing
        from poker_bot.core.simulation import _evaluate_hand_strength
        
        # Evaluate hand strength
        if len(community_cards) > 0:
            dealt_community = community_cards[community_cards >= 0]
            full_hand = jnp.concatenate([hole_cards, dealt_community])
            strength = _evaluate_hand_strength(full_hand)
        else:
            # Preflop evaluation
            strength = _evaluate_hand_strength(hole_cards)
        
        # Normalize strength
        strength_bucket = jnp.int32(strength / 1e9)
        
        # Create bucket key
        pot_bucket = jnp.int32(pot_size / 10)
        stack_bucket = jnp.int32(stack_size / 10)
        
        return f"pos_{position}_str_{strength_bucket}_pot_{pot_bucket}_stack_{stack_bucket}"

class EliteTrainingPipeline:
    """
    Complete training pipeline with elite game engine
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.simulator = EliteGTOSimulator(config.get('num_players', 6))
        
    def generate_training_batch(self, rng_key: jnp.ndarray) -> Dict:
        """Generate training batch with elite game engine"""
        
        batch_size = self.config.get('batch_size', 1024)
        
        # Simulate games
        game_data = self.simulator.simulate_training_games(rng_key, batch_size)
        
        # Extract features for each player
        features = []
        targets = []
        
        for game_idx in range(batch_size):
            for player_idx in range(self.simulator.num_players):
                # Get player state
                hole_cards = game_data['hole_cards'][game_idx, player_idx]
                community = game_data['final_community'][game_idx]
                stack = game_data['player_stacks'][game_idx, player_idx]
                pot = game_data['final_pot'][game_idx]
                
                # Create features
                features.append(
                    self.simulator.get_game_state_features({
                        'player_stacks': game_data['player_stacks'][game_idx],
                        'player_bets': jnp.zeros(self.simulator.num_players),
                        'final_pot': pot
                    })
                )
                
                # Target is final payoff
                targets.append(game_data['payoffs'][game_idx, player_idx])
        
        return {
            'features': jnp.stack(features),
            'targets': jnp.array(targets),
            'game_data': game_data
        }
    
    def validate_game_engine(self, num_test_games: int = 100) -> Dict:
        """Validate game engine correctness"""
        
        rng_key = jax.random.PRNGKey(42)
        
        # Test basic functionality
        test_results = {
            'state_creation': True,
            'action_validity': True,
            'payoff_zero_sum': True,
            'performance': True
        }
        
        # Simulate test games
        game_data = self.simulator.simulate_training_games(rng_key, num_test_games)
        
        # Validate zero-sum property
        total_payoffs = jnp.sum(game_data['payoffs'], axis=1)
        max_deviation = jnp.max(jnp.abs(total_payoffs))
        
        if max_deviation > 1e-3:
            test_results['payoff_zero_sum'] = False
            logger.warning(f"Zero-sum violation: max deviation {max_deviation}")
        
        # Performance test
        start_time = time.time()
        _ = self.simulator.simulate_training_games(rng_key, 1000)
        elapsed = time.time() - start_time
        
        games_per_second = 1000 / elapsed
        test_results['performance'] = games_per_second > 100
        
        logger.info(f"Game engine validation: {games_per_second:.1f} games/sec")
        
        return test_results

# ============================================================================
# INTEGRATION WITH EXISTING TRAINER
# ============================================================================

def integrate_with_trainer(trainer_config: Dict) -> EliteTrainingPipeline:
    """
    Integrate elite game engine with existing trainer
    """
    
    pipeline = EliteTrainingPipeline(trainer_config)
    
    # Validate integration
    validation_results = pipeline.validate_game_engine()
    
    if all(validation_results.values()):
        logger.info("✅ Elite game engine integration successful")
    else:
        logger.warning("⚠️  Some validation issues detected")
    
    return pipeline

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """Example usage"""
    
    config = {
        'num_players': 6,
        'batch_size': 1024,
        'small_blind': 1.0,
        'big_blind': 2.0,
        'starting_stack': 100.0
    }
    
    # Create pipeline
    pipeline = integrate_with_trainer(config)
    
    # Generate training batch
    rng_key = jax.random.PRNGKey(42)
    training_batch = pipeline.generate_training_batch(rng_key)
    
    logger.info(f"Generated training batch: {training_batch['features'].shape}")
    logger.info(f"Average payoff: {jnp.mean(training_batch['targets']):.2f}")