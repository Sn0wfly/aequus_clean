"""
🎯 JAX-Native Game Engine Validation - Simplified Version
Tests the complete NLHE implementation with JAX compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poker_bot.core.jax_game_engine import (
    create_initial_state, get_legal_actions, apply_action, 
    is_round_complete, advance_round, simulate_game, batch_simulate,
    PlayerAction
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JAXGameValidator:
    """Validation for JAX-native game engine"""
    
    def run_all_tests(self) -> dict:
        """Run all validation tests"""
        results = {}
        
        logger.info("🚀 Starting JAX-Native Game Engine Validation")
        logger.info("=" * 60)
        
        # Test 1: State creation
        results['state_creation'] = self.test_state_creation()
        
        # Test 2: Legal actions
        results['legal_actions'] = self.test_legal_actions()
        
        # Test 3: Action application
        results['action_application'] = self.test_action_application()
        
        # Test 4: Round advancement
        results['round_advancement'] = self.test_round_advancement()
        
        # Test 5: Game completion
        results['game_completion'] = self.test_game_completion()
        
        # Test 6: Vectorized simulation
        results['vectorized_simulation'] = self.test_vectorized_simulation()
        
        # Test 7: Performance benchmark
        results['performance'] = self.test_performance()
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def test_state_creation(self) -> bool:
        """Test initial state creation"""
        logger.info("Testing state creation...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            state = create_initial_state(rng_key, 6, 1.0, 2.0, 100.0)
            
            # Validate dimensions
            assert state['player_stacks'].shape == (6,)
            assert state['player_bets'].shape == (6,)
            assert state['hole_cards'].shape == (6, 2)
            assert state['community_cards'].shape == (5,)
            
            # Validate initial values
            assert state['pot'] == 3.0  # SB + BB
            assert state['current_player'] == 2
            assert state['round'] == 0
            
            logger.info("✅ State creation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ State creation test failed: {e}")
            return False
    
    def test_legal_actions(self) -> bool:
        """Test legal actions calculation"""
        logger.info("Testing legal actions...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            state = create_initial_state(rng_key, 6, 1.0, 2.0, 100.0)
            
            # Get legal actions for UTG
            legal_actions = get_legal_actions(state)
            
            # Should be able to fold, call, or raise
            assert legal_actions[PlayerAction.FOLD] == True
            assert legal_actions[PlayerAction.CALL] == True
            assert legal_actions[PlayerAction.RAISE] == True
            assert legal_actions[PlayerAction.CHECK] == False
            
            logger.info("✅ Legal actions test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Legal actions test failed: {e}")
            return False
    
    def test_action_application(self) -> bool:
        """Test action application"""
        logger.info("Testing action application...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            state = create_initial_state(rng_key, 6, 1.0, 2.0, 100.0)
            
            # Test call action
            new_state = apply_action(state, PlayerAction.CALL, 2.0)
            
            # Check stack and bet updates
            assert new_state['player_stacks'][2] == 98.0  # Stack decreased
            assert new_state['player_bets'][2] == 2.0     # Current bet matches
            assert new_state['pot'] > 3.0                 # Pot increased
            
            logger.info("✅ Action application test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Action application test failed: {e}")
            return False
    
    def test_round_advancement(self) -> bool:
        """Test betting round advancement"""
        logger.info("Testing round advancement...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            state = create_initial_state(rng_key, 6, 1.0, 2.0, 100.0)
            
            # Simulate complete preflop round (simplified)
            # All players call
            for i in range(6):
                if not state['player_folded'][i] and not state['player_all_in'][i]:
                    state = apply_action(state, PlayerAction.CALL, 2.0)
            
            # Advance round
            new_state = advance_round(state)
            
            # Check round advancement
            assert new_state['round'] == 1  # FLOP
            assert jnp.sum(new_state['community_cards'][:3] >= 0) == 3  # 3 flop cards
            
            logger.info("✅ Round advancement test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Round advancement test failed: {e}")
            return False
    
    def test_game_completion(self) -> bool:
        """Test game completion scenarios"""
        logger.info("Testing game completion...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            
            # Test different scenarios
            scenarios = [
                (6, 100.0),  # Standard 6-max
                (3, 50.0),   # Short-handed
                (2, 200.0),  # Heads-up
            ]
            
            for num_players, stack in scenarios:
                result = simulate_game(rng_key, num_players, 1.0, 2.0, stack)
                
                # Validate completion
                assert result['payoffs'].shape[0] == num_players
                assert jnp.abs(jnp.sum(result['payoffs'])) < 1.0  # Zero-sum (within tolerance)
            
            logger.info("✅ Game completion test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Game completion test failed: {e}")
            return False
    
    def test_vectorized_simulation(self) -> bool:
        """Test vectorized game simulation"""
        logger.info("Testing vectorized simulation...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            rng_keys = jax.random.split(rng_key, 10)
            
            # Vectorized simulation
            results = batch_simulate(rng_keys, 6, 1.0, 2.0, 100.0)
            
            # Validate results
            assert 'payoffs' in results
            assert 'final_community' in results
            assert 'hole_cards' in results
            assert results['payoffs'].shape == (10, 6)
            
            logger.info("✅ Vectorized simulation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vectorized simulation test failed: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance benchmark"""
        logger.info("Testing performance...")
        
        try:
            # Benchmark single game
            rng_key = jax.random.PRNGKey(42)
            
            # JIT compile
            jitted_simulate = jax.jit(
                lambda key: simulate_game(key, 6, 1.0, 2.0, 100.0)
            )
            
            # Warm up
            _ = jitted_simulate(rng_key)
            
            # Benchmark
            start_time = time.time()
            num_games = 1000
            
            for i in range(num_games):
                key = jax.random.fold_in(rng_key, i)
                _ = jitted_simulate(key)
            
            elapsed = time.time() - start_time
            games_per_second = num_games / elapsed
            
            logger.info(f"✅ Performance test passed - {games_per_second:.1f} games/sec")
            return games_per_second > 100  # Target 100+ games/sec
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    def print_summary(self, results: dict):
        """Print validation summary"""
        logger.info("\n" + "=" * 60)
        logger.info("🎯 VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            logger.info(f"{test_name:20} {status}")
        
        logger.info("-" * 60)
        logger.info(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - JAX-Native Game Engine Ready!")
        else:
            logger.warning("⚠️  Some tests failed - Review and fix issues")

def main():
    """Main validation script"""
    validator = JAXGameValidator()
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()