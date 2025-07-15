"""
🎯 Elite Game Engine Validation Script
Tests the complete NLHE implementation based on OpenSpiel patterns
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Dict, List
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poker_bot.core.elite_game_engine import EliteGameEngine, GameState, PlayerAction, BettingRound
from poker_bot.core.betting_tree import EliteBettingTree, TreeAnalyzer
from poker_bot.core.game_state import GameStateReconstructor, StateValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameEngineValidator:
    """Comprehensive validation suite for the elite game engine"""
    
    def __init__(self):
        self.engine = EliteGameEngine(num_players=6)
        self.tree = EliteBettingTree()
        self.reconstructor = GameStateReconstructor()
        
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all validation tests"""
        results = {}
        
        logger.info("🚀 Starting Elite Game Engine Validation")
        logger.info("=" * 60)
        
        # Test 1: Basic game initialization
        results['game_initialization'] = self.test_game_initialization()
        
        # Test 2: Action application
        results['action_application'] = self.test_action_application()
        
        # Test 3: Side pot calculation
        results['side_pots'] = self.test_side_pots()
        
        # Test 4: All-in scenarios
        results['all_in_scenarios'] = self.test_all_in_scenarios()
        
        # Test 5: Betting tree construction
        results['betting_tree'] = self.test_betting_tree()
        
        # Test 6: State reconstruction
        results['state_reconstruction'] = self.test_state_reconstruction()
        
        # Test 7: Performance benchmark
        results['performance'] = self.test_performance()
        
        # Test 8: OpenSpiel compatibility
        results['openspiel_compat'] = self.test_openspiel_compatibility()
        
        # Summary
        self.print_summary(results)
        
        return results
    
    def test_game_initialization(self) -> bool:
        """Test game initialization"""
        logger.info("Testing game initialization...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            starting_stacks = jnp.array([100.0] * 6)
            
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            
            # Validate basic properties
            assert len(game_state.players) == 6
            assert game_state.pot == 3.0  # SB + BB
            assert game_state.current_round == BettingRound.PREFLOP
            assert game_state.current_player == 2  # UTG
            
            # Validate player states
            for i, player in enumerate(game_state.players):
                assert player.player_id == i
                assert len(player.hole_cards) == 2
                assert player.stack >= 0
            
            logger.info("✅ Game initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Game initialization test failed: {e}")
            return False
    
    def test_action_application(self) -> bool:
        """Test action application"""
        logger.info("Testing action application...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            starting_stacks = jnp.array([100.0] * 6)
            
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            
            # Test fold action
            new_state = self.engine.apply_action(game_state, PlayerAction.FOLD)
            assert new_state.players[2].is_folded
            
            # Test call action
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            new_state = self.engine.apply_action(game_state, PlayerAction.CALL, 2.0)
            assert new_state.players[2].current_bet == 2.0
            
            # Test raise action
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            new_state = self.engine.apply_action(game_state, PlayerAction.RAISE, 6.0)
            assert new_state.players[2].current_bet == 6.0
            
            logger.info("✅ Action application test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Action application test failed: {e}")
            return False
    
    def test_side_pots(self) -> bool:
        """Test side pot calculation"""
        logger.info("Testing side pot calculation...")
        
        try:
            # Create scenario with different all-in amounts
            rng_key = jax.random.PRNGKey(42)
            starting_stacks = jnp.array([50.0, 75.0, 100.0, 100.0, 100.0, 100.0])
            
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            
            # Simulate all-in scenarios
            players = game_state.players
            players[0].total_bet = 50.0  # Short stack all-in
            players[1].total_bet = 75.0  # Medium stack all-in
            players[2].total_bet = 100.0  # Deep stack
            
            side_pots = self.engine._calculate_side_pots(players)
            
            # Validate side pots
            assert len(side_pots) >= 2  # Should have multiple side pots
            
            total_pot = sum(pot.amount for pot in side_pots)
            expected_total = 50.0 + 75.0 + 100.0 + 2.0 + 1.0  # Including blinds
            
            assert abs(total_pot - expected_total) < 1e-6
            
            logger.info("✅ Side pot calculation test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Side pot calculation test failed: {e}")
            return False
    
    def test_all_in_scenarios(self) -> bool:
        """Test all-in scenarios"""
        logger.info("Testing all-in scenarios...")
        
        try:
            rng_key = jax.random.PRNGKey(42)
            starting_stacks = jnp.array([10.0, 100.0, 100.0, 100.0, 100.0, 100.0])
            
            game_state = self.engine._initialize_game(rng_key, starting_stacks)
            
            # Force all-in scenario
            new_state = self.engine.apply_action(game_state, PlayerAction.ALL_IN)
            
            # Validate all-in state
            player = new_state.players[2]
            assert player.is_all_in
            assert player.stack == 0.0
            assert player.current_bet == 10.0
            
            logger.info("✅ All-in scenarios test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ All-in scenarios test failed: {e}")
            return False
    
    def test_betting_tree(self) -> bool:
        """Test betting tree construction"""
        logger.info("Testing betting tree construction...")
        
        try:
            # Build betting tree
            root = self.tree.build_tree(num_players=6)
            
            # Analyze tree
            counts = TreeAnalyzer.count_nodes(root)
            
            # Validate tree structure
            assert counts['total'] > 0
            assert counts['player'] > 0
            assert counts['terminal'] > 0
            
            logger.info(f"✅ Betting tree test passed - {counts['total']} nodes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Betting tree test failed: {e}")
            return False
    
    def test_state_reconstruction(self) -> bool:
        """Test state reconstruction"""
        logger.info("Testing state reconstruction...")
        
        try:
            # Create test game data
            game_data = {
                'hole_cards': jnp.array([
                    [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]
                ]),
                'community_cards': jnp.array([-1, -1, -1, -1, -1]),
                'action_history': [
                    (2, PlayerAction.CALL, 2.0),
                    (3, PlayerAction.RAISE, 6.0),
                    (4, PlayerAction.FOLD, 0.0)
                ],
                'starting_stacks': jnp.array([100.0] * 6)
            }
            
            # Reconstruct states
            states = self.reconstructor.reconstruct_from_game_data(game_data)
            
            # Validate states
            assert len(states) == 3  # One state per action
            
            for state in states:
                assert StateValidator.validate_state(state)
            
            logger.info("✅ State reconstruction test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ State reconstruction test failed: {e}")
            return False
    
    def test_performance(self) -> bool:
        """Test performance benchmarks"""
        logger.info("Testing performance...")
        
        try:
            # Benchmark game simulation
            start_time = time.time()
            
            num_games = 1000
            rng_key = jax.random.PRNGKey(42)
            
            for i in range(num_games):
                key = jax.random.fold_in(rng_key, i)
                starting_stacks = jnp.array([100.0] * 6)
                game_data = self.engine.simulate_game(key, starting_stacks)
            
            elapsed = time.time() - start_time
            games_per_second = num_games / elapsed
            
            logger.info(f"✅ Performance test passed - {games_per_second:.1f} games/sec")
            return games_per_second > 100  # At least 100 games/sec
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            return False
    
    def test_openspiel_compatibility(self) -> bool:
        """Test OpenSpiel compatibility"""
        logger.info("Testing OpenSpiel compatibility...")
        
        try:
            # Test state format compatibility
            rng_key = jax.random.PRNGKey(42)
            starting_stacks = jnp.array([100.0] * 6)
            
            game_data = self.engine.simulate_game(rng_key, starting_stacks)
            
            # Validate required fields
            required_fields = ['payoffs', 'final_community', 'hole_cards', 'final_pot']
            for field in required_fields:
                assert field in game_data
            
            # Validate tensor shapes
            assert game_data['payoffs'].shape == (6,)
            assert game_data['hole_cards'].shape == (6, 2)
            assert game_data['final_community'].shape == (5,)
            
            logger.info("✅ OpenSpiel compatibility test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenSpiel compatibility test failed: {e}")
            return False
    
    def print_summary(self, results: Dict[str, bool]):
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
            logger.info("🎉 ALL TESTS PASSED - Elite Game Engine Ready!")
        else:
            logger.warning("⚠️  Some tests failed - Review and fix issues")

def main():
    """Main validation script"""
    validator = GameEngineValidator()
    results = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()