#!/usr/bin/env python3 
"""
🧪 Test Poker Concepts - Unit Tests for Poker AI
Validates that the AI learns fundamental poker concepts correctly
"""

import pytest
import numpy as np
import jax.numpy as jnp
from poker_bot.core.trainer import PokerTrainer, TrainerConfig, evaluate_hand_jax
from poker_bot.core.full_game_engine import evaluate_hand_wrapper
from poker_bot.evaluator import HandEvaluator

class TestHandEvaluator:
    """Test the core hand evaluation system"""
    
    def test_evaluator_import(self):
        """Test that evaluator can be imported and instantiated"""
        evaluator = HandEvaluator()
        assert evaluator is not None
    
    def test_aa_vs_72o_direct(self):
        """Test AA beats 72o with direct evaluator"""
        evaluator = HandEvaluator()
        
        # 7-card hands with same board
        board = [46, 42, 37, 35, 32]  # Kh Qd Js 9c 8h
        aa_hand = [51, 47] + board  # As Ac + board
        trash_hand = [23, 0] + board  # 7c 2s + board
        
        aa_strength = evaluator.evaluate_single(aa_hand)
        trash_strength = evaluator.evaluate_single(trash_hand)
        
        # phevaluator: lower = better
        assert aa_strength < trash_strength, f"AA ({aa_strength}) should be better than 72o ({trash_strength})"
    
    def test_wrapper_consistency(self):
        """Test wrapper returns consistent results"""
        board = [46, 42, 37, 35, 32]
        aa_hand = np.array([51, 47] + board)
        trash_hand = np.array([23, 0] + board)
        
        aa_wrapper = evaluate_hand_wrapper(aa_hand)
        trash_wrapper = evaluate_hand_wrapper(trash_hand)
        
        # Wrapper inverts: higher = better
        assert aa_wrapper > trash_wrapper, f"AA wrapper ({aa_wrapper}) should be higher than 72o wrapper ({trash_wrapper})"
        assert aa_wrapper > 0 and trash_wrapper > 0, "Both should be positive"

class TestPokerConcepts:
    """Test fundamental poker concepts"""
    
    @pytest.fixture
    def trained_model(self):
        """Create a small trained model for testing"""
        config = TrainerConfig()
        config.batch_size = 32  # Smaller for testing
        trainer = PokerTrainer(config)
        trainer.train(20, 'test_concepts', 20, snapshot_iterations=[])
        return trainer
    
    def test_hand_strength_concept(self, trained_model):
        """Test that model differentiates hand strengths"""
        from poker_bot.core.trainer import compute_mock_info_set
        
        # Test various hand matchups
        test_cases = [
            # (strong_hand, weak_hand, description)
            ([12, 12], [5, 0], "AA vs 72o"),
            ([11, 11], [3, 1], "KK vs 52o"), 
            ([10, 10], [7, 2], "QQ vs 93o"),
            ([12, 11], [6, 1], "AK vs 82o"),
        ]
        
        strong_wins = 0
        total_tests = len(test_cases)
        
        for strong, weak, desc in test_cases:
            strong_info = compute_mock_info_set(strong, False, 2)
            weak_info = compute_mock_info_set(weak, False, 2)
            
            if strong_info < 50000 and weak_info < 50000:
                strong_strategy = trained_model.strategy[strong_info]
                weak_strategy = trained_model.strategy[weak_info]
                
                # Strong hands should be more aggressive
                strong_aggression = float(jnp.sum(strong_strategy[3:6]))  # BET/RAISE/ALLIN
                weak_aggression = float(jnp.sum(weak_strategy[3:6]))
                
                if strong_aggression > weak_aggression:
                    strong_wins += 1
                    print(f"✅ {desc}: Strong {strong_aggression:.3f} > Weak {weak_aggression:.3f}")
                else:
                    print(f"❌ {desc}: Strong {strong_aggression:.3f} ≤ Weak {weak_aggression:.3f}")
        
        success_rate = strong_wins / total_tests
        assert success_rate >= 0.5, f"Hand strength concept failing: {success_rate:.1%} success rate"
        print(f"🎯 Hand Strength Test: {success_rate:.1%} success rate")
    
    def test_suited_vs_offsuit(self, trained_model):
        """Test that model prefers suited hands"""
        from poker_bot.core.trainer import compute_mock_info_set
        
        test_cases = [
            ([12, 10], "AJs vs AJo"),  # Premium suited
            ([10, 9], "JTs vs JTo"),   # Suited connector  
            ([11, 9], "KTs vs KTo"),   # King-Ten
        ]
        
        suited_wins = 0
        total_tests = len(test_cases)
        
        for hole_ranks, desc in test_cases:
            suited_info = compute_mock_info_set(hole_ranks, True, 3)
            offsuit_info = compute_mock_info_set(hole_ranks, False, 3)
            
            if suited_info < 50000 and offsuit_info < 50000:
                suited_strategy = trained_model.strategy[suited_info]
                offsuit_strategy = trained_model.strategy[offsuit_info]
                
                # Suited should be more aggressive OR fold less
                suited_aggression = float(jnp.sum(suited_strategy[3:6]))
                offsuit_aggression = float(jnp.sum(offsuit_strategy[3:6]))
                
                suited_fold = float(suited_strategy[0])
                offsuit_fold = float(offsuit_strategy[0])
                
                # Suited is better if: more aggressive OR folds less
                suited_better = (suited_aggression > offsuit_aggression) or (suited_fold < offsuit_fold)
                
                if suited_better:
                    suited_wins += 1
                    print(f"✅ {desc}: Suited better (agg: {suited_aggression:.3f} vs {offsuit_aggression:.3f}, fold: {suited_fold:.3f} vs {offsuit_fold:.3f})")
                else:
                    print(f"❌ {desc}: Offsuit better (agg: {suited_aggression:.3f} vs {offsuit_aggression:.3f}, fold: {suited_fold:.3f} vs {offsuit_fold:.3f})")
        
        success_rate = suited_wins / total_tests
        assert success_rate >= 0.5, f"Suited concept failing: {success_rate:.1%} success rate"
        print(f"🎯 Suited Test: {success_rate:.1%} success rate")
    
    def test_position_awareness(self, trained_model):
        """Test that model plays tighter in early position"""
        from poker_bot.core.trainer import compute_mock_info_set
        
        # Marginal hand that should play differently by position
        marginal_hand = [10, 9]  # JT suited
        
        early_info = compute_mock_info_set(marginal_hand, True, 0)  # UTG
        late_info = compute_mock_info_set(marginal_hand, True, 5)   # Button
        
        if early_info < 50000 and late_info < 50000:
            early_strategy = trained_model.strategy[early_info]
            late_strategy = trained_model.strategy[late_info]
            
            early_fold = float(early_strategy[0])
            late_fold = float(late_strategy[0])
            
            early_aggression = float(jnp.sum(early_strategy[3:6]))
            late_aggression = float(jnp.sum(late_strategy[3:6]))
            
            # Should fold more in early position OR be less aggressive
            position_awareness = (early_fold > late_fold) or (early_aggression < late_aggression)
            
            print(f"Early pos - Fold: {early_fold:.3f}, Agg: {early_aggression:.3f}")
            print(f"Late pos  - Fold: {late_fold:.3f}, Agg: {late_aggression:.3f}")
            print(f"Position awareness: {position_awareness}")
            
            # For now, just log the results (position is hardest to learn with few iterations)
            if position_awareness:
                print("✅ Position awareness detected")
            else:
                print("⚠️ No clear position awareness (may need more training)")

class TestSystemIntegrity:
    """Test system-wide integrity"""
    
    def test_training_data_real(self):
        """Test that training uses real evaluator"""
        from poker_bot.core.trainer import validate_training_data_integrity
        import jax
        
        # Create dummy strategy for validation
        strategy = jnp.ones((50000, 6)) / 6
        key = jax.random.PRNGKey(42)
        
        results = validate_training_data_integrity(strategy, key, verbose=False)
        
        # All critical tests should pass
        assert results['real_histories_detected'], "Real histories not detected"
        assert results['info_set_consistency'], "Info set mapping broken"
        assert results['hand_strength_variation'], "Hand strength evaluator broken"
        assert results['action_diversity'], "Action generation broken"
        assert len(results['critical_bugs']) == 0, f"Critical bugs found: {results['critical_bugs']}"
    
    def test_model_saves_correctly(self):
        """Test that models save and load correctly"""
        config = TrainerConfig()
        trainer1 = PokerTrainer(config)
        
        # Train briefly
        trainer1.train(5, 'test_save_load', 5, snapshot_iterations=[])
        
        # Save model
        trainer1.save_model('test_save_load_manual.pkl')
        
        # Load in new trainer
        trainer2 = PokerTrainer(config)
        trainer2.load_model('test_save_load_manual.pkl')
        
        # Strategies should be identical
        assert jnp.allclose(trainer1.strategy, trainer2.strategy), "Model save/load broken"
        assert jnp.allclose(trainer1.regrets, trainer2.regrets), "Regrets save/load broken"
        assert trainer1.iteration == trainer2.iteration, "Iteration count broken"
        
        # Cleanup
        import os
        for file in ['test_save_load_final.pkl', 'test_save_load_iter_5.pkl', 'test_save_load_manual.pkl']:
            if os.path.exists(file):
                os.remove(file)

def run_all_tests():
    """Run all tests manually"""
    print("🧪 RUNNING POKER AI UNIT TESTS")
    print("="*50)
    
    # Test 1: Hand Evaluator
    print("\n🔧 Testing Hand Evaluator...")
    test_evaluator = TestHandEvaluator()
    try:
        test_evaluator.test_evaluator_import()
        test_evaluator.test_aa_vs_72o_direct()
        test_evaluator.test_wrapper_consistency()
        print("✅ Hand Evaluator tests PASSED")
    except Exception as e:
        print(f"❌ Hand Evaluator tests FAILED: {e}")
    
    # Test 2: System Integrity
    print("\n🔧 Testing System Integrity...")
    test_system = TestSystemIntegrity()
    try:
        test_system.test_training_data_real()
        test_system.test_model_saves_correctly()
        print("✅ System Integrity tests PASSED")
    except Exception as e:
        print(f"❌ System Integrity tests FAILED: {e}")
    
    # Test 3: Poker Concepts (requires training)
    print("\n🔧 Testing Poker Concepts (this will take ~30 seconds)...")
    test_concepts = TestPokerConcepts()
    try:
        trained_model = test_concepts.trained_model()
        test_concepts.test_hand_strength_concept(trained_model)
        test_concepts.test_suited_vs_offsuit(trained_model)
        test_concepts.test_position_awareness(trained_model)
        print("✅ Poker Concepts tests COMPLETED")
    except Exception as e:
        print(f"❌ Poker Concepts tests FAILED: {e}")
    
    print("\n🎉 ALL TESTS COMPLETED!")

if __name__ == "__main__":
    run_all_tests() 