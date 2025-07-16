#!/usr/bin/env python3
"""
🔍 Debug Evaluator Detailed - Find why AA and 72o both return 0
"""

def test_evaluator_detailed():
    """Debug what's happening in the validation"""
    print("🔍 DEBUGGING EVALUATOR DETAILED...")
    
    try:
        import jax.numpy as jnp
        import numpy as np
        from poker_bot.core.trainer import evaluate_hand_jax
        from poker_bot.core.full_game_engine import evaluate_hand_wrapper
        from poker_bot.evaluator import HandEvaluator
        
        print("✅ All imports successful")
        
        # Test what the validation is actually testing
        print("\n🧪 Testing validation scenario:")
        
        # AA cards: As=51, Ac=47 (same as test_real_evaluator.py but only 2 cards)
        aa_cards_jax = jnp.array([51, 47], dtype=jnp.int8)
        trash_cards_jax = jnp.array([23, 0], dtype=jnp.int8)  # 7c=23, 2s=0
        
        print(f"  AA cards: {aa_cards_jax}")
        print(f"  72o cards: {trash_cards_jax}")
        
        # Test direct wrapper
        print("\n🔧 Testing evaluate_hand_wrapper directly:")
        aa_wrapper = evaluate_hand_wrapper(np.array([51, 47]))
        trash_wrapper = evaluate_hand_wrapper(np.array([23, 0]))
        
        print(f"  AA wrapper (2 cards): {aa_wrapper}")
        print(f"  72o wrapper (2 cards): {trash_wrapper}")
        
        # Test with 7 cards (5 community + 2 hole)
        print("\n🔧 Testing with 7 cards (hole + community):")
        aa_7cards = np.array([51, 47, 46, 42, 37, 35, 32])  # AA + board
        trash_7cards = np.array([23, 0, 46, 42, 37, 35, 32])  # 72o + board
        
        aa_7_wrapper = evaluate_hand_wrapper(aa_7cards)
        trash_7_wrapper = evaluate_hand_wrapper(trash_7cards)
        
        print(f"  AA wrapper (7 cards): {aa_7_wrapper}")
        print(f"  72o wrapper (7 cards): {trash_7_wrapper}")
        
        # Test JAX evaluator
        print("\n🎯 Testing evaluate_hand_jax:")
        try:
            # This will fail if called outside JIT, but let's see the error
            aa_jax = evaluate_hand_jax(aa_cards_jax)
            trash_jax = evaluate_hand_jax(trash_cards_jax)
            print(f"  AA JAX: {aa_jax}")
            print(f"  72o JAX: {trash_jax}")
        except Exception as e:
            print(f"  JAX evaluator error (expected): {e}")
        
        # Test real evaluator directly
        print("\n🏆 Testing HandEvaluator directly:")
        evaluator = HandEvaluator()
        
        try:
            aa_real = evaluator.evaluate_single([51, 47, 46, 42, 37, 35, 32])
            trash_real = evaluator.evaluate_single([23, 0, 46, 42, 37, 35, 32])
            print(f"  AA real evaluator: {aa_real}")
            print(f"  72o real evaluator: {trash_real}")
        except Exception as e:
            print(f"  Real evaluator error: {e}")
        
        # Diagnosis
        print(f"\n📊 DIAGNOSIS:")
        if aa_wrapper == 0 and trash_wrapper == 0:
            print(f"  ❌ PROBLEM: Both return 0 with 2 cards - wrapper needs ≥5 cards")
        elif aa_7_wrapper > trash_7_wrapper:
            print(f"  ✅ GOOD: Wrapper works with 7 cards")
        else:
            print(f"  ❌ PROBLEM: Wrapper broken even with 7 cards")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_evaluator_detailed() 