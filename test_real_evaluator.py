#!/usr/bin/env python3
"""
🧪 Test Real Hand Evaluator - Verify AA vs 72o and other known hands
"""

def test_real_evaluator():
    """Test that real evaluator gives correct results for known hands"""
    print("🧪 Testing Real Hand Evaluator...")
    
    try:
        from poker_bot.evaluator import HandEvaluator
        evaluator = HandEvaluator()
        
        print("✅ HandEvaluator imported successfully")
        
        # Test known hands
        print("\n🃏 Testing known hand comparisons:")
        
        # AA vs 72o (5 card board: Kh Qd Js 9c 8h)
        aa_hand = [51, 47, 46, 42, 37, 35, 32]  # As Ac Kh Qd Js 9c 8h  
        trash_hand = [23, 0, 46, 42, 37, 35, 32]  # 7c 2s Kh Qd Js 9c 8h
        
        aa_strength = evaluator.evaluate_single(aa_hand)
        trash_strength = evaluator.evaluate_single(trash_hand)
        
        print(f"  AA strength: {aa_strength}")
        print(f"  72o strength: {trash_strength}")
        print(f"  AA better? {aa_strength < trash_strength} (lower = better in phevaluator)")
        
        # Royal Flush vs High Card
        royal_flush = [48, 44, 40, 36, 32]  # As Ks Qs Js Ts (all spades)
        high_card = [50, 46, 42, 38, 34]    # Ad Kh Qd Jh Td (no flush, no straight)
        
        royal_strength = evaluator.evaluate_single(royal_flush)
        high_strength = evaluator.evaluate_single(high_card)
        
        print(f"\n  Royal Flush: {royal_strength}")
        print(f"  High Card: {high_strength}")
        print(f"  Royal better? {royal_strength < high_strength}")
        
        # Test the wrapper function
        print("\n🔧 Testing evaluate_hand_wrapper:")
        from poker_bot.core.full_game_engine import evaluate_hand_wrapper
        import numpy as np
        
        aa_wrapper = evaluate_hand_wrapper(np.array(aa_hand))
        trash_wrapper = evaluate_hand_wrapper(np.array(trash_hand))
        
        print(f"  AA via wrapper: {aa_wrapper}")
        print(f"  72o via wrapper: {trash_wrapper}")
        print(f"  AA better via wrapper? {aa_wrapper > trash_wrapper} (higher = better in wrapper)")
        
        if aa_wrapper > trash_wrapper:
            print("  ✅ Wrapper working correctly!")
        else:
            print("  ❌ Wrapper still broken!")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run: pip install phevaluator")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_real_evaluator()
    if success:
        print("\n🎉 Real evaluator test completed!")
    else:
        print("\n💀 Real evaluator test FAILED - Fix before training!") 