"""
Hand Evaluator Real - CUDA Integration Module

This module provides the interface to CUDA hand evaluation with proper
fallback handling when CUDA is not available.
"""

import ctypes
import os
import sys
from typing import Optional, List
import random


def load_cuda_library() -> Optional[ctypes.CDLL]:
    """
    Load CUDA hand evaluation library.
    
    Returns:
        CUDA library if available, None otherwise
    """
    try:
        # Get the directory where this module is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try different possible library names
        library_names = [
            "libhand_evaluator_real.so",
            "hand_evaluator_real.so", 
            "libhand_evaluator.so",
            "hand_evaluator.so",
            "hand_evaluator_real.dll",
            "hand_evaluator.dll"
        ]
        
        for lib_name in library_names:
            lib_path = os.path.join(current_dir, lib_name)
            if os.path.exists(lib_path):
                try:
                    lib = ctypes.CDLL(lib_path)
                    
                    # Test if required functions exist
                    if hasattr(lib, 'cuda_evaluate_hand'):
                        return lib
                except OSError:
                    continue
        
        # If no library found, return None for fallback
        return None
        
    except Exception as e:
        print(f"Warning: Could not load CUDA library: {e}")
        return None


class HandEvaluatorFallback:
    """
    Fallback hand evaluator when CUDA is not available.
    
    Provides basic poker hand evaluation without CUDA acceleration.
    """
    
    # Hand rankings (higher number = better hand)
    HAND_RANKINGS = {
        'high_card': 1,
        'pair': 2,
        'two_pair': 3,
        'three_kind': 4,
        'straight': 5,
        'flush': 6,
        'full_house': 7,
        'four_kind': 8,
        'straight_flush': 9,
        'royal_flush': 10
    }
    
    @staticmethod
    def evaluate_hand(cards: List[int]) -> int:
        """
        Evaluate a 7-card poker hand (2 hole + 5 community).
        
        Args:
            cards: List of 7 integers representing cards
            
        Returns:
            Hand strength score (higher = better)
        """
        if len(cards) != 7:
            raise ValueError("Must provide exactly 7 cards")
        
        # Convert card integers to rank/suit
        card_data = []
        for card_int in cards:
            rank = card_int // 4
            suit = card_int % 4
            card_data.append((rank, suit))
        
        # Find best 5-card hand from 7 cards
        best_score = 0
        
        # Check all combinations of 5 cards from 7
        from itertools import combinations
        
        for five_cards in combinations(card_data, 5):
            score = HandEvaluatorFallback._evaluate_five_cards(five_cards)
            best_score = max(best_score, score)
        
        return best_score
    
    @staticmethod
    def _evaluate_five_cards(cards: List[tuple]) -> int:
        """Evaluate a 5-card poker hand."""
        ranks = [card[0] for card in cards]
        suits = [card[1] for card in cards]
        
        # Count ranks
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        # Sort ranks by count, then by rank value
        sorted_counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        counts = [count for rank, count in sorted_counts]
        sorted_ranks = [rank for rank, count in sorted_counts]
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        sorted_ranks_unique = sorted(set(ranks))
        is_straight = False
        
        if len(sorted_ranks_unique) == 5:
            # Normal straight
            if sorted_ranks_unique[-1] - sorted_ranks_unique[0] == 4:
                is_straight = True
            # Ace-low straight (A,2,3,4,5)
            elif sorted_ranks_unique == [2, 3, 4, 5, 14]:
                is_straight = True
        
        # Determine hand type and calculate score
        base_score = 0
        
        if is_straight and is_flush:
            if sorted_ranks_unique == [10, 11, 12, 13, 14]:
                # Royal flush
                base_score = HandEvaluatorFallback.HAND_RANKINGS['royal_flush'] * 1000000
            else:
                # Straight flush
                base_score = HandEvaluatorFallback.HAND_RANKINGS['straight_flush'] * 1000000
                base_score += max(sorted_ranks_unique) * 1000
        elif counts[0] == 4:
            # Four of a kind
            base_score = HandEvaluatorFallback.HAND_RANKINGS['four_kind'] * 1000000
            base_score += sorted_ranks[0] * 1000  # Four of a kind rank
            base_score += sorted_ranks[1]  # Kicker
        elif counts[0] == 3 and counts[1] == 2:
            # Full house
            base_score = HandEvaluatorFallback.HAND_RANKINGS['full_house'] * 1000000
            base_score += sorted_ranks[0] * 1000  # Three of a kind rank
            base_score += sorted_ranks[1]  # Pair rank
        elif is_flush:
            # Flush
            base_score = HandEvaluatorFallback.HAND_RANKINGS['flush'] * 1000000
            # Add high cards in descending order
            for i, rank in enumerate(sorted(ranks, reverse=True)):
                base_score += rank * (100 ** (4-i))
        elif is_straight:
            # Straight
            base_score = HandEvaluatorFallback.HAND_RANKINGS['straight'] * 1000000
            base_score += max(sorted_ranks_unique) * 1000
        elif counts[0] == 3:
            # Three of a kind
            base_score = HandEvaluatorFallback.HAND_RANKINGS['three_kind'] * 1000000
            base_score += sorted_ranks[0] * 1000  # Three of a kind rank
            # Add kickers
            kickers = sorted([r for r in ranks if r != sorted_ranks[0]], reverse=True)
            base_score += kickers[0] * 10 + kickers[1]
        elif counts[0] == 2 and counts[1] == 2:
            # Two pair
            base_score = HandEvaluatorFallback.HAND_RANKINGS['two_pair'] * 1000000
            pairs = sorted([sorted_ranks[0], sorted_ranks[1]], reverse=True)
            base_score += pairs[0] * 10000 + pairs[1] * 100
            # Add kicker
            kicker = [r for r in ranks if r not in pairs][0]
            base_score += kicker
        elif counts[0] == 2:
            # One pair
            base_score = HandEvaluatorFallback.HAND_RANKINGS['pair'] * 1000000
            base_score += sorted_ranks[0] * 10000  # Pair rank
            # Add kickers
            kickers = sorted([r for r in ranks if r != sorted_ranks[0]], reverse=True)
            for i, kicker in enumerate(kickers):
                base_score += kicker * (100 ** (2-i))
        else:
            # High card
            base_score = HandEvaluatorFallback.HAND_RANKINGS['high_card'] * 1000000
            # Add high cards in descending order
            for i, rank in enumerate(sorted(ranks, reverse=True)):
                base_score += rank * (100 ** (4-i))
        
        return base_score


# Global CUDA library instance
_cuda_lib = None
_fallback_evaluator = HandEvaluatorFallback()


def get_hand_evaluator():
    """Get the hand evaluator (CUDA or fallback)."""
    global _cuda_lib
    
    if _cuda_lib is None:
        _cuda_lib = load_cuda_library()
    
    return _cuda_lib


def evaluate_hand(cards: List[int]) -> int:
    """
    Evaluate a poker hand using CUDA if available, fallback otherwise.
    
    Args:
        cards: List of 7 integers representing cards
        
    Returns:
        Hand strength score (higher = better)
    """
    cuda_lib = get_hand_evaluator()
    
    if cuda_lib is not None:
        try:
            # Use CUDA evaluation
            cards_array = (ctypes.c_int * len(cards))(*cards)
            return cuda_lib.cuda_evaluate_hand(cards_array)
        except Exception:
            # Fall back to Python if CUDA fails
            pass
    
    # Use fallback evaluator
    return _fallback_evaluator.evaluate_hand(cards)


def test_hand_evaluator():
    """Test the hand evaluator with known hands."""
    print("Testing hand evaluator...")
    
    # Test hands (rank * 4 + suit format)
    test_hands = [
        # Royal flush in spades (10s, Js, Qs, Ks, As + two low cards)
        ([10*4+0, 11*4+0, 12*4+0, 13*4+0, 14*4+0, 2*4+1, 3*4+2], "Royal Flush"),
        
        # Four of a kind Aces
        ([14*4+0, 14*4+1, 14*4+2, 14*4+3, 2*4+0, 3*4+1, 4*4+2], "Four Aces"),
        
        # Pair of Aces
        ([14*4+0, 14*4+1, 2*4+2, 3*4+3, 4*4+0, 5*4+1, 6*4+2], "Pair of Aces"),
        
        # High card (worst hand)
        ([2*4+0, 4*4+1, 6*4+2, 8*4+3, 10*4+0, 12*4+1, 13*4+2], "High Card"),
    ]
    
    scores = []
    for hand, description in test_hands:
        score = evaluate_hand(hand)
        scores.append(score)
        print(f"  {description}: {score}")
    
    # Verify ordering (higher scores should be better hands)
    if scores[0] > scores[1] > scores[2] > scores[3]:
        print("✓ Hand evaluator working correctly")
        return True
    else:
        print("✗ Hand evaluator not working correctly")
        print(f"  Scores: {scores}")
        return False


if __name__ == "__main__":
    test_hand_evaluator() 