"""
🧠 History-Aware Bucketing System
Phase 2 Enhancement: 200k buckets with action history
"""

import cupy as cp
import numpy as np
from typing import Tuple, List
import hashlib

class HistoryAwareBucketing:
    """
    Advanced bucketing with action history and position dynamics
    200k buckets capturing range polarization and balance
    """
    
    def __init__(self):
        self.action_encoding = {
            'FOLD': 0, 'CHECK': 1, 'CALL': 2, 'BET_33': 3, 'BET_50': 4,
            'BET_75': 5, 'BET_100': 6, 'BET_150': 7, 'BET_200': 8, 'RAISE': 9
        }
        
    def encode_action_history(self, history: List[str]) -> int:
        """Encode last 2 actions into compact integer"""
        encoded = 0
        for i, action in enumerate(history[-2:]):
            if action in self.action_encoding:
                encoded += self.action_encoding[action] * (10 ** i)
        return encoded
    
    def create_history_buckets(self, 
                             hole_cards: cp.ndarray,
                             community_cards: cp.ndarray,
                             positions: cp.ndarray,
                             stack_sizes: cp.ndarray,
                             pot_sizes: cp.ndarray,
                             num_actives: cp.ndarray,
                             action_history: List[str] = None) -> cp.ndarray:
        """
        Create 200k history-aware buckets
        Args:
            hole_cards: (batch, 2) hole cards
            community_cards: (batch, 5) community cards
            positions: (batch,) player positions
            stack_sizes: (batch,) stack sizes
            pot_sizes: (batch,) pot sizes
            num_actives: (batch,) active players
            action_history: List of last actions
        Returns:
            bucket_ids: (batch,) enhanced bucket IDs (0-199999)
        """
        batch_size = hole_cards.shape[0]
        
        # Base bucketing from Phase 1
        from .icm_modeling import ICMAwareBucketing
        base_bucketing = ICMAwareBucketing()
        base_buckets = base_bucketing.create_icm_buckets(
            hole_cards, stack_sizes, positions, pot_sizes, num_actives
        )
        
        # History encoding
        history_encoded = 0
        if action_history:
            history_encoded = self.encode_action_history(action_history)
        
        # Dynamic stack depth classes (10 classes)
        stack_depth = cp.clip(stack_sizes / 20.0, 0, 9).astype(cp.uint32)
        
        # Position dynamics (early/middle/late)
        position_class = cp.where(positions < 2, 0,  # Early
                         cp.where(positions < 4, 1,  # Middle
                                  2))  # Late
        
        # Pot commitment factor
        commitment = cp.clip(pot_sizes / stack_sizes, 0, 5).astype(cp.uint32)
        
        # Street classification
        num_comm = cp.sum(community_cards != -1, axis=1)
        street = cp.where(num_comm == 0, 0,  # Preflop
                 cp.where(num_comm == 3, 1,  # Flop
                 cp.where(num_comm == 4, 2,  # Turn
                          3)))  # River
        
        # Combine all factors into 200k buckets
        bucket_id = (
            base_buckets * 100 +           # Base bucket (0-49999)
            history_encoded * 1000 +       # Action history (0-999)
            stack_depth * 100 +            # Stack depth (0-9)
            position_class * 10 +          # Position class (0-2)
            commitment * 2 +               # Commitment (0-5)
            street                         # Street (0-3)
        )
        
        return bucket_id % 200000

class AdvancedHandEvaluator:
    """
    Phase 2 enhanced evaluation with history context
    """
    
    def __init__(self):
        self.range_weights = self._init_range_weights()
        
    def _init_range_weights(self) -> dict:
        """Initialize range weights based on position and history"""
        return {
            'early': {'premium': 0.8, 'broadway': 0.6, 'suited': 0.4, 'other': 0.2},
            'middle': {'premium': 0.7, 'broadway': 0.7, 'suited': 0.5, 'other': 0.3},
            'late': {'premium': 0.6, 'broadway': 0.8, 'suited': 0.7, 'other': 0.5}
        }
    
    def evaluate_with_history(self, hole_cards: cp.ndarray,
                            position: int,
                            action_history: List[str],
                            stack_depth: float) -> cp.ndarray:
        """
        Evaluate hand strength with full context
        """
        batch_size = hole_cards.shape[0]
        
        # Base evaluation
        from .enhanced_eval import EnhancedHandEvaluator
        base_eval = EnhancedHandEvaluator()
        base_strength = base_eval.enhanced_hand_strength(
            hole_cards, cp.full((batch_size, 5), -1)
        )
        
        # Position adjustment
        position_factor = 1.0
        if position < 2:  # Early position
            position_factor = 0.9
        elif position > 3:  # Late position
            position_factor = 1.1
            
        # Stack depth adjustment
        if stack_depth < 15:  # Short stack
            stack_factor = 1.2
        elif stack_depth > 100:  # Deep stack
            stack_factor = 0.9
        else:
            stack_factor = 1.0
            
        # Action history adjustment
        history_factor = 1.0
        if action_history:
            aggressive_actions = sum(1 for a in action_history[-2:] if 'BET' in a or 'RAISE' in a)
            history_factor = 1.0 + (aggressive_actions * 0.1)
        
        # Final adjustment
        adjusted_strength = base_strength * position_factor * stack_factor * history_factor
        
        return cp.clip(adjusted_strength, 0, 1000).astype(cp.uint32)

# Phase 2 configuration
PHASE2_BUCKET_COUNT = 200000
PHASE2_CONFIG = {
    'num_buckets': PHASE2_BUCKET_COUNT,
    'history_depth': 2,
    'stack_classes': 10,
    'position_classes': 3,
    'commitment_levels': 6,
    'street_classes': 4
}