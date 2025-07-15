"""
🎯 Elite Game State Management - Complete NLHE State Reconstruction
Based on OpenSpiel patterns for professional-grade poker state handling

This module provides:
- Complete game state reconstruction from training data
- Street-by-street state tracking
- Professional hand evaluation integration
- Memory-efficient state representation
- OpenSpiel-compatible state format
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict
from enum import IntEnum
import logging
from poker_bot.core.elite_game_engine import EliteGameEngine, GameState, PlayerAction, BettingRound

logger = logging.getLogger(__name__)

# ============================================================================
# STATE REPRESENTATION CONSTANTS
# ============================================================================

class StateFeature(IntEnum):
    """Feature indices for state representation"""
    # Player features (6 players × 6 features)
    PLAYER_STACK = 0
    PLAYER_BET = 1
    PLAYER_FOLDED = 2
    PLAYER_ALL_IN = 3
    PLAYER_POSITION = 4
    PLAYER_ACTIVE = 5
    
    # Community cards (5 cards × 2 features)
    COMM_CARD_RANK = 36
    COMM_CARD_SUIT = 41
    
    # Game state features
    POT_SIZE = 46
    CURRENT_BET = 47
    MIN_RAISE = 48
    ROUND = 49
    NUM_ACTIVE = 50
    
    # Action history (last 10 actions × 3 features)
    ACTION_PLAYER = 51
    ACTION_TYPE = 61
    ACTION_AMOUNT = 71
    
    # Total state size
    STATE_SIZE = 81

# ============================================================================
# STATE DATA STRUCTURES
# ============================================================================

@dataclass
class GameStateVector:
    """Complete game state as a fixed-size vector"""
    state: jnp.ndarray  # [STATE_SIZE] float32
    
    @classmethod
    def zeros(cls) -> 'GameStateVector':
        """Create zero-initialized state vector"""
        return cls(state=jnp.zeros(StateFeature.STATE_SIZE, dtype=jnp.float32))

@dataclass
class PlayerStateFeatures:
    """Features for a single player"""
    stack: float
    current_bet: float
    is_folded: bool
    is_all_in: bool
    position: int
    is_active: bool

@dataclass
class ActionHistory:
    """Action history for state reconstruction"""
    player: int
    action: PlayerAction
    amount: float
    round: BettingRound

# ============================================================================
# STATE RECONSTRUCTOR
# ============================================================================

class GameStateReconstructor:
    """
    Professional game state reconstruction for training
    
    Features:
    - Complete state reconstruction from game data
    - OpenSpiel-compatible format
    - Memory-efficient representation
    - Street-by-street tracking
    """
    
    def __init__(self, num_players: int = 6):
        self.num_players = num_players
        self.engine = EliteGameEngine(num_players=num_players)
        
    def reconstruct_state_vector(self, 
                                hole_cards: jnp.ndarray,
                                community_cards: jnp.ndarray,
                                player_states: List[PlayerStateFeatures],
                                pot_size: float,
                                current_bet: float,
                                betting_round: BettingRound,
                                action_history: List[ActionHistory]) -> GameStateVector:
        """
        Reconstruct complete game state vector
        
        Args:
            hole_cards: [num_players, 2] hole cards for each player
            community_cards: [5] community cards (-1 for undealt)
            player_states: List of player state features
            pot_size: Current pot size
            current_bet: Current bet to call
            betting_round: Current betting round
            action_history: List of actions taken
            
        Returns:
            Complete state vector for training
        """
        state_vector = GameStateVector.zeros()
        
        # Fill player features
        for i, player_state in enumerate(player_states):
            base_idx = i * 6
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_STACK].set(player_state.stack)
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_BET].set(player_state.current_bet)
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_FOLDED].set(float(player_state.is_folded))
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_ALL_IN].set(float(player_state.is_all_in))
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_POSITION].set(float(player_state.position))
            state_vector.state = state_vector.state.at[base_idx + StateFeature.PLAYER_ACTIVE].set(float(player_state.is_active))
        
        # Fill community card features
        dealt_community = community_cards[community_cards >= 0]
        for i, card in enumerate(dealt_community):
            rank = card % 13
            suit = card // 13
            state_vector.state = state_vector.state.at[StateFeature.COMM_CARD_RANK + i].set(float(rank))
            state_vector.state = state_vector.state.at[StateFeature.COMM_CARD_SUIT + i].set(float(suit))
        
        # Fill game state features
        state_vector.state = state_vector.state.at[StateFeature.POT_SIZE].set(pot_size)
        state_vector.state = state_vector.state.at[StateFeature.CURRENT_BET].set(current_bet)
        state_vector.state = state_vector.state.at[StateFeature.MIN_RAISE].set(self.engine.big_blind)
        state_vector.state = state_vector.state.at[StateFeature.ROUND].set(float(betting_round))
        
        # Calculate active players
        num_active = sum(1 for p in player_states if p.is_active and not p.is_folded)
        state_vector.state = state_vector.state.at[StateFeature.NUM_ACTIVE].set(float(num_active))
        
        # Fill action history
        recent_actions = action_history[-10:]  # Last 10 actions
        for i, action in enumerate(recent_actions):
            base_idx = StateFeature.ACTION_PLAYER + i * 3
            state_vector.state = state_vector.state.at[base_idx].set(float(action.player))
            state_vector.state = state_vector.state.at[base_idx + 1].set(float(action.action))
            state_vector.state = state_vector.state.at[base_idx + 2].set(action.amount)
        
        return state_vector
    
    def reconstruct_from_game_data(self, game_data: Dict) -> List[GameStateVector]:
        """
        Reconstruct all game states from complete game data
        
        Args:
            game_data: Complete game data including:
                - hole_cards: [num_players, 2]
                - community_cards: [5]
                - action_history: List of (player, action, amount)
                - starting_stacks: [num_players]
                
        Returns:
            List of state vectors for each decision point
        """
        hole_cards = game_data['hole_cards']
        community_cards = game_data['community_cards']
        action_history = game_data['action_history']
        starting_stacks = game_data['starting_stacks']
        
        # Initialize game state
        player_states = []
        for i in range(self.num_players):
            player_states.append(PlayerStateFeatures(
                stack=float(starting_stacks[i]),
                current_bet=0.0,
                is_folded=False,
                is_all_in=False,
                position=i,
                is_active=True
            ))
        
        # Track game progression
        states = []
        current_round = BettingRound.PREFLOP
        pot_size = self.engine.small_blind + self.engine.big_blind
        current_bet = self.engine.big_blind
        
        # Process action history
        for player_id, action, amount in action_history:
            # Create state before action
            state_vector = self.reconstruct_state_vector(
                hole_cards=hole_cards,
                community_cards=community_cards,
                player_states=player_states,
                pot_size=pot_size,
                current_bet=current_bet,
                betting_round=current_round,
                action_history=[ActionHistory(p, a, amt, current_round) 
                               for p, a, amt in action_history[:len(states)]]
            )
            states.append(state_vector)
            
            # Update player state based on action
            player_state = player_states[player_id]
            
            if action == PlayerAction.FOLD:
                player_state.is_folded = True
                player_state.is_active = False
                
            elif action == PlayerAction.CALL:
                call_amount = min(amount, player_state.stack)
                player_state.stack -= call_amount
                player_state.current_bet += call_amount
                pot_size += call_amount
                
            elif action in [PlayerAction.BET, PlayerAction.RAISE]:
                bet_amount = min(amount, player_state.stack)
                player_state.stack -= bet_amount
                player_state.current_bet = bet_amount
                pot_size += bet_amount
                current_bet = max(current_bet, bet_amount)
                
                if player_state.stack <= 0:
                    player_state.is_all_in = True
                    player_state.is_active = False
                    
            elif action == PlayerAction.ALL_IN:
                all_in_amount = player_state.stack
                player_state.stack = 0.0
                player_state.current_bet += all_in_amount
                player_state.total_bet += all_in_amount
                player_state.is_all_in = True
                player_state.is_active = False
                pot_size += all_in_amount
                current_bet = max(current_bet, all_in_amount)
        
        return states
    
    def create_batch_states(self, batch_games: List[Dict]) -> jnp.ndarray:
        """
        Create batch of state vectors for training
        
        Args:
            batch_games: List of game data dictionaries
            
        Returns:
            [batch_size, STATE_SIZE] state tensor
        """
        all_states = []
        
        for game_data in batch_games:
            states = self.reconstruct_from_game_data(game_data)
            all_states.extend(states)
        
        # Convert to JAX tensor
        if all_states:
            state_array = jnp.stack([s.state for s in all_states])
        else:
            state_array = jnp.empty((0, StateFeature.STATE_SIZE))
        
        return state_array
    
    def get_legal_actions_mask(self, state_vector: GameStateVector) -> jnp.ndarray:
        """
        Get legal actions mask for a given state
        
        Args:
            state_vector: Game state vector
            
        Returns:
            [num_actions] boolean mask
        """
        # Extract relevant features
        current_bet = state_vector.state[StateFeature.CURRENT_BET]
        player_idx = int(state_vector.state[StateFeature.PLAYER_POSITION])
        player_stack = state_vector.state[player_idx * 6 + StateFeature.PLAYER_STACK]
        player_bet = state_vector.state[player_idx * 6 + StateFeature.PLAYER_BET]
        
        # Calculate legal actions
        call_amount = current_bet - player_bet
        legal_actions = jnp.zeros(11, dtype=jnp.bool_)  # 11 possible actions
        
        if call_amount <= 0:
            # Can check or bet
            legal_actions = legal_actions.at[PlayerAction.CHECK].set(True)
            legal_actions = legal_actions.at[PlayerAction.BET].set(True)
        else:
            # Can fold, call, or raise
            legal_actions = legal_actions.at[PlayerAction.FOLD].set(True)
            legal_actions = legal_actions.at[PlayerAction.CALL].set(True)
            
            if player_stack > call_amount:
                legal_actions = legal_actions.at[PlayerAction.RAISE].set(True)
        
        # Always allow all-in if stack > 0
        if player_stack > 0:
            legal_actions = legal_actions.at[PlayerAction.ALL_IN].set(True)
        
        return legal_actions

# ============================================================================
# STATE ENCODING AND DECODING
# ============================================================================

class StateEncoder:
    """Encode and decode game states for efficient storage"""
    
    @staticmethod
    def encode_state(state_vector: GameStateVector) -> bytes:
        """Encode state vector to bytes"""
        # Convert to numpy and compress
        state_np = np.array(state_vector.state, dtype=np.float32)
        return state_np.tobytes()
    
    @staticmethod
    def decode_state(encoded: bytes) -> GameStateVector:
        """Decode bytes back to state vector"""
        state_np = np.frombuffer(encoded, dtype=np.float32)
        return GameStateVector(state=jnp.array(state_np))
    
    @staticmethod
    def encode_batch(states: jnp.ndarray) -> bytes:
        """Encode batch of states"""
        states_np = np.array(states, dtype=np.float32)
        return states_np.tobytes()
    
    @staticmethod
    def decode_batch(encoded: bytes, batch_size: int) -> jnp.ndarray:
        """Decode batch of states"""
        states_np = np.frombuffer(encoded, dtype=np.float32)
        states_np = states_np.reshape(batch_size, StateFeature.STATE_SIZE)
        return jnp.array(states_np)

# ============================================================================
# STATE VALIDATION
# ============================================================================

class StateValidator:
    """Validate game state vectors"""
    
    @staticmethod
    def validate_state(state_vector: GameStateVector) -> bool:
        """Validate a single state vector"""
        # Check dimensions
        if state_vector.state.shape != (StateFeature.STATE_SIZE,):
            return False
        
        # Check for NaN or infinite values
        if jnp.any(jnp.isnan(state_vector.state)) or jnp.any(jnp.isinf(state_vector.state)):
            return False
        
        # Check ranges
        if jnp.any(state_vector.state < -1e6) or jnp.any(state_vector.state > 1e6):
            return False
        
        return True
    
    @staticmethod
    def validate_batch(states: jnp.ndarray) -> bool:
        """Validate batch of states"""
        if len(states.shape) != 2:
            return False
        
        if states.shape[1] != StateFeature.STATE_SIZE:
            return False
        
        return not (jnp.any(jnp.isnan(states)) or jnp.any(jnp.isinf(states)))

# ============================================================================
# STATE UTILITIES
# ============================================================================

class StateUtilities:
    """Utility functions for state manipulation"""
    
    @staticmethod
    def normalize_states(states: jnp.ndarray) -> jnp.ndarray:
        """Normalize state values for training"""
        # Normalize stack sizes (divide by 100)
        stack_indices = [i * 6 + StateFeature.PLAYER_STACK for i in range(6)]
        states = states.at[:, stack_indices].divide(100.0)
        
        # Normalize pot size
        states = states.at[:, StateFeature.POT_SIZE].divide(100.0)
        
        # Normalize bet amounts
        bet_indices = [i * 6 + StateFeature.PLAYER_BET for i in range(6)]
        states = states.at[:, bet_indices].divide(100.0)
        
        return states
    
    @staticmethod
    def denormalize_states(states: jnp.ndarray) -> jnp.ndarray:
        """Denormalize state values"""
        # Reverse normalization
        stack_indices = [i * 6 + StateFeature.PLAYER_STACK for i in range(6)]
        states = states.at[:, stack_indices].multiply(100.0)
        
        states = states.at[:, StateFeature.POT_SIZE].multiply(100.0)
        
        bet_indices = [i * 6 + StateFeature.PLAYER_BET for i in range(6)]
        states = states.at[:, bet_indices].multiply(100.0)
        
        return states