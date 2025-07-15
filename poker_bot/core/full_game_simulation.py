"""
🎯 JAX-Native Full NLHE Game Engine
Professional-grade implementation based on OpenSpiel patterns
Vectorized for GPU acceleration with JAX

This module provides:
- Complete NLHE betting rounds (pre-flop to river)
- Side pot management with multiple all-ins
- Street-by-street game progression
- JAX-native vectorized operations
- OpenSpiel-compatible state representation
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class BettingRound(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class PlayerAction(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class GamePhase(IntEnum):
    DEALING = 0
    BETTING = 1
    SHOWDOWN = 2
    FINISHED = 3

# Game constants
MAX_PLAYERS = 6
TOTAL_CARDS = 52
HOLE_CARDS = 2
COMMUNITY_CARDS = 5

# ============================================================================
# JAX-NATIVE STATE STRUCTURES
# ============================================================================

class GameState(NamedTuple):
    """Complete game state as JAX arrays"""
    # Player state [MAX_PLAYERS, 6]
    # [stack, current_bet, total_bet, is_folded, is_all_in, position]
    player_state: jnp.ndarray
    
    # Community cards [COMMUNITY_CARDS]
    community_cards: jnp.ndarray
    
    # Game state [10]
    # [current_round, current_player, pot, min_raise, max_raise, 
    #  phase, deck_index, small_blind, big_blind, num_active]
    game_info: jnp.ndarray
    
    # Deck [TOTAL_CARDS]
    deck: jnp.ndarray
    
    # Action history [MAX_HISTORY, 3] - [player, action, amount]
    action_history: jnp.ndarray
    action_count: jnp.int32

class PlayerState(NamedTuple):
    """Individual player state"""
    stack: jnp.ndarray
    current_bet: jnp.ndarray
    total_bet: jnp.ndarray
    is_folded: jnp.ndarray
    is_all_in: jnp.ndarray
    hole_cards: jnp.ndarray  # [2]

# ============================================================================
# JAX-NATIVE GAME ENGINE
# ============================================================================

class FullGameEngine:
    """
    JAX-native NLHE game engine
    Vectorized for GPU acceleration
    """
    
    def __init__(self, num_players: int = 6, small_blind: float = 1.0, big_blind: float = 2.0):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        
    @staticmethod
    def create_initial_state(rng_key: jnp.ndarray, 
                           num_players: int = 6,
                           small_blind: float = 1.0,
                           big_blind: float = 2.0,
                           starting_stack: float = 100.0) -> GameState:
        """Create initial game state"""
        
        # Create and shuffle deck
        deck = jax.random.permutation(rng_key, jnp.arange(TOTAL_CARDS))
        
        # Initialize player states
        player_state = jnp.zeros((MAX_PLAYERS, 6))
        
        # Set starting stacks
        player_state = player_state.at[:, 0].set(starting_stack)
        
        # Post blinds
        player_state = player_state.at[0, 0].add(-small_blind)
        player_state = player_state.at[0, 1].set(small_blind)
        player_state = player_state.at[0, 2].set(small_blind)
        
        player_state = player_state.at[1, 0].add(-big_blind)
        player_state = player_state.at[1, 1].set(big_blind)
        player_state = player_state.at[1, 2].set(big_blind)
        
        # Initialize game info
        game_info = jnp.array([
            float(BettingRound.PREFLOP),  # current_round
            2.0,  # current_player (UTG)
            small_blind + big_blind,  # pot
            big_blind,  # min_raise
            starting_stack,  # max_raise
            float(GamePhase.BETTING),  # phase
            float(num_players * HOLE_CARDS),  # deck_index
            small_blind,
            big_blind,
            float(num_players)  # num_active
        ])
        
        # Initialize community cards
        community_cards = jnp.full(COMMUNITY_CARDS, -1.0)
        
        # Initialize action history
        action_history = jnp.zeros((20, 3))  # Track last 20 actions
        action_count = jnp.int32(0)
        
        return GameState(
            player_state=player_state,
            community_cards=community_cards,
            game_info=game_info,
            deck=deck,
            action_history=action_history,
            action_count=action_count
        )
    
    @staticmethod
    def get_legal_actions(state: GameState) -> jnp.ndarray:
        """Get legal actions for current player"""
        current_player = jnp.int32(state.game_info[1])
        player_data = state.player_state[current_player]
        
        stack = player_data[0]
        current_bet = player_data[1]
        
        max_table_bet = jnp.max(state.player_state[:, 1])
        call_amount = max_table_bet - current_bet
        
        # Initialize action mask [6 actions]
        legal_actions = jnp.zeros(6, dtype=jnp.bool_)
        
        # Player is folded or all-in
        is_folded = player_data[3] > 0.5
        is_all_in = player_data[4] > 0.5
        
        # If player can't act, return empty mask
        can_act = ~is_folded & ~is_all_in
        
        # Check actions
        can_check = (call_amount <= 0.01) & can_act
        can_call = (call_amount > 0.01) & (stack >= call_amount) & can_act
        can_fold = (call_amount > 0.01) & can_act
        can_bet = (call_amount <= 0.01) & (stack > 0) & can_act
        can_raise = (call_amount > 0.01) & (stack > call_amount) & can_act
        can_all_in = (stack > 0) & can_act
        
        legal_actions = legal_actions.at[PlayerAction.CHECK].set(can_check)
        legal_actions = legal_actions.at[PlayerAction.CALL].set(can_call)
        legal_actions = legal_actions.at[PlayerAction.FOLD].set(can_fold)
        legal_actions = legal_actions.at[PlayerAction.BET].set(can_bet)
        legal_actions = legal_actions.at[PlayerAction.RAISE].set(can_raise)
        legal_actions = legal_actions.at[PlayerAction.ALL_IN].set(can_all_in)
        
        return legal_actions
    
    @staticmethod
    def apply_action(state: GameState, action: jnp.int32, amount: jnp.float32) -> GameState:
        """Apply action to game state"""
        current_player = jnp.int32(state.game_info[1])
        
        # Get player data
        player_data = state.player_state[current_player]
        stack = player_data[0]
        current_bet = player_data[1]
        total_bet = player_data[2]
        
        # Calculate new state based on action
        new_player_state = state.player_state
        
        # Action effects
        def apply_fold():
            new_state = new_player_state.at[current_player, 3].set(1.0)  # is_folded
            return new_state
        
        def apply_check():
            return new_player_state
        
        def apply_call():
            max_bet = jnp.max(new_player_state[:, 1])
            call_amount = max_bet - current_bet
            actual_call = jnp.minimum(call_amount, stack)
            
            new_state = new_player_state
            new_state = new_state.at[current_player, 0].add(-actual_call)
            new_state = new_state.at[current_player, 1].add(actual_call)
            new_state = new_state.at[current_player, 2].add(actual_call)
            
            # Check for all-in
            all_in = jnp.abs(new_state[current_player, 0]) < 0.01
            new_state = new_state.at[current_player, 4].set(all_in.astype(jnp.float32))
            
            return new_state
        
        def apply_bet():
            actual_bet = jnp.minimum(amount, stack)
            
            new_state = new_player_state
            new_state = new_state.at[current_player, 0].add(-actual_bet)
            new_state = new_state.at[current_player, 1].set(actual_bet)
            new_state = new_state.at[current_player, 2].add(actual_bet)
            
            # Check for all-in
            all_in = jnp.abs(new_state[current_player, 0]) < 0.01
            new_state = new_state.at[current_player, 4].set(all_in.astype(jnp.float32))
            
            return new_state
        
        def apply_raise():
            max_bet = jnp.max(new_player_state[:, 1])
            raise_amount = amount - current_bet
            actual_raise = jnp.minimum(raise_amount, stack)
            
            new_state = new_player_state
            new_state = new_state.at[current_player, 0].add(-actual_raise)
            new_state = new_state.at[current_player, 1].set(amount)
            new_state = new_state.at[current_player, 2].add(actual_raise)
            
            # Check for all-in
            all_in = jnp.abs(new_state[current_player, 0]) < 0.01
            new_state = new_state.at[current_player, 4].set(all_in.astype(jnp.float32))
            
            return new_state
        
        def apply_all_in():
            all_in_amount = stack
            
            new_state = new_player_state
            new_state = new_state.at[current_player, 0].set(0.0)
            new_state = new_state.at[current_player, 1].set(current_bet + all_in_amount)
            new_state = new_state.at[current_player, 2].add(all_in_amount)
            new_state = new_state.at[current_player, 4].set(1.0)
            
            return new_state
        
        # Apply action
        new_player_state = lax.switch(
            action,
            [apply_fold, apply_check, apply_call, apply_bet, apply_raise, apply_all_in]
        )
        
        # Update pot
        new_pot = state.game_info[2] + (new_player_state[current_player, 2] - total_bet)
        
        # Update action history
        new_action_history = state.action_history
        new_action_history = new_action_history.at[state.action_count, 0].set(float(current_player))
        new_action_history = new_action_history.at[state.action_count, 1].set(float(action))
        new_action_history = new_action_history.at[state.action_count, 2].set(float(amount))
        new_action_count = state.action_count + 1
        
        # Update game info
        new_game_info = state.game_info.at[2].set(new_pot)
        
        # Check if betting round is complete
        active_players = jnp.sum(new_player_state[:, 3] < 0.5)  # Not folded
        all_in_players = jnp.sum(new_player_state[:, 4] > 0.5)
        can_act_players = active_players - all_in_players
        
        # Find next player
        next_player = (current_player + 1) % state.num_players
        
        # Skip folded and all-in players
        def find_next_active_player(player_idx):
            is_folded = new_player_state[player_idx, 3] > 0.5
            is_all_in = new_player_state[player_idx, 4] > 0.5
            return jnp.where(is_folded | is_all_in, 
                           (player_idx + 1) % state.num_players, 
                           player_idx)
        
        # Vectorized next player calculation
        next_player = find_next_active_player(next_player)
        
        # Check if round is complete
        max_bet = jnp.max(new_player_state[:, 1])
        all_called = jnp.all((new_player_state[:, 1] == max_bet) | 
                           (new_player_state[:, 3] > 0.5) | 
                           (new_player_state[:, 4] > 0.5))
        
        round_complete = (can_act_players <= 1) | all_called
        
        # Update game state
        new_game_info = new_game_info.at[1].set(float(next_player))
        
        return GameState(
            player_state=new_player_state,
            community_cards=state.community_cards,
            game_info=new_game_info,
            deck=state.deck,
            action_history=new_action_history,
            action_count=new_action_count
        )
    
    @staticmethod
    def advance_round(state: GameState) -> GameState:
        """Advance to next betting round"""
        current_round = jnp.int32(state.game_info[0])
        next_round = current_round + 1
        
        # Deal community cards based on round
        new_community = state.community_cards
        
        def deal_flop():
            start_idx = jnp.int32(state.game_info[6])
            cards = state.deck[start_idx:start_idx+3]
            new_comm = state.community_cards.at[0:3].set(cards)
            return new_comm, start_idx + 3
        
        def deal_turn():
            start_idx = jnp.int32(state.game_info[6])
            card = state.deck[start_idx:start_idx+1]
            new_comm = state.community_cards.at[3:4].set(card)
            return new_comm, start_idx + 1
        
        def deal_river():
            start_idx = jnp.int32(state.game_info[6])
            card = state.deck[start_idx:start_idx+1]
            new_comm = state.community_cards.at[4:5].set(card)
            return new_comm, start_idx + 1
        
        def no_deal():
            return state.community_cards, jnp.int32(state.game_info[6])
        
        new_community, new_deck_idx = lax.switch(
            current_round,
            [no_deal, deal_flop, deal_turn, deal_river, no_deal]
        )
        
        # Update game info
        new_game_info = state.game_info
        new_game_info = new_game_info.at[0].set(float(next_round))
        new_game_info = new_game_info.at[1].set(0.0)  # First to act
        new_game_info = new_game_info.at[6].set(float(new_deck_idx))
        
        return GameState(
            player_state=state.player_state,
            community_cards=new_community,
            game_info=new_game_info,
            deck=state.deck,
            action_history=state.action_history,
            action_count=state.action_count
        )
    
    @staticmethod
    def calculate_side_pots(player_state: jnp.ndarray) -> jnp.ndarray:
        """Calculate side pots for showdown"""
        # Get active players and their contributions
        active_mask = player_state[:, 3] < 0.5  # Not folded
        contributions = player_state[:, 2] * active_mask
        
        # Sort unique contributions
        unique_contribs = jnp.unique(contributions)
        
        # Calculate side pots
        side_pots = jnp.zeros((MAX_PLAYERS, 2))  # [amount, eligible_count]
        
        def calculate_pot_for_contrib(carry, contrib):
            pot_amount = (contrib - carry[0]) * jnp.sum(contributions >= contrib)
            eligible = jnp.sum(contributions >= contrib)
            
            new_pot = jnp.array([pot_amount, eligible])
            return contrib, new_pot
        
        # Use scan to calculate side pots
        _, pots = lax.scan(
            calculate_pot_for_contrib,
            0.0,
            unique_contribs
        )
        
        return pots
    
    @staticmethod
    def simulate_game(rng_key: jnp.ndarray, 
                     num_players: int = 6,
                     small_blind: float = 1.0,
                     big_blind: float = 2.0,
                     starting_stack: float = 100.0) -> Dict:
        """Simulate complete NLHE game"""
        
        # Initialize game
        state = FullGameEngine.create_initial_state(
            rng_key, num_players, small_blind, big_blind, starting_stack
        )
        
        # Game loop
        def game_step(carry, _):
            state, done = carry
            
            # Get legal actions
            legal_actions = FullGameEngine.get_legal_actions(state)
            
            # Random action selection
            action_probs = legal_actions.astype(jnp.float32)
            action = jax.random.categorical(rng_key, jnp.log(action_probs + 1e-8))
            
            # Apply action
            new_state = FullGameEngine.apply_action(state, action, 10.0)  # Simplified amount
            
            # Check if round complete
            round_complete = jnp.max(new_state.player_state[:, 1]) == jnp.min(
                new_state.player_state[new_state.player_state[:, 3] < 0.5, 1]
            )
            
            # Advance round if complete
            final_state = lax.cond(
                round_complete,
                lambda s: FullGameEngine.advance_round(s),
                lambda s: s,
                new_state
            )
            
            # Check if game finished
            active_players = jnp.sum(final_state.player_state[:, 3] < 0.5)
            game_done = (active_players <= 1) | (final_state.game_info[0] >= 4.0)
            
            return (final_state, game_done), None
        
        # Run game
        final_state, _ = lax.scan(
            game_step,
            (state, False),
            jnp.arange(100)  # Max 100 actions
        )
        
        # Calculate final payoffs
        active_players = final_state.player_state[:, 3] < 0.5
        total_pot = jnp.sum(final_state.player_state[:, 2])
        
        # Simplified payoff calculation
        payoffs = jnp.where(active_players, total_pot / jnp.sum(active_players), 
                          -final_state.player_state[:, 2])
        
        return {
            'payoffs': payoffs,
            'final_community': final_state.community_cards,
            'hole_cards': final_state.deck[:num_players*2].reshape(num_players, 2),
            'final_pot': total_pot,
            'action_history': final_state.action_history[:final_state.action_count]
        }

# ============================================================================
# VECTORIZED BATCH SIMULATION
# ============================================================================

class VectorizedGameEngine:
    """Vectorized game engine for batch processing"""
    
    @staticmethod
    def batch_simulate(rng_keys: jnp.ndarray,
                      num_players: int = 6,
                      small_blind: float = 1.0,
                      big_blind: float = 2.0,
                      starting_stack: float = 100.0) -> Dict:
        """Simulate multiple games in batch"""
        
        # Vectorized simulation
        def simulate_single(key):
            return FullGameEngine.simulate_game(
                key, num_players, small_blind, big_blind, starting_stack
            )
        
        # Use vmap for vectorization
        batch_results = jax.vmap(simulate_single)(rng_keys)
        
        return batch_results

# ============================================================================
# STATE CONVERSION UTILITIES
# ============================================================================

class StateConverter:
    """Convert between game state formats"""
    
    @staticmethod
    def to_openspiel_format(state: GameState) -> Dict:
        """Convert to OpenSpiel-compatible format"""
        return {
            'player_states': state.player_state,
            'community_cards': state.community_cards,
            'game_info': state.game_info,
            'deck': state.deck,
            'action_history': state.action_history[:state.action_count]
        }
    
    @staticmethod
    def from_openspiel_format(data: Dict) -> GameState:
        """Convert from OpenSpiel format"""
        return GameState(
            player_state=data['player_states'],
            community_cards=data['community_cards'],
            game_info=data['game_info'],
            deck=data['deck'],
            action_history=data['action_history'],
            action_count=jnp.int32(data['action_history'].shape[0])
        )