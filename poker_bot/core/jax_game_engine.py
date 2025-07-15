"""
🎯 JAX-Native NLHE Game Engine - Simplified Version
Professional-grade implementation with JAX compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple
from enum import IntEnum

class PlayerAction(IntEnum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class GameState:
    """Simple game state structure"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_initial_state(rng_key: jnp.ndarray, 
                        num_players: int = 6,
                        small_blind: float = 1.0,
                        big_blind: float = 2.0,
                        starting_stack: float = 100.0) -> Dict:
    """Create initial game state"""
    
    # Create deck
    deck = jax.random.permutation(rng_key, jnp.arange(52))
    
    # Deal hole cards
    hole_cards = deck[:num_players * 2].reshape(num_players, 2)
    
    # Initialize player states
    player_stacks = jnp.full(num_players, starting_stack)
    player_bets = jnp.zeros(num_players)
    player_folded = jnp.zeros(num_players, dtype=jnp.bool_)
    player_all_in = jnp.zeros(num_players, dtype=jnp.bool_)
    
    # Post blinds
    player_stacks = player_stacks.at[0].add(-small_blind)
    player_bets = player_bets.at[0].set(small_blind)
    
    player_stacks = player_stacks.at[1].add(-big_blind)
    player_bets = player_bets.at[1].set(big_blind)
    
    # Initialize community cards
    community_cards = jnp.full(5, -1)
    
    return {
        'player_stacks': player_stacks,
        'player_bets': player_bets,
        'player_folded': player_folded,
        'player_all_in': player_all_in,
        'hole_cards': hole_cards,
        'community_cards': community_cards,
        'deck': deck,
        'pot': small_blind + big_blind,
        'current_player': 2,
        'round': 0,  # 0=preflop, 1=flop, 2=turn, 3=river
        'min_bet': big_blind,
        'deck_index': num_players * 2
    }

def get_legal_actions(state: Dict) -> jnp.ndarray:
    """Get legal actions for current player"""
    current_player = state['current_player']
    stack = state['player_stacks'][current_player]
    current_bet = state['player_bets'][current_player]
    
    max_bet = jnp.max(state['player_bets'])
    call_amount = max_bet - current_bet
    
    # Action mask: [fold, check, call, bet, raise, all_in]
    actions = jnp.zeros(6, dtype=jnp.bool_)
    
    # Can fold if facing a bet
    actions = actions.at[PlayerAction.FOLD].set(call_amount > 0)
    
    # Can check if no bet to call
    actions = actions.at[PlayerAction.CHECK].set(call_amount == 0)
    
    # Can call if have enough chips
    actions = actions.at[PlayerAction.CALL].set(
        (call_amount > 0) & (stack >= call_amount)
    )
    
    # Can bet if no bet and have chips
    actions = actions.at[PlayerAction.BET].set(
        (call_amount == 0) & (stack > 0)
    )
    
    # Can raise if facing bet and have chips
    actions = actions.at[PlayerAction.RAISE].set(
        (call_amount > 0) & (stack > call_amount)
    )
    
    # Can always go all-in if have chips
    actions = actions.at[PlayerAction.ALL_IN].set(stack > 0)
    
    return actions

def apply_action(state: Dict, action: int, amount: float = 0.0) -> Dict:
    """Apply action to game state"""
    current_player = state['current_player']
    
    # Create new state
    new_state = {k: v.copy() if hasattr(v, 'copy') else v for k, v in state.items()}
    
    if action == PlayerAction.FOLD:
        new_state['player_folded'] = new_state['player_folded'].at[current_player].set(True)
        
    elif action == PlayerAction.CALL:
        max_bet = jnp.max(new_state['player_bets'])
        call_amount = max_bet - new_state['player_bets'][current_player]
        actual_call = jnp.minimum(call_amount, new_state['player_stacks'][current_player])
        
        new_state['player_stacks'] = new_state['player_stacks'].at[current_player].add(-actual_call)
        new_state['player_bets'] = new_state['player_bets'].at[current_player].add(actual_call)
        new_state['pot'] += actual_call
        
        # Check for all-in
        if new_state['player_stacks'][current_player] < 0.01:
            new_state['player_all_in'] = new_state['player_all_in'].at[current_player].set(True)
            
    elif action == PlayerAction.RAISE:
        raise_amount = amount - new_state['player_bets'][current_player]
        actual_raise = jnp.minimum(raise_amount, new_state['player_stacks'][current_player])
        
        new_state['player_stacks'] = new_state['player_stacks'].at[current_player].add(-actual_raise)
        new_state['player_bets'] = new_state['player_bets'].at[current_player].set(amount)
        new_state['pot'] += actual_raise
        
        # Check for all-in
        if new_state['player_stacks'][current_player] < 0.01:
            new_state['player_all_in'] = new_state['player_all_in'].at[current_player].set(True)
            
    elif action == PlayerAction.ALL_IN:
        all_in_amount = new_state['player_stacks'][current_player]
        
        new_state['player_stacks'] = new_state['player_stacks'].at[current_player].set(0.0)
        new_state['player_bets'] = new_state['player_bets'].at[current_player].add(all_in_amount)
        new_state['pot'] += all_in_amount
        new_state['player_all_in'] = new_state['player_all_in'].at[current_player].set(True)
    
    # Find next active player
    next_player = (current_player + 1) % len(new_state['player_stacks'])
    
    # Skip folded and all-in players
    for _ in range(len(new_state['player_stacks'])):
        if (not new_state['player_folded'][next_player] and 
            not new_state['player_all_in'][next_player]):
            break
        next_player = (next_player + 1) % len(new_state['player_stacks'])
    
    new_state['current_player'] = next_player
    
    return new_state

def is_round_complete(state: Dict) -> bool:
    """Check if betting round is complete"""
    active_players = ~state['player_folded'] & ~state['player_all_in']
    if jnp.sum(active_players) <= 1:
        return True
    
    max_bet = jnp.max(state['player_bets'])
    all_called = jnp.all(
        (state['player_bets'] == max_bet) | 
        state['player_folded'] | 
        state['player_all_in']
    )
    
    return all_called

def advance_round(state: Dict) -> Dict:
    """Advance to next betting round"""
    new_state = {k: v.copy() if hasattr(v, 'copy') else v for k, v in state.items()}
    
    current_round = state['round']
    
    if current_round == 0:  # Preflop -> Flop
        new_state['community_cards'] = new_state['community_cards'].at[0:3].set(
            new_state['deck'][new_state['deck_index']:new_state['deck_index']+3]
        )
        new_state['deck_index'] += 3
        
    elif current_round == 1:  # Flop -> Turn
        new_state['community_cards'] = new_state['community_cards'].at[3:4].set(
            new_state['deck'][new_state['deck_index']:new_state['deck_index']+1]
        )
        new_state['deck_index'] += 1
        
    elif current_round == 2:  # Turn -> River
        new_state['community_cards'] = new_state['community_cards'].at[4:5].set(
            new_state['deck'][new_state['deck_index']:new_state['deck_index']+1]
        )
        new_state['deck_index'] += 1
        
    new_state['round'] += 1
    new_state['current_player'] = 0  # First to act after blinds
    new_state['player_bets'] = jnp.zeros_like(new_state['player_bets'])
    
    return new_state

def calculate_payoffs(state: Dict) -> jnp.ndarray:
    """Calculate final payoffs"""
    from poker_bot.core.simulation import _evaluate_hand_strength
    
    num_players = len(state['player_stacks'])
    payoffs = jnp.zeros(num_players)
    
    # Get active players
    active_mask = ~state['player_folded']
    active_indices = jnp.where(active_mask)[0]
    
    if len(active_indices) == 1:
        # Single winner gets entire pot
        winner = active_indices[0]
        payoffs = payoffs.at[winner].set(state['pot'])
        return payoffs - state['player_bets']
    
    # Evaluate hands
    hand_strengths = []
    for i in active_indices:
        # Combine hole cards with community cards
        hole = state['hole_cards'][i]
        community = state['community_cards'][state['community_cards'] >= 0]
        full_hand = jnp.concatenate([hole, community])
        strength = _evaluate_hand_strength(full_hand)
        hand_strengths.append((i, strength))
    
    # Find winner(s)
    if hand_strengths:
        strengths = jnp.array([s[1] for s in hand_strengths])
        max_strength = jnp.max(strengths)
        winners = [idx for idx, s in hand_strengths if s[1] == max_strength]
        
        # Split pot among winners
        winner_share = state['pot'] / len(winners)
        for winner in winners:
            payoffs = payoffs.at[winner].set(winner_share)
    
    return payoffs - state['player_bets']

def simulate_game(rng_key: jnp.ndarray,
                 num_players: int = 6,
                 small_blind: float = 1.0,
                 big_blind: float = 2.0,
                 starting_stack: float = 100.0) -> Dict:
    """Simulate complete NLHE game"""
    
    state = create_initial_state(rng_key, num_players, small_blind, big_blind, starting_stack)
    
    # Game loop
    max_actions = 100
    for _ in range(max_actions):
        # Check if game should end
        active_players = jnp.sum(~state['player_folded'])
        if active_players <= 1 or state['round'] >= 4:
            break
            
        # Check if round complete
        if is_round_complete(state):
            if state['round'] < 3:  # Not river yet
                state = advance_round(state)
            else:
                break  # River complete, go to showdown
            continue
            
        # Get legal actions
        legal_actions = get_legal_actions(state)
        
        # Simple random action (for testing)
        valid_actions = jnp.where(legal_actions)[0]
        if len(valid_actions) > 0:
            action = valid_actions[0]  # Take first valid action
            amount = 10.0 if action in [PlayerAction.BET, PlayerAction.RAISE] else 0.0
            state = apply_action(state, action, amount)
    
    # Calculate final payoffs
    payoffs = calculate_payoffs(state)
    
    return {
        'payoffs': payoffs,
        'final_community': state['community_cards'],
        'hole_cards': state['hole_cards'],
        'final_pot': state['pot'],
        'player_stacks': state['player_stacks'],
        'player_bets': state['player_bets']
    }

# Vectorized version
@jax.jit
def batch_simulate(rng_keys: jnp.ndarray,
                  num_players: int = 6,
                  small_blind: float = 1.0,
                  big_blind: float = 2.0,
                  starting_stack: float = 100.0) -> Dict:
    """Vectorized batch simulation"""
    
    def simulate_single(key):
        return simulate_game(key, num_players, small_blind, big_blind, starting_stack)
    
    return jax.vmap(simulate_single)(rng_keys)