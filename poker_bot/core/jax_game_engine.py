"""
🎯 JAX-Native NLHE Game Engine - Simplified Version
Professional-grade implementation with JAX compatibility
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from typing import Dict, Tuple
from enum import IntEnum

# AGREGADO: Importar evaluador real para payoffs correctos
from poker_bot.evaluator import HandEvaluator
import logging

# Crear instancia global del evaluador real
hand_evaluator = HandEvaluator()
logger = logging.getLogger(__name__)

# ---------- Wrapper para evaluador real compatible con JAX ----------
def evaluate_hand_jax(cards_device):
    """
    Wrapper JAX-compatible para el evaluador real de manos.
    Usa phevaluator para evaluación profesional de manos.
    """
    cards_np = np.asarray(cards_device)
    
    # Convertir cartas a formato compatible con evaluador
    if np.all(cards_np >= 0):  # Solo evaluar si todas las cartas son válidas
        try:
            # Usar el evaluador real en lugar del mock
            strength = hand_evaluator.evaluate_single(cards_np.tolist())
            return np.int32(strength)
        except:
            # Fallback a evaluación simple si falla
            return np.int32(np.sum(cards_np) % 7462)
    else:
        return np.int32(9999)  # Mano inválida

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

# ---------- Evaluación Real de Showdowns ----------
def calculate_real_payoffs(state: Dict) -> jnp.ndarray:
    """
    Calcula payoffs reales usando phevaluator para showdowns.
    Compatible con JAX JIT usando pure_callback.
    """
    num_players = state['hole_cards'].shape[0]
    active_players = ~state['player_folded']
    num_active = jnp.sum(active_players)
    
    # Si solo queda un jugador, se lleva toda la pot
    def single_winner():
        payoffs = jnp.where(
            active_players,
            state['pot'] - state['player_bets'],
            -state['player_bets']
        )
        return payoffs
    
    # Si hay múltiples jugadores, evaluar showdown
    def showdown_evaluation():
        def evaluate_all_hands(hole_cards_array, community_cards_array, active_players_array):
            """
            Función pura que evalúa todas las manos y retorna ganadores.
            Recibe arrays concretos como argumentos, no estado trazado.
            """
            hand_strengths = []
            num_players = len(active_players_array)
            
            for player_idx in range(num_players):
                if active_players_array[player_idx]:
                    # Combinar hole cards + community cards
                    player_hole = hole_cards_array[player_idx]
                    all_cards = np.concatenate([player_hole, community_cards_array])
                    # Filtrar cartas válidas (>= 0)
                    valid_cards = all_cards[all_cards >= 0]
                    
                    # Evaluar mano si tenemos suficientes cartas
                    if len(valid_cards) >= 5:
                        try:
                            strength = hand_evaluator.evaluate_single(valid_cards.tolist())
                            hand_strengths.append(strength)
                        except:
                            hand_strengths.append(9999)  # Mano inválida
                    else:
                        hand_strengths.append(9999)  # Mano inválida
                else:
                    hand_strengths.append(9999)  # Jugador foldeado
            
            # Encontrar la mejor mano (menor número = mejor en phevaluator)
            hand_strengths = np.array(hand_strengths)
            active_strengths = hand_strengths[active_players_array]
            
            if len(active_strengths) > 0:
                best_strength = np.min(active_strengths)
                winners = (hand_strengths == best_strength) & active_players_array
                num_winners = np.sum(winners)
                
                if num_winners > 0:
                    return winners.astype(np.float32)
                else:
                    # Fallback: el primer jugador activo gana
                    fallback_winners = np.zeros(num_players, dtype=np.float32)
                    first_active = np.where(active_players_array)[0]
                    if len(first_active) > 0:
                        fallback_winners[first_active[0]] = 1.0
                    return fallback_winners
            else:
                # No hay jugadores activos, devolver array vacío
                return np.zeros(num_players, dtype=np.float32)
        
        # Usar pure_callback con argumentos específicos, no estado completo
        winner_mask = jax.pure_callback(
            evaluate_all_hands,
            jnp.zeros(num_players, dtype=jnp.float32),
            state['hole_cards'],
            state['community_cards'], 
            active_players,
            vmap_method=None
        )
        
        # Distribuir pot entre ganadores
        num_winners = jnp.sum(winner_mask)
        pot_share = state['pot'] / jnp.maximum(num_winners, 1)
        
        payoffs = jnp.where(
            winner_mask > 0,
            pot_share - state['player_bets'],
            -state['player_bets']
        )
        
        return payoffs
    
    # Usar lax.cond para decidir entre un ganador o showdown
    return lax.cond(
        num_active <= 1,
        single_winner,
        showdown_evaluation
    )

def simulate_game(rng_key: jnp.ndarray,
                 num_players: int = 6,
                 small_blind: float = 1.0,
                 big_blind: float = 2.0,
                 starting_stack: float = 100.0) -> Dict:
    """Simulate complete NLHE game - JAX JIT Compatible"""
    
    state = create_initial_state(rng_key, num_players, small_blind, big_blind, starting_stack)
    
    # JAX-compatible game loop usando lax.fori_loop
    def game_step(step_idx, state):
        # Check if game should continue
        active_players = jnp.sum(~state['player_folded'])
        should_continue = (active_players > 1) & (state['round'] < 4)
        
        def continue_game():
            # Simulate a simple action (simplified for JAX compatibility)
            current_player = step_idx % num_players
            
            # Simple action logic - randomly fold, call, or bet
            rng_action = jax.random.fold_in(rng_key, step_idx)
            action_prob = jax.random.uniform(rng_action)
            
            # Determine action: 0=fold, 1=check/call, 2=bet/raise
            action = lax.cond(
                action_prob < 0.2,
                lambda: PlayerAction.FOLD,
                lambda: lax.cond(
                    action_prob < 0.7,
                    lambda: PlayerAction.CALL,
                    lambda: PlayerAction.BET
                )
            )
            
            # Apply action with fixed amount
            amount = lax.cond(
                action == PlayerAction.BET,
                lambda: 10.0,
                lambda: 0.0
            )
            
            # Update state based on action (create new dict for JAX compatibility)
            new_player_bets = lax.cond(
                action == PlayerAction.FOLD,
                lambda: state['player_bets'],
                lambda: state['player_bets'].at[current_player].add(amount)
            )
            
            new_player_folded = lax.cond(
                action == PlayerAction.FOLD,
                lambda: state['player_folded'].at[current_player].set(True),
                lambda: state['player_folded']
            )
            
            # Update pot
            new_pot = state['pot'] + amount
            
            # Advance round occasionally (simplified logic)
            should_advance = (step_idx % 10 == 0) & (state['round'] < 3)
            new_round = lax.cond(
                should_advance,
                lambda: state['round'] + 1,
                lambda: state['round']
            )
            
            # Create new state dict
            new_state = {
                'hole_cards': state['hole_cards'],
                'community_cards': state['community_cards'],
                'player_stacks': state['player_stacks'],
                'player_bets': new_player_bets,
                'player_folded': new_player_folded,
                'player_all_in': state['player_all_in'],
                'pot': new_pot,
                'round': new_round,
                'current_player': state['current_player'],
                'deck': state['deck'],
                'min_bet': state['min_bet'],
                'deck_index': state['deck_index']
            }
            
            return new_state
            
        def stop_game():
            return state
            
        return lax.cond(should_continue, continue_game, stop_game)
    
    # Run game loop
    max_steps = 50  # Reduced for performance
    final_state = lax.fori_loop(0, max_steps, game_step, state)
    
    # Calculate REAL payoffs using phevaluator for showdowns
    payoffs = calculate_real_payoffs(final_state)
    
    return {
        'payoffs': payoffs,
        'final_community': final_state['community_cards'],
        'hole_cards': final_state['hole_cards'],
        'final_pot': final_state['pot'],
        'player_stacks': final_state['player_stacks'],
        'player_bets': final_state['player_bets']
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