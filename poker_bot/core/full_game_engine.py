import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from dataclasses import dataclass
from poker_bot.evaluator import HandEvaluator
from jax.tree_util import register_pytree_node_class

# --- Constantes y Wrapper de Evaluador ---
MAX_GAME_LENGTH = 60
evaluator = HandEvaluator()

def evaluate_hand_wrapper(cards_np: np.ndarray) -> np.int32:
    valid_cards = cards_np[cards_np != -1]
    cards_list = valid_cards.tolist()
    if len(cards_list) < 5:
        return np.int32(9999)
    return np.int32(evaluator.evaluate_single(cards_list))

# --- Estructura de Datos Principal (Pytree) ---
@register_pytree_node_class
@dataclass(frozen=True)
class GameState:
    stacks: Array
    bets: Array
    player_status: Array
    hole_cards: Array
    community_cards: Array
    current_player_idx: Array
    street: Array
    pot_size: Array
    deck: Array
    deck_pointer: Array
    num_players_acted_this_round: Array

    def tree_flatten(self):
        children = (self.stacks, self.bets, self.player_status, self.hole_cards,
                    self.community_cards, self.current_player_idx, self.street,
                    self.pot_size, self.deck, self.deck_pointer,
                    self.num_players_acted_this_round)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# --- UNIDADES DE TRABAJO COMPILADAS (RÁPIDAS) ---
@jax.jit
def create_initial_state(key: Array) -> GameState:
    deck = jnp.arange(52, dtype=jnp.int8)
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, deck)
    hole_cards = shuffled_deck[:12].reshape((6, 2))
    
    initial_stacks = jnp.full((6,), 1000.0)
    initial_bets = jnp.zeros((6,))
    sb_player, bb_player = 0, 1
    sb_amount, bb_amount = 5.0, 10.0
    
    stacks = initial_stacks.at[sb_player].set(initial_stacks[sb_player] - sb_amount)
    stacks = stacks.at[bb_player].set(stacks[bb_player] - bb_amount)
    bets = initial_bets.at[sb_player].set(sb_amount)
    bets = bets.at[bb_player].set(bb_amount)
    
    return GameState(
        stacks=stacks, bets=bets, player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=hole_cards, community_cards=jnp.full((5,), -1, dtype=jnp.int8),
        current_player_idx=jnp.array([2], dtype=jnp.int8), street=jnp.array([0], dtype=jnp.int8),
        pot_size=jnp.array([sb_amount + bb_amount]), deck=shuffled_deck,
        deck_pointer=jnp.array([12], dtype=jnp.int32),
        num_players_acted_this_round=jnp.array([0], dtype=jnp.int32)
    )

@jax.jit
def get_legal_actions(state: GameState, num_actions: int = 14) -> Array:
    mask = jnp.zeros(num_actions, dtype=jnp.bool_)
    player_idx = state.current_player_idx[0]
    status = state.player_status[player_idx]
    can_act = (status == 0)
    mask = mask.at[0].set(can_act)
    current_bet = state.bets[player_idx]
    max_bet = jnp.max(state.bets)
    can_check = (current_bet == max_bet)
    mask = mask.at[1].set(can_check & can_act)
    to_call = max_bet - current_bet
    can_call = (to_call > 0) & (state.stacks[player_idx] >= to_call)
    mask = mask.at[2].set(can_call & can_act)
    mask = mask.at[3:].set((~can_check) & can_act)
    return mask

@jax.jit
def _update_turn(state: GameState) -> GameState:
    start_idx = state.current_player_idx[0]
    next_player = start_idx
    for i in range(1, 7):
        next_player = (start_idx + i) % 6
        if state.player_status[next_player] == 0:
            break
    return GameState(**{**state.__dict__, "current_player_idx": jnp.array([next_player], dtype=jnp.int8)})

@jax.jit
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]
    if action == 0: # Fold
        new_status = state.player_status.at[player_idx].set(1)
        return GameState(**{**state.__dict__, "player_status": new_status})
    elif action == 1: # Check
        return state
    elif action == 2: # Call
        amount = jnp.max(state.bets) - state.bets[player_idx]
        new_stacks = state.stacks.at[player_idx].add(-amount)
        new_bets = state.bets.at[player_idx].add(amount)
        new_pot = state.pot_size + amount
        return GameState(**{**state.__dict__, "stacks": new_stacks, "bets": new_bets, "pot_size": new_pot})
    else: # Bet/Raise
        bet_size = 20.0
        new_stacks = state.stacks.at[player_idx].add(-bet_size)
        new_bets = state.bets.at[player_idx].add(bet_size)
        new_pot = state.pot_size + bet_size
        return GameState(**{**state.__dict__, "stacks": new_stacks, "bets": new_bets, "pot_size": new_pot})

# --- ORQUESTACIÓN EN PYTHON PURO ---

def run_betting_round(state: GameState, policy_logits: Array, key: Array):
    history_in_round = []
    for _ in range(30):
        num_active = jnp.sum(state.player_status == 0)
        if num_active <= 1: break
        
        player_idx = state.current_player_idx[0]
        bets_equal = (state.bets[player_idx] == jnp.max(state.bets))
        all_acted = (state.num_players_acted_this_round[0] >= num_active)
        if bets_equal and all_acted and state.num_players_acted_this_round[0] > 0: break
            
        logits = policy_logits[player_idx]
        legal_mask = get_legal_actions(state)
        masked_logits = jnp.where(legal_mask, logits, -1e9)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, masked_logits)
        
        history_in_round.append(action)
        state_after_action = step(state, action)
        state = _update_turn(state_after_action)

        is_aggressive = (action >= 3)
        num_acted = state.num_players_acted_this_round[0] + 1 if not is_aggressive else 1
        state = GameState(**{**state.__dict__, "num_players_acted_this_round": jnp.array([num_acted])})

    return state, history_in_round

def play_game(initial_state: GameState, policy_logits: Array, key: Array):
    state = initial_state
    full_history = []
    
    # Preflop
    key, subkey = jax.random.split(key)
    state, history = run_betting_round(state, policy_logits, subkey)
    full_history.extend(history)
    
    # Flop
    if jnp.sum(state.player_status != 1) > 1:
        state = _deal_community_cards(state, 3)
        key, subkey = jax.random.split(key)
        state, history = run_betting_round(state, policy_logits, subkey)
        full_history.extend(history)

    # Turn
    if jnp.sum(state.player_status != 1) > 1:
        state = _deal_community_cards(state, 1)
        key, subkey = jax.random.split(key)
        state, history = run_betting_round(state, policy_logits, subkey)
        full_history.extend(history)

    # River
    if jnp.sum(state.player_status != 1) > 1:
        state = _deal_community_cards(state, 1)
        key, subkey = jax.random.split(key)
        state, history = run_betting_round(state, policy_logits, subkey)
        full_history.extend(history)
        
    padded_history = np.full((MAX_GAME_LENGTH,), -1, dtype=np.int32)
    padded_history[:len(full_history)] = full_history
    return state, jnp.array(padded_history)

@jax.jit
def resolve_showdown(state: GameState) -> Array:
    active_mask = (state.player_status != 1)
    pot = jnp.sum(state.bets)
    
    def single_winner_case():
        winner_idx = jnp.argmax(active_mask)
        payoffs = -state.bets
        return payoffs.at[winner_idx].add(pot)
        
    def showdown_case():
        def player_hand_eval(i):
            cards = jnp.concatenate([state.hole_cards[i], state.community_cards])
            return jax.pure_callback(evaluate_hand_wrapper, ShapeDtypeStruct((), np.int32), cards)
        strengths = jnp.array([jax.lax.cond(active_mask[i], lambda: player_hand_eval(i), lambda: 9999) for i in range(6)])
        best_strength = jnp.min(strengths)
        winners_mask = (strengths == best_strength) & active_mask
        win_share = pot / jnp.sum(winners_mask)
        return -state.bets + (winners_mask * win_share)
        
    can_showdown = (jnp.sum(state.community_cards != -1) >= 5) & (jnp.sum(active_mask) > 1)
    return jax.lax.cond(can_showdown, showdown_case, single_winner_case)

def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    keys = jax.random.split(key, batch_size)
    final_states, histories, initial_states = [], [], []
    
    vmap_initial_state = jax.vmap(create_initial_state)
    initial_states_batch = vmap_initial_state(keys)

    for i in range(batch_size):
        initial_state = jax.tree_util.tree_map(lambda x: x[i], initial_states_batch)
        final_state, history = play_game(initial_state, policy_logits, keys[i])
        final_states.append(final_state)
        histories.append(history)
        initial_states.append(initial_state)

    stacked_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *final_states)
    stacked_histories = jnp.stack(histories)
    stacked_initial_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *initial_states)
    
    vmap_resolve_showdown = jax.vmap(resolve_showdown)
    payoffs = vmap_resolve_showdown(stacked_states)
    
    return stacked_states, payoffs, stacked_histories, stacked_initial_states