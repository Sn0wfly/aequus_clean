import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from dataclasses import dataclass
from poker_bot.evaluator import HandEvaluator
from jax.tree_util import register_pytree_node_class

# --- Constantes y Wrapper ---
MAX_GAME_LENGTH = 60
evaluator = HandEvaluator()

def evaluate_hand_wrapper(cards_np: np.ndarray) -> np.int32:
    valid_cards = cards_np[cards_np != -1]
    cards_list = valid_cards.tolist()
    if len(cards_list) < 5:
        return np.int32(9999)
    return np.int32(evaluator.evaluate_single(cards_list))

# --- Estructura de Datos (Pytree) ---
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
    history: Array
    action_counter: Array

    def tree_flatten(self):
        return ((self.stacks, self.bets, self.player_status, self.hole_cards,
                 self.community_cards, self.current_player_idx, self.street,
                 self.pot_size, self.deck, self.deck_pointer,
                 self.num_players_acted_this_round, self.history, self.action_counter), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# --- Unidades de Trabajo Compiladas (JIT) ---
@jax.jit
def create_initial_state(key: Array) -> GameState:
    deck = jnp.arange(52, dtype=jnp.int8)
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, deck)
    hole_cards = shuffled_deck[:12].reshape((6, 2))
    
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].add(-5.0).at[1].add(-10.0)

    return GameState(
        stacks=stacks, bets=bets, player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=hole_cards, community_cards=jnp.full((5,), -1, dtype=jnp.int8),
        current_player_idx=jnp.array([2], dtype=jnp.int8), street=jnp.array([0], dtype=jnp.int8),
        pot_size=jnp.array([15.0]), deck=shuffled_deck,
        deck_pointer=jnp.array([12]), num_players_acted_this_round=jnp.array([0]),
        history=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        action_counter=jnp.array([0])
    )

@jax.jit
def get_legal_actions(state: GameState, num_actions: int = 14) -> Array:
    mask = jnp.zeros(num_actions, dtype=jnp.bool_)
    player_idx = state.current_player_idx[0]
    can_act = (state.player_status[player_idx] == 0)
    
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
    next_player_idx = (start_idx + 1) % 6
    def cond_fun(val):
        i, p_idx = val
        return (state.player_status[p_idx] != 0) & (i < 6)
    def body_fun(val):
        i, p_idx = val
        return (i + 1, (p_idx + 1) % 6)
    _, final_next_player = jax.lax.while_loop(cond_fun, body_fun, (0, next_player_idx))
    return GameState(**{**state.__dict__, "current_player_idx": jnp.array([final_next_player], dtype=jnp.int8)})

@jax.jit
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]
    branches = [
        lambda s: GameState(**{**s.__dict__, "player_status": s.player_status.at[player_idx].set(1)}), # Fold
        lambda s: s, # Check
        lambda s: GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-(jnp.max(s.bets) - s.bets[player_idx])), 
                                           "bets": s.bets.at[player_idx].set(jnp.max(s.bets)), 
                                           "pot_size": s.pot_size + (jnp.max(s.bets) - s.bets[player_idx])}), # Call
        lambda s: GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-20.0), 
                                           "bets": s.bets.at[player_idx].add(20.0), 
                                           "pot_size": s.pot_size + 20.0}) # Bet/Raise
    ]
    state_after_action = jax.lax.switch(jnp.clip(action, 0, 3), branches, state)
    return _update_turn(state_after_action)

@jax.jit
def _deal_community(state, num_to_deal, street_num):
    start = state.deck_pointer[0]
    cards = jax.lax.dynamic_slice(state.deck, (start,), (num_to_deal,))
    num_dealt = jnp.sum(state.community_cards != -1)
    new_community = jax.lax.dynamic_update_slice(state.community_cards, cards, (num_dealt,))
    return GameState(**{**state.__dict__, 
                        "community_cards": new_community, 
                        "deck_pointer": state.deck_pointer + num_to_deal,
                        "street": jnp.array([street_num], dtype=jnp.int8),
                        "num_players_acted_this_round": jnp.array([0])})

# --- Orquestación y Vectorización (TODO COMPILADO) ---
@jax.jit
def play_game(initial_state: GameState, policy_logits: Array, key: Array):
    
    def street_loop_body(street_idx, state_key_tuple):
        state, key = state_key_tuple
        
        # Repartir cartas si aplica
        state = jax.lax.cond(street_idx == 1, lambda s: _deal_community(s, 3, 1), lambda s: s, state)
        state = jax.lax.cond(street_idx == 2, lambda s: _deal_community(s, 1, 2), lambda s: s, state)
        state = jax.lax.cond(street_idx == 3, lambda s: _deal_community(s, 1, 3), lambda s: s, state)
        
        # Ronda de apuestas
        def betting_loop_cond(carry):
            state, key, i = carry
            num_active = jnp.sum(state.player_status == 0)
            player_idx = state.current_player_idx[0]
            bets_equal = (state.bets[player_idx] == jnp.max(state.bets))
            all_acted = (state.num_players_acted_this_round[0] >= num_active)
            round_is_active = (num_active > 1) & (~(all_acted & bets_equal) | (state.num_players_acted_this_round[0]==0))
            return round_is_active & (i < 30)

        def betting_loop_body(carry):
            state, key, i = carry
            player_idx = state.current_player_idx[0]
            
            logits = policy_logits[player_idx]
            legal_mask = get_legal_actions(state)
            masked_logits = jnp.where(legal_mask, logits, -1e9)
            
            action_key, next_key = jax.random.split(key)
            action = jax.random.categorical(action_key, masked_logits)
            
            ac = state.action_counter[0]
            new_history = state.history.at[ac].set(action)
            state_before_step = GameState(**{**state.__dict__, "history": new_history, "action_counter": state.action_counter + 1})
            
            state_after_action = step(state_before_step, action)

            is_aggressive = (action >= 3)
            num_acted = jax.lax.cond(is_aggressive, lambda: 1, lambda: state.num_players_acted_this_round[0] + 1)
            final_state = GameState(**{**state_after_action.__dict__, "num_players_acted_this_round": jnp.array([num_acted])})
            return final_state, next_key, i + 1
        
        is_game_over = jnp.sum(state.player_status != 1) <= 1
        
        final_state, final_key, _ = jax.lax.cond(
            is_game_over,
            lambda c: c,
            lambda c: jax.lax.while_loop(betting_loop_cond, betting_loop_body, c),
            (state, key, 0)
        )
        return final_state, final_key

    final_state, _ = jax.lax.fori_loop(0, 4, street_loop_body, (initial_state, key))
    return final_state

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
        win_share = pot / jnp.maximum(1, jnp.sum(winners_mask))
        return -state.bets + (winners_mask * win_share)
        
    can_showdown = (jnp.sum(state.community_cards != -1) >= 5) & (jnp.sum(active_mask) > 1)
    return jax.lax.cond(can_showdown, showdown_case, single_winner_case)

def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    keys = jax.random.split(key, batch_size)
    vmap_initial_state = jax.vmap(create_initial_state)
    initial_states = vmap_initial_state(keys)

    vmap_play_game = jax.vmap(play_game, in_axes=(0, None, 0))
    final_states = vmap_play_game(initial_states, policy_logits, keys)
    
    vmap_resolve_showdown = jax.vmap(resolve_showdown)
    payoffs = vmap_resolve_showdown(final_states)
    
    return final_states, payoffs, final_states.history, initial_states