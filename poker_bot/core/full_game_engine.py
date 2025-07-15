import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, ShapeDtypeStruct
from dataclasses import dataclass
from poker_bot.evaluator import HandEvaluator
from jax.tree_util import register_pytree_node_class
from functools import partial

# --- Constantes y Wrapper de Evaluador ---
MAX_GAME_LENGTH = 60
evaluator = HandEvaluator()

def evaluate_hand_wrapper(cards_np: np.ndarray) -> np.int32:
    valid_cards = cards_np[cards_np != -1]
    cards_list = valid_cards.tolist()
    if len(cards_list) < 5:
        # Si no hay suficientes cartas para una mano de póker, devuelve la peor puntuación posible.
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
    history: Array
    action_counter: Array

    def tree_flatten(self):
        children = (self.stacks, self.bets, self.player_status, self.hole_cards,
                    self.community_cards, self.current_player_idx, self.street,
                    self.pot_size, self.deck, self.deck_pointer,
                    self.num_players_acted_this_round, self.history, self.action_counter)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# --- Unidades de Trabajo Compiladas con JIT ---
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
    
    stacks = initial_stacks.at[sb_player].add(-sb_amount).at[bb_player].add(-bb_amount)
    bets = initial_bets.at[sb_player].set(sb_amount).at[bb_player].set(bb_amount)
    
    return GameState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=hole_cards,
        community_cards=jnp.full((5,), -1, dtype=jnp.int8),
        current_player_idx=jnp.array([2], dtype=jnp.int8),
        street=jnp.array([0], dtype=jnp.int8),
        pot_size=jnp.array([sb_amount + bb_amount]),
        deck=shuffled_deck,
        deck_pointer=jnp.array([12], dtype=jnp.int32),
        num_players_acted_this_round=jnp.array([0], dtype=jnp.int32),
        history=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        action_counter=jnp.array([0], dtype=jnp.int32)
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
    def body_fun(i, current_idx):
        next_idx = (start_idx + i + 1) % 6
        is_active = (state.player_status[next_idx] == 0)
        return jax.lax.cond(is_active, lambda: next_idx, lambda: current_idx)
    next_player = jax.lax.fori_loop(0, 6, body_fun, start_idx)
    return GameState(**{**state.__dict__, "current_player_idx": jnp.array([next_player], dtype=jnp.int8)})

@jax.jit
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]
    def do_fold(s):
        new_status = s.player_status.at[player_idx].set(1)
        return GameState(**{**s.__dict__, "player_status": new_status})
    def do_check(s):
        return s
    def do_call(s):
        amount = jnp.max(s.bets) - s.bets[player_idx]
        new_stacks = s.stacks.at[player_idx].add(-amount)
        new_bets = s.bets.at[player_idx].add(amount)
        new_pot = s.pot_size + amount
        return GameState(**{**s.__dict__, "stacks": new_stacks, "bets": new_bets, "pot_size": new_pot})
    def do_bet_raise(s):
        bet_size = 20.0
        new_stacks = s.stacks.at[player_idx].add(-bet_size)
        new_bets = s.bets.at[player_idx].add(bet_size)
        new_pot = s.pot_size + bet_size
        return GameState(**{**s.__dict__, "stacks": new_stacks, "bets": new_bets, "pot_size": new_pot})
    state_after_action = jax.lax.switch(jnp.clip(action, 0, 3), [do_fold, do_check, do_call, do_bet_raise], state)
    return _update_turn(state_after_action)

@partial(jax.jit, static_argnums=(1,))
def _deal_community_cards(state: GameState, num_to_deal: int) -> GameState:
    start = state.deck_pointer[0]
    cards = jax.lax.dynamic_slice(state.deck, (start,), (num_to_deal,))
    num_dealt = jnp.sum(state.community_cards != -1)
    new_community = jax.lax.dynamic_update_slice(state.community_cards, cards, (num_dealt,))
    return GameState(**{**state.__dict__, "community_cards": new_community, "deck_pointer": state.deck_pointer + num_to_deal, "street": state.street + 1})

# --- Orquestación y Vectorización ---

@jax.jit
def play_game(initial_state: GameState, policy_logits: Array, key: Array):
    state = initial_state
    
    # Bucle de las calles (Preflop, Flop, Turn, River)
    def street_loop_body(street_idx, state_key_tuple):
        state, key = state_key_tuple
        
        # Repartir cartas si no es preflop
        state = jax.lax.cond(street_idx > 0,
                             lambda s: _deal_community_cards(s, jax.lax.switch(street_idx, [0, 3, 1, 1])),
                             lambda s: s,
                             state)
        
        # Bucle de la ronda de apuestas
        def betting_loop_cond(carry):
            state, key = carry
            num_active = jnp.sum(state.player_status == 0)
            player_idx = state.current_player_idx[0]
            bets_equal = (state.bets[player_idx] == jnp.max(state.bets))
            all_acted = (state.num_players_acted_this_round[0] >= num_active)
            return (num_active > 1) & (~(bets_equal & all_acted))

        def betting_loop_body(carry):
            state, key = carry
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
            return final_state, next_key

        key, betting_key = jax.random.split(key)
        state, _ = jax.lax.while_loop(betting_loop_cond, betting_loop_body, (state, betting_key))
        
        # Resetear contador para la siguiente calle
        state = GameState(**{**state.__dict__, "num_players_acted_this_round": jnp.array([0])})
        return state, key

    # Ejecutar el bucle de las calles
    state, _ = jax.lax.fori_loop(0, 4, street_loop_body, (state, key))
    return state

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
            return jax.pure_callback(evaluate_hand_wrapper, ShapeDtypeStruct((), np.int32), cards, vmap_method='sequential')
        
        strengths = jnp.array([jax.lax.cond(active_mask[i], lambda: player_hand_eval(i), lambda: 9999) for i in range(6)])
        best_strength = jnp.min(strengths)
        
        winners_mask = (strengths == best_strength) & active_mask
        num_winners = jnp.sum(winners_mask)
        win_share = pot / jnp.sum(winners_mask)
        
        return -state.bets + (winners_mask * win_share)
    
    # Solo ir a showdown si hay cartas comunitarias (es decir, post-flop)
    # y más de un jugador activo.
    can_showdown = (jnp.sum(state.community_cards != -1) >= 3) & (jnp.sum(active_mask) > 1)
    
    return jax.lax.cond(can_showdown, showdown_case, single_winner_case)

def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    keys = jax.random.split(key, batch_size)
    
    vmap_initial_state = jax.vmap(create_initial_state)
    vmap_play_game = jax.vmap(play_game, in_axes=(0, None, 0))
    vmap_resolve_showdown = jax.vmap(resolve_showdown)

    initial_states = vmap_initial_state(keys)
    final_states = vmap_play_game(initial_states, policy_logits, keys)
    payoffs = vmap_resolve_showdown(final_states)
    
    return final_states, payoffs, final_states.history, initial_states