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
    # phevaluator requiere entre 5 y 7 cartas.
    if len(cards_list) < 5:
        return np.int32(9999) # Devuelve la peor puntuación si no hay suficientes cartas.
    return np.int32(evaluator.evaluate_single(cards_list))

# --- Estructura de Datos Principal (Pytree) ---
@register_pytree_node_class
@dataclass(frozen=True)
class GameState:
    # Define todos los campos para el estado del juego.
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
        stacks=stacks, bets=bets, player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=hole_cards, community_cards=jnp.full((5,), -1, dtype=jnp.int8),
        current_player_idx=jnp.array([2], dtype=jnp.int8), street=jnp.array([0], dtype=jnp.int8),
        pot_size=jnp.array([sb_amount + bb_amount]), deck=shuffled_deck,
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
        next_idx = (start_idx + 1 + i) % 6
        is_active = (state.player_status[next_idx] == 0)
        return jax.lax.cond(is_active, lambda: next_idx, lambda: current_idx)
    next_player = jax.lax.fori_loop(0, 6, body_fun, start_idx)
    # Recreamos el objeto para asegurar la inmutabilidad
    return GameState(
        **{**state.__dict__, "current_player_idx": jnp.array([next_player], dtype=jnp.int8)}
    )

@jax.jit
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]
    # Usamos jax.lax.switch, que es la forma correcta de hacer un 'if/elif' en JIT
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
    
    # Se usa clip para asegurar que la acción esté en el rango de nuestras funciones
    state_after_action = jax.lax.switch(
        jnp.clip(action, 0, 3), 
        [do_fold, do_check, do_call, do_bet_raise],
        state
    )
    return _update_turn(state_after_action)

@jax.jit
def _deal_community_cards(state: GameState, num_to_deal: int) -> GameState:
    start = state.deck_pointer[0]
    # Usamos dynamic_slice con tamaños estáticos donde es posible
    cards = jax.lax.dynamic_slice_in_dim(state.deck, start, num_to_deal)
    num_dealt = jnp.sum(state.community_cards != -1)
    new_community = jax.lax.dynamic_update_slice(state.community_cards, cards, (num_dealt,))
    return GameState(**{**state.__dict__, "community_cards": new_community, "deck_pointer": state.deck_pointer + num_to_deal})

# --- ORQUESTACIÓN Y VECTORIZACIÓN (TODO COMPILADO) ---
@jax.jit
def play_game(initial_state: GameState, policy_logits: Array, key: Array):
    
    def street_loop_body(street_idx, state_key_tuple):
        state, key = state_key_tuple
        
        # Lógica de reparto de cartas
        def deal_flop(s): return _deal_community_cards(s, 3)
        def deal_turn_river(s): return _deal_community_cards(s, 1)
        
        state = jax.lax.switch(street_idx, [
            lambda s: s, # Preflop
            deal_flop,   # Flop
            deal_turn_river, # Turn
            deal_turn_river  # River
        ], state)
        
        # Ronda de apuestas
        def betting_loop_cond(carry):
            state, key, i = carry
            num_active = jnp.sum(state.player_status == 0)
            player_idx = state.current_player_idx[0]
            bets_equal = (state.bets[player_idx] == jnp.max(state.bets))
            all_acted = (state.num_players_acted_this_round[0] >= num_active)
            # La ronda continúa si hay más de 1 jugador y la ronda no ha terminado
            return (num_active > 1) & (~(all_acted & bets_equal) | (state.num_players_acted_this_round[0] == 0)) & (i < 30)

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
        
        key, betting_key = jax.random.split(key)
        state, _, _ = jax.lax.while_loop(betting_loop_cond, betting_loop_body, (state, betting_key, 0))
        
        state = GameState(**{**state.__dict__, "street": state.street + 1, "num_players_acted_this_round": jnp.array([0])})
        return state, key

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

# Función de interfaz final
def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    keys = jax.random.split(key, batch_size)
    
    # Vectorizamos todo el proceso
    vmap_play_and_resolve = jax.vmap(
        lambda k: resolve_showdown(play_game(create_initial_state(k), policy_logits, k)),
        in_axes=(0)
    )
    
    # Obtenemos los payoffs directamente
    payoffs = vmap_play_and_resolve(keys)
    
    # Devolvemos una estructura compatible (placeholders para histories y states)
    # En una implementación real, si se necesitaran, se devolverían de la vmap.
    dummy_histories = jnp.full((batch_size, MAX_GAME_LENGTH), -1)
    vmap_initial_state = jax.vmap(create_initial_state)
    initial_states = vmap_initial_state(keys)

    return initial_states, payoffs, dummy_histories, initial_states