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

# --- Unidades de Trabajo JIT ---

@jax.jit
def create_initial_state(key: Array) -> GameState:
    deck = jnp.arange(52, dtype=jnp.int8)
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, deck)
    hole_cards = shuffled_deck[:12].reshape((6, 2))
    
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].set(stacks[0] - 5.0).at[1].set(stacks[1] - 10.0)
    
    return GameState(
        stacks=stacks, bets=bets, player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=hole_cards, community_cards=jnp.full((5,), -1, dtype=jnp.int8),
        current_player_idx=jnp.array([2], dtype=jnp.int8), street=jnp.array([0], dtype=jnp.int8),
        pot_size=jnp.array([15.0]), deck=shuffled_deck,
        deck_pointer=jnp.array([12]), num_players_acted_this_round=jnp.array([0])
    )

@jax.jit
def get_legal_actions(state: GameState, num_actions: int = 14) -> Array:
    mask = jnp.zeros(num_actions, dtype=jnp.bool_)
    player_idx = state.current_player_idx[0]
    status = state.player_status[player_idx]
    can_act = (status == 0)
    
    mask = mask.at[0].set(can_act) # Fold
    current_bet = state.bets[player_idx]
    max_bet = jnp.max(state.bets)
    can_check = (current_bet == max_bet)
    mask = mask.at[1].set(can_check & can_act) # Check
    to_call = max_bet - current_bet
    can_call = (to_call > 0) & (state.stacks[player_idx] >= to_call)
    mask = mask.at[2].set(can_call & can_act) # Call
    mask = mask.at[3:].set((~can_check) & can_act) # Bet/Raise
    return mask

@jax.jit
def _update_turn(state: GameState) -> GameState:
    start_idx = state.current_player_idx[0]
    def body_fun(i, current_idx):
        next_idx = (start_idx + 1 + i) % 6
        return jax.lax.cond(state.player_status[next_idx] == 0, lambda: next_idx, lambda: current_idx)
    next_player = jax.lax.fori_loop(0, 6, body_fun, start_idx)
    return GameState(**{**state.__dict__, "current_player_idx": jnp.array([next_player], dtype=jnp.int8)})

@jax.jit
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]
    
    def do_fold(s): return GameState(**{**s.__dict__, "player_status": s.player_status.at[player_idx].set(1)})
    def do_check(s): return s
    def do_call(s):
        amount = jnp.max(s.bets) - s.bets[player_idx]
        return GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-amount), "bets": s.bets.at[player_idx].add(amount), "pot_size": s.pot_size + amount})
    def do_bet_raise(s):
        bet_size = 20.0
        return GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-bet_size), "bets": s.bets.at[player_idx].add(bet_size), "pot_size": s.pot_size + bet_size})
        
    state_after_action = jax.lax.switch(jnp.clip(action, 0, 3), [do_fold, do_check, do_call, do_bet_raise], state)
    
    is_aggressive = (action >= 3)
    num_acted = jax.lax.cond(is_aggressive, lambda: 1, lambda: state.num_players_acted_this_round[0] + 1)
    state_with_counter = GameState(**{**state_after_action.__dict__, "num_players_acted_this_round": jnp.array([num_acted])})
    
    return _update_turn(state_with_counter)

@jax.jit
def _deal_community_cards(state: GameState, street_idx: int) -> GameState:
    def deal_flop(s):
        cards = jax.lax.dynamic_slice_in_dim(s.deck, s.deck_pointer[0], 3)
        return GameState(**{**s.__dict__, "community_cards": s.community_cards.at[:3].set(cards), "deck_pointer": s.deck_pointer + 3})
    def deal_turn_river(s, street):
        card = jax.lax.dynamic_slice_in_dim(s.deck, s.deck_pointer[0], 1)
        new_comm = jax.lax.dynamic_update_slice(s.community_cards, card, (2 + street,))
        return GameState(**{**s.__dict__, "community_cards": new_comm, "deck_pointer": s.deck_pointer + 1})

    return jax.lax.switch(street_idx, [
        lambda s: s,
        deal_flop,
        lambda s: deal_turn_river(s, 2),
        lambda s: deal_turn_river(s, 3),
    ], state)

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
    return jax.lax.cond(jnp.sum(active_mask) <= 1, single_winner_case, 
                       lambda: jax.lax.cond(can_showdown, showdown_case, single_winner_case))

# --- ARQUITECTURA FINAL: UN ÚNICO BLOQUE COMPILABLE ---
@jax.jit
def batch_play_and_resolve(key: Array, policy_logits: Array, batch_size: int):
    
    keys = jax.random.split(key, batch_size)
    
    # 1. Crear batch de estados iniciales
    initial_states = jax.vmap(create_initial_state)(keys)

    # 2. Bucle de juego principal sobre el tiempo (max_steps)
    def game_loop_body(i, carry):
        state, continue_mask, histories, key = carry
        
        # 2a. Avanzar calle (solo al principio de cada calle)
        is_flop = (i == 15) # Estimación de cuándo repartir
        is_turn = (i == 20)
        is_river = (i == 25)
        
        state = jax.vmap(_deal_community_cards, in_axes=(0, None))(state, jax.lax.cond(is_flop, lambda:1, lambda:0))
        state = jax.vmap(_deal_community_cards, in_axes=(0, None))(state, jax.lax.cond(is_turn, lambda:2, lambda:0))
        state = jax.vmap(_deal_community_cards, in_axes=(0, None))(state, jax.lax.cond(is_river, lambda:3, lambda:0))

        # 2b. Seleccionar acción para todos los juegos activos
        player_idx = state.current_player_idx[:, 0]
        batch_indices = jnp.arange(batch_size)
        
        logits = policy_logits[player_idx]
        legal = jax.vmap(get_legal_actions)(state)
        masked = jnp.where(legal, logits, -1e9)
        
        keys = jax.random.split(key, batch_size + 1)
        action_keys, next_key = keys[:-1], keys[-1]
        
        actions = jax.vmap(jax.random.categorical)(action_keys, masked)
        
        # 2c. Aplicar step a todos los juegos
        next_state = jax.vmap(step)(state, actions)
        
        # 2d. Mantener el estado anterior si el juego ya terminó
        final_state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(continue_mask[:, None], new, old) if old.ndim > 1 else jnp.where(continue_mask, new, old),
            state, next_state
        )

        # 2e. Actualizar historial y máscara de continuación
        histories = histories.at[batch_indices, i].set(jnp.where(continue_mask, actions, -1))
        
        num_active = jax.vmap(lambda s: jnp.sum(s.player_status == 0))(final_state)
        game_is_over = num_active <= 1
        new_continue_mask = continue_mask & ~game_is_over

        return final_state, new_continue_mask, histories, next_key

    # Inicializar el bucle
    histories = jnp.full((batch_size, MAX_GAME_LENGTH), -1, dtype=jnp.int32)
    continue_mask = jnp.ones(batch_size, dtype=jnp.bool_)
    
    # Ejecutar el bucle de juego
    final_states, _, final_histories, _ = jax.lax.fori_loop(0, MAX_GAME_LENGTH, game_loop_body, (initial_states, continue_mask, histories, key))
    
    # 3. Resolver todos los juegos
    payoffs = jax.vmap(resolve_showdown)(final_states)
    
    return final_states, payoffs, final_histories, initial_states

# --- Interfaz para el Trainer ---
# Esta función simplemente envuelve la función JIT principal.
def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    return batch_play_and_resolve(key, policy_logits, batch_size)