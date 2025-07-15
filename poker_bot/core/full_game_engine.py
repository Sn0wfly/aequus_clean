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

# --- Unidades de Trabajo Compiladas con JIT ---
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
        deck_pointer=jnp.array([12]), num_players_acted_this_round=jnp.array([0])
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
def step(state: GameState, action: int) -> GameState:
    player_idx = state.current_player_idx[0]

    def _update_turn(s: GameState) -> GameState:
        start_idx = s.current_player_idx[0]
        # Bucle fori_loop es seguro para JIT
        def body_fun(i, current_idx):
            next_idx = (start_idx + 1 + i) % 6
            return jax.lax.cond(s.player_status[next_idx] == 0, lambda: next_idx, lambda: current_idx)
        next_player = jax.lax.fori_loop(0, 6, body_fun, start_idx)
        return GameState(**{**s.__dict__, "current_player_idx": jnp.array([next_player], dtype=jnp.int8)})

    def do_fold(s):
        return GameState(**{**s.__dict__, "player_status": s.player_status.at[player_idx].set(1)})
    def do_check(s):
        return s
    def do_call(s):
        amount = jnp.max(s.bets) - s.bets[player_idx]
        return GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-amount), "bets": s.bets.at[player_idx].add(amount), "pot_size": s.pot_size + amount})
    def do_bet_raise(s):
        bet_size = 20.0
        return GameState(**{**s.__dict__, "stacks": s.stacks.at[player_idx].add(-bet_size), "bets": s.bets.at[player_idx].add(bet_size), "pot_size": s.pot_size + bet_size})
        
    state_after_action = jax.lax.switch(
        jnp.clip(action, 0, 3), [do_fold, do_check, do_call, do_bet_raise], state
    )
    return _update_turn(state_after_action)

@jax.jit
def _deal_community_cards(state: GameState, num_to_deal: int) -> GameState:
    start = state.deck_pointer[0]
    # Usamos dynamic_slice_in_dim que no requiere static_argnums en este contexto
    cards = jax.lax.dynamic_slice_in_dim(state.deck, start, num_to_deal)
    num_dealt = jnp.sum(state.community_cards != -1)
    new_community = jax.lax.dynamic_update_slice(state.community_cards, cards, (num_dealt,))
    return GameState(**{**state.__dict__, "community_cards": new_community, "deck_pointer": state.deck_pointer + num_to_deal})

# --- Orquestación Híbrida (Bucles de Python, sin JIT) ---
def run_betting_round(state: GameState, policy_logits: Array, key: Array):
    history_in_round = []
    for _ in range(30):
        num_active = jnp.sum(state.player_status == 0)
        if num_active <= 1: break
        
        player_idx = state.current_player_idx[0]
        all_acted = (state.num_players_acted_this_round[0] >= num_active)
        bets_settled = (state.bets[player_idx] == jnp.max(state.bets))
        
        # Corregir la condición de parada para la primera acción
        if all_acted and bets_settled: break

        logits = policy_logits[player_idx]
        legal_mask = get_legal_actions(state) # Llama a la función JIT
        masked_logits = jnp.where(legal_mask, logits, -1e9)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, masked_logits)
        
        history_in_round.append(int(action))
        state_after_action = step(state, action) # Llama a la función JIT
        
        is_aggressive = (action >= 3)
        num_acted = 1 if is_aggressive else state.num_players_acted_this_round[0] + 1
        state = GameState(**{**state_after_action.__dict__, "num_players_acted_this_round": jnp.array([num_acted])})
        
    final_state = GameState(**{**state.__dict__, "num_players_acted_this_round": jnp.array([0])})
    return final_state, history_in_round

def play_game(initial_state: GameState, policy_logits: Array, key: Array):
    state = initial_state
    full_history = []
    
    for street_idx, num_cards in enumerate([0, 3, 1, 1]):
        # Solo repartir post-flop
        if street_idx > 0:
            if jnp.sum(state.player_status != 1) > 1:
                state = _deal_community_cards(state, num_cards)
            else:
                break # Termina el juego si solo queda uno
        
        state = GameState(**{**state.__dict__, "street": jnp.array([street_idx])})
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
        win_share = pot / jnp.maximum(1, jnp.sum(winners_mask))
        return -state.bets + (winners_mask * win_share)
        
    can_showdown = (jnp.sum(state.community_cards != -1) >= 5) & (jnp.sum(active_mask) > 1)
    return jax.lax.cond(can_showdown, showdown_case, single_winner_case)

def batch_play_game(batch_size: int, policy_logits: Array, key: Array):
    keys = jax.random.split(key, batch_size)
    
    vmap_initial_state = jax.vmap(create_initial_state)
    initial_states = vmap_initial_state(keys)
    
    final_states, histories = [], []
    for i in range(batch_size):
        initial_state = jax.tree_util.tree_map(lambda x: x[i], initial_states)
        final_state, history = play_game(initial_state, policy_logits, keys[i])
        final_states.append(final_state)
        histories.append(history)

    stacked_states = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *final_states)
    stacked_histories = jnp.stack(histories)
    
    vmap_resolve_showdown = jax.vmap(resolve_showdown)
    payoffs = vmap_resolve_showdown(stacked_states)
    
    return stacked_states, payoffs, stacked_histories, initial_states