import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass, replace
from functools import partial
import numpy as np
from poker_bot.evaluator import HandEvaluator
from jax import ShapeDtypeStruct

# ------------- Estado y auxiliares -------------
@dataclass
class GameState:
    stacks: jax.Array
    bets: jax.Array
    player_status: jax.Array        # 0 = activo, 1 = fold
    hole_cards: jax.Array
    comm_cards: jax.Array
    cur_player: jax.Array           # escalar int8
    street: jax.Array               # 0..3
    pot: jax.Array
    deck: jax.Array
    deck_ptr: jax.Array
    acted_this_round: jax.Array     # contador de jugadores que han actuado
    key: jax.Array                  # RNG key
    action_hist: jax.Array          # (MAX_GAME_LENGTH,)
    hist_ptr: jax.Array             # escalar int32

MAX_GAME_LENGTH = 60

evaluator = HandEvaluator()
def evaluate_hand_wrapper(cards_device):
    cards_list = np.asarray(cards_device).tolist()
    valid_cards = [c for c in cards_list if c != -1]
    if len(valid_cards) < 5:
        return np.int32(9999)
    return np.int32(evaluator.evaluate_single(valid_cards))

# ------------- helpers JIT puros -------------
@jax.jit
def next_active_player(ps, start):
    idx = (start + jnp.arange(6, dtype=jnp.int8)) % 6
    mask = ps[idx] == 0
    return jnp.where(mask.any(), idx[jnp.argmax(mask)], start).astype(jnp.int8)

@jax.jit
def is_betting_done(status, bets, acted, prev_len):
    active = (status != 1)
    num_active = active.sum()
    max_bet = jnp.max(bets * active)
    all_called = (acted >= num_active) & (bets == max_bet).all()
    return (num_active <= 1) | all_called

@jax.jit
def get_legal_actions(state: GameState) -> jnp.ndarray:
    mask = jnp.zeros(3, dtype=jnp.bool_)
    p = state.cur_player[0]
    status = state.player_status[p]
    can_act = (status == 0)
    current_bet = state.bets[p]
    max_bet = jnp.max(state.bets)
    can_check = (current_bet == max_bet)
    to_call = max_bet - current_bet
    can_call = (to_call > 0) & (state.stacks[p] >= to_call)
    mask = mask.at[0].set(can_act)  # fold
    mask = mask.at[1].set(can_act & ((to_call == 0) | (state.stacks[p] >= to_call)))  # check/call
    mask = mask.at[2].set(can_act & (state.stacks[p] > 0) & (~can_check))  # bet/raise solo si no puede check
    return mask

# ------------- Acción detallada -------------
@jax.jit
def apply_action(state, action):
    p = state.cur_player[0]
    status = state.player_status[p]
    current_bet = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current_bet
    player_stack = state.stacks[p]

    def do_fold(s):
        new_ps = s.player_status.at[p].set(1)
        return s.replace(player_status=new_ps)
    def do_check_call(s):
        amount = jnp.where(to_call > 0, to_call, 0.)
        new_stack = s.stacks.at[p].add(-amount)
        new_bet = s.bets.at[p].add(amount)
        new_pot = s.pot + amount
        return s.replace(stacks=new_stack, bets=new_bet, pot=new_pot)
    def do_bet_raise(s):
        bet_size = jnp.minimum(20.0, s.stacks[p])
        new_stack = s.stacks.at[p].add(-bet_size)
        new_bet = s.bets.at[p].add(bet_size)
        new_pot = s.pot + bet_size
        return s.replace(stacks=new_stack, bets=new_bet, pot=new_pot)

    # Elige la acción
    state2 = lax.switch(
        jnp.clip(action, 0, 2),
        [do_fold, do_check_call, do_bet_raise],
        state
    )
    # Avanza el historial
    new_hist = state2.action_hist.at[state2.hist_ptr[0]].set(action)
    new_ptr = state2.hist_ptr + 1
    # Avanza el contador de actuados
    new_acted = state2.acted_this_round + 1
    return state2.replace(
        action_hist=new_hist,
        hist_ptr=new_ptr,
        acted_this_round=new_acted
    )

# ------------- Una ronda de apuestas (while_loop) -------------
def _betting_round_body(state):
    legal = get_legal_actions(state)
    logits = jnp.where(legal, 0.0, -1e9 * jnp.ones_like(legal))
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, logits)
    # Actualiza key
    state = state.replace(key=key)
    # Aplica la acción
    state = apply_action(state, action)
    # Siguiente jugador activo
    next_p = next_active_player(state.player_status, (state.cur_player[0] + 1) % 6)
    state = state.replace(cur_player=jnp.array([next_p], dtype=jnp.int8))
    return state

@jax.jit
def run_betting_round(init_state):
    def cond_fun(s):
        return ~is_betting_done(s.player_status, s.bets, s.acted_this_round, s.street)
    return lax.while_loop(cond_fun, _betting_round_body, init_state)

# ------------- Una calle (scan) -------------
#@jax.jit
def play_street(state, num_cards):
    def deal_cards(s):
        start = s.deck_ptr[0]
        cards = s.deck[start:start+num_cards]
        comm = s.comm_cards.at[start:start+num_cards].set(cards)
        return s.replace(
            comm_cards=comm,
            deck_ptr=s.deck_ptr + num_cards,
            acted_this_round=jnp.array([0], dtype=jnp.int8),
            cur_player=jnp.array([0], dtype=jnp.int8)
        )
    state = lax.cond(num_cards > 0, deal_cards, lambda s: s, state)
    state = run_betting_round(state)
    return state

# ------------- Showdown detallado -------------
def resolve_showdown(state: GameState) -> jnp.ndarray:
    active_mask = (state.player_status != 1)
    pot = state.pot
    bets = state.bets
    def single_winner_case():
        winner = jnp.argmax(active_mask)
        payoffs = -bets
        return payoffs.at[winner].add(pot)
    def showdown_case():
        def player_hand_eval(i):
            cards = jnp.concatenate([state.hole_cards[i], state.comm_cards])
            return jax.pure_callback(evaluate_hand_wrapper, ShapeDtypeStruct((), np.int32), cards)
        strengths = jnp.array([lax.cond(active_mask[i], lambda: player_hand_eval(i), lambda: 9999) for i in range(6)])
        best_strength = jnp.min(strengths)
        winners_mask = (strengths == best_strength) & active_mask
        win_share = pot / jnp.maximum(1, jnp.sum(winners_mask))
        return -bets + (winners_mask * win_share)
    can_showdown = (jnp.sum(state.comm_cards != -1) >= 5) & (jnp.sum(active_mask) > 1)
    return lax.cond(jnp.sum(active_mask) <= 1, single_winner_case, lambda: lax.cond(can_showdown, showdown_case, single_winner_case))

# ------------- Juego completo -------------
def play_one_game(key):
    deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
    key, subkey = jax.random.split(key)
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].add(-5.0).at[1].add(-10.0)
    state = GameState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=deck[:12].reshape((6, 2)),
        comm_cards=jnp.full((5,), -1, dtype=jnp.int8),
        cur_player=jnp.array([2], dtype=jnp.int8),
        street=jnp.array([0], dtype=jnp.int8),
        pot=jnp.array([15.0]),
        deck=deck,
        deck_ptr=jnp.array([12], dtype=jnp.int8),
        acted_this_round=jnp.array([0], dtype=jnp.int8),
        key=subkey,
        action_hist=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        hist_ptr=jnp.array([0], dtype=jnp.int32)
    )
    # Preflop ya hecho
    # Flop
    state = play_street(state, 3)
    # Turn
    state = play_street(state, 1)
    # River
    state = play_street(state, 1)
    # Showdown
    payoffs = resolve_showdown(state)
    return payoffs, state.action_hist

play_one_game_jit = jax.jit(play_one_game)
batch_play = jax.vmap(play_one_game_jit)

# --------------------------------------------------
# API que espera trainer.py
# --------------------------------------------------
@jax.jit
def initial_state_for_idx(idx: int) -> GameState:
    """
    Devuelve el estado inicial correspondiente al juego `idx`
    dentro del batch.  Como nuestro motor es stateless,
    generamos el mismo estado que produciría `play_one_game`
    pero sin ejecutar todo el juego.
    """
    # Simulamos la misma semilla que usaría `batch_play`
    master_key = jax.random.PRNGKey(0)
    key = jax.random.fold_in(master_key, idx)
    deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
    key, subkey = jax.random.split(key)
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,)).at[0].set(5.0).at[1].set(10.0)
    stacks = stacks.at[0].add(-5.0).at[1].add(-10.0)
    return GameState(
        stacks=stacks,
        bets=bets,
        player_status=jnp.zeros((6,), dtype=jnp.int8),
        hole_cards=deck[:12].reshape((6, 2)),
        comm_cards=jnp.full((5,), -1, dtype=jnp.int8),
        cur_player=jnp.array([2], dtype=jnp.int8),
        street=jnp.array([0], dtype=jnp.int8),
        pot=jnp.array([15.0]),
        deck=deck,
        deck_ptr=jnp.array([12], dtype=jnp.int8),
        acted_this_round=jnp.array([0], dtype=jnp.int8),
        key=subkey,
        action_hist=jnp.full((MAX_GAME_LENGTH,), -1, dtype=jnp.int32),
        hist_ptr=jnp.array([0], dtype=jnp.int32)
    )


def batch_play(keys):
    """
    Entrada:  keys con shape (batch_size,)
    Salida:   payoffs (batch, 6) , histories (batch, MAX_GAME_LENGTH)
    """
    return jax.vmap(play_one_game)(keys)