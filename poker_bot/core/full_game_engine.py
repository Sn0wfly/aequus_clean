# full_game_engine.py
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass, replace
import numpy as np
from jax.tree_util import register_pytree_node_class
from jax import ShapeDtypeStruct

from poker_bot.evaluator import HandEvaluator

MAX_GAME_LENGTH = 60
evaluator = HandEvaluator()

# ---------- Pytree dataclass ----------
@register_pytree_node_class
@dataclass
class GameState:
    stacks: jax.Array
    bets: jax.Array
    player_status: jax.Array
    hole_cards: jax.Array
    comm_cards: jax.Array
    cur_player: jax.Array
    street: jax.Array
    pot: jax.Array
    deck: jax.Array
    deck_ptr: jax.Array
    acted_this_round: jax.Array
    key: jax.Array
    action_hist: jax.Array
    hist_ptr: jax.Array

    def tree_flatten(self):
        children = (self.stacks, self.bets, self.player_status, self.hole_cards,
                    self.comm_cards, self.cur_player, self.street, self.pot,
                    self.deck, self.deck_ptr, self.acted_this_round,
                    self.key, self.action_hist, self.hist_ptr)
        return children, None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

# ---------- Callback ----------
def evaluate_hand_wrapper(cards_device):
    cards_list = np.asarray(cards_device).tolist()
    valid_cards = [c for c in cards_list if c != -1]
    return np.int32(evaluator.evaluate_single(valid_cards)) if len(valid_cards) >= 5 else np.int32(9999)

# ---------- Helpers ----------
@jax.jit
def next_active_player(ps, start):
    idx = (start + jnp.arange(6, dtype=jnp.int8)) % 6
    mask = ps[idx] == 0
    return jnp.where(mask.any(), idx[jnp.argmax(mask)], start).astype(jnp.int8)

@jax.jit
def is_betting_done(status, bets, acted, _):
    active = status != 1
    num_active = active.sum()
    max_bet = jnp.max(bets * active)
    all_called = (acted >= num_active) & (bets == max_bet).all()
    return (num_active <= 1) | all_called

@jax.jit
def get_legal_actions(state: GameState):
    mask = jnp.zeros(3, dtype=jnp.bool_)
    p = state.cur_player[0]
    status = state.player_status[p]
    can_act = status == 0
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current
    mask = mask.at[0].set(can_act)  # fold
    mask = mask.at[1].set(can_act & ((to_call == 0) | (state.stacks[p] >= to_call)))  # check/call
    mask = mask.at[2].set(can_act & (state.stacks[p] > 0) & (current != max_bet))  # bet/raise
    return mask

# ---------- Step ----------
@jax.jit
def apply_action(state, action):
    p = state.cur_player[0]
    current = state.bets[p]
    max_bet = jnp.max(state.bets)
    to_call = max_bet - current

    def do_fold(s):
        return s.replace(player_status=s.player_status.at[p].set(1))

    def do_check_call(s):
        amt = jnp.where(to_call > 0, to_call, 0.0)
        return s.replace(
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    def do_bet_raise(s):
        amt = jnp.minimum(20.0, s.stacks[p])
        return s.replace(
            stacks=s.stacks.at[p].add(-amt),
            bets=s.bets.at[p].add(amt),
            pot=s.pot + amt
        )

    state2 = lax.switch(jnp.clip(action, 0, 2), [do_fold, do_check_call, do_bet_raise], state)
    new_hist = state2.action_hist.at[state2.hist_ptr[0]].set(action)
    return state2.replace(
        action_hist=new_hist,
        hist_ptr=state2.hist_ptr + 1,
        acted_this_round=state2.acted_this_round + 1
    )

# ---------- Betting round ----------
def _betting_body(state):
    legal = get_legal_actions(state)
    key, subkey = jax.random.split(state.key)
    action = jax.random.categorical(subkey, jnp.where(legal, 0.0, -1e9))
    state = state.replace(key=key)
    state = apply_action(state, action)
    next_p = next_active_player(state.player_status, (state.cur_player[0] + 1) % 6)
    return state.replace(cur_player=jnp.array([next_p], dtype=jnp.int8))

@jax.jit
def run_betting_round(init_state):
    cond = lambda s: ~is_betting_done(s.player_status, s.bets, s.acted_this_round, s.street)
    return lax.while_loop(cond, _betting_body, init_state)

# ---------- Street ----------
def play_street(state, num_cards):
    def deal(s):
        start = s.deck_ptr[0]
        # Extrae exactamente `num_cards` cartas a partir de `start`
        cards = lax.dynamic_slice(s.deck, (start,), (num_cards,))
        # Inserta las cartas extraídas en `comm_cards` a partir de `start`
        comm = lax.dynamic_update_slice(s.comm_cards, cards, (start,))
        return s.replace(
            comm_cards=comm,
            deck_ptr=s.deck_ptr + num_cards,
            acted_this_round=jnp.array([0], dtype=jnp.int8),
            cur_player=jnp.array([0], dtype=jnp.int8)
        )
    state = lax.cond(num_cards > 0, deal, lambda x: x, state)
    return run_betting_round(state)

# ---------- Showdown ----------
def resolve_showdown(state: GameState) -> jax.Array:
    active = state.player_status != 1
    pot = state.pot

    def single():
        winner = jnp.argmax(active)
        return -state.bets.at[winner].add(pot)

    def full():
        def eval_i(i):
            cards = jnp.concatenate([state.hole_cards[i], state.comm_cards])
            return jax.pure_callback(evaluate_hand_wrapper, ShapeDtypeStruct((), np.int32), cards)
        strengths = jnp.array([lax.cond(active[i], lambda: eval_i(i), lambda: 9999) for i in range(6)])
        best = jnp.min(strengths)
        winners = (strengths == best) & active
        share = pot / jnp.maximum(1, winners.sum())
        return -state.bets + winners * share

    can_show = (state.comm_cards != -1).sum() >= 5
    return lax.cond(active.sum() <= 1, single, lambda: lax.cond(can_show, full, single))

# ---------- Single game ----------
@jax.jit
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

    state = play_street(state, 3)  # flop
    state = play_street(state, 1)  # turn
    state = play_street(state, 1)  # river
    payoffs = resolve_showdown(state)
    return payoffs, state.action_hist

# ---------- Batch API ----------
@jax.jit
def initial_state_for_idx(idx: int) -> GameState:
    key = jax.random.fold_in(jax.random.PRNGKey(0), idx)
    return play_one_game(key)[0]  # devuelve solo el estado inicial

def batch_play(keys):
    return jax.vmap(play_one_game)(keys)