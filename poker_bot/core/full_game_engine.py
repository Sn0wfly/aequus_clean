import jax
import jax.numpy as jnp
from jax import Array
from dataclasses import dataclass
from typing import Optional
from poker_bot.evaluator import evaluate_hand

@dataclass
class GameState:
    stacks: Array  # (6,)
    bets: Array  # (6,)
    player_status: Array  # (6,) int8: 0=Activo, 1=Folded, 2=All-in
    hole_cards: Array  # (6, 2)
    community_cards: Array  # (5,)
    current_player_idx: Array  # (1,)
    street: Array  # (1,) 0=Preflop, 1=Flop, 2=Turn, 3=River
    pot_size: Array  # (1,)
    deck: Array  # (52,)
    deck_pointer: Array  # (1,)

@jax.jit
def get_legal_actions(state: GameState, num_actions: int = 14) -> jnp.ndarray:
    """
    Devuelve un array booleano de tamaño num_actions donde True indica que la acción es legal.
    Índices:
      0: FOLD
      1: CHECK
      2: CALL
      3-13: BET/RAISE (simplificado: legales solo si no se puede hacer CHECK)
    """
    legal_actions_mask = jnp.zeros(num_actions, dtype=jnp.bool_)

    player_idx = state.current_player_idx[0]
    current_bet = state.bets[player_idx]
    max_bet_on_table = jnp.max(state.bets)
    player_stack = state.stacks[player_idx]
    player_status = state.player_status[player_idx]

    # FOLD: True si el jugador está activo (player_status != 1)
    can_fold = (player_status == 0)
    legal_actions_mask = legal_actions_mask.at[0].set(can_fold)

    # CHECK: True si la apuesta del jugador es igual a la máxima
    can_check = (current_bet == max_bet_on_table)
    legal_actions_mask = legal_actions_mask.at[1].set(can_check)

    # CALL: True si el jugador puede igualar la apuesta máxima y tiene fichas suficientes
    to_call = max_bet_on_table - current_bet
    can_call = (to_call > 0) & (player_stack >= to_call) & (player_status == 0)
    legal_actions_mask = legal_actions_mask.at[2].set(can_call)

    # BET/RAISE: índices 3 a 13, legales solo si no se puede hacer CHECK
    def set_bet_raise_true(mask):
        # Solo se pueden apostar si el jugador está activo
        return mask.at[3:14].set(player_status == 0)
    def set_bet_raise_false(mask):
        return mask
    legal_actions_mask = jax.lax.cond(~can_check, set_bet_raise_true, set_bet_raise_false, legal_actions_mask)

    return legal_actions_mask

@jax.jit
def _update_turn(state: GameState) -> GameState:
    """
    Avanza el turno al siguiente jugador activo (player_status == 0).
    """
    def cond_fun(carry):
        idx, found = carry
        return ~found

    def body_fun(carry):
        idx, _ = carry
        idx = (idx + 1) % 6
        found = (state.player_status[idx] == 0)
        return (idx, found)

    start_idx = (state.current_player_idx[0] + 1) % 6
    init_found = (state.player_status[start_idx] == 0)
    idx, _ = jax.lax.while_loop(cond_fun, body_fun, (start_idx, init_found))
    # Devuelve un nuevo estado con el current_player_idx actualizado
    return GameState(
        stacks=state.stacks,
        bets=state.bets,
        player_status=state.player_status,
        hole_cards=state.hole_cards,
        community_cards=state.community_cards,
        current_player_idx=jnp.array([idx]),
        street=state.street,
        pot_size=state.pot_size
    )

@jax.jit
def step(state: GameState, action: int) -> GameState:
    """
    Aplica la acción al estado y devuelve un nuevo GameState actualizado.
    """
    player_idx = state.current_player_idx[0]

    def do_fold(state):
        # Actualiza player_status[player_idx] a 1 (Folded)
        new_player_status = state.player_status.at[player_idx].set(1)
        new_state = GameState(
            stacks=state.stacks,
            bets=state.bets,
            player_status=new_player_status,
            hole_cards=state.hole_cards,
            community_cards=state.community_cards,
            current_player_idx=state.current_player_idx,
            street=state.street,
            pot_size=state.pot_size
        )
        return _update_turn(new_state)

    def do_check(state):
        # No cambios, solo avanza el turno
        return _update_turn(state)

    def do_call(state):
        amount_to_call = jnp.max(state.bets) - state.bets[player_idx]
        new_stack = state.stacks.at[player_idx].add(-amount_to_call)
        new_bet = state.bets.at[player_idx].add(amount_to_call)
        new_pot = state.pot_size.at[0].add(amount_to_call)
        new_state = GameState(
            stacks=new_stack,
            bets=new_bet,
            player_status=state.player_status,
            hole_cards=state.hole_cards,
            community_cards=state.community_cards,
            current_player_idx=state.current_player_idx,
            street=state.street,
            pot_size=new_pot
        )
        return _update_turn(new_state)

    def do_bet_raise(state):
        bet_size = 20.0
        new_stack = state.stacks.at[player_idx].add(-bet_size)
        new_bet = state.bets.at[player_idx].add(bet_size)
        new_pot = state.pot_size.at[0].add(bet_size)
        new_state = GameState(
            stacks=new_stack,
            bets=new_bet,
            player_status=state.player_status,
            hole_cards=state.hole_cards,
            community_cards=state.community_cards,
            current_player_idx=state.current_player_idx,
            street=state.street,
            pot_size=new_pot
        )
        return _update_turn(new_state)

    # jax.lax.switch para seleccionar la acción
    def action_fn(idx):
        return jax.lax.switch(
            idx,
            [lambda _: do_fold(state),
             lambda _: do_check(state),
             lambda _: do_call(state)] +
            [lambda _: do_bet_raise(state)] * 11,
            None
        )

    # action: 0=FOLD, 1=CHECK, 2=CALL, 3-13=BET/RAISE
    return action_fn(jnp.clip(action, 0, 13))

@jax.jit
def run_betting_round(initial_state: GameState, policy_logits: jnp.ndarray, key: jax.random.KeyArray = None) -> GameState:
    """
    Simula una ronda de apuestas completa usando la política dada (logits) para cada jugador.
    Args:
        initial_state: Estado inicial de la ronda.
        policy_logits: (6, 14) logits de política para cada jugador.
        key: PRNGKey de JAX para muestreo estocástico.
    Returns:
        GameState final tras la ronda de apuestas.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    max_steps = 30

    def _betting_step(state_and_key, _):
        state, key = state_and_key
        player_idx = state.current_player_idx[0]
        logits = policy_logits[player_idx]
        legal_mask = get_legal_actions(state)
        masked_logits = jnp.where(legal_mask, logits, -1e9)
        key, subkey = jax.random.split(key)
        action = jax.random.categorical(subkey, masked_logits)
        new_state = step(state, action)
        return (new_state, key), new_state

    (final_state, _), states = jax.lax.scan(
        _betting_step,
        (initial_state, key),
        xs=None,
        length=max_steps
    )
    return final_state

@jax.jit
def create_initial_state(key: jax.random.KeyArray) -> GameState:
    # Baraja la baraja
    deck = jnp.arange(52)
    key, subkey = jax.random.split(key)
    shuffled_deck = jax.random.permutation(subkey, deck)
    # Reparte las cartas de mano a los 6 jugadores (2 cada uno)
    hole_cards = shuffled_deck[:12].reshape((6, 2))
    # Inicializa stacks, apuestas, etc.
    stacks = jnp.full((6,), 1000.0)
    bets = jnp.zeros((6,))
    player_status = jnp.zeros((6,), dtype=jnp.int8)
    community_cards = jnp.full((5,), -1)  # -1 indica carta no repartida
    current_player_idx = jnp.array([0])
    street = jnp.array([0])
    pot_size = jnp.array([0.0])
    deck_pointer = jnp.array([12])
    return GameState(
        stacks=stacks,
        bets=bets,
        player_status=player_status,
        hole_cards=hole_cards,
        community_cards=community_cards,
        current_player_idx=current_player_idx,
        street=street,
        pot_size=pot_size,
        deck=shuffled_deck,
        deck_pointer=deck_pointer
    )

@jax.jit
def _deal_community_cards(state: GameState, num_cards_to_deal: int) -> GameState:
    # Reparte num_cards_to_deal cartas del deck a las posiciones libres de community_cards
    start = state.deck_pointer[0]
    end = start + num_cards_to_deal
    cards = state.deck[start:end]
    # Encuentra las posiciones libres (-1) en community_cards
    mask = (state.community_cards == -1)
    idxs = jnp.where(mask, size=num_cards_to_deal, fill_value=0)[0]
    new_community = state.community_cards.at[idxs[:num_cards_to_deal]].set(cards)
    new_deck_pointer = state.deck_pointer + num_cards_to_deal
    return GameState(
        stacks=state.stacks,
        bets=state.bets,
        player_status=state.player_status,
        hole_cards=state.hole_cards,
        community_cards=new_community,
        current_player_idx=state.current_player_idx,
        street=state.street,
        pot_size=state.pot_size,
        deck=state.deck,
        deck_pointer=new_deck_pointer
    )

@jax.jit
def play_game(initial_state: GameState, policy_logits: jnp.ndarray, key: jax.random.KeyArray) -> GameState:
    """
    Simula una mano completa de póker (hasta 4 rondas de apuestas).
    Args:
        initial_state: Estado inicial del juego.
        policy_logits: (6, 14) logits de política para cada jugador.
        key: PRNGKey de JAX para muestreo estocástico.
    Returns:
        GameState final tras la mano.
    """
    def body_fun(street_idx, state_and_key):
        state, key = state_and_key
        # Reparto real de cartas comunitarias
        def deal_cards(state):
            return jax.lax.switch(
                street_idx,
                [lambda s: s,  # Preflop: no se reparten comunitarias
                 lambda s: _deal_community_cards(s, 3),  # Flop
                 lambda s: _deal_community_cards(s, 1),  # Turn
                 lambda s: _deal_community_cards(s, 1)],  # River
                state
            )
        state = deal_cards(state)
        # Ejecuta la ronda de apuestas
        key, subkey = jax.random.split(key)
        state_after_round = run_betting_round(state, policy_logits, subkey)
        # Comprueba si la mano ha terminado (solo un jugador activo)
        num_active = jnp.sum(state_after_round.player_status != 1)
        def early_exit(_):
            return (state_after_round, key)
        def continue_game(_):
            return (state_after_round, key)
        state_and_key = jax.lax.cond(num_active == 1, early_exit, continue_game, operand=None)
        return state_and_key

    final_state, _ = jax.lax.fori_loop(
        0, 4, body_fun, (initial_state, key)
    )
    return final_state 

@jax.jit
def resolve_showdown(state: GameState) -> jnp.ndarray:
    """
    Calcula los payoffs finales para cada jugador al terminar la mano.
    Devuelve un array (6,) con las ganancias/pérdidas de cada jugador.
    """
    # 1. ¿Solo queda un jugador activo?
    active_mask = (state.player_status != 1)
    num_active = jnp.sum(active_mask)
    pot = state.pot_size[0]
    bets = state.bets

    def single_winner_case(_):
        winner = jnp.argmax(active_mask)
        payoffs = -bets
        payoffs = payoffs.at[winner].set(pot - bets[winner])
        return payoffs

    def showdown_case(_):
        # Solo jugadores activos
        def player_hand_eval(i):
            return evaluate_hand(state.hole_cards[i], state.community_cards)
        idxs = jnp.arange(6)
        hand_strengths = jax.vmap(lambda i: jax.lax.cond(active_mask[i], player_hand_eval, lambda _: -1, i))(idxs)
        max_strength = jnp.max(hand_strengths)
        winners_mask = (hand_strengths == max_strength) & active_mask
        num_winners = jnp.sum(winners_mask)
        win_share = pot / num_winners
        payoffs = -bets + (winners_mask.astype(jnp.float32) * win_share)
        return payoffs

    payoffs = jax.lax.cond(num_active == 1, single_winner_case, showdown_case, operand=None)
    return payoffs 

# Vectoriza la creación de estados
batch_create_initial_state = jax.vmap(create_initial_state)

# Vectoriza la simulación del juego
batch_play_game_vmapped = jax.vmap(play_game, in_axes=(0, None, 0))

# Vectoriza la resolución final
def _resolve_showdown_vmapped(state_batch):
    return jax.vmap(resolve_showdown)(state_batch)
batch_resolve_showdown = _resolve_showdown_vmapped

def batch_play_game(batch_size: int, policy_logits: jnp.ndarray, key: jax.random.KeyArray):
    keys = jax.random.split(key, batch_size)
    initial_states = batch_create_initial_state(keys)
    final_states = batch_play_game_vmapped(initial_states, policy_logits, keys)
    payoffs = batch_resolve_showdown(final_states)
    return final_states, payoffs 