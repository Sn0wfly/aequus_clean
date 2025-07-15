# poker_bot/core/simulation.py

"""
🚀 Motor de Simulación de Póker Vectorizado

Contiene el núcleo computacional para simular partidas de Texas Hold'em
a máxima velocidad en la GPU usando JAX.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax

# ============================================================================
# FUNCIONES DE EVALUACIÓN DE MANOS (BAJO NIVEL)
# ============================================================================

@jax.jit
def _evaluate_straight_vectorized(ranks: jnp.ndarray) -> bool:
    """Detecta una escalera en un conjunto de rangos de cartas."""
    # `set` elimina duplicados, `bincount` cuenta ocurrencias de cada rango
    unique_ranks = jnp.bincount(ranks, length=14) > 0
    
    # Caso especial para la rueda (A-2-3-4-5)
    is_wheel = jnp.all(unique_ranks[jnp.array([0, 1, 2, 3, 12])])
    
    # Busca 5 rangos consecutivos usando una convolución
    kernel = jnp.ones(5, dtype=jnp.int32)
    consecutive_counts = jnp.convolve(unique_ranks[:13], kernel, mode='valid')
    
    return jnp.any(consecutive_counts >= 5) | is_wheel

@jax.jit
def _evaluate_hand_strength(cards: jnp.ndarray) -> jnp.ndarray:
    """Evalúa la fuerza de una mano de 7 cartas, devolviendo un score numérico."""
    suits = cards // 13
    ranks = cards % 13

    # Detección de color y escalera
    is_flush = jnp.bincount(suits, length=4).max() >= 5
    is_straight = _evaluate_straight_vectorized(ranks)
    
    # Contar pares, tríos, etc.
    rank_counts = jnp.bincount(ranks, length=13)
    counts = jnp.bincount(rank_counts, length=5)
    
    # La puntuación se basa en la rareza de la mano (mayor es mejor) y un kicker
    # para desempates. El kicker se calcula a partir de los rangos ordenados.
    sorted_ranks = jnp.sort(rank_counts)[::-1]
    kicker_score = jnp.dot(sorted_ranks, 13**jnp.arange(13, 0, -1)).astype(jnp.float32)

    # Asignar puntuación basada en el tipo de mano
    score = kicker_score.astype(jnp.float32)
    score = jax.lax.cond(counts[2] >= 1, lambda: 1e10 + kicker_score, lambda: score.astype(jnp.float32)) # Par
    score = jax.lax.cond(counts[2] >= 2, lambda: 2e10 + kicker_score, lambda: score.astype(jnp.float32)) # Dos Pares
    score = jax.lax.cond(counts[3] >= 1, lambda: 3e10 + kicker_score, lambda: score.astype(jnp.float32)) # Trío
    score = jax.lax.cond(is_straight,    lambda: 4e10 + jnp.max(ranks).astype(jnp.float32), lambda: score.astype(jnp.float32)) # Escalera
    score = jax.lax.cond(is_flush,       lambda: 5e10 + kicker_score, lambda: score.astype(jnp.float32)) # Color
    score = jax.lax.cond((counts[3] >= 1) & (counts[2] >= 2), lambda: 6e10 + kicker_score, lambda: score.astype(jnp.float32)) # Full
    score = jax.lax.cond(counts[4] >= 1, lambda: 7e10 + kicker_score, lambda: score.astype(jnp.float32)) # Póker
    score = jax.lax.cond(is_straight & is_flush, lambda: 8e10 + jnp.max(ranks).astype(jnp.float32), lambda: score.astype(jnp.float32)) # Escalera de Color

    return score

# ============================================================================
# SIMULACIÓN DE JUEGO (ALTO NIVEL)
# ============================================================================

def _simulate_single_game_vectorized(rng_key: jnp.ndarray, game_config: dict) -> dict:
    """Simula una ÚNICA partida de Hold'em. Esta es la función que se vectorizará."""
    MAX_PLAYERS = 6
    players = game_config['players']
    starting_stack = game_config['starting_stack']
    small_blind = game_config['small_blind']
    big_blind = game_config['big_blind']
    
    # --- Estado del Juego ---
    stacks = jnp.full(MAX_PLAYERS, starting_stack)
    bets = jnp.zeros(MAX_PLAYERS)
    is_folded = jnp.zeros(MAX_PLAYERS, dtype=bool)
    is_all_in = jnp.zeros(MAX_PLAYERS, dtype=bool)
    
    # --- Reparto ---
    rng_key, deck_key = jax.random.split(rng_key)
    deck = jax.random.permutation(deck_key, jnp.arange(52))
    
    num_hole_cards = MAX_PLAYERS * 2
    hole_cards = lax.dynamic_slice(deck, (0,), (num_hole_cards,))
    hole_cards = hole_cards.reshape((MAX_PLAYERS, 2))
    
    community_start_index = num_hole_cards
    community_cards_full = lax.dynamic_slice(deck, (community_start_index,), (5,))
    
    # Solo usamos los primeros 'players' en la lógica de juego
    # ... el resto de la función debe usar hole_cards[:players] y stacks[:players] según corresponda ...
    
    # --- Rondas de Apuestas (simplificado para JIT) ---
    # Esto es una abstracción. Un motor de juego real tendría un bucle complejo.
    # Para nuestro propósito (generar datos para CFVFP), una simulación de resultados
    # basada en la fuerza de la mano es suficiente y muchísimo más rápida.
    
    # Simulamos el resultado final directamente, que es lo que nuestro trainer necesita.
    
    # --- Showdown ---
    def evaluate_player_hand(player_idx):
        # Combina las cartas del jugador con las comunitarias
        player_cards = jnp.concatenate((hole_cards[player_idx], community_cards_full))
        return _evaluate_hand_strength(player_cards)

    # Evalúa las manos de todos los jugadores
    hand_strengths = jax.vmap(evaluate_player_hand)(jnp.arange(MAX_PLAYERS))
    # Solo usar hand_strengths[:players] en la lógica posterior
    
    # Determina el ganador
    winner_idx = jnp.argmax(hand_strengths)
    
    # --- Payoffs (simplificado) ---
    # Asumimos un bote final basado en una acción promedio.
    final_pot = big_blind * 20 
    payoffs = jnp.zeros(MAX_PLAYERS)
    payoffs = payoffs.at[winner_idx].set(final_pot)

    # Salida compatible con JAX: hole_cards de tamaño fijo
    # Eliminamos el uso de slicing dinámico y usamos asignación directa
    hole_cards_out = jnp.full((MAX_PLAYERS, 2), -1, dtype=hole_cards.dtype)
    mask = jnp.arange(MAX_PLAYERS) < players  # [True, ..., False]
    # Asignamos solo los jugadores activos
    hole_cards_out = jnp.where(mask[:, None], hole_cards, hole_cards_out)

    # Un resultado simplificado para el trainer, que es lo que importa
    return {
        'payoffs': payoffs,
        'final_community': community_cards_full,
        'hole_cards': hole_cards_out,
        'final_pot': final_pot,
    }

# --- FUNCIÓN PÚBLICA PRINCIPAL ---
@jax.jit
def batch_simulate_real_holdem(rng_keys: jnp.ndarray, game_config: dict) -> dict:
    """
    Vectoriza la simulación y asegura la ejecución en GPU.
    """
    # FORZAR ENTRADAS A GPU
    rng_keys_gpu = jax.device_put(rng_keys)  # Mueve las claves a la GPU por defecto
    return jax.vmap(_simulate_single_game_vectorized, in_axes=(0, None))(rng_keys_gpu, game_config)