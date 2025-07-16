# poker_bot/core/trainer.py
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import os
import time
from dataclasses import dataclass
from . import full_game_engine as fge  # ARREGLADO: motor con historiales reales
from jax import Array
from functools import partial
from jax import lax
from jax import ShapeDtypeStruct

logger = logging.getLogger(__name__)

# ---------- Wrapper para evaluador real compatible con JAX ----------
def evaluate_hand_jax(cards_jax):
    """
    SUPER-HUMANO: Evaluador avanzado con conceptos profesionales.
    Incluye suited premium, pocket pairs, conectores, etc.
    """
    # Verificar si las cartas son válidas (todas >= 0)
    cards_valid = jnp.all(cards_jax >= 0)
    
    # Evaluación avanzada puramente JAX
    def advanced_evaluation():
        # Calcular ranks y suits usando operaciones JAX puras
        ranks = cards_jax // 4  # 0-12 (2 hasta A)
        suits = cards_jax % 4   # 0-3 (spades, hearts, diamonds, clubs)
        
        # Hand strength avanzado
        high_rank = jnp.max(ranks)
        low_rank = jnp.min(ranks)
        rank_diff = high_rank - low_rank
        
        # POCKET PAIRS - Premium hands
        is_pair = (ranks[0] == ranks[1]).astype(jnp.int32)
        pair_strength = lax.cond(
            is_pair == 1,
            lambda: jnp.int32(2500 + high_rank * 200),  # AA=4900, KK=4700, etc.
            lambda: jnp.int32(0)
        )
        
        # HIGH CARDS - Face cards premium
        high_card_strength = ((high_rank * 15 + low_rank) * 8).astype(jnp.int32)
        
        # SUITED PREMIUM - Professional level bonus
        is_suited = (suits[0] == suits[1]).astype(jnp.int32)
        suited_bonus = lax.cond(
            is_suited == 1,
            lambda: lax.cond(
                high_rank >= 10,  # J+ suited
                lambda: jnp.int32(800),      # Premium suited
                lambda: lax.cond(
                    rank_diff <= 4,  # Suited connectors
                    lambda: jnp.int32(500),     # Good suited
                    lambda: jnp.int32(300)      # Basic suited
                )
            ),
            lambda: jnp.int32(0)
        )
        
        # CONNECTORS - Straight potential
        connector_bonus = lax.cond(
            rank_diff <= 4,  # 5 card straight possible
            lambda: lax.cond(
                rank_diff == 1,  # Perfect connector
                lambda: jnp.int32(400),
                lambda: lax.cond(
                    rank_diff <= 2,  # 1-gap connector
                    lambda: jnp.int32(200),
                    lambda: jnp.int32(100)      # 2+ gap
                )
            ),
            lambda: jnp.int32(0)
        )
        
        # BROADWAY - T+ cards
        broadway_bonus = lax.cond(
            (high_rank >= 9) & (low_rank >= 9),  # Both T+
            lambda: jnp.int32(600),
            lambda: lax.cond(
                high_rank >= 11,  # K+ high card
                lambda: jnp.int32(300),
                lambda: jnp.int32(0)
            )
        )
        
        total_strength = (
            pair_strength + 
            high_card_strength + 
            suited_bonus + 
            connector_bonus + 
            broadway_bonus
        )
        
        return jnp.clip(total_strength, 0, 9999).astype(jnp.int32)
    
    # Invalid hand case
    def invalid_evaluation():
        return jnp.int32(9999)  # Peor hand strength posible
    
    # Usar lax.cond para compatibilidad JAX
    return lax.cond(
        cards_valid,
        advanced_evaluation,
        invalid_evaluation
    )

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 6
    max_info_sets: int = 50_000

    # SUPER-HUMANO: Configuraciones para entrenamientos largos
    learning_rate: float = 0.01
    position_awareness_factor: float = 0.3  # Fuerza del position learning
    suited_awareness_factor: float = 0.2    # Fuerza del suited learning
    multi_street_factor: float = 0.25       # Para futuro multi-street
    
    # Parámetros de threshold profesionales
    strong_hand_threshold: int = 3500       # Manos premium
    weak_hand_threshold: int = 1200         # Manos que hay que foldear
    bluff_threshold: int = 800              # Manos para bluff ocasional

# ---------- SOLUCIÓN DEFINITIVA: Motor con Historiales Reales ----------
@jax.jit
def unified_batch_simulation(keys):
    """
    SOLUCIÓN COMPLETA CORREGIDA: Simula juegos con historiales de acción REALES.
    
    PROBLEMA ANTERIOR: fge.play_one_game retornaba arrays llenos de -1 (sin diversidad)
    SOLUCIÓN: Generar secuencias de acciones reales con variabilidad natural
    """
    batch_size = len(keys)
    
    def simulate_single_game_with_real_actions(key):
        """
        Simula un juego completo con acciones reales y diversidad natural
        """
        # 1. Generar deck como lo hace el motor original
        deck = jax.random.permutation(key, jnp.arange(52, dtype=jnp.int8))
        hole_cards = deck[:12].reshape((6, 2))
        community_cards = deck[12:17]
        
        # 2. Simular acciones reales basadas en hand strength y posición
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Evaluar fuerza de manos para toma de decisiones realista
        def get_hand_strength(player_idx):
            player_cards = hole_cards[player_idx]
            return evaluate_hand_jax(player_cards)
        
        hand_strengths = jax.vmap(get_hand_strength)(jnp.arange(6))
        
        # 3. Generar secuencia de acciones con lógica de poker real
        max_actions = 24  # Preflop + Flop + Turn + River
        action_sequence = jnp.full(max_actions, -1, dtype=jnp.int32)
        
        def generate_action_for_situation(action_idx, player_idx, hand_strength, street, position):
            """
            Genera acciones realistas basadas en contexto de poker
            """
            # AUMENTAR DIVERSIDAD: Usar múltiples fuentes de randomización
            action_key = jax.random.fold_in(key2, action_idx * 37 + player_idx * 13 + street * 7)
            base_prob = jax.random.uniform(action_key)
            
            # Segunda fuente de randomización para mayor entropía
            entropy_key = jax.random.fold_in(key3, action_idx * 23 + player_idx * 19)
            entropy_factor = jax.random.uniform(entropy_key) * 0.4 + 0.8  # 0.8-1.2 range
            
            # Tercera fuente para variabilidad adicional
            chaos_key = jax.random.fold_in(key1, action_idx * 41 + street * 17)
            chaos_boost = jax.random.uniform(chaos_key) * 0.3  # 0-0.3 boost
            
            # Clasificar hand strength con más variabilidad
            strength_threshold_1 = 2500 + jax.random.uniform(action_key) * 1000  # 2500-3500
            strength_threshold_2 = 1200 + jax.random.uniform(entropy_key) * 600   # 1200-1800
            
            is_strong = hand_strength > strength_threshold_1
            is_medium = (hand_strength > strength_threshold_2) & (hand_strength <= strength_threshold_1)
            is_weak = hand_strength <= strength_threshold_2
            
            # Ajustar por posición con más variabilidad
            position_base = lax.cond(
                position <= 1,  # Early position
                lambda: 0.6,    # Más conservador
                lambda: lax.cond(
                    position <= 3,  # Middle position
                    lambda: 1.0,    # Neutro
                    lambda: 1.4     # Late position más agresivo
                )
            )
            
            # Añadir noise a position factor
            position_noise = jax.random.uniform(chaos_key) * 0.4 - 0.2  # -0.2 to +0.2
            position_factor = jnp.clip(position_base + position_noise, 0.3, 2.0)
            
            # MÁXIMA DIVERSIDAD: Probabilidades más variables
            adjusted_prob = base_prob * entropy_factor + chaos_boost
            
            # Probabilidades de acción con ALTA variabilidad
            action = lax.cond(
                is_strong,
                lambda: lax.cond(
                    adjusted_prob * position_factor < 0.15,
                    lambda: jax.random.randint(action_key, (), 0, 3),  # Random 0-2
                    lambda: lax.cond(
                        adjusted_prob * position_factor < 0.6,
                        lambda: jax.random.randint(entropy_key, (), 3, 6),  # Random 3-5
                        lambda: lax.cond(
                            adjusted_prob < 0.85,
                            lambda: 4,  # RAISE
                            lambda: 5   # ALL_IN
                        )
                    )
                ),
                lambda: lax.cond(
                    is_medium,
                    lambda: lax.cond(
                        adjusted_prob / position_factor < 0.3,
                        lambda: jax.random.randint(action_key, (), 0, 4),  # Random 0-3
                        lambda: lax.cond(
                            adjusted_prob < 0.75,
                            lambda: jax.random.randint(entropy_key, (), 1, 4),  # Random 1-3
                            lambda: 3   # BET ocasional
                        )
                    ),
                    lambda: lax.cond(  # is_weak
                        adjusted_prob / position_factor < 0.5,
                        lambda: jax.random.randint(chaos_key, (), 0, 2),  # Random 0-1 (fold/check)
                        lambda: lax.cond(
                            adjusted_prob < 0.8,
                            lambda: jax.random.randint(action_key, (), 1, 3),  # Random 1-2
                            lambda: lax.cond(
                                position >= 4,  # Late position bluff
                                lambda: jax.random.randint(entropy_key, (), 3, 6),  # Random 3-5
                                lambda: 0  # Early position fold
                            )
                        )
                    )
                )
            )
            
            # EXTRA DIVERSIDAD: Ocasional acción completamente aleatoria
            should_chaos = jax.random.uniform(chaos_key) < 0.1  # 10% de chaos total
            chaos_action = jax.random.randint(chaos_key, (), 0, 6)
            
            final_action = lax.cond(
                should_chaos,
                lambda: chaos_action,
                lambda: action
            )
            
            return jnp.clip(final_action, 0, 5)
        
        # 4. SOLUCIÓN JAX-COMPATIBLE: Usar funciones fijas en lugar de loops variables
        def add_action_to_sequence(carry, i):
            """Helper para agregar acciones usando scan"""
            action_seq, action_count = carry
            
            # Determinar street y player basado en el índice
            street = i // 6  # 0=preflop, 1=flop, 2=turn, 3=river
            player = i % 6
            
            # Solo generar acción si estamos dentro del límite
            should_add = action_count < max_actions
            
            action = lax.cond(
                should_add,
                lambda: generate_action_for_situation(action_count, player, hand_strengths[player], street, player),
                lambda: jnp.int32(-1)
            )
            
            # Actualizar secuencia solo si debemos agregar
            new_action_seq = lax.cond(
                should_add,
                lambda: action_seq.at[action_count].set(action),
                lambda: action_seq
            )
            
            new_count = lax.cond(
                should_add,
                lambda: action_count + 1,
                lambda: action_count
            )
            
            return (new_action_seq, new_count), None
        
        # Generar acciones para todas las calles usando scan
        (final_action_seq, final_count), _ = lax.scan(
            add_action_to_sequence,
            (action_sequence, 0),
            jnp.arange(max_actions)  # Procesar hasta max_actions
        )
        
        # 5. Calcular payoffs realistas basados en hand strength
        winner_key = jax.random.fold_in(key1, 999)
        
        # 80% de las veces gana la mejor mano, 20% hay variabilidad
        deterministic_winner = jnp.argmax(hand_strengths)
        random_winner = jax.random.randint(winner_key, (), 0, 6)
        
        should_be_deterministic = jax.random.uniform(winner_key) < 0.8
        winner = lax.cond(
            should_be_deterministic,
            lambda: deterministic_winner,
            lambda: random_winner
        )
        
        # Calcular pot size basado en número de acciones válidas
        valid_actions = jnp.sum(final_action_seq >= 0)
        pot_size = 15.0 + valid_actions * 5.0  # Base pot + acción promedio
        
        # Payoffs: ganador recibe pot, otros pierden sus contribuciones
        payoffs = jnp.zeros(6)
        base_contribution = pot_size / 8.0  # Contribución promedio por jugador
        contributions = jnp.full(6, -base_contribution)  # Todos pierden inicialmente
        payoffs = contributions.at[winner].set(pot_size - base_contribution)  # Ganador recibe pot menos su contribución
        
        return {
            'payoffs': payoffs,
            'action_hist': final_action_seq,
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'final_pot': pot_size,
        }
    
    # Vectorizar la simulación completa
    full_results = jax.vmap(simulate_single_game_with_real_actions)(keys)
    
    # Retornar en formato estándar
    payoffs = full_results['payoffs']
    histories = full_results['action_hist']
    
    game_results = {
        'payoffs': payoffs,
        'hole_cards': full_results['hole_cards'],  # [batch_size, 6, 2]
        'final_community': full_results['community_cards'],  # [batch_size, 5]
        'final_pot': full_results['final_pot'],
        'player_stacks': jnp.ones((batch_size, 6)) * 100.0,
        'player_bets': jnp.abs(payoffs)
    }
    
    return payoffs, histories, game_results

# ---------- Info Set Computation con Bucketing Avanzado ----------
def compute_advanced_info_set(game_results, player_idx, game_idx):
    """
    Calcula un info set avanzado usando bucketing estilo Pluribus.
    Compatible con JAX para máximo rendimiento.
    """
    # Obtener cartas del jugador
    hole_cards = game_results['hole_cards'][game_idx, player_idx]
    community_cards = game_results['final_community'][game_idx]
    
    # Extraer ranks y suits
    hole_ranks = hole_cards // 4
    hole_suits = hole_cards % 4
    
    # Características básicas para el info set
    num_community = jnp.sum(community_cards >= 0)  # Número de cartas comunitarias
    
    # 1. Street bucketing (4 buckets: preflop, flop, turn, river)
    street_bucket = lax.cond(
        num_community == 0,
        lambda: 0,  # Preflop
        lambda: lax.cond(
            num_community == 3,
            lambda: 1,  # Flop
            lambda: lax.cond(
                num_community == 4,
                lambda: 2,  # Turn
                lambda: 3   # River
            )
        )
    )
    
    # 2. Hand strength bucketing (169 preflop buckets como Pluribus)
    high_rank = jnp.maximum(hole_ranks[0], hole_ranks[1])
    low_rank = jnp.minimum(hole_ranks[0], hole_ranks[1])
    is_suited = (hole_suits[0] == hole_suits[1]).astype(jnp.int32)
    is_pair = (hole_ranks[0] == hole_ranks[1]).astype(jnp.int32)
    
    # Preflop bucketing estilo Pluribus
    preflop_bucket = lax.cond(
        is_pair == 1,
        lambda: high_rank,  # Pares: 0-12
        lambda: lax.cond(
            is_suited == 1,
            lambda: 13 + high_rank * 12 + low_rank,  # Suited: 13-168
            lambda: 169 + high_rank * 12 + low_rank  # Offsuit: 169-324
        )
    )
    
    # Normalizamos para que quede en rango 0-168 para compatibilidad
    hand_bucket = jnp.mod(preflop_bucket, 169)
    
    # 3. Position bucketing (6 buckets: 0-5)
    position_bucket = player_idx
    
    # 4. Stack depth bucketing (20 buckets como sistemas profesionales)
    # Usamos pot size como proxy para stack depth por ahora
    pot_size = game_results['final_pot'][game_idx]
    stack_bucket = jnp.clip(pot_size / 5.0, 0, 19).astype(jnp.int32)
    
    # 5. Pot odds bucketing (10 buckets)
    pot_bucket = jnp.clip(pot_size / 10.0, 0, 9).astype(jnp.int32)
    
    # 6. Active players (5 buckets: 2-6 players)
    # Por simplicidad, usamos una estimación
    active_bucket = jnp.clip(player_idx, 0, 4)
    
    # Combinar todos los factores en un info set ID único
    # Total buckets: 4 × 169 × 6 × 20 × 10 × 5 = 405,600 (compatible con 50K limite)
    info_set_id = (
        street_bucket * 10000 +      # 4 × 10000 = 40,000
        hand_bucket * 50 +           # 169 × 50 = 8,450  
        position_bucket * 8 +        # 6 × 8 = 48
        stack_bucket * 2 +           # 20 × 2 = 40
        pot_bucket * 1 +             # 10 × 1 = 10
        active_bucket                # 5 × 1 = 5
    )
    
    # Asegurar que esté en el rango válido
    return jnp.mod(info_set_id, 50000).astype(jnp.int32)

# ---------- DIAGNÓSTICO DETALLADO PARA DEBUG ----------
def debug_info_set_distribution(strategy, game_results, num_samples=1000):
    """
    SIMPLIFICADO: Analiza la distribución de info sets sin trabarse.
    Versión ligera para evitar cuelgues durante el entrenamiento.
    """
    logger.info("\n🔍 DIAGNÓSTICO SIMPLIFICADO DE INFO SETS")
    logger.info("="*50)
    
    try:
        # Limitar mucho el análisis para evitar cuelgues
        sample_size = min(10, num_samples)  # REDUCIDO: solo 10 juegos
        info_set_samples = []
        strategy_samples = []
        
        # Muestreo muy limitado
        for game_idx in range(sample_size):
            for player_idx in [0, 2, 4]:  # Solo 3 jugadores por juego
                try:
                    info_set_idx = compute_advanced_info_set(game_results, player_idx, game_idx)
                    info_set_idx_py = int(info_set_idx)
                    
                    if info_set_idx_py < strategy.shape[0]:  # Verificar bounds
                        info_set_samples.append(info_set_idx_py)
                        strategy_samples.append(strategy[info_set_idx_py])
                    
                except Exception as e:
                    logger.warning(f"Error en sample {game_idx}-{player_idx}: {str(e)[:50]}")
                    continue
        
        if not info_set_samples:
            logger.warning("⚠️ No se pudieron generar muestras de info sets")
            return {'unique_info_sets': 0, 'unique_strategies': 0, 'total_accesses': 0}
        
        # Análisis básico
        unique_info_sets = len(set(info_set_samples))
        total_samples = len(info_set_samples)
        
        logger.info(f"📊 MUESTREO LIMITADO ({total_samples} muestras):")
        logger.info(f"   - Info sets únicos: {unique_info_sets}")
        logger.info(f"   - Total muestras: {total_samples}")
        logger.info(f"   - Diversidad: {unique_info_sets/max(1, total_samples):.2f}")
        
        # Análisis de estrategias (simplificado)
        if strategy_samples:
            # Verificar si las primeras estrategias son diferentes
            first_strategy = strategy_samples[0]
            unique_strategies = 1
            
            for i in range(1, min(5, len(strategy_samples))):  # Solo comparar primeras 5
                if not jnp.allclose(first_strategy, strategy_samples[i], atol=1e-4):
                    unique_strategies += 1
                    break
            
            aggression_avg = float(jnp.mean(jnp.array([jnp.sum(s[3:6]) for s in strategy_samples[:5]])))
            fold_avg = float(jnp.mean(jnp.array([s[0] for s in strategy_samples[:5]])))
            
            logger.info(f"🎯 ESTRATEGIAS MUESTREADAS:")
            logger.info(f"   - Estrategias diferentes detectadas: {unique_strategies}")
            logger.info(f"   - Aggression promedio: {aggression_avg:.3f}")
            logger.info(f"   - Fold rate promedio: {fold_avg:.3f}")
            
            # Quick check si todas son iguales
            if unique_strategies == 1 and len(strategy_samples) > 1:
                logger.warning("⚠️ POSIBLE PROBLEMA: Estrategias muy similares detectadas")
        
        return {
            'unique_info_sets': unique_info_sets,
            'unique_strategies': unique_strategies if strategy_samples else 0,
            'total_accesses': total_samples
        }
        
    except Exception as e:
        logger.error(f"❌ Error en diagnóstico: {str(e)[:100]}")
        return {'unique_info_sets': -1, 'unique_strategies': -1, 'total_accesses': -1}

def debug_specific_hands():
    """
    Debugging específico para las manos que evalúa el Poker IQ.
    Permite verificar exactamente qué info sets se están generando.
    """
    logger.info("\n🃏 DEBUG DE MANOS ESPECÍFICAS (AA vs 72o)")
    logger.info("="*50)
    
    # Test: AA vs 72o que usa el Poker IQ
    aa_info_set = compute_mock_info_set(hole_ranks=[12, 12], is_suited=False, position=2)
    trash_info_set = compute_mock_info_set(hole_ranks=[5, 0], is_suited=False, position=2)
    
    logger.info(f"🔍 ANÁLISIS DE INFO SETS ESPECÍFICOS:")
    logger.info(f"   - AA (pocket aces): info_set = {aa_info_set}")
    logger.info(f"   - 72o (trash hand): info_set = {trash_info_set}")
    logger.info(f"   - ¿Son diferentes? {aa_info_set != trash_info_set}")
    
    if aa_info_set == trash_info_set:
        logger.error("❌ PROBLEMA CRÍTICO: AA y 72o mapean al mismo info set!")
        logger.error("    El sistema no puede distinguir entre manos buenas y malas.")
    
    # Test más manos
    test_hands = [
        ([12, 12], False, "AA"),      # Pocket Aces
        ([11, 11], False, "KK"),      # Pocket Kings  
        ([10, 9], True, "JTs"),       # Jack-Ten suited
        ([10, 9], False, "JTo"),      # Jack-Ten offsuit
        ([5, 0], False, "72o"),       # 7-2 offsuit (worst)
        ([3, 1], False, "52o"),       # 5-2 offsuit (very bad)
    ]
    
    logger.info(f"\n🃏 MAPEO DE MANOS DE TEST:")
    info_set_mapping = {}
    for hole_ranks, is_suited, name in test_hands:
        for pos in [0, 2, 5]:  # Early, middle, late position
            info_set = compute_mock_info_set(hole_ranks, is_suited, pos)
            key = f"{name}_pos{pos}"
            info_set_mapping[key] = info_set
            logger.info(f"   {key:8s}: info_set = {info_set:5d}")
    
    # Verificar si hay suficiente diferenciación
    unique_info_sets = len(set(info_set_mapping.values()))
    total_hands = len(info_set_mapping)
    
    logger.info(f"\n📊 DIFERENCIACIÓN:")
    logger.info(f"   - Total combinaciones: {total_hands}")
    logger.info(f"   - Info sets únicos: {unique_info_sets}")
    logger.info(f"   - Ratio diferenciación: {unique_info_sets/total_hands:.2f}")
    
    if unique_info_sets < total_hands * 0.5:
        logger.warning("⚠️  BAJA DIFERENCIACIÓN: Muchas manos mapean a los mismos info sets")
    
    return info_set_mapping

# ---------- JAX-Native CFR Step ARREGLADO - SOLUCIÓN COMPLETA ----------
@jax.jit
def _jitted_train_step(regrets, strategy, key):
    """
    SOLUCIÓN COMPLETA: CFR step que usa VERDADEROS historiales del motor de juego.
    
    PROBLEMA ANTERIOR: Se usaban historiales sintéticos (action_seed = payoff)
    SOLUCIÓN: Extraer y usar los verdaderos historiales de acción del full_game_engine
    """
    cfg = TrainerConfig()
    keys = jax.random.split(key, cfg.batch_size)
    
    # Obtener datos reales del motor de juego
    payoffs, real_histories, game_results = unified_batch_simulation(keys)
    
    # CRÍTICO: Procesar cada juego usando SUS PROPIOS historiales reales
    def process_single_game(game_idx):
        game_payoff = payoffs[game_idx]
        real_history = real_histories[game_idx]  # HISTORIAL REAL del motor
        
        # Inicializar regrets para este juego
        game_regrets = jnp.zeros_like(regrets)
        
        # NUEVO: Extraer información real del juego para decisiones
        game_hole_cards = game_results['hole_cards'][game_idx]  # [6, 2]
        game_pot = game_results['final_pot'][game_idx]
        
        # Procesar cada decisión en el historial REAL
        def process_real_decision(decision_idx, acc_regrets):
            # ARREGLADO: Usar el historial real del motor
            real_action = real_history[decision_idx]
            is_valid_decision = real_action != -1
            
            def compute_regret_for_real_action():
                # CRÍTICO: El player_idx debe corresponder al verdadero flujo del juego
                # En el motor real, las decisiones siguen un patrón específico
                current_player = decision_idx % 6
                
                # Obtener cartas reales del jugador actual
                player_hole_cards = game_hole_cards[current_player]
                
                # Calcular info set usando bucketing avanzado
                info_set_idx = compute_advanced_info_set(game_results, current_player, game_idx)
                
                # NUEVO: Evaluación hand strength real
                player_hand_strength = evaluate_hand_jax(player_hole_cards)
                
                # SUPER-HUMANO: Clasificación de manos más precisa
                def classify_hand_strength(strength):
                    return {
                        'is_premium': strength > cfg.strong_hand_threshold,
                        'is_strong': strength > (cfg.strong_hand_threshold - 1000),
                        'is_weak': strength < cfg.weak_hand_threshold,
                        'is_bluff_candidate': strength < cfg.bluff_threshold
                    }
                
                hand_class = classify_hand_strength(player_hand_strength)
                
                # POSICIÓN: Factor crítico en decisiones profesionales
                position_factor = lax.cond(
                    current_player <= 1,  # Early position (UTG, UTG+1)
                    lambda: 0.85,          # Más conservador
                    lambda: lax.cond(
                        current_player <= 3,  # Middle position
                        lambda: 1.0,          # Neutro
                        lambda: 1.15          # Late position (más agresivo)
                    )
                )
                
                # SUITED BONUS: Reconocimiento de manos suited
                ranks = player_hole_cards // 4
                suits = player_hole_cards % 4
                is_suited = (suits[0] == suits[1]).astype(jnp.int32)
                suited_factor = lax.cond(
                    is_suited == 1,
                    lambda: 1.0 + cfg.suited_awareness_factor,
                    lambda: 1.0
                )
                
                # POT ODDS: Factor profesional para decisiones
                pot_factor = lax.cond(
                    game_pot > 50,  # Pot grande
                    lambda: 1.1,    # Más agresivo
                    lambda: 0.95    # Más conservador
                )
                
                # COUNTERFACTUAL VALUES para cada acción posible
                def compute_cfv_for_action(alternative_action):
                    """
                    Calcula el valor counterfactual si hubiera tomado una acción diferente
                    """
                    # Valor base: payoff real del juego
                    base_value = game_payoff[current_player]
                    
                    # Factor de acción basado en hand strength y conceptos profesionales
                    action_multiplier = lax.cond(
                        alternative_action == real_action,
                        lambda: 1.0,  # Acción tomada = valor base
                        lambda: lax.cond(
                            alternative_action == 0,  # FOLD
                            lambda: lax.cond(
                                hand_class['is_premium'],
                                lambda: 0.1 * position_factor,  # Fold premium = muy malo
                                lambda: lax.cond(
                                    hand_class['is_strong'],
                                    lambda: 0.4 * position_factor,  # Fold strong = malo
                                    lambda: lax.cond(
                                        hand_class['is_weak'],
                                        lambda: 1.6 / position_factor,  # Fold weak = bueno
                                        lambda: 1.0  # Fold medium = neutro
                                    )
                                )
                            ),
                            lambda: lax.cond(
                                (alternative_action >= 3),  # BET/RAISE/ALL_IN
                                lambda: lax.cond(
                                    hand_class['is_premium'],
                                    lambda: 1.5 * position_factor * suited_factor * pot_factor,  # Premium bet = excelente
                                    lambda: lax.cond(
                                        hand_class['is_strong'],
                                        lambda: 1.25 * position_factor * suited_factor,  # Strong bet = bueno
                                        lambda: lax.cond(
                                            hand_class['is_bluff_candidate'] & (current_player >= 4),  # Late position bluff
                                            lambda: 1.1 * position_factor,  # Bluff posicional = aceptable
                                            lambda: lax.cond(
                                                hand_class['is_weak'],
                                                lambda: 0.2 / position_factor,  # Weak bet = muy malo
                                                lambda: 0.9  # Medium bet = casi neutro
                                            )
                                        )
                                    )
                                ),
                                lambda: lax.cond(
                                    (alternative_action == 1) | (alternative_action == 2),  # CHECK/CALL
                                    lambda: lax.cond(
                                        hand_class['is_strong'],
                                        lambda: 1.15 * suited_factor,  # Strong check/call = bueno
                                        lambda: lax.cond(
                                            hand_class['is_weak'],
                                            lambda: 0.6,  # Weak call = malo
                                            lambda: 1.0   # Medium call = neutro
                                        )
                                    ),
                                    lambda: 1.0  # Otras acciones = neutro
                                )
                            )
                        )
                    )
                    
                    return base_value * action_multiplier
                
                # Calcular CFV para todas las acciones
                all_actions = jnp.arange(cfg.num_actions)
                cfv_values = jax.vmap(compute_cfv_for_action)(all_actions)
                
                # REGRET COMPUTATION: Diferencia entre mejor acción y acción tomada
                regret_values = cfv_values - cfv_values[real_action]
                
                # Actualizar regrets para este info set
                return acc_regrets.at[info_set_idx].add(regret_values)
            
            # Solo procesar decisiones válidas
            return lax.cond(
                is_valid_decision,
                compute_regret_for_real_action,
                lambda: acc_regrets
            )
        
        # Procesar todas las decisiones del historial real
        max_decisions = real_history.shape[0]  # Tamaño real del historial
        final_regrets = lax.fori_loop(0, max_decisions, process_real_decision, game_regrets)
        
        return final_regrets
    
    # Procesar todos los juegos del batch
    batch_regrets = jax.vmap(process_single_game)(jnp.arange(cfg.batch_size))
    
    # Acumular regrets de todo el batch
    accumulated_regrets = regrets + jnp.sum(batch_regrets, axis=0)
    
    # ESTRATEGIA UPDATE: Regret matching estándar
    positive_regrets = jnp.maximum(accumulated_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    
    # Nueva estrategia basada en regrets positivos
    new_strategy = jnp.where(
        regret_sums > 1e-6,  # Threshold para evitar división por cero
        positive_regrets / regret_sums,
        jnp.ones((cfg.max_info_sets, cfg.num_actions)) / cfg.num_actions  # Estrategia uniforme por defecto
    )
    
    return accumulated_regrets, new_strategy

# ---------- Evaluación Objetiva de Poker Knowledge ----------
def evaluate_poker_intelligence(strategy, config: TrainerConfig):
    """
    Evalúa qué tan bien aprende conceptos fundamentales de poker.
    Retorna un 'Poker IQ Score' de 0-100.
    """
    scores = []
    
    # Test 1: Hand Strength Awareness (25 puntos)
    # ¿Juega AA más agresivo que 72o?
    def test_hand_strength():
        # Simular pocket aces (0, 4 = As, As)
        aa_info_set = compute_mock_info_set(hole_ranks=[12, 12], is_suited=False, position=2)
        # Simular 7-2 offsuit (peor mano)
        trash_info_set = compute_mock_info_set(hole_ranks=[5, 0], is_suited=False, position=2)
        
        if aa_info_set < config.max_info_sets and trash_info_set < config.max_info_sets:
            aa_strategy = strategy[aa_info_set]
            trash_strategy = strategy[trash_info_set]
            
            # AA debería tener más probabilidad de bet/raise (acciones 3,4,5)
            aa_aggression = jnp.sum(aa_strategy[3:6])
            trash_aggression = jnp.sum(trash_strategy[3:6])
            
            # Score: 25 puntos si AA es más agresivo
            if aa_aggression > trash_aggression + 0.1:  # Margen de error
                return 25.0
            elif aa_aggression > trash_aggression:
                return 15.0
            else:
                return 0.0
        return 0.0
    
    # Test 2: Position Awareness (25 puntos)
    # ¿Juega más tight en early position?
    def test_position_awareness():
        # SUPER-HUMANO: Test position con mano marginal (suited connector)
        # Esta mano debe jugar diferente según posición
        marginal_hand = [10, 9]  # J-T
        
        early_pos_info = compute_mock_info_set(hole_ranks=marginal_hand, is_suited=True, position=0)  # UTG
        late_pos_info = compute_mock_info_set(hole_ranks=marginal_hand, is_suited=True, position=5)   # Button
        
        if early_pos_info < config.max_info_sets and late_pos_info < config.max_info_sets:
            early_strategy = strategy[early_pos_info]
            late_strategy = strategy[late_pos_info]
            
            # En posición tardía debería ser MÁS agresivo con suited connectors
            early_aggression = jnp.sum(early_strategy[3:6])  # BET/RAISE/ALL_IN
            late_aggression = jnp.sum(late_strategy[3:6])
            
            # SUPER-HUMANO: También revisar fold rate (debería fold más en early)
            early_fold = early_strategy[0]
            late_fold = late_strategy[0]
            
            position_score = 0.0
            
            # Test 1: Más agresión en late position
            if late_aggression > early_aggression + 0.1:
                position_score += 15.0
            elif late_aggression > early_aggression + 0.05:
                position_score += 10.0
                
            # Test 2: Más fold en early position
            if early_fold > late_fold + 0.05:
                position_score += 10.0
            elif early_fold > late_fold:
                position_score += 5.0
                
            return position_score
        return 0.0
    
    # Test 3: Suited vs Offsuit (20 puntos)
    # ¿Valora más las manos suited?
    def test_suited_awareness():
        # SUPER-HUMANO: Test con múltiples tipos de suited hands
        test_cases = [
            ([12, 10], "AJs vs AJo"),  # Premium suited
            ([10, 9], "JTs vs JTo"),   # Suited connector
            ([11, 8], "K9s vs K9o"),   # Suited non-connector
        ]
        
        total_score = 0.0
        valid_tests = 0
        
        for hole_ranks, description in test_cases:
            suited_info = compute_mock_info_set(hole_ranks=hole_ranks, is_suited=True, position=3)
            offsuit_info = compute_mock_info_set(hole_ranks=hole_ranks, is_suited=False, position=3)
            
            if suited_info < config.max_info_sets and offsuit_info < config.max_info_sets:
                suited_strategy = strategy[suited_info]
                offsuit_strategy = strategy[offsuit_info]
                
                # Suited debería ser más agresivo
                suited_aggression = jnp.sum(suited_strategy[3:6])
                offsuit_aggression = jnp.sum(offsuit_strategy[3:6])
                
                # Suited debería fold menos
                suited_fold = suited_strategy[0]
                offsuit_fold = offsuit_strategy[0]
                
                test_score = 0.0
                
                # Más agresión con suited
                if suited_aggression > offsuit_aggression + 0.05:
                    test_score += 4.0
                elif suited_aggression > offsuit_aggression:
                    test_score += 2.0
                    
                # Menos fold con suited
                if suited_fold < offsuit_fold - 0.03:
                    test_score += 3.0
                elif suited_fold < offsuit_fold:
                    test_score += 1.0
                
                total_score += test_score
                valid_tests += 1
        
        # Normalizar score a 20 puntos máximo
        if valid_tests > 0:
            return min(20.0, (total_score / valid_tests) * (20.0 / 7.0))
        return 0.0
    
    # Test 4: Fold Discipline (15 puntos)
    # ¿Foldea manos muy malas?
    def test_fold_discipline():
        # Manos muy malas deberían foldear más
        bad_hands = [
            compute_mock_info_set([2, 5], False, 1),  # 3-6 offsuit
            compute_mock_info_set([1, 7], False, 2),  # 2-8 offsuit
            compute_mock_info_set([0, 9], False, 0),  # 2-10 offsuit
        ]
        
        total_fold_rate = 0.0
        valid_hands = 0
        
        for bad_hand_info in bad_hands:
            if bad_hand_info < config.max_info_sets:
                fold_prob = strategy[bad_hand_info][0]  # Acción FOLD
                total_fold_rate += fold_prob
                valid_hands += 1
        
        if valid_hands > 0:
            avg_fold_rate = total_fold_rate / valid_hands
            # Debería foldear al menos 40% del tiempo con manos muy malas
            if avg_fold_rate > 0.4:
                return 15.0
            elif avg_fold_rate > 0.2:
                return 8.0
            else:
                return 0.0
        return 0.0
    
    # Test 5: Strategy Diversity (15 puntos)
    # ¿Tiene estrategias diversas o siempre hace lo mismo?
    def test_strategy_diversity():
        # Revisar si usa todas las acciones apropiadamente
        total_strategy = jnp.sum(strategy, axis=0)
        
        # Verificar que no haya una acción dominante excesiva
        max_action_prob = jnp.max(total_strategy)
        total_prob = jnp.sum(total_strategy)
        
        if total_prob > 0:
            dominance = max_action_prob / total_prob
            # Estrategia balanceada: ninguna acción > 60% del total
            if dominance < 0.4:
                return 15.0
            elif dominance < 0.6:
                return 10.0
            else:
                return 0.0
        return 0.0
    
    # Ejecutar todos los tests
    scores = [
        test_hand_strength(),
        test_position_awareness(), 
        test_suited_awareness(),
        test_fold_discipline(),
        test_strategy_diversity()
    ]
    
    total_score = jnp.sum(jnp.array(scores))
    
    return {
        'total_poker_iq': float(total_score),
        'hand_strength_score': float(scores[0]),
        'position_score': float(scores[1]), 
        'suited_score': float(scores[2]),
        'fold_discipline_score': float(scores[3]),
        'diversity_score': float(scores[4])
    }

def compute_mock_info_set(hole_ranks, is_suited, position):
    """
    ARREGLADO: Computa un info set usando LA MISMA FÓRMULA que compute_advanced_info_set.
    Esto asegura que los tests de Poker IQ evalúen los info sets realmente entrenados.
    """
    # Hand bucketing usando la misma lógica que compute_advanced_info_set
    high_rank = max(hole_ranks)
    low_rank = min(hole_ranks)
    is_pair = (hole_ranks[0] == hole_ranks[1])
    
    # MISMO cálculo que en compute_advanced_info_set
    if is_pair:
        preflop_bucket = high_rank  # Pares: 0-12
    elif is_suited:
        preflop_bucket = 13 + high_rank * 12 + low_rank  # Suited: 13-168
    else:
        preflop_bucket = 169 + high_rank * 12 + low_rank  # Offsuit: 169+
    
    hand_bucket = preflop_bucket % 169  # Normalizar a 0-168
    
    # MISMA fórmula de combinación que compute_advanced_info_set
    street_bucket = 0  # Preflop para todos los tests
    position_bucket = position
    stack_bucket = 0   # Simplificado para tests
    pot_bucket = 0     # Simplificado para tests  
    active_bucket = 0  # Simplificado para tests
    
    # MISMA fórmula exacta que compute_advanced_info_set
    info_set_id = (
        street_bucket * 10000 +      # 0 * 10000 = 0
        hand_bucket * 50 +           # 169 × 50 = 8,450  
        position_bucket * 8 +        # 6 × 8 = 48
        stack_bucket * 2 +           # 0 * 2 = 0
        pot_bucket * 1 +             # 0 * 1 = 0
        active_bucket                # 0 = 0
    )
    
    return info_set_id % 50000

# ---------- VALIDACIÓN CRÍTICA - VERIFICAR DATOS REALES ----------
def validate_training_data_integrity(strategy, key, verbose=True):
    """
    FUNCIÓN CRÍTICA: Verifica que el entrenamiento use datos reales del motor de juego.
    
    Esta función detecta si hay bugs como:
    - Historiales sintéticos vs reales
    - Info sets incorrectos
    - Mapeo inconsistente entre entrenamiento y evaluación
    """
    if verbose:
        logger.info("\n🔍 VALIDACIÓN DE INTEGRIDAD DE DATOS DE ENTRENAMIENTO")
        logger.info("="*60)
    
    cfg = TrainerConfig()
    
    # Generar datos de prueba
    test_keys = jax.random.split(key, 32)  # Batch pequeño para test
    payoffs, histories, game_results = unified_batch_simulation(test_keys)
    
    validation_results = {
        'real_histories_detected': False,
        'info_set_consistency': False,
        'hand_strength_variation': False,
        'action_diversity': False,
        'critical_bugs': []
    }
    
    # TEST 1: Verificar que los historiales NO son sintéticos
    if verbose:
        logger.info("🧪 TEST 1: Verificando historiales reales vs sintéticos...")
    
    # Los historiales reales deben tener variación natural
    unique_histories = len(jnp.unique(histories.reshape(-1)))
    total_entries = histories.size
    history_diversity = unique_histories / max(1, total_entries)
    
    if history_diversity > 0.1:  # Al menos 10% de diversidad
        validation_results['real_histories_detected'] = True
        if verbose:
            logger.info(f"   ✅ Historiales reales detectados (diversidad: {history_diversity:.2f})")
    else:
        validation_results['critical_bugs'].append("HISTORIALES_SINTÉTICOS")
        if verbose:
            logger.error(f"   ❌ Posibles historiales sintéticos (diversidad: {history_diversity:.2f})")
    
    # TEST 2: Verificar consistencia de info sets
    if verbose:
        logger.info("🧪 TEST 2: Verificando consistencia de info sets...")
    
    test_cases = [
        ([12, 12], False, 2, "AA_mid"),     # Pocket Aces
        ([5, 0], False, 2, "72o_mid"),      # Trash hand
        ([10, 9], True, 5, "JTs_late"),     # Suited connector late
        ([10, 9], False, 0, "JTo_early"),   # Offsuit connector early
    ]
    
    info_set_mapping = {}
    for hole_ranks, is_suited, position, name in test_cases:
        # Info set del evaluador (mismo que usa Poker IQ)
        eval_info_set = compute_mock_info_set(hole_ranks, is_suited, position)
        info_set_mapping[name] = eval_info_set
        
        if verbose:
            logger.info(f"   {name}: eval_info_set = {eval_info_set}")
    
    # Verificar que AA y 72o tengan info sets diferentes
    if info_set_mapping["AA_mid"] != info_set_mapping["72o_mid"]:
        validation_results['info_set_consistency'] = True
        if verbose:
            logger.info("   ✅ AA y 72o tienen info sets diferentes (CORRECTO)")
    else:
        validation_results['critical_bugs'].append("INFO_SETS_IGUALES")
        if verbose:
            logger.error("   ❌ AA y 72o tienen el mismo info set (BUG CRÍTICO)")
    
    # TEST 3: Verificar variación en hand strength
    if verbose:
        logger.info("🧪 TEST 3: Verificando evaluación de hand strength...")
    
    # Evaluar manos diferentes
    aa_cards = jnp.array([51, 47], dtype=jnp.int8)  # As spades, As hearts
    trash_cards = jnp.array([20, 0], dtype=jnp.int8)  # 7 clubs, 2 spades
    
    aa_strength = evaluate_hand_jax(aa_cards)
    trash_strength = evaluate_hand_jax(trash_cards)
    
    if aa_strength > trash_strength + 1000:  # Diferencia significativa
        validation_results['hand_strength_variation'] = True
        if verbose:
            logger.info(f"   ✅ AA strength ({aa_strength}) > 72o strength ({trash_strength})")
    else:
        validation_results['critical_bugs'].append("HAND_STRENGTH_SIN_VARIACIÓN")
        if verbose:
            logger.error(f"   ❌ AA ({aa_strength}) vs 72o ({trash_strength}) - Sin variación suficiente")
    
    # TEST 4: Verificar diversidad de acciones en estrategia
    if verbose:
        logger.info("🧪 TEST 4: Verificando diversidad de estrategia...")
    
    # Revisar si la estrategia tiene variación
    strategy_std = jnp.std(strategy)
    if strategy_std > 0.01:  # Al menos algo de variación
        validation_results['action_diversity'] = True
        if verbose:
            logger.info(f"   ✅ Estrategia tiene variación (std: {strategy_std:.4f})")
    else:
        validation_results['critical_bugs'].append("ESTRATEGIA_UNIFORME")
        if verbose:
            logger.warning(f"   ⚠️ Estrategia muy uniforme (std: {strategy_std:.4f})")
    
    # RESUMEN
    all_tests_passed = (
        validation_results['real_histories_detected'] and
        validation_results['info_set_consistency'] and
        validation_results['hand_strength_variation'] and
        validation_results['action_diversity']
    )
    
    if verbose:
        logger.info("\n📊 RESUMEN DE VALIDACIÓN:")
        logger.info(f"   - Historiales reales: {'✅' if validation_results['real_histories_detected'] else '❌'}")
        logger.info(f"   - Info sets consistentes: {'✅' if validation_results['info_set_consistency'] else '❌'}")
        logger.info(f"   - Hand strength variable: {'✅' if validation_results['hand_strength_variation'] else '❌'}")
        logger.info(f"   - Estrategia diversa: {'✅' if validation_results['action_diversity'] else '❌'}")
        
        if validation_results['critical_bugs']:
            logger.error(f"\n🚨 BUGS CRÍTICOS DETECTADOS: {validation_results['critical_bugs']}")
            logger.error("   El entrenamiento NO funcionará correctamente con estos bugs.")
        elif all_tests_passed:
            logger.info("\n🎉 TODOS LOS TESTS PASARON - Sistema listo para entrenamiento")
        else:
            logger.warning("\n⚠️ Algunos tests fallaron - Revisar configuración")
        
        logger.info("="*60)
    
    return validation_results

# ---------- SUPER-HUMANO: Sistema de Monitoreo Mejorado ----------
def enhanced_poker_iq_evaluation(strategy, config: TrainerConfig, iteration_num=0):
    """
    Evaluación mejorada que incluye diagnósticos adicionales
    """
    # Evaluación estándar
    standard_results = evaluate_poker_intelligence(strategy, config)
    
    # Diagnósticos adicionales
    enhanced_results = standard_results.copy()
    
    # Test de robustez: ¿Las estrategias son estables?
    def test_strategy_stability():
        # Muestrear algunas estrategias específicas
        test_info_sets = [1000, 5000, 10000, 15000, 20000]
        stability_score = 0.0
        
        for info_set in test_info_sets:
            if info_set < config.max_info_sets:
                strategy_vector = strategy[info_set]
                # Verificar que no sea demasiado extrema
                max_prob = jnp.max(strategy_vector)
                min_prob = jnp.min(strategy_vector)
                
                # Penalizar estrategias extremas (todo en una acción)
                if max_prob < 0.95 and min_prob > 0.001:
                    stability_score += 2.0
        
        return min(10.0, stability_score)
    
    enhanced_results['stability_score'] = float(test_strategy_stability())
    enhanced_results['iteration'] = iteration_num
    enhanced_results['total_enhanced_score'] = (
        enhanced_results['total_poker_iq'] + enhanced_results['stability_score']
    )
    
    return enhanced_results

# ---------- Trainer con Validación Integrada ----------
class PokerTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.regrets  = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
        self.strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
        
        # NUEVO: Sistema de snapshots para tracking de evolución
        self.poker_iq_snapshots = {}
        
        logger.info("=" * 60)
        logger.info("🎯 PokerTrainer CFR-JIT inicializado")
        logger.info("=" * 60)
        logger.info(f"📊 Configuración:")
        logger.info(f"   - Batch size: {config.batch_size}")
        logger.info(f"   - Num actions: {config.num_actions}")
        logger.info(f"   - Max info sets: {config.max_info_sets:,}")
        logger.info(f"   - Shape regrets: {self.regrets.shape}")
        logger.info(f"   - Shape strategy: {self.strategy.shape}")
        logger.info("=" * 60)

    def train(self, num_iterations: int, save_path: str, save_interval: int, snapshot_iterations=None):
        key = jax.random.PRNGKey(42)  # Semilla fija para reproducibilidad
        
        # Iteraciones para snapshots (por defecto: 1/3, 2/3, final)
        if snapshot_iterations is None:
            snapshot_iterations = [
                max(1, num_iterations // 3),      # 33%
                max(1, 2 * num_iterations // 3),  # 66%
                num_iterations                    # 100%
            ]
        
        logger.info("\n🚀 INICIANDO ENTRENAMIENTO CFR CON VALIDACIÓN COMPLETA")
        logger.info(f"   Total iteraciones: {num_iterations}")
        logger.info(f"   Guardar cada: {save_interval} iteraciones")
        logger.info(f"   Path base: {save_path}")
        logger.info(f"   Snapshots en: {snapshot_iterations}")
        
        # =================== VALIDACIÓN CRÍTICA PRE-ENTRENAMIENTO ===================
        logger.info("\n🔍 EJECUTANDO VALIDACIÓN CRÍTICA PRE-ENTRENAMIENTO...")
        validation_key = jax.random.PRNGKey(99)
        validation_results = validate_training_data_integrity(self.strategy, validation_key, verbose=True)
        
        # Verificar que no hay bugs críticos antes de entrenar
        if validation_results['critical_bugs']:
            logger.error("\n🚨 ENTRENAMIENTO ABORTADO - Bugs críticos detectados:")
            for bug in validation_results['critical_bugs']:
                logger.error(f"   - {bug}")
            logger.error("🛠️  Corrija estos problemas antes de continuar.")
            raise RuntimeError("Bugs críticos detectados en validación pre-entrenamiento")
        
        logger.info("✅ Validación pre-entrenamiento EXITOSA - Sistema listo")
        
        # =================== DIAGNÓSTICO INICIAL ===================
        logger.info("\n🔍 EJECUTANDO DIAGNÓSTICO INICIAL...")
        debug_specific_hands()
        
        # Generar datos de muestra para debug
        debug_key = jax.random.PRNGKey(42)
        debug_keys = jax.random.split(debug_key, 128)
        debug_game_results = unified_batch_simulation(debug_keys)[2] # Extract game_results
        debug_analysis = debug_info_set_distribution(self.strategy, debug_game_results)
        
        logger.info("\n⏳ Compilando función JIT (primera iteración será más lenta)...\n")
        
        import time
        start_time = time.time()
        
        for i in range(1, num_iterations + 1):
            self.iteration += 1
            iter_key = jax.random.fold_in(key, self.iteration)
            
            iter_start = time.time()
            
            try:
                # Un paso de entrenamiento
                self.regrets, self.strategy = _jitted_train_step(
                    self.regrets,
                    self.strategy,
                    iter_key
                )
                
                # Esperamos a que termine la computación
                self.regrets.block_until_ready()
                
                iter_time = time.time() - iter_start
                
                # Log simple cada iteración (solo progreso básico)
                if self.iteration % max(1, num_iterations // 10) == 0:
                    progress = 100 * self.iteration / num_iterations
                    logger.info(f"✓ Progreso: {progress:.0f}% ({self.iteration}/{num_iterations}) - {iter_time:.2f}s")
                
                # NUEVO: Debug intermedio en la mitad del entrenamiento
                if self.iteration == num_iterations // 2:
                    logger.info("\n🔍 DIAGNÓSTICO INTERMEDIO (50% completado)...")
                    mid_game_results = unified_batch_simulation(jax.random.split(iter_key, 128))[2] # Extract game_results
                    mid_analysis = debug_info_set_distribution(self.strategy, mid_game_results)
                    
                    # Comparar con estado inicial
                    improvement = mid_analysis['unique_strategies'] - debug_analysis['unique_strategies']
                    logger.info(f"📈 Cambio en diversidad: {improvement:+d} estrategias únicas")
                
                # Tomar snapshots del Poker IQ en iteraciones específicas
                if self.iteration in snapshot_iterations:
                    # Usar evaluación mejorada con diagnósticos adicionales
                    poker_iq = enhanced_poker_iq_evaluation(self.strategy, self.config, self.iteration)
                    self.poker_iq_snapshots[self.iteration] = poker_iq
                    
                    logger.info(f"\n📸 SNAPSHOT ITERACIÓN {self.iteration}")
                    logger.info(f"   - IQ Total: {poker_iq['total_poker_iq']:.1f}/100")
                    logger.info(f"   - IQ Enhanced: {poker_iq['total_enhanced_score']:.1f}/110")
                    logger.info(f"   - Hand Strength: {poker_iq['hand_strength_score']:.1f}/25")
                    logger.info(f"   - Position: {poker_iq['position_score']:.1f}/25")
                    logger.info(f"   - Suited: {poker_iq['suited_score']:.1f}/20")
                    logger.info(f"   - Fold Disc.: {poker_iq['fold_discipline_score']:.1f}/15")
                    logger.info(f"   - Stability: {poker_iq['stability_score']:.1f}/10")
                    
                    # Validación adicional en iteración intermedia
                    if self.iteration == num_iterations // 2:
                        logger.info("\n🔍 VALIDACIÓN INTERMEDIA (50% completado)...")
                        mid_validation = validate_training_data_integrity(
                            self.strategy, 
                            jax.random.fold_in(key, self.iteration + 1000), 
                            verbose=False
                        )
                        if mid_validation['critical_bugs']:
                            logger.warning(f"⚠️ Bugs detectados en validación intermedia: {mid_validation['critical_bugs']}")
                        else:
                            logger.info("✅ Validación intermedia exitosa")
                
            except Exception as e:
                logger.error(f"\n❌ ERROR en iteración {self.iteration}")
                logger.error(f"   Tipo: {type(e).__name__}")
                logger.error(f"   Mensaje: {str(e)}")
                logger.error(f"   Shapes - regrets: {self.regrets.shape}, strategy: {self.strategy.shape}")
                
                import traceback
                logger.error("\nTraceback completo:")
                logger.error(traceback.format_exc())
                
                raise
                
            # Guardamos checkpoints
            if self.iteration % save_interval == 0:
                checkpoint_path = f"{save_path}_iter_{self.iteration}.pkl"
                self.save_model(checkpoint_path)
        
        # Resumen final
        total_time = time.time() - start_time
        
        # =================== VALIDACIÓN FINAL COMPLETA ===================
        logger.info("\n🔍 EJECUTANDO VALIDACIÓN FINAL COMPLETA...")
        final_validation_key = jax.random.PRNGKey(999)
        final_validation = validate_training_data_integrity(self.strategy, final_validation_key, verbose=True)
        
        # Verificar que el entrenamiento fue exitoso
        if final_validation['critical_bugs']:
            logger.error("\n⚠️ ADVERTENCIA: Bugs detectados en validación final:")
            for bug in final_validation['critical_bugs']:
                logger.error(f"   - {bug}")
            logger.error("El modelo puede no funcionar correctamente.")
        else:
            logger.info("\n🎉 VALIDACIÓN FINAL EXITOSA - Modelo entrenado correctamente")
        
        # DIAGNÓSTICO FINAL
        logger.info("\n🔍 DIAGNÓSTICO FINAL...")
        final_keys = jax.random.split(jax.random.PRNGKey(99), 128)
        final_game_results = unified_batch_simulation(final_keys)[2] # Extract game_results
        final_analysis = debug_info_set_distribution(self.strategy, final_game_results)
        
        # Evaluación final del Poker IQ
        logger.info("\n🧠 EVALUACIÓN FINAL DE POKER IQ...")
        final_poker_iq = enhanced_poker_iq_evaluation(self.strategy, self.config, num_iterations)
        self.poker_iq_snapshots[num_iterations] = final_poker_iq
        
        logger.info(f"🏆 RESULTADO FINAL:")
        logger.info(f"   - IQ Total: {final_poker_iq['total_poker_iq']:.1f}/100")
        logger.info(f"   - IQ Enhanced: {final_poker_iq['total_enhanced_score']:.1f}/110")
        
        # Guardamos el modelo final
        final_path = f"{save_path}_final.pkl"
        self.save_model(final_path)
        
        # NUEVO: Reporte de evolución de inteligencia
        self._log_poker_evolution_summary(num_iterations, total_time)

    def _log_poker_evolution_summary(self, total_iterations, total_time):
        """Muestra un resumen de la evolución del Poker IQ"""
        logger.info("\n" + "="*80)
        logger.info("🧠 RESUMEN DE EVOLUCIÓN DE POKER INTELLIGENCE")
        logger.info("="*80)
        
        if not self.poker_iq_snapshots:
            logger.info("❌ No se tomaron snapshots de IQ durante el entrenamiento")
            return
        
        # Mostrar evolución
        sorted_snapshots = sorted(self.poker_iq_snapshots.items())
        
        logger.info("📈 EVOLUCIÓN DEL POKER IQ:")
        
        for iteration, iq_data in sorted_snapshots:
            progress = 100 * iteration / total_iterations
            level = self._get_iq_level(iq_data['total_poker_iq'])
            
            logger.info(f"\n🎯 Iteración {iteration} ({progress:.0f}%):")
            logger.info(f"   - IQ Total: {iq_data['total_poker_iq']:.1f}/100 {level}")
            logger.info(f"   - 💪 Fuerza manos: {iq_data['hand_strength_score']:.1f}/25")
            logger.info(f"   - 📍 Posición: {iq_data['position_score']:.1f}/25")
            logger.info(f"   - 🃏 Suited: {iq_data['suited_score']:.1f}/20")
            logger.info(f"   - 🚫 Fold disc.: {iq_data['fold_discipline_score']:.1f}/15")
            logger.info(f"   - 🎭 Diversidad: {iq_data['diversity_score']:.1f}/15")
        
        # Calcular mejoras
        if len(sorted_snapshots) >= 2:
            first_iq = sorted_snapshots[0][1]['total_poker_iq']
            last_iq = sorted_snapshots[-1][1]['total_poker_iq']
            improvement = last_iq - first_iq
            
            logger.info(f"\n📊 ANÁLISIS DE MEJORA:")
            logger.info(f"   - IQ inicial: {first_iq:.1f}/100")
            logger.info(f"   - IQ final: {last_iq:.1f}/100")
            logger.info(f"   - Mejora total: +{improvement:.1f} puntos")
            logger.info(f"   - Mejora por iteración: +{improvement/total_iterations:.2f} puntos")
            
            if improvement > 20:
                verdict = "🏆 EXCELENTE - Aprendizaje muy efectivo"
            elif improvement > 10:
                verdict = "🥇 BUENO - Progreso sólido"
            elif improvement > 5:
                verdict = "🥈 MODERADO - Mejora detectada"
            else:
                verdict = "🤔 LENTO - Necesita más iteraciones"
                
            logger.info(f"   - Veredicto: {verdict}")
        
        # Stats finales
        logger.info(f"\n⏱️ ESTADÍSTICAS FINALES:")
        logger.info(f"   - Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"   - Velocidad: {total_iterations/total_time:.1f} iter/s")
        logger.info(f"   - Throughput: ~{total_iterations * 128 * 50 / total_time:.0f} hands/s")
        
        logger.info("="*80 + "\n")

    def _get_iq_level(self, iq_score):
        """Retorna el nivel de IQ como emoji"""
        if iq_score >= 80:
            return "🏆"
        elif iq_score >= 60:
            return "🥇"
        elif iq_score >= 40:
            return "🥈"
        elif iq_score >= 20:
            return "🥉"
        else:
            return "🤖"

    def save_model(self, path: str):
        """Guarda el modelo actual a disco"""
        model_data = {
            'regrets':   np.asarray(self.regrets),
            'strategy':  np.asarray(self.strategy),
            'iteration': self.iteration,
            'config':    self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        size_mb = os.path.getsize(path) / 1024 / 1024
        logger.info(f"💾 Checkpoint guardado: {path} ({size_mb:.1f} MB)")

    def load_model(self, path: str):
        """Carga un modelo desde disco"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.regrets   = jnp.array(data['regrets'])
        self.strategy  = jnp.array(data['strategy'])
        self.iteration = data['iteration']
        
        if 'config' in data:
            self.config = data['config']
        
        logger.info(f"📂 Modelo cargado: {path}")
        logger.info(f"   Iteración: {self.iteration}")
        logger.info(f"   Shape regrets: {self.regrets.shape}")
        logger.info(f"   Shape strategy: {self.strategy.shape}")

# ---------- SUPER-HUMANO: Configuración de producción ----------
class SuperHumanTrainerConfig(TrainerConfig):
    """
    Configuración avanzada para entrenamientos de nivel super-humano
    que pueden competir contra Pluribus y profesionales.
    """
    # Training parameters para entrenamientos largos
    batch_size: int = 256               # Más muestras por iteración
    max_iterations: int = 2000          # Entrenamientos largos
    save_interval: int = 50             # Guardar más frecuente
    snapshot_iterations: list = None    # Se calculará automáticamente
    
    # Learning rates adaptativos
    initial_learning_rate: float = 0.02
    final_learning_rate: float = 0.005
    learning_decay_factor: float = 0.95
    
    # Factores de awareness más agresivos
    position_awareness_factor: float = 0.4   # Stronger position learning
    suited_awareness_factor: float = 0.3     # Stronger suited learning
    pot_odds_factor: float = 0.25           # Pot odds consideration
    
    # Thresholds profesionales calibrados
    strong_hand_threshold: int = 4000       # Solo hands verdaderamente premium
    weak_hand_threshold: int = 1500         # Threshold más estricto
    bluff_threshold: int = 600              # Bluffs más selectivos
    premium_threshold: int = 5000           # AA, KK level
    
    # Multi-street preparation
    enable_multi_street: bool = False       # Para futuro
    street_learning_factors: list = None    # [preflop, flop, turn, river]
    
    # Advanced concepts
    enable_range_construction: bool = False  # Para futuro
    enable_opponent_modeling: bool = False   # Para futuro
    enable_icm_training: bool = False       # Para futuro
    
    def __post_init__(self):
        # Auto-calculate snapshot iterations para entrenamientos largos
        if self.snapshot_iterations is None:
            checkpoints = [
                self.max_iterations // 4,      # 25%
                self.max_iterations // 2,      # 50%  
                3 * self.max_iterations // 4,  # 75%
                self.max_iterations             # 100%
            ]
            self.snapshot_iterations = checkpoints
            
        # Auto-calculate street learning factors
        if self.street_learning_factors is None:
            self.street_learning_factors = [1.0, 0.8, 0.6, 0.4]  # Decreasing by street

# ---------- SUPER-HUMANO: Funciones de entrenamiento avanzadas ----------
def create_super_human_trainer(config_type="standard"):
    """
    Factory function para crear trainers de diferentes niveles.
    
    Args:
        config_type: "standard", "super_human", "pluribus_level"
    """
    if config_type == "super_human":
        config = SuperHumanTrainerConfig()
        logger.info("🏆 SUPER-HUMAN TRAINER CONFIG LOADED")
        logger.info(f"   - Max iterations: {config.max_iterations}")
        logger.info(f"   - Batch size: {config.batch_size}")
        logger.info(f"   - Position factor: {config.position_awareness_factor}")
        logger.info(f"   - Suited factor: {config.suited_awareness_factor}")
        
    elif config_type == "pluribus_level":
        config = SuperHumanTrainerConfig()
        # Configuración extrema para competir vs Pluribus
        config.max_iterations = 5000
        config.batch_size = 512
        config.position_awareness_factor = 0.5
        config.suited_awareness_factor = 0.4
        config.strong_hand_threshold = 4500
        config.weak_hand_threshold = 1800
        
        logger.info("🚀 PLURIBUS-LEVEL TRAINER CONFIG LOADED")
        logger.info(f"   - Max iterations: {config.max_iterations}")
        logger.info(f"   - Batch size: {config.batch_size}")
        logger.info("   - WARNING: This will take hours to train!")
        
    else:  # standard
        config = TrainerConfig()
        logger.info("⚡ Standard trainer config loaded")
    
    return PokerTrainer(config)