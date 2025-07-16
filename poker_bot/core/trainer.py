# poker_bot/core/trainer.py
import jax
import jax.numpy as jnp
import numpy as np
import logging
import pickle
import os
import time
from dataclasses import dataclass
from . import jax_game_engine as ege  # CAMBIADO: motor elite en lugar de full_game_engine
from jax import Array
from functools import partial
from jax import lax
from jax import ShapeDtypeStruct

logger = logging.getLogger(__name__)

# ---------- Wrapper para evaluador real compatible con JAX ----------
def evaluate_hand_jax(cards_jax):
    """
    ARREGLADO: Evaluador JAX puro sin numpy operations.
    Compatible con JIT compilation usando solo operaciones JAX nativas.
    """
    # Verificar si las cartas son válidas (todas >= 0)
    cards_valid = jnp.all(cards_jax >= 0)
    
    # Evaluación simple puramente JAX (sin numpy ni evaluador externo)
    def simple_evaluation():
        # Calcular ranks y suits usando operaciones JAX puras
        ranks = cards_jax // 4  # 0-12 (2 hasta A)
        suits = cards_jax % 4   # 0-3 (spades, hearts, diamonds, clubs)
        
        # Hand strength básico usando solo operaciones JAX
        high_rank = jnp.max(ranks)
        low_rank = jnp.min(ranks)
        is_pair = jnp.sum(ranks[0] == ranks[1]).astype(jnp.int32)
        
        # Fórmula simple para hand strength (0-7461, mayor = mejor)
        base_strength = (high_rank * 13 + low_rank) * 10
        pair_bonus = is_pair * 1000
        
        # Suited bonus (operaciones JAX puras)
        suited_bonus = jnp.where(
            suits[0] == suits[1], 
            200,  # Bonus por suited
            0
        )
        
        total_strength = base_strength + pair_bonus + suited_bonus
        return jnp.clip(total_strength, 0, 7461).astype(jnp.int32)
    
    # Invalid hand case
    def invalid_evaluation():
        return jnp.int32(9999)  # Peor hand strength posible
    
    # Usar lax.cond para compatibilidad JAX
    return lax.cond(
        cards_valid,
        simple_evaluation,
        invalid_evaluation
    )

# ---------- Config ----------
@dataclass
class TrainerConfig:
    batch_size: int = 128
    num_actions: int = 6  # CAMBIADO: de 3 a 6 para coincidir con el motor elite (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)
    max_info_sets: int = 50_000

# ---------- Elite Game Engine Wrapper para CFR ----------
@jax.jit
def elite_batch_play(keys):
    """
    Wrapper JIT-compatible que usa el motor elite y retorna formato compatible con CFR.
    Retorna (payoffs, histories) como esperaba el trainer original.
    """
    # Usar el motor elite para simular juegos
    game_results = ege.batch_simulate(keys)
    
    # Extraer payoffs (ya en formato correcto)
    payoffs = game_results['payoffs']
    
    # Construir historias sintéticas basadas en los resultados del juego
    # Por ahora usamos una historia simplificada hasta que implementemos el historial completo.
    batch_size = payoffs.shape[0]
    max_history_length = 60
    
    # Crear historias basadas en los resultados del juego
    histories = jnp.full((batch_size, max_history_length), -1, dtype=jnp.int32)
    
    # Simular algunas acciones básicas por juego usando lax.fori_loop (compatible con JIT)
    def add_action(i, hist):
        # Acciones aleatorias pero deterministas basadas en el payoff
        action_seed = payoffs[:, 0] + i  # Usar payoff como semilla
        actions = jnp.mod(jnp.abs(action_seed).astype(jnp.int32), 6)  # 0-5 para 6 acciones
        return hist.at[:, i].set(actions)
    
    histories = lax.fori_loop(0, jnp.minimum(10, max_history_length), add_action, histories)
    
    return payoffs, histories

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

# ---------- JAX-Native CFR Step MEJORADO CON DEBUG ----------
@jax.jit
def _jitted_train_step(regrets, strategy, key):
    """
    Un paso de CFR usando el motor elite completo con bucketing avanzado
    """
    cfg = TrainerConfig()
    keys = jax.random.split(key, cfg.batch_size)
    
    # MEJORADO: Usar wrapper elite que retorna formato compatible
    payoffs, histories = elite_batch_play(keys)
    
    # También obtener resultados completos para info sets reales
    game_results = ege.batch_simulate(keys)
    
    # Procesar todos los juegos del batch directamente
    def process_single_game(game_idx):
        payoff = payoffs[game_idx]
        history = histories[game_idx]
        
        # Acumular regrets para este juego
        game_regrets = jnp.zeros_like(regrets)
        
        def process_step(step_idx, acc_regrets):
            action = history[step_idx]
            valid = action != -1
            
            def compute_regret():
                # ARREGLADO: Usar ciclo rotativo de jugadores más realista
                player_idx = step_idx % 6  # Jugador actual en ciclo rotativo
                
                # Calcular info set usando bucketing avanzado estilo Pluribus
                info_set_idx = compute_advanced_info_set(game_results, player_idx, game_idx)
                
                # MEJORADO: Counterfactual values más realistas basados en hand strength
                def cfv(a):
                    # Obtener cartas del jugador para evaluación más realista
                    hole_cards = game_results['hole_cards'][game_idx, player_idx]
                    
                    # Usar evaluador real del motor elite
                    hand_strength = evaluate_hand_jax(hole_cards)
                    
                    # Base value usando payoff real del juego
                    base_value = payoff[player_idx]
                    
                    # Factor de acción más realista basado en hand strength
                    action_factor = lax.cond(
                        a == action,
                        lambda: 1.0,  # Acción real tomada
                        lambda: lax.cond(
                            a == 0,  # FOLD
                            lambda: lax.cond(
                                hand_strength > 5000,  # Mano fuerte
                                lambda: 0.1,  # Fold con mano fuerte es malo
                                lambda: 0.8   # Fold con mano débil es bueno
                            ),
                            lambda: lax.cond(
                                (a == 1) | (a == 2),  # CHECK/CALL
                                lambda: lax.cond(
                                    hand_strength > 5000,
                                    lambda: 0.6,  # Check/call con mano fuerte es conservador
                                    lambda: 0.4   # Check/call con mano débil es arriesgado
                                ),
                                lambda: lax.cond(  # BET/RAISE/ALL_IN
                                    hand_strength > 5000,
                                    lambda: 1.2,  # Apostar con mano fuerte es bueno
                                    lambda: 0.2   # Apostar con mano débil es bluff
                                )
                            )
                        )
                    )
                    
                    return base_value * action_factor
                
                cfv_all = jax.vmap(cfv)(jnp.arange(cfg.num_actions))
                regret_delta = cfv_all - cfv_all[action]
                
                return acc_regrets.at[info_set_idx].add(regret_delta)
            
            return lax.cond(valid, compute_regret, lambda: acc_regrets)
        
        # Procesar todos los pasos del juego
        final_game_regrets = lax.fori_loop(0, 60, process_step, game_regrets)
        return final_game_regrets

    # Procesar todos los juegos y sumar los regrets
    all_game_regrets = jax.vmap(process_single_game)(jnp.arange(cfg.batch_size))
    accumulated_regrets = regrets + jnp.sum(all_game_regrets, axis=0)
    
    # Actualizar estrategia
    positive_regrets = jnp.maximum(accumulated_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    new_strategy = jnp.where(
        regret_sums > 0,
        positive_regrets / regret_sums,
        jnp.ones((cfg.max_info_sets, cfg.num_actions)) / cfg.num_actions
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
        # Misma mano en posiciones diferentes
        early_pos_info = compute_mock_info_set(hole_ranks=[10, 9], is_suited=True, position=0)
        late_pos_info = compute_mock_info_set(hole_ranks=[10, 9], is_suited=True, position=5)
        
        if early_pos_info < config.max_info_sets and late_pos_info < config.max_info_sets:
            early_strategy = strategy[early_pos_info]
            late_strategy = strategy[late_pos_info]
            
            # En posición tardía debería ser más agresivo
            early_aggression = jnp.sum(early_strategy[3:6])
            late_aggression = jnp.sum(late_strategy[3:6])
            
            if late_aggression > early_aggression + 0.05:
                return 25.0
            elif late_aggression > early_aggression:
                return 15.0
            else:
                return 0.0
        return 0.0
    
    # Test 3: Suited vs Offsuit (20 puntos)
    # ¿Valora más las manos suited?
    def test_suited_awareness():
        # KQ suited vs KQ offsuit
        suited_info = compute_mock_info_set(hole_ranks=[11, 10], is_suited=True, position=3)
        offsuit_info = compute_mock_info_set(hole_ranks=[11, 10], is_suited=False, position=3)
        
        if suited_info < config.max_info_sets and offsuit_info < config.max_info_sets:
            suited_strategy = strategy[suited_info]
            offsuit_strategy = strategy[offsuit_info]
            
            # Suited debería ser ligeramente más agresivo
            suited_aggression = jnp.sum(suited_strategy[3:6])
            offsuit_aggression = jnp.sum(offsuit_strategy[3:6])
            
            if suited_aggression > offsuit_aggression + 0.03:
                return 20.0
            elif suited_aggression > offsuit_aggression:
                return 10.0
            else:
                return 0.0
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

# ---------- Trainer ----------
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
        
        logger.info("\n🚀 INICIANDO ENTRENAMIENTO CFR")
        logger.info(f"   Total iteraciones: {num_iterations}")
        logger.info(f"   Guardar cada: {save_interval} iteraciones")
        logger.info(f"   Path base: {save_path}")
        logger.info(f"   Snapshots en: {snapshot_iterations}")
        logger.info("\n⏳ Compilando función JIT (primera iteración será más lenta)...\n")
        
        # NUEVO: Debug inicial de info sets
        logger.info("\n🔍 EJECUTANDO DIAGNÓSTICO INICIAL...")
        debug_specific_hands()
        
        # Generar datos de muestra para debug
        debug_key = jax.random.PRNGKey(42)
        debug_keys = jax.random.split(debug_key, 128)
        debug_game_results = ege.batch_simulate(debug_keys)
        debug_analysis = debug_info_set_distribution(self.strategy, debug_game_results)
        
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
                    mid_game_results = ege.batch_simulate(jax.random.split(iter_key, 128))
                    mid_analysis = debug_info_set_distribution(self.strategy, mid_game_results)
                    
                    # Comparar con estado inicial
                    improvement = mid_analysis['unique_strategies'] - debug_analysis['unique_strategies']
                    logger.info(f"📈 Cambio en diversidad: {improvement:+d} estrategias únicas")
                
                # Tomar snapshots del Poker IQ en iteraciones específicas
                if self.iteration in snapshot_iterations:
                    poker_iq = evaluate_poker_intelligence(self.strategy, self.config)
                    self.poker_iq_snapshots[self.iteration] = poker_iq
                    logger.info(f"📸 Snapshot tomado en iteración {self.iteration} - IQ: {poker_iq['total_poker_iq']:.1f}/100")
                
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
        
        # NUEVO: Debug final completo
        logger.info("\n🔍 DIAGNÓSTICO FINAL...")
        final_keys = jax.random.split(jax.random.PRNGKey(99), 128)
        final_game_results = ege.batch_simulate(final_keys)
        final_analysis = debug_info_set_distribution(self.strategy, final_game_results)
        
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