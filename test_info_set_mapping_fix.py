#!/usr/bin/env python3
"""
TEST CRÍTICO: Verificar que la corrección de mapeo de info sets funciona.

Este script verifica que compute_mock_info_set (evaluación) ahora genere
info sets que coincidan con los generados durante el entrenamiento real.
"""

import logging
import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, 
    compute_mock_info_set, compute_advanced_info_set,
    unified_batch_simulation, _jitted_train_step
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_info_set_mapping_fix():
    """
    Prueba crítica: ¿Los info sets de evaluación ahora coinciden con los de entrenamiento?
    """
    logger.info("🔧 TEST CRÍTICO: Verificación de corrección de mapeo de info sets")
    logger.info("="*70)
    
    # 1. Generar datos de entrenamiento reales
    logger.info("🎲 Generando datos de entrenamiento...")
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 128)
    
    # Obtener datos del motor de entrenamiento
    payoffs, histories, game_results = unified_batch_simulation(keys)
    
    # 2. Extraer info sets que realmente se generan durante entrenamiento
    logger.info("🔍 Extrayendo info sets generados durante entrenamiento...")
    training_info_sets = set()
    
    # Muestrear los primeros 32 juegos para análisis
    for game_idx in range(32):
        for player_idx in range(6):
            try:
                info_set = compute_advanced_info_set(game_results, player_idx, game_idx)
                training_info_sets.add(int(info_set))
            except Exception as e:
                logger.warning(f"Error extrayendo info set game={game_idx}, player={player_idx}: {e}")
    
    logger.info(f"📊 Info sets únicos extraídos del entrenamiento: {len(training_info_sets)}")
    sorted_training = sorted(training_info_sets)
    logger.info(f"   Muestra: {sorted_training[:10]}...{sorted_training[-5:]}")
    
    # 3. Generar info sets usando la función de evaluación CORREGIDA
    logger.info("🧪 Generando info sets con función de evaluación CORREGIDA...")
    
    test_hands = [
        ([12, 12], False, 0, "AA_early"),     # Pocket Aces early position
        ([12, 12], False, 2, "AA_mid"),       # Pocket Aces middle position  
        ([12, 12], False, 5, "AA_late"),      # Pocket Aces late position
        ([11, 11], False, 2, "KK_mid"),       # Pocket Kings
        ([10, 9], True, 0, "JTs_early"),      # Suited connector early
        ([10, 9], True, 5, "JTs_late"),       # Suited connector late
        ([10, 9], False, 2, "JTo_mid"),       # Offsuit connector
        ([5, 0], False, 2, "72o_mid"),        # Trash hand
        ([8, 6], False, 3, "T8o_mid"),        # Medium hand
        ([12, 8], True, 4, "ATs_late"),       # Ace suited
    ]
    
    evaluation_info_sets = {}
    overlapping_info_sets = []
    
    logger.info("📋 Info sets generados por evaluación CORREGIDA:")
    logger.info("-" * 60)
    
    for hole_ranks, is_suited, position, name in test_hands:
        eval_info_set = compute_mock_info_set(hole_ranks, is_suited, position)
        evaluation_info_sets[name] = eval_info_set
        
        # ¿Este info set coincide con alguno del entrenamiento?
        is_trained = eval_info_set in training_info_sets
        status = "✅ ENTRENADO" if is_trained else "❌ NO ENTRENADO"
        
        logger.info(f"   {name:12s}: {eval_info_set:5d} | {status}")
        
        if is_trained:
            overlapping_info_sets.append((name, eval_info_set))
    
    # 4. Estadísticas de overlap
    overlap_count = len(overlapping_info_sets)
    total_tests = len(test_hands)
    overlap_percentage = (overlap_count / total_tests) * 100
    
    logger.info("\n📊 RESULTADOS DEL MAPEO:")
    logger.info("="*50)
    logger.info(f"   Total manos de test: {total_tests}")
    logger.info(f"   Info sets que coinciden: {overlap_count}")
    logger.info(f"   Porcentaje de overlap: {overlap_percentage:.1f}%")
    
    if overlap_count > 0:
        logger.info(f"\n🎯 MANOS CON INFO SETS ENTRENADOS:")
        for name, info_set in overlapping_info_sets:
            logger.info(f"   - {name}: info_set {info_set}")
    
    # 5. Test específico de entrenamiento corto para verificar estrategias
    if overlap_count >= 2:  # Si tenemos al menos 2 overlaps, hacer un test
        logger.info("\n🚀 EJECUTANDO TEST DE ENTRENAMIENTO CORTO...")
        
        # Entrenar muy pocas iteraciones
        config = TrainerConfig()
        config.batch_size = 64
        trainer = PokerTrainer(config)
        
        # 10 iteraciones de entrenamiento
        train_key = jax.random.PRNGKey(123)
        for i in range(10):
            iter_key = jax.random.fold_in(train_key, i)
            trainer.regrets, trainer.strategy = _jitted_train_step(
                trainer.regrets, trainer.strategy, iter_key
            )
            trainer.strategy.block_until_ready()  # Esperar computación
        
        logger.info("\n🧠 ANALIZANDO ESTRATEGIAS DE INFO SETS ENTRENADOS:")
        logger.info("-" * 55)
        
        for name, info_set in overlapping_info_sets[:3]:  # Solo los primeros 3
            strategy = trainer.strategy[info_set]
            fold_rate = float(strategy[0])
            aggression = float(jnp.sum(strategy[3:6]))  # BET/RAISE/ALL_IN
            
            logger.info(f"   {name:12s} (info_set {info_set}):")
            logger.info(f"      Fold rate: {fold_rate:.3f}")
            logger.info(f"      Aggression: {aggression:.3f}")
            logger.info(f"      Strategy: {[f'{float(x):.3f}' for x in strategy]}")
    
    # 6. Veredicto final
    logger.info("\n🏆 VEREDICTO FINAL:")
    logger.info("="*40)
    
    if overlap_percentage >= 50:
        logger.info("✅ CORRECCIÓN EXITOSA!")
        logger.info(f"   {overlap_percentage:.1f}% de los info sets de evaluación ahora coinciden")
        logger.info("   con info sets realmente entrenados.")
        logger.info("   El Poker IQ debería mejorar significativamente.")
        success = True
        
    elif overlap_percentage >= 20:
        logger.info("⚠️ CORRECCIÓN PARCIAL")
        logger.info(f"   {overlap_percentage:.1f}% de overlap - mejor que antes, pero")
        logger.info("   puede necesitar ajustes adicionales en los valores default.")
        success = True
        
    else:
        logger.info("❌ CORRECCIÓN INSUFICIENTE")
        logger.info(f"   Solo {overlap_percentage:.1f}% de overlap.")
        logger.info("   Los valores default necesitan más ajuste.")
        success = False
    
    return {
        'overlap_percentage': overlap_percentage,
        'overlapping_count': overlap_count,
        'total_tests': total_tests,
        'overlapping_info_sets': overlapping_info_sets,
        'success': success
    }

if __name__ == "__main__":
    results = test_info_set_mapping_fix()
    
    if results['success']:
        logger.info("\n🎉 TEST EXITOSO - La corrección mejora el mapeo de info sets!")
    else:
        logger.info("\n❌ TEST FALLIDO - Necesita más ajustes en compute_mock_info_set")
    
    logger.info(f"\nResumen: {results['overlapping_count']}/{results['total_tests']} manos tienen info sets entrenados") 