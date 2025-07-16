#!/usr/bin/env python3
"""
Test rápido para verificar que la diversidad de historiales mejoró
después de los cambios en unified_batch_simulation.
"""

import jax
import jax.numpy as jnp
import logging
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

try:
    from poker_bot.core.trainer import unified_batch_simulation
    logger.info("✅ Módulos importados correctamente")
except ImportError as e:
    logger.error(f"❌ Error importando módulos: {e}")
    sys.exit(1)

def test_historia_diversity():
    """
    Test específico para verificar que la diversidad de historiales mejoró.
    """
    logger.info("\n🔍 TESTING DIVERSIDAD DE HISTORIALES POST-FIX")
    logger.info("="*60)
    
    # Generar datos de test
    key = jax.random.PRNGKey(12345)
    test_keys = jax.random.split(key, 64)  # Batch de 64 juegos
    
    logger.info("🎯 Generando historiales de test...")
    payoffs, histories, game_results = unified_batch_simulation(test_keys)
    
    logger.info(f"📊 ANÁLISIS DE HISTORIALES:")
    logger.info(f"   - Shape: {histories.shape}")
    
    # Analizar valores únicos
    unique_values = jnp.unique(histories)
    logger.info(f"   - Valores únicos: {unique_values}")
    
    # Contar acciones válidas vs padding
    valid_actions_mask = histories >= 0
    valid_actions_only = histories[valid_actions_mask]
    
    total_positions = histories.size
    valid_positions = jnp.sum(valid_actions_mask)
    padding_positions = jnp.sum(histories == -1)
    
    logger.info(f"   - Total posiciones: {total_positions}")
    logger.info(f"   - Posiciones válidas: {valid_positions}")
    logger.info(f"   - Posiciones padding (-1): {padding_positions}")
    logger.info(f"   - Ratio padding: {padding_positions/total_positions:.3f}")
    
    # Calcular diversidad CORRECTA (solo acciones válidas)
    if len(valid_actions_only) > 0:
        unique_valid_actions = len(jnp.unique(valid_actions_only))
        total_valid_actions = len(valid_actions_only)
        history_diversity = unique_valid_actions / max(1, total_valid_actions)
        
        logger.info(f"\n🎯 CÁLCULO DE DIVERSIDAD:")
        logger.info(f"   - Acciones válidas únicas: {unique_valid_actions}")
        logger.info(f"   - Total acciones válidas: {total_valid_actions}")
        logger.info(f"   - Diversidad calculada: {history_diversity:.4f}")
        
        # Análisis de distribución de acciones
        logger.info(f"\n📈 DISTRIBUCIÓN DE ACCIONES:")
        for action in range(6):
            count = jnp.sum(valid_actions_only == action)
            percentage = count / total_valid_actions * 100
            logger.info(f"   - Acción {action}: {count:4d} veces ({percentage:5.1f}%)")
        
        # Threshold test
        threshold = 0.02  # 2% como en el código
        
        if history_diversity > threshold:
            logger.info(f"\n✅ TEST PASSED: Diversidad {history_diversity:.4f} > threshold {threshold}")
            logger.info("   Los historiales tienen suficiente diversidad")
            return True
        else:
            logger.error(f"\n❌ TEST FAILED: Diversidad {history_diversity:.4f} <= threshold {threshold}")
            logger.error("   Los historiales siguen siendo demasiado uniformes")
            return False
    else:
        logger.error("\n❌ No hay acciones válidas en los historiales")
        return False

def test_variabilidad_longitud():
    """
    Test para verificar que los juegos tienen longitudes variables.
    """
    logger.info("\n🔍 TESTING VARIABILIDAD EN LONGITUD DE JUEGOS")
    logger.info("="*60)
    
    key = jax.random.PRNGKey(54321)
    test_keys = jax.random.split(key, 32)
    
    payoffs, histories, game_results = unified_batch_simulation(test_keys)
    
    # Calcular longitud de cada juego (número de acciones válidas)
    game_lengths = []
    for game_idx in range(histories.shape[0]):
        game_history = histories[game_idx]
        valid_actions = jnp.sum(game_history >= 0)
        game_lengths.append(int(valid_actions))
    
    min_length = min(game_lengths)
    max_length = max(game_lengths)
    avg_length = sum(game_lengths) / len(game_lengths)
    unique_lengths = len(set(game_lengths))
    
    logger.info(f"📊 ESTADÍSTICAS DE LONGITUD:")
    logger.info(f"   - Longitud mínima: {min_length}")
    logger.info(f"   - Longitud máxima: {max_length}")
    logger.info(f"   - Longitud promedio: {avg_length:.1f}")
    logger.info(f"   - Longitudes únicas: {unique_lengths}")
    logger.info(f"   - Variabilidad: {max_length - min_length}")
    
    # Test de variabilidad
    if unique_lengths >= 5 and (max_length - min_length) >= 10:
        logger.info(f"\n✅ VARIABILIDAD BUENA: {unique_lengths} longitudes diferentes")
        return True
    else:
        logger.warning(f"\n⚠️  VARIABILIDAD BAJA: Solo {unique_lengths} longitudes diferentes")
        return False

if __name__ == "__main__":
    logger.info("🚀 INICIANDO TEST DE DIVERSIDAD POST-FIX")
    
    try:
        # Test 1: Diversidad de historiales
        diversity_pass = test_historia_diversity()
        
        # Test 2: Variabilidad de longitud
        variability_pass = test_variabilidad_longitud()
        
        # Resultado final
        logger.info("\n" + "="*60)
        logger.info("🏁 RESULTADO FINAL:")
        logger.info(f"   - Test diversidad: {'✅ PASS' if diversity_pass else '❌ FAIL'}")
        logger.info(f"   - Test variabilidad: {'✅ PASS' if variability_pass else '❌ FAIL'}")
        
        if diversity_pass and variability_pass:
            logger.info("\n🎉 TODOS LOS TESTS PASARON - Fix aplicado exitosamente")
            logger.info("   El sistema ahora genera historiales con suficiente diversidad")
        elif diversity_pass:
            logger.info("\n🥈 DIVERSIDAD MEJORADA - Pero falta variabilidad en longitudes")
        else:
            logger.error("\n❌ FIX NO COMPLETO - Aún hay problemas de diversidad")
            logger.error("   Necesitan más ajustes en unified_batch_simulation")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n💥 ERROR DURANTE TEST: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1) 