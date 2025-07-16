#!/usr/bin/env python3
"""
Debug específico para analizar los regrets y ver por qué el regret matching no funciona.
"""

import logging
import numpy as np
import jax.numpy as jnp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🔍 DEBUG DE REGRETS - Análisis del Regret Matching")
    logger.info("="*60)
    
    from poker_bot.core.trainer import PokerTrainer, TrainerConfig
    
    # Crear trainer y entrenar pocas iteraciones para debug
    config = TrainerConfig()
    config.batch_size = 32  # Más pequeño para debug
    trainer = PokerTrainer(config)
    
    # Entrenar solo 5 iteraciones para análisis rápido
    logger.info("🚀 Entrenando 5 iteraciones para debug...")
    trainer.train(
        num_iterations=5,
        save_path="models/debug_regrets",
        save_interval=5,
        snapshot_iterations=[5]
    )
    
    # Analizar regrets específicos
    logger.info("\n🔍 ANÁLISIS DE REGRETS ESPECÍFICOS")
    logger.info("="*50)
    
    # Info sets de test
    aa_info_set = 616  # AA middle position
    trash_info_set = 3016  # 72o middle position
    
    # Obtener regrets para estas manos específicas
    aa_regrets = trainer.regrets[aa_info_set]
    trash_regrets = trainer.regrets[trash_info_set]
    
    aa_strategy = trainer.strategy[aa_info_set]
    trash_strategy = trainer.strategy[trash_info_set]
    
    logger.info(f"📊 AA (info_set {aa_info_set}):")
    logger.info(f"   Regrets: {[float(x) for x in aa_regrets]}")
    logger.info(f"   Strategy: {[float(x) for x in aa_strategy]}")
    logger.info(f"   Regret sum: {float(jnp.sum(aa_regrets))}")
    logger.info(f"   Positive regrets: {[float(x) for x in jnp.maximum(aa_regrets, 0.0)]}")
    logger.info(f"   Positive sum: {float(jnp.sum(jnp.maximum(aa_regrets, 0.0)))}")
    
    logger.info(f"\n📊 72o (info_set {trash_info_set}):")
    logger.info(f"   Regrets: {[float(x) for x in trash_regrets]}")
    logger.info(f"   Strategy: {[float(x) for x in trash_strategy]}")
    logger.info(f"   Regret sum: {float(jnp.sum(trash_regrets))}")
    logger.info(f"   Positive regrets: {[float(x) for x in jnp.maximum(trash_regrets, 0.0)]}")
    logger.info(f"   Positive sum: {float(jnp.sum(jnp.maximum(trash_regrets, 0.0)))}")
    
    # Análisis estadístico global
    logger.info(f"\n📈 ANÁLISIS ESTADÍSTICO GLOBAL:")
    logger.info(f"   Total regrets shape: {trainer.regrets.shape}")
    logger.info(f"   Total strategy shape: {trainer.strategy.shape}")
    
    # Estadísticas de regrets
    regret_stats = {
        'min': float(jnp.min(trainer.regrets)),
        'max': float(jnp.max(trainer.regrets)),
        'mean': float(jnp.mean(trainer.regrets)),
        'std': float(jnp.std(trainer.regrets)),
        'nonzero_count': int(jnp.sum(trainer.regrets != 0.0))
    }
    
    logger.info(f"   Regret min: {regret_stats['min']:.6f}")
    logger.info(f"   Regret max: {regret_stats['max']:.6f}")
    logger.info(f"   Regret mean: {regret_stats['mean']:.6f}")
    logger.info(f"   Regret std: {regret_stats['std']:.6f}")
    logger.info(f"   Non-zero regrets: {regret_stats['nonzero_count']}/{trainer.regrets.size}")
    
    # Análisis de positive regrets
    positive_regrets = jnp.maximum(trainer.regrets, 0.0)
    positive_sums = jnp.sum(positive_regrets, axis=1)
    
    # Contar cuántos info sets tienen positive regrets
    valid_info_sets = jnp.sum(positive_sums > 1e-6)
    
    logger.info(f"\n📊 ANÁLISIS DE POSITIVE REGRETS:")
    logger.info(f"   Info sets con positive regrets > 1e-6: {int(valid_info_sets)}/{config.max_info_sets}")
    logger.info(f"   Porcentaje: {float(valid_info_sets/config.max_info_sets)*100:.2f}%")
    
    # Encontrar info sets con más regrets
    top_indices = jnp.argsort(positive_sums)[-10:]  # Top 10
    
    logger.info(f"\n🏆 TOP 10 INFO SETS CON MÁS POSITIVE REGRETS:")
    for i, idx in enumerate(reversed(top_indices)):
        regret_sum = float(positive_sums[idx])
        strategy = trainer.strategy[idx]
        logger.info(f"   {i+1}. Info set {int(idx)}: sum={regret_sum:.6f}")
        logger.info(f"      Strategy: {[f'{float(x):.3f}' for x in strategy]}")
    
    # Test manual de regret matching
    logger.info(f"\n🧪 TEST MANUAL DE REGRET MATCHING:")
    
    # Simular el proceso manualmente para AA
    aa_positive = jnp.maximum(aa_regrets, 0.0)
    aa_sum = jnp.sum(aa_positive)
    
    logger.info(f"   AA positive regrets: {[float(x) for x in aa_positive]}")
    logger.info(f"   AA positive sum: {float(aa_sum)}")
    
    if aa_sum > 1e-6:
        manual_strategy = aa_positive / aa_sum
        logger.info(f"   Manual strategy (should match): {[float(x) for x in manual_strategy]}")
        logger.info(f"   Actual strategy: {[float(x) for x in aa_strategy]}")
        
        # Verificar si coinciden
        matches = jnp.allclose(manual_strategy, aa_strategy, atol=1e-6)
        logger.info(f"   ¿Coinciden? {matches}")
    else:
        logger.info(f"   AA sum <= 1e-6, debería usar estrategia uniforme")
        uniform = jnp.ones(6) / 6
        matches = jnp.allclose(uniform, aa_strategy, atol=1e-6)
        logger.info(f"   ¿Es uniforme como esperado? {matches}")
    
    logger.info(f"\n✅ Debug de regrets completado")

if __name__ == "__main__":
    main() 