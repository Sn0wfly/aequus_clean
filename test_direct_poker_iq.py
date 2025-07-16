#!/usr/bin/env python3
"""
TEST DIRECTO: Evalúa Poker IQ usando directamente los info sets entrenados,
sin depender de compute_mock_info_set que puede estar mal mapeado.
"""

import logging
import jax
import jax.numpy as jnp
import numpy as np
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, 
    compute_advanced_info_set, unified_batch_simulation, _jitted_train_step
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def evaluate_direct_poker_iq():
    """
    Evaluación DIRECTA de Poker IQ usando info sets realmente entrenados
    """
    logger.info("🧠 EVALUACIÓN DIRECTA DE POKER IQ")
    logger.info("="*50)
    
    # 1. Entrenar un modelo corto
    logger.info("🚀 Entrenando modelo corto (50 iteraciones)...")
    config = TrainerConfig()
    config.batch_size = 128
    trainer = PokerTrainer(config)
    
    # Entrenamiento breve pero significativo
    train_key = jax.random.PRNGKey(42)
    for i in range(50):
        iter_key = jax.random.fold_in(train_key, i)
        trainer.regrets, trainer.strategy = _jitted_train_step(
            trainer.regrets, trainer.strategy, iter_key
        )
        if i % 10 == 0:
            trainer.strategy.block_until_ready()
            logger.info(f"   Iteración {i}/50 completada")
    
    trainer.strategy.block_until_ready()
    logger.info("✅ Entrenamiento completado")
    
    # 2. Extraer info sets con regrets significativos
    logger.info("\n🔍 Analizando info sets con regrets significativos...")
    regret_sums = jnp.sum(jnp.abs(trainer.regrets), axis=1)
    significant_threshold = 10.0
    
    significant_indices = jnp.where(regret_sums > significant_threshold)[0]
    significant_count = len(significant_indices)
    
    logger.info(f"📊 Info sets con regrets > {significant_threshold}: {significant_count}")
    
    if significant_count == 0:
        logger.error("❌ No hay info sets con regrets significativos")
        return {"error": "No trained info sets found"}
    
    # 3. Analizar las estrategias de los info sets entrenados
    logger.info("\n🎯 ANALIZANDO ESTRATEGIAS DE INFO SETS ENTRENADOS:")
    logger.info("-" * 60)
    
    # Extraer datos de los top info sets
    top_indices = significant_indices[jnp.argsort(-regret_sums[significant_indices])[:20]]
    
    strategies_data = []
    
    for i, info_set_idx in enumerate(top_indices):
        strategy = trainer.strategy[info_set_idx]
        regret_sum = float(regret_sums[info_set_idx])
        
        fold_rate = float(strategy[0])
        aggression = float(jnp.sum(strategy[3:6]))  # BET/RAISE/ALL_IN
        
        # Categorizar estrategia
        if aggression > 0.6:
            category = "🔥 AGGRESSIVE"
        elif aggression > 0.4:
            category = "⚡ BALANCED"
        elif fold_rate > 0.4:
            category = "🛡️ TIGHT"
        else:
            category = "📊 NEUTRAL"
        
        strategies_data.append({
            'info_set': int(info_set_idx),
            'regret_sum': regret_sum,
            'fold_rate': fold_rate,
            'aggression': aggression,
            'category': category,
            'strategy': strategy
        })
        
        if i < 10:  # Mostrar solo los primeros 10
            logger.info(f"{i+1:2d}. Info set {info_set_idx:5d}: regrets={regret_sum:8.1f} | {category}")
            logger.info(f"    Fold: {fold_rate:.3f}, Aggression: {aggression:.3f}")
    
    # 4. Evaluar conceptos de poker usando datos reales
    logger.info(f"\n🧪 EVALUACIÓN DE CONCEPTOS DE POKER:")
    logger.info("=" * 45)
    
    # Test 1: Diversidad de estrategias
    def test_strategy_diversity():
        """¿Hay suficiente diversidad en las estrategias?"""
        aggressive_count = len([s for s in strategies_data if s['aggression'] > 0.6])
        tight_count = len([s for s in strategies_data if s['fold_rate'] > 0.4])
        
        diversity_score = 0.0
        
        # Premio por tener estrategias agresivas (manos fuertes)
        if aggressive_count >= 3:
            diversity_score += 10.0
        elif aggressive_count >= 1:
            diversity_score += 5.0
            
        # Premio por tener estrategias tight (manos débiles)
        if tight_count >= 2:
            diversity_score += 10.0
        elif tight_count >= 1:
            diversity_score += 5.0
            
        # Premio por variedad general
        categories = set([s['category'] for s in strategies_data])
        if len(categories) >= 3:
            diversity_score += 5.0
        
        return min(25.0, diversity_score)
    
    # Test 2: Comportamiento racional
    def test_rational_behavior():
        """¿Los info sets actúan de manera racional?"""
        rational_score = 0.0
        
        # Verificar que hay info sets que no siempre folden
        non_folding = [s for s in strategies_data if s['fold_rate'] < 0.8]
        if len(non_folding) >= 10:
            rational_score += 15.0
        elif len(non_folding) >= 5:
            rational_score += 10.0
        
        # Verificar que hay variedad en niveles de agresión
        aggression_levels = [s['aggression'] for s in strategies_data]
        aggression_std = np.std(aggression_levels)
        
        if aggression_std > 0.2:
            rational_score += 10.0
        elif aggression_std > 0.1:
            rational_score += 5.0
        
        return min(25.0, rational_score)
    
    # Test 3: Aprendizaje efectivo
    def test_learning_effectiveness():
        """¿El sistema está aprendiendo efectivamente?"""
        learning_score = 0.0
        
        # Verificar que hay regrets altos (indica aprendizaje activo)
        high_regret_count = len([s for s in strategies_data if s['regret_sum'] > 100])
        if high_regret_count >= 5:
            learning_score += 15.0
        elif high_regret_count >= 2:
            learning_score += 10.0
        
        # Verificar que las estrategias no son uniformes
        non_uniform_count = 0
        for s in strategies_data:
            strategy_std = float(jnp.std(s['strategy']))
            if strategy_std > 0.05:  # No uniforme
                non_uniform_count += 1
        
        if non_uniform_count >= 15:
            learning_score += 10.0
        elif non_uniform_count >= 10:
            learning_score += 5.0
        
        return min(25.0, learning_score)
    
    # Test 4: Complejidad estratégica  
    def test_strategic_complexity():
        """¿Las estrategias muestran complejidad apropiada?"""
        complexity_score = 0.0
        
        # Verificar uso de todas las acciones
        all_actions_used = True
        for action_idx in range(6):
            action_usage = np.mean([float(s['strategy'][action_idx]) for s in strategies_data])
            if action_usage < 0.05:  # Acción casi nunca usada
                all_actions_used = False
                break
        
        if all_actions_used:
            complexity_score += 15.0
        
        # Verificar que hay estrategias mixtas (no determinísticas)
        mixed_strategies = 0
        for s in strategies_data:
            max_prob = float(jnp.max(s['strategy']))
            if max_prob < 0.9:  # No determinística
                mixed_strategies += 1
        
        if mixed_strategies >= 15:
            complexity_score += 10.0
        elif mixed_strategies >= 10:
            complexity_score += 5.0
        
        return min(25.0, complexity_score)
    
    # Ejecutar todos los tests
    diversity_score = test_strategy_diversity()
    rational_score = test_rational_behavior()
    learning_score = test_learning_effectiveness()
    complexity_score = test_strategic_complexity()
    
    total_score = diversity_score + rational_score + learning_score + complexity_score
    
    logger.info(f"🎯 RESULTADOS DE EVALUACIÓN:")
    logger.info(f"   📊 Diversidad de estrategias: {diversity_score:.1f}/25")
    logger.info(f"   🧠 Comportamiento racional: {rational_score:.1f}/25") 
    logger.info(f"   📈 Efectividad de aprendizaje: {learning_score:.1f}/25")
    logger.info(f"   🎭 Complejidad estratégica: {complexity_score:.1f}/25")
    logger.info(f"   🏆 POKER IQ DIRECTO: {total_score:.1f}/100")
    
    # Interpretación del resultado
    if total_score >= 80:
        verdict = "🏆 EXCELENTE - Nivel profesional"
    elif total_score >= 60:
        verdict = "🥇 BUENO - Aprendizaje sólido"
    elif total_score >= 40:
        verdict = "🥈 MODERADO - Progreso visible"
    elif total_score >= 20:
        verdict = "🥉 BÁSICO - Aprendizaje inicial"
    else:
        verdict = "❌ INSUFICIENTE - Necesita más entrenamiento"
    
    logger.info(f"   📋 Veredicto: {verdict}")
    
    return {
        'total_score': total_score,
        'diversity_score': diversity_score,
        'rational_score': rational_score,
        'learning_score': learning_score,
        'complexity_score': complexity_score,
        'significant_info_sets': significant_count,
        'verdict': verdict
    }

if __name__ == "__main__":
    logger.info("🎯 INICIANDO EVALUACIÓN DIRECTA DE POKER IQ")
    logger.info("   (Sin depender de compute_mock_info_set)")
    
    results = evaluate_direct_poker_iq()
    
    if 'error' not in results:
        logger.info(f"\n🎉 EVALUACIÓN COMPLETADA")
        logger.info(f"   Poker IQ Directo: {results['total_score']:.1f}/100")
        logger.info(f"   Info sets entrenados: {results['significant_info_sets']}")
        logger.info(f"   {results['verdict']}")
    else:
        logger.error(f"\n❌ Error en evaluación: {results['error']}") 