#!/usr/bin/env python3
"""
EVALUADOR DE MODELOS: Evalúa modelos .pkl guardados usando evaluación directa.

Uso:
  python evaluate_model.py models/poker_bot_20250116_083000_final.pkl
  python evaluate_model.py models/poker_bot_20250116_083000_iter_800.pkl
"""

import sys
import logging
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def evaluate_saved_model(model_path):
    """
    Evalúa un modelo guardado usando evaluación directa
    """
    print(f"🧠 EVALUANDO MODELO: {model_path}")
    print("="*60)
    
    try:
        # Cargar modelo
        config = TrainerConfig()
        trainer = PokerTrainer(config)
        trainer.load_model(model_path)
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   - Iteración: {trainer.iteration}")
        print(f"   - Shape strategy: {trainer.strategy.shape}")
        print(f"   - Shape regrets: {trainer.regrets.shape}")
        print()
        
        # Evaluación directa
        from test_direct_poker_iq import evaluate_direct_poker_iq
        
        # Crear un evaluador temporal que use la estrategia cargada
        import jax.numpy as jnp
        
        print("🔍 Analizando info sets con regrets significativos...")
        regret_sums = jnp.sum(jnp.abs(trainer.regrets), axis=1)
        significant_threshold = 10.0
        significant_indices = jnp.where(regret_sums > significant_threshold)[0]
        significant_count = len(significant_indices)
        
        print(f"📊 Info sets entrenados: {significant_count}")
        
        if significant_count == 0:
            print("❌ No hay info sets entrenados en este modelo")
            return
        
        print(f"🎯 Ejecutando evaluación directa...")
        print()
        
        # Analizar estrategias top
        top_indices = significant_indices[jnp.argsort(-regret_sums[significant_indices])[:15]]
        
        strategies_data = []
        aggressive_count = 0
        balanced_count = 0
        tight_count = 0
        
        print("🏆 TOP 10 INFO SETS ENTRENADOS:")
        print("-" * 50)
        
        for i, info_set_idx in enumerate(top_indices[:10]):
            strategy = trainer.strategy[info_set_idx]
            regret_sum = float(regret_sums[info_set_idx])
            
            fold_rate = float(strategy[0])
            aggression = float(jnp.sum(strategy[3:6]))  # BET/RAISE/ALL_IN
            
            # Categorizar
            if aggression > 0.6:
                category = "🔥 AGGRESSIVE"
                aggressive_count += 1
            elif aggression > 0.4:
                category = "⚡ BALANCED"
                balanced_count += 1
            elif fold_rate > 0.4:
                category = "🛡️ TIGHT"
                tight_count += 1
            else:
                category = "📊 NEUTRAL"
            
            strategies_data.append({
                'fold_rate': fold_rate,
                'aggression': aggression,
                'category': category
            })
            
            print(f"{i+1:2d}. Info set {info_set_idx:5d}: {category}")
            print(f"    Fold: {fold_rate:.3f}, Aggression: {aggression:.3f}")
        
        print()
        
        # Calcular Poker IQ simplificado
        diversity_score = 0.0
        if aggressive_count >= 2: diversity_score += 10.0
        if balanced_count >= 3: diversity_score += 10.0
        if tight_count >= 1: diversity_score += 5.0
        
        rational_score = 0.0
        non_folding = len([s for s in strategies_data if s['fold_rate'] < 0.8])
        if non_folding >= 8: rational_score += 20.0
        elif non_folding >= 5: rational_score += 15.0
        
        learning_score = 0.0
        if significant_count >= 50: learning_score += 20.0
        elif significant_count >= 30: learning_score += 15.0
        elif significant_count >= 10: learning_score += 10.0
        
        complexity_score = 0.0
        if aggressive_count + balanced_count + tight_count >= 8:
            complexity_score += 15.0
        
        # Bonus por iteraciones
        iteration_bonus = min(20.0, trainer.iteration / 50.0)  # Max 20 puntos, 1 por cada 50 iter
        
        total_score = diversity_score + rational_score + learning_score + complexity_score + iteration_bonus
        
        print(f"🎯 POKER IQ EVALUACIÓN:")
        print(f"   📊 Diversidad: {diversity_score:.1f}/25")
        print(f"   🧠 Racionalidad: {rational_score:.1f}/25") 
        print(f"   📈 Aprendizaje: {learning_score:.1f}/25")
        print(f"   🎭 Complejidad: {complexity_score:.1f}/25")
        print(f"   ⏱️ Bonus iteraciones: {iteration_bonus:.1f}/20")
        print(f"   🏆 POKER IQ TOTAL: {total_score:.1f}/120")
        
        # Veredicto
        if total_score >= 90:
            verdict = "🏆 EXCELENTE - Nivel profesional"
        elif total_score >= 70:
            verdict = "🥇 BUENO - Nivel competente"
        elif total_score >= 50:
            verdict = "🥈 MODERADO - Progreso sólido"
        elif total_score >= 30:
            verdict = "🥉 BÁSICO - Aprendizaje inicial"
        else:
            verdict = "❌ INSUFICIENTE"
        
        print(f"   📋 Veredicto: {verdict}")
        
        # Estadísticas de categorías
        print(f"\n📊 DISTRIBUCIÓN DE ESTRATEGIAS:")
        print(f"   🔥 Aggressive: {aggressive_count}")
        print(f"   ⚡ Balanced: {balanced_count}")
        print(f"   🛡️ Tight: {tight_count}")
        print(f"   📊 Total categorizado: {aggressive_count + balanced_count + tight_count}")
        
        return {
            'poker_iq': total_score,
            'significant_info_sets': significant_count,
            'iteration': trainer.iteration,
            'verdict': verdict
        }
        
    except Exception as e:
        print(f"❌ Error evaluando modelo: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❌ Uso: python evaluate_model.py <ruta_al_modelo.pkl>")
        print()
        print("Ejemplos:")
        print("  python evaluate_model.py models/poker_bot_20250116_083000_final.pkl")
        print("  python evaluate_model.py models/poker_bot_20250116_083000_iter_800.pkl")
        sys.exit(1)
    
    model_path = sys.argv[1]
    results = evaluate_saved_model(model_path)
    
    if results:
        print(f"\n✅ Evaluación completada")
        print(f"   Poker IQ: {results['poker_iq']:.1f}/120")
        print(f"   Iteración: {results['iteration']}")
    else:
        print(f"\n❌ Error evaluando {model_path}") 