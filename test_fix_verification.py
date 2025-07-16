#!/usr/bin/env python3
"""
VERIFICACIÓN RÁPIDA: ¿Se arregló el bug de CFR?
Debe mostrar muchos más info sets being trained ahora.
"""

import jax
import jax.numpy as jnp
from poker_bot.core.trainer import (
    PokerTrainer, TrainerConfig, _jitted_train_step,
    unified_batch_simulation, compute_advanced_info_set
)

def test_bug_fix():
    """Test rápido para verificar que el bug está arreglado"""
    print("🔧 VERIFICANDO FIX: Bug de acumulación de regrets")
    print("="*50)
    
    config = TrainerConfig()
    config.batch_size = 16  # Un poco más grande para ver el efecto
    
    # Estado inicial
    regrets = jnp.zeros((config.max_info_sets, config.num_actions), dtype=jnp.float32)
    strategy = jnp.ones((config.max_info_sets, config.num_actions), dtype=jnp.float32) / config.num_actions
    
    key = jax.random.PRNGKey(42)
    
    print(f"📊 Configuración:")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Máximo teórico info sets visitados: {config.batch_size * 6} (batch × jugadores)")
    
    # Ejecutar UN paso de CFR
    print(f"\n🔄 Ejecutando UN paso de CFR corregido...")
    
    new_regrets, new_strategy = _jitted_train_step(regrets, strategy, key)
    
    # Analizar cuántos info sets fueron entrenados
    positive_regrets = jnp.maximum(new_regrets, 0.0)
    regret_sums = jnp.sum(positive_regrets, axis=1, keepdims=True)
    
    info_sets_trained = jnp.sum(regret_sums > 1e-6)
    theoretical_max = config.batch_size * 6  # 6 jugadores por juego
    
    print(f"\n📈 RESULTADOS:")
    print(f"   - Info sets entrenados: {info_sets_trained}")
    print(f"   - Teórico máximo: {theoretical_max}")
    print(f"   - Cobertura: {float(info_sets_trained)/theoretical_max:.2%}")
    print(f"   - Máximo regret sum: {jnp.max(regret_sums):.3f}")
    
    # Verificar si el fix funcionó
    expected_minimum = theoretical_max * 0.5  # Al menos 50% de cobertura esperada
    
    if info_sets_trained >= expected_minimum:
        print(f"\n✅ ¡FIX EXITOSO!")
        print(f"   - Antes: ~18 info sets por batch")
        print(f"   - Ahora: {info_sets_trained} info sets por batch")
        print(f"   - Mejora: ~{info_sets_trained/18:.1f}x más info sets entrenados")
        return True
    else:
        print(f"\n❌ Fix no funcionó completamente")
        print(f"   - Esperado: >= {expected_minimum}")
        print(f"   - Actual: {info_sets_trained}")
        return False

def test_aa_vs_72o_learning():
    """Test específico: ¿Pueden AA y 72o aprender estrategias diferentes ahora?"""
    print(f"\n🃏 TEST ESPECÍFICO: AA vs 72o")
    print("="*40)
    
    config = TrainerConfig()
    config.batch_size = 64  # Batch más grande para mayor probabilidad de entrenar estas manos
    
    trainer = PokerTrainer(config)
    
    # Entrenar por unas pocas iteraciones
    print("⏳ Entrenando por 10 iteraciones rápidas...")
    
    for i in range(10):
        key = jax.random.PRNGKey(i + 100)
        trainer.regrets, trainer.strategy = _jitted_train_step(
            trainer.regrets, trainer.strategy, key
        )
    
    # Verificar estrategias de AA vs 72o
    from poker_bot.core.trainer import compute_mock_info_set
    
    aa_info = compute_mock_info_set([12, 12], False, 2)  # AA middle position
    trash_info = compute_mock_info_set([5, 0], False, 2)  # 72o middle position
    
    if aa_info < config.max_info_sets and trash_info < config.max_info_sets:
        aa_strategy = trainer.strategy[aa_info]
        trash_strategy = trainer.strategy[trash_info]
        
        aa_aggression = float(jnp.sum(aa_strategy[3:6]))  # BET/RAISE/ALLIN
        trash_aggression = float(jnp.sum(trash_strategy[3:6]))
        
        aa_fold = float(aa_strategy[0])
        trash_fold = float(trash_strategy[0])
        
        print(f"🎯 RESULTADOS:")
        print(f"   AA strategy: FOLD={aa_fold:.3f}, AGG={aa_aggression:.3f}")
        print(f"   72o strategy: FOLD={trash_fold:.3f}, AGG={trash_aggression:.3f}")
        
        strategies_different = abs(aa_aggression - trash_aggression) > 0.01
        
        if strategies_different:
            print(f"\n✅ ¡ÉXITO! AA y 72o tienen estrategias diferentes")
            print(f"   - Diferencia en agresión: {aa_aggression - trash_aggression:+.3f}")
            return True
        else:
            print(f"\n⚠️ Aún muy similares, pero esto es normal para 10 iteraciones")
            print(f"   - Diferencia en agresión: {aa_aggression - trash_aggression:+.3f}")
            return False
    else:
        print(f"❌ Error: Info sets fuera de rango")
        return False

def main():
    """Ejecutar verificaciones"""
    print("🚨 VERIFICACIÓN DE FIX CFR")
    print("="*60)
    
    # Test 1: Verificar que más info sets se entrenan
    try:
        fix_works = test_bug_fix()
    except Exception as e:
        print(f"❌ Error en test_bug_fix: {e}")
        fix_works = False
    
    # Test 2: Verificar que AA vs 72o pueden diferenciarse
    if fix_works:
        try:
            aa_test = test_aa_vs_72o_learning()
        except Exception as e:
            print(f"❌ Error en test_aa_vs_72o: {e}")
            aa_test = False
    else:
        aa_test = False
    
    print(f"\n🎯 RESUMEN:")
    print(f"   - Fix de info sets: {'✅' if fix_works else '❌'}")
    print(f"   - AA vs 72o test: {'✅' if aa_test else '⚠️ '}")
    
    if fix_works:
        print(f"\n🎉 ¡El fix está funcionando! Puedes proceder con el entrenamiento completo.")
    else:
        print(f"\n🔧 El fix necesita más trabajo.")

if __name__ == "__main__":
    main() 