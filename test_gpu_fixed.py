#!/usr/bin/env python3
"""
Test rápido para verificar que el GPU trainer AHORA SÍ aprenda
con toda la lógica CFR completa aplicada
"""

import time
import jax
import jax.numpy as jnp
from trainer_mccfr_gpu_optimized import create_gpu_trainer

def test_gpu_learning_fixed():
    """Test que el GPU trainer ahora SÍ aprenda con la lógica CFR completa"""
    print("🔥 TEST GPU TRAINER - LÓGICA CFR COMPLETA")
    print("="*55)
    
    # Verificar GPU
    devices = jax.devices()
    print(f"🖥️  JAX devices: {devices}")
    gpu_detected = 'cuda' in str(devices).lower()
    print(f"🚀 GPU: {'✅ DETECTADA' if gpu_detected else '❌ NO DETECTADA'}")
    
    # Crear trainer
    trainer = create_gpu_trainer('standard')
    
    # Estado inicial
    print(f"\n📊 ESTADO INICIAL:")
    initial_strategy = trainer.strategy.copy()
    initial_std = float(jnp.std(initial_strategy))
    initial_mean = float(jnp.mean(initial_strategy))
    print(f"   - Strategy STD: {initial_std:.6f}")
    print(f"   - Strategy mean: {initial_mean:.6f}")
    print(f"   - ¿Uniforme? {'SÍ' if abs(initial_mean - 1/6) < 0.001 else 'NO'}")
    
    # Test corto pero efectivo
    print(f"\n🚀 ENTRENAMIENTO CON LÓGICA CFR COMPLETA (50 iteraciones)...")
    start_time = time.time()
    
    trainer.train(
        num_iterations=50,
        save_path="gpu_fixed_test",
        save_interval=50
    )
    
    training_time = time.time() - start_time
    speed = 50 / training_time
    
    # Verificar aprendizaje
    print(f"\n🧠 VERIFICACIÓN DE APRENDIZAJE:")
    trained_strategy = trainer.strategy.copy()
    trained_std = float(jnp.std(trained_strategy))
    trained_mean = float(jnp.mean(trained_strategy))
    
    # Cambio en estrategia
    strategy_change = float(jnp.mean(jnp.abs(trained_strategy - initial_strategy)))
    max_change = float(jnp.max(jnp.abs(trained_strategy - initial_strategy)))
    
    print(f"   - Strategy STD final: {trained_std:.6f}")
    print(f"   - Strategy mean final: {trained_mean:.6f}")
    print(f"   - Cambio promedio: {strategy_change:.6f}")
    print(f"   - Cambio máximo: {max_change:.6f}")
    print(f"   - STD ratio: {trained_std/max(initial_std, 1e-8):.2f}x")
    
    # Criterios de aprendizaje
    learning_detected = strategy_change > 1e-4
    significant_learning = strategy_change > 1e-3
    diversification = trained_std > initial_std * 2
    
    print(f"\n📈 ANÁLISIS DE APRENDIZAJE:")
    print(f"   - Cambio detectado: {'✅ SÍ' if learning_detected else '❌ NO'} (>{1e-4:.0e})")
    print(f"   - Cambio significativo: {'✅ SÍ' if significant_learning else '❌ NO'} (>{1e-3:.0e})")
    print(f"   - Diversificación: {'✅ SÍ' if diversification else '❌ NO'} (STD aumentó)")
    
    # Performance
    print(f"\n⚡ PERFORMANCE:")
    print(f"   - Tiempo: {training_time:.2f}s")
    print(f"   - Velocidad: {speed:.1f} it/s")
    print(f"   - Throughput: ~{50 * 256 * 50 / training_time:.0f} hands/s")
    
    # Análisis de regrets
    print(f"\n🎯 ANÁLISIS DE REGRETS:")
    non_zero_regrets = jnp.sum(trainer.regrets != 0.0)
    regret_std = float(jnp.std(trainer.regrets))
    max_regret = float(jnp.max(jnp.abs(trainer.regrets)))
    
    print(f"   - Regrets no-cero: {non_zero_regrets:,}")
    print(f"   - Regret STD: {regret_std:.6f}")
    print(f"   - Max regret: {max_regret:.2f}")
    
    # VEREDICTO FINAL
    all_criteria_met = learning_detected and diversification and non_zero_regrets > 1000
    
    print(f"\n🏆 VEREDICTO FINAL:")
    if all_criteria_met:
        print(f"   ✅ GPU TRAINER CORREGIDO - ¡AHORA SÍ APRENDE!")
        print(f"   🎉 Lógica CFR completa funcionando")
        print(f"   🚀 Velocidad: {speed:.1f} it/s")
        print(f"   💎 Listo para entrenamientos largos")
        verdict = "SUCCESS"
    elif learning_detected:
        print(f"   ⚠️  Aprendizaje detectado pero débil")
        print(f"   🔧 Necesita más iteraciones para convergencia")
        verdict = "PARTIAL"
    else:
        print(f"   ❌ AÚN NO APRENDE - Revisar lógica CFR")
        print(f"   🐛 Posible bug en implementación")
        verdict = "FAILED"
    
    return {
        'verdict': verdict,
        'learning_detected': learning_detected,
        'strategy_change': strategy_change,
        'speed': speed,
        'non_zero_regrets': int(non_zero_regrets)
    }

def test_comparison_with_working_version():
    """Comparar con la versión que sabemos que funciona"""
    print(f"\n🔬 COMPARACIÓN CON VERSIÓN QUE FUNCIONA")
    print("="*50)
    
    # Test rápido con trainer original (CPU)
    print(f"📚 Importando trainer original...")
    from poker_bot.core.trainer import PokerTrainer, TrainerConfig
    
    cpu_trainer = PokerTrainer(TrainerConfig())
    
    print(f"🔄 Test rápido CPU (20 iteraciones)...")
    start_time = time.time()
    
    cpu_trainer.train(
        num_iterations=20,
        save_path="cpu_comparison_test",
        save_interval=20,
        snapshot_iterations=[]
    )
    
    cpu_time = time.time() - start_time
    cpu_speed = 20 / cpu_time
    
    cpu_strategy_std = float(jnp.std(cpu_trainer.strategy))
    cpu_regrets = jnp.sum(cpu_trainer.regrets != 0.0)
    
    print(f"   - CPU velocidad: {cpu_speed:.2f} it/s")
    print(f"   - CPU strategy STD: {cpu_strategy_std:.6f}")
    print(f"   - CPU regrets: {cpu_regrets:,}")
    
    return {
        'cpu_speed': cpu_speed,
        'cpu_learning': cpu_strategy_std > 1e-6
    }

if __name__ == "__main__":
    print("🔥 INICIANDO TEST GPU TRAINER CORREGIDO")
    print("="*60)
    
    results = test_gpu_learning_fixed()
    
    if results['verdict'] == 'SUCCESS':
        print(f"\n🎉 ¡ÉXITO! GPU trainer ahora aprende correctamente")
        print(f"   Ejecuta: python launch_gpu_training.py standard")
        print(f"   Para entrenar modelo de 1000 iteraciones en ~3 minutos")
    else:
        print(f"\n🤔 Test completado - Revisar resultados arriba")
        
    # Comparación opcional
    print(f"\n❓ ¿Comparar con CPU trainer? (y/n): ", end="")
    try:
        if input().lower() == 'y':
            comparison = test_comparison_with_working_version()
            print(f"   GPU vs CPU: {results['speed']:.1f} vs {comparison['cpu_speed']:.1f} it/s")
            print(f"   Speedup: {results['speed']/comparison['cpu_speed']:.0f}x más rápido")
    except:
        pass 