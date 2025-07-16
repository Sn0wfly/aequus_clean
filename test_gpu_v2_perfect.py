#!/usr/bin/env python3
"""
Test para GPU Trainer V2 - Lo mejor de ambos mundos
✅ CFR Correcto + ✅ GPU Máximo
"""

import time
import jax
import jax.numpy as jnp
from trainer_mccfr_gpu_optimized_v2 import create_gpu_trainer_v2

def test_gpu_v2_perfect():
    """Test que debe tener aprendizaje real + velocidad alta"""
    print("🚀 TEST GPU TRAINER V2 - PERFECTO")
    print("="*50)
    print("CFR Correcto + GPU Máximo")
    
    # Verificar GPU
    devices = jax.devices()
    print(f"\n🖥️  JAX devices: {devices}")
    gpu_detected = 'cuda' in str(devices).lower()
    print(f"🚀 GPU: {'✅ DETECTADA' if gpu_detected else '❌ NO DETECTADA'}")
    
    # Crear trainer V2
    trainer = create_gpu_trainer_v2('standard')
    
    # Estado inicial
    print(f"\n📊 ESTADO INICIAL:")
    initial_strategy = trainer.strategy.copy()
    initial_std = float(jnp.std(initial_strategy))
    initial_mean = float(jnp.mean(initial_strategy))
    print(f"   - Strategy STD: {initial_std:.6f}")
    print(f"   - Strategy mean: {initial_mean:.6f}")
    print(f"   - ¿Uniforme? {'SÍ' if abs(initial_mean - 1/6) < 0.001 else 'NO'}")
    
    # Test inicial corto para velocidad
    print(f"\n🚀 TEST VELOCIDAD (20 iteraciones)...")
    start_time = time.time()
    
    trainer.train(
        num_iterations=20,
        save_path="gpu_v2_speed_test",
        save_interval=20
    )
    
    speed_time = time.time() - start_time
    speed = 20 / speed_time
    
    print(f"\n⚡ ANÁLISIS DE VELOCIDAD:")
    print(f"   - Tiempo: {speed_time:.2f}s")
    print(f"   - Velocidad: {speed:.1f} it/s")
    print(f"   - Throughput: ~{20 * 256 * 50 / speed_time:.0f} hands/s")
    
    # Verificar si es veloz (>50 it/s esperado para V2)
    is_fast = speed > 50
    print(f"   - ¿Rápido? {'✅ SÍ' if is_fast else '❌ NO'} (>{50} it/s)")
    
    # Test de aprendizaje (más iteraciones)
    print(f"\n🧠 TEST APRENDIZAJE (50 iteraciones adicionales)...")
    
    # Crear nuevo trainer para test limpio de aprendizaje
    learning_trainer = create_gpu_trainer_v2('standard')
    initial_strategy_learning = learning_trainer.strategy.copy()
    
    start_learning = time.time()
    
    learning_trainer.train(
        num_iterations=50,
        save_path="gpu_v2_learning_test", 
        save_interval=50
    )
    
    learning_time = time.time() - start_learning
    learning_speed = 50 / learning_time
    
    # Verificar aprendizaje
    trained_strategy = learning_trainer.strategy.copy()
    strategy_change = float(jnp.mean(jnp.abs(trained_strategy - initial_strategy_learning)))
    strategy_std = float(jnp.std(trained_strategy))
    max_change = float(jnp.max(jnp.abs(trained_strategy - initial_strategy_learning)))
    
    print(f"\n🧠 ANÁLISIS DE APRENDIZAJE:")
    print(f"   - Tiempo: {learning_time:.2f}s")
    print(f"   - Velocidad: {learning_speed:.1f} it/s")
    print(f"   - Strategy STD final: {strategy_std:.6f}")
    print(f"   - Cambio promedio: {strategy_change:.6f}")
    print(f"   - Cambio máximo: {max_change:.6f}")
    
    # Análisis de regrets
    non_zero_regrets = jnp.sum(learning_trainer.regrets != 0.0)
    regret_std = float(jnp.std(learning_trainer.regrets))
    max_regret = float(jnp.max(jnp.abs(learning_trainer.regrets)))
    
    print(f"\n🎯 ANÁLISIS DE REGRETS:")
    print(f"   - Regrets no-cero: {non_zero_regrets:,}")
    print(f"   - Regret STD: {regret_std:.6f}")
    print(f"   - Max regret: {max_regret:.2f}")
    
    # Criterios de éxito
    learning_detected = strategy_change > 1e-4
    significant_learning = strategy_change > 1e-3
    diversification = strategy_std > 1e-5
    substantial_regrets = non_zero_regrets > 1000
    
    print(f"\n📈 CRITERIOS DE APRENDIZAJE:")
    print(f"   - Cambio detectado: {'✅ SÍ' if learning_detected else '❌ NO'} (>{1e-4:.0e})")
    print(f"   - Cambio significativo: {'✅ SÍ' if significant_learning else '❌ NO'} (>{1e-3:.0e})")
    print(f"   - Diversificación: {'✅ SÍ' if diversification else '❌ NO'} (>1e-5)")
    print(f"   - Regrets sustanciales: {'✅ SÍ' if substantial_regrets else '❌ NO'} (>1000)")
    
    # VEREDICTO FINAL
    perfect_performance = is_fast and speed > 100  # Muy rápido
    perfect_learning = learning_detected and significant_learning and substantial_regrets
    
    print(f"\n🏆 VEREDICTO V2:")
    
    if perfect_performance and perfect_learning:
        print(f"   ✅ ¡PERFECTO! CFR correcto + GPU máximo")
        print(f"   🚀 Velocidad: {learning_speed:.1f} it/s (excelente)")
        print(f"   🧠 Aprendizaje: Detectado y significativo")
        print(f"   💎 Listo para entrenamientos de producción")
        verdict = "PERFECT"
    elif perfect_learning and is_fast:
        print(f"   ✅ EXCELENTE - Aprendizaje correcto + velocidad buena")
        print(f"   🚀 Velocidad: {learning_speed:.1f} it/s")
        print(f"   🧠 Aprendizaje: Correcto")
        verdict = "EXCELLENT"
    elif perfect_learning:
        print(f"   ⚠️  Aprendizaje correcto pero velocidad moderada")
        print(f"   🧠 CFR: ✅ Funcionando")
        print(f"   🚀 GPU: Necesita optimización")
        verdict = "LEARNING_OK"
    elif is_fast:
        print(f"   ⚠️  Velocidad buena pero aprendizaje débil")
        print(f"   🚀 GPU: ✅ Funcionando")
        print(f"   🧠 CFR: Necesita revisión")
        verdict = "SPEED_OK"
    else:
        print(f"   ❌ Problemas en velocidad Y aprendizaje")
        print(f"   🔧 Necesita trabajo adicional")
        verdict = "NEEDS_WORK"
    
    # Comparar con versiones anteriores
    print(f"\n📊 COMPARACIÓN:")
    print(f"   - V1 (CPU fallback): ~1.7 it/s + aprendizaje OK")
    print(f"   - V0 (GPU puro): ~300 it/s + sin aprendizaje")
    print(f"   - V2 (híbrido): {learning_speed:.1f} it/s + aprendizaje {'✅' if learning_detected else '❌'}")
    
    return {
        'verdict': verdict,
        'speed': learning_speed,
        'learning_detected': learning_detected,
        'strategy_change': strategy_change,
        'non_zero_regrets': int(non_zero_regrets)
    }

def quick_comparison_test():
    """Quick test de las 3 versiones para comparar"""
    print(f"\n🔬 COMPARACIÓN RÁPIDA DE VERSIONES")
    print("="*60)
    
    results = {}
    
    # Test V2
    print(f"🚀 Testing V2 (perfecto)...")
    trainer_v2 = create_gpu_trainer_v2('fast')  # Config rápida
    
    start = time.time()
    trainer_v2.train(10, "comparison_v2", save_interval=10)
    v2_time = time.time() - start
    v2_speed = 10 / v2_time
    v2_learning = float(jnp.std(trainer_v2.strategy)) > 1e-6
    
    results['v2'] = {
        'speed': v2_speed,
        'learning': v2_learning,
        'time': v2_time
    }
    
    print(f"   - V2: {v2_speed:.1f} it/s, aprendizaje: {'✅' if v2_learning else '❌'}")
    
    # Test V1 (si está disponible)
    try:
        from trainer_mccfr_gpu_optimized import create_gpu_trainer
        print(f"🔄 Testing V1 (CPU fallback)...")
        
        trainer_v1 = create_gpu_trainer('fast')
        start = time.time()
        trainer_v1.train(10, "comparison_v1", save_interval=10)
        v1_time = time.time() - start
        v1_speed = 10 / v1_time
        v1_learning = float(jnp.std(trainer_v1.strategy)) > 1e-6
        
        results['v1'] = {
            'speed': v1_speed,
            'learning': v1_learning,
            'time': v1_time
        }
        
        print(f"   - V1: {v1_speed:.1f} it/s, aprendizaje: {'✅' if v1_learning else '❌'}")
        
        # Comparación
        print(f"\n📊 COMPARACIÓN FINAL:")
        print(f"   - V2 vs V1 velocidad: {v2_speed/v1_speed:.1f}x")
        print(f"   - V2 aprendizaje: {'✅' if v2_learning else '❌'}")
        print(f"   - V1 aprendizaje: {'✅' if v1_learning else '❌'}")
        
        if v2_speed > v1_speed and v2_learning:
            print(f"   🏆 V2 GANA: Más rápido Y aprende")
        elif v2_learning and v1_learning:
            print(f"   ⚖️  EMPATE: Ambos aprenden, V2 más rápido")
        else:
            print(f"   🤔 Revisar resultados...")
            
    except ImportError:
        print(f"   ⚠️ V1 no disponible para comparación")
    
    return results

if __name__ == "__main__":
    print("🚀 INICIANDO TEST GPU TRAINER V2 - PERFECCIÓN")
    print("="*70)
    
    results = test_gpu_v2_perfect()
    
    if results['verdict'] in ['PERFECT', 'EXCELLENT']:
        print(f"\n🎉 ¡V2 ES LA SOLUCIÓN DEFINITIVA!")
        print(f"   Usa: from trainer_mccfr_gpu_optimized_v2 import create_gpu_trainer_v2")
        print(f"   trainer = create_gpu_trainer_v2('standard')")
        print(f"   trainer.train(1000, 'production_model')")
        
        # Estimaciones de producción
        speed = results['speed']
        estimates = [
            (100, "Test rápido"),
            (1000, "Modelo estándar"),
            (5000, "Nivel profesional"),
            (10000, "Nivel elite")
        ]
        
        print(f"\n⏱️  ESTIMACIONES DE PRODUCCIÓN:")
        for iters, desc in estimates:
            time_est = iters / speed
            if time_est < 60:
                time_str = f"{time_est:.0f}s"
            elif time_est < 3600:
                time_str = f"{time_est/60:.1f}min"
            else:
                time_str = f"{time_est/3600:.1f}hrs"
            print(f"   - {desc:15s}: {iters:5d} iter = {time_str}")
    else:
        print(f"\n🤔 V2 necesita más trabajo - revisar arriba")
        
    # Comparación opcional
    try:
        response = input(f"\n❓ ¿Comparación con otras versiones? (y/n): ")
        if response.lower() == 'y':
            comparison = quick_comparison_test()
    except:
        pass 