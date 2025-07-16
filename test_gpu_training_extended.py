#!/usr/bin/env python3
"""
Test script para verificar que el entrenamiento GPU funciona correctamente
con sesiones más largas y produce aprendizaje real.
"""

import time
import jax
from trainer_mccfr_gpu_optimized import create_gpu_trainer

def test_extended_gpu_training():
    """Test entrenamiento GPU extendido con verificación de aprendizaje"""
    print("🧪 TEST GPU TRAINING EXTENDIDO")
    print("="*50)
    
    # Crear trainer GPU
    trainer = create_gpu_trainer('standard')
    
    # Test 1: Verificar estado inicial
    print("\n📊 ESTADO INICIAL:")
    initial_strategy = trainer.strategy.copy()
    initial_std = float(jax.numpy.std(initial_strategy))
    print(f"   - Estrategia STD inicial: {initial_std:.6f}")
    print(f"   - Estrategia shape: {initial_strategy.shape}")
    
    # Test 2: Entrenamiento corto pero sostenido (100 iteraciones)
    print("\n🚀 ENTRENAMIENTO CORTO (100 iteraciones)...")
    start_time = time.time()
    
    trainer.train(
        num_iterations=100,
        save_path="gpu_test_model",
        save_interval=50,
        snapshot_iterations=[]  # Sin evaluaciones para velocidad pura
    )
    
    short_time = time.time() - start_time
    short_speed = 100 / short_time
    
    print(f"\n📈 RESULTADOS 100 ITERACIONES:")
    print(f"   - Tiempo: {short_time:.2f}s")
    print(f"   - Velocidad: {short_speed:.1f} it/s")
    print(f"   - Throughput: ~{100 * 128 * 50 / short_time:.0f} hands/s")
    
    # Test 3: Verificar aprendizaje después de 100 iteraciones
    trained_strategy = trainer.strategy.copy()
    trained_std = float(jax.numpy.std(trained_strategy))
    
    # Verificar que cambió la estrategia
    strategy_change = float(jax.numpy.mean(jax.numpy.abs(trained_strategy - initial_strategy)))
    
    print(f"\n🧠 VERIFICACIÓN DE APRENDIZAJE:")
    print(f"   - Estrategia STD final: {trained_std:.6f}")
    print(f"   - Cambio promedio: {strategy_change:.6f}")
    print(f"   - Ratio de cambio: {trained_std/max(initial_std, 1e-8):.2f}x")
    
    learning_detected = strategy_change > 1e-4
    print(f"   - ¿Aprendizaje detectado? {'✅ SÍ' if learning_detected else '❌ NO'}")
    
    # Test 4: Entrenamiento mediano (500 iteraciones) para velocidad sostenida
    print("\n🚀 ENTRENAMIENTO MEDIANO (500 iteraciones)...")
    start_time = time.time()
    
    # Crear nuevo trainer para test limpio
    trainer_2 = create_gpu_trainer('standard')
    trainer_2.train(
        num_iterations=500,
        save_path="gpu_test_model_500",
        save_interval=250,
        snapshot_iterations=[]  # Sin evaluaciones para velocidad pura
    )
    
    medium_time = time.time() - start_time
    medium_speed = 500 / medium_time
    
    print(f"\n📈 RESULTADOS 500 ITERACIONES:")
    print(f"   - Tiempo: {medium_time:.2f}s ({medium_time/60:.1f} min)")
    print(f"   - Velocidad: {medium_speed:.1f} it/s")
    print(f"   - Throughput: ~{500 * 128 * 50 / medium_time:.0f} hands/s")
    
    # Test 5: Estimaciones para entrenamientos largos
    print(f"\n⏱️ ESTIMACIONES PARA ENTRENAMIENTOS LARGOS:")
    
    avg_speed = (short_speed + medium_speed) / 2
    
    estimates = [
        (1000, "1K iter"),
        (5000, "5K iter"),
        (10000, "10K iter"),
        (25000, "25K iter"),
    ]
    
    for iters, label in estimates:
        est_time = iters / avg_speed
        est_minutes = est_time / 60
        est_hours = est_time / 3600
        
        if est_hours >= 1:
            time_str = f"{est_hours:.1f} horas"
        elif est_minutes >= 1:
            time_str = f"{est_minutes:.1f} min"
        else:
            time_str = f"{est_time:.0f}s"
            
        print(f"   - {label:8s}: {time_str}")
    
    # Test 6: Comparación con velocidad original
    original_speed = 2.23  # it/s
    speedup = avg_speed / original_speed
    
    print(f"\n🏆 COMPARACIÓN FINAL:")
    print(f"   - GPU optimized: {avg_speed:.1f} it/s")
    print(f"   - Original CPU: {original_speed:.1f} it/s")
    print(f"   - Speedup: {speedup:.0f}x más rápido")
    
    # Cálculo de ahorro de tiempo
    for iters, label in [(1000, "1K"), (10000, "10K")]:
        gpu_time = iters / avg_speed / 60  # minutos
        cpu_time = iters / original_speed / 60  # minutos
        savings = cpu_time - gpu_time
        
        print(f"   - {label}: GPU {gpu_time:.1f}min vs CPU {cpu_time:.0f}min (ahorra {savings:.0f}min)")
    
    print(f"\n🎯 CONCLUSIÓN:")
    if learning_detected and avg_speed > 100:
        print("✅ GPU Training VERIFICADO: Velocidad alta + Aprendizaje real")
        print("🚀 Listo para entrenamientos de producción!")
    elif learning_detected:
        print("⚠️ Aprendizaje detectado pero velocidad moderada")
    else:
        print("❌ Problema: Sin aprendizaje detectado")
    
    return {
        'avg_speed': avg_speed,
        'learning_detected': learning_detected,
        'speedup': speedup
    }

if __name__ == "__main__":
    print("🔥 INICIANDO TEST EXTENDIDO GPU TRAINER")
    print("="*60)
    
    # Verificar que JAX detecta GPU
    print(f"🖥️  JAX devices: {jax.devices()}")
    if 'gpu' in str(jax.devices()).lower():
        print("✅ GPU detectada por JAX")
    else:
        print("⚠️ Solo CPU detectada")
    
    results = test_extended_gpu_training()
    
    print("\n" + "="*60)
    print("🏁 TEST COMPLETADO!")
    if results['learning_detected'] and results['avg_speed'] > 100:
        print("🎉 GPU TRAINER 100% FUNCIONAL!")
    else:
        print("🤔 Revisar resultados arriba") 