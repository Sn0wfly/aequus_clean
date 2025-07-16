#!/usr/bin/env python3
"""
🏁 BENCHMARK MCCFR MERCEDES-BENZ
Test de velocidad para entrenamientos largos

Mide:
- Iteraciones por segundo (it/s)
- Tiempo estimado para entrenamientos largos
- Comparación entre configuraciones
- Recomendaciones de hardware
"""

import time
import jax
import jax.numpy as jnp
from trainer_mccfr_real import create_mccfr_trainer, MCCFRTrainer, MCCFRConfig

def benchmark_mccfr_speed():
    """🏁 Benchmark completo de velocidad MCCFR"""
    print("🏁 BENCHMARK MCCFR MERCEDES-BENZ")
    print("="*60)
    
    configs_to_test = [
        ("fast", "Configuración Rápida", 25_000, 128),
        ("standard", "Configuración Estándar", 50_000, 256), 
        ("large", "Configuración Grande", 100_000, 512)
    ]
    
    results = {}
    
    for config_name, display_name, max_info_sets, batch_size in configs_to_test:
        print(f"\n🧪 TESTEANDO: {display_name}")
        print(f"   📊 Info sets: {max_info_sets:,}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Compilando JIT...")
        
        try:
            # Crear trainer
            trainer = create_mccfr_trainer(config_name)
            
            # Warmup JIT compilation (no contar este tiempo)
            print(f"   ⏳ Warmup (compilación JIT)...")
            start_warmup = time.time()
            trainer.train(3, f"benchmark_warmup_{config_name}", save_interval=999)
            warmup_time = time.time() - start_warmup
            print(f"   ✅ Warmup completado: {warmup_time:.1f}s")
            
            # Benchmark real (10 iteraciones)
            print(f"   🏁 Benchmark real (10 iteraciones)...")
            start_bench = time.time()
            trainer.train(10, f"benchmark_real_{config_name}", save_interval=999)
            bench_time = time.time() - start_bench
            
            # Calcular métricas
            iterations_per_second = 10.0 / bench_time
            seconds_per_iteration = bench_time / 10.0
            
            # Almacenar resultados
            results[config_name] = {
                'display_name': display_name,
                'max_info_sets': max_info_sets,
                'batch_size': batch_size,
                'warmup_time': warmup_time,
                'bench_time': bench_time,
                'it_per_sec': iterations_per_second,
                'sec_per_it': seconds_per_iteration
            }
            
            print(f"   📈 Velocidad: {iterations_per_second:.2f} it/s")
            print(f"   ⏱️  Tiempo por iteración: {seconds_per_iteration:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error en {config_name}: {str(e)}")
            results[config_name] = None
    
    # Mostrar resumen comparativo
    print(f"\n📊 RESUMEN COMPARATIVO")
    print("="*60)
    
    for config_name, result in results.items():
        if result:
            print(f"\n🏷️  {result['display_name']}:")
            print(f"   ⚡ Velocidad: {result['it_per_sec']:.2f} it/s")
            print(f"   📊 Info sets: {result['max_info_sets']:,}")
            print(f"   📦 Batch size: {result['batch_size']}")
            print(f"   ⏱️  Por iteración: {result['sec_per_it']:.2f}s")
    
    # Estimaciones de tiempo para entrenamientos largos
    print(f"\n⏰ ESTIMACIONES PARA ENTRENAMIENTOS LARGOS")
    print("="*60)
    
    training_scenarios = [
        (1000, "Entrenamiento Profesional"),
        (5000, "Entrenamiento Elite"),
        (10000, "Entrenamiento Super-Humano")
    ]
    
    best_config = None
    best_speed = 0
    
    for config_name, result in results.items():
        if result and result['it_per_sec'] > best_speed:
            best_speed = result['it_per_sec']
            best_config = result
    
    if best_config:
        print(f"\n🏆 RECOMENDACIÓN: {best_config['display_name']}")
        print(f"   (Velocidad: {best_config['it_per_sec']:.2f} it/s)")
        
        for iterations, scenario_name in training_scenarios:
            estimated_time = iterations / best_config['it_per_sec']
            hours = estimated_time / 3600
            minutes = (estimated_time % 3600) / 60
            
            print(f"\n🎯 {scenario_name} ({iterations:,} iteraciones):")
            if hours >= 1:
                print(f"   ⏰ Tiempo estimado: {hours:.1f} horas ({minutes:.0f} min)")
            else:
                print(f"   ⏰ Tiempo estimado: {minutes:.1f} minutos")
    
    # Hardware info
    print(f"\n💻 INFORMACIÓN DEL SISTEMA")
    print("="*60)
    try:
        import platform
        print(f"   🖥️  Sistema: {platform.system()} {platform.release()}")
        print(f"   🐍 Python: {platform.python_version()}")
        
        # JAX device info
        devices = jax.devices()
        print(f"   🔧 JAX devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"      • Device {i}: {device.device_kind}")
            
    except Exception as e:
        print(f"   ⚠️  No se pudo obtener info del sistema: {e}")
    
    return results

def quick_speed_test():
    """🚀 Test rápido de velocidad (solo configuración estándar)"""
    print("🚀 QUICK SPEED TEST - Mercedes-Benz")
    print("="*50)
    
    trainer = create_mccfr_trainer("standard")
    
    # Warmup
    print("⏳ Compilando JIT...")
    start = time.time()
    trainer.train(1, "speed_warmup", save_interval=999)
    warmup = time.time() - start
    print(f"✅ JIT compiled: {warmup:.1f}s")
    
    # Benchmark
    print("🏁 Midiendo velocidad (5 iteraciones)...")
    start = time.time()
    trainer.train(5, "speed_test", save_interval=999)
    elapsed = time.time() - start
    
    speed = 5.0 / elapsed
    print(f"\n📈 RESULTADO:")
    print(f"   ⚡ Velocidad: {speed:.2f} it/s")
    print(f"   ⏱️  Por iteración: {elapsed/5:.2f}s")
    
    # Estimaciones rápidas
    scenarios = [(1000, "1K iter"), (5000, "5K iter"), (10000, "10K iter")]
    print(f"\n⏰ ESTIMACIONES:")
    for iters, name in scenarios:
        time_hours = (iters / speed) / 3600
        if time_hours < 1:
            time_mins = (iters / speed) / 60
            print(f"   🎯 {name}: {time_mins:.1f} minutos")
        else:
            print(f"   🎯 {name}: {time_hours:.1f} horas")
    
    return speed

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test
        quick_speed_test()
    else:
        # Full benchmark
        benchmark_mccfr_speed()
    
    print(f"\n🚀 ¡Listo para entrenamiento largo!")
    print(f"Uso:")
    print(f"  python benchmark_mccfr_speed.py quick    # Test rápido")
    print(f"  python benchmark_mccfr_speed.py          # Benchmark completo") 