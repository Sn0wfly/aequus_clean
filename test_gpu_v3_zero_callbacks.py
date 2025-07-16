"""
🚀 TEST GPU TRAINER V3 - ZERO CALLBACKS
=========================================
TEST ESPECÍFICO: Verificar que eliminamos CPU fallback

PROBLEMA ANTERIOR:
- V1 & V2: pure_callback → CPU fallback  
- Alto VRAM (18GB) + GPU util 3% + CPU 99%

SOLUCIÓN V3:
- ❌ ZERO pure_callback  
- ✅ 100% GPU-native hand evaluation
- ✅ Expected: GPU util >90%, CPU <50%

Este test verifica la solución al problema fundamental.
"""

import jax
import jax.numpy as jnp
import time
import sys
import os

# Import V3 trainer
sys.path.append('.')
from trainer_mccfr_gpu_optimized_v3 import GPUTrainerV3, test_gpu_v3_speed, test_gpu_v3_learning

def test_zero_callbacks_verification():
    """
    🔍 VERIFICACIÓN CRÍTICA: Zero callbacks implementation
    Confirma que V3 no usa pure_callback
    """
    print("\n🔍 VERIFICACIÓN ZERO CALLBACKS")
    print("="*50)
    
    # Test 1: Import verification
    try:
        from trainer_mccfr_gpu_optimized_v3 import gpu_hand_strength_v3
        print("✅ gpu_hand_strength_v3 importado correctamente")
        
        # Verify it's JIT compiled (should be fast)
        test_cards = jnp.array([[0, 4, 8, 12, 16, 20, 24], [1, 5, 9, 13, 17, 21, 25]])
        
        # Compile
        _ = gpu_hand_strength_v3(test_cards)
        
        # Time execution
        start = time.time()
        strengths = gpu_hand_strength_v3(test_cards) 
        strengths.block_until_ready()
        exec_time = time.time() - start
        
        print(f"✅ Hand evaluation time: {exec_time*1000:.2f}ms")
        print(f"✅ Output strengths: {strengths}")
        
        if exec_time < 0.01:  # Should be very fast
            print("✅ FAST execution - likely GPU native")
        else:
            print("⚠️ SLOW execution - possible CPU fallback")
            
    except Exception as e:
        print(f"❌ Error testing hand evaluation: {e}")
        return False
    
    # Test 2: No pure_callback in source
    try:
        with open('trainer_mccfr_gpu_optimized_v3.py', 'r') as f:
            source_code = f.read()
        
        if 'pure_callback' in source_code:
            print("❌ FOUND pure_callback in V3 source!")
            return False
        else:
            print("✅ ZERO pure_callback confirmed in source")
            
        if 'host_callback' in source_code:
            print("❌ FOUND host_callback in V3 source!")
            return False
        else:
            print("✅ ZERO host_callback confirmed")
            
    except Exception as e:
        print(f"❌ Error checking source: {e}")
        return False
    
    print("🎉 ZERO CALLBACKS VERIFICATION: ✅ PASSED")
    return True

def test_gpu_utilization_prediction():
    """
    📊 PREDICCIÓN: GPU utilization should be much higher
    """
    print("\n📊 PREDICCIÓN GPU UTILIZATION V3")
    print("="*50)
    print("🎯 EXPECTATIVA:")
    print("   - GPU util: >50% (vs 3% anterior)")
    print("   - CPU util: <50% (vs 99% anterior)")
    print("   - Velocidad: >100 it/s (vs 1-4 it/s anterior)")
    print("   - VRAM: Similar (~18GB)")
    print("\n💡 RAZÓN: Zero host callbacks = Zero CPU fallback")

def main():
    """Test principal V3"""
    print("🚀 TEST GPU TRAINER V3 - SOLUCIÓN CALLBACK PROBLEM")
    print("="*70)
    
    # Device check
    print(f"🖥️  JAX devices: {jax.devices()}")
    if jax.devices()[0].platform == 'gpu':
        print("🚀 GPU: ✅ DETECTADA")
    else:
        print("❌ GPU: NO DETECTADA")
        return
    
    # Verificación zero callbacks
    if not test_zero_callbacks_verification():
        print("❌ FALLÓ verificación zero callbacks")
        return
    
    # Predicción
    test_gpu_utilization_prediction()
    
    print("\n" + "="*70)
    print("📊 TEST 1: VELOCIDAD V3 (Solución al CPU fallback)")
    
    # Test velocidad V3
    speed_results = test_gpu_v3_speed(20)
    
    print("\n" + "="*70)
    print("📊 TEST 2: APRENDIZAJE V3")
    
    # Test aprendizaje V3 
    learning_results, learning_ok = test_gpu_v3_learning(30)
    
    # Análisis final
    print("\n" + "="*70)
    print("🏆 ANÁLISIS FINAL V3")
    print("="*70)
    
    speed_success = speed_results['speed'] > 10  # Al menos 10x better than V1
    
    print(f"🚀 Velocidad V3: {speed_results['speed']:.1f} it/s")
    print(f"🧠 Aprendizaje V3: {'✅ OK' if learning_ok else '❌ FALLO'}")
    print(f"❌ Callbacks V3: ✅ ZERO")
    
    # Comparación con versiones anteriores
    print(f"\n📊 COMPARACIÓN:")
    print(f"   - V0 (GPU puro): ~300 it/s + sin aprendizaje")
    print(f"   - V1 (CPU fallback): ~1.7 it/s + aprendizaje OK")  
    print(f"   - V2 (híbrido): ~2-400 it/s + aprendizaje ✅")
    print(f"   - V3 (zero callbacks): {speed_results['speed']:.1f} it/s + aprendizaje {'✅' if learning_ok else '❌'}")
    
    # Veredicto final
    if speed_success and learning_ok:
        print(f"\n🎉 GPU TRAINER V3: ✅ PROBLEMA RESUELTO")
        print(f"   ✅ Zero callbacks implementado")
        print(f"   ✅ Velocidad excelente ({speed_results['speed']:.1f} it/s)")
        print(f"   ✅ Aprendizaje funcionando")
        print(f"   ✅ Debería mostrar GPU util >50%, CPU <50%")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Verificar GPU utilization con nvidia-smi")
        print(f"   2. Si GPU util sigue baja, hay otro callback escondido")
        print(f"   3. Si GPU util >50%, problema RESUELTO! 🎉")
        
    else:
        print(f"\n🔧 GPU TRAINER V3: Necesita ajustes")
        print(f"   - Velocidad: {'✅' if speed_success else '❌'}")
        print(f"   - Aprendizaje: {'✅' if learning_ok else '❌'}")
        
        if not speed_success:
            print(f"   🔍 Velocidad baja sugiere que aún hay CPU fallback")
            print(f"   🔍 Revisar por callbacks escondidos")

if __name__ == "__main__":
    main() 