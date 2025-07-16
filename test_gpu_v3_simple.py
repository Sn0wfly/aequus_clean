"""
🚀 SIMPLE TEST GPU V3 - Para ejecutar en vast.ai
=================================================
Test simplificado del GPU Trainer V3 - Solución CPU fallback

EJECUTAR EN VAST.AI:
python test_gpu_v3_simple.py

EXPECTED RESULTS:
- GPU utilization >50% (vs 3% anterior)  
- CPU utilization <50% (vs 99% anterior)
- Velocidad >50 it/s (vs 1-4 it/s anterior)
"""

import jax
import jax.numpy as jnp
import time

def main():
    print("🚀 GPU TRAINER V3 - SOLUCIÓN CPU FALLBACK")
    print("="*60)
    
    # Check devices
    print(f"🖥️  JAX devices: {jax.devices()}")
    
    # Import and test V3
    try:
        from trainer_mccfr_gpu_optimized_v3 import GPUTrainerV3
        print("✅ V3 trainer imported successfully")
        
        # Quick verification: no pure_callback in source
        with open('trainer_mccfr_gpu_optimized_v3.py', 'r') as f:
            source = f.read()
        
        if 'pure_callback' in source:
            print("❌ ERROR: pure_callback found in V3!")
        else:
            print("✅ CONFIRMED: Zero pure_callback in V3")
            
        print("\n🚀 TESTING V3 SPEED...")
        trainer = GPUTrainerV3()
        
        # Test speed with short run
        start_time = time.time()
        results = trainer.train(10, verbose=True)  # Just 10 iterations
        total_time = time.time() - start_time
        
        speed = 10 / total_time
        print(f"\n📊 RESULTS:")
        print(f"   - Speed: {speed:.1f} it/s")
        print(f"   - Expected: >50 it/s (zero callbacks)")
        
        if speed > 50:
            print("\n🎉 SUCCESS: Ultra high speed achieved!")
            print("   ✅ Likely zero CPU fallback")
            print("   ✅ Check nvidia-smi for GPU util >50%")
        elif speed > 10:
            print("\n✅ GOOD: High speed achieved")
            print("   ✅ Major improvement vs V1/V2")
            print("   ✅ Check nvidia-smi for GPU utilization")
        else:
            print("\n🔧 SLOW: May still have CPU fallback")
            print("   🔍 Check for hidden callbacks")
            
        # Learning test
        print(f"\n🧠 TESTING LEARNING...")
        initial_std = float(jnp.std(trainer.strategy))
        trainer.train(20, verbose=False)  # Quick learning test
        final_std = float(jnp.std(trainer.strategy))
        
        learning_change = final_std - initial_std
        
        print(f"   - Initial STD: {initial_std:.6f}")
        print(f"   - Final STD: {final_std:.6f}")
        print(f"   - Change: {learning_change:.6f}")
        
        if learning_change > 1e-4:
            print("   ✅ Learning detected!")
        else:
            print("   ❌ No learning detected")
            
        print(f"\n🏆 VEREDICTO:")
        if speed > 10 and learning_change > 1e-4:
            print("   ✅ V3 SUCCESS: Speed + Learning")
            print("   ✅ Zero callbacks solution working")
            print("   ✅ Monitor GPU util with nvidia-smi")
        else:
            print("   🔧 Needs investigation")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 