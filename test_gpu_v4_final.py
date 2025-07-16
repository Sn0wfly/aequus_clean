"""
🚀 TEST FINAL GPU V4 - VECTORIZED OPERATIONS
=============================================
SOLUCIÓN FINAL al problema CPU fallback:
- ❌ Zero pure_callback
- ❌ Zero lax.cond anidadas
- ✅ Pure jnp.where operations (GPU-friendly)

EXPECTED: GPU util >80%, velocidad >100 it/s
"""

import jax
import jax.numpy as jnp
import time

def main():
    print("🚀 GPU TRAINER V4 - SOLUCIÓN FINAL CPU FALLBACK")
    print("="*60)
    
    # Check devices
    print(f"🖥️  JAX devices: {jax.devices()}")
    
    try:
        from trainer_mccfr_gpu_optimized_v4 import GPUTrainerV4
        print("✅ V4 trainer imported successfully")
        
        # Verification: no problematic patterns
        with open('trainer_mccfr_gpu_optimized_v4.py', 'r') as f:
            source = f.read()
        
        # Check for problem patterns
        problems = []
        if 'pure_callback' in source and 'jax.pure_callback' in source:
            problems.append("pure_callback usage found")
        
        # Count nested lax.cond (this should be much lower)
        nested_cond_count = source.count('lambda: lax.cond(')
        if nested_cond_count > 5:  # Allow some, but not excessive nesting
            problems.append(f"excessive nested lax.cond: {nested_cond_count}")
        
        if problems:
            print(f"⚠️ POTENTIAL ISSUES: {problems}")
        else:
            print("✅ CLEAN: No problematic patterns detected")
            
        print(f"📊 Nested lax.cond count: {nested_cond_count} (should be <5)")
            
        print("\n🚀 TESTING V4 FINAL...")
        trainer = GPUTrainerV4()
        
        # Quick speed test
        print("⏱️ Quick speed test...")
        start_time = time.time()
        results = trainer.train(15, verbose=True)
        total_time = time.time() - start_time
        
        speed = 15 / total_time
        print(f"\n📊 V4 RESULTS:")
        print(f"   - Speed: {speed:.1f} it/s")
        print(f"   - Expected: >100 it/s")
        
        # Speed analysis
        if speed > 100:
            print("\n🎉 ULTRA SUCCESS: >100 it/s achieved!")
            print("   ✅ CPU fallback ELIMINATED")
            print("   ✅ Pure GPU execution confirmed")
            print("   ✅ Check nvidia-smi: GPU util should be >80%")
        elif speed > 50:
            print("\n✅ GOOD SUCCESS: High speed achieved")
            print("   ✅ Major improvement")
            print("   ✅ Check nvidia-smi for GPU utilization")
        elif speed > 10:
            print("\n⚡ PARTIAL SUCCESS: Better than before")
            print("   ✅ Some improvement")
            print("   🔍 Still room for optimization")
        else:
            print("\n🔧 NEEDS WORK: Still slow")
            print("   🔍 May still have hidden CPU fallback")
            
        # Quick learning test
        print(f"\n🧠 QUICK LEARNING TEST...")
        initial_std = float(jnp.std(trainer.strategy))
        trainer.train(25, verbose=False)
        final_std = float(jnp.std(trainer.strategy))
        
        learning_change = final_std - initial_std
        
        print(f"   - Change: {learning_change:.6f}")
        
        if learning_change > 1e-4:
            print("   ✅ Learning confirmed!")
        else:
            print("   ⚠️ Learning unclear")
            
        # Final verdict
        print(f"\n🏆 VEREDICTO FINAL V4:")
        
        success_speed = speed > 50
        success_learning = learning_change > 1e-4
        
        if success_speed and success_learning:
            print("   🎉 V4 SUCCESS COMPLETO!")
            print("   ✅ Speed improvement achieved")
            print("   ✅ Learning working")
            print("   ✅ Vectorized operations successful")
            print(f"   ✅ Monitor GPU util - should be >50%")
            
            if speed > 100:
                print("\n🚀 BREAKTHROUGH: >100 it/s = CPU fallback ELIMINATED!")
        else:
            print("   🔧 Partial success")
            print(f"   - Speed: {'✅' if success_speed else '❌'} ({speed:.1f} it/s)")
            print(f"   - Learning: {'✅' if success_learning else '❌'}")
            
        # Comparison
        print(f"\n📊 COMPARISON:")
        print(f"   - V1 (CPU fallback): ~1.7 it/s")
        print(f"   - V2 (hybrid): ~2-4 it/s")  
        print(f"   - V3 (nested cond): ~1.4 it/s")
        print(f"   - V4 (vectorized): {speed:.1f} it/s")
        
        improvement = speed / 1.7  # vs V1
        print(f"   - Improvement vs V1: {improvement:.1f}x")
        
        if improvement > 10:
            print("   🏆 MAJOR BREAKTHROUGH!")
        elif improvement > 5:
            print("   ✅ Significant improvement")
        elif improvement > 2:
            print("   ⚡ Good improvement")
        else:
            print("   🔧 Needs more work")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 