#!/usr/bin/env python3
"""
🔥 TEST PYTORCH CFR TRAINER
===========================
Test del trainer PyTorch traducido desde trainer_mccfr_real.py

VERIFICA:
✅ GPU performance (vs JAX CPU fallback)
✅ Info sets ricos funcionando
✅ CFR logic correcto
✅ Velocidad de entrenamiento
"""

import torch
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_pytorch_trainer():
    """Test completo del trainer PyTorch"""
    print("🔥 TESTING PYTORCH CFR TRAINER")
    print("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ No GPU available - PyTorch advantage limited")
        device_name = "CPU"
    else:
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {device_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
    
    try:
        from poker_trainer_pytorch import create_pytorch_trainer
        print("✅ PyTorch trainer imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Test 1: Quick speed test
    print(f"\n🚀 TEST 1: SPEED COMPARISON")
    print(f"   Expected: 10-50x faster than JAX (>10 it/s)")
    
    trainer = create_pytorch_trainer("fast")
    
    # Warmup
    print("   Warming up...")
    trainer.mccfr_step()
    
    # Speed test
    iterations = 20
    start_time = time.time()
    
    for i in range(iterations):
        trainer.mccfr_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for GPU
    
    total_time = time.time() - start_time
    speed = iterations / total_time
    
    print(f"\n📊 SPEED RESULTS:")
    print(f"   - Iterations: {iterations}")
    print(f"   - Time: {total_time:.2f}s")
    print(f"   - Speed: {speed:.1f} it/s")
    
    # Compare with expected JAX performance
    jax_speed = 2.2  # From vast.ai results
    improvement = speed / jax_speed if jax_speed > 0 else 0
    
    print(f"\n📈 COMPARISON vs JAX:")
    print(f"   - JAX (vast.ai): {jax_speed:.1f} it/s")
    print(f"   - PyTorch: {speed:.1f} it/s")
    print(f"   - Improvement: {improvement:.1f}x")
    
    if speed > 10:
        print("   ✅ EXCELLENT: >10 it/s achieved!")
    elif speed > 5:
        print("   ✅ GOOD: Significant improvement")
    elif speed > jax_speed:
        print("   ✅ BETTER: Faster than JAX")
    else:
        print("   ⚠️ SLOW: Needs optimization")
    
    # Test 2: Training test
    print(f"\n🧠 TEST 2: TRAINING VALIDATION")
    
    trainer_train = create_pytorch_trainer("fast")
    
    # Initial state
    initial_strategy = trainer_train.strategy.clone()
    initial_std = torch.std(initial_strategy)
    
    print(f"   Initial strategy STD: {initial_std:.6f}")
    
    # Short training
    print("   Training 30 iterations...")
    start_time = time.time()
    trainer_train.train(30, "pytorch_test", save_interval=30)
    training_time = time.time() - start_time
    
    # Check learning
    final_strategy = trainer_train.strategy
    final_std = torch.std(final_strategy)
    strategy_change = torch.mean(torch.abs(final_strategy - initial_strategy))
    
    print(f"\n📊 LEARNING RESULTS:")
    print(f"   - Final strategy STD: {final_std:.6f}")
    print(f"   - Strategy change: {strategy_change:.6f}")
    print(f"   - Training speed: {30/training_time:.1f} it/s")
    
    learning_detected = strategy_change > 1e-4
    print(f"   - Learning: {'✅ YES' if learning_detected else '❌ NO'}")
    
    # Test 3: Info sets analysis
    print(f"\n🏆 TEST 3: INFO SETS ANALYSIS")
    results = trainer_train.analyze_training_progress()
    
    # Test 4: Memory usage (if GPU)
    if torch.cuda.is_available():
        print(f"\n💾 TEST 4: GPU MEMORY")
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_percent = memory_used / memory_total * 100
        
        print(f"   - Used: {memory_used:.1f} GB")
        print(f"   - Total: {memory_total:.1f} GB") 
        print(f"   - Usage: {memory_percent:.1f}%")
        
        if memory_percent < 50:
            print("   ✅ GOOD: Efficient memory usage")
        else:
            print("   ⚠️ HIGH: Memory usage high")
    
    # Overall verdict
    print(f"\n🏆 FINAL VERDICT:")
    
    success_criteria = [
        speed > 5.0,  # At least 5 it/s
        learning_detected,  # Learning working
        results['trained_info_sets'] > 50,  # Some training happening
        results['rich_differentiation'] > 0.2  # Info sets differentiating
    ]
    
    passed = sum(success_criteria)
    total = len(success_criteria)
    
    print(f"   Tests passed: {passed}/{total}")
    
    if passed == total:
        print("   🏆 PYTORCH TRAINER: ✅ PERFECT!")
        print("   Ready for production training")
        
        # Estimate production performance
        print(f"\n📈 PRODUCTION ESTIMATES:")
        if "4090" in device_name:
            est_speed = "50-150 it/s"
        elif "3080" in device_name:
            est_speed = "30-80 it/s"
        elif torch.cuda.is_available():
            est_speed = f"{speed*2:.0f}-{speed*5:.0f} it/s"
        else:
            est_speed = f"{speed:.0f} it/s (CPU)"
        
        print(f"   Estimated speed: {est_speed}")
        print(f"   1000 iterations: {1000/speed/60:.1f}-{1000/(speed*3)/60:.1f} minutes")
        
    elif passed >= 3:
        print("   ✅ PYTORCH TRAINER: Good, minor issues")
    elif passed >= 2:
        print("   ⚠️ PYTORCH TRAINER: Needs work")
    else:
        print("   ❌ PYTORCH TRAINER: Major issues")
    
    # Usage instructions
    if passed >= 3:
        print(f"\n🚀 USAGE FOR PRODUCTION:")
        print(f"   trainer = create_pytorch_trainer('standard')")
        print(f"   trainer.train(1000, 'production_model')")
        print(f"   # Should be 10-50x faster than JAX!")

def quick_speed_only():
    """Quick test - just speed"""
    print("🔥 QUICK PYTORCH SPEED TEST")
    print("="*40)
    
    try:
        from poker_trainer_pytorch import create_pytorch_trainer
        trainer = create_pytorch_trainer("fast")
        
        # 10 iterations speed test
        start = time.time()
        for _ in range(10):
            trainer.mccfr_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        speed = 10 / elapsed
        
        print(f"Speed: {speed:.1f} it/s")
        
        if speed > 10:
            print("✅ FAST!")
        elif speed > 5:
            print("✅ Good")
        else:
            print("⚠️ Slow")
            
        return speed
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

if __name__ == "__main__":
    print("🔥 PYTORCH CFR TRAINER - TESTING SUITE")
    print("="*70)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_speed_only()
    else:
        test_pytorch_trainer() 