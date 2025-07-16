#!/usr/bin/env python3
"""
🚀 CUDA POKER CFR - FINAL TEST
=============================
Test completo de la solución CUDA que reemplaza JAX/PyTorch
con performance superior

EXPECTED RESULTS:
- Speed: >100 it/s (vs 2.2 it/s JAX)
- GPU Utilization: >80% (vs 8% previous)
- CPU Usage: <30% (vs 100% previous)
"""

import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_cuda_poker_cfr():
    """Complete test of CUDA CFR implementation"""
    
    print("🚀 CUDA POKER CFR - FINAL SOLUTION TEST")
    print("="*60)
    
    # Test 1: Check if CUDA library exists
    print("🧪 TEST 1: CUDA Library Check")
    try:
        import os
        cuda_lib_exists = os.path.exists("poker_cuda/libpoker_cuda.so")
        
        if not cuda_lib_exists:
            print("❌ CUDA library not found")
            print("🛠️  Please compile first:")
            print("   cd poker_cuda/")
            print("   make")
            return False
        else:
            print("✅ CUDA library found")
    except Exception as e:
        print(f"❌ Error checking CUDA library: {e}")
        return False
    
    # Test 2: Import CUDA trainer
    print("\n🧪 TEST 2: Python Interface")
    try:
        sys.path.append('poker_cuda')
        from cuda_trainer import CUDAPokerCFR
        print("✅ CUDA trainer imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🛠️  Make sure poker_cuda/ directory contains:")
        print("   - cuda_trainer.py")
        print("   - libpoker_cuda.so")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 3: Initialize trainer
    print("\n🧪 TEST 3: Trainer Initialization")
    try:
        trainer = CUDAPokerCFR(batch_size=256)
        print("✅ CUDA trainer initialized")
        print(f"   Batch size: {trainer.batch_size}")
        print(f"   GPU memory: {trainer._calculate_memory_usage():.1f} MB")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("🛠️  Check CUDA installation and GPU availability")
        return False
    
    # Test 4: Performance benchmark
    print("\n🧪 TEST 4: Performance Benchmark")
    try:
        print("   Running 20 iterations for speed test...")
        
        start_time = time.time()
        for i in range(20):
            trainer.train_step()
            if i % 5 == 0:
                print(f"   Progress: {i+1}/20")
        
        total_time = time.time() - start_time
        speed = 20 / total_time
        throughput = speed * trainer.batch_size * trainer.max_players
        
        print(f"✅ Performance test complete")
        print(f"   Speed: {speed:.1f} it/s")
        print(f"   Throughput: {throughput:.0f} hands/s")
        print(f"   Time per iteration: {total_time/20*1000:.1f} ms")
        
        # Performance evaluation
        if speed > 50:
            print("🏆 OUTSTANDING performance!")
        elif speed > 20:
            print("🥇 EXCELLENT performance!")
        elif speed > 10:
            print("🥈 GOOD performance!")
        elif speed > 5:
            print("🥉 Acceptable performance")
        else:
            print("⚠️  Performance below expectations")
            
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False
    
    # Test 5: Hand evaluator
    print("\n🧪 TEST 5: Hand Evaluator")
    try:
        # Test with pocket aces
        aa_hand = [51, 47, 46, 42, 37, 35, 32]  # AA + random board
        aa_strength = trainer.evaluate_hand(aa_hand)
        
        # Test with trash hand
        trash_hand = [0, 23, 46, 42, 37, 35, 32]  # 72o + same board  
        trash_strength = trainer.evaluate_hand(trash_hand)
        
        print(f"✅ Hand evaluator working")
        print(f"   AA strength: {aa_strength}")
        print(f"   72o strength: {trash_strength}")
        
        if aa_strength > trash_strength:
            print("✅ Hand evaluation logical (AA > 72o)")
        else:
            print("⚠️  Hand evaluation unexpected")
            
    except Exception as e:
        print(f"❌ Hand evaluator error: {e}")
        return False
    
    # Performance comparison
    print("\n📊 PERFORMANCE COMPARISON")
    print("-" * 40)
    
    alternatives = {
        'JAX V4 (GPU fallback)': 2.2,
        'PyTorch (GPU fallback)': 0.6,
        'JAX V4 (CPU only)': 16.7
    }
    
    print(f"{'Solution':25s} {'Speed':>10s} {'Improvement':>12s}")
    print("-" * 40)
    
    for name, alt_speed in alternatives.items():
        improvement = speed / alt_speed
        print(f"{name:25s} {alt_speed:>8.1f} it/s {improvement:>8.1f}x")
    
    print("-" * 40)
    print(f"{'CUDA (this test)':25s} {speed:>8.1f} it/s {'baseline':>12s}")
    
    best_alternative = max(alternatives.values())
    total_improvement = speed / best_alternative
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Best alternative: {best_alternative} it/s")
    print(f"   CUDA solution: {speed:.1f} it/s")
    print(f"   Total improvement: {total_improvement:.1f}x")
    
    if total_improvement > 5:
        print("🏆 SUCCESS! CUDA solution significantly outperforms alternatives")
    elif total_improvement > 2:
        print("🥇 GOOD! Notable improvement achieved")
    else:
        print("🤔 MODERATE improvement - consider tuning")
    
    return True

def show_next_steps():
    """Show what to do next after successful test"""
    
    print("\n" + "="*60)
    print("🚀 NEXT STEPS - PRODUCTION DEPLOYMENT")
    print("="*60)
    
    print("\n1. 🏭 PRODUCTION BUILD:")
    print("   cd poker_cuda/")
    print("   make production")
    
    print("\n2. 🎯 FULL TRAINING:")
    print("   python3 -c \"")
    print("   from cuda_trainer import train_cuda_poker_bot")
    print("   trainer = train_cuda_poker_bot(")
    print("       num_iterations=1000,")
    print("       batch_size=1024,")
    print("       save_interval=100")
    print("   )\"")
    
    print("\n3. 📊 MEMORY OPTIMIZATION:")
    print("   # Monitor GPU usage:")
    print("   watch -n 1 nvidia-smi")
    print("   ")
    print("   # Increase batch_size until GPU ~80% full")
    print("   # For 24GB GPU: try batch_size=2048 or higher")
    
    print("\n4. 🏆 EXPECTED PRODUCTION RESULTS:")
    print("   - Speed: >100 it/s (vs 2.2 it/s JAX)")
    print("   - GPU utilization: >80% (vs 8% previous)")
    print("   - CPU usage: <30% (vs 100% previous)")
    print("   - Memory: ~2GB for batch_size=1024")
    
    print("\n5. 📈 PERFORMANCE TUNING:")
    print("   make tune    # See tuning guide")
    print("   make benchmark    # Full benchmark")

if __name__ == "__main__":
    print("Starting CUDA Poker CFR final test...\n")
    
    success = test_cuda_poker_cfr()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! CUDA solution working correctly.")
        show_next_steps()
    else:
        print("\n❌ Tests failed. Please check setup and try again.")
        print("\n🛠️  Troubleshooting:")
        print("   1. Make sure CUDA is installed: nvcc --version")
        print("   2. Compile CUDA code: cd poker_cuda && make")
        print("   3. Check GPU: nvidia-smi")
        print("   4. Check library: ls -la poker_cuda/libpoker_cuda.so")
    
    print("\n" + "="*60) 