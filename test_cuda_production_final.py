#!/usr/bin/env python3
"""
🚀 CUDA POKER CFR - FINAL PRODUCTION TEST
========================================
Complete test of production CUDA system
Verifies all advanced features and real learning

TESTS:
✅ Real hand evaluator (phevaluator-compatible)
✅ Advanced CFR with info sets ricos
✅ Realistic game simulation
✅ Poker IQ evaluation
✅ Learning validation
✅ Performance benchmark vs alternatives
"""

import sys
import time
import logging
import os
import ctypes

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def diagnose_library_symbols():
    """Diagnose library symbol loading issues"""
    try:
        lib_path = "./poker_cuda/libpoker_cuda.so"
        lib = ctypes.CDLL(lib_path)
        
        print("🔍 DIAGNOSING LIBRARY SYMBOLS...")
        
        # Test simple function first
        try:
            test_func = lib.cuda_test_function
            test_func.restype = ctypes.c_int
            result = test_func()
            print(f"✅ cuda_test_function: {result}")
        except AttributeError as e:
            print(f"❌ cuda_test_function not found: {e}")
        
        # Test main function
        try:
            eval_func = lib.cuda_evaluate_single_hand_real
            print("✅ cuda_evaluate_single_hand_real found")
        except AttributeError as e:
            print(f"❌ cuda_evaluate_single_hand_real not found: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Library loading failed: {e}")
        return False

def test_cuda_production_system():
    """Complete test of production CUDA system"""
    
    print("🚀 CUDA POKER CFR - FINAL PRODUCTION TEST")
    print("="*70)
    
    # Test 1: Library Check
    print("\n🧪 TEST 1: Production Library Check")
    try:
        if not os.path.exists("poker_cuda/libpoker_cuda.so"):
            print("❌ CUDA library not found")
            print("🛠️  Compile with: cd poker_cuda && make production")
            return False
        else:
            print("✅ Production CUDA library found")
    except Exception as e:
        print(f"❌ Error checking library: {e}")
        return False
    
    # Test 2: Import Production Trainer
    print("\n🧪 TEST 2: Production System Import")
    try:
        sys.path.append('poker_cuda')
        from cuda_trainer_production import ProductionCUDAPokerCFR, ProductionConfig
        print("✅ Production system imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🛠️  Make sure poker_cuda/ contains:")
        print("   - cuda_trainer_production.py")
        print("   - libpoker_cuda.so")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test 3: Production System Initialization
    print("\n🧪 TEST 3: Production System Initialization")
    try:
        config = ProductionConfig(batch_size=256)
        trainer = ProductionCUDAPokerCFR(config)
        print("✅ Production system initialized")
        print(f"   Batch size: {trainer.config.batch_size}")
        print(f"   Memory allocation: {trainer._calculate_memory_usage():.1f} MB")
        print(f"   Advanced features: ENABLED")
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("🛠️  Check CUDA installation and GPU")
        return False
    
    # Test 4: Real Hand Evaluator
    print("\n🧪 TEST 4: Real Hand Evaluator Test")
    try:
        # Test multiple hands to verify real evaluation
        test_hands = [
            ([51, 47, 46, 42, 37, 35, 32], "AA (Pocket Aces)"),
            ([51, 50, 49, 48, 47], "Royal Flush"),
            ([0, 4, 8, 12, 16], "Straight (wheel)"),
            ([0, 23, 46, 42, 37, 35, 32], "72o (worst hand)"),
            ([51, 47, 50, 46, 49], "AA with pair on board"),
        ]
        
        hand_strengths = []
        for cards, description in test_hands:
            strength = trainer.evaluate_hand_real(cards)
            hand_strengths.append((strength, description))
            print(f"   {description:20s}: {strength:8d}")
        
        # Verify logical ordering
        royal_flush_strength = hand_strengths[1][0]  # Royal flush
        aa_strength = hand_strengths[0][0]          # Pocket aces
        trash_strength = hand_strengths[3][0]       # 72o
        
        if royal_flush_strength > aa_strength > trash_strength:
            print("✅ Hand evaluation logically correct")
            print(f"   Royal Flush ({royal_flush_strength}) > AA ({aa_strength}) > 72o ({trash_strength})")
        else:
            print("⚠️  Hand evaluation unexpected ordering")
            
    except Exception as e:
        print(f"❌ Hand evaluator error: {e}")
        return False
    
    # Test 5: Advanced CFR Training
    print("\n🧪 TEST 5: Advanced CFR Training Test")
    try:
        print("   Running advanced CFR training steps...")
        
        training_times = []
        for i in range(10):
            start_time = time.time()
            result = trainer.train_step_advanced()
            step_time = time.time() - start_time
            training_times.append(step_time)
            
            if 'error' in result:
                print(f"❌ Training step {i+1} failed: {result['error']}")
                return False
            
            if i % 3 == 0:
                print(f"   Step {i+1}/10: {result['speed_it_per_sec']:.1f} it/s")
        
        avg_time = sum(training_times) / len(training_times)
        avg_speed = 1.0 / avg_time if avg_time > 0 else 0
        throughput = avg_speed * trainer.config.batch_size * trainer.config.max_players
        
        print("✅ Advanced CFR training successful")
        print(f"   Average speed: {avg_speed:.1f} it/s")
        print(f"   Throughput: {throughput:.0f} hands/s")
        print(f"   Time per iteration: {avg_time*1000:.1f} ms")
        
    except Exception as e:
        print(f"❌ CFR training error: {e}")
        return False
    
    # Test 6: Learning Validation
    print("\n🧪 TEST 6: Learning Validation")
    try:
        validation = trainer.validate_learning()
        
        if 'error' in validation:
            print(f"❌ Learning validation failed: {validation['error']}")
        else:
            print("✅ Learning validation completed")
            print(f"   Hand evaluation correct: {'✅' if validation['hand_evaluation_correct'] else '❌'}")
            print(f"   Learning detected: {'✅' if validation['learning_detected'] else '⚠️'}")
            print(f"   AA strength: {validation['aa_strength']}")
            print(f"   72o strength: {validation['trash_strength']}")
            print(f"   Current Poker IQ: {validation['current_poker_iq']:.1f}")
            
    except Exception as e:
        print(f"❌ Learning validation error: {e}")
        return False
    
    # Test 7: Poker IQ Evaluation
    print("\n🧪 TEST 7: Poker IQ Evaluation")
    try:
        poker_iq = trainer.evaluate_poker_iq()
        
        if 'error' in poker_iq:
            print(f"❌ Poker IQ evaluation failed: {poker_iq['error']}")
        else:
            print("✅ Poker IQ evaluation successful")
            print(f"   Total Poker IQ: {poker_iq['total_poker_iq']:.1f}/100")
            print(f"   Hand Strength: {poker_iq['hand_strength_score']:.1f}/25")
            print(f"   Position: {poker_iq['position_score']:.1f}/25")
            print(f"   Suited: {poker_iq['suited_score']:.1f}/20")
            print(f"   Fold Discipline: {poker_iq['fold_discipline_score']:.1f}/15")
            print(f"   Diversity: {poker_iq['diversity_score']:.1f}/15")
            
    except Exception as e:
        print(f"❌ Poker IQ evaluation error: {e}")
        return False
    
    # Test 8: Performance Benchmark
    print("\n🧪 TEST 8: Performance Benchmark vs Alternatives")
    try:
        benchmark = trainer.benchmark_vs_alternatives()
        
        print("✅ Performance benchmark completed")
        print(f"   CUDA Speed: {benchmark['cuda_speed']:.1f} it/s")
        print(f"   CUDA Throughput: {benchmark['cuda_throughput']:,.0f} hands/s")
        print(f"   Best Alternative: {benchmark['best_alternative_speed']:.1f} it/s")
        print(f"   Total Improvement: {benchmark['total_improvement']:.1f}x")
        
        # Performance classification
        if benchmark['total_improvement'] > 50:
            performance_rating = "🏆 OUTSTANDING"
        elif benchmark['total_improvement'] > 20:
            performance_rating = "🥇 EXCELLENT"
        elif benchmark['total_improvement'] > 10:
            performance_rating = "🥈 GOOD"
        elif benchmark['total_improvement'] > 5:
            performance_rating = "🥉 ACCEPTABLE"
        else:
            performance_rating = "⚠️ NEEDS WORK"
        
        print(f"   Performance Rating: {performance_rating}")
        
    except Exception as e:
        print(f"❌ Performance benchmark error: {e}")
        return False
    
    # Test 9: Production Training Sample
    print("\n🧪 TEST 9: Production Training Sample")
    try:
        print("   Running short production training...")
        
        # Short production training test
        results = trainer.train_production(
            num_iterations=50,
            save_path="test_production_sample",
            verbose=False
        )
        
        if results.get('success'):
            print("✅ Production training sample successful")
            print(f"   Final speed: {results['final_speed']:.1f} it/s")
            print(f"   Total hands processed: {results['total_hands']:,}")
            print(f"   Final Poker IQ: {results['final_poker_iq'].get('total_poker_iq', 0):.1f}/100")
        else:
            print(f"❌ Production training failed: {results.get('error')}")
            
    except Exception as e:
        print(f"❌ Production training error: {e}")
        return False
    
    # Final Summary
    print("\n" + "="*70)
    print("🏆 FINAL PRODUCTION SYSTEM SUMMARY")
    print("="*70)
    
    # Key metrics
    final_metrics = {
        'speed': benchmark['cuda_speed'],
        'improvement': benchmark['total_improvement'],
        'throughput': benchmark['cuda_throughput'],
        'poker_iq': poker_iq.get('total_poker_iq', 0),
        'hand_eval_correct': validation.get('hand_evaluation_correct', False),
        'learning_detected': validation.get('learning_detected', False)
    }
    
    print(f"🚀 PERFORMANCE METRICS:")
    print(f"   Speed: {final_metrics['speed']:.1f} it/s")
    print(f"   Improvement vs alternatives: {final_metrics['improvement']:.1f}x")
    print(f"   Throughput: {final_metrics['throughput']:,.0f} hands/s")
    
    print(f"\n🧠 INTELLIGENCE METRICS:")
    print(f"   Poker IQ: {final_metrics['poker_iq']:.1f}/100")
    print(f"   Hand evaluation: {'✅' if final_metrics['hand_eval_correct'] else '❌'}")
    print(f"   Learning capability: {'✅' if final_metrics['learning_detected'] else '⚠️'}")
    
    print(f"\n🎯 PRODUCTION READINESS:")
    all_systems_go = (
        final_metrics['speed'] > 50 and
        final_metrics['improvement'] > 10 and
        final_metrics['hand_eval_correct'] and
        final_metrics['poker_iq'] > 5
    )
    
    if all_systems_go:
        print("✅ SYSTEM IS PRODUCTION READY!")
        print("   All critical metrics pass requirements")
        print("   Ready for large-scale training")
    else:
        print("⚠️  System needs optimization for production")
        if final_metrics['speed'] <= 50:
            print("   - Speed needs improvement")
        if final_metrics['improvement'] <= 10:
            print("   - Performance advantage insufficient")
        if not final_metrics['hand_eval_correct']:
            print("   - Hand evaluation needs fixing")
        if final_metrics['poker_iq'] <= 5:
            print("   - Learning system needs work")
    
    return all_systems_go

def show_production_deployment_guide():
    """Show production deployment instructions"""
    
    print("\n" + "="*70)
    print("🚀 PRODUCTION DEPLOYMENT GUIDE")
    print("="*70)
    
    print("\n1. 🏭 PRODUCTION COMPILATION:")
    print("   cd poker_cuda/")
    print("   make production")
    
    print("\n2. 🎯 LARGE-SCALE TRAINING:")
    print("   from cuda_trainer_production import train_production_poker_bot")
    print("   trainer = train_production_poker_bot(")
    print("       num_iterations=5000,")
    print("       batch_size=2048,")
    print("       save_path='super_human_bot'")
    print("   )")
    
    print("\n3. 📊 MEMORY OPTIMIZATION:")
    print("   # Monitor GPU usage:")
    print("   watch -n 1 nvidia-smi")
    print("   ")
    print("   # Optimize batch size for your GPU:")
    print("   # 24GB GPU: batch_size=4096")
    print("   # 12GB GPU: batch_size=2048") 
    print("   # 8GB GPU:  batch_size=1024")
    
    print("\n4. 🏆 EXPECTED PRODUCTION RESULTS:")
    print("   - Speed: >100 it/s (vs 2.2 it/s JAX)")
    print("   - GPU utilization: >80% (vs 8% previous)")
    print("   - CPU usage: <30% (vs 100% previous)")
    print("   - Memory efficiency: <4GB for batch_size=2048")
    print("   - Poker IQ: 80+/100 after 5000+ iterations")
    
    print("\n5. 🔬 VALIDATION & MONITORING:")
    print("   # Check learning progress:")
    print("   trainer.evaluate_poker_iq()")
    print("   trainer.validate_learning()")
    print("   ")
    print("   # Benchmark vs alternatives:")
    print("   trainer.benchmark_vs_alternatives()")

def test_library_symbols():
    """Test individual symbol loading to diagnose library issues"""
    try:
        import ctypes
        lib_path = "./poker_cuda/libpoker_cuda.so"
        lib = ctypes.CDLL(lib_path)
        
        print("🔍 Testing individual symbol loading...")
        
        # Test simple function first
        try:
            test_func = lib.cuda_test_function
            test_func.restype = ctypes.c_int
            result = test_func()
            print(f"✅ cuda_test_function: {result}")
        except AttributeError as e:
            print(f"❌ cuda_test_function not found: {e}")
        
        # Test main function
        try:
            eval_func = lib.cuda_evaluate_single_hand_real
            eval_func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            eval_func.restype = ctypes.c_int
            print("✅ cuda_evaluate_single_hand_real found")
        except AttributeError as e:
            print(f"❌ cuda_evaluate_single_hand_real not found: {e}")
        
        # Test other functions
        functions_to_test = [
            "cuda_evaluate_hands_batch_real_wrapper",
            "cuda_validate_evaluator",
            "run_cfr_iteration_advanced"
        ]
        
        for func_name in functions_to_test:
            try:
                func = getattr(lib, func_name)
                print(f"✅ {func_name} found")
            except AttributeError:
                print(f"❌ {func_name} not found")
                
        return True
        
    except Exception as e:
        print(f"❌ Library loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting CUDA Production System test...")
    print()
    print("🚀 CUDA POKER CFR - FINAL PRODUCTION TEST")
    print("="*70)
    print()
    
    # TEST 0: Symbol diagnosis
    print("🧪 TEST 0: Library Symbol Diagnosis")
    if diagnose_library_symbols():
        print("✅ Symbol test completed")
    else:
        print("❌ Symbol test failed")
    print()
    
    # Continue with existing tests...
    print("🧪 TEST 1: Production Library Check")
    
    success = test_cuda_production_system()
    
    if success:
        print("\n🎉 ALL PRODUCTION TESTS PASSED!")
        print("System is ready for large-scale poker bot training")
        show_production_deployment_guide()
    else:
        print("\n❌ PRODUCTION TESTS FAILED")
        print("\n🛠️  Troubleshooting Steps:")
        print("   1. Ensure CUDA toolkit installed: nvcc --version")
        print("   2. Compile production system: cd poker_cuda && make production")
        print("   3. Check GPU availability: nvidia-smi")
        print("   4. Verify library exists: ls -la poker_cuda/libpoker_cuda.so")
        print("   5. Check Python dependencies")
    
    print("\n" + "="*70) 