#!/usr/bin/env python3
"""
Script de entrenamiento de producción con GPU optimizado.
Niveles: quick, standard, professional, elite, superhuman
"""

import os
import time
import argparse
import jax
from trainer_mccfr_gpu_optimized import create_gpu_trainer

def main():
    parser = argparse.ArgumentParser(description="GPU Poker Training")
    parser.add_argument('level', choices=['quick', 'standard', 'professional', 'elite', 'superhuman'], 
                       help='Training level')
    parser.add_argument('--name', default='gpu_model', help='Model name prefix')
    parser.add_argument('--iterations', type=int, help='Override default iterations')
    
    args = parser.parse_args()
    
    # Training configurations
    configs = {
        'quick': {
            'iterations': 100,
            'save_interval': 50,
            'description': 'Quick test training (2-3 min)',
            'expected_time': '2-3 min'
        },
        'standard': {
            'iterations': 1000,
            'save_interval': 100,
            'description': 'Standard training (3-5 min)',
            'expected_time': '3-5 min'
        },
        'professional': {
            'iterations': 5000,
            'save_interval': 250,
            'description': 'Professional level (15-20 min)',
            'expected_time': '15-20 min'
        },
        'elite': {
            'iterations': 10000,
            'save_interval': 500,
            'description': 'Elite level (30-40 min)',
            'expected_time': '30-40 min'
        },
        'superhuman': {
            'iterations': 25000,
            'save_interval': 1000,
            'description': 'Superhuman level (1.5-2 hours)',
            'expected_time': '1.5-2 hours'
        }
    }
    
    config = configs[args.level]
    if args.iterations:
        config['iterations'] = args.iterations
    
    print("🚀 GPU POKER TRAINING - PRODUCTION MODE")
    print("="*60)
    print(f"🎯 Level: {args.level.upper()}")
    print(f"📝 Description: {config['description']}")
    print(f"⏱️  Expected time: {config['expected_time']}")
    print(f"🔢 Iterations: {config['iterations']:,}")
    print(f"💾 Save interval: {config['save_interval']}")
    print(f"📂 Model name: {args.name}")
    
    # Verify GPU
    devices = jax.devices()
    print(f"\n🖥️  JAX devices: {devices}")
    
    if 'cuda' in str(devices).lower():
        print("✅ GPU DETECTED - Ready for training!")
    else:
        print("⚠️  WARNING: No GPU detected. Training will be slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Training cancelled.")
            return
    
    # Warning for long trainings
    if config['iterations'] >= 10000:
        print(f"\n⚠️  WARNING: This is a LONG training ({config['expected_time']})")
        print("   - Make sure your vast.ai instance won't terminate")
        print("   - Consider using 'screen' or 'tmux' for persistence")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("❌ Training cancelled.")
            return
    
    print(f"\n🚀 STARTING {args.level.upper()} TRAINING...")
    print("="*60)
    
    # Create trainer
    trainer = create_gpu_trainer('standard')
    
    # Start training
    start_time = time.time()
    
    try:
        trainer.train(
            num_iterations=config['iterations'],
            save_path=args.name,
            save_interval=config['save_interval']
        )
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"🚀 Speed: {config['iterations']/total_time:.1f} it/s")
        print(f"💾 Model saved: {args.name}_final.pkl")
        
        # Estimate quality based on iterations
        if config['iterations'] >= 10000:
            quality = "🏆 ELITE QUALITY"
        elif config['iterations'] >= 5000:
            quality = "🥇 PROFESSIONAL QUALITY"
        elif config['iterations'] >= 1000:
            quality = "🥈 GOOD QUALITY"
        else:
            quality = "🥉 BASIC QUALITY"
        
        print(f"🎯 Estimated quality: {quality}")
        
        # Usage instructions
        print(f"\n📖 HOW TO USE YOUR TRAINED MODEL:")
        print(f"   from trainer_mccfr_gpu_optimized import MCCFRTrainerGPU")
        print(f"   trainer = MCCFRTrainerGPU()")
        print(f"   trainer.load_model('{args.name}_final.pkl')")
        print(f"   # Now use trainer.strategy for decisions")
        
        # Suggest next steps
        if args.level == 'quick':
            print(f"\n💡 NEXT STEPS:")
            print(f"   - Try 'standard' level for better quality")
            print(f"   - Run: python launch_gpu_training.py standard")
        elif args.level == 'standard':
            print(f"\n💡 NEXT STEPS:")
            print(f"   - Try 'professional' level for tournament play")
            print(f"   - Run: python launch_gpu_training.py professional")
        elif config['iterations'] < 25000:
            print(f"\n💡 NEXT STEPS:")
            print(f"   - Try 'superhuman' level for maximum strength")
            print(f"   - Run: python launch_gpu_training.py superhuman")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  TRAINING INTERRUPTED by user")
        print(f"   Checkpoints saved in {args.name}_iter_*.pkl")
        print(f"   You can resume or use the latest checkpoint")
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED:")
        print(f"   Error: {str(e)}")
        print(f"   Check logs above for details")
        
        # Save emergency checkpoint if possible
        try:
            emergency_path = f"{args.name}_emergency.pkl"
            trainer.save_model(emergency_path)
            print(f"💾 Emergency checkpoint saved: {emergency_path}")
        except:
            print("❌ Could not save emergency checkpoint")

if __name__ == "__main__":
    main() 