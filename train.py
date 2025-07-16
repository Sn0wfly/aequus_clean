#!/usr/bin/env python3
"""
🚀 Poker AI Training CLI
Train poker models without editing code
"""

import argparse
import sys
import os
import time
from poker_bot.core.trainer import PokerTrainer, TrainerConfig, SuperHumanTrainerConfig

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="🃏 Train Poker AI models using CFR algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 iterations)
  python train.py --iterations 10 --output quick_test
  
  # Standard training (1000 iterations)
  python train.py --iterations 1000 --output standard_model
  
  # Super-human training (2000 iterations with snapshots)
  python train.py --config superhuman --iterations 2000 --output elite_model --snapshots 500 1000 1500 2000
  
  # Professional training (5000 iterations)
  python train.py --config superhuman --iterations 5000 --batch-size 256 --output professional_model
        """)
    
    # Required arguments
    parser.add_argument('--iterations', '-i', type=int, required=True,
                       help='Number of training iterations')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output model name/path (without .pkl extension)')
    
    # Configuration options
    parser.add_argument('--config', '-c', choices=['standard', 'superhuman'], default='standard',
                       help='Training configuration preset (default: standard)')
    
    # Training parameters
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                       help='Batch size for training (default: config dependent)')
    parser.add_argument('--save-interval', '-s', type=int, default=None,
                       help='Save checkpoint every N iterations (default: iterations/10)')
    
    # Evaluation options
    parser.add_argument('--snapshots', nargs='*', type=int, default=None,
                       help='Iteration numbers for poker IQ snapshots (default: 25%%, 50%%, 75%%, 100%%)')
    parser.add_argument('--no-snapshots', action='store_true',
                       help='Disable poker IQ snapshots for faster training')
    
    # Advanced options
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (default: config dependent)')
    parser.add_argument('--max-info-sets', type=int, default=50000,
                       help='Maximum number of information sets (default: 50000)')
    
    # Validation options
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip pre-training validation (faster startup)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output during training')
    
    return parser

def validate_args(args):
    """Validate command line arguments"""
    errors = []
    
    if args.iterations < 1:
        errors.append("Iterations must be positive")
    
    if args.iterations > 10000:
        print("⚠️  Warning: >10000 iterations will take several hours")
    
    if args.batch_size and args.batch_size < 16:
        errors.append("Batch size should be >= 16 for stable training")
    
    if args.save_interval and args.save_interval > args.iterations:
        errors.append("Save interval cannot be larger than total iterations")
    
    if args.snapshots and any(s > args.iterations for s in args.snapshots):
        errors.append("Snapshot iterations cannot exceed total iterations")
    
    if errors:
        print("❌ Argument validation errors:")
        for error in errors:
            print(f"   - {error}")
        sys.exit(1)

def create_config(args):
    """Create training configuration from arguments"""
    if args.config == 'superhuman':
        config = SuperHumanTrainerConfig()
        print("🏆 Using Super-Human configuration")
    else:
        config = TrainerConfig()
        print("⚡ Using Standard configuration")
    
    # Override config with command line arguments
    if args.batch_size:
        config.batch_size = args.batch_size
        print(f"   Batch size: {config.batch_size}")
    
    if args.learning_rate:
        config.learning_rate = args.learning_rate
        print(f"   Learning rate: {config.learning_rate}")
    
    if args.max_info_sets != 50000:
        config.max_info_sets = args.max_info_sets
        print(f"   Max info sets: {config.max_info_sets}")
    
    return config

def setup_snapshots(args):
    """Setup snapshot iterations"""
    if args.no_snapshots:
        return []
    
    if args.snapshots is not None:
        return args.snapshots
    
    # Default snapshots: 25%, 50%, 75%, 100%
    snapshots = [
        args.iterations // 4,
        args.iterations // 2, 
        3 * args.iterations // 4,
        args.iterations
    ]
    return [s for s in snapshots if s > 0]

def estimate_training_time(iterations, batch_size):
    """Estimate training time"""
    # Based on observed performance: ~100 iter/s for batch_size=128
    base_rate = 100  # iterations per second
    scale_factor = batch_size / 128
    adjusted_rate = base_rate / scale_factor
    
    estimated_seconds = iterations / adjusted_rate
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"~{estimated_seconds/60:.1f} minutes"
    else:
        return f"~{estimated_seconds/3600:.1f} hours"

def main():
    """Main training function"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🃏 POKER AI TRAINER")
    print("="*50)
    
    # Validate arguments
    validate_args(args)
    
    # Create configuration
    config = create_config(args)
    
    # Setup save interval
    save_interval = args.save_interval or max(1, args.iterations // 10)
    
    # Setup snapshots
    snapshots = setup_snapshots(args)
    
    # Estimate time
    time_estimate = estimate_training_time(args.iterations, config.batch_size)
    
    # Print training plan
    print(f"\n📋 TRAINING PLAN:")
    print(f"   🎯 Iterations: {args.iterations}")
    print(f"   📦 Batch size: {config.batch_size}")
    print(f"   💾 Save every: {save_interval} iterations")
    print(f"   📸 Snapshots: {snapshots if snapshots else 'Disabled'}")
    print(f"   ⏱️  Estimated time: {time_estimate}")
    print(f"   📁 Output: {args.output}_final.pkl")
    
    # Confirm for long trainings
    if args.iterations >= 1000:
        response = input(f"\n⚠️  This will train for {args.iterations} iterations ({time_estimate}). Continue? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled")
            sys.exit(0)
    
    # Create trainer
    print(f"\n🚀 STARTING TRAINING...")
    print(f"⏰ Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    trainer = PokerTrainer(config)
    
    try:
        # Run training
        start_time = time.time()
        
        trainer.train(
            num_iterations=args.iterations,
            save_path=f"models/{args.output}",
            save_interval=save_interval,
            snapshot_iterations=snapshots if snapshots else []
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Success message
        print(f"\n🎉 TRAINING COMPLETED!")
        print(f"⏱️  Actual time: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"🚀 Speed: {args.iterations/duration:.1f} iter/s")
        print(f"📁 Model saved: models/{args.output}_final.pkl")
        
        # Suggest next steps
        print(f"\n📋 NEXT STEPS:")
        print(f"   # Evaluate model performance:")
        print(f"   python evaluate_model.py models/{args.output}_final.pkl")
        print(f"   ")
        print(f"   # Run unit tests:")
        print(f"   python test_poker_concepts.py")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 