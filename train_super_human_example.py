#!/usr/bin/env python3
"""
🎯 Super-Human Poker AI Training Example
=========================================

This script demonstrates how to train poker AI models at different levels:
- Standard: Quick testing and development
- Super-Human: Professional competition level  
- Pluribus-Level: Elite AI competition

Examples:
  python train_super_human_example.py --demo standard
  python train_super_human_example.py --demo super_human
  python train_super_human_example.py --demo pluribus_level
  python train_super_human_example.py --custom
"""

import argparse
import logging
from datetime import datetime
from poker_bot.core.trainer import create_super_human_trainer, SuperHumanTrainerConfig

# Configure logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def demo_standard_training():
    """
    Demo: Standard training for quick testing and development
    Expected: 40-60 IQ in 2-3 minutes
    """
    logger.info("🎯 DEMO: Standard Training")
    logger.info("Purpose: Quick testing and development")
    logger.info("Expected time: 2-3 minutes")
    logger.info("Expected IQ: 40-60/100")
    
    trainer = create_super_human_trainer("standard")
    trainer.train(
        num_iterations=100,
        save_path="demo_standard",
        save_interval=25,
        snapshot_iterations=[25, 50, 75, 100]
    )
    
    logger.info("✅ Standard training demo completed!")

def demo_super_human_training():
    """
    Demo: Super-human training for professional competition
    Expected: 70-85 IQ in 30-60 minutes
    """
    logger.info("🏆 DEMO: Super-Human Training")
    logger.info("Purpose: Professional competition level")
    logger.info("Expected time: 30-60 minutes")
    logger.info("Expected IQ: 70-85/100")
    logger.info("Features: Position + suited awareness, advanced evaluation")
    
    trainer = create_super_human_trainer("super_human")
    
    # Override for demo (shorter training)
    trainer.config.max_iterations = 500  # Reduced for demo
    
    trainer.train(
        num_iterations=500,
        save_path="demo_super_human",
        save_interval=100,
        snapshot_iterations=[125, 250, 375, 500]
    )
    
    logger.info("✅ Super-human training demo completed!")

def demo_pluribus_level_training():
    """
    Demo: Pluribus-level training for elite AI competition
    Expected: 85-95 IQ in 2-4 hours
    """
    logger.info("🚀 DEMO: Pluribus-Level Training")
    logger.info("Purpose: Elite AI competition")
    logger.info("Expected time: 2-4 hours")
    logger.info("Expected IQ: 85-95/100")
    logger.info("Features: All advanced concepts, extreme precision")
    logger.info("WARNING: This will take a long time!")
    
    # Ask for confirmation
    confirm = input("Continue with Pluribus-level training? (y/N): ")
    if confirm.lower() != 'y':
        logger.info("Pluribus-level training cancelled.")
        return
    
    trainer = create_super_human_trainer("pluribus_level")
    
    # Override for demo (shorter training than full 5000)
    trainer.config.max_iterations = 1000  # Reduced for demo
    
    trainer.train(
        num_iterations=1000,
        save_path="demo_pluribus",
        save_interval=200,
        snapshot_iterations=[250, 500, 750, 1000]
    )
    
    logger.info("✅ Pluribus-level training demo completed!")

def custom_training_example():
    """
    Example: Custom training configuration
    Shows how to create and configure training manually
    """
    logger.info("🛠️ CUSTOM TRAINING EXAMPLE")
    
    # Create custom configuration
    config = SuperHumanTrainerConfig()
    
    # Customize parameters
    config.batch_size = 256  # Larger batch for better learning
    config.position_awareness_factor = 0.5  # Strong position learning
    config.suited_awareness_factor = 0.4    # Strong suited learning
    config.strong_hand_threshold = 4200     # More selective premium hands
    config.weak_hand_threshold = 1300       # More selective folding
    
    logger.info("Custom configuration:")
    logger.info(f"  - Batch size: {config.batch_size}")
    logger.info(f"  - Position factor: {config.position_awareness_factor}")
    logger.info(f"  - Suited factor: {config.suited_awareness_factor}")
    logger.info(f"  - Strong threshold: {config.strong_hand_threshold}")
    logger.info(f"  - Weak threshold: {config.weak_hand_threshold}")
    
    # Create trainer with custom config
    from poker_bot.core.trainer import PokerTrainer
    trainer = PokerTrainer(config)
    
    # Custom training with adaptive snapshots
    iterations = 300
    snapshots = [50, 100, 150, 200, 250, 300]  # More frequent snapshots
    
    trainer.train(
        num_iterations=iterations,
        save_path="custom_training",
        save_interval=50,
        snapshot_iterations=snapshots
    )
    
    logger.info("✅ Custom training example completed!")

def production_training_template():
    """
    Production template: Ready-to-use training for serious competition
    """
    logger.info("🏭 PRODUCTION TRAINING TEMPLATE")
    logger.info("This is a template for serious competition training")
    
    # Generate unique model name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"production_bot_{timestamp}"
    
    logger.info(f"Model name: {model_name}")
    
    # Create super-human trainer
    trainer = create_super_human_trainer("super_human")
    
    # Production settings
    production_iterations = 2000
    production_batch_size = 512  # Large batch for stability
    
    trainer.config.batch_size = production_batch_size
    
    # Production snapshots for analysis
    snapshots = [
        production_iterations // 8,   # 12.5%
        production_iterations // 4,   # 25%
        production_iterations // 2,   # 50%
        3 * production_iterations // 4,  # 75%
        7 * production_iterations // 8,  # 87.5%
        production_iterations         # 100%
    ]
    
    logger.info("Production training parameters:")
    logger.info(f"  - Iterations: {production_iterations:,}")
    logger.info(f"  - Batch size: {production_batch_size}")
    logger.info(f"  - Snapshots: {snapshots}")
    logger.info(f"  - Estimated time: 45-90 minutes")
    logger.info(f"  - Target IQ: 75-85/100")
    
    # Confirm before starting
    confirm = input(f"Start production training for {model_name}? (y/N): ")
    if confirm.lower() != 'y':
        logger.info("Production training cancelled.")
        return
    
    # Start production training
    trainer.train(
        num_iterations=production_iterations,
        save_path=model_name,
        save_interval=200,  # Save every 200 iterations
        snapshot_iterations=snapshots
    )
    
    logger.info(f"✅ Production training completed: {model_name}")
    logger.info(f"Models saved: {model_name}_*.pkl")

def main():
    parser = argparse.ArgumentParser(
        description='🎯 Super-Human Poker AI Training Examples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demos
  python train_super_human_example.py --demo standard
  python train_super_human_example.py --demo super_human
  python train_super_human_example.py --demo pluribus_level
  
  # Advanced usage  
  python train_super_human_example.py --custom
  python train_super_human_example.py --production
  
  # Show all options
  python train_super_human_example.py --help
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--demo', choices=['standard', 'super_human', 'pluribus_level'],
                      help='Run a training demo at specified level')
    group.add_argument('--custom', action='store_true',
                      help='Run custom training configuration example')
    group.add_argument('--production', action='store_true',
                      help='Run production training template')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🎯 SUPER-HUMAN POKER AI TRAINING EXAMPLES")
    logger.info("="*80)
    
    try:
        if args.demo:
            if args.demo == 'standard':
                demo_standard_training()
            elif args.demo == 'super_human':
                demo_super_human_training()
            elif args.demo == 'pluribus_level':
                demo_pluribus_level_training()
                
        elif args.custom:
            custom_training_example()
            
        elif args.production:
            production_training_template()
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise
    
    logger.info("\n" + "="*80)
    logger.info("🎉 TRAINING EXAMPLE SESSION COMPLETED")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("1. Analyze the Poker IQ evolution in the logs")
    logger.info("2. Compare different models using the diagnostics")
    logger.info("3. Use the best model for competition")
    logger.info("4. Continue training with more iterations if needed")

if __name__ == "__main__":
    main() 