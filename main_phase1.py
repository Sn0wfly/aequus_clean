#!/usr/bin/env python3
"""
Phase 1 Enhanced Training Script for Aequus
Production-ready poker AI with enhanced evaluation and ICM modeling
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from poker_bot.core.trainer import PokerTrainer, TrainerConfig
from poker_bot.core.simulation import batch_simulate_real_holdem
from poker_bot.core.enhanced_eval import EnhancedHandEvaluator
from poker_bot.core.icm_modeling import ICMModel
import jax.random as jr
import time
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_phase1_config():
    """Create Phase 1 enhanced configuration"""
    return TrainerConfig(
        batch_size=32768,        # H100 optimized
        learning_rate=0.05,      # Slightly reduced for stability
        temperature=1.0,
        num_actions=14,          # Expanded action set
        dtype='bfloat16',
        accumulation_dtype='float32',
        max_info_sets=50000,     # Increased from 25k
        growth_factor=1.5,
        chunk_size=20000,
        gpu_bucket=False,
        use_pluribus_bucketing=True,
        N_rollouts=100           # Reduced for Phase 1 testing
    )

def load_phase1_config(config_path: str = "config/phase1_config.yaml") -> dict:
    """Load Phase 1 configuration from YAML"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Phase 1 config loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"⚠️ Phase 1 config not found, using defaults")
        return create_phase1_config()

def train_phase1(config: TrainerConfig, num_iterations: int, save_every: int, save_path: str):
    """Enhanced training with Phase 1 improvements"""
    logger.info("🚀 Starting Phase 1 Enhanced Training")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Batch size: {config.batch_size:,}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Num actions: {config.num_actions}")
    logger.info(f"  Max info sets: {config.max_info_sets:,}")
    logger.info(f"  N rollouts: {config.N_rollouts}")
    logger.info(f"  Enhanced evaluation: ✅ Enabled")
    logger.info(f"  ICM modeling: ✅ Enabled")
    logger.info(f"  Iterations: {num_iterations:,}")
    logger.info(f"  Save every: {save_every}")
    logger.info(f"  Save path: {save_path}")
    
    # Initialize enhanced components
    enhanced_eval = EnhancedHandEvaluator()
    icm_model = ICMModel()
    
    # Initialize trainer
    trainer = PokerTrainer(config)
    
    # Game configuration
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Training loop
    start_time = time.time()
    
    for iteration in range(num_iterations):
        # Generate random keys for this iteration
        rng_key = jr.PRNGKey(iteration)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        # Simulate games
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        
        # Enhanced training step with Phase 1 improvements
        results = trainer.train_step(game_results, iteration=iteration)
        
        # Log enhanced metrics
        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (iteration + 1) / elapsed
            eta = (num_iterations - iteration - 1) / rate if rate > 0 else 0
            
            logger.info(f"Iteration {iteration+1:,}/{num_iterations:,}")
            logger.info(f"  Unique info sets: {results['unique_info_sets']:,}")
            logger.info(f"  Info sets processed: {results['info_sets_processed']:,}")
            logger.info(f"  Avg payoff: {results['avg_payoff']:.3f}")
            logger.info(f"  Strategy entropy: {float(results['strategy_entropy']):.3f}")
            logger.info(f"  Elapsed: {elapsed:.1f}s, Rate: {rate:.1f} it/s, ETA: {eta:.1f}s")
            logger.info("-" * 40)
        
        # Save checkpoint
        if (iteration + 1) % save_every == 0:
            checkpoint_path = f"{save_path}_phase1_checkpoint_{iteration+1}.pkl"
            trainer.save_model(checkpoint_path)
            logger.info(f"💾 Phase 1 checkpoint saved: {checkpoint_path}")
    
    # Final save
    final_path = f"{save_path}_phase1_final.pkl"
    trainer.save_model(final_path)
    
    total_time = time.time() - start_time
    logger.info("🎉 Phase 1 training completed!")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.info(f"Final model saved: {final_path}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Enhanced Training")
    parser.add_argument("--config", choices=["phase1", "debug"], 
                       default="phase1", help="Training configuration")
    parser.add_argument("--iterations", type=int, default=10000, 
                       help="Number of training iterations")
    parser.add_argument("--save_every", type=int, default=1000, 
                       help="Save checkpoint every N iterations")
    parser.add_argument("--save_path", type=str, default="aequus_phase1", 
                       help="Base path for saving models")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config == "phase1":
        config = create_phase1_config()
    else:
        config = create_phase1_config()  # Use debug settings
    
    # Train model
    trainer = train_phase1(config, args.iterations, args.save_every, args.save_path)
    
    return trainer

if __name__ == "__main__":
    main()