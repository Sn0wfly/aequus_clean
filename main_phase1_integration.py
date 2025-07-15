"""
🎯 Elite Poker AI - Phase 1 Integration
Complete NLHE game engine with training pipeline
"""

import jax
import jax.numpy as jnp
import yaml
import logging
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from poker_bot.core.elite_gto_trainer import EliteTrainingPipeline
from poker_bot.core.jax_game_engine import simulate_game, batch_simulate
from poker_bot.core.betting_tree import EliteBettingTree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase1Integration:
    """Complete Phase 1 integration for elite poker AI"""
    
    def __init__(self, config_path: str = "poker_bot/core/elite_config.yaml"):
        """Initialize Phase 1 integration"""
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("🚀 Initializing Elite Poker AI - Phase 1")
        logger.info("=" * 60)
        
        # Initialize components
        self.game_engine = EliteTrainingPipeline(self.config['game_engine'])
        self.betting_tree = EliteBettingTree()
        
        # Performance metrics
        self.metrics = {
            'games_simulated': 0,
            'training_iterations': 0,
            'total_time': 0.0
        }
        
    def validate_system(self) -> bool:
        """Complete system validation"""
        
        logger.info("🔍 Validating Phase 1 System...")
        
        validation_results = {
            'game_engine': self._validate_game_engine(),
            'betting_tree': self._validate_betting_tree(),
            'integration': self._validate_integration()
        }
        
        all_passed = all(validation_results.values())
        
        if all_passed:
            logger.info("✅ Phase 1 System Validation PASSED")
        else:
            logger.error("❌ Phase 1 System Validation FAILED")
            
        return all_passed
    
    def _validate_game_engine(self) -> bool:
        """Validate game engine"""
        try:
            rng_key = jax.random.PRNGKey(42)
            
            # Test single game
            result = simulate_game(rng_key, 6, 1.0, 2.0, 100.0)
            
            # Validate dimensions
            assert result['payoffs'].shape == (6,)
            assert result['hole_cards'].shape == (6, 2)
            assert result['final_community'].shape == (5,)
            
            # Validate zero-sum
            assert abs(jnp.sum(result['payoffs'])) < 1e-3
            
            # Test batch simulation
            rng_keys = jax.random.split(rng_key, 100)
            batch_results = batch_simulate(rng_keys, 6, 1.0, 2.0, 100.0)
            
            assert batch_results['payoffs'].shape == (100, 6)
            
            logger.info("✅ Game engine validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Game engine validation failed: {e}")
            return False
    
    def _validate_betting_tree(self) -> bool:
        """Validate betting tree"""
        try:
            # Test tree construction
            tree = self.betting_tree.build_tree(6)
            
            # Validate tree structure
            assert tree is not None
            
            logger.info("✅ Betting tree validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Betting tree validation failed: {e}")
            return False
    
    def _validate_integration(self) -> bool:
        """Validate integration"""
        try:
            # Test training pipeline
            rng_key = jax.random.PRNGKey(42)
            training_batch = self.game_engine.generate_training_batch(rng_key)
            
            # Validate batch structure
            assert 'features' in training_batch
            assert 'targets' in training_batch
            assert 'game_data' in training_batch
            
            logger.info("✅ Integration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration validation failed: {e}")
            return False
    
    def run_training_session(self, num_iterations: int = 1000) -> Dict:
        """Run complete training session"""
        
        logger.info(f"🎯 Starting training session: {num_iterations} iterations")
        
        start_time = time.time()
        
        # Generate training data
        rng_key = jax.random.PRNGKey(int(time.time()))
        training_data = self.game_engine.generate_training_batch(rng_key)
        
        # Training metrics
        metrics = {
            'games_processed': training_data['game_data']['payoffs'].shape[0],
            'average_payoff': float(jnp.mean(training_data['targets'])),
            'training_time': time.time() - start_time,
            'games_per_second': training_data['game_data']['payoffs'].shape[0] / (time.time() - start_time)
        }
        
        logger.info(f"✅ Training session completed")
        logger.info(f"   Games processed: {metrics['games_processed']:,}")
        logger.info(f"   Average payoff: {metrics['average_payoff']:.3f}")
        logger.info(f"   Games/sec: {metrics['games_per_second']:.1f}")
        
        return metrics
    
    def benchmark_performance(self, num_games: int = 10000) -> Dict:
        """Benchmark system performance"""
        
        logger.info(f"🏃 Running performance benchmark: {num_games} games")
        
        rng_key = jax.random.PRNGKey(42)
        
        # Warm up
        _ = simulate_game(rng_key, 6, 1.0, 2.0, 100.0)
        
        # Benchmark
        start_time = time.time()
        
        rng_keys = jax.random.split(rng_key, num_games)
        results = batch_simulate(rng_keys, 6, 1.0, 2.0, 100.0)
        
        elapsed = time.time() - start_time
        
        metrics = {
            'total_games': num_games,
            'total_time': elapsed,
            'games_per_second': num_games / elapsed,
            'throughput_mb': (num_games * 6 * 8) / (elapsed * 1024 * 1024)  # 8 bytes per float
        }
        
        logger.info("🎯 Performance Benchmark Results")
        logger.info("=" * 40)
        logger.info(f"Total games: {metrics['total_games']:,}")
        logger.info(f"Total time: {metrics['total_time']:.2f}s")
        logger.info(f"Games/sec: {metrics['games_per_second']:.1f}")
        logger.info(f"Throughput: {metrics['throughput_mb']:.2f} MB/s")
        
        return metrics
    
    def generate_elite_dataset(self, num_games: int = 100000) -> Dict:
        """Generate elite training dataset"""
        
        logger.info(f"📊 Generating elite dataset: {num_games} games")
        
        rng_key = jax.random.PRNGKey(42)
        
        # Generate in batches to manage memory
        batch_size = 10000
        num_batches = (num_games + batch_size - 1) // batch_size
        
        all_data = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_games)
            batch_games = end_idx - start_idx
            
            rng_keys = jax.random.split(rng_key, batch_games)
            batch_data = batch_simulate(
                rng_keys, 
                self.num_players,
                self.small_blind,
                self.big_blind,
                self.starting_stack
            )
            
            all_data.append(batch_data)
            rng_key = jax.random.fold_in(rng_key, batch_idx)
            
            logger.info(f"   Batch {batch_idx + 1}/{num_batches} completed")
        
        # Combine all data
        combined_data = {
            key: jnp.concatenate([batch[key] for batch in all_data])
            for key in all_data[0].keys()
        }
        
        logger.info(f"✅ Elite dataset generated: {combined_data['payoffs'].shape[0]:,} games")
        
        return combined_data
    
    def save_checkpoint(self, data: Dict, path: str):
        """Save training checkpoint"""
        
        import pickle
        
        checkpoint = {
            'config': self.config,
            'data': data,
            'metrics': self.metrics,
            'timestamp': time.time()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def run_phase1_demo(self):
        """Run complete Phase 1 demonstration"""
        
        logger.info("🎮 Running Phase 1 Demo")
        logger.info("=" * 60)
        
        # 1. Validate system
        if not self.validate_system():
            logger.error("System validation failed")
            return
        
        # 2. Run performance benchmark
        perf_metrics = self.benchmark_performance(1000)
        
        # 3. Run training session
        train_metrics = self.run_training_session(100)
        
        # 4. Generate sample dataset
        sample_data = self.generate_elite_dataset(1000)
        
        # 5. Save results
        self.save_checkpoint(sample_data, "models/phase1_sample_data.pkl")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PHASE 1 DEMO SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Performance: {perf_metrics['games_per_second']:.1f} games/sec")
        logger.info(f"Training: {train_metrics['games_per_second']:.1f} games/sec")
        logger.info(f"Dataset: {sample_data['payoffs'].shape[0]:,} games")
        logger.info("✅ Phase 1 Integration Complete!")

def main():
    """Main execution"""
    
    # Create integration instance
    integration = Phase1Integration()
    
    # Run demo
    integration.run_phase1_demo()
    
    logger.info("\n🎉 Phase 1 Ready for Training!")
    logger.info("Next steps:")
    logger.info("1. Run full training: python main_phase1_integration.py --train")
    logger.info("2. Generate large dataset: python main_phase1_integration.py --dataset 100000")
    logger.info("3. Benchmark performance: python main_phase1_integration.py --benchmark")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Elite Poker AI Phase 1')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--dataset', type=int, default=1000, help='Generate dataset size')
    parser.add_argument('--config', type=str, default='poker_bot/core/elite_config.yaml', 
                       help='Configuration file')
    
    args = parser.parse_args()
    
    integration = Phase1Integration(args.config)
    
    if args.benchmark:
        integration.benchmark_performance(10000)
    elif args.train:
        integration.run_training_session(1000)
    else:
        integration.run_phase1_demo()