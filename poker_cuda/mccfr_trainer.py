"""
Production MCCFR Trainer for Texas Hold'em

This module provides a production-ready MCCFR trainer with:
- Multiple sampling schemes (External, Outcome)
- Comprehensive logging and metrics
- Strategy checkpointing and loading
- Performance monitoring
- Configurable training parameters
"""

import os
import json
import time
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .mccfr_core import MCCFRBase, ExternalSamplingMCCFR, OutcomeSamplingMCCFR, InfoSet
from .poker_game import TexasHoldemHistory


@dataclass
class MCCFRConfig:
    """Configuration for MCCFR training."""
    # Training parameters
    iterations: int = 10000
    algorithm: str = "external"  # "external" or "outcome"
    num_players: int = 2
    
    # Game parameters
    small_blind: int = 1
    big_blind: int = 2
    starting_stacks: int = 200
    
    # Outcome sampling specific
    epsilon: float = 0.6
    
    # Logging and checkpointing
    log_interval: int = 1000
    checkpoint_interval: int = 5000
    save_strategy: bool = True
    
    # Output directories
    output_dir: str = "mccfr_results"
    checkpoint_dir: str = "checkpoints"
    
    # Performance monitoring
    evaluate_exploitability: bool = True
    exploitability_interval: int = 1000
    
    def __post_init__(self):
        """Create output directories."""
        Path(self.output_dir).mkdir(exist_ok=True)
        Path(os.path.join(self.output_dir, self.checkpoint_dir)).mkdir(exist_ok=True)


class MCCFRTrainer:
    """
    Production MCCFR trainer with comprehensive features.
    
    Features:
    - Multiple MCCFR algorithms
    - Detailed performance tracking
    - Strategy saving/loading
    - Progress monitoring
    - Exploitability evaluation
    """
    
    def __init__(self, config: MCCFRConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize algorithm
        if config.algorithm.lower() == "external":
            self.algorithm = ExternalSamplingMCCFR(config.num_players)
        elif config.algorithm.lower() == "outcome":
            self.algorithm = OutcomeSamplingMCCFR(config.num_players, config.epsilon)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
        
        # Training metrics
        self.metrics = {
            'iterations': [],
            'exploitability': [],
            'training_time': [],
            'iteration_time': [],
            'info_sets_count': [],
            'memory_usage': []
        }
        
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        log_file = os.path.join(self.config.output_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def train(self) -> Dict[str, InfoSet]:
        """
        Main training loop with comprehensive monitoring.
        
        Returns:
            Dictionary of trained information sets
        """
        self.logger.info("="*60)
        self.logger.info("STARTING MCCFR TRAINING")
        self.logger.info("="*60)
        self.logger.info(f"Algorithm: {self.config.algorithm}")
        self.logger.info(f"Iterations: {self.config.iterations}")
        self.logger.info(f"Players: {self.config.num_players}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        self.start_time = time.time()
        
        # Create root game state
        root_history = TexasHoldemHistory(
            num_players=self.config.num_players,
            small_blind=self.config.small_blind,
            big_blind=self.config.big_blind,
            starting_stacks=self.config.starting_stacks
        )
        
        try:
            # Training loop with monitoring
            for iteration in range(self.config.iterations):
                iteration_start = time.time()
                
                # Train single iteration
                self._train_iteration(root_history)
                
                iteration_time = time.time() - iteration_start
                
                # Periodic logging and evaluation
                if (iteration + 1) % self.config.log_interval == 0:
                    self._log_progress(iteration + 1, iteration_time)
                
                if (iteration + 1) % self.config.exploitability_interval == 0:
                    self._evaluate_exploitability(root_history, iteration + 1)
                
                if (iteration + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(iteration + 1)
                
                # Update metrics
                self.metrics['iteration_time'].append(iteration_time)
            
            # Final evaluation and saving
            self._finalize_training(root_history)
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self._save_checkpoint(iteration + 1, suffix="_interrupted")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        return self.algorithm.info_sets
    
    def _train_iteration(self, root_history: TexasHoldemHistory):
        """Train a single iteration."""
        if isinstance(self.algorithm, ExternalSamplingMCCFR):
            # External sampling trains each player separately
            for player in range(self.config.num_players):
                self.algorithm._external_sampling_update(root_history, player, 1.0, 1.0)
        elif isinstance(self.algorithm, OutcomeSamplingMCCFR):
            # Outcome sampling trains each player separately  
            for player in range(self.config.num_players):
                self.algorithm._outcome_sampling_update(root_history, player, 1.0, 1.0, 1.0)
        
        self.algorithm.iteration += 1
    
    def _log_progress(self, iteration: int, iteration_time: float):
        """Log training progress."""
        elapsed_time = time.time() - self.start_time
        avg_iteration_time = np.mean(self.metrics['iteration_time'][-1000:])  # Last 1000 iterations
        estimated_remaining = (self.config.iterations - iteration) * avg_iteration_time
        
        info_sets_count = len(self.algorithm.info_sets)
        
        self.logger.info(f"Iteration {iteration:,}/{self.config.iterations:,}")
        self.logger.info(f"  Elapsed time: {elapsed_time:.1f}s")
        self.logger.info(f"  Iteration time: {iteration_time:.4f}s (avg: {avg_iteration_time:.4f}s)")
        self.logger.info(f"  Information sets: {info_sets_count:,}")
        self.logger.info(f"  Estimated remaining: {estimated_remaining:.1f}s")
        
        # Update metrics
        self.metrics['iterations'].append(iteration)
        self.metrics['training_time'].append(elapsed_time)
        self.metrics['info_sets_count'].append(info_sets_count)
    
    def _evaluate_exploitability(self, root_history: TexasHoldemHistory, iteration: int):
        """Evaluate current strategy exploitability."""
        if not self.config.evaluate_exploitability:
            return
        
        try:
            exploitability_start = time.time()
            exploitability = self.algorithm.calculate_exploitability(root_history)
            eval_time = time.time() - exploitability_start
            
            self.logger.info(f"  Exploitability: {exploitability:.6f} (evaluated in {eval_time:.2f}s)")
            self.metrics['exploitability'].append(exploitability)
            
        except Exception as e:
            self.logger.warning(f"Could not evaluate exploitability: {e}")
    
    def _save_checkpoint(self, iteration: int, suffix: str = ""):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            self.config.checkpoint_dir,
            f"checkpoint_iter_{iteration}{suffix}.pkl"
        )
        
        checkpoint_data = {
            'iteration': iteration,
            'config': asdict(self.config),
            'info_sets': self.algorithm.info_sets,
            'metrics': self.metrics,
            'algorithm_type': type(self.algorithm).__name__
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _finalize_training(self, root_history: TexasHoldemHistory):
        """Finalize training with final evaluation and saving."""
        total_time = time.time() - self.start_time
        
        self.logger.info("="*60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("="*60)
        self.logger.info(f"Total training time: {total_time:.1f}s")
        self.logger.info(f"Total iterations: {self.config.iterations:,}")
        self.logger.info(f"Average iteration time: {np.mean(self.metrics['iteration_time']):.4f}s")
        self.logger.info(f"Information sets created: {len(self.algorithm.info_sets):,}")
        
        # Final exploitability evaluation
        if self.config.evaluate_exploitability:
            try:
                final_exploitability = self.algorithm.calculate_exploitability(root_history)
                self.logger.info(f"Final exploitability: {final_exploitability:.6f}")
            except Exception as e:
                self.logger.warning(f"Could not evaluate final exploitability: {e}")
        
        # Save final strategy
        if self.config.save_strategy:
            self._save_final_strategy()
        
        # Save final checkpoint
        self._save_checkpoint(self.config.iterations, "_final")
        
        # Save training metrics
        self._save_metrics()
    
    def _save_final_strategy(self):
        """Save final average strategy."""
        strategy_path = os.path.join(self.config.output_dir, "final_strategy.json")
        
        try:
            strategy = self.algorithm.get_strategy_profile()
            
            # Convert to JSON-serializable format
            json_strategy = {}
            for info_set_key, action_probs in strategy.items():
                json_strategy[info_set_key] = action_probs
            
            with open(strategy_path, 'w') as f:
                json.dump(json_strategy, f, indent=2)
            
            self.logger.info(f"Final strategy saved: {strategy_path}")
            
            # Also save strategy summary
            self._save_strategy_summary(json_strategy)
            
        except Exception as e:
            self.logger.error(f"Failed to save final strategy: {e}")
    
    def _save_strategy_summary(self, strategy: Dict[str, Dict[str, float]]):
        """Save human-readable strategy summary."""
        summary_path = os.path.join(self.config.output_dir, "strategy_summary.txt")
        
        try:
            with open(summary_path, 'w') as f:
                f.write("MCCFR STRATEGY SUMMARY\n")
                f.write("="*50 + "\n\n")
                
                # Group by betting round
                preflop_sets = []
                flop_sets = []
                turn_sets = []
                river_sets = []
                
                for info_set_key, action_probs in strategy.items():
                    parts = info_set_key.split('|')
                    if len(parts) >= 2:
                        community = parts[1]
                        if len(community) == 0:
                            preflop_sets.append((info_set_key, action_probs))
                        elif len(community) == 6:  # 3 cards = 6 chars
                            flop_sets.append((info_set_key, action_probs))
                        elif len(community) == 8:  # 4 cards = 8 chars
                            turn_sets.append((info_set_key, action_probs))
                        elif len(community) == 10:  # 5 cards = 10 chars
                            river_sets.append((info_set_key, action_probs))
                
                # Write summaries
                self._write_round_summary(f, "PREFLOP", preflop_sets)
                self._write_round_summary(f, "FLOP", flop_sets)
                self._write_round_summary(f, "TURN", turn_sets)
                self._write_round_summary(f, "RIVER", river_sets)
            
            self.logger.info(f"Strategy summary saved: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save strategy summary: {e}")
    
    def _write_round_summary(self, file, round_name: str, info_sets: List[Tuple[str, Dict[str, float]]]):
        """Write summary for a betting round."""
        if not info_sets:
            return
        
        file.write(f"{round_name} STRATEGY\n")
        file.write("-" * 30 + "\n")
        
        for info_set_key, action_probs in info_sets[:10]:  # Show first 10
            file.write(f"Info Set: {info_set_key}\n")
            for action, prob in action_probs.items():
                file.write(f"  {action}: {prob:.3f}\n")
            file.write("\n")
        
        if len(info_sets) > 10:
            file.write(f"... and {len(info_sets) - 10} more info sets\n")
        
        file.write("\n")
    
    def _save_metrics(self):
        """Save training metrics."""
        metrics_path = os.path.join(self.config.output_dir, "training_metrics.json")
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.logger.info(f"Training metrics saved: {metrics_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.
        
        Returns:
            Iteration number from checkpoint
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore algorithm state
            self.algorithm.info_sets = checkpoint_data['info_sets']
            self.algorithm.iteration = checkpoint_data['iteration']
            
            # Restore metrics
            self.metrics = checkpoint_data['metrics']
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            self.logger.info(f"Resumed from iteration {checkpoint_data['iteration']}")
            
            return checkpoint_data['iteration']
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def evaluate_strategy(self, root_history: TexasHoldemHistory, num_hands: int = 1000) -> Dict[str, float]:
        """
        Evaluate trained strategy through simulation.
        
        Args:
            root_history: Starting game state
            num_hands: Number of hands to simulate
            
        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating strategy over {num_hands} hands...")
        
        results = {
            'total_hands': num_hands,
            'player_winnings': [0.0] * self.config.num_players,
            'average_pot_size': 0.0,
            'hands_to_showdown': 0,
            'average_hand_length': 0.0
        }
        
        total_pot = 0.0
        total_actions = 0
        
        for hand in range(num_hands):
            # Simulate hand using current strategy
            history = TexasHoldemHistory(
                self.config.num_players,
                self.config.small_blind,
                self.config.big_blind,
                self.config.starting_stacks
            )
            
            actions_in_hand = 0
            
            while not history.is_terminal():
                if history.is_chance_node():
                    action = history.sample_chance()
                    history = history.create_child(action)
                else:
                    player = history.get_player()
                    info_set = self.algorithm.get_info_set(history, player)
                    avg_strategy = info_set.get_average_strategy()
                    
                    # Sample action according to average strategy
                    actions = list(avg_strategy.keys())
                    probs = list(avg_strategy.values())
                    action = np.random.choice(actions, p=probs)
                    
                    history = history.create_child(action)
                    actions_in_hand += 1
            
            # Record results
            total_pot += history.pot
            total_actions += actions_in_hand
            
            if len(history.community_cards) == 5:
                results['hands_to_showdown'] += 1
            
            # Calculate winnings
            for player in range(self.config.num_players):
                utility = history.get_utility(player)
                results['player_winnings'][player] += utility
        
        # Calculate averages
        results['average_pot_size'] = total_pot / num_hands
        results['average_hand_length'] = total_actions / num_hands
        results['showdown_rate'] = results['hands_to_showdown'] / num_hands
        
        # Average winnings
        for player in range(self.config.num_players):
            results['player_winnings'][player] /= num_hands
        
        self.logger.info("Strategy evaluation completed:")
        for player in range(self.config.num_players):
            self.logger.info(f"  Player {player} average winnings: {results['player_winnings'][player]:.3f}")
        self.logger.info(f"  Average pot size: {results['average_pot_size']:.2f}")
        self.logger.info(f"  Showdown rate: {results['showdown_rate']:.3f}")
        self.logger.info(f"  Average hand length: {results['average_hand_length']:.1f} actions")
        
        return results


def create_default_config() -> MCCFRConfig:
    """Create default configuration for MCCFR training."""
    return MCCFRConfig(
        iterations=10000,
        algorithm="external",
        log_interval=1000,
        checkpoint_interval=5000,
        evaluate_exploitability=True,
        exploitability_interval=1000
    ) 