"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) Core Framework

Based on "Monte Carlo Sampling for Regret Minimization in Extensive Games"
by Lanctot et al. (2009) and related academic literature.

This module provides the theoretical foundation for MCCFR algorithms
with proper abstractions for different sampling schemes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class InfoSet:
    """
    Information Set representation for MCCFR.
    
    An information set contains all game states that are indistinguishable
    to a player. Each info set maintains:
    - Strategy (action probabilities) 
    - Cumulative regrets
    - Cumulative strategy weights
    """
    key: str
    actions: List[str]
    regret_sum: Dict[str, float] = field(default_factory=dict)
    strategy_sum: Dict[str, float] = field(default_factory=dict)
    strategy: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize regrets, strategy sums and calculate initial strategy."""
        for action in self.actions:
            self.regret_sum.setdefault(action, 0.0)
            self.strategy_sum.setdefault(action, 0.0)
        self.calculate_strategy()
    
    def calculate_strategy(self) -> Dict[str, float]:
        """
        Calculate current strategy using regret matching.
        
        Returns strategy where each action's probability is proportional
        to its positive regret. If no positive regrets exist, uses uniform.
        """
        # Positive regrets only
        positive_regret_sum = sum(max(regret, 0) for regret in self.regret_sum.values())
        
        if positive_regret_sum > 0:
            # Regret matching: prob ∝ positive regret
            for action in self.actions:
                self.strategy[action] = max(self.regret_sum[action], 0) / positive_regret_sum
        else:
            # Uniform strategy if no positive regrets
            uniform_prob = 1.0 / len(self.actions)
            for action in self.actions:
                self.strategy[action] = uniform_prob
        
        return self.strategy.copy()
    
    def get_average_strategy(self) -> Dict[str, float]:
        """
        Calculate the average strategy over all iterations.
        
        This is what converges to Nash equilibrium as T → ∞.
        """
        strategy_sum_total = sum(self.strategy_sum.values())
        
        if strategy_sum_total > 0:
            return {action: self.strategy_sum[action] / strategy_sum_total 
                   for action in self.actions}
        else:
            # Fallback to uniform if no strategy history
            uniform_prob = 1.0 / len(self.actions)
            return {action: uniform_prob for action in self.actions}
    
    def update_regret(self, action: str, regret: float):
        """Add regret for an action."""
        self.regret_sum[action] += regret
    
    def update_strategy_sum(self, reach_prob: float):
        """Update cumulative strategy weighted by reach probability."""
        for action in self.actions:
            self.strategy_sum[action] += reach_prob * self.strategy[action]


class GameHistory(ABC):
    """
    Abstract base class for game state/history.
    
    This represents a sequence of actions in the game tree.
    Different games (Kuhn poker, Texas Hold'em, etc.) inherit from this.
    """
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state."""
        pass
    
    @abstractmethod
    def get_player(self) -> int:
        """Get the player to act at this history."""
        pass
    
    @abstractmethod
    def is_chance_node(self) -> bool:
        """Check if this is a chance node (random event)."""
        pass
    
    @abstractmethod
    def get_actions(self) -> List[str]:
        """Get available actions at this history."""
        pass
    
    @abstractmethod
    def get_info_set_key(self, player: int) -> str:
        """Get information set key for a player."""
        pass
    
    @abstractmethod
    def create_child(self, action: str) -> 'GameHistory':
        """Create child history by taking an action."""
        pass
    
    @abstractmethod
    def get_utility(self, player: int) -> float:
        """Get utility for player at terminal node."""
        pass
    
    @abstractmethod
    def sample_chance(self) -> str:
        """Sample a chance action."""
        pass
    
    @abstractmethod
    def get_chance_prob(self, action: str) -> float:
        """Get probability of chance action."""
        pass


class MCCFRBase(ABC):
    """
    Base class for Monte Carlo CFR algorithms.
    
    Implements the core MCCFR framework with different sampling schemes
    as specified in Lanctot et al. 2009.
    """
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.info_sets: Dict[str, InfoSet] = {}
        self.iteration = 0
        
    def get_info_set(self, history: GameHistory, player: int) -> InfoSet:
        """Get or create information set for player at history."""
        key = history.get_info_set_key(player)
        
        if key not in self.info_sets:
            actions = history.get_actions()
            self.info_sets[key] = InfoSet(key=key, actions=actions)
        
        return self.info_sets[key]
    
    @abstractmethod
    def train(self, root_history: GameHistory, iterations: int) -> Dict[str, InfoSet]:
        """
        Main training loop. Each subclass implements different sampling scheme.
        
        Args:
            root_history: Starting game state
            iterations: Number of training iterations
            
        Returns:
            Dictionary of trained information sets
        """
        pass
    
    def get_strategy_profile(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy for all information sets."""
        return {key: info_set.get_average_strategy() 
                for key, info_set in self.info_sets.items()}
    
    def calculate_exploitability(self, root_history: GameHistory) -> float:
        """
        Calculate exploitability of current average strategy.
        
        Exploitability = sum of best response values for all players
        """
        total_exploitability = 0.0
        
        for player in range(self.num_players):
            best_response_value = self._calculate_best_response_value(root_history, player)
            total_exploitability += best_response_value
            
        return total_exploitability
    
    def _calculate_best_response_value(self, history: GameHistory, player: int) -> float:
        """Calculate best response value for a player against average strategy."""
        if history.is_terminal():
            return history.get_utility(player)
        
        if history.is_chance_node():
            value = 0.0
            for action in history.get_actions():
                child = history.create_child(action)
                prob = history.get_chance_prob(action)
                value += prob * self._calculate_best_response_value(child, player)
            return value
        
        current_player = history.get_player()
        
        if current_player == player:
            # Best response: choose action with highest value
            best_value = float('-inf')
            for action in history.get_actions():
                child = history.create_child(action)
                value = self._calculate_best_response_value(child, player)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent plays according to average strategy
            info_set = self.get_info_set(history, current_player)
            avg_strategy = info_set.get_average_strategy()
            
            value = 0.0
            for action in history.get_actions():
                child = history.create_child(action)
                prob = avg_strategy[action]
                value += prob * self._calculate_best_response_value(child, player)
            return value


class ExternalSamplingMCCFR(MCCFRBase):
    """
    External Sampling MCCFR implementation.
    
    Samples opponent and chance actions, explores all actions for the training player.
    According to Lanctot et al., this has the best theoretical guarantees with
    asymptotic improvement over vanilla CFR.
    """
    
    def train(self, root_history: GameHistory, iterations: int) -> Dict[str, InfoSet]:
        """
        Train using External Sampling MCCFR.
        
        For each iteration and each player:
        1. Sample opponent and chance actions
        2. Explore all actions for training player 
        3. Update regrets using sampled utilities
        """
        logger.info(f"Starting External Sampling MCCFR training for {iterations} iterations")
        
        for iteration in range(iterations):
            self.iteration = iteration
            
            # Train each player
            for player in range(self.num_players):
                self._external_sampling_update(root_history, player, 1.0, 1.0)
            
            if (iteration + 1) % 1000 == 0:
                exploitability = self.calculate_exploitability(root_history)
                logger.info(f"Iteration {iteration + 1}: Exploitability = {exploitability:.6f}")
        
        logger.info("External Sampling MCCFR training completed")
        return self.info_sets
    
    def _external_sampling_update(self, history: GameHistory, player: int, 
                                reach_prob_player: float, reach_prob_others: float) -> float:
        """
        Recursive update for external sampling.
        
        Args:
            history: Current game state
            player: Player being trained
            reach_prob_player: Probability player reaches this history
            reach_prob_others: Probability others reach this history
            
        Returns:
            Expected utility for the player
        """
        if history.is_terminal():
            return history.get_utility(player)
        
        if history.is_chance_node():
            # Sample chance action
            action = history.sample_chance()
            child = history.create_child(action)
            return self._external_sampling_update(
                child, player, reach_prob_player, reach_prob_others
            )
        
        current_player = history.get_player()
        info_set = self.get_info_set(history, current_player)
        
        if current_player == player:
            # Training player: explore all actions
            action_utilities = {}
            utility = 0.0
            
            # Calculate utility for each action
            for action in info_set.actions:
                child = history.create_child(action)
                action_utilities[action] = self._external_sampling_update(
                    child, player, 
                    reach_prob_player * info_set.strategy[action], 
                    reach_prob_others
                )
                utility += info_set.strategy[action] * action_utilities[action]
            
            # Update regrets
            for action in info_set.actions:
                regret = action_utilities[action] - utility
                info_set.update_regret(action, reach_prob_others * regret)
            
            # Update strategy sum
            info_set.update_strategy_sum(reach_prob_player)
            
            # Recalculate strategy for next iteration
            info_set.calculate_strategy()
            
            return utility
        
        else:
            # Opponent: sample action according to current strategy
            actions = info_set.actions
            probs = [info_set.strategy[action] for action in actions]
            sampled_action = np.random.choice(actions, p=probs)
            
            child = history.create_child(sampled_action)
            return self._external_sampling_update(
                child, player, reach_prob_player, 
                reach_prob_others * info_set.strategy[sampled_action]
            )


class OutcomeSamplingMCCFR(MCCFRBase):
    """
    Outcome Sampling MCCFR implementation.
    
    Samples a single trajectory through the game tree per iteration.
    Useful for online learning when opponent's strategy is unknown.
    """
    
    def __init__(self, num_players: int = 2, epsilon: float = 0.6):
        super().__init__(num_players)
        self.epsilon = epsilon  # Exploration parameter for sampling
    
    def train(self, root_history: GameHistory, iterations: int) -> Dict[str, InfoSet]:
        """Train using Outcome Sampling MCCFR."""
        logger.info(f"Starting Outcome Sampling MCCFR training for {iterations} iterations")
        
        for iteration in range(iterations):
            self.iteration = iteration
            
            # Sample single trajectory for each player
            for player in range(self.num_players):
                self._outcome_sampling_update(root_history, player, 1.0, 1.0, 1.0)
            
            if (iteration + 1) % 1000 == 0:
                exploitability = self.calculate_exploitability(root_history)
                logger.info(f"Iteration {iteration + 1}: Exploitability = {exploitability:.6f}")
        
        logger.info("Outcome Sampling MCCFR training completed")
        return self.info_sets
    
    def _outcome_sampling_update(self, history: GameHistory, player: int,
                               reach_prob_player: float, reach_prob_others: float,
                               sampling_prob: float) -> float:
        """Recursive update for outcome sampling."""
        if history.is_terminal():
            return history.get_utility(player) / sampling_prob
        
        if history.is_chance_node():
            action = history.sample_chance()
            child = history.create_child(action)
            chance_prob = history.get_chance_prob(action)
            return self._outcome_sampling_update(
                child, player, reach_prob_player, reach_prob_others, 
                sampling_prob * chance_prob
            )
        
        current_player = history.get_player()
        info_set = self.get_info_set(history, current_player)
        
        # Sample action using epsilon-greedy
        if np.random.random() < self.epsilon:
            sampled_action = np.random.choice(info_set.actions)
            sample_prob = self.epsilon / len(info_set.actions) + \
                         (1 - self.epsilon) * info_set.strategy[sampled_action]
        else:
            actions = info_set.actions
            probs = [info_set.strategy[action] for action in actions]
            sampled_action = np.random.choice(actions, p=probs)
            sample_prob = self.epsilon / len(info_set.actions) + \
                         (1 - self.epsilon) * info_set.strategy[sampled_action]
        
        child = history.create_child(sampled_action)
        
        if current_player == player:
            # Calculate utility
            utility = self._outcome_sampling_update(
                child, player, 
                reach_prob_player * info_set.strategy[sampled_action],
                reach_prob_others, sampling_prob * sample_prob
            )
            
            # Update regrets for all actions
            for action in info_set.actions:
                if action == sampled_action:
                    regret = utility * (1 - info_set.strategy[action])
                else:
                    regret = -utility * info_set.strategy[action]
                
                info_set.update_regret(action, reach_prob_others * regret)
            
            # Update strategy sum
            info_set.update_strategy_sum(reach_prob_player)
            info_set.calculate_strategy()
            
            return utility
        
        else:
            return self._outcome_sampling_update(
                child, player, reach_prob_player,
                reach_prob_others * info_set.strategy[sampled_action],
                sampling_prob * sample_prob
            ) 