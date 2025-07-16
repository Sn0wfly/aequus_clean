"""
CUDA-accelerated Monte Carlo Counterfactual Regret Minimization for Poker

This package provides a production-ready MCCFR implementation for poker
with CUDA-accelerated hand evaluation.
"""

from .mccfr_core import (
    InfoSet,
    GameHistory, 
    MCCFRBase,
    ExternalSamplingMCCFR,
    OutcomeSamplingMCCFR
)

from .poker_game import (
    PokerCard,
    PokerDeck,
    TexasHoldemHistory
)

from .mccfr_trainer import (
    MCCFRConfig,
    MCCFRTrainer,
    create_default_config
)

__version__ = "1.0.0"
__author__ = "MCCFR Production Team"

__all__ = [
    # Core MCCFR
    "InfoSet",
    "GameHistory",
    "MCCFRBase", 
    "ExternalSamplingMCCFR",
    "OutcomeSamplingMCCFR",
    
    # Poker game
    "PokerCard",
    "PokerDeck", 
    "TexasHoldemHistory",
    
    # Training
    "MCCFRConfig",
    "MCCFRTrainer",
    "create_default_config"
] 