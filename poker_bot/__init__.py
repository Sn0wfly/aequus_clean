"""
🎯 PokerTrainer - GPU-Native Poker AI

A high-performance poker bot using JAX + MCCFR for GPU acceleration.

Phase 1: Modern CFR (CFVFP) and GPU optimization
Phase 2: Performance optimization with multi-GPU, advanced algorithms, and smart caching
"""

__version__ = "1.0.0"
__author__ = "PokerTrainer Team"

from .core.trainer import PokerTrainer, TrainerConfig
from .bot import PokerBot
from .evaluator import HandEvaluator

__all__ = ["PokerTrainer", "TrainerConfig", "PokerBot", "HandEvaluator"] 