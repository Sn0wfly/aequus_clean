"""
🎯 Elite Poker Game Engine - Full NLHE Implementation
Based on OpenSpiel patterns for professional-grade poker simulation

This module implements a complete No-Limit Texas Hold'em game engine with:
- Full betting rounds (pre-flop, flop, turn, river)
- Side pot management
- All-in scenarios
- Street-by-street betting
- Professional-grade game state reconstruction
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# GAME CONSTANTS AND ENUMS
# ============================================================================

class BettingRound(IntEnum):
    """Betting rounds in Texas Hold'em"""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class PlayerAction(IntEnum):
    """Standard poker actions"""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class GamePhase(IntEnum):
    """Game phases for state tracking"""
    DEALING = 0
    BETTING = 1
    SHOWDOWN = 2
    FINISHED = 3

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PlayerState:
    """Complete player state including stack, bets, and status"""
    player_id: int
    stack: float
    current_bet: float
    total_bet: float
    is_folded: bool
    is_all_in: bool
    hole_cards: jnp.ndarray  # [2] cards
    hand_strength: float = 0.0

@dataclass
class Pot:
    """Represents a pot or side pot"""
    amount: float
    eligible_players: List[int]  # Player IDs eligible for this pot
    max_bet: float  # Maximum bet that created this pot

@dataclass
class GameState:
    """Complete game state for NLHE"""
    players: List[PlayerState]
    community_cards: jnp.ndarray  # [5] cards, -1 for undealt
    current_round: BettingRound
    current_player: int
    min_raise: float
    max_raise: float
    pot: float
    side_pots: List[Pot]
    action_history: List[Tuple[int, PlayerAction, float]]
    deck: jnp.ndarray  # [52] shuffled deck
    deck_index: int
    small_blind: float
    big_blind: float
    phase: GamePhase

# ============================================================================
# ELITE GAME ENGINE
# ============================================================================

class EliteGameEngine:
    """
    Professional-grade NLHE game engine based on OpenSpiel patterns
    
    Features:
    - Complete betting structure with unlimited raises
    - Side pot calculation with multiple all-ins
    - Street-by-street game progression
    - Professional hand evaluation
    - Full game state reconstruction
    """
    
    def __init__(self, num_players: int = 6, small_blind: float = 1.0, big_blind: float = 2.0):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_players = 6
        
    @jax.jit
    def _create_deck(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Create and shuffle a standard 52-card deck"""
        return jax.random.permutation(rng_key, jnp.arange(52))
    
    @jax.jit
    def _deal_cards(self, deck: jnp.ndarray, start_idx: int, num_cards: int) -> Tuple[jnp.ndarray, int]:
        """Deal cards from the deck"""
        cards = lax.dynamic_slice(deck, (start_idx,), (num_cards,))
        return cards, start_idx + num_cards
    
    def _initialize_game(self, rng_key: jnp.ndarray, starting_stacks: jnp.ndarray) -> GameState:
        """Initialize a new NLHE game"""
        deck = self._create_deck(rng_key)
        
        # Deal hole cards to all players
        hole_cards = []
        deck_idx = 0
        for player_id in range(self.num_players):
            cards, deck_idx = self._deal_cards(deck, deck_idx, 2)
            hole_cards.append(cards)
        
        # Initialize players
        players = []
        for i in range(self.num_players):
            player = PlayerState(
                player_id=i,
                stack=float(starting_stacks[i]),
                current_bet=0.0,
                total_bet=0.0,
                is_folded=False,
                is_all_in=False,
                hole_cards=hole_cards[i]
            )
            players.append(player)
        
        # Post blinds
        sb_player = 0
        bb_player = 1
        
        players[sb_player].stack -= self.small_blind
        players[sb_player].current_bet = self.small_blind
        players[sb_player].total_bet = self.small_blind
        
        players[bb_player].stack -= self.big_blind
        players[bb_player].current_bet = self.big_blind
        players[bb_player].total_bet = self.big_blind
        
        return GameState(
            players=players,
            community_cards=jnp.full(5, -1),
            current_round=BettingRound.PREFLOP,
            current_player=2,  # UTG
            min_raise=self.big_blind,
            max_raise=float('inf'),
            pot=self.small_blind + self.big_blind,
            side_pots=[],
            action_history=[],
            deck=deck,
            deck_index=deck_idx,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            phase=GamePhase.BETTING
        )
    
    def _calculate_side_pots(self, players: List[PlayerState]) -> List[Pot]:
        """Calculate side pots based on player bets and all-ins"""
        active_players = [p for p in players if not p.is_folded]
        if not active_players:
            return []
        
        # Group players by their total contribution
        contributions = {}
        for player in active_players:
            contrib = player.total_bet
            if contrib not in contributions:
                contributions[contrib] = []
            contributions[contrib].append(player.player_id)
        
        # Sort contributions
        sorted_contribs = sorted(contributions.keys())
        
        side_pots = []
        prev_contrib = 0.0
        
        for contrib in sorted_contribs:
            if contrib > prev_contrib:
                # Create new side pot
                pot_amount = (contrib - prev_contrib) * len([
                    p for p in active_players 
                    if p.total_bet >= contrib
                ])
                
                eligible_players = [
                    p.player_id for p in active_players 
                    if p.total_bet >= contrib
                ]
                
                side_pots.append(Pot(
                    amount=pot_amount,
                    eligible_players=eligible_players,
                    max_bet=contrib
                ))
                
                prev_contrib = contrib
        
        return side_pots
    
    def _is_betting_round_complete(self, game_state: GameState) -> bool:
        """Check if current betting round is complete"""
        active_players = [p for p in game_state.players if not p.is_folded and not p.is_all_in]
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have acted and bets are equal
        max_bet = max(p.current_bet for p in game_state.players if not p.is_folded)
        
        for player in active_players:
            if player.current_bet != max_bet:
                return False
        
        # Check if all players have had a chance to act
        last_raiser = None
        for player_id, action, amount in reversed(game_state.action_history):
            if action in [PlayerAction.BET, PlayerAction.RAISE, PlayerAction.ALL_IN]:
                last_raiser = player_id
                break
        
        if last_raiser is None:
            return False
        
        # All players after last raiser have acted
        current_player = game_state.current_player
        return current_player == (last_raiser + 1) % self.num_players
    
    def _deal_community_cards(self, game_state: GameState) -> GameState:
        """Deal community cards for the current round"""
        if game_state.current_round == BettingRound.FLOP:
            # Deal 3 flop cards
            cards, new_deck_idx = self._deal_cards(
                game_state.deck, 
                game_state.deck_index, 
                3
            )
            new_community = game_state.community_cards.at[0:3].set(cards)
            return game_state._replace(
                community_cards=new_community,
                deck_index=new_deck_idx
            )
        elif game_state.current_round == BettingRound.TURN:
            # Deal 1 turn card
            cards, new_deck_idx = self._deal_cards(
                game_state.deck, 
                game_state.deck_index, 
                1
            )
            new_community = game_state.community_cards.at[3:4].set(cards)
            return game_state._replace(
                community_cards=new_community,
                deck_index=new_deck_idx
            )
        elif game_state.current_round == BettingRound.RIVER:
            # Deal 1 river card
            cards, new_deck_idx = self._deal_cards(
                game_state.deck, 
                game_state.deck_index, 
                1
            )
            new_community = game_state.community_cards.at[4:5].set(cards)
            return game_state._replace(
                community_cards=new_community,
                deck_index=new_deck_idx
            )
        
        return game_state
    
    def _evaluate_showdown(self, game_state: GameState) -> jnp.ndarray:
        """Evaluate hands at showdown and determine winners"""
        from poker_bot.core.simulation import _evaluate_hand_strength
        
        payoffs = jnp.zeros(self.num_players)
        
        # Get active players
        active_players = [
            p for p in game_state.players 
            if not p.is_folded
        ]
        
        if len(active_players) == 1:
            # Single winner
            winner = active_players[0]
            payoffs = payoffs.at[winner.player_id].set(game_state.pot)
            return payoffs
        
        # Evaluate all hands
        hand_strengths = []
        for player in active_players:
            full_hand = jnp.concatenate([
                player.hole_cards,
                game_state.community_cards[game_state.community_cards >= 0]
            ])
            strength = _evaluate_hand_strength(full_hand)
            hand_strengths.append((player.player_id, strength))
        
        # Calculate side pot distributions
        total_paid = 0.0
        for pot in game_state.side_pots:
            # Find best hand among eligible players
            eligible_hands = [
                (pid, strength) for pid, strength in hand_strengths
                if pid in pot.eligible_players
            ]
            
            if eligible_hands:
                best_strength = max(strength for _, strength in eligible_hands)
                winners = [pid for pid, strength in eligible_hands if strength == best_strength]
                
                # Split pot among winners
                winner_share = pot.amount / len(winners)
                for winner in winners:
                    payoffs = payoffs.at[winner].add(winner_share)
        
        # Calculate net payoffs
        for player in game_state.players:
            net = payoffs[player.player_id] - player.total_bet
            payoffs = payoffs.at[player.player_id].set(net)
        
        return payoffs
    
    def apply_action(self, game_state: GameState, action: PlayerAction, amount: float = 0.0) -> GameState:
        """Apply a player action to the game state"""
        if game_state.phase != GamePhase.BETTING:
            return game_state
        
        current_player = game_state.players[game_state.current_player]
        
        if action == PlayerAction.FOLD:
            current_player.is_folded = True
            current_player.current_bet = 0.0
            
        elif action == PlayerAction.CHECK:
            if current_player.current_bet < max(p.current_bet for p in game_state.players):
                raise ValueError("Cannot check when facing a bet")
                
        elif action == PlayerAction.CALL:
            max_bet = max(p.current_bet for p in game_state.players)
            call_amount = max_bet - current_player.current_bet
            
            if call_amount >= current_player.stack:
                # All-in
                call_amount = current_player.stack
                current_player.is_all_in = True
            
            current_player.stack -= call_amount
            current_player.current_bet += call_amount
            current_player.total_bet += call_amount
            
        elif action == PlayerAction.BET:
            if amount < game_state.min_raise:
                raise ValueError(f"Bet must be at least {game_state.min_raise}")
            
            if amount > current_player.stack:
                amount = current_player.stack
                current_player.is_all_in = True
            
            current_player.stack -= amount
            current_player.current_bet = amount
            current_player.total_bet += amount
            
        elif action == PlayerAction.RAISE:
            max_bet = max(p.current_bet for p in game_state.players)
            min_raise = max_bet + game_state.min_raise
            
            if amount < min_raise:
                raise ValueError(f"Raise must be at least {min_raise}")
            
            if amount > current_player.stack:
                amount = current_player.stack
                current_player.is_all_in = True
            
            current_player.stack -= (amount - current_player.current_bet)
            current_player.current_bet = amount
            current_player.total_bet += (amount - current_player.current_bet)
            
        elif action == PlayerAction.ALL_IN:
            all_in_amount = current_player.stack
            current_player.stack = 0.0
            current_player.current_bet += all_in_amount
            current_player.total_bet += all_in_amount
            current_player.is_all_in = True
        
        # Update action history
        new_history = game_state.action_history + [(game_state.current_player, action, amount)]
        
        # Calculate next player
        next_player = (game_state.current_player + 1) % self.num_players
        
        # Skip folded and all-in players
        while (game_state.players[next_player].is_folded or 
               game_state.players[next_player].is_all_in):
            next_player = (next_player + 1) % self.num_players
        
        # Check if betting round is complete
        if self._is_betting_round_complete(game_state._replace(
            current_player=next_player,
            action_history=new_history
        )):
            # Move to next round or showdown
            return self._advance_round(game_state._replace(
                action_history=new_history
            ))
        
        return game_state._replace(
            current_player=next_player,
            action_history=new_history
        )
    
    def _advance_round(self, game_state: GameState) -> GameState:
        """Advance to the next betting round or showdown"""
        # Reset current bets
        new_players = []
        for player in game_state.players:
            new_player = player._replace(current_bet=0.0)
            new_players.append(new_player)
        
        # Calculate side pots
        side_pots = self._calculate_side_pots(new_players)
        
        # Check if we should go to showdown
        active_players = [p for p in new_players if not p.is_folded]
        if len(active_players) <= 1:
            # Immediate winner
            payoffs = jnp.zeros(self.num_players)
            if active_players:
                winner = active_players[0]
                payoffs = payoffs.at[winner.player_id].set(game_state.pot)
                for player in new_players:
                    net = payoffs[player.player_id] - player.total_bet
                    payoffs = payoffs.at[player.player_id].set(net)
            
            return game_state._replace(
                players=new_players,
                side_pots=side_pots,
                phase=GamePhase.FINISHED
            )
        
        # Move to next round
        next_round = BettingRound(game_state.current_round + 1)
        
        if next_round == BettingRound.SHOWDOWN:
            # Evaluate hands
            payoffs = self._evaluate_showdown(game_state._replace(
                players=new_players,
                side_pots=side_pots
            ))
            
            return game_state._replace(
                players=new_players,
                side_pots=side_pots,
                phase=GamePhase.FINISHED
            )
        
        # Deal community cards for next round
        updated_state = self._deal_community_cards(game_state._replace(
            players=new_players,
            current_round=next_round,
            current_player=0,  # First to act after blinds
            side_pots=side_pots,
            action_history=[]
        ))
        
        return updated_state
    
    def simulate_game(self, rng_key: jnp.ndarray, starting_stacks: jnp.ndarray) -> Dict:
        """
        Simulate a complete NLHE game with random actions
        This is for training data generation
        """
        game_state = self._initialize_game(rng_key, starting_stacks)
        
        # Simple random action selection for training
        while game_state.phase == GamePhase.BETTING:
            current_player = game_state.players[game_state.current_player]
            
            if current_player.is_folded or current_player.is_all_in:
                continue
            
            # Random action selection (simplified for training)
            max_bet = max(p.current_bet for p in game_state.players)
            call_amount = max_bet - current_player.current_bet
            
            # Simple action probabilities
            if call_amount == 0:
                # Can check or bet
                action = PlayerAction.CHECK if jax.random.uniform(rng_key) < 0.7 else PlayerAction.BET
                amount = game_state.min_raise if action == PlayerAction.BET else 0.0
            else:
                # Can fold, call, or raise
                rand = jax.random.uniform(rng_key)
                if rand < 0.3:
                    action = PlayerAction.FOLD
                    amount = 0.0
                elif rand < 0.8:
                    action = PlayerAction.CALL
                    amount = call_amount
                else:
                    action = PlayerAction.RAISE
                    amount = call_amount + game_state.min_raise
            
            game_state = self.apply_action(game_state, action, amount)
        
        # Convert to training format
        hole_cards = jnp.stack([p.hole_cards for p in game_state.players])
        community_cards = game_state.community_cards
        
        # Create payoffs array
        payoffs = jnp.zeros(self.num_players)
        if game_state.phase == GamePhase.FINISHED:
            # Calculate final payoffs based on game result
            active_players = [p for p in game_state.players if not p.is_folded]
            if len(active_players) == 1:
                # Single winner
                winner = active_players[0]
                payoffs = payoffs.at[winner.player_id].set(game_state.pot)
            else:
                # Multiple players - use showdown evaluation
                payoffs = self._evaluate_showdown(game_state)
        
        return {
            'payoffs': payoffs,
            'final_community': community_cards,
            'hole_cards': hole_cards,
            'final_pot': game_state.pot,
            'action_history': game_state.action_history,
            'side_pots': [(p.amount, p.eligible_players) for p in game_state.side_pots]
        }

# ============================================================================
# GAME STATE RECONSTRUCTION FOR TRAINING
# ============================================================================

class GameStateReconstructor:
    """Reconstruct game states from training data for CFR"""
    
    @staticmethod
    def reconstruct_from_history(
        hole_cards: jnp.ndarray,
        community_cards: jnp.ndarray,
        action_history: List[Tuple[int, PlayerAction, float]],
        starting_stacks: jnp.ndarray
    ) -> List[Dict]:
        """Reconstruct game states at each decision point"""
        engine = EliteGameEngine(num_players=len(hole_cards))
        
        # Initialize game
        rng_key = jax.random.PRNGKey(0)  # Dummy key for reconstruction
        game_state = engine._initialize_game(rng_key, starting_stacks)
        
        states = []
        
        # Replay actions to reconstruct states
        for player_id, action, amount in action_history:
            # Create state representation at this decision point
            state_info = {
                'hole_cards': hole_cards[player_id],
                'community_cards': game_state.community_cards,
                'current_round': game_state.current_round,
                'pot': game_state.pot,
                'stack': game_state.players[player_id].stack,
                'current_bet': game_state.players[player_id].current_bet,
                'position': player_id,
                'num_active': len([p for p in game_state.players if not p.is_folded]),
                'legal_actions': GameStateReconstructor._get_legal_actions(game_state, player_id)
            }
            states.append(state_info)
            
            # Apply action
            game_state = engine.apply_action(game_state, action, amount)
        
        return states
    
    @staticmethod
    def _get_legal_actions(game_state: GameState, player_id: int) -> List[PlayerAction]:
        """Get legal actions for a player in current state"""
        player = game_state.players[player_id]
        
        if player.is_folded or player.is_all_in:
            return []
        
        max_bet = max(p.current_bet for p in game_state.players)
        call_amount = max_bet - player.current_bet
        
        actions = []
        
        if call_amount == 0:
            actions.extend([PlayerAction.CHECK, PlayerAction.BET])
        else:
            actions.extend([PlayerAction.FOLD, PlayerAction.CALL])
            if player.stack > call_amount:
                actions.append(PlayerAction.RAISE)
        
        if player.stack > 0:
            actions.append(PlayerAction.ALL_IN)
        
        return actions