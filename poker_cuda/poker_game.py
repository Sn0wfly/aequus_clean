"""
Texas Hold'em Game Implementation for MCCFR

This module implements the Texas Hold'em poker game logic that integrates
with the MCCFR core framework and uses CUDA hand evaluation.
"""

from typing import List, Dict, Optional, Tuple, Set
import random
import itertools
import ctypes
from dataclasses import dataclass, field
import numpy as np

from .mccfr_core import GameHistory
from .hand_evaluator_real import load_cuda_library

# Load CUDA library once at module level
cuda_lib = None
try:
    cuda_lib = load_cuda_library()
except Exception as e:
    print(f"Warning: Could not load CUDA library: {e}")


@dataclass
class PokerCard:
    """Represents a playing card."""
    rank: int  # 2-14 (2-9, T, J, Q, K, A)
    suit: int  # 0-3 (spades, hearts, diamonds, clubs)
    
    def __str__(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['s', 'h', 'd', 'c']
        return f"{ranks[self.rank-2]}{suits[self.suit]}"
    
    def to_int(self) -> int:
        """Convert card to integer representation for CUDA."""
        return self.rank * 4 + self.suit
    
    def __lt__(self, other):
        """Less than comparison for sorting."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit
    
    def __eq__(self, other):
        """Equality comparison."""
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        """Hash for use in sets/dicts."""
        return hash((self.rank, self.suit))


class PokerDeck:
    """Standard 52-card deck."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset deck to all 52 cards."""
        self.cards = [PokerCard(rank, suit) 
                     for rank in range(2, 15) 
                     for suit in range(4)]
        random.shuffle(self.cards)
        self.dealt_count = 0
    
    def deal(self, count: int) -> List[PokerCard]:
        """Deal specified number of cards."""
        if self.dealt_count + count > 52:
            raise ValueError("Not enough cards in deck")
        
        cards = self.cards[self.dealt_count:self.dealt_count + count]
        self.dealt_count += count
        return cards
    
    def remaining_cards(self) -> List[PokerCard]:
        """Get list of remaining cards in deck."""
        return self.cards[self.dealt_count:]


class TexasHoldemHistory(GameHistory):
    """
    Texas Hold'em game state implementation.
    
    Represents a specific point in a poker hand including:
    - Player hole cards
    - Community cards (flop, turn, river)
    - Betting action history
    - Current player to act
    - Pot size and player stacks
    """
    
    def __init__(self, num_players: int = 2, small_blind: int = 1, big_blind: int = 2,
                 starting_stacks: int = 200):
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stacks = starting_stacks
        
        # Game state
        self.deck = PokerDeck()
        self.hole_cards: List[List[PokerCard]] = [[] for _ in range(num_players)]
        self.community_cards: List[PokerCard] = []
        self.action_history: List[str] = []
        self.betting_round = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.current_player = 0
        
        # Betting state
        self.pot = 0
        self.stacks = [starting_stacks] * num_players
        self.current_bets = [0] * num_players
        self.folded = [False] * num_players
        self.all_in = [False] * num_players
        
        # Deal initial cards
        self._deal_hole_cards()
        self._post_blinds()
        self._find_next_player()
    
    def _deal_hole_cards(self):
        """Deal 2 hole cards to each player."""
        for player in range(self.num_players):
            self.hole_cards[player] = self.deck.deal(2)
    
    def _post_blinds(self):
        """Post small and big blinds."""
        # Small blind (player 0)
        sb_amount = min(self.small_blind, self.stacks[0])
        self.current_bets[0] = sb_amount
        self.stacks[0] -= sb_amount
        self.pot += sb_amount
        
        # Big blind (player 1)
        bb_amount = min(self.big_blind, self.stacks[1])
        self.current_bets[1] = bb_amount
        self.stacks[1] -= bb_amount
        self.pot += bb_amount
        
        if self.stacks[0] == 0:
            self.all_in[0] = True
        if self.stacks[1] == 0:
            self.all_in[1] = True
    
    def _find_next_player(self):
        """Find next player to act."""
        # In heads-up poker:
        # Preflop: Small blind (player 0) acts first after blinds are posted
        # Postflop: Small blind (player 0) acts first
        
        start_player = 0  # Small blind always acts first in heads-up
        
        for i in range(self.num_players):
            player = (start_player + i) % self.num_players
            if not self.folded[player] and not self.all_in[player]:
                self.current_player = player
                return
        
        # No active players found - hand should be terminal
        self.current_player = -1
    
    def is_terminal(self) -> bool:
        """Check if hand is over."""
        active_players = sum(1 for i in range(self.num_players) 
                           if not self.folded[i])
        
        # Only one player left
        if active_players <= 1:
            return True
        
        # All betting rounds complete and all players checked/called
        if self.betting_round >= 4:
            return True
        
        # All players all-in
        if all(self.folded[i] or self.all_in[i] for i in range(self.num_players)):
            return True
        
        return False
    
    def get_player(self) -> int:
        """Get current player to act."""
        return self.current_player
    
    def is_chance_node(self) -> bool:
        """Check if we need to deal community cards."""
        if not self._betting_round_complete():
            return False
        
        # Need to deal flop, turn, or river
        if self.betting_round == 0 and len(self.community_cards) == 0:
            return True
        if self.betting_round == 1 and len(self.community_cards) == 3:
            return True  
        if self.betting_round == 2 and len(self.community_cards) == 4:
            return True
        
        return False
    
    def _betting_round_complete(self) -> bool:
        """Check if current betting round is complete."""
        active_players = [i for i in range(self.num_players) 
                         if not self.folded[i] and not self.all_in[i]]
        
        if len(active_players) <= 1:
            return True
        
        max_bet = max(self.current_bets)
        
        # All active players have matched max bet or are all-in
        for player in active_players:
            if self.current_bets[player] < max_bet:
                return False
        
        return True
    
    def get_actions(self) -> List[str]:
        """Get available actions for current player."""
        if self.is_terminal() or self.is_chance_node():
            return []
        
        actions = []
        max_bet = max(self.current_bets)
        current_bet = self.current_bets[self.current_player]
        stack = self.stacks[self.current_player]
        
        # Can always fold (except when no bet to face)
        if max_bet > current_bet:
            actions.append('fold')
        
        # Can check if no bet to face, call if there is
        if max_bet == current_bet:
            actions.append('check')
        else:
            call_amount = max_bet - current_bet
            if call_amount <= stack:
                actions.append('call')
            else:
                actions.append('allin')
        
        # Can bet/raise if have chips
        if stack > 0:
            if max_bet == current_bet:
                # No bet to face - can bet
                min_bet = self.big_blind
                if stack >= min_bet:
                    actions.append('bet')
            else:
                # Bet to face - can raise
                min_raise = (max_bet - current_bet) * 2  # Min raise = size of current bet
                if stack >= max_bet - current_bet + min_raise:
                    actions.append('raise')
        
        return actions
    
    def get_info_set_key(self, player: int) -> str:
        """
        Generate information set key for a player.
        
        Info set includes:
        - Player's hole cards
        - Community cards
        - Betting action history
        """
        hole_str = ''.join(str(card) for card in sorted(self.hole_cards[player]))
        community_str = ''.join(str(card) for card in self.community_cards)
        action_str = ''.join(self.action_history)
        
        return f"{hole_str}|{community_str}|{action_str}"
    
    def create_child(self, action: str) -> 'TexasHoldemHistory':
        """Create new history by taking an action."""
        # Create new instance without calling __init__
        new_history = object.__new__(TexasHoldemHistory)
        
        # Copy basic attributes
        new_history.num_players = self.num_players
        new_history.small_blind = self.small_blind
        new_history.big_blind = self.big_blind
        new_history.starting_stacks = self.starting_stacks
        
        # Copy deck state
        new_history.deck = object.__new__(PokerDeck)
        new_history.deck.cards = self.deck.cards.copy()
        new_history.deck.dealt_count = self.deck.dealt_count
        
        # Copy game state
        new_history.hole_cards = [cards.copy() for cards in self.hole_cards]
        new_history.community_cards = self.community_cards.copy()
        new_history.action_history = self.action_history.copy()
        new_history.betting_round = self.betting_round
        new_history.current_player = self.current_player
        
        # Copy betting state
        new_history.pot = self.pot
        new_history.stacks = self.stacks.copy()
        new_history.current_bets = self.current_bets.copy()
        new_history.folded = self.folded.copy()
        new_history.all_in = self.all_in.copy()
        
        # Apply the action
        if action in ['flop', 'turn', 'river']:
            new_history._deal_community_cards(action)
        else:
            new_history._apply_action(action)
        
        return new_history
    
    def _deal_community_cards(self, action: str):
        """Deal community cards."""
        if action == 'flop':
            self.community_cards.extend(self.deck.deal(3))
        elif action == 'turn':
            self.community_cards.extend(self.deck.deal(1))
        elif action == 'river':
            self.community_cards.extend(self.deck.deal(1))
        
        self.betting_round += 1
        self.current_bets = [0] * self.num_players
        self._find_next_player()
    
    def _apply_action(self, action: str):
        """Apply betting action."""
        player = self.current_player
        max_bet = max(self.current_bets)
        current_bet = self.current_bets[player]
        
        if action == 'fold':
            self.folded[player] = True
        elif action == 'check':
            pass  # No money changes hands
        elif action == 'call':
            call_amount = min(max_bet - current_bet, self.stacks[player])
            self.current_bets[player] += call_amount
            self.stacks[player] -= call_amount
            self.pot += call_amount
            if self.stacks[player] == 0:
                self.all_in[player] = True
        elif action == 'bet':
            bet_amount = min(self.big_blind, self.stacks[player])
            self.current_bets[player] += bet_amount
            self.stacks[player] -= bet_amount
            self.pot += bet_amount
            if self.stacks[player] == 0:
                self.all_in[player] = True
        elif action == 'raise':
            current_bet_size = max_bet - current_bet
            raise_amount = min(current_bet_size * 2, self.stacks[player])
            self.current_bets[player] += current_bet_size + raise_amount
            self.stacks[player] -= current_bet_size + raise_amount
            self.pot += current_bet_size + raise_amount
            if self.stacks[player] == 0:
                self.all_in[player] = True
        elif action == 'allin':
            allin_amount = self.stacks[player]
            self.current_bets[player] += allin_amount
            self.stacks[player] = 0
            self.pot += allin_amount
            self.all_in[player] = True
        
        self.action_history.append(action)
        
        # Move to next player or next betting round
        if self._betting_round_complete():
            if not self.is_terminal():
                self.betting_round += 1
                self.current_bets = [0] * self.num_players
        
        self._find_next_player()
    
    def sample_chance(self) -> str:
        """Sample chance action (community cards)."""
        if self.betting_round == 0:
            return 'flop'
        elif self.betting_round == 1:
            return 'turn'
        elif self.betting_round == 2:
            return 'river'
        else:
            raise ValueError("No chance action available")
    
    def get_chance_prob(self, action: str) -> float:
        """Get probability of chance action (all equal for card dealing)."""
        return 1.0  # Simplified - in practice would be 1/C(remaining_cards, cards_dealt)
    
    def get_utility(self, player: int) -> float:
        """Get utility for player at terminal node."""
        if not self.is_terminal():
            return 0.0
        
        # If player folded, they get 0
        if self.folded[player]:
            return 0.0
        
        active_players = [i for i in range(self.num_players) if not self.folded[i]]
        
        # If only one active player, they win the pot
        if len(active_players) == 1 and player in active_players:
            return self.pot
        elif len(active_players) == 1:
            return 0.0
        
        # Showdown - evaluate hands using CUDA
        if len(self.community_cards) == 5:
            return self._evaluate_showdown(player)
        else:
            # Hand ended before showdown, split pot among active players
            return self.pot / len(active_players) if player in active_players else 0.0
    
    def _evaluate_showdown(self, player: int) -> float:
        """Evaluate showdown using CUDA hand evaluator."""
        if cuda_lib is None:
            # Fallback to random if CUDA not available
            active_players = [i for i in range(self.num_players) if not self.folded[i]]
            if player in active_players:
                return self.pot / len(active_players)
            return 0.0
        
        try:
            # Convert cards to integers for CUDA
            player_cards = [card.to_int() for card in self.hole_cards[player]]
            community_ints = [card.to_int() for card in self.community_cards]
            
            # Evaluate player's hand
            cards_array = (ctypes.c_int * 7)(*player_cards + community_ints)
            player_score = cuda_lib.cuda_evaluate_hand(cards_array)
            
            # Compare with other active players
            active_players = [i for i in range(self.num_players) if not self.folded[i]]
            scores = {}
            
            for opp in active_players:
                opp_cards = [card.to_int() for card in self.hole_cards[opp]]
                opp_cards_array = (ctypes.c_int * 7)(*opp_cards + community_ints)
                scores[opp] = cuda_lib.cuda_evaluate_hand(opp_cards_array)
            
            # Find winners (higher score = better hand)
            max_score = max(scores.values())
            winners = [p for p, score in scores.items() if score == max_score]
            
            if player in winners:
                return self.pot / len(winners)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error in CUDA evaluation: {e}")
            # Fallback
            active_players = [i for i in range(self.num_players) if not self.folded[i]]
            if player in active_players:
                return self.pot / len(active_players)
            return 0.0 