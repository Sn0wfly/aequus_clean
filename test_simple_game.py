"""
Simple game logic test to debug the recursion issue.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from poker_cuda import TexasHoldemHistory

def test_simple_game_flow():
    """Test that game can reach terminal states correctly."""
    print("Testing simple game flow...")
    
    game = TexasHoldemHistory(num_players=2)
    print(f"Initial state:")
    print(f"  Player: {game.get_player()}")
    print(f"  Actions: {game.get_actions()}")
    print(f"  Terminal: {game.is_terminal()}")
    print(f"  Chance: {game.is_chance_node()}")
    print(f"  Pot: {game.pot}")
    
    # Take a few actions
    step = 0
    max_steps = 10
    
    while not game.is_terminal() and step < max_steps:
        step += 1
        print(f"\nStep {step}:")
        
        if game.is_chance_node():
            action = game.sample_chance()
            print(f"  Chance action: {action}")
        else:
            actions = game.get_actions()
            if not actions:
                print("  No actions available but not terminal!")
                break
            action = actions[0]  # Take first action
            print(f"  Player {game.get_player()} action: {action}")
        
        try:
            game = game.create_child(action)
            print(f"  New pot: {game.pot}")
            print(f"  Terminal: {game.is_terminal()}")
            print(f"  Chance: {game.is_chance_node()}")
        except Exception as e:
            print(f"  Error creating child: {e}")
            break
    
    if step >= max_steps:
        print(f"\nReached max steps ({max_steps}) - game may not terminate properly")
        return False
    
    print(f"\nGame completed in {step} steps")
    print(f"Final terminal state: {game.is_terminal()}")
    
    if game.is_terminal():
        print("Utilities:")
        for player in range(2):
            utility = game.get_utility(player)
            print(f"  Player {player}: {utility}")
    
    return game.is_terminal()

if __name__ == "__main__":
    test_simple_game_flow() 