"""
Test game completion to see if games reach terminal states.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from poker_cuda import TexasHoldemHistory

def play_complete_game():
    """Play a complete game to see if it terminates."""
    print("Playing complete game...")
    
    game = TexasHoldemHistory(num_players=2)
    step = 0
    max_steps = 50  # Reasonable limit
    
    print(f"Initial state: pot={game.pot}, player={game.current_player}")
    print(f"  Terminal: {game.is_terminal()}, Chance: {game.is_chance_node()}")
    
    while not game.is_terminal() and step < max_steps:
        step += 1
        print(f"\nStep {step}:")
        
        if game.is_chance_node():
            action = game.sample_chance()
            print(f"  Chance: {action}")
        else:
            actions = game.get_actions()
            print(f"  Player {game.current_player} actions: {actions}")
            
            if not actions:
                print("  ERROR: No actions but not terminal!")
                break
                
            # Play a simple strategy: fold sometimes, call otherwise
            if 'fold' in actions and step > 2:  # Fold sometimes to end game
                action = 'fold'
            elif 'call' in actions:
                action = 'call'
            elif 'check' in actions:
                action = 'check'
            else:
                action = actions[0]
            
            print(f"  Chosen action: {action}")
        
        game = game.create_child(action)
        print(f"  New pot: {game.pot}")
        print(f"  Terminal: {game.is_terminal()}, Chance: {game.is_chance_node()}")
        
        if game.is_terminal():
            print("  GAME ENDED!")
            print("  Final utilities:")
            for player in range(2):
                utility = game.get_utility(player)
                print(f"    Player {player}: {utility}")
            break
    
    if step >= max_steps:
        print(f"\nWARNING: Game didn't terminate in {max_steps} steps")
        print("This could cause infinite recursion in MCCFR")
        return False
    else:
        print(f"\nGame completed successfully in {step} steps")
        return True

def test_multiple_games():
    """Test multiple games to see if they all terminate."""
    print("Testing multiple games...")
    
    for i in range(5):
        print(f"\n{'='*20} Game {i+1} {'='*20}")
        success = play_complete_game()
        if not success:
            print(f"Game {i+1} failed to terminate!")
            return False
    
    print("\nAll games terminated successfully!")
    return True

if __name__ == "__main__":
    test_multiple_games() 