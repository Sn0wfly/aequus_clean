"""
Minimal test to isolate the exact issue.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_minimal():
    print("Starting minimal test...")
    
    try:
        print("1. Importing TexasHoldemHistory...")
        from poker_cuda import TexasHoldemHistory
        print("   Import successful")
        
        print("2. Creating game instance...")
        game = TexasHoldemHistory(num_players=2)
        print("   Game created successfully")
        
        print("3. Checking basic properties...")
        print(f"   Current player: {game.current_player}")
        print(f"   Pot: {game.pot}")
        print(f"   Is terminal: {game.is_terminal()}")
        print(f"   Is chance: {game.is_chance_node()}")
        
        print("4. Getting actions...")
        actions = game.get_actions()
        print(f"   Actions: {actions}")
        
        if actions:
            print("5. Creating child with first action...")
            action = actions[0]
            print(f"   Taking action: {action}")
            child = game.create_child(action)
            print(f"   Child created successfully")
            print(f"   Child pot: {child.pot}")
        
        print("Test completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_minimal() 