#!/usr/bin/env python3
"""
Verification script for Aequus cleanup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_imports():
    """Verify all core modules can be imported"""
    try:
        from poker_bot.core.trainer import PokerTrainer, TrainerConfig
        print("✅ trainer.py imports successfully")
        
        from poker_bot.core.simulation import batch_simulate_real_holdem
        print("✅ simulation.py imports successfully")
        
        from poker_bot.core.enhanced_eval import EnhancedHandEvaluator
        print("✅ enhanced_eval.py imports successfully")
        
        from poker_bot.core.icm_modeling import ICMModel
        print("✅ icm_modeling.py imports successfully")
        
        from poker_bot.core.mccfr_gpu import mccfr_rollout_gpu
        print("✅ mccfr_gpu.py imports successfully")
        
        from poker_bot.core.cfr_gpu import cfr_step_gpu
        print("✅ cfr_gpu.py imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def verify_config():
    """Verify configuration is valid"""
    try:
        import yaml
        with open('config/phase1_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Phase 1 config is valid")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def main():
    print("🔍 Verifying Aequus cleanup...")
    print("=" * 50)
    
    success = True
    
    # Test imports
    success &= verify_imports()
    
    # Test config
    success &= verify_config()
    
    print("=" * 50)
    if success:
        print("✅ All verifications passed - cleanup successful!")
        print("Ready to run: python test_phase1.py")
    else:
        print("❌ Some verifications failed")
        sys.exit(1)

if __name__ == "__main__":
    main()