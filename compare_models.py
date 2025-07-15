#!/usr/bin/env python3
"""
Compare different model checkpoints
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import glob
import argparse
import numpy as np

def load_model(path):
    """Load model from pickle file"""
    with open(path, 'rb') as f:
        return pickle.load(f)

def compare_models(model1_path, model2_path):
    """Compare two model checkpoints"""
    print(f"🔍 Comparing models:")
    print(f"  Model 1: {model1_path}")
    print(f"  Model 2: {model2_path}")
    print("=" * 60)
    
    model1 = load_model(model1_path)
    model2 = load_model(model2_path)
    
    # Basic comparison
    print("📊 Basic Metrics:")
    print(f"  Model 1 - Unique info sets: {model1['unique_info_sets']:,}")
    print(f"  Model 2 - Unique info sets: {model2['unique_info_sets']:,}")
    print(f"  Growth: {model2['unique_info_sets'] - model1['unique_info_sets']:,} ({((model2['unique_info_sets'] - model1['unique_info_sets']) / model1['unique_info_sets'] * 100):.1f}%)")
    
    print(f"  Model 1 - Total games: {model1['total_games']:,}")
    print(f"  Model 2 - Total games: {model2['total_games']:,}")
    
    # Strategy comparison
    if 'strategies' in model1 and 'strategies' in model2:
        strategies1 = model1['strategies']
        strategies2 = model2['strategies']
        
        # Compare strategy entropy
        entropy1 = -np.sum(strategies1 * np.log(strategies1 + 1e-8), axis=1)
        entropy2 = -np.sum(strategies2 * np.log(strategies2 + 1e-8), axis=1)
        
        print("\n🎯 Strategy Analysis:")
        print(f"  Model 1 - Avg entropy: {np.mean(entropy1):.3f}")
        print(f"  Model 2 - Avg entropy: {np.mean(entropy2):.3f}")
    
    return {
        'model1_info_sets': model1['unique_info_sets'],
        'model2_info_sets': model2['unique_info_sets'],
        'growth': model2['unique_info_sets'] - model1['unique_info_sets'],
        'model1_games': model1['total_games'],
        'model2_games': model2['total_games']
    }

def find_latest_checkpoints(pattern="*.pkl"):
    """Find latest checkpoint files"""
    files = glob.glob(pattern)
    if not files:
        print("❌ No checkpoint files found")
        return None
    
    # Sort by creation time
    files.sort(key=os.path.getctime)
    return files[-2:] if len(files) >= 2 else files

def main():
    parser = argparse.ArgumentParser(description="Compare model checkpoints")
    parser.add_argument("files", nargs="*", help="Model files to compare")
    parser.add_argument("--pattern", default="*.pkl", help="Pattern for checkpoint files")
    
    args = parser.parse_args()
    
    if args.files:
        if len(args.files) == 2:
            compare_models(args.files[0], args.files[1])
        else:
            print("❌ Please provide exactly 2 files to compare")
    else:
        # Auto-find latest checkpoints
        latest = find_latest_checkpoints(args.pattern)
        if latest and len(latest) >= 2:
            compare_models(latest[-2], latest[-1])
        else:
            print("❌ Not enough checkpoint files found")

if __name__ == "__main__":
    main()