# 🎯 Aequus - Production-Ready Poker AI

**Professional Texas Hold'em AI with 83M+ unique info sets**

## 🚀 Quick Start Guide

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/Sn0wfly/aequus_clean.git
cd aequus_clean

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_cleanup.py
```

### **2. Testing Your Setup**
```bash
# Run comprehensive tests
python test_phase1.py

# Verify system components
python -c "from poker_bot.core.trainer import PokerTrainer; print('✅ System ready')"
```

## 📋 Complete Training Guide

### **🎯 Basic Training Commands**

#### **Start New Training**
```bash
# Quick test (recommended first run)
python main_phase1.py --iterations 100 --save_every 20 --save_path test_run

# Standard training
python main_phase1.py --iterations 10000 --save_every 1000 --save_path aequus_production

# Large scale training
python main_phase1.py --iterations 50000 --save_every 5000 --save_path aequus_large
```

#### **Resume Training**
```bash
# Resume from checkpoint
python main_phase1.py --iterations 10000 --save_every 1000 --save_path my_bot --resume checkpoint_2000.pkl

# Continue with more iterations
python main_phase1.py --iterations 20000 --save_every 1000 --save_path my_bot --resume aequus_production_phase1_checkpoint_5000.pkl
```

### **📊 Model Analysis Commands**

#### **Compare Checkpoints**
```bash
# Compare two checkpoints
python compare_models.py checkpoint_1000.pkl checkpoint_2000.pkl

# Compare latest checkpoints
python compare_models.py *.pkl

# Specific comparison
python compare_models.py aequus_production_phase1_checkpoint_1000.pkl aequus_production_phase1_checkpoint_5000.pkl
```

#### **Monitor Training Progress**
```bash
# Check training logs
tail -f phase1_training.log

# List all checkpoints
ls -la *.pkl

# Check checkpoint details
python -c "
import pickle
with open('aequus_production_phase1_checkpoint_1000.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f'Unique info sets: {data[\"unique_info_sets\"]:,}')
    print(f'Total games: {data[\"total_games\"]:,}')
"
```

### **🤖 CLI Interface Commands**

#### **Using the CLI**
```bash
# Install CLI (optional)
pip install -e .

# Train via CLI
poker-bot train --iterations 10000 --batch-size 32768 --save-interval 1000 --model-path models/aequus.pkl

# Test bot
poker-bot play --model aequus_production_phase1_final.pkl

# System evaluation
poker-bot evaluate
```

### **☁️ Vast.ai Deployment**

#### **Deploy to Cloud**
```bash
# Make deployment script executable
chmod +x scripts/deploy_phase1_vastai.sh

# Deploy to Vast.ai
./scripts/deploy_phase1_vastai.sh

# Monitor cloud training
./monitor_phase1.sh
```

#### **Cloud Training Commands**
```bash
# On Vast.ai instance
python main_phase1.py --iterations 50000 --save_every 5000 --save_path aequus_cloud

# Background training
nohup python main_phase1.py --iterations 100000 --save_every 10000 --save_path aequus_full > training.log 2>&1 &
```

## 🎯 Training Examples

### **Example 1: Quick Test**
```bash
# 100 iterations, save every 20
python main_phase1.py --iterations 100 --save_every 20 --save_path quick_test

# Expected: ~600k unique info sets in 100 iterations
```

### **Example 2: Production Training**
```bash
# 10k iterations, save every 1k
python main_phase1.py --iterations 10000 --save_every 1000 --save_path production_run

# Expected: ~40M unique info sets in 10k iterations
```

### **Example 3: Large Scale**
```bash
# 50k iterations, save every 5k
python main_phase1.py --iterations 50000 --save_every 5000 --save_path large_scale

# Expected: ~83M+ unique info sets in 50k iterations
```

## 📊 Understanding Results

### **Key Metrics to Monitor**
- **Unique info sets**: Should grow consistently (target: 40M+ for 10k iterations)
- **Total games**: Batch size × iterations (32,768 × iterations)
- **Strategy entropy**: Should stabilize around 2.6-2.7
- **Growth rate**: Expect 200-300% growth between checkpoints

### **Sample Output Analysis**
```bash
# After running compare_models.py
📊 Basic Metrics:
  Model 1 - Unique info sets: 592,641
  Model 2 - Unique info sets: 1,988,141
  Growth: 1,395,500 (235.5%) ✅ Healthy growth
  Model 1 - Total games: 327,680
  Model 2 - Total games: 1,638,400
```

## 🛠️ Troubleshooting

### **Common Issues**
```bash
# CUDA/GPU issues
python -c "import jax; print(jax.devices())"

# Memory issues (reduce batch size)
python main_phase1.py --iterations 1000 --save_every 100 --save_path low_memory --batch-size 8192

# Import issues
python verify_cleanup.py
```

### **Performance Tuning**
```bash
# For H100 GPUs (optimal)
python main_phase1.py --iterations 10000 --batch-size 32768

# For A100 GPUs
python main_phase1.py --iterations 10000 --batch-size 16384

# For RTX 4090
python main_phase1.py --iterations 10000 --batch-size 8192
```

## 📁 File Structure After Training

```
aequus_clean/
├── aequus_production_phase1_checkpoint_*.pkl  # Training checkpoints
├── aequus_production_phase1_final.pkl         # Final model
├── phase1_training.log                        # Training logs
├── COMMANDS.md                                # This guide
└── ... (original clean structure)
```

## 🎉 Next Steps

1. **Start with quick test**: `python main_phase1.py --iterations 100 --save_every 20 --save_path test`
2. **Scale up gradually**: Increase iterations based on your GPU capacity
3. **Monitor progress**: Use `compare_models.py` to track growth
4. **Deploy to Vast.ai**: Use `scripts/deploy_phase1_vastai.sh` for cloud training

**Happy training!** 🚀