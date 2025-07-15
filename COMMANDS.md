# 🎯 Aequus Poker Bot - Available Commands

## Production Commands

### Training
```bash
# Phase 1 Enhanced Training
python main_phase1.py --iterations 10000 --save_every 1000 --save_path aequus_production

# Resume training
python main_phase1.py --iterations 10000 --save_every 1000 --save_path my_bot --resume checkpoint_2000.pkl

# Quick test
python main_phase1.py --iterations 1000 --save_every 100 --save_path test_run
```

### Testing
```bash
# Run comprehensive tests
python test_phase1.py

# Compare model checkpoints
python compare_models.py checkpoint_1000.pkl checkpoint_2000.pkl
```

### CLI Interface (via poker-bot command)
```bash
# Install first: pip install -e .
poker-bot train --iterations 10000 --batch-size 32768 --save-interval 1000
poker-bot play --model path/to/model.pkl
poker-bot evaluate
```

### Vast.ai Deployment
```bash
# Deploy to Vast.ai
bash scripts/deploy_phase1_vastai.sh

# Monitor training
./monitor_phase1.sh

# Run benchmarks
./benchmark_phase1.sh
```

## Configuration Files

- `config/phase1_config.yaml` - Production Phase 1 configuration
- `config/training_config.yaml` - Legacy training configuration (deprecated)

## Core Modules Used

- `poker_bot/core/trainer.py` - Main trainer (✅ Production ready)
- `poker_bot/core/simulation.py` - Game simulation (✅ Production ready)
- `poker_bot/core/enhanced_eval.py` - Enhanced evaluation (✅ Production ready)
- `poker_bot/core/icm_modeling.py` - ICM calculations (✅ Production ready)
- `poker_bot/core/mccfr_gpu.py` - GPU MCCFR (✅ Production ready)
- `poker_bot/core/cfr_gpu.py` - GPU CFR (✅ Production ready)

## Unused/Legacy Modules (to be removed)
- `poker_bot/core/advanced_mccfr.py` - Legacy
- `poker_bot/core/bucket_fine.py` - Legacy
- `poker_bot/core/bucket_gpu.py` - Legacy
- `poker_bot/core/production_optimization.py` - Legacy
- `poker_bot/core/history_aware_bucketing.py` - Phase 2 feature