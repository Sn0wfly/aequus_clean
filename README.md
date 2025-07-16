# 🃏 Aequus Poker AI

A professional-grade Texas Hold'em poker AI using Counterfactual Regret Minimization (CFR) with JAX GPU acceleration.

## 🎯 Features

- **Real Hand Evaluation**: Uses `phevaluator` for accurate poker hand ranking
- **GPU-Accelerated Training**: JAX-based CFR implementation for fast training
- **Professional Architecture**: 50,000 information sets with advanced bucketing
- **Comprehensive Testing**: Unit tests for poker concepts and system integrity
- **Easy CLI Interface**: Train models without editing code
- **Real-Time Evaluation**: Built-in Poker IQ scoring system

## 🚀 Quick Start

### Prerequisites

```bash
# GPU environment (vast.ai, Google Colab, etc.)
pip install -r requirements.txt
```

### Train Your First Model

```bash
# Quick test (30 seconds)
python train.py --iterations 50 --output my_first_model

# Standard model (5-10 minutes) 
python train.py --iterations 1000 --output standard_model

# Professional model (1-2 hours)
python train.py --config superhuman --iterations 2000 --output pro_model
```

### Evaluate Model Performance

```bash
# Comprehensive evaluation with Poker IQ scoring
python evaluate_model.py models/standard_model_final.pkl

# Run unit tests to verify concepts
python test_poker_concepts.py
```

## 📊 Performance Benchmarks

| Model | Iterations | Training Time | Poker IQ | Level |
|-------|------------|---------------|----------|-------|
| Quick Test | 50 | ~30s | 45-55/120 | 🥉 Beginner |
| Standard | 1000 | ~10min | 75-85/120 | 🥈 Intermediate |
| Professional | 2000+ | ~1-2h | 85-100/120 | 🏆 Advanced |
| Elite | 5000+ | ~3-5h | 95-110/120 | 🚀 Expert |

## 🧠 Poker IQ Evaluation

The system evaluates models across 5 core poker concepts:

- **Hand Strength** (25 pts): Does it play AA more aggressively than 72o?
- **Position Awareness** (25 pts): Does it play tighter in early position?
- **Suited Recognition** (20 pts): Does it prefer suited hands?
- **Fold Discipline** (15 pts): Does it fold weak hands appropriately?
- **Strategy Diversity** (15 pts): Does it use varied strategies?

**Bonus**: Iteration bonus (20 pts) rewards longer training.

## 🔧 CLI Usage

### Basic Training

```bash
python train.py --iterations 1000 --output my_model
```

### Advanced Training

```bash
# Super-human configuration with custom parameters
python train.py \
  --config superhuman \
  --iterations 2000 \
  --batch-size 256 \
  --output elite_model \
  --snapshots 500 1000 1500 2000
```

### CLI Options

| Option | Description | Example |
|--------|-------------|---------|
| `--iterations` | Number of training iterations | `--iterations 1000` |
| `--output` | Model name (without .pkl) | `--output my_model` |
| `--config` | standard or superhuman | `--config superhuman` |
| `--batch-size` | Training batch size | `--batch-size 256` |
| `--snapshots` | IQ evaluation points | `--snapshots 500 1000` |
| `--no-snapshots` | Disable evaluations (faster) | `--no-snapshots` |

## 🧪 Testing

### Run All Tests

```bash
python test_poker_concepts.py
```

### Individual Test Categories

- **Hand Evaluator**: Verifies phevaluator integration
- **System Integrity**: Validates training pipeline
- **Poker Concepts**: Tests learned poker knowledge

### Expected Test Results

```
✅ Hand Evaluator tests PASSED
✅ System Integrity tests PASSED  
✅ Poker Concepts tests COMPLETED
🎯 Hand Strength Test: 75% success rate
🎯 Suited Test: 67% success rate
```

## 📁 Project Structure

```
aequus_clean/
├── poker_bot/
│   ├── core/
│   │   ├── trainer.py          # Main CFR training logic
│   │   ├── full_game_engine.py # Game simulation
│   │   └── ...
│   └── evaluator.py            # Hand evaluation (phevaluator wrapper)
├── models/                     # Trained models (.pkl files)
├── train.py                    # CLI training interface
├── evaluate_model.py           # Model evaluation script
├── test_poker_concepts.py      # Unit tests
└── requirements.txt            # Dependencies
```

## 🔬 Technical Details

### Algorithm: Counterfactual Regret Minimization (CFR)

- **Information Sets**: 50,000 unique game situations
- **Actions**: 6 (FOLD, CHECK, CALL, BET, RAISE, ALL_IN)
- **Bucketing**: Advanced multi-dimensional bucketing:
  - Street (preflop/flop/turn/river)
  - Hand strength (169 preflop combinations)
  - Position (6 positions)
  - Stack depth (20 buckets)
  - Pot odds (10 buckets)

### GPU Acceleration

- **JAX JIT Compilation**: ~10x speedup over pure Python
- **Vectorized Operations**: Batch processing for efficiency
- **Memory Optimized**: Efficient storage of 50K x 6 strategy matrices

### Hand Evaluation

- **phevaluator**: Ultra-fast C++ poker hand evaluator
- **7-card evaluation**: Handles hole cards + community cards
- **Real showdowns**: No mock/synthetic hand rankings

## 🐛 Troubleshooting

### Common Issues

1. **Import Error: No module named 'phevaluator'**
   ```bash
   pip install phevaluator
   ```

2. **JAX CUDA Issues**
   ```bash
   pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

3. **Low Poker IQ Scores**
   - Ensure you're using the real evaluator (not mocks)
   - Train for more iterations (1000+ recommended)
   - Use superhuman config for better learning

4. **Training Too Slow**
   - Use `--no-snapshots` for faster training
   - Reduce batch size if memory issues
   - Ensure GPU is available and JAX detects it

### Validation Errors

If training aborts with validation errors:

```bash
# Check evaluator works
python test_real_evaluator.py

# Run comprehensive debugging
python debug_evaluator_detailed.py
```

## 🏆 Advanced Usage

### Custom Training Configurations

```python
from poker_bot.core.trainer import PokerTrainer, TrainerConfig

# Create custom config
config = TrainerConfig()
config.batch_size = 512
config.learning_rate = 0.02
config.position_awareness_factor = 0.5

# Train with custom config
trainer = PokerTrainer(config)
trainer.train(2000, 'custom_model', 500)
```

### Model Analysis

```python
# Load and analyze trained model
trainer = PokerTrainer(TrainerConfig())
trainer.load_model('models/my_model_final.pkl')

# Examine strategies for specific situations
from poker_bot.core.trainer import compute_mock_info_set
aa_info = compute_mock_info_set([12, 12], False, 2)  # AA in middle position
strategy = trainer.strategy[aa_info]
print(f"AA strategy: {strategy}")  # [fold, check, call, bet, raise, all_in]
```

## 📈 Roadmap

- [ ] **Multi-street Training**: Full preflop → river training
- [ ] **Opponent Modeling**: Adaptive strategies against different player types
- [ ] **Tournament ICM**: Independent Chip Model for tournament play
- [ ] **Real-time Interface**: API for live poker integration
- [ ] **Pluribus-level**: 6-max no-limit training to match state-of-the-art

## 🤝 Contributing

1. **Bug Reports**: Use GitHub issues with full error traces
2. **Feature Requests**: Describe use case and implementation ideas
3. **Code Contributions**: Follow existing code style and add tests
4. **Testing**: Run `python test_poker_concepts.py` before submitting

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **phevaluator**: Fast poker hand evaluation library
- **JAX**: High-performance machine learning library
- **CFR Algorithm**: Introduced by Zinkevich et al. (2008)
- **Pluribus**: Inspiration from Facebook's superhuman poker AI

## 📞 Support

- **Issues**: GitHub issue tracker
- **Questions**: Create discussion in GitHub
- **Performance**: Share benchmark results and hardware specs

---

*Built with ❤️ for the poker AI community*