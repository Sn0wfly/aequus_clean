#!/bin/bash
# 🚀 Phase 1 Deployment Script for Vast.ai
# Enhanced poker AI with pro-level quality improvements

set -e

echo "🎯 Phase 1 Enhanced Poker AI Deployment"
echo "======================================="

# Configuration
INSTANCE_TYPE="H100"  # or "A100" based on availability
CUDA_VERSION="12.1"
PYTHON_VERSION="3.10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running on Vast.ai
if [[ -f /etc/vast_ai_image ]]; then
    log "✅ Running on Vast.ai instance"
else
    warn "Not running on Vast.ai - some optimizations may not apply"
fi

# System setup
log "📦 Setting up system..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-dev git build-essential

# Install CUDA (if not already present)
if ! command -v nvidia-smi &> /dev/null; then
    log "🔧 Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
fi

# Verify GPU
log "🔍 Checking GPU availability..."
nvidia-smi

# Create virtual environment
log "🐍 Setting up Python environment..."
python3 -m venv /opt/poker_env
source /opt/poker_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install JAX with CUDA support
log "⚡ Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
log "📚 Installing dependencies..."
pip install -r requirements.txt
pip install cupy-cuda12x  # For GPU acceleration

# Install project in development mode
log "🔨 Installing Aequus..."
pip install -e .

# Verify installation
log "✅ Verifying installation..."
python -c "import jax; print('JAX version:', jax.__version__)"
python -c "import cupy; print('CuPy version:', cupy.__version__)"
python -c "import poker_bot; print('Aequus installed successfully')"

# Create directories
log "📁 Creating directories..."
mkdir -p models/phase1
mkdir -p logs/phase1
mkdir -p benchmarks/phase1

# Pre-compute lookup tables
log "🧮 Pre-computing lookup tables..."
python -c "
from poker_bot.core.enhanced_eval import EnhancedHandEvaluator
from poker_bot.core.icm_modeling import ICMModel
print('Pre-computing enhanced evaluation tables...')
evaluator = EnhancedHandEvaluator()
print('Pre-computing ICM tables...')
icm = ICMModel()
print('✅ Lookup tables ready')
"

# Run Phase 1 tests
log "🧪 Running Phase 1 tests..."
python test_phase1.py

# Start training
log "🚀 Starting Phase 1 training..."
echo "Available commands:"
echo "  python main_phase1.py --benchmark          # Run benchmarks"
echo "  python main_phase1.py --iterations 1000    # Quick test"
echo "  python main_phase1.py --iterations 10000   # Full training"
echo "  python main_phase1.py --help               # Show all options"

# Create monitoring script
cat > monitor_phase1.sh << 'EOF'
#!/bin/bash
# Phase 1 monitoring script
source /opt/poker_env/bin/activate

echo "📊 Phase 1 Training Monitor"
echo "=========================="
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
echo ""
echo "Training Progress:"
tail -n 20 phase1_training.log
echo ""
echo "Latest Model:"
ls -la models/phase1/ | tail -n 5
EOF

chmod +x monitor_phase1.sh

# Create benchmark script
cat > benchmark_phase1.sh << 'EOF'
#!/bin/bash
# Phase 1 benchmark script
source /opt/poker_env/bin/activate

echo "🔥 Phase 1 Benchmark"
echo "===================="
python main_phase1.py --benchmark
EOF

chmod +x benchmark_phase1.sh

log "✅ Phase 1 deployment complete!"
log ""
log "🎯 Next steps:"
log "  1. Run: ./monitor_phase1.sh"
log "  2. Test: ./benchmark_phase1.sh"
log "  3. Train: python main_phase1.py --iterations 10000"
log ""
log "📊 Performance targets:"
log "   - Games/sec: >800 (vs current 1000+)"
log "   - Memory: <4GB (vs current 2.5GB)"
log "   - Quality: 2.5x better convergence"