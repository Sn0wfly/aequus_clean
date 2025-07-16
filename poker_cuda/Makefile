# 🚀 POKER CUDA CFR - Makefile
# ===========================
# Compiles CUDA kernels and creates shared library for Python

# CUDA compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_70 --compiler-options '-fPIC' -shared -Xcompiler -fopenmp
CUDA_LIBS = -lcurand

# Determine CUDA path automatically
CUDA_PATH ?= $(shell which nvcc | sed 's/\/bin\/nvcc//')
ifeq ($(CUDA_PATH),)
	CUDA_PATH = /usr/local/cuda
endif

# Include and library paths
CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB_PATH = -L$(CUDA_PATH)/lib64

# Source files
HAND_EVALUATOR_SRC = hand_evaluator.cu
CFR_KERNELS_SRC = cfr_kernels.cu
OUTPUT_LIB = libpoker_cuda.so

# Build targets
.PHONY: all clean test install benchmark

all: $(OUTPUT_LIB)

$(OUTPUT_LIB): $(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC)
	@echo "🔥 Compiling CUDA Poker CFR kernels..."
	@echo "CUDA Path: $(CUDA_PATH)"
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC) \
		$(CUDA_LIBS) -o $(OUTPUT_LIB)
	@echo "✅ Compilation successful: $(OUTPUT_LIB)"
	@ls -lh $(OUTPUT_LIB)

# Quick compile for development
quick:
	$(NVCC) -O2 -arch=sm_70 --compiler-options '-fPIC' -shared \
		$(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC) \
		-lcurand -o $(OUTPUT_LIB)

# Production build with maximum optimization
production:
	@echo "🏆 Production build with maximum optimization..."
	$(NVCC) -O3 -arch=sm_70,sm_75,sm_80 --compiler-options '-fPIC -march=native' \
		-shared -Xcompiler -fopenmp --use_fast_math \
		$(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC) \
		-lcurand -lcublas -o $(OUTPUT_LIB)
	@echo "✅ Production build complete"

# Test compilation and basic functionality
test: $(OUTPUT_LIB)
	@echo "🧪 Testing CUDA library..."
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(OUTPUT_LIB)'); print('✅ Library loaded successfully')"

# Run Python trainer test
test_trainer: $(OUTPUT_LIB)
	@echo "🚀 Testing CUDA CFR trainer..."
	python3 cuda_trainer.py

# Performance benchmark
benchmark: $(OUTPUT_LIB)
	@echo "📊 Running performance benchmark..."
	python3 -c "from cuda_trainer import benchmark_cuda_vs_alternatives; benchmark_cuda_vs_alternatives(batch_size=512)"

# Install to system (optional)
install: $(OUTPUT_LIB)
	@echo "📦 Installing CUDA library..."
	sudo cp $(OUTPUT_LIB) /usr/local/lib/
	sudo ldconfig
	@echo "✅ Installed to /usr/local/lib/"

# Debug build with symbols
debug:
	$(NVCC) -g -G -O0 -arch=sm_70 --compiler-options '-fPIC' -shared \
		$(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC) \
		-lcurand -o $(OUTPUT_LIB)
	@echo "🐛 Debug build complete"

# Check CUDA environment
check_cuda:
	@echo "🔍 CUDA Environment Check:"
	@echo "NVCC Version:"
	@nvcc --version || echo "❌ nvcc not found"
	@echo ""
	@echo "CUDA Devices:"
	@nvidia-smi || echo "❌ nvidia-smi not found" 
	@echo ""
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "Include: $(CUDA_INCLUDE)"
	@echo "Lib Path: $(CUDA_LIB_PATH)"

# Clean build files
clean:
	rm -f $(OUTPUT_LIB) *.o *.so
	@echo "🧹 Cleaned build files"

# Show help
help:
	@echo "🚀 CUDA Poker CFR - Build System"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Build shared library (default)"
	@echo "  quick       - Fast development build"
	@echo "  production  - Optimized production build"
	@echo "  test        - Test library loading"
	@echo "  test_trainer- Test full Python trainer"
	@echo "  benchmark   - Run performance benchmark"
	@echo "  debug       - Build with debug symbols"
	@echo "  check_cuda  - Check CUDA environment"
	@echo "  clean       - Remove build files"
	@echo "  install     - Install to system (sudo)"

# Default help when no target specified
.DEFAULT_GOAL := help 