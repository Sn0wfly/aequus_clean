# 🚀 POKER CUDA CFR - PRODUCTION Makefile
# =====================================
# Compiles complete CUDA poker system

# CUDA compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_70 --compiler-options '-fPIC' -shared -Xcompiler -fopenmp
CUDA_LIBS = -lcurand

# CUDA path
CUDA_PATH ?= $(shell which nvcc | sed 's/\/bin\/nvcc//')
ifeq ($(CUDA_PATH),)
	CUDA_PATH = /usr/local/cuda
endif

CUDA_INCLUDE = -I$(CUDA_PATH)/include
CUDA_LIB_PATH = -L$(CUDA_PATH)/lib64

# Source files
HAND_EVALUATOR_SRC = hand_evaluator.cu hand_evaluator_real.cu
CFR_KERNELS_SRC = cfr_kernels.cu cfr_advanced.cu
ALL_SOURCES = $(HAND_EVALUATOR_SRC) $(CFR_KERNELS_SRC)
OUTPUT_LIB = libpoker_cuda.so

.PHONY: all clean test production quick

all: $(OUTPUT_LIB)

$(OUTPUT_LIB): $(ALL_SOURCES)
	@echo "🔥 Compiling COMPLETE CUDA Poker system..."
	@echo "CUDA Path: $(CUDA_PATH)"
	$(NVCC) $(CUDA_FLAGS) $(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(ALL_SOURCES) $(CUDA_LIBS) -o $(OUTPUT_LIB)
	@echo "✅ Complete compilation successful: $(OUTPUT_LIB)"
	@ls -lh $(OUTPUT_LIB)

# Production build with maximum optimization
production:
	@echo "🏆 PRODUCTION BUILD - Maximum optimization..."
	$(NVCC) -O3 -arch=sm_70 --compiler-options '-fPIC -march=native' \
		-shared -Xcompiler -fopenmp --use_fast_math \
		$(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(ALL_SOURCES) $(CUDA_LIBS) -o $(OUTPUT_LIB)
	@echo "✅ Production build complete"

# Quick compile for development
quick:
	$(NVCC) -O2 -arch=sm_70 --compiler-options '-fPIC' -shared \
		$(CUDA_INCLUDE) $(CUDA_LIB_PATH) \
		$(ALL_SOURCES) -lcurand -o $(OUTPUT_LIB)

# Test production system
test_production: $(OUTPUT_LIB)
	@echo "🚀 Testing PRODUCTION CUDA system..."
	python3 cuda_trainer_production.py

# Basic library test
test: $(OUTPUT_LIB)
	@echo "🧪 Testing CUDA library..."
	python3 -c "import ctypes; lib = ctypes.CDLL('./$(OUTPUT_LIB)'); print('✅ Library loaded')"

# Check CUDA environment
check_cuda:
	@echo "🔍 CUDA Environment:"
	@nvcc --version || echo "❌ nvcc not found"
	@nvidia-smi || echo "❌ nvidia-smi not found"

# Clean build files
clean:
	rm -f $(OUTPUT_LIB) *.o *.so
	@echo "🧹 Cleaned build files"

help:
	@echo "🚀 CUDA Poker CFR - PRODUCTION Build System"
	@echo "==========================================="
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build complete system"
	@echo "  production      - Optimized production build"
	@echo "  quick           - Fast development build"
	@echo "  test            - Test library loading"
	@echo "  test_production - Test complete system"
	@echo "  check_cuda      - Check CUDA environment"
	@echo "  clean           - Remove build files"

.DEFAULT_GOAL := all 