# Makefile
#
# Usage:
#   make            Build the core benchmark binary (no NCCL, single-GPU only)
#   make single     Alias for the default build — explicit single-GPU target
#   make USE_NCCL=1 Build with NCCL for multi-GPU tensor-parallel support
#   make clean      Remove all build artifacts and the benchmark binary
#
# The output binary is always named 'benchmark'.
# The core build (USE_NCCL=0, the default) covers:
#   - ar          : autoregressive decode
#   - sd          : speculative decoding (single GPU)
#   - ssd-single  : async speculative decoding on one GPU (two host threads)
# Building with USE_NCCL=1 additionally enables:
#   - ar (TP)     : tensor-parallel AR across multiple GPUs
#   - ssd (multi) : full multi-GPU async speculative decoding

NVCC      = nvcc
CXX_HOST  = /usr/bin/g++-13
EXTRA_FLAGS ?=
NVCCFLAGS = -O2 -ccbin $(CXX_HOST) -std=c++17 $(EXTRA_FLAGS)
LIBS      = -lcublas

# Set USE_NCCL=1 to enable tensor-parallel (multi-GPU) support
USE_NCCL ?= 0

# Source and build directories
SRC_DIR = src
BUILD_DIR = build

# Sources
SRCS = $(SRC_DIR)/main.cu \
       $(SRC_DIR)/forward_single.cu \
       $(SRC_DIR)/bench_common.cu \
       $(SRC_DIR)/bench_ar.cu \
       $(SRC_DIR)/bench_sd.cu \
       $(SRC_DIR)/bench_ssd.cu \
       $(SRC_DIR)/bench_ssd_single.cu

# Headers (for dependency tracking)
HDRS = $(SRC_DIR)/config.h \
       $(SRC_DIR)/check.h \
       $(SRC_DIR)/gemm.h \
       $(SRC_DIR)/gpu_context.h \
       $(SRC_DIR)/forward_single.h \
       $(SRC_DIR)/bench_common.h \
       $(SRC_DIR)/bench_ar.h \
       $(SRC_DIR)/bench_sd.h \
       $(SRC_DIR)/bench_ssd.h \
       $(SRC_DIR)/bench_ssd_single.h \
       $(SRC_DIR)/thread_safe_queue.h

ifeq ($(USE_NCCL),1)
  NVCCFLAGS += -DUSE_NCCL
  LIBS      += -lnccl -lnvToolsExt
  SRCS      += $(SRC_DIR)/forward_tp.cu
  HDRS      += $(SRC_DIR)/forward_tp.h
endif

# Object files
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Binary
TARGET = benchmark

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LIBS)

# Compile .cu -> .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HDRS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Convenience alias: build without NCCL (core single-GPU binary only)
single: USE_NCCL = 0
single: all

# multi-gpu: build with NCCL for tensor-parallel AR, SD, and multi-GPU SSD.
# Requires NCCL and a multi-GPU machine (see scripts/README.md for PSC setup).
multi-gpu:
	$(MAKE) USE_NCCL=1

.PHONY: all single multi-gpu clean
