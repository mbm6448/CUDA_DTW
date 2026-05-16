# CUDA DTW - build rules
#
# Targets:
#   make            -> bin/dtw_test   (benchmark + functional driver)
#   make test       -> runs gtest unit tests
#   make bench      -> runs the benchmark driver
#   make basic      -> runs functional tests only
#   make query      -> bin/query (small CUDA device-info utility)
#   make clean      -> remove build artifacts

NVCC  ?= nvcc
CXX   ?= g++

# Architectures. CUDA 13 dropped support for pre-Hopper (sm < 100), so this
# is the supported range. Override on the command line if you need to retarget,
# e.g.  make GENCODE_FLAGS="-gencode arch=compute_120,code=sm_120"
GENCODE_FLAGS ?= \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120

NVCC_FLAGS := -O3 -std=c++17 $(GENCODE_FLAGS) -lineinfo \
              -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -Wextra \
              -Iinclude

LDFLAGS := -Xcompiler -fopenmp

BIN_DIR  := bin
BUILD    := build

DTW_OBJ  := $(BUILD)/DTW.o
MAIN_OBJ := $(BUILD)/main.o
TEST_OBJ := $(BUILD)/TestDTW.o
QUERY_OBJ:= $(BUILD)/query.o

MAIN_BIN  := $(BIN_DIR)/dtw_test
TEST_BIN  := $(BIN_DIR)/test_run
QUERY_BIN := $(BIN_DIR)/query

GTEST_LIBS := -lgtest -lgtest_main -lpthread

.PHONY: all test bench basic query clean dirs

all: dirs $(MAIN_BIN)

dirs:
	@mkdir -p $(BIN_DIR) $(BUILD)

$(MAIN_BIN): $(MAIN_OBJ) $(DTW_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^

$(TEST_BIN): $(TEST_OBJ) $(DTW_OBJ)
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ $^ $(GTEST_LIBS)

$(QUERY_BIN): $(QUERY_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(DTW_OBJ): src/DTW.cu include/DTW.h | dirs
	$(NVCC) $(NVCC_FLAGS) -c src/DTW.cu -o $@

$(MAIN_OBJ): src/main.cu include/DTW.h | dirs
	$(NVCC) $(NVCC_FLAGS) -c src/main.cu -o $@

$(TEST_OBJ): test/TestDTW.cu include/DTW.h | dirs
	$(NVCC) $(NVCC_FLAGS) -c test/TestDTW.cu -o $@

$(QUERY_OBJ): src/query.cu | dirs
	$(NVCC) $(NVCC_FLAGS) -c src/query.cu -o $@

test: dirs $(TEST_BIN)
	./$(TEST_BIN)

bench: $(MAIN_BIN)
	./$(MAIN_BIN) --perf

basic: $(MAIN_BIN)
	./$(MAIN_BIN) --basic

query: dirs $(QUERY_BIN)
	./$(QUERY_BIN)

clean:
	rm -rf $(BUILD) $(BIN_DIR)
