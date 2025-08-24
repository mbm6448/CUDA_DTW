# CUDA compiler and flags
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_70  
NVCC_FLAGS = $(CUDA_FLAGS) -Iinclude -I/home/mbm6448/Desktop/googlTest/googletest/googletest/include

# Object files
MAIN_OBJS = obj/DTW.o obj/main.o
TEST_OBJS = obj/DTW.o obj/TestDTW.o
QUERY_OBJ = obj/query.o

# Targets
MAIN_TARGET = bin/main
TEST_TARGET = bin/test_run
QUERY_TARGET = bin/query

# Google Test library
GTEST_LIB = -L/home/mbm6448/Desktop/googlTest/googletest/build/lib/ -lgtest -lgtest_main -lpthread

# Default target
all: $(MAIN_TARGET) $(TEST_TARGET) $(QUERY_TARGET)

# Main executable
$(MAIN_TARGET): $(MAIN_OBJS)
	@mkdir -p bin
	$(NVCC) $(CUDA_FLAGS) $(MAIN_OBJS) -o $(MAIN_TARGET)
	@echo "Built main executable: $(MAIN_TARGET)"

# Test executable
$(TEST_TARGET): $(TEST_OBJS)
	@mkdir -p bin
	$(NVCC) $(CUDA_FLAGS) $(TEST_OBJS) -o $(TEST_TARGET) $(GTEST_LIB)
	@echo "Built test executable: $(TEST_TARGET)"

# Query executable for checking CUDA capability
$(QUERY_TARGET): $(QUERY_OBJ)
	@mkdir -p bin
	$(NVCC) $(CUDA_FLAGS) $(QUERY_OBJ) -o $(QUERY_TARGET)
	@echo "Built query executable: $(QUERY_TARGET)"

# Compile DTW.cu
obj/DTW.o: src/DTW.cu include/DTW.h
	@mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c src/DTW.cu -o obj/DTW.o

# Compile main.cu
obj/main.o: src/main.cu include/DTW.h
	@mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c src/main.cu -o obj/main.o

# Compile TestDTW.cu
obj/TestDTW.o: test/TestDTW.cu include/DTW.h
	@mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c test/TestDTW.cu -o obj/TestDTW.o

# Compile query.cu
obj/query.o: src/query.cu
	@mkdir -p obj
	$(NVCC) $(CUDA_FLAGS) -c src/query.cu -o obj/query.o

# Build and run main
run: $(MAIN_TARGET)
	./$(MAIN_TARGET)

# Build and run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Check CUDA capability
check: $(QUERY_TARGET)
	./$(QUERY_TARGET)

# Clean build artifacts
clean:
	rm -rf obj bin
	@echo "Cleaned build artifacts"

# Debug build
debug: CUDA_FLAGS = -g -G -arch=sm_70
debug: NVCC_FLAGS = $(CUDA_FLAGS) -Iinclude -I/home/mbm6448/Desktop/googlTest/googletest/googletest/include
debug: all
	@echo "Built with debug symbols"

# Performance profiling build
profile: CUDA_FLAGS = -O3 -lineinfo -arch=sm_70
profile: NVCC_FLAGS = $(CUDA_FLAGS) -Iinclude -I/home/mbm6448/Desktop/googlTest/googletest/googletest/include
profile: all
	@echo "Built with profiling support"

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build all executables (default)"
	@echo "  run     - Build and run main program"
	@echo "  test    - Build and run tests"
	@echo "  check   - Check CUDA device capabilities"
	@echo "  clean   - Remove all build artifacts"
	@echo "  debug   - Build with debug symbols"
	@echo "  profile - Build with profiling support"
	@echo "  help    - Show this help message"

.PHONY: all run test check clean debug profile help