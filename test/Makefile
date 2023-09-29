# Variables
NVCC = nvcc
ARCH := -arch=sm_$(shell nvidia-smi --query-gpu=compute_capability --format=csv,noheader,nounits | head -n 1 | tr -d '.')
NVCC_FLAGS = -O3 $(ARCH)
OBJS = TestDTW.o
TARGET = test_run
GTEST_LIB = -lgtest -lgtest_main -lpthread

# Compile and Link
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET) $(GTEST_LIB)

# Compile each source file
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean object files and executable
clean:
	rm -f $(OBJS) $(TARGET)
