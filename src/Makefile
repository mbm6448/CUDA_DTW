# Variables
NVCC = nvcc
ARCH := -arch=sm_$(shell nvidia-smi --query-gpu=compute_capability --format=csv,noheader,nounits | head -n 1 | tr -d '.')
NVCC_FLAGS = -O3 $(ARCH)
OBJS = DTW.o main.o
TARGET = main

# Compile and Link
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET)

# Compile each source file
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean object files and executable
clean:
	rm -f $(OBJS) $(TARGET)
