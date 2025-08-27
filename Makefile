# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14
NVCC_FLAGS += -Xcompiler -Wall -Xcompiler -Wextra


# Target executable
TARGET = dtw_test

# Source files
SOURCES = main.cu DTW.cu

# Object files
OBJECTS = main.o DTW.o

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compile main.cu
main.o: main.cu DTW.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu -o $@

# Compile DTW.cu
DTW.o: DTW.cu DTW.h
	$(NVCC) $(NVCC_FLAGS) -c DTW.cu -o $@

# Run all tests
test: $(TARGET)
	./$(TARGET) --all

# Run performance benchmark
benchmark: $(TARGET)
	./$(TARGET) --perf

# Run basic tests only
basic: $(TARGET)
	./$(TARGET) --basic

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install (optional - adjust path as needed)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: all test benchmark basic clean install