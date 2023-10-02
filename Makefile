NVCC = nvcc
NVCC_FLAGS = -O3  -Iinclude -I/home/mbm6448/Desktop/googlTest/googletest/googletest/include
MAIN_OBJS = obj/DTW.o obj/main.o
TEST_OBJS = obj/DTW.o obj/TestDTW.o
MAIN_TARGET = bin/main
TEST_TARGET = bin/test_run
GTEST_LIB = -L/home/mbm6448/Desktop/googlTest/googletest/build/lib/ -lgtest -lgtest_main -lpthread

# Main Compile and Link
$(MAIN_TARGET): $(MAIN_OBJS)
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(MAIN_OBJS) -o $(MAIN_TARGET)

# Test Compile and Link
$(TEST_TARGET): $(TEST_OBJS)
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(TEST_OBJS) -o $(TEST_TARGET) $(GTEST_LIB)

# Compile each source file
obj/%.o: src/%.cu
	mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile each test file
obj/%.o: test/%.cu
	mkdir -p obj
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean object files and executable
clean:
	rm -rf obj bin

# Build everything
all: $(MAIN_TARGET) $(TEST_TARGET)

# Build tests
test: $(TEST_TARGET)



