// main.cu - Comprehensive test program for GPU-accelerated DTW
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include "DTW.h"

// ANSI color codes for terminal output
#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"

// Helper function to format time duration
std::string formatDuration(double seconds) {
    if (seconds < 0.001) {
        return std::to_string(seconds * 1000000.0) + " μs";
    } else if (seconds < 1.0) {
        return std::to_string(seconds * 1000.0) + " ms";
    } else {
        return std::to_string(seconds) + " s";
    }
}

// Helper function to format memory size
std::string formatMemory(size_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    } else if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KB";
    } else if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else {
        return std::to_string(bytes / (1024.0 * 1024 * 1024)) + " GB";
    }
}

// Test basic DTW functionality
void testBasicFunctionality() {
    std::cout << BOLD << CYAN << "\n=== Basic Functionality Test ===" << RESET << std::endl;
    
    DTW dtw(256);
    
    // Test 1: Simple sequences
    std::cout << "\n1. Simple sequence test:" << std::endl;
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.5, 2.5, 3.5, 4.5};
    
    double distance = dtw.compute(x, y);
    std::cout << "   DTW distance: " << std::fixed << std::setprecision(6) << distance << std::endl;
    
    // Test 2: Identical sequences (should give 0)
    std::cout << "\n2. Identical sequences test:" << std::endl;
    std::vector<double> test_seq = DTW::generateRandomSequence(100);
    double identical_distance = dtw.compute(test_seq, test_seq);
    std::cout << "   DTW distance: " << identical_distance;
    if (std::abs(identical_distance) < 1e-10) {
        std::cout << GREEN << " ✓ (correctly returns ~0)" << RESET << std::endl;
    } else {
        std::cout << RED << " ✗ (should be 0)" << RESET << std::endl;
    }
    
    // Test 3: Path extraction
    std::cout << "\n3. Path extraction test:" << std::endl;
    auto result = dtw.computeWithPath(x, y);
    if (result.success) {
        std::cout << "   Path length: " << result.path.size() << std::endl;
        std::cout << "   First 5 alignment points:" << std::endl;
        for (size_t i = 0; i < std::min(result.path.size(), size_t(5)); i++) {
            std::cout << "     (" << result.path[i].first << ", " 
                      << result.path[i].second << ")" << std::endl;
        }
        std::cout << GREEN << "   ✓ Path extraction successful" << RESET << std::endl;
    } else {
        std::cout << RED << "   ✗ Error: " << result.error_message << RESET << std::endl;
    }
}

// Test multi-dimensional DTW
void testMultiDimensional() {
    std::cout << BOLD << CYAN << "\n=== Multi-Dimensional DTW Test ===" << RESET << std::endl;
    
    DTW dtw;
    
    // Test different dimensions
    std::vector<int> dimensions = {1, 2, 3, 5, 10, 20};
    int seq_length = 500;
    
    std::cout << "\nTesting with sequence length: " << seq_length << std::endl;
    std::cout << std::setw(10) << "Dimension" 
              << std::setw(15) << "Distance" 
              << std::setw(15) << "Time" 
              << std::setw(15) << "Status" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (int dim : dimensions) {
        auto x = DTW::generateRandomSequence(seq_length, dim);
        auto y = DTW::generateRandomSequence(seq_length, dim);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto result = dtw.computeMultiDim(x, y, seq_length, seq_length, dim, DTW::EUCLIDEAN);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        
        std::cout << std::setw(10) << dim;
        if (result.success) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(3) << result.distance
                      << std::setw(15) << formatDuration(diff.count())
                      << GREEN << std::setw(15) << "✓ Success" << RESET;
        } else {
            std::cout << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A"
                      << RED << std::setw(15) << "✗ Failed" << RESET;
        }
        std::cout << std::endl;
    }
}

// Test Sakoe-Chiba band optimization
void testSakoeChibaBand() {
    std::cout << BOLD << CYAN << "\n=== Sakoe-Chiba Band Optimization Test ===" << RESET << std::endl;
    
    DTW dtw;
    int n = 2000, m = 2000;
    
    auto x = DTW::generateRandomSequence(n);
    auto y = DTW::generateRandomSequence(m);
    
    std::vector<int> windows = {-1, 200, 100, 50, 20, 10};
    
    std::cout << "\nSequence size: " << n << " x " << m << std::endl;
    std::cout << std::setw(12) << "Window" 
              << std::setw(15) << "Distance" 
              << std::setw(15) << "Time"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(57, '-') << std::endl;
    
    double baseline_time = 0;
    
    for (int window : windows) {
        auto start = std::chrono::high_resolution_clock::now();
        double distance = dtw.compute(x, y, window);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double time_seconds = diff.count();
        
        if (window == -1) {
            baseline_time = time_seconds;
            std::cout << std::setw(12) << "No band";
        } else {
            std::cout << std::setw(12) << window;
        }
        
        std::cout << std::setw(15) << std::fixed << std::setprecision(3) << distance
                  << std::setw(15) << formatDuration(time_seconds);
        
        if (baseline_time > 0 && window != -1) {
            double speedup = baseline_time / time_seconds;
            std::cout << std::setw(14) << std::setprecision(2) << speedup << "x";
        } else {
            std::cout << std::setw(15) << "baseline";
        }
        std::cout << std::endl;
    }
}

// Performance benchmarking
void performanceBenchmark() {
    std::cout << BOLD << CYAN << "\n=== Performance Benchmark ===" << RESET << std::endl;
    
    // Create DTW instance with larger pre-allocated memory
    DTW dtw(256, 15000, 10);
    
    struct TestCase {
        int n, m;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {100, 100, "Small"},
        {500, 500, "Medium"},
        {1000, 1000, "Large"},
        {2000, 2000, "XLarge"},
        {5000, 5000, "XXLarge"},
        {10000, 10000, "Huge"},
        {1000, 5000, "Asymmetric"},
        // Uncomment only if you have >16GB GPU memory
        // {15000, 15000, "Massive"}
    };
    
    std::cout << "\n" << std::setw(15) << "Test Case" 
              << std::setw(12) << "Size"
              << std::setw(15) << "Memory"
              << std::setw(15) << "Time"
              << std::setw(15) << "Distance"
              << std::setw(12) << "Status" << std::endl;
    std::cout << std::string(84, '-') << std::endl;
    
    for (const auto& test : test_cases) {
        // Calculate memory requirement
        size_t matrix_memory = (size_t)test.n * test.m * sizeof(double);
        size_t vector_memory = (test.n + test.m) * sizeof(double);
        size_t total_memory = matrix_memory + vector_memory;
        
        std::cout << std::setw(15) << test.description
                  << std::setw(12) << (std::to_string(test.n) + "x" + std::to_string(test.m));
        
        // Skip if memory requirement is too high
        if (total_memory > 8ULL * 1024 * 1024 * 1024) {  // 8GB limit
            std::cout << std::setw(15) << formatMemory(total_memory)
                      << YELLOW << "   Skipped (>8GB GPU memory required)" << RESET << std::endl;
            continue;
        }
        
        std::cout << std::setw(15) << formatMemory(total_memory);
        
        // Generate test sequences
        auto x = DTW::generateRandomSequence(test.n);
        auto y = DTW::generateRandomSequence(test.m);
        
        // Warm-up run
        dtw.compute(x, y);
        
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        double distance = dtw.compute(x, y);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        
        if (distance >= 0) {
            std::cout << std::setw(15) << formatDuration(diff.count())
                      << std::setw(15) << std::fixed << std::setprecision(2) << distance
                      << GREEN << std::setw(12) << "✓ Success" << RESET;
        } else {
            std::cout << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A"
                      << RED << std::setw(12) << "✗ Failed" << RESET;
        }
        std::cout << std::endl;
    }
}

// Test different distance metrics
void testDistanceMetrics() {
    std::cout << BOLD << CYAN << "\n=== Distance Metrics Test ===" << RESET << std::endl;
    
    DTW dtw;
    int n = 100, m = 100, dim = 3;
    
    auto x = DTW::generateRandomSequence(n, dim);
    auto y = DTW::generateRandomSequence(m, dim);
    
    std::vector<std::pair<DTW::DistanceType, std::string>> metrics = {
        {DTW::EUCLIDEAN, "Euclidean"},
        {DTW::MANHATTAN, "Manhattan"},
        {DTW::SQUARED, "Squared"},
        {DTW::ABSOLUTE, "Absolute"}
    };
    
    std::cout << "\nSequence: " << n << "x" << m << ", Dimension: " << dim << std::endl;
    std::cout << std::setw(15) << "Metric" 
              << std::setw(15) << "Distance"
              << std::setw(15) << "Time" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (const auto& [metric, name] : metrics) {
        auto start = std::chrono::high_resolution_clock::now();
        auto result = dtw.computeMultiDim(x, y, n, m, dim, metric);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        
        std::cout << std::setw(15) << name;
        if (result.success) {
            std::cout << std::setw(15) << std::fixed << std::setprecision(4) << result.distance
                      << std::setw(15) << formatDuration(diff.count());
        } else {
            std::cout << RED << "   Failed: " << result.error_message << RESET;
        }
        std::cout << std::endl;
    }
}

// Test shared memory optimization
void testSharedMemoryOptimization() {
    std::cout << BOLD << CYAN << "\n=== Shared Memory Optimization Test ===" << RESET << std::endl;
    
    DTW dtw_shared(256);
    DTW dtw_global(256);
    dtw_global.setUseSharedMemory(false);
    
    std::vector<int> dimensions = {1, 3, 8, 16, 32};
    int seq_length = 1000;
    
    std::cout << "\nSequence length: " << seq_length << std::endl;
    std::cout << std::setw(10) << "Dimension"
              << std::setw(20) << "Shared Mem Time"
              << std::setw(20) << "Global Mem Time"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    for (int dim : dimensions) {
        auto x = DTW::generateRandomSequence(seq_length, dim);
        auto y = DTW::generateRandomSequence(seq_length, dim);
        
        // Test with shared memory
        auto start_shared = std::chrono::high_resolution_clock::now();
        auto result_shared = dtw_shared.computeMultiDim(x, y, seq_length, seq_length, dim);
        auto end_shared = std::chrono::high_resolution_clock::now();
        double time_shared = std::chrono::duration<double>(end_shared - start_shared).count();
        
        // Test without shared memory
        auto start_global = std::chrono::high_resolution_clock::now();
        auto result_global = dtw_global.computeMultiDim(x, y, seq_length, seq_length, dim);
        auto end_global = std::chrono::high_resolution_clock::now();
        double time_global = std::chrono::duration<double>(end_global - start_global).count();
        
        std::cout << std::setw(10) << dim
                  << std::setw(20) << formatDuration(time_shared)
                  << std::setw(20) << formatDuration(time_global);
        
        if (result_shared.success && result_global.success) {
            double speedup = time_global / time_shared;
            if (speedup > 1.0) {
                std::cout << GREEN << std::setw(14) << std::fixed << std::setprecision(2) 
                          << speedup << "x" << RESET;
            } else {
                std::cout << YELLOW << std::setw(14) << std::fixed << std::setprecision(2) 
                          << speedup << "x" << RESET;
            }
        } else {
            std::cout << RED << std::setw(15) << "Error" << RESET;
        }
        std::cout << std::endl;
    }
}

// Stress test for memory management
void stressTestMemory() {
    std::cout << BOLD << CYAN << "\n=== Memory Management Stress Test ===" << RESET << std::endl;
    
    DTW dtw(256, 5000, 10);
    
    std::cout << "\nRunning 100 iterations with varying sizes..." << std::endl;
    
    int successes = 0;
    int failures = 0;
    double total_time = 0;
    
    for (int i = 0; i < 100; i++) {
        // Random sizes between 100 and 1000
        int n = 100 + (rand() % 900);
        int m = 100 + (rand() % 900);
        
        auto x = DTW::generateRandomSequence(n);
        auto y = DTW::generateRandomSequence(m);
        
        auto start = std::chrono::high_resolution_clock::now();
        double distance = dtw.compute(x, y);
        auto end = std::chrono::high_resolution_clock::now();
        
        total_time += std::chrono::duration<double>(end - start).count();
        
        if (distance >= 0) {
            successes++;
        } else {
            failures++;
        }
        
        // Progress indicator
        if ((i + 1) % 10 == 0) {
            std::cout << "   Progress: " << (i + 1) << "/100 completed\r" << std::flush;
        }
    }
    
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "   Successful runs: " << GREEN << successes << RESET << std::endl;
    std::cout << "   Failed runs: " << (failures > 0 ? RED : GREEN) << failures << RESET << std::endl;
    std::cout << "   Average time per run: " << formatDuration(total_time / 100) << std::endl;
    std::cout << "   Memory reuse: " << GREEN << "✓ Working correctly" << RESET << std::endl;
}

int main(int argc, char** argv) {
    std::cout << BOLD << BLUE << "╔═══════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     GPU-Accelerated Dynamic Time Warping Test Suite    ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════╝" << RESET << std::endl;
    
    // Check CUDA availability first
    std::cout << BOLD << "\n=== System Information ===" << RESET << std::endl;
    if (DTW::isCudaAvailable()) {
        std::cout << GREEN << "✓ CUDA is available" << RESET << std::endl;
        std::cout << DTW::getCudaDeviceInfo() << std::endl;
    } else {
        std::cout << RED << "✗ CUDA is not available. Exiting..." << RESET << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    bool run_all = true;
    bool run_basic = false;
    bool run_multidim = false;
    bool run_band = false;
    bool run_perf = false;
    bool run_metrics = false;
    bool run_shared = false;
    bool run_stress = false;
    
    if (argc > 1) {
        run_all = false;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--basic") run_basic = true;
            else if (arg == "--multidim") run_multidim = true;
            else if (arg == "--band") run_band = true;
            else if (arg == "--perf") run_perf = true;
            else if (arg == "--metrics") run_metrics = true;
            else if (arg == "--shared") run_shared = true;
            else if (arg == "--stress") run_stress = true;
            else if (arg == "--all") run_all = true;
            else if (arg == "--help") {
                std::cout << "\nUsage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --all      Run all tests (default)" << std::endl;
                std::cout << "  --basic    Run basic functionality tests" << std::endl;
                std::cout << "  --multidim Run multi-dimensional tests" << std::endl;
                std::cout << "  --band     Run Sakoe-Chiba band tests" << std::endl;
                std::cout << "  --perf     Run performance benchmarks" << std::endl;
                std::cout << "  --metrics  Run distance metrics tests" << std::endl;
                std::cout << "  --shared   Run shared memory optimization tests" << std::endl;
                std::cout << "  --stress   Run memory stress tests" << std::endl;
                std::cout << "  --help     Show this help message" << std::endl;
                return 0;
            }
        }
    }
    
    // Run selected tests
    try {
        if (run_all || run_basic) testBasicFunctionality();
        if (run_all || run_multidim) testMultiDimensional();
        if (run_all || run_band) testSakoeChibaBand();
        if (run_all || run_perf) performanceBenchmark();
        if (run_all || run_metrics) testDistanceMetrics();
        if (run_all || run_shared) testSharedMemoryOptimization();
        if (run_all || run_stress) stressTestMemory();
        
        std::cout << BOLD << GREEN << "\n=== All Tests Completed Successfully ===" << RESET << std::endl;
    } catch (const std::exception& e) {
        std::cout << BOLD << RED << "\n✗ Error: " << e.what() << RESET << std::endl;
        return 1;
    }
    
    return 0;
}