#include <iostream>
#include <chrono>
#include <iomanip>
#include "DTW.h"

int main() {
    // Test with different sizes
    std::vector<std::pair<int, int>> test_sizes = {
        {100, 100},
        {500, 500},
        {1000, 1000},
        {5000, 5000},
        {10000, 10000},
        // Note: 30000x50000 would require ~12GB of GPU memory just for the matrix
        // Only uncomment if you have sufficient GPU memory
        // {30000, 50000}
    };
    
    std::cout << "Dynamic Time Warping CUDA Implementation Test\n";
    std::cout << "=============================================\n\n";
    
    for (const auto& size_pair : test_sizes) {
        int n = size_pair.first;
        int m = size_pair.second;
        
        std::cout << "Testing with sequence sizes: " << n << " x " << m << std::endl;
        
        // Calculate memory requirement
        size_t memory_needed = (size_t)n * m * sizeof(double) + 
                              (n + m) * sizeof(double);
        double memory_gb = memory_needed / (1024.0 * 1024.0 * 1024.0);
        std::cout << "Estimated GPU memory needed: " << std::fixed << std::setprecision(2) 
                  << memory_gb << " GB" << std::endl;
        
        if (memory_gb > 8.0) {
            std::cout << "Warning: This may exceed typical GPU memory. Skipping...\n\n";
            continue;
        }
        
        DTW dtw(256);  // Use block size of 256
        
        // Generate random sequences
        std::cout << "Generating random sequences..." << std::endl;
        std::vector<double> x = DTW::generateRandomSequence(n);
        std::vector<double> y = DTW::generateRandomSequence(m);
        
        // Compute DTW distance with timing
        std::cout << "Computing DTW distance..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        double dtw_distance = dtw.compute(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        
        if (dtw_distance >= 0) {
            std::cout << "DTW distance: " << dtw_distance << std::endl;
            std::cout << "Computation time: " << diff.count() << " seconds" << std::endl;
        } else {
            std::cout << "Error computing DTW distance" << std::endl;
        }
        
        std::cout << "----------------------------------------\n\n";
    }
    
    // Test with identical sequences (should give 0 distance)
    std::cout << "Testing with identical sequences (100 elements):" << std::endl;
    DTW dtw;
    std::vector<double> test_seq = {0.1, 0.2, 0.3, 0.4, 0.5};
    for (int i = 0; i < 95; i++) {
        test_seq.push_back(rand() / static_cast<double>(RAND_MAX));
    }
    
    double identical_distance = dtw.compute(test_seq, test_seq);
    std::cout << "DTW distance for identical sequences: " << identical_distance;
    std::cout << " (should be 0.0)" << std::endl;
    
    return 0;
}