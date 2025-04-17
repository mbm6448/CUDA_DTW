#ifndef DTW_H
#define DTW_H

#include <vector>

/**
 * Dynamic Time Warping (DTW) implementation using CUDA
 * Computes the minimum distance between two time series
 */
class DTW {
public:
    /**
     * Constructor
     * @param blockSize The CUDA block size to use for computation
     */
    DTW(int blockSize = 16);
    
    /**
     * Compute the DTW distance between two sequences
     * @param x First sequence
     * @param y Second sequence
     * @return The DTW distance, or -1.0 if an error occurred
     */
    double compute(const std::vector<double>& x, const std::vector<double>& y);
    
    /**
     * Generate a random sequence for testing
     * @param length The length of the sequence
     * @return A vector containing random doubles between 0 and 1
     */
    std::vector<double> generateRandomSequence(int length);

private:
    int BLOCK_SIZE; // Block size for CUDA computation
};

#endif // DTW_H
