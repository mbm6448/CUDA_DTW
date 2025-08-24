#ifndef DTW_H
#define DTW_H

#include <vector>

/**
 * Dynamic Time Warping (DTW) implementation using CUDA
 * Computes the minimum distance between two time series
 * 
 * This implementation uses a wave-front parallel approach where
 * anti-diagonals of the DTW matrix are processed in parallel.
 */
class DTW {
public:
    /**
     * Constructor
     * @param blockSize The CUDA block size to use for computation (default: 256)
     *                  Valid range: 1-1024. Will default to 256 if invalid.
     */
    DTW(int blockSize = 256);
    
    /**
     * Compute the DTW distance between two sequences
     * Uses wave-front parallelization for efficient GPU computation
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
    static std::vector<double> generateRandomSequence(int length);

private:
    int BLOCK_SIZE; // Block size for CUDA computation
};

#endif // DTW_H