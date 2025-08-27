// DTW.h - Header file for GPU-accelerated Dynamic Time Warping
#ifndef DTW_H
#define DTW_H

#include <vector>
#include <functional>
#include <limits>

/**
 * Dynamic Time Warping (DTW) implementation using CUDA
 * Computes the minimum distance between two time series
 * 
 * Features:
 * - Wave-front parallel approach for GPU acceleration
 * - Multi-dimensional sequence support
 * - Path extraction capability
 * - Sakoe-Chiba band constraint support
 * - Multiple distance metrics
 * - Robust error handling
 */
class DTW {
public:
    // Distance metric types
    enum DistanceType {
        EUCLIDEAN,
        MANHATTAN,
        ABSOLUTE,
        SQUARED
    };
    
    // Structure to hold DTW results
    struct DTWResult {
        double distance;
        std::vector<std::pair<int, int>> path;
        bool success;
        std::string error_message;
        
        DTWResult() : distance(-1.0), success(false) {}
    };
    
    /**
     * Constructor
     * @param blockSize The CUDA block size to use for computation (default: 256)
     * @param maxLength Maximum sequence length to pre-allocate (default: 10000)
     * @param maxDim Maximum dimensions per point (default: 128)
     */
    DTW(int blockSize = 256, int maxLength = 10000, int maxDim = 128);
    
    /**
     * Destructor - cleans up GPU memory
     */
    ~DTW();
    
    /**
     * Compute the DTW distance between two 1D sequences
     * @param x First sequence
     * @param y Second sequence
     * @param window Sakoe-Chiba band width (-1 for no constraint)
     * @return The DTW distance, or -1.0 if an error occurred
     */
    double compute(const std::vector<double>& x, const std::vector<double>& y, int window = -1);
    
    /**
     * Compute DTW distance between two multi-dimensional sequences
     * @param x First sequence (n x dim)
     * @param y Second sequence (m x dim)
     * @param n Length of first sequence
     * @param m Length of second sequence
     * @param dim Dimensions per point
     * @param distType Distance metric to use
     * @param window Sakoe-Chiba band width (-1 for no constraint)
     * @return DTWResult containing distance and optional path
     */
    DTWResult computeMultiDim(const std::vector<double>& x, const std::vector<double>& y,
                              int n, int m, int dim,
                              DistanceType distType = EUCLIDEAN,
                              int window = -1);
    
    /**
     * Compute DTW with path extraction
     * @param x First sequence
     * @param y Second sequence
     * @param window Sakoe-Chiba band width (-1 for no constraint)
     * @return DTWResult containing distance and alignment path
     */
    DTWResult computeWithPath(const std::vector<double>& x, const std::vector<double>& y, 
                              int window = -1);
    
    /**
     * Compute DTW with custom distance function (CPU fallback)
     * @param x First sequence
     * @param y Second sequence
     * @param distFunc Custom distance function
     * @return The DTW distance
     */
    double computeCustom(const std::vector<double>& x, const std::vector<double>& y,
                         std::function<double(double, double)> distFunc);
    
    /**
     * Get the full DTW matrix (for debugging/visualization)
     * @param result Output matrix (must be pre-allocated to n*m)
     * @param n Number of rows
     * @param m Number of columns
     * @return true if successful
     */
    bool getMatrix(std::vector<double>& result, int n, int m);
    
    /**
     * Set whether to use shared memory optimization
     * @param use true to enable shared memory optimization
     */
    void setUseSharedMemory(bool use) { useSharedMemory = use; }
    
    /**
     * Check if CUDA is available
     * @return true if CUDA device is available
     */
    static bool isCudaAvailable();
    
    /**
     * Get CUDA device properties
     * @return Device name or error message
     */
    static std::string getCudaDeviceInfo();
    
    /**
     * Generate a random sequence for testing
     * @param length The length of the sequence
     * @param dim Number of dimensions (default: 1)
     * @return A vector containing random doubles between 0 and 1
     */
    static std::vector<double> generateRandomSequence(int length, int dim = 1);

private:
    int BLOCK_SIZE;
    int MAX_LENGTH;
    int MAX_DIM;
    bool useSharedMemory;
    
    // Device pointers
    double *d_matrix;
    double *d_x;
    double *d_y;
    
    // Pre-allocated sizes
    size_t allocated_matrix_size;
    size_t allocated_x_size;
    size_t allocated_y_size;
    
    // Helper methods
    bool ensureMemoryAllocated(int n, int m, int dim);
    void cleanup();
    DTWResult extractPath(int n, int m);
};

#endif // DTW_H