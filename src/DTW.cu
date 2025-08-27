// DTW.cu - CUDA implementation file
#include "DTW.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " (" << cudaGetErrorString(error) << ")" << std::endl; \
            return false; \
        } \
    } while(0)

// Distance functions for device
__device__ inline double euclideanDistance(const double* x, const double* y, int dim) {
    double sum = 0.0;
    for (int d = 0; d < dim; d++) {
        double diff = x[d] - y[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__device__ inline double manhattanDistance(const double* x, const double* y, int dim) {
    double sum = 0.0;
    for (int d = 0; d < dim; d++) {
        sum += fabs(x[d] - y[d]);
    }
    return sum;
}

__device__ inline double squaredDistance(const double* x, const double* y, int dim) {
    double sum = 0.0;
    for (int d = 0; d < dim; d++) {
        double diff = x[d] - y[d];
        sum += diff * diff;
    }
    return sum;
}

__device__ inline double absoluteDifference(double x, double y) {
    return fabs(x - y);
}

// Kernel for initializing the DTW matrix
__global__ void initializeMatrix(double *d, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * m;
    
    if (idx < total) {
        d[idx] = DBL_MAX;
    }
}

// Fixed kernel for processing anti-diagonals with proper shared memory usage
__global__ void dtwDiagonalKernel(double *d, const double *x, const double *y, 
                                  int n, int m, int dim, int diagonal, int window) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the range of valid (i,j) pairs for this diagonal
    int start_i = max(0, diagonal - m + 1);
    int end_i = min(diagonal + 1, n);
    int num_elements = end_i - start_i;
    
    if (tid >= num_elements) return;
    
    int i = start_i + tid;
    int j = diagonal - i;
    
    // Apply Sakoe-Chiba band constraint if specified
    if (window > 0 && abs(i - j) > window) {
        return;  // Skip cells outside the band
    }
    
    if (i >= 0 && i < n && j >= 0 && j < m) {
        // Calculate distance based on dimensionality
        double cost;
        if (dim == 1) {
            cost = absoluteDifference(x[i], y[j]);
        } else {
            cost = euclideanDistance(&x[i * dim], &y[j * dim], dim);
        }
        
        if (i == 0 && j == 0) {
            d[0] = cost;
        } else {
            double min_prev = DBL_MAX;
            
            if (i > 0 && j > 0) {
                double diag = d[(i-1) * m + (j-1)];
                if (diag < min_prev) min_prev = diag;
            }
            if (i > 0) {
                double up = d[(i-1) * m + j];
                if (up < min_prev) min_prev = up;
            }
            if (j > 0) {
                double left = d[i * m + (j-1)];
                if (left < min_prev) min_prev = left;
            }
            
            d[i * m + j] = cost + min_prev;
        }
    }
}

// Optimized kernel with proper shared memory usage
__global__ void dtwDiagonalKernelShared(double *d, const double *x, const double *y, 
                                        int n, int m, int dim, int diagonal, int window) {
    extern __shared__ double shared_mem[];
    double* shared_x = shared_mem;
    double* shared_y = &shared_mem[blockDim.x * dim];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    int start_i = max(0, diagonal - m + 1);
    int end_i = min(diagonal + 1, n);
    int num_elements = end_i - start_i;
    
    if (tid >= num_elements) return;
    
    int i = start_i + tid;
    int j = diagonal - i;
    
    // Apply Sakoe-Chiba band constraint
    if (window > 0 && abs(i - j) > window) {
        return;
    }
    
    // Cooperative loading of sequence data into shared memory
    if (i < n && local_tid < blockDim.x) {
        for (int d = 0; d < dim; d++) {
            shared_x[local_tid * dim + d] = x[i * dim + d];
        }
    }
    
    if (j < m && local_tid < blockDim.x) {
        for (int d = 0; d < dim; d++) {
            shared_y[local_tid * dim + d] = y[j * dim + d];
        }
    }
    
    __syncthreads();
    
    if (i >= 0 && i < n && j >= 0 && j < m) {
        // Calculate cost using shared memory
        double cost;
        if (dim == 1) {
            cost = fabs(shared_x[local_tid] - shared_y[local_tid]);
        } else {
            cost = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = shared_x[local_tid * dim + d] - shared_y[local_tid * dim + d];
                cost += diff * diff;
            }
            cost = sqrt(cost);
        }
        
        if (i == 0 && j == 0) {
            d[0] = cost;
        } else {
            double min_prev = DBL_MAX;
            
            if (i > 0 && j > 0) {
                min_prev = fmin(min_prev, d[(i-1) * m + (j-1)]);
            }
            if (i > 0) {
                min_prev = fmin(min_prev, d[(i-1) * m + j]);
            }
            if (j > 0) {
                min_prev = fmin(min_prev, d[i * m + (j-1)]);
            }
            
            d[i * m + j] = cost + min_prev;
        }
    }
}

// Constructor
DTW::DTW(int blockSize, int maxLength, int maxDim) 
    : BLOCK_SIZE(blockSize), MAX_LENGTH(maxLength), MAX_DIM(maxDim),
      useSharedMemory(true), d_matrix(nullptr), d_x(nullptr), d_y(nullptr),
      allocated_matrix_size(0), allocated_x_size(0), allocated_y_size(0) {
    
    if (BLOCK_SIZE <= 0 || BLOCK_SIZE > 1024) {
        BLOCK_SIZE = 256;
    }
    
    // Check CUDA availability
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "Warning: No CUDA devices available. GPU acceleration disabled." << std::endl;
    }
}

// Destructor
DTW::~DTW() {
    cleanup();
}

void DTW::cleanup() {
    if (d_matrix) {
        cudaFree(d_matrix);
        d_matrix = nullptr;
    }
    if (d_x) {
        cudaFree(d_x);
        d_x = nullptr;
    }
    if (d_y) {
        cudaFree(d_y);
        d_y = nullptr;
    }
    allocated_matrix_size = 0;
    allocated_x_size = 0;
    allocated_y_size = 0;
}

bool DTW::ensureMemoryAllocated(int n, int m, int dim) {
    size_t required_matrix = (size_t)n * m * sizeof(double);
    size_t required_x = (size_t)n * dim * sizeof(double);
    size_t required_y = (size_t)m * dim * sizeof(double);
    
    cudaError_t err;
    
    // Reallocate matrix if needed
    if (required_matrix > allocated_matrix_size) {
        if (d_matrix) cudaFree(d_matrix);
        err = cudaMalloc((void**)&d_matrix, required_matrix);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate matrix memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        allocated_matrix_size = required_matrix;
    }
    
    // Reallocate x if needed
    if (required_x > allocated_x_size) {
        if (d_x) cudaFree(d_x);
        err = cudaMalloc((void**)&d_x, required_x);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate x memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        allocated_x_size = required_x;
    }
    
    // Reallocate y if needed
    if (required_y > allocated_y_size) {
        if (d_y) cudaFree(d_y);
        err = cudaMalloc((void**)&d_y, required_y);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate y memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        allocated_y_size = required_y;
    }
    
    return true;
}

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y, int window) {
    DTWResult result = computeMultiDim(x, y, x.size(), y.size(), 1, ABSOLUTE, window);
    return result.success ? result.distance : -1.0;
}

DTW::DTWResult DTW::computeMultiDim(const std::vector<double>& x, const std::vector<double>& y,
                                     int n, int m, int dim,
                                     DistanceType distType, int window) {
    DTWResult result;
    
    // Input validation
    if (n <= 0 || m <= 0 || dim <= 0) {
        result.error_message = "Invalid dimensions";
        return result;
    }
    
    if (x.size() != (size_t)(n * dim) || y.size() != (size_t)(m * dim)) {
        result.error_message = "Input size mismatch";
        return result;
    }
    
    // Check for very large matrices
    if ((long long)n * m > 1e9) {
        std::cerr << "Warning: Very large matrix (" << n << " x " << m << ")" << std::endl;
    }
    
    // Ensure sufficient memory is allocated
    if (!ensureMemoryAllocated(n, m, dim)) {
        result.error_message = "Memory allocation failed";
        return result;
    }
    
    cudaError_t err;
    
    // Copy input data to device
    err = cudaMemcpy(d_x, x.data(), n * dim * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        result.error_message = "Failed to copy x to device";
        return result;
    }
    
    err = cudaMemcpy(d_y, y.data(), m * dim * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        result.error_message = "Failed to copy y to device";
        return result;
    }
    
    // Initialize DTW matrix
    int total_elements = n * m;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    initializeMatrix<<<num_blocks, BLOCK_SIZE>>>(d_matrix, n, m);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        result.error_message = "Kernel launch failed (init)";
        return result;
    }
    
    // Process anti-diagonals
    int total_diagonals = n + m - 1;
    
    for (int diagonal = 0; diagonal < total_diagonals; diagonal++) {
        int start_i = std::max(0, diagonal - m + 1);
        int end_i = std::min(diagonal + 1, n);
        int num_elements = end_i - start_i;
        
        if (num_elements > 0) {
            int blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // Use shared memory optimization for small dimensions
            if (useSharedMemory && dim <= 16) {
                size_t shared_size = 2 * BLOCK_SIZE * dim * sizeof(double);
                dtwDiagonalKernelShared<<<blocks, BLOCK_SIZE, shared_size>>>(
                    d_matrix, d_x, d_y, n, m, dim, diagonal, window);
            } else {
                dtwDiagonalKernel<<<blocks, BLOCK_SIZE>>>(
                    d_matrix, d_x, d_y, n, m, dim, diagonal, window);
            }
            
            // Synchronize after each diagonal
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                result.error_message = "Synchronization failed";
                return result;
            }
        }
    }
    
    // Copy result back
    err = cudaMemcpy(&result.distance, &d_matrix[(n-1) * m + (m-1)], 
                     sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        result.error_message = "Failed to copy result";
        return result;
    }
    
    result.success = true;
    return result;
}

DTW::DTWResult DTW::computeWithPath(const std::vector<double>& x, const std::vector<double>& y, 
                                     int window) {
    int n = x.size();
    int m = y.size();
    
    // First compute the distance
    DTWResult result = computeMultiDim(x, y, n, m, 1, ABSOLUTE, window);
    
    if (!result.success) {
        return result;
    }
    
    // Extract the path
    return extractPath(n, m);
}

DTW::DTWResult DTW::extractPath(int n, int m) {
    DTWResult result;
    
    // Copy matrix to host
    std::vector<double> h_matrix(n * m);
    cudaError_t err = cudaMemcpy(h_matrix.data(), d_matrix, 
                                 n * m * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        result.error_message = "Failed to copy matrix for path extraction";
        return result;
    }
    
    // Backtrack to find path
    std::vector<std::pair<int, int>> path;
    int i = n - 1, j = m - 1;
    path.push_back({i, j});
    
    while (i > 0 || j > 0) {
        if (i == 0) {
            j--;
        } else if (j == 0) {
            i--;
        } else {
            double diag = h_matrix[(i-1) * m + (j-1)];
            double left = h_matrix[i * m + (j-1)];
            double up = h_matrix[(i-1) * m + j];
            
            if (diag <= left && diag <= up) {
                i--; j--;
            } else if (left <= up) {
                j--;
            } else {
                i--;
            }
        }
        path.push_back({i, j});
    }
    
    std::reverse(path.begin(), path.end());
    
    result.distance = h_matrix[(n-1) * m + (m-1)];
    result.path = path;
    result.success = true;
    
    return result;
}

bool DTW::getMatrix(std::vector<double>& result, int n, int m) {
    if (result.size() != (size_t)(n * m)) {
        std::cerr << "Output vector size mismatch" << std::endl;
        return false;
    }
    
    cudaError_t err = cudaMemcpy(result.data(), d_matrix, 
                                 n * m * sizeof(double), cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
}

double DTW::computeCustom(const std::vector<double>& x, const std::vector<double>& y,
                          std::function<double(double, double)> distFunc) {
    int n = x.size();
    int m = y.size();
    
    // CPU fallback for custom distance functions
    std::vector<double> matrix(n * m, std::numeric_limits<double>::infinity());
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double cost = distFunc(x[i], y[j]);
            
            if (i == 0 && j == 0) {
                matrix[0] = cost;
            } else if (i == 0) {
                matrix[j] = matrix[j-1] + cost;
            } else if (j == 0) {
                matrix[i * m] = matrix[(i-1) * m] + cost;
            } else {
                double diag = matrix[(i-1) * m + (j-1)];
                double left = matrix[i * m + (j-1)];
                double up = matrix[(i-1) * m + j];
                matrix[i * m + j] = std::min({diag, left, up}) + cost;
            }
        }
    }
    
    return matrix[(n-1) * m + (m-1)];
}

bool DTW::isCudaAvailable() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

std::string DTW::getCudaDeviceInfo() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        return "CUDA Error: " + std::string(cudaGetErrorString(error));
    }
    
    if (deviceCount == 0) {
        return "No CUDA devices found";
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::stringstream ss;
    ss << "Device: " << prop.name << "\n";
    ss << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    ss << "Total memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    ss << "Max threads per block: " << prop.maxThreadsPerBlock;
    
    return ss.str();
}

std::vector<double> DTW::generateRandomSequence(int length, int dim) {
    std::vector<double> sequence(length * dim);
    for(int i = 0; i < length * dim; i++) {
        sequence[i] = rand() / static_cast<double>(RAND_MAX);
    }
    return sequence;
}

// Example usage program (main.cpp)
#include "DTW.h"
#include <iostream>
#include <chrono>
#include <iomanip>

void testBasicDTW() {
    std::cout << "=== Basic DTW Test ===" << std::endl;
    
    DTW dtw(256);
    
    // Create test sequences
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.5, 2.5, 3.5, 4.5};
    
    double distance = dtw.compute(x, y);
    std::cout << "DTW distance: " << distance << std::endl;
    
    // Test with path extraction
    auto result = dtw.computeWithPath(x, y);
    if (result.success) {
        std::cout << "Path length: " << result.path.size() << std::endl;
        std::cout << "Alignment path (first 10):" << std::endl;
        for (size_t i = 0; i < std::min(result.path.size(), size_t(10)); i++) {
            std::cout << "  (" << result.path[i].first << ", " 
                      << result.path[i].second << ")" << std::endl;
        }
    }
}

void testMultiDimensional() {
    std::cout << "\n=== Multi-dimensional DTW Test ===" << std::endl;
    
    DTW dtw;
    
    int n = 100, m = 100, dim = 3;
    auto x = DTW::generateRandomSequence(n, dim);
    auto y = DTW::generateRandomSequence(m, dim);
    
    auto result = dtw.computeMultiDim(x, y, n, m, dim, DTW::EUCLIDEAN);
    
    if (result.success) {
        std::cout << "3D sequence DTW distance: " << result.distance << std::endl;
    } else {
        std::cout << "Error: " << result.error_message << std::endl;
    }
}

void testPerformance() {
    std::cout << "\n=== Performance Test ===" << std::endl;
    
    DTW dtw(256, 10000, 10);
    
    std::vector<int> sizes = {100, 500, 1000, 2000, 5000};
    
    for (int size : sizes) {
        auto x = DTW::generateRandomSequence(size);
        auto y = DTW::generateRandomSequence(size);
        
        auto start = std::chrono::high_resolution_clock::now();
        double distance = dtw.compute(x, y);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Size " << size << "x" << size << ": "
                  << std::fixed << std::setprecision(3) 
                  << duration.count() / 1000.0 << " ms";
        
        if (distance > 0) {
            std::cout << " (distance: " << distance << ")";
        }
        std::cout << std::endl;
    }
}

void testSakoeChibaBand() {
    std::cout << "\n=== Sakoe-Chiba Band Test ===" << std::endl;
    
    DTW dtw;
    
    auto x = DTW::generateRandomSequence(1000);
    auto y = DTW::generateRandomSequence(1000);
    
    // Test with different window sizes
    std::vector<int> windows = {-1, 100, 50, 20, 10};
    
    for (int window : windows) {
        auto start = std::chrono::high_resolution_clock::now();
        double distance = dtw.compute(x, y, window);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Window " << std::setw(3) << window << ": "
                  << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms"
                  << " (distance: " << distance << ")" << std::endl;
    }
}

int main() {
    // Check CUDA availability
    std::cout << "=== CUDA Information ===" << std::endl;
    if (DTW::isCudaAvailable()) {
        std::cout << DTW::getCudaDeviceInfo() << std::endl;
    } else {
        std::cout << "CUDA not available. Tests will fail." << std::endl;
        return 1;
    }
    
    // Run tests
    testBasicDTW();
    testMultiDimensional();
    testPerformance();
    testSakoeChibaBand();
    
    return 0;
}