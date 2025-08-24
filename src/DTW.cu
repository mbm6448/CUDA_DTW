#include "DTW.h"
#include <cuda_runtime.h>
#include <float.h>  // For DBL_MAX
#include <iostream>
#include <algorithm>

// Device function for computing absolute difference
__device__ double absoluteDifference(double x, double y) {
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

// Kernel for processing anti-diagonals in the DTW matrix
__global__ void dtwDiagonalKernel(double *d, const double *x, const double *y, 
                                  int n, int m, int diagonal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the range of valid (i,j) pairs for this diagonal
    // For diagonal k, we have i + j = k
    int start_i = max(0, diagonal - m + 1);
    int end_i = min(diagonal + 1, n);
    int num_elements = end_i - start_i;
    
    if (tid >= num_elements) return;
    
    int i = start_i + tid;
    int j = diagonal - i;
    
    if (i >= 0 && i < n && j >= 0 && j < m) {
        double cost = absoluteDifference(x[i], y[j]);
        
        if (i == 0 && j == 0) {
            // Base case: first cell
            d[0] = cost;
        } else {
            double min_prev = DBL_MAX;
            
            // Check three possible predecessors
            if (i > 0 && j > 0) {
                // Diagonal predecessor
                min_prev = fmin(min_prev, d[(i-1) * m + (j-1)]);
            }
            if (i > 0) {
                // Upper predecessor
                min_prev = fmin(min_prev, d[(i-1) * m + j]);
            }
            if (j > 0) {
                // Left predecessor
                min_prev = fmin(min_prev, d[i * m + (j-1)]);
            }
            
            d[i * m + j] = cost + min_prev;
        }
    }
}

// Alternative kernel using shared memory for better performance
__global__ void dtwDiagonalKernelShared(double *d, const double *x, const double *y, 
                                        int n, int m, int diagonal) {
    extern __shared__ double shared_mem[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int start_i = max(0, diagonal - m + 1);
    int end_i = min(diagonal + 1, n);
    int num_elements = end_i - start_i;
    
    if (tid >= num_elements) return;
    
    int i = start_i + tid;
    int j = diagonal - i;
    
    // Load relevant x and y values into shared memory if within block
    if (threadIdx.x < blockDim.x && i < n) {
        shared_mem[threadIdx.x] = x[i];
    }
    if (threadIdx.x < blockDim.x && j < m) {
        shared_mem[blockDim.x + threadIdx.x] = y[j];
    }
    __syncthreads();
    
    if (i >= 0 && i < n && j >= 0 && j < m) {
        double cost = absoluteDifference(
            (threadIdx.x < blockDim.x && i < n) ? shared_mem[threadIdx.x] : x[i],
            (threadIdx.x < blockDim.x && j < m) ? shared_mem[blockDim.x + threadIdx.x] : y[j]
        );
        
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

DTW::DTW(int blockSize) : BLOCK_SIZE(blockSize) {
    if (BLOCK_SIZE <= 0 || BLOCK_SIZE > 1024) {
        BLOCK_SIZE = 256;  // Default to reasonable value
    }
}

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    int m = y.size();
    
    if (n <= 0 || m <= 0) {
        std::cerr << "Error: Empty input sequences" << std::endl;
        return -1.0;
    }
    
    // For very large sequences, consider limiting size or using approximations
    if ((long long)n * m > 1e9) {
        std::cerr << "Warning: Very large matrix size (" << n << " x " << m << ")" << std::endl;
        std::cerr << "Consider using approximation methods for sequences this large" << std::endl;
    }
    
    double *d_device = nullptr, *x_device = nullptr, *y_device = nullptr;
    double result = -1.0;
    cudaError_t err;
    
    // Allocate memory on device
    size_t matrix_size = (size_t)n * m * sizeof(double);
    err = cudaMalloc((void**)&d_device, matrix_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for matrix: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&x_device, n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for x: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    err = cudaMalloc((void**)&y_device, m * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for y: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    // Copy input data to device
    err = cudaMemcpy(x_device, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for x: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    err = cudaMemcpy(y_device, y.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for y: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
    // Initialize DTW matrix with infinity
    {
        int total_elements = n * m;
        int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        initializeMatrix<<<num_blocks, BLOCK_SIZE>>>(d_device, n, m);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel failed (init): " << cudaGetErrorString(err) << std::endl;
            goto cleanup;
        }
    }
    
    // Process anti-diagonals
    for (int diagonal = 0; diagonal < n + m - 1; diagonal++) {
        int start_i = std::max(0, diagonal - m + 1);
        int end_i = std::min(diagonal + 1, n);
        int num_elements = end_i - start_i;
        
        if (num_elements > 0) {
            int num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // Use shared memory version for better performance
            size_t shared_size = 2 * BLOCK_SIZE * sizeof(double);
            dtwDiagonalKernelShared<<<num_blocks, BLOCK_SIZE, shared_size>>>(
                d_device, x_device, y_device, n, m, diagonal);
            
            // Synchronize after each diagonal to ensure dependencies are met
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(err) << std::endl;
                goto cleanup;
            }
            
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                std::cerr << "CUDA kernel failed at diagonal " << diagonal << ": " 
                          << cudaGetErrorString(err) << std::endl;
                goto cleanup;
            }
        }
    }
    
    // Copy result back
    err = cudaMemcpy(&result, &d_device[(n-1) * m + (m-1)], sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed for result: " << cudaGetErrorString(err) << std::endl;
        goto cleanup;
    }
    
cleanup:
    // Free device memory
    if (d_device) cudaFree(d_device);
    if (x_device) cudaFree(x_device);
    if (y_device) cudaFree(y_device);
    
    return result;
}

std::vector<double> DTW::generateRandomSequence(int length) {
    std::vector<double> sequence(length);
    for(int i = 0; i < length; i++) {
        sequence[i] = rand() / static_cast<double>(RAND_MAX);
    }
    return sequence;
}