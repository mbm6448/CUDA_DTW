#include "DTW.h"
#include <cuda_runtime.h>
#include <float.h>  // For DBL_MAX
#include <iostream>

// Forward declarations
__device__ double absoluteDifference(double x, double y);
__global__ void dtwKernel(double *d, const double *x, const double *y, int n, int m);

DTW::DTW(int blockSize) : BLOCK_SIZE(blockSize) {}

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    int m = y.size();
    
    if (n <= 0 || m <= 0) {
        std::cerr << "Error: Empty input sequences" << std::endl;
        return -1.0;
    }
    
    double *d_device, *x_device, *y_device;
    double result;
    
    // Allocate memory on device
    cudaError_t err;
    err = cudaMalloc((void**)&d_device, n * m * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    err = cudaMalloc((void**)&x_device, n * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_device);
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    err = cudaMalloc((void**)&y_device, m * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_device);
        cudaFree(x_device);
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    // Copy input data to device
    err = cudaMemcpy(x_device, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_device);
        cudaFree(x_device);
        cudaFree(y_device);
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    err = cudaMemcpy(y_device, y.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_device);
        cudaFree(x_device);
        cudaFree(y_device);
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    // Initialize DTW matrix with infinity
    err = cudaMemset(d_device, 0xFF, n * m * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_device);
        cudaFree(x_device);
        cudaFree(y_device);
        std::cerr << "CUDA memset failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    // Launch kernel for each anti-diagonal
    dim3 block(BLOCK_SIZE);
    
    // We need to process the matrix in anti-diagonal order to respect dependencies
    for (int k = 0; k < n + m - 1; k++) {
        int start_i = std::max(0, k - m + 1);
        int count = std::min(k + 1, std::min(n, m + n - k - 1)) - start_i;
        
        if (count > 0) {
            dim3 grid((count + block.x - 1) / block.x);
            dtwKernel<<<grid, block>>>(d_device, x_device, y_device, n, m);
            
            // Check for kernel errors
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                cudaFree(d_device);
                cudaFree(x_device);
                cudaFree(y_device);
                std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
                return -1.0;
            }
        }
    }
    
    // Copy result back
    err = cudaMemcpy(&result, &d_device[(n-1) * m + (m-1)], sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_device);
        cudaFree(x_device);
        cudaFree(y_device);
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        return -1.0;
    }
    
    // Free device memory
    cudaFree(d_device);
    cudaFree(x_device);
    cudaFree(y_device);
    
    return result;
}

std::vector<double> DTW::generateRandomSequence(int length) {
    std::vector<double> sequence(length);
    for(int i = 0; i < length; i++) {
        sequence[i] = rand() / static_cast<double>(RAND_MAX);
    }
    return sequence;
}

__device__ double absoluteDifference(double x, double y) {
    return fabs(x - y);  // Use fabs for floating point absolute value
}

__global__ void dtwKernel(double *d, const double *x, const double *y, int n, int m) {
    // Calculate anti-diagonal wave
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for better access pattern
    __shared__ double localX[32];  // Assuming block size <= 32
    __shared__ double localY[32];
    
    // Load shared memory
    if (threadIdx.x < blockDim.x) {
        if (k < n) localX[threadIdx.x] = x[k];
        if (k < m) localY[threadIdx.x] = y[k];
    }
    __syncthreads();
    
    // Process cells in current anti-diagonal
    for (int i = 0; i <= k; i++) {
        int j = k - i;
        
        if (i < n && j < m) {
            double cost = absoluteDifference(x[i], y[j]);
            
            if (i == 0 && j == 0) {
                d[0] = cost;  // First cell
            } else {
                // Get values from neighbors with boundary checking
                double diagonal = (i > 0 && j > 0) ? d[(i-1) * m + (j-1)] : DBL_MAX;
                double left = (j > 0) ? d[i * m + (j-1)] : DBL_MAX;
                double up = (i > 0) ? d[(i-1) * m + j] : DBL_MAX;
                
                // Calculate current cell value
                d[i * m + j] = cost + fmin(diagonal, fmin(left, up));
            }
        }
    }
}
