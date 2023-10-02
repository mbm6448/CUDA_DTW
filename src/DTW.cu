#include "DTW.h"
#include <cuda_runtime.h>

#define WARP_WIDTH 32

DTW::DTW(int blockSize) : BLOCK_SIZE(blockSize) {}

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    int m = y.size();
    double *d_device, *x_device, *y_device;
    std::vector<double> d(n * m);

    // Allocate memory on device
    cudaMalloc((void**)&d_device, n * m * sizeof(double));
    cudaMalloc((void**)&x_device, n * sizeof(double));
    cudaMalloc((void**)&y_device, m * sizeof(double));

    // Copy input data to device
    cudaMemcpy(x_device, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y.data(), m * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

    dtwKernel<<<grid, block>>>(d_device, x_device, y_device, n, m, BLOCK_SIZE);

    cudaMemcpy(d.data(), d_device, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_device);
    cudaFree(x_device);
    cudaFree(y_device);

    return d[n * m - 1];
}

std::vector<double> DTW::generateRandomSequence(int length) {
    std::vector<double> sequence(length);
    for(int i = 0; i < length; i++) {
        sequence[i] = rand() / (double)RAND_MAX;
    }
    return sequence;
}

__device__ double absoluteDifference(double x, double y) {
    return abs(x - y);
}

__global__ void dtwKernel(double *d, double *x, double *y, int n, int m, int blockSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Using shared memory for local block's data
    __shared__ double localX[WARP_WIDTH];
    __shared__ double localY[WARP_WIDTH];

    if (threadIdx.x < blockSize && i < n) {
        localX[threadIdx.x] = x[i];
    }
    if (threadIdx.y < blockSize && j < m) {
        localY[threadIdx.y] = y[j];
    }

    __syncthreads();

    if (i < n && j < m) {
        double cost = absoluteDifference(localX[threadIdx.x], localY[threadIdx.y]);

        double diagonal = (i > 0 && j > 0) ? d[(i - 1) * m + (j - 1)] : INFINITY;
        double left = (i > 0) ? d[(i - 1) * m + j] : INFINITY;
        double top = (j > 0) ? d[i * m + (j - 1)] : INFINITY;

        if (i == 0 && j == 0) {
            d[i * m + j] = cost;
        } else {
            d[i * m + j] = cost + min(diagonal, min(left, top));
        }
    }
}

