#include "DTW.h"
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

DTW::DTW(int blockSize) : BLOCK_SIZE(blockSize) {}

std::vector<double> DTW::generateRandomSequence(int length) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    std::vector<double> sequence(length);
    for (int i = 0; i < length; i++) {
        sequence[i] = dis(gen);
    }

    return sequence;
}

__device__ double DTW::absoluteDifference(double x, double y) {
    return fabs(x - y);
}

__global__ void DTW::dtwKernel(double *d, double *x, double *y, int n, int m) {
    extern __shared__ double shared[];
    double *s_x = shared;
    double *s_y = &shared[blockDim.x];

    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid_x < n && tid_y < m) {
        s_x[threadIdx.x] = x[tid_x];
        s_y[threadIdx.y] = y[tid_y];

        __syncthreads();

        double dtw[BLOCK_SIZE][BLOCK_SIZE] = {{0.0}};

        dtw[0][0] = absoluteDifference(s_x[0], s_y[0]);

        for (int i = 1; i < BLOCK_SIZE && tid_x + i < n; i++) {
            dtw[i][0] = dtw[i - 1][0] + absoluteDifference(s_x[i], s_y[0]);
        }

        for (int i = 1; i < BLOCK_SIZE && tid_y + i < m; i++) {
            dtw[0][i] = dtw[0][i - 1] + absoluteDifference(s_x[0], s_y[i]);
        }

        for (int i = 1; i < BLOCK_SIZE && tid_x + i < n; i++) {
            for (int j = 1; j < BLOCK_SIZE && tid_y + j < m; j++) {
                dtw[i][j] = absoluteDifference(s_x[i], s_y[j]) +
                            fmin(dtw[i - 1][j],
                                 fmin(dtw[i][j - 1], dtw[i - 1][j - 1]));
            }
        }

        d[tid_x * m + tid_y] = dtw[BLOCK_SIZE - 1][BLOCK_SIZE - 1];
    }
}

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    int m = y.size();

    double *d_d, *d_x, *d_y;

    cudaError_t err = cudaMalloc((void **)&d_d, n * m * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector d (error code " << cudaGetErrorString(err) << ")!\n";
        cudaFree(d_d);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_x, n * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector x (error code " << cudaGetErrorString(err) << ")!\n";
        cudaFree(d_d);
        cudaFree(d_x);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_y, m * sizeof(double));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device vector y (error code " << cudaGetErrorString(err) << ")!\n";
        cudaFree(d_d);
        cudaFree(d_x);
        cudaFree(d_y);
        exit(EXIT_FAILURE);
    }

    cudaMemcpyAsync(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_y, y.data(), m * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x, (m + dimBlock.y - 1) / dimBlock.y);

    dtwKernel<<<dimGrid, dimBlock, 2 * BLOCK_SIZE * sizeof(double)>>>(d_d, d_x, d_y, n, m);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    std::vector<double> h_d(n * m);
    cudaMemcpyAsync(h_d.data(), d_d, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    double dtw = h_d[n * m - 1];

    cudaFree(d_d);
    cudaFree(d_x);
    cudaFree(d_y);

    return dtw;
}

