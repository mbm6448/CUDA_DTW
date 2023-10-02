#pragma once

#include <vector>
#include <cuda.h>

class DTW {
public:
    DTW(int blockSize = 32);
    ~DTW() = default;

    double compute(const std::vector<double>& x, const std::vector<double>& y);
    static std::vector<double> generateRandomSequence(int length);

private:
    int BLOCK_SIZE;
};

__device__ double absoluteDifference(double x, double y);
__global__ void dtwKernel(double *d, double *x, double *y, int n, int m, int blockSize);

