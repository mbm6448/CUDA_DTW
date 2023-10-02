#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printf("Device %d compute capability: %d.%d\n", i, devProp.major, devProp.minor);
    }
    return 0;
}

