// DTW.h - GPU-accelerated Dynamic Time Warping
#ifndef DTW_H
#define DTW_H

#include <vector>
#include <functional>
#include <string>
#include <utility>

#include <cuda_runtime.h>

/*
 * Dynamic Time Warping on CUDA.
 *
 * Implementation notes (so the reviewer doesn't have to guess):
 *   - Wave-front parallel: each anti-diagonal of the DP matrix is computed
 *     in parallel, in order. Within a diagonal, all cells are independent.
 *   - Anti-diagonal storage layout: cells of one diagonal are stored
 *     contiguously, so a warp of threads accesses contiguous memory ->
 *     fully coalesced global loads and stores.
 *   - Rolling 3-buffer scheme: only the previous two diagonals are needed
 *     to compute the next one, so device memory is O(min(n,m)) when the
 *     alignment path is not requested.
 *   - Shared memory holds the relevant slabs of the previous two diagonals
 *     once per block; every neighbour read by every cell in the block is
 *     then a shared-memory hit. (This is the actual shared-memory
 *     optimization, not a memcpy with no reuse.)
 *   - Sakoe-Chiba band: the host clips each diagonal's [k_lo, k_hi] to the
 *     band so kernels only launch threads that do real work.
 *
 * Path extraction is the slow path; it allocates the full n*m matrix.
 */
class DTW {
public:
    enum DistanceType {
        EUCLIDEAN,
        MANHATTAN,
        ABSOLUTE,
        SQUARED
    };

    struct DTWResult {
        double distance;
        std::vector<std::pair<int, int>> path;
        bool success;
        std::string error_message;

        DTWResult() : distance(-1.0), success(false) {}
    };

    DTW(int blockSize = 256, int maxLength = 10000, int maxDim = 128);
    ~DTW();

    // 1D convenience entry point. window < 0 means no constraint.
    double compute(const std::vector<double>& x, const std::vector<double>& y, int window = -1);

    // Multi-dim entry point. x has n*dim doubles, y has m*dim doubles, row-major.
    DTWResult computeMultiDim(const std::vector<double>& x, const std::vector<double>& y,
                              int n, int m, int dim,
                              DistanceType distType = EUCLIDEAN,
                              int window = -1);

    // Computes distance and the alignment path (slow path: allocates full n*m matrix).
    DTWResult computeWithPath(const std::vector<double>& x, const std::vector<double>& y,
                              int window = -1);

    // CPU fallback for arbitrary distance functions.
    double computeCustom(const std::vector<double>& x, const std::vector<double>& y,
                         std::function<double(double, double)> distFunc);

    // Copies the full DTW cost matrix from device to host.
    // Only valid after computeWithPath() (the lean kernels don't keep it).
    bool getMatrix(std::vector<double>& result, int n, int m);

    // Toggle the shared-memory kernel. Default on. Off is mainly for benchmarking
    // the contribution of the shared-memory tile vs the plain coalesced kernel.
    void setUseSharedMemory(bool use) { useSharedMemory = use; }

    static bool isCudaAvailable();
    static std::string getCudaDeviceInfo();

    // Generate length*dim random doubles in [0,1) for testing.
    static std::vector<double> generateRandomSequence(int length, int dim = 1);

    // CPU reference implementations.
    // Single-threaded plain DP. Used as the correctness oracle and as
    // the "GPU vs 1 CPU thread" benchmark denominator.
    static double cpuDtwSerial(const std::vector<double>& x,
                               const std::vector<double>& y,
                               int n, int m, int dim,
                               DistanceType distType = ABSOLUTE,
                               int window = -1);

    // Wave-front parallel CPU using OpenMP. Used as the
    // "GPU vs all CPU cores" benchmark denominator.
    static double cpuDtwOpenMP(const std::vector<double>& x,
                               const std::vector<double>& y,
                               int n, int m, int dim,
                               DistanceType distType = ABSOLUTE,
                               int window = -1);

private:
    int BLOCK_SIZE;
    int MAX_LENGTH;
    int MAX_DIM;
    bool useSharedMemory;

    // Sequence buffers (device).
    double* d_x;
    double* d_y;
    size_t allocated_x_size;
    size_t allocated_y_size;

    // Three rolling diagonal buffers (device). Length = min(n,m)+2.
    double* d_diag[3];
    size_t allocated_diag_size;

    // Full n*m matrix, allocated only when path extraction is requested.
    double* d_full;
    size_t allocated_full_size;
    bool full_matrix_valid;

    // CUDA Graph cache. The wave-front loop launches n+m-1 small kernels.
    // We capture them into a graph on the first call with a given (n,m,dim,
    // distType,window,useShared,withMatrix) tuple and replay it on every
    // subsequent call. This collapses ~16k host-side launches at 8k size
    // into one cudaGraphLaunch.
    cudaStream_t graph_stream;
    cudaGraph_t  cached_graph;
    cudaGraphExec_t cached_exec;
    bool   cache_valid;
    int    cache_n, cache_m, cache_dim, cache_distType, cache_window;
    bool   cache_useShared, cache_withMatrix;

    bool ensureSequenceMem(int n, int m, int dim);
    bool ensureDiagMem(int n, int m);
    bool ensureFullMem(int n, int m);
    void destroyCachedGraph();
    void cleanup();

    DTWResult runLean(const std::vector<double>& x, const std::vector<double>& y,
                      int n, int m, int dim, DistanceType distType, int window);
    DTWResult runWithMatrix(const std::vector<double>& x, const std::vector<double>& y,
                            int n, int m, int dim, DistanceType distType, int window);
    DTWResult extractPath(int n, int m);
};

#endif // DTW_H
