// DTW.cu - GPU-accelerated DTW using wave-front parallelism with anti-diagonal
// storage layout, rolling 3-buffer scheme, and shared-memory tiling.
//
// See include/DTW.h for the design notes.

#include "DTW.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// ---------- Device helpers ----------

__device__ __forceinline__ double pointCost1D(const double* x, const double* y,
                                              int i, int j,
                                              int distType) {
    double d = x[i] - y[j];
    if (distType == DTW::SQUARED) return d * d;
    return fabs(d);  // EUCLIDEAN/MANHATTAN/ABSOLUTE all collapse to |.| in 1D
}

__device__ __forceinline__ double pointCostMD(const double* x, const double* y,
                                              int i, int j, int dim,
                                              int distType) {
    double sum = 0.0;
    const double* xi = x + (size_t)i * dim;
    const double* yj = y + (size_t)j * dim;
    if (distType == DTW::MANHATTAN || distType == DTW::ABSOLUTE) {
        for (int d = 0; d < dim; ++d) sum += fabs(xi[d] - yj[d]);
        return sum;
    }
    // EUCLIDEAN and SQUARED both start from sum of squares
    for (int d = 0; d < dim; ++d) {
        double diff = xi[d] - yj[d];
        sum += diff * diff;
    }
    return (distType == DTW::EUCLIDEAN) ? sqrt(sum) : sum;
}

// ---------- Tiled cooperative wave-front (single launch) ----------
//
// Matrix is divided into T*T tiles. Tiles also form an anti-diagonal pattern;
// tile (a, b) depends only on (a-1, b), (a, b-1), and (a-1, b-1). We process
// tiles in waves (a + b = w), one wave per outer-loop iteration, with a grid
// barrier between waves. This collapses the (n+m-1) per-cell anti-diagonals
// into ((n+m)/T - 1) tile-waves and gets us out of the launch/sync overhead
// regime entirely.
//
// Each block handles ONE tile per wave. Inside the tile, work runs as a
// micro wave-front in shared memory across 2T-1 internal anti-diagonals,
// with __syncthreads() (effectively warp-sync since T = warpSize) between.
//
// 1D / single-dim only — multi-dim uses the per-diagonal cooperative kernel.
//
// Storage: full n*m matrix in row-major. Predecessor reads for cells on the
// tile's top row or left column come from neighbouring tiles via global D.
template<int T>
__global__ void dtwTileCoopKernel(
    const double* __restrict__ x, const double* __restrict__ y,
    int n, int m, int distType, int window,
    double* __restrict__ D)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int nT = (n + T - 1) / T;
    int mT = (m + T - 1) / T;

    int tid = (int)threadIdx.x;
    int gtid    = (int)blockIdx.x * (int)blockDim.x + tid;
    long long gstride = (long long)gridDim.x * blockDim.x;

    // 1. Init D to DBL_MAX.
    long long total = (long long)n * m;
    for (long long t = gtid; t < total; t += gstride) D[t] = DBL_MAX;
    grid.sync();

    __shared__ double tile[T][T];
    __shared__ double sx[T];
    __shared__ double sy[T];

    int total_waves = nT + mT - 1;

    for (int w = 0; w < total_waves; ++w) {
        int a_lo = (w - mT + 1 > 0) ? (w - mT + 1) : 0;
        int a_hi = (w < nT - 1) ? w : (nT - 1);
        int wave_tiles = a_hi - a_lo + 1;
        int my_tile_in_wave = (int)blockIdx.x;
        bool has_tile = (my_tile_in_wave < wave_tiles);

        if (has_tile) {
            int a = a_lo + my_tile_in_wave;
            int b = w - a;
            int i_start = a * T;
            int j_start = b * T;
            int rc = (n - i_start < T) ? (n - i_start) : T;
            int cc = (m - j_start < T) ? (m - j_start) : T;

            // Whole-tile band cull. The tile occupies i in [i_start, i_start+rc-1]
            // and j in [j_start, j_start+cc-1]. min |i-j| in this rectangle is
            //   max(0, max(i_start - (j_start+cc-1), j_start - (i_start+rc-1)))
            // If that exceeds window, the entire tile is outside the band.
            bool tile_in_band = true;
            if (window >= 0) {
                int lo_ij = i_start - (j_start + cc - 1);
                int lo_ji = j_start - (i_start + rc - 1);
                int min_absdiff = 0;
                if (lo_ij > 0) min_absdiff = lo_ij;
                else if (lo_ji > 0) min_absdiff = lo_ji;
                if (min_absdiff > window) tile_in_band = false;
            }

            if (tile_in_band) {
                // Cooperative load of sequence segments.
                if (tid < rc) sx[tid] = x[i_start + tid];
                if (tid < cc) sy[tid] = y[j_start + tid];
                __syncthreads();

                // Internal wave-front: 2*max(rc,cc)-1 anti-diagonals.
                int inner_diags = rc + cc - 1;
                for (int dd = 0; dd < inner_diags; ++dd) {
                    int li_lo = (dd - cc + 1 > 0) ? (dd - cc + 1) : 0;
                    int li_hi = (dd < rc - 1) ? dd : (rc - 1);
                    int diag_len = li_hi - li_lo + 1;

                    if (tid < diag_len) {
                        int li = li_lo + tid;
                        int lj = dd - li;
                        int gi = i_start + li;
                        int gj = j_start + lj;

                        bool in_band = (window < 0) || (abs(gi - gj) <= window);
                        double result = DBL_MAX;
                        if (in_band) {
                            double cost = (distType == 3 /*SQUARED*/)
                                          ? (sx[li] - sy[lj]) * (sx[li] - sy[lj])
                                          : fabs(sx[li] - sy[lj]);

                            double up, left, dgv;
                            if (li > 0)              up = tile[li - 1][lj];
                            else if (gi > 0)         up = D[(size_t)(gi - 1) * m + gj];
                            else                     up = DBL_MAX;

                            if (lj > 0)              left = tile[li][lj - 1];
                            else if (gj > 0)         left = D[(size_t)gi * m + (gj - 1)];
                            else                     left = DBL_MAX;

                            if (li > 0 && lj > 0)        dgv = tile[li - 1][lj - 1];
                            else if (gi > 0 && gj > 0)   dgv = D[(size_t)(gi - 1) * m + (gj - 1)];
                            else                         dgv = DBL_MAX;

                            double base = fmin(up, fmin(left, dgv));
                            result = (gi == 0 && gj == 0) ? cost : cost + base;
                        }
                        tile[li][lj] = result;
                    }
                    __syncthreads();
                }

                // Store tile back to global D.
                for (int row = 0; row < rc; ++row) {
                    if (tid < cc) {
                        D[(size_t)(i_start + row) * m + (j_start + tid)] = tile[row][tid];
                    }
                }
            }
        }

        grid.sync();
    }
}

// Keep the per-diagonal cooperative kernel for multi-dim sequences and for
// pedagogical comparison. Same shape and contract as before.
__global__ void dtwCoopKernel(
    const double* __restrict__ x, const double* __restrict__ y,
    int n, int m, int dim, int distType, int window,
    double* __restrict__ d0, double* __restrict__ d1, double* __restrict__ d2,
    double* full_matrix /* nullptr in lean mode */)
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    extern __shared__ double smem[];
    // Layout: s_prev1[blockDim.x + 1], then s_prev2[blockDim.x].
    double* s_prev1 = smem;
    double* s_prev2 = smem + (blockDim.x + 1);

    int gtid   = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int gstride = (int)gridDim.x * (int)blockDim.x;
    int local  = threadIdx.x;

    // 1. Init diagonal buffers (and full matrix if present) to DBL_MAX.
    int diag_buf_len = (n < m ? n : m) + 2;
    for (int t = gtid; t < diag_buf_len; t += gstride) {
        d0[t] = DBL_MAX;
        d1[t] = DBL_MAX;
        d2[t] = DBL_MAX;
    }
    if (full_matrix) {
        long long total = (long long)n * m;
        for (long long t = gtid; t < total; t += gstride) {
            full_matrix[t] = DBL_MAX;
        }
    }
    grid.sync();

    // 2. Walk anti-diagonals. Each block handles a contiguous chunk of cells
    //    of the diagonal (block b handles active_lo + b*blockDim.x .. ).
    double* diags[3] = { d0, d1, d2 };
    int total_diags = n + m - 1;
    int prev_il1 = 0, prev_il2 = 0;

    for (int diag = 0; diag < total_diags; ++diag) {
        int il = (diag - m + 1 > 0) ? (diag - m + 1) : 0;
        int ih = (diag < n - 1) ? diag : (n - 1);

        int active_lo = il, active_hi = ih;
        if (window >= 0) {
            int lo_band = (int)ceil((diag - (double)window) / 2.0);
            int hi_band = (int)floor((diag + (double)window) / 2.0);
            if (active_lo < lo_band) active_lo = lo_band;
            if (active_hi > hi_band) active_hi = hi_band;
        }
        int active_count = active_hi - active_lo + 1;
        if (active_count < 0) active_count = 0;

        double* curr  = diags[diag % 3];
        double* prev1 = diags[((diag - 1) % 3 + 3) % 3];
        double* prev2 = diags[((diag - 2) % 3 + 3) % 3];

        int il0 = il, il1 = prev_il1, il2 = prev_il2;
        int block_first_i = active_lo + (int)blockIdx.x * (int)blockDim.x;

        // Only blocks whose first cell lies inside the active range do work.
        // ALL blocks must still reach grid.sync() below.
        bool block_has_work = (block_first_i <= active_hi);

        if (block_has_work) {
            // Cooperatively load the prev1 slab: indices
            //   [block_first_i - il1 - 1 .. block_first_i - il1 + blockDim.x - 1]
            int p1_base = block_first_i - il1 - 1;
            int idx = p1_base + local;
            s_prev1[local] = (idx >= 0) ? prev1[idx] : DBL_MAX;
            if (local == 0) {
                int idx_extra = p1_base + (int)blockDim.x;
                s_prev1[blockDim.x] = (idx_extra >= 0) ? prev1[idx_extra] : DBL_MAX;
            }
            // prev2 slab: [block_first_i - il2 - 1 .. block_first_i - il2 + blockDim.x - 2]
            int p2_base = block_first_i - il2 - 1;
            int idx2 = p2_base + local;
            s_prev2[local] = (idx2 >= 0) ? prev2[idx2] : DBL_MAX;
        }
        __syncthreads();

        if (block_has_work) {
            int i = block_first_i + local;
            if (i <= active_hi) {
                int j = diag - i;
                int k = i - il0;

                double cost = (dim == 1) ? pointCost1D(x, y, i, j, distType)
                                         : pointCostMD(x, y, i, j, dim, distType);

                double up   = (i > 0)          ? s_prev1[local]     : DBL_MAX;
                double left = (j > 0)          ? s_prev1[local + 1] : DBL_MAX;
                double dgv  = (i > 0 && j > 0) ? s_prev2[local]     : DBL_MAX;

                double base = fmin(up, fmin(left, dgv));
                double result = (i == 0 && j == 0) ? cost : cost + base;
                curr[k] = result;
                if (full_matrix) full_matrix[(size_t)i * m + j] = result;
            }
        }

        grid.sync();
        prev_il2 = prev_il1;
        prev_il1 = il;
    }
}

// ---------- Host helpers ----------

inline void warnCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (" << what << "): "
                  << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace

// ---------- DTW ctor/dtor/memory ----------

DTW::DTW(int blockSize, int maxLength, int maxDim)
    : BLOCK_SIZE(blockSize), MAX_LENGTH(maxLength), MAX_DIM(maxDim),
      useSharedMemory(true),
      d_x(nullptr), d_y(nullptr),
      allocated_x_size(0), allocated_y_size(0),
      d_full(nullptr), allocated_full_size(0), full_matrix_valid(false),
      graph_stream(nullptr), cached_graph(nullptr), cached_exec(nullptr),
      cache_valid(false),
      cache_n(0), cache_m(0), cache_dim(0), cache_distType(0), cache_window(0),
      cache_useShared(false), cache_withMatrix(false)
{
    if (BLOCK_SIZE <= 0 || BLOCK_SIZE > 1024) BLOCK_SIZE = 256;
    for (int i = 0; i < 3; ++i) d_diag[i] = nullptr;
    allocated_diag_size = 0;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "Warning: no CUDA device available. "
                     "DTW will not be able to run on the GPU." << std::endl;
        return;
    }
    cudaStreamCreate(&graph_stream);
}

DTW::~DTW() { cleanup(); }

void DTW::destroyCachedGraph() {
    if (cached_exec)  { cudaGraphExecDestroy(cached_exec); cached_exec = nullptr; }
    if (cached_graph) { cudaGraphDestroy(cached_graph);    cached_graph = nullptr; }
    cache_valid = false;
}

void DTW::cleanup() {
    destroyCachedGraph();
    if (graph_stream) { cudaStreamDestroy(graph_stream); graph_stream = nullptr; }
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_y) { cudaFree(d_y); d_y = nullptr; }
    for (int i = 0; i < 3; ++i) {
        if (d_diag[i]) { cudaFree(d_diag[i]); d_diag[i] = nullptr; }
    }
    if (d_full) { cudaFree(d_full); d_full = nullptr; }
    allocated_x_size = allocated_y_size = 0;
    allocated_diag_size = 0;
    allocated_full_size = 0;
    full_matrix_valid = false;
}

bool DTW::ensureSequenceMem(int n, int m, int dim) {
    size_t need_x = (size_t)n * dim * sizeof(double);
    size_t need_y = (size_t)m * dim * sizeof(double);
    bool reallocated = false;
    if (need_x > allocated_x_size) {
        if (d_x) cudaFree(d_x);
        cudaError_t e = cudaMalloc((void**)&d_x, need_x);
        if (e != cudaSuccess) { warnCuda(e, "alloc x"); return false; }
        allocated_x_size = need_x;
        reallocated = true;
    }
    if (need_y > allocated_y_size) {
        if (d_y) cudaFree(d_y);
        cudaError_t e = cudaMalloc((void**)&d_y, need_y);
        if (e != cudaSuccess) { warnCuda(e, "alloc y"); return false; }
        allocated_y_size = need_y;
        reallocated = true;
    }
    if (reallocated) destroyCachedGraph();
    return true;
}

bool DTW::ensureDiagMem(int n, int m) {
    size_t need = (size_t)(std::min(n, m) + 2) * sizeof(double);
    if (need <= allocated_diag_size) return true;
    for (int i = 0; i < 3; ++i) {
        if (d_diag[i]) cudaFree(d_diag[i]);
        cudaError_t e = cudaMalloc((void**)&d_diag[i], need);
        if (e != cudaSuccess) { warnCuda(e, "alloc diag"); return false; }
    }
    allocated_diag_size = need;
    destroyCachedGraph();
    return true;
}

bool DTW::ensureFullMem(int n, int m) {
    size_t need = (size_t)n * m * sizeof(double);
    if (need <= allocated_full_size) return true;
    if (d_full) cudaFree(d_full);
    cudaError_t e = cudaMalloc((void**)&d_full, need);
    if (e != cudaSuccess) { warnCuda(e, "alloc full"); return false; }
    allocated_full_size = need;
    destroyCachedGraph();
    return true;
}

// ---------- Public entry points ----------

double DTW::compute(const std::vector<double>& x, const std::vector<double>& y, int window) {
    DTWResult r = computeMultiDim(x, y, (int)x.size(), (int)y.size(), 1, ABSOLUTE, window);
    return r.success ? r.distance : -1.0;
}

DTW::DTWResult DTW::computeMultiDim(const std::vector<double>& x, const std::vector<double>& y,
                                    int n, int m, int dim,
                                    DistanceType distType, int window) {
    DTWResult result;
    if (n <= 0 || m <= 0 || dim <= 0) {
        result.error_message = "Invalid dimensions"; return result;
    }
    if (x.size() != (size_t)(n * dim) || y.size() != (size_t)(m * dim)) {
        result.error_message = "Input size mismatch"; return result;
    }
    return runLean(x, y, n, m, dim, distType, window);
}

DTW::DTWResult DTW::computeWithPath(const std::vector<double>& x, const std::vector<double>& y,
                                    int window) {
    int n = (int)x.size();
    int m = (int)y.size();
    if (n <= 0 || m <= 0) {
        DTWResult r; r.error_message = "Empty input"; return r;
    }
    DTWResult res = runWithMatrix(x, y, n, m, 1, ABSOLUTE, window);
    if (!res.success) return res;
    return extractPath(n, m);
}

// ---------- Wave-front driver ----------

namespace {

struct DiagGeom {
    int il;        // i_lo (unclipped)
    int ih;        // i_hi (unclipped)
    int len;       // ih - il + 1
    int active_lo; // band-clipped lo
    int active_hi; // band-clipped hi
    int active;    // active_hi - active_lo + 1 (>= 0)
};

DiagGeom diagGeom(int diag, int n, int m, int window) {
    DiagGeom g;
    g.il = std::max(0, diag - m + 1);
    g.ih = std::min(diag, n - 1);
    g.len = g.ih - g.il + 1;
    if (window < 0) {
        g.active_lo = g.il;
        g.active_hi = g.ih;
    } else {
        // |i - j| <= window, j = diag - i  =>  (diag - w)/2 <= i <= (diag + w)/2
        int lo_band = (diag - window + 1) / 2;  // ceil((diag - w)/2)
        if ((diag - window) < 0 && ((diag - window) % 2) != 0) {
            // floor toward -inf adjustment; safer formula below
        }
        // Use safer ceil/floor:
        lo_band = (int)std::ceil((diag - (double)window) / 2.0);
        int hi_band = (int)std::floor((diag + (double)window) / 2.0);
        g.active_lo = std::max(g.il, lo_band);
        g.active_hi = std::min(g.ih, hi_band);
    }
    g.active = g.active_hi - g.active_lo + 1;
    if (g.active < 0) g.active = 0;
    return g;
}

}  // namespace

namespace {

constexpr int TILE_T = 32;

// Launch the tile-cooperative kernel for 1D sequences. Writes results into
// the full n*m matrix `D`. Single launch, no per-diagonal overhead.
cudaError_t launchTileCoop(cudaStream_t stream,
                           int n, int m, int distType, int window,
                           const double* d_x, const double* d_y,
                           double* d_full) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    int max_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_per_sm, dtwTileCoopKernel<TILE_T>, TILE_T, 0);
    int max_resident = max_per_sm * p.multiProcessorCount;
    if (max_resident < 1) max_resident = 1;

    int nT = (n + TILE_T - 1) / TILE_T;
    int mT = (m + TILE_T - 1) / TILE_T;
    int max_wave_tiles = (nT < mT) ? nT : mT;  // largest tile-wave
    int grid = (max_wave_tiles < max_resident) ? max_wave_tiles : max_resident;
    if (grid < 1) grid = 1;

    const double* x_ptr = d_x;
    const double* y_ptr = d_y;
    double* d_ptr = d_full;
    int n_v = n, m_v = m, dist_v = distType, win_v = window;
    void* args[] = {
        (void*)&x_ptr, (void*)&y_ptr,
        (void*)&n_v, (void*)&m_v, (void*)&dist_v, (void*)&win_v,
        (void*)&d_ptr
    };
    dim3 gridDim(grid), blockDim(TILE_T);
    return cudaLaunchCooperativeKernel((void*)dtwTileCoopKernel<TILE_T>,
                                       gridDim, blockDim, args, 0, stream);
}

// Per-diagonal cooperative kernel launcher (used for multi-dim or as fallback).
cudaError_t launchCoopWavefront(cudaStream_t stream, int blockSize,
                                int n, int m, int dim, int distType, int window,
                                const double* d_x, const double* d_y,
                                double* d0, double* d1, double* d2,
                                double* full_matrix) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);

    size_t shmem = ((size_t)2 * blockSize + 1) * sizeof(double);
    int max_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_per_sm, dtwCoopKernel,
                                                  blockSize, shmem);
    int max_resident = max_per_sm * p.multiProcessorCount;
    if (max_resident < 1) max_resident = 1;

    int min_nm = (n < m) ? n : m;
    int needed = (min_nm + blockSize - 1) / blockSize;
    int grid = needed < max_resident ? needed : max_resident;
    if (grid < 1) grid = 1;

    const double* x_ptr = d_x;
    const double* y_ptr = d_y;
    double* p0 = d0; double* p1 = d1; double* p2 = d2;
    double* pf = full_matrix;
    int n_v = n, m_v = m, dim_v = dim, dist_v = distType, win_v = window;
    void* args[] = {
        (void*)&x_ptr, (void*)&y_ptr,
        (void*)&n_v, (void*)&m_v, (void*)&dim_v, (void*)&dist_v, (void*)&win_v,
        (void*)&p0,  (void*)&p1,  (void*)&p2,
        (void*)&pf
    };

    dim3 gridDim(grid), blockDim(blockSize);
    return cudaLaunchCooperativeKernel((void*)dtwCoopKernel, gridDim, blockDim,
                                       args, shmem, stream);
}

}  // namespace

// Common runner. Uses a single cooperative-kernel launch to execute the
// whole wave-front, eliminating the (n+m-1) per-diagonal launch overhead.
DTW::DTWResult DTW::runLean(const std::vector<double>& x, const std::vector<double>& y,
                            int n, int m, int dim, DistanceType distType, int window) {
    DTWResult result;
    full_matrix_valid = false;

    // Single-dim fast path: use the tiled cooperative kernel + full matrix.
    // For multi-dim we fall back to the per-diagonal cooperative kernel with
    // 3 rolling buffers (no full matrix).
    bool use_tile = (dim == 1);
    if (use_tile) {
        if (!ensureSequenceMem(n, m, dim) || !ensureFullMem(n, m)) {
            result.error_message = "GPU memory allocation failed"; return result;
        }
    } else {
        if (!ensureSequenceMem(n, m, dim) || !ensureDiagMem(n, m)) {
            result.error_message = "GPU memory allocation failed"; return result;
        }
    }

    cudaError_t e;
    e = cudaMemcpyAsync(d_x, x.data(), (size_t)n * dim * sizeof(double),
                        cudaMemcpyHostToDevice, graph_stream);
    if (e != cudaSuccess) { result.error_message = "H2D x"; return result; }
    e = cudaMemcpyAsync(d_y, y.data(), (size_t)m * dim * sizeof(double),
                        cudaMemcpyHostToDevice, graph_stream);
    if (e != cudaSuccess) { result.error_message = "H2D y"; return result; }

    if (use_tile) {
        e = launchTileCoop(graph_stream, n, m, (int)distType, window,
                           d_x, d_y, d_full);
        if (e != cudaSuccess) {
            result.error_message = std::string("tile launch: ") + cudaGetErrorString(e);
            return result;
        }
        // Read final cell straight from D.
        e = cudaMemcpyAsync(&result.distance,
                            d_full + (size_t)(n - 1) * m + (m - 1),
                            sizeof(double), cudaMemcpyDeviceToHost, graph_stream);
    } else {
        e = launchCoopWavefront(graph_stream, BLOCK_SIZE,
                                n, m, dim, (int)distType, window,
                                d_x, d_y, d_diag[0], d_diag[1], d_diag[2], nullptr);
        if (e != cudaSuccess) {
            result.error_message = std::string("coop launch: ") + cudaGetErrorString(e);
            return result;
        }
        int last_diag = n + m - 2;
        DiagGeom gl = diagGeom(last_diag, n, m, window);
        int result_k = (n - 1) - gl.il;
        double* last_buf = d_diag[last_diag % 3];
        e = cudaMemcpyAsync(&result.distance, last_buf + result_k,
                            sizeof(double), cudaMemcpyDeviceToHost, graph_stream);
    }
    if (e != cudaSuccess) { result.error_message = "D2H result"; return result; }

    e = cudaStreamSynchronize(graph_stream);
    if (e != cudaSuccess) {
        result.error_message = std::string("Stream sync: ") + cudaGetErrorString(e);
        return result;
    }
    if (window >= 0 && result.distance == DBL_MAX) {
        result.error_message = "Band too narrow: alignment infeasible";
        return result;
    }
    if (use_tile) full_matrix_valid = true;  // D contains the computed matrix
    result.success = true;
    return result;
}

DTW::DTWResult DTW::runWithMatrix(const std::vector<double>& x, const std::vector<double>& y,
                                  int n, int m, int dim, DistanceType distType, int window) {
    DTWResult result;
    bool use_tile = (dim == 1);
    if (use_tile) {
        if (!ensureSequenceMem(n, m, dim) || !ensureFullMem(n, m)) {
            result.error_message = "GPU memory allocation failed"; return result;
        }
    } else {
        if (!ensureSequenceMem(n, m, dim) || !ensureDiagMem(n, m) ||
            !ensureFullMem(n, m)) {
            result.error_message = "GPU memory allocation failed"; return result;
        }
    }
    full_matrix_valid = false;

    cudaError_t e;
    e = cudaMemcpyAsync(d_x, x.data(), (size_t)n * dim * sizeof(double),
                        cudaMemcpyHostToDevice, graph_stream);
    if (e != cudaSuccess) { result.error_message = "H2D x"; return result; }
    e = cudaMemcpyAsync(d_y, y.data(), (size_t)m * dim * sizeof(double),
                        cudaMemcpyHostToDevice, graph_stream);
    if (e != cudaSuccess) { result.error_message = "H2D y"; return result; }

    if (use_tile) {
        e = launchTileCoop(graph_stream, n, m, (int)distType, window,
                           d_x, d_y, d_full);
        if (e != cudaSuccess) {
            result.error_message = std::string("tile launch: ") + cudaGetErrorString(e);
            return result;
        }
        e = cudaMemcpyAsync(&result.distance,
                            d_full + (size_t)(n - 1) * m + (m - 1),
                            sizeof(double), cudaMemcpyDeviceToHost, graph_stream);
    } else {
        e = launchCoopWavefront(graph_stream, BLOCK_SIZE,
                                n, m, dim, (int)distType, window,
                                d_x, d_y, d_diag[0], d_diag[1], d_diag[2], d_full);
        if (e != cudaSuccess) {
            result.error_message = std::string("coop launch: ") + cudaGetErrorString(e);
            return result;
        }
        int last_diag = n + m - 2;
        DiagGeom gl = diagGeom(last_diag, n, m, window);
        int result_k = (n - 1) - gl.il;
        double* last_buf = d_diag[last_diag % 3];
        e = cudaMemcpyAsync(&result.distance, last_buf + result_k,
                            sizeof(double), cudaMemcpyDeviceToHost, graph_stream);
    }
    if (e != cudaSuccess) { result.error_message = "D2H result"; return result; }
    e = cudaStreamSynchronize(graph_stream);
    if (e != cudaSuccess) {
        result.error_message = std::string("Device sync: ") + cudaGetErrorString(e);
        return result;
    }

    if (window >= 0 && result.distance == DBL_MAX) {
        result.error_message = "Band too narrow: alignment infeasible";
        return result;
    }

    full_matrix_valid = true;
    result.success = true;
    return result;
}

DTW::DTWResult DTW::extractPath(int n, int m) {
    DTWResult result;
    if (!full_matrix_valid) {
        result.error_message = "Full matrix not available; call computeWithPath";
        return result;
    }

    std::vector<double> h(n * m);
    cudaError_t e = cudaMemcpy(h.data(), d_full, (size_t)n * m * sizeof(double),
                               cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { result.error_message = "D2H full matrix"; return result; }

    std::vector<std::pair<int, int>> path;
    int i = n - 1, j = m - 1;
    path.emplace_back(i, j);
    while (i > 0 || j > 0) {
        if (i == 0)      { --j; }
        else if (j == 0) { --i; }
        else {
            double diag = h[(size_t)(i - 1) * m + (j - 1)];
            double up   = h[(size_t)(i - 1) * m +  j     ];
            double left = h[(size_t)  i      * m + (j - 1)];
            if (diag <= up && diag <= left)      { --i; --j; }
            else if (left <= up)                 { --j; }
            else                                 { --i; }
        }
        path.emplace_back(i, j);
    }
    std::reverse(path.begin(), path.end());

    result.distance = h[(size_t)(n - 1) * m + (m - 1)];
    result.path = std::move(path);
    result.success = true;
    return result;
}

bool DTW::getMatrix(std::vector<double>& result, int n, int m) {
    if (!full_matrix_valid) {
        std::cerr << "getMatrix: no full matrix is currently resident on the GPU. "
                     "Call computeWithPath first." << std::endl;
        return false;
    }
    if (result.size() != (size_t)n * m) {
        std::cerr << "getMatrix: output buffer size mismatch" << std::endl;
        return false;
    }
    cudaError_t e = cudaMemcpy(result.data(), d_full,
                               (size_t)n * m * sizeof(double), cudaMemcpyDeviceToHost);
    return e == cudaSuccess;
}

double DTW::computeCustom(const std::vector<double>& x, const std::vector<double>& y,
                          std::function<double(double, double)> distFunc) {
    int n = (int)x.size();
    int m = (int)y.size();
    if (n <= 0 || m <= 0) return -1.0;

    std::vector<double> d((size_t)n * m, std::numeric_limits<double>::infinity());
    d[0] = distFunc(x[0], y[0]);
    for (int j = 1; j < m; ++j) d[j] = d[j - 1] + distFunc(x[0], y[j]);
    for (int i = 1; i < n; ++i) d[(size_t)i * m] = d[(size_t)(i - 1) * m] + distFunc(x[i], y[0]);
    for (int i = 1; i < n; ++i) {
        for (int j = 1; j < m; ++j) {
            double c = distFunc(x[i], y[j]);
            double dg = d[(size_t)(i - 1) * m + (j - 1)];
            double up = d[(size_t)(i - 1) * m +  j     ];
            double lf = d[(size_t)  i      * m + (j - 1)];
            d[(size_t)i * m + j] = c + std::min(dg, std::min(up, lf));
        }
    }
    return d[(size_t)(n - 1) * m + (m - 1)];
}

bool DTW::isCudaAvailable() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaSuccess && n > 0;
}

std::string DTW::getCudaDeviceInfo() {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) return std::string("CUDA error: ") + cudaGetErrorString(e);
    if (n == 0) return "No CUDA devices found";
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::stringstream ss;
    ss << "Device: " << p.name << "\n";
    ss << "Compute capability: " << p.major << "." << p.minor << "\n";
    ss << "Total memory: " << (p.totalGlobalMem / (1024 * 1024)) << " MB\n";
    ss << "Multiprocessors: " << p.multiProcessorCount << "\n";
    ss << "Max threads/block: " << p.maxThreadsPerBlock;
    return ss.str();
}

std::vector<double> DTW::generateRandomSequence(int length, int dim) {
    std::vector<double> s((size_t)length * dim);
    for (size_t i = 0; i < s.size(); ++i) {
        s[i] = rand() / static_cast<double>(RAND_MAX);
    }
    return s;
}

// ---------- CPU reference implementations ----------

namespace {

inline double cpuPointCost(const double* xi, const double* yj, int dim,
                           DTW::DistanceType distType) {
    if (dim == 1) {
        double d = xi[0] - yj[0];
        if (distType == DTW::SQUARED) return d * d;
        return std::fabs(d);
    }
    double sum = 0.0;
    if (distType == DTW::MANHATTAN || distType == DTW::ABSOLUTE) {
        for (int d = 0; d < dim; ++d) sum += std::fabs(xi[d] - yj[d]);
        return sum;
    }
    for (int d = 0; d < dim; ++d) {
        double diff = xi[d] - yj[d];
        sum += diff * diff;
    }
    return (distType == DTW::EUCLIDEAN) ? std::sqrt(sum) : sum;
}

}  // namespace

double DTW::cpuDtwSerial(const std::vector<double>& x, const std::vector<double>& y,
                         int n, int m, int dim,
                         DistanceType distType, int window) {
    if (n <= 0 || m <= 0 || dim <= 0) return -1.0;
    if (x.size() != (size_t)(n * dim) || y.size() != (size_t)(m * dim)) return -1.0;

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> d((size_t)n * m, INF);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (window >= 0 && std::abs(i - j) > window) continue;
            double c = cpuPointCost(&x[(size_t)i * dim], &y[(size_t)j * dim], dim, distType);
            if (i == 0 && j == 0)      d[0] = c;
            else if (i == 0)           d[j] = d[j - 1] + c;
            else if (j == 0)           d[(size_t)i * m] = d[(size_t)(i - 1) * m] + c;
            else {
                double dg = d[(size_t)(i - 1) * m + (j - 1)];
                double up = d[(size_t)(i - 1) * m +  j     ];
                double lf = d[(size_t)  i      * m + (j - 1)];
                d[(size_t)i * m + j] = c + std::min(dg, std::min(up, lf));
            }
        }
    }
    double r = d[(size_t)(n - 1) * m + (m - 1)];
    return std::isinf(r) ? -1.0 : r;
}

double DTW::cpuDtwOpenMP(const std::vector<double>& x, const std::vector<double>& y,
                         int n, int m, int dim,
                         DistanceType distType, int window) {
    if (n <= 0 || m <= 0 || dim <= 0) return -1.0;
    if (x.size() != (size_t)(n * dim) || y.size() != (size_t)(m * dim)) return -1.0;

    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> d((size_t)n * m, INF);

    int total_diags = n + m - 1;
    for (int diag = 0; diag < total_diags; ++diag) {
        int il = std::max(0, diag - m + 1);
        int ih = std::min(diag, n - 1);
        int active_lo = il, active_hi = ih;
        if (window >= 0) {
            int lo_band = (int)std::ceil((diag - (double)window) / 2.0);
            int hi_band = (int)std::floor((diag + (double)window) / 2.0);
            active_lo = std::max(active_lo, lo_band);
            active_hi = std::min(active_hi, hi_band);
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = active_lo; i <= active_hi; ++i) {
            int j = diag - i;
            double c = cpuPointCost(&x[(size_t)i * dim], &y[(size_t)j * dim], dim, distType);
            if (i == 0 && j == 0) {
                d[0] = c;
            } else {
                double up = (i > 0)            ? d[(size_t)(i - 1) * m +  j     ] : INF;
                double lf = (j > 0)            ? d[(size_t)  i      * m + (j - 1)] : INF;
                double dg = (i > 0 && j > 0)   ? d[(size_t)(i - 1) * m + (j - 1)] : INF;
                d[(size_t)i * m + j] = c + std::min(dg, std::min(up, lf));
            }
        }
    }
    double r = d[(size_t)(n - 1) * m + (m - 1)];
    return std::isinf(r) ? -1.0 : r;
}
