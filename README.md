# GPU-Accelerated Dynamic Time Warping (CUDA)

This repo implements **Dynamic Time Warping** on the GPU using a *tiled
cooperative wave-front* kernel. It computes the DTW distance (and, optionally,
the alignment path) between two real-valued time series faster than a
single-threaded C++ CPU reference by **~12× at 4k×4k**, **~20× at 8k×8k**, and
**~24× at 12k×12k** on an NVIDIA RTX 5080 (CUDA 13.1).

## What's inside

- A **wave-front parallel** DTW kernel. The anti-diagonals of the DP matrix are
  the natural parallel dimension: cells on the same anti-diagonal are
  independent and can be computed simultaneously.
- **Anti-diagonal storage layout** so consecutive threads in a warp access
  contiguous memory — fully coalesced loads and stores.
- **Tiled execution**: the matrix is split into 32×32 tiles; tiles also flow
  in anti-diagonal waves. This reduces the number of grid-wide barriers from
  `n + m - 1` (one per cell-diagonal) to `(n + m) / T - 1` (one per
  tile-wave). At 8k×8k that's ~500 instead of ~16,000.
- **Single-launch cooperative kernel** (`cudaLaunchCooperativeKernel` +
  `cooperative_groups::this_grid().sync()`). The host launches **once**; the
  kernel walks all tile-waves internally.
- **Shared-memory tile** in every block: each block keeps its 32×32 working
  set in `__shared__` and only touches global memory at tile boundaries.
- **Sakoe–Chiba band** constraint, applied at both tile and cell granularity
  to skip work outside the band.
- **CPU reference** implementations (single-threaded and OpenMP wave-front)
  used both as a correctness oracle in the GoogleTest suite and as the
  benchmark baseline.

## Build

Requires CUDA Toolkit 13+ and a Hopper-or-newer GPU (sm_100 or sm_120 by
default). Override `GENCODE_FLAGS` to retarget.

```sh
make                # builds bin/dtw_test
make test           # builds bin/test_run and runs GoogleTest suite
make bench          # runs the benchmark driver
```

For an older GPU, e.g.:

```sh
make GENCODE_FLAGS="-gencode arch=compute_120,code=sm_120"
```

## Run

```sh
./bin/dtw_test --basic     # functional tests
./bin/dtw_test --perf      # GPU vs CPU benchmark
./bin/dtw_test --help
```

## Benchmark (RTX 5080, CUDA 13.1)

`./bin/dtw_test --perf` produces:

```
    size           GPU        CPU-1T       CPU-OMP         vs 1T        vs OMP
     256    358.789 us    309.966 us     41.786 ms         0.86x       116.46x
     512    727.746 us      1.418 ms      4.482 ms         1.95x         6.16x
      1k      1.445 ms      5.527 ms     16.094 ms         3.83x        11.14x
      2k      2.887 ms     21.134 ms     26.967 ms         7.32x         9.34x
      4k      6.525 ms     74.219 ms     71.395 ms        11.38x        10.94x
      8k     13.350 ms    267.354 ms    164.254 ms        20.03x        12.30x
     12k     20.809 ms          skip    491.258 ms             -        23.61x
     16k     28.568 ms          skip    507.943 ms             -        17.78x
```

* `CPU-1T` is a plain single-threaded triple-loop in `-O3` C++.
* `CPU-OMP` is wave-front parallel CPU using 32 OpenMP threads (also the
  correctness baseline, used by the test suite).

Speedup grows with sequence length because the GPU is compute-bound while the
CPU is memory-bandwidth-bound: doubling `n` quadruples CPU work but only
doubles GPU work (number of tile-waves).

## File structure

```
CUDA_DTW/
├── include/DTW.h        Public API and design notes
├── src/DTW.cu           Tiled wave-front kernel + host driver + CPU references
├── src/main.cu          Functional + performance driver
├── src/query.cu         Standalone CUDA device-info utility
├── test/TestDTW.cu      GoogleTest suite (24 tests)
├── Makefile             Build rules
└── README.md            This file
```

## API

```cpp
DTW dtw;                                     // default blockSize=256

// 1-D
double dist = dtw.compute(x, y);             // x, y are std::vector<double>
auto   res  = dtw.computeWithPath(x, y);     // returns DTWResult { distance, path }

// Multi-dimensional (n x dim, m x dim, row-major)
auto r = dtw.computeMultiDim(x, y, n, m, dim, DTW::EUCLIDEAN, /*window=*/-1);

// CPU references
double c = DTW::cpuDtwSerial(x, y, n, m, dim);
double c = DTW::cpuDtwOpenMP(x, y, n, m, dim);
```

## License

Apache 2.0. See `LICENSE`.
