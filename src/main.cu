// main.cu - Functional + performance driver for the CUDA DTW implementation.
//
// Usage:
//   ./bin/dtw_test [--basic|--multidim|--band|--perf|--metrics|--shared|--stress|--all|--help]
//
// Performance benchmark prints, for each size:
//   GPU ms | CPU-1T ms | CPU-OMP ms | speedup vs 1T | speedup vs OMP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "DTW.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define RESET   "\033[0m"
#define BOLD    "\033[1m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define CYAN    "\033[36m"

static std::string fmtTime(double seconds) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(3);
    if (seconds < 1e-3)      s << (seconds * 1e6) << " us";
    else if (seconds < 1.0)  s << (seconds * 1e3) << " ms";
    else                     s << seconds << " s";
    return s.str();
}

static double sec(std::chrono::high_resolution_clock::time_point a,
                  std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// ---------- Tests ----------

static void testBasicFunctionality() {
    std::cout << BOLD << CYAN << "\n=== Basic Functionality ===" << RESET << "\n";
    DTW dtw(256);

    std::cout << "\n1. Simple sequences:\n";
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {1.5, 2.5, 3.5, 4.5};
    double gpu = dtw.compute(x, y);
    double cpu = DTW::cpuDtwSerial(x, y, (int)x.size(), (int)y.size(), 1);
    std::cout << "   GPU = " << gpu << "   CPU = " << cpu
              << "   diff = " << std::abs(gpu - cpu) << "\n";

    std::cout << "\n2. Identical sequences (expect 0):\n";
    auto t = DTW::generateRandomSequence(200);
    double d_id = dtw.compute(t, t);
    std::cout << "   DTW(t,t) = " << d_id
              << (std::abs(d_id) < 1e-10 ? GREEN " OK" RESET : RED " FAIL" RESET) << "\n";

    std::cout << "\n3. Known small case (x=[1,2,3], y=[2,3,4]) expect 2.0:\n";
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {2.0, 3.0, 4.0};
    double dab = dtw.compute(a, b);
    std::cout << "   GPU = " << dab
              << (std::abs(dab - 2.0) < 1e-9 ? GREEN " OK" RESET : RED " FAIL" RESET) << "\n";

    std::cout << "\n4. Path extraction:\n";
    auto pr = dtw.computeWithPath(x, y);
    if (pr.success) {
        std::cout << "   path length = " << pr.path.size() << "\n";
        for (size_t i = 0; i < std::min<size_t>(pr.path.size(), 5); ++i) {
            std::cout << "     (" << pr.path[i].first << ", " << pr.path[i].second << ")\n";
        }
        std::cout << GREEN << "   OK" << RESET << "\n";
    } else {
        std::cout << RED << "   FAIL: " << pr.error_message << RESET << "\n";
    }
}

static void testMultiDimensional() {
    std::cout << BOLD << CYAN << "\n=== Multi-dimensional ===" << RESET << "\n";
    DTW dtw;
    std::vector<int> dims = {1, 2, 3, 5, 10, 20};
    int N = 500;
    std::cout << std::setw(6) << "dim"
              << std::setw(16) << "GPU dist"
              << std::setw(16) << "CPU dist"
              << std::setw(14) << "GPU time" << "\n";
    std::cout << std::string(52, '-') << "\n";
    for (int dim : dims) {
        auto x = DTW::generateRandomSequence(N, dim);
        auto y = DTW::generateRandomSequence(N, dim);
        auto t0 = std::chrono::high_resolution_clock::now();
        auto r = dtw.computeMultiDim(x, y, N, N, dim, DTW::EUCLIDEAN);
        auto t1 = std::chrono::high_resolution_clock::now();
        double c = DTW::cpuDtwSerial(x, y, N, N, dim, DTW::EUCLIDEAN);
        std::cout << std::setw(6) << dim
                  << std::setw(16) << std::fixed << std::setprecision(4) << r.distance
                  << std::setw(16) << c
                  << std::setw(14) << fmtTime(sec(t0, t1)) << "\n";
    }
}

static void testSakoeChibaBand() {
    std::cout << BOLD << CYAN << "\n=== Sakoe-Chiba Band ===" << RESET << "\n";
    DTW dtw;
    int n = 2000, m = 2000;
    auto x = DTW::generateRandomSequence(n);
    auto y = DTW::generateRandomSequence(m);
    std::vector<int> ws = {-1, 200, 100, 50, 20};
    std::cout << std::setw(10) << "window"
              << std::setw(16) << "distance"
              << std::setw(14) << "time"
              << std::setw(14) << "speedup" << "\n";
    double base = 0.0;
    for (int w : ws) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double d = dtw.compute(x, y, w);
        auto t1 = std::chrono::high_resolution_clock::now();
        double t = sec(t0, t1);
        if (w < 0) { base = t; std::cout << std::setw(10) << "none"; }
        else        { std::cout << std::setw(10) << w; }
        std::cout << std::setw(16) << std::fixed << std::setprecision(4) << d
                  << std::setw(14) << fmtTime(t);
        if (w < 0) std::cout << std::setw(14) << "baseline";
        else       std::cout << std::setw(13) << std::setprecision(2) << (base / t) << "x";
        std::cout << "\n";
    }
}

static void performanceBenchmark() {
    std::cout << BOLD << CYAN << "\n=== GPU vs CPU Benchmark ===" << RESET << "\n";
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "(OpenMP disabled at compile time -- CPU-OMP column = serial)\n";
#endif

    DTW dtw(256, 12000, 10);

    struct Case { int n, m; const char* label; };
    std::vector<Case> cases = {
        {  256,   256, "256"},
        {  512,   512, "512"},
        { 1000,  1000, "1k"},
        { 2000,  2000, "2k"},
        { 4000,  4000, "4k"},
        { 8000,  8000, "8k"},
        {12000, 12000, "12k"},
        {16000, 16000, "16k"},
    };

    std::cout << std::setw(8)  << "size"
              << std::setw(14) << "GPU"
              << std::setw(14) << "CPU-1T"
              << std::setw(14) << "CPU-OMP"
              << std::setw(14) << "vs 1T"
              << std::setw(14) << "vs OMP"
              << std::setw(16) << "distance" << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (const auto& c : cases) {
        auto x = DTW::generateRandomSequence(c.n);
        auto y = DTW::generateRandomSequence(c.m);

        // GPU: one warm-up, then timed.
        dtw.compute(x, y);
        cudaDeviceSynchronize();
        auto g0 = std::chrono::high_resolution_clock::now();
        double dgpu = dtw.compute(x, y);
        cudaDeviceSynchronize();
        auto g1 = std::chrono::high_resolution_clock::now();
        double tg = sec(g0, g1);

        double tc1 = 0.0, dcpu1 = 0.0, tco = 0.0;

        // Skip the slow 1T CPU for the largest cases so the bench actually finishes.
        // 4k uses ~70ms, 8k would use ~280ms, 16k ~1.1s -- still tolerable, but
        // we cap at 8k to keep total bench time reasonable.
        bool run_cpu_1t = (c.n <= 8000);
        if (run_cpu_1t) {
            auto c0 = std::chrono::high_resolution_clock::now();
            dcpu1 = DTW::cpuDtwSerial(x, y, c.n, c.m, 1);
            auto c1 = std::chrono::high_resolution_clock::now();
            tc1 = sec(c0, c1);
        }
        {
            auto c0 = std::chrono::high_resolution_clock::now();
            DTW::cpuDtwOpenMP(x, y, c.n, c.m, 1);
            auto c1 = std::chrono::high_resolution_clock::now();
            tco = sec(c0, c1);
        }

        std::cout << std::setw(8) << c.label
                  << std::setw(14) << fmtTime(tg)
                  << std::setw(14) << (run_cpu_1t ? fmtTime(tc1) : std::string("skip"))
                  << std::setw(14) << fmtTime(tco);
        if (run_cpu_1t) {
            std::ostringstream s; s.setf(std::ios::fixed); s.precision(2); s << (tc1 / tg) << "x";
            std::cout << std::setw(14) << s.str();
        } else {
            std::cout << std::setw(14) << "-";
        }
        {
            std::ostringstream s; s.setf(std::ios::fixed); s.precision(2); s << (tco / tg) << "x";
            std::cout << std::setw(14) << s.str();
        }
        std::cout << std::setw(16) << std::fixed << std::setprecision(2) << dgpu;
        if (run_cpu_1t && std::abs(dgpu - dcpu1) > 1e-6 * std::max(1.0, std::abs(dcpu1))) {
            std::cout << RED << "  MISMATCH" << RESET;
        }
        std::cout << "\n";
    }
}

static void testDistanceMetrics() {
    std::cout << BOLD << CYAN << "\n=== Distance Metrics ===" << RESET << "\n";
    DTW dtw;
    int n = 200, m = 200, dim = 3;
    auto x = DTW::generateRandomSequence(n, dim);
    auto y = DTW::generateRandomSequence(m, dim);

    struct M { DTW::DistanceType t; const char* name; };
    std::vector<M> metrics = {
        {DTW::EUCLIDEAN, "Euclidean"},
        {DTW::MANHATTAN, "Manhattan"},
        {DTW::SQUARED,   "Squared"},
        {DTW::ABSOLUTE,  "Absolute"},
    };
    std::cout << std::setw(12) << "metric"
              << std::setw(16) << "GPU dist"
              << std::setw(16) << "CPU dist" << "\n";
    for (auto& mm : metrics) {
        auto r = dtw.computeMultiDim(x, y, n, m, dim, mm.t);
        double c = DTW::cpuDtwSerial(x, y, n, m, dim, mm.t);
        std::cout << std::setw(12) << mm.name
                  << std::setw(16) << std::fixed << std::setprecision(4) << r.distance
                  << std::setw(16) << c << "\n";
    }
}

static void testSharedVsPlain() {
    std::cout << BOLD << CYAN << "\n=== Kernel: shared-memory vs plain ===" << RESET << "\n";
    DTW dtw_shared(256);
    DTW dtw_plain(256);  dtw_plain.setUseSharedMemory(false);
    int N = 4000;
    auto x = DTW::generateRandomSequence(N);
    auto y = DTW::generateRandomSequence(N);

    // Warm-up
    dtw_shared.compute(x, y);
    dtw_plain.compute(x, y);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    double ds = dtw_shared.compute(x, y);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double dp = dtw_plain.compute(x, y);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    double ts = sec(t0, t1), tp = sec(t1, t2);

    std::cout << "  size " << N << "x" << N << "\n";
    std::cout << "  shared = " << fmtTime(ts) << "  plain = " << fmtTime(tp)
              << "  shared/plain = " << std::fixed << std::setprecision(2) << (tp / ts) << "x\n";
    std::cout << "  result match: "
              << (std::abs(ds - dp) < 1e-6 * std::max(1.0, std::abs(dp)) ? GREEN "OK" RESET
                                                                          : RED "FAIL" RESET)
              << "\n";
}

static void stressTestMemory() {
    std::cout << BOLD << CYAN << "\n=== Memory reuse stress ===" << RESET << "\n";
    DTW dtw(256, 2000, 4);
    int ok = 0;
    for (int it = 0; it < 50; ++it) {
        int n = 100 + (rand() % 900);
        int m = 100 + (rand() % 900);
        auto x = DTW::generateRandomSequence(n);
        auto y = DTW::generateRandomSequence(m);
        double d = dtw.compute(x, y);
        if (d >= 0) ++ok;
    }
    std::cout << "  passes: " << ok << " / 50\n";
}

int main(int argc, char** argv) {
    std::cout << BOLD << BLUE
              << "============================================================\n"
              << "      GPU-Accelerated Dynamic Time Warping  (CUDA, C++)     \n"
              << "============================================================" << RESET << "\n";

    if (!DTW::isCudaAvailable()) {
        std::cout << RED << "No CUDA device. Exiting." << RESET << "\n";
        return 1;
    }
    std::cout << DTW::getCudaDeviceInfo() << "\n";

    bool run_all = (argc == 1);
    bool basic = false, multi = false, band = false, perf = false,
         metrics = false, shared = false, stress = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--basic")    basic = true;
        else if (a == "--multidim") multi = true;
        else if (a == "--band")     band  = true;
        else if (a == "--perf")     perf  = true;
        else if (a == "--metrics")  metrics = true;
        else if (a == "--shared")   shared  = true;
        else if (a == "--stress")   stress  = true;
        else if (a == "--all")      run_all = true;
        else if (a == "--help") {
            std::cout << "\nUsage: " << argv[0]
                      << " [--all|--basic|--multidim|--band|--perf|--metrics|--shared|--stress]\n";
            return 0;
        }
    }

    if (run_all || basic)   testBasicFunctionality();
    if (run_all || multi)   testMultiDimensional();
    if (run_all || band)    testSakoeChibaBand();
    if (run_all || perf)    performanceBenchmark();
    if (run_all || metrics) testDistanceMetrics();
    if (run_all || shared)  testSharedVsPlain();
    if (run_all || stress)  stressTestMemory();

    std::cout << BOLD << GREEN << "\n=== Done ===" << RESET << "\n";
    return 0;
}
