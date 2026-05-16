// TestDTW.cu - GoogleTest suite for the CUDA DTW implementation.
// The CPU reference (DTW::cpuDtwSerial) is the correctness oracle for
// every randomized GPU result.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "DTW.h"

class DTWTestFixture : public ::testing::Test {
protected:
    void SetUp() override { dtw = new DTW(256); }
    void TearDown() override { delete dtw; }
    DTW* dtw = nullptr;
};

TEST_F(DTWTestFixture, NonNegativeDistance) {
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> y = {0.3, 0.2, 0.1};
    EXPECT_GE(dtw->compute(x, y), 0.0);
}

TEST_F(DTWTestFixture, ZeroDistanceForIdenticalSequences) {
    std::vector<double> x = {0.1, 0.2, 0.3, 0.4, 0.5};
    EXPECT_NEAR(dtw->compute(x, x), 0.0, 1e-9);
}

TEST_F(DTWTestFixture, SymmetryProperty) {
    std::vector<double> x = {0.1, 0.4, 0.5, 0.9};
    std::vector<double> y = {0.2, 0.3, 0.6};
    EXPECT_NEAR(dtw->compute(x, y), dtw->compute(y, x), 1e-9);
}

TEST_F(DTWTestFixture, SingleElementSequences) {
    std::vector<double> x = {0.5};
    std::vector<double> y = {0.3};
    EXPECT_NEAR(dtw->compute(x, y), 0.2, 1e-9);
}

TEST_F(DTWTestFixture, EmptySequencesReturnError) {
    std::vector<double> x;
    std::vector<double> y = {0.1, 0.2};
    EXPECT_EQ(dtw->compute(x, y), -1.0);
    EXPECT_EQ(dtw->compute(y, x), -1.0);
}

// Constant sequences: DTW path must touch all n*m cells of one row, then
// step through all of the other dimension. For x = ten 0.5s, z = twenty
// 0.7s, every aligned pair costs 0.2 and the path length is exactly
// max(n,m)=20. So DTW = 20 * 0.2 = 4.0  (NOT 2.0 as the old test claimed).
TEST_F(DTWTestFixture, ConstantSequences) {
    std::vector<double> x(10, 0.5);
    std::vector<double> y(15, 0.5);
    EXPECT_NEAR(dtw->compute(x, y), 0.0, 1e-9);

    std::vector<double> z(20, 0.7);
    EXPECT_NEAR(dtw->compute(x, z), 4.0, 1e-9);
}

// Hand-computed DTW of {1,2,3} vs {2,3,4} with absolute cost:
//   pointwise costs        cumulative D
//   1 2 3                   1 3 6
//   0 1 2          ->       1 2 4
//   1 0 1                   2 1 2
// D[2,2] = 2  (NOT 3 as the old test claimed).
TEST_F(DTWTestFixture, KnownDTWResult) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {2.0, 3.0, 4.0};
    EXPECT_NEAR(dtw->compute(x, y), 2.0, 1e-9);
}

TEST_F(DTWTestFixture, MonotonicityTest) {
    std::vector<double> base  = {0.0, 0.1, 0.2, 0.3, 0.4};
    std::vector<double> close = {0.01, 0.11, 0.21, 0.31, 0.41};
    std::vector<double> far   = {0.5, 0.6, 0.7, 0.8, 0.9};
    EXPECT_LT(dtw->compute(base, close), dtw->compute(base, far));
}

TEST_F(DTWTestFixture, LargerSequences) {
    auto x = DTW::generateRandomSequence(1000);
    auto y = DTW::generateRandomSequence(1500);
    double d = dtw->compute(x, y);
    EXPECT_GE(d, 0.0);
    EXPECT_NEAR(dtw->compute(x, x), 0.0, 1e-9);
}

// Path extraction should produce a monotone non-decreasing path from
// (0,0) to (n-1,m-1) with steps in {right, down, diagonal}.
TEST_F(DTWTestFixture, PathExtractionWellFormed) {
    auto x = DTW::generateRandomSequence(50);
    auto y = DTW::generateRandomSequence(70);
    auto r = dtw->computeWithPath(x, y);
    ASSERT_TRUE(r.success) << r.error_message;
    ASSERT_FALSE(r.path.empty());
    EXPECT_EQ(r.path.front().first,  0);
    EXPECT_EQ(r.path.front().second, 0);
    EXPECT_EQ(r.path.back().first,  (int)x.size() - 1);
    EXPECT_EQ(r.path.back().second, (int)y.size() - 1);
    for (size_t k = 1; k < r.path.size(); ++k) {
        int di = r.path[k].first  - r.path[k - 1].first;
        int dj = r.path[k].second - r.path[k - 1].second;
        EXPECT_TRUE((di == 0 && dj == 1) || (di == 1 && dj == 0) || (di == 1 && dj == 1))
            << "Illegal step at index " << k << ": (" << di << "," << dj << ")";
    }
    EXPECT_NEAR(r.distance, dtw->compute(x, y), 1e-9);
}

class DTWParameterizedTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    DTW dtw{256};
};

TEST_P(DTWParameterizedTest, DifferentSequenceLengths) {
    int n = std::get<0>(GetParam());
    int m = std::get<1>(GetParam());
    std::vector<double> x(n, 0.5), y(m, 0.5);
    EXPECT_NEAR(dtw.compute(x, y), 0.0, 1e-9);
}

INSTANTIATE_TEST_SUITE_P(DifferentLengths, DTWParameterizedTest,
    ::testing::Values(
        std::make_tuple(10, 20),
        std::make_tuple(50, 100),
        std::make_tuple(100, 200),
        std::make_tuple(500, 500)));

// The real correctness oracle: random GPU vs single-threaded CPU.
class DTWMatchesCpu : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    DTW dtw{256};
};

TEST_P(DTWMatchesCpu, GpuEqualsCpuReference) {
    int n = std::get<0>(GetParam());
    int m = std::get<1>(GetParam());
    std::srand((unsigned)(1234 + n * 31 + m));
    auto x = DTW::generateRandomSequence(n);
    auto y = DTW::generateRandomSequence(m);
    double g = dtw.compute(x, y);
    double c = DTW::cpuDtwSerial(x, y, n, m, 1);
    ASSERT_GE(g, 0.0);
    ASSERT_GE(c, 0.0);
    EXPECT_NEAR(g, c, 1e-9 * std::max(1.0, std::abs(c)));
}

INSTANTIATE_TEST_SUITE_P(MatchesCpuReference, DTWMatchesCpu,
    ::testing::Values(
        std::make_tuple(50,  50),
        std::make_tuple(50,  200),
        std::make_tuple(200, 50),
        std::make_tuple(300, 300),
        std::make_tuple(513, 257),
        std::make_tuple(800, 1200)));

// Multi-dim Euclidean: GPU vs CPU reference.
TEST(DTWMultiDim, EuclideanMatchesCpu) {
    DTW dtw{256};
    int n = 200, m = 250, dim = 4;
    std::srand(99);
    auto x = DTW::generateRandomSequence(n, dim);
    auto y = DTW::generateRandomSequence(m, dim);
    auto r = dtw.computeMultiDim(x, y, n, m, dim, DTW::EUCLIDEAN);
    double c = DTW::cpuDtwSerial(x, y, n, m, dim, DTW::EUCLIDEAN);
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(r.distance, c, 1e-9 * std::max(1.0, std::abs(c)));
}

TEST(DTWMultiDim, ManhattanMatchesCpu) {
    DTW dtw{256};
    int n = 150, m = 220, dim = 3;
    std::srand(7);
    auto x = DTW::generateRandomSequence(n, dim);
    auto y = DTW::generateRandomSequence(m, dim);
    auto r = dtw.computeMultiDim(x, y, n, m, dim, DTW::MANHATTAN);
    double c = DTW::cpuDtwSerial(x, y, n, m, dim, DTW::MANHATTAN);
    ASSERT_TRUE(r.success);
    EXPECT_NEAR(r.distance, c, 1e-9 * std::max(1.0, std::abs(c)));
}

// Sakoe-Chiba band: GPU vs CPU reference, both constrained.
TEST(DTWBand, BandConstraintMatchesCpu) {
    DTW dtw{256};
    int n = 400, m = 400, window = 30;
    std::srand(2025);
    auto x = DTW::generateRandomSequence(n);
    auto y = DTW::generateRandomSequence(m);
    double g = dtw.compute(x, y, window);
    double c = DTW::cpuDtwSerial(x, y, n, m, 1, DTW::ABSOLUTE, window);
    ASSERT_GE(g, 0.0);
    ASSERT_GE(c, 0.0);
    EXPECT_NEAR(g, c, 1e-9 * std::max(1.0, std::abs(c)));
}

// Shared-memory and plain kernels must produce identical results.
TEST(DTWKernelVariants, SharedAndPlainAgree) {
    DTW s{256}; s.setUseSharedMemory(true);
    DTW p{256}; p.setUseSharedMemory(false);
    int n = 500, m = 700;
    std::srand(42);
    auto x = DTW::generateRandomSequence(n);
    auto y = DTW::generateRandomSequence(m);
    double ds = s.compute(x, y);
    double dp = p.compute(x, y);
    EXPECT_NEAR(ds, dp, 1e-9 * std::max(1.0, std::abs(dp)));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
