#include "gtest/gtest.h"
#include "../include/DTW.h"
#include <cmath>

class DTWTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        dtw = new DTW(256);  // Use consistent block size
    }

    void TearDown() override {
        delete dtw;
    }

    DTW* dtw;
    
    // Helper function to compare doubles with tolerance
    bool areClose(double a, double b, double tolerance = 1e-9) {
        return std::abs(a - b) < tolerance;
    }
};

// Test that DTW distance is non-negative
TEST_F(DTWTestFixture, NonNegativeDistance) {
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> y = {0.3, 0.2, 0.1};
    double distance = dtw->compute(x, y);
    EXPECT_GE(distance, 0.0);
}

// Test that identical sequences have DTW distance of 0
TEST_F(DTWTestFixture, ZeroDistanceForIdenticalSequences) {
    std::vector<double> x = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> y = x;
    double distance = dtw->compute(x, y);
    EXPECT_NEAR(distance, 0.0, 1e-9);
}

// Test symmetry property: d(x, y) = d(y, x)
TEST_F(DTWTestFixture, SymmetryProperty) {
    std::vector<double> x = {0.1, 0.4, 0.5, 0.9};
    std::vector<double> y = {0.2, 0.3, 0.6};
    double distance_xy = dtw->compute(x, y);
    double distance_yx = dtw->compute(y, x);
    EXPECT_NEAR(distance_xy, distance_yx, 1e-9);
}

// Test with single-element sequences
TEST_F(DTWTestFixture, SingleElementSequences) {
    std::vector<double> x = {0.5};
    std::vector<double> y = {0.3};
    double distance = dtw->compute(x, y);
    EXPECT_NEAR(distance, 0.2, 1e-9);  // |0.5 - 0.3| = 0.2
}

// Test with empty sequences (should return error)
TEST_F(DTWTestFixture, EmptySequences) {
    std::vector<double> x = {};
    std::vector<double> y = {0.1, 0.2};
    double distance = dtw->compute(x, y);
    EXPECT_EQ(distance, -1.0);  // Error case
    
    x = {0.1, 0.2};
    y = {};
    distance = dtw->compute(x, y);
    EXPECT_EQ(distance, -1.0);  // Error case
}

// Test triangle inequality (may not hold strictly for DTW)
TEST_F(DTWTestFixture, TriangleInequalityCheck) {
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> y = {0.4, 0.5, 0.6};
    std::vector<double> z = {0.7, 0.8, 0.9};
    
    double d_xy = dtw->compute(x, y);
    double d_yz = dtw->compute(y, z);
    double d_xz = dtw->compute(x, z);
    
    // DTW doesn't always satisfy triangle inequality strictly,
    // but we can check that distances are reasonable
    EXPECT_GT(d_xy, 0);
    EXPECT_GT(d_yz, 0);
    EXPECT_GT(d_xz, 0);
}

// Test with constant sequences
TEST_F(DTWTestFixture, ConstantSequences) {
    std::vector<double> x(10, 0.5);  // All 0.5
    std::vector<double> y(15, 0.5);  // All 0.5
    double distance = dtw->compute(x, y);
    EXPECT_NEAR(distance, 0.0, 1e-9);
    
    std::vector<double> z(20, 0.7);  // All 0.7
    distance = dtw->compute(x, z);
    // Should be |0.5 - 0.7| * min(10, 20) = 0.2 * 10 = 2.0
    EXPECT_NEAR(distance, 2.0, 1e-9);
}

// Test known DTW result
TEST_F(DTWTestFixture, KnownDTWResult) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {2.0, 3.0, 4.0};
    
    // DTW matrix:
    // |1  2  4|
    // |2  2  3|
    // |4  3  3|
    // Result should be 3.0
    
    double distance = dtw->compute(x, y);
    EXPECT_NEAR(distance, 3.0, 1e-9);
}

// Parameterized test for different sequence lengths
class DTWParameterizedTest : public ::testing::TestWithParam<std::tuple<int, int>> {
protected:
    DTW dtw{256};
};

TEST_P(DTWParameterizedTest, DifferentSequenceLengths) {
    int lenX = std::get<0>(GetParam());
    int lenY = std::get<1>(GetParam());

    std::vector<double> x(lenX, 0.5);
    std::vector<double> y(lenY, 0.5);

    double distance = dtw.compute(x, y);
    EXPECT_NEAR(distance, 0.0, 1e-9);
}

INSTANTIATE_TEST_SUITE_P(
    DifferentLengths,
    DTWParameterizedTest,
    ::testing::Values(
        std::make_tuple(10, 20),
        std::make_tuple(50, 100),
        std::make_tuple(100, 200),
        std::make_tuple(500, 500)
    )
);

// Stress test with larger sequences
TEST_F(DTWTestFixture, LargerSequences) {
    int n = 1000;
    int m = 1500;
    
    std::vector<double> x = DTW::generateRandomSequence(n);
    std::vector<double> y = DTW::generateRandomSequence(m);
    
    double distance = dtw->compute(x, y);
    EXPECT_GE(distance, 0.0);
    
    // Test with same sequence
    double self_distance = dtw->compute(x, x);
    EXPECT_NEAR(self_distance, 0.0, 1e-9);
}

// Test monotonicity: closer sequences should have smaller distances
TEST_F(DTWTestFixture, MonotonicityTest) {
    std::vector<double> base = {0.0, 0.1, 0.2, 0.3, 0.4};
    std::vector<double> close = {0.01, 0.11, 0.21, 0.31, 0.41};
    std::vector<double> far = {0.5, 0.6, 0.7, 0.8, 0.9};
    
    double d_close = dtw->compute(base, close);
    double d_far = dtw->compute(base, far);
    
    EXPECT_LT(d_close, d_far);  // Closer sequence should have smaller distance
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}