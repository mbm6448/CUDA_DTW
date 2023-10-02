#include "gtest/gtest.h"
#include "../include/DTW.h"

class DTWTestFixture : public ::testing::Test {
protected:
    // This function runs before each TEST_F in this test suite
    void SetUp() override {
        dtw = new DTW();
    }

    // This function runs after each TEST_F in this test suite
    void TearDown() override {
        delete dtw;
    }

    DTW* dtw;
};

// Basic test to ensure that DTW distance is non-negative
TEST_F(DTWTestFixture, NonNegativeDistance) {
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> y = {0.3, 0.2, 0.1};
    double distance = dtw->compute(x, y);
    EXPECT_GE(distance, 0);
}

// Basic test to ensure that identical sequences have DTW distance of 0
TEST_F(DTWTestFixture, ZeroDistanceForIdenticalSequences) {
    std::vector<double> x = {0.1, 0.2, 0.3};
    std::vector<double> y = x;
    double distance = dtw->compute(x, y);
    EXPECT_DOUBLE_EQ(distance, 0);
}

// Testing symmetry property of DTW distance: d(x, y) = d(y, x)
TEST_F(DTWTestFixture, SymmetryProperty) {
    std::vector<double> x = {0.1, 0.4, 0.5, 0.9};
    std::vector<double> y = {0.2, 0.3, 0.6};
    double distance_xy = dtw->compute(x, y);
    double distance_yx = dtw->compute(y, x);
    EXPECT_DOUBLE_EQ(distance_xy, distance_yx);
}

// Parameterized test for different sequence lengths
class DTWParameterizedTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(DTWParameterizedTest, DifferentSequenceLengths) {
    int lenX = std::get<0>(GetParam());
    int lenY = std::get<1>(GetParam());

    std::vector<double> x(lenX, 0.5); // a sequence of length lenX with all elements as 0.5
    std::vector<double> y(lenY, 0.5); // a sequence of length lenY with all elements as 0.5

    DTW dtw;
    double distance = dtw.compute(x, y);

    // Since both sequences have the same values, the distance should be zero
    EXPECT_DOUBLE_EQ(distance, 0);
}

INSTANTIATE_TEST_SUITE_P(
    DifferentLengths,
    DTWParameterizedTest,
    ::testing::Values(
        std::make_tuple(10, 20),
        std::make_tuple(100, 200),
        std::make_tuple(50, 100)
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
