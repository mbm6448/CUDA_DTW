#include <iostream>
#include "DTW.h"

int main() {
    DTW dtw;

    std::vector<double> x = DTW::generateRandomSequence(30000);
    std::vector<double> y = DTW::generateRandomSequence(50000);

    double dtw_distance = dtw.compute(x, y);

    std::cout << "DTW distance: " << dtw_distance << std::endl;

    return 0;
}

