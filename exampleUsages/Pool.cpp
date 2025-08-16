#include <iostream>
#include <Eigen/Dense>
#include "PoolingLayer.hpp"

int main() {
    Eigen::MatrixXf input(4, 4);
    input << 1, 2, 3, 4,
             5, 6, 7, 8,
             9,10,11,12,
            13,14,15,16;

    std::vector<Eigen::MatrixXf> input_channels = {input};  // single-channel test

    std::cout << "Input:\n" << input << "\n\n";

    // Test Max Pooling
    PoolingLayer maxPool(2, 2, 2, PoolingLayer::MAX);
    std::vector<Eigen::MatrixXf> maxOut = maxPool.forward(input_channels);

    std::cout << "Max Pooling Output:\n" << maxOut[0] << "\n\n";

    std::vector<Eigen::MatrixXf> d_out = {Eigen::MatrixXf::Ones(maxOut[0].rows(), maxOut[0].cols())};
    std::vector<Eigen::MatrixXf> maxGrad = maxPool.backward(d_out);
    std::cout << "Max Pooling Backward Gradient:\n" << maxGrad[0] << "\n\n";

    // Test Average Pooling
    PoolingLayer avgPool(2, 2, 2, PoolingLayer::AVERAGE);
    std::vector<Eigen::MatrixXf> avgOut = avgPool.forward(input_channels);
    std::cout << "Average Pooling Output:\n" << avgOut[0] << "\n\n";

    std::vector<Eigen::MatrixXf> avgGrad = avgPool.backward(d_out);
    std::cout << "Average Pooling Backward Gradient:\n" << avgGrad[0] << "\n\n";

    return 0;
}

// compile with : g++ -std=c++17 -I./include -I./eigen-master src/PoolingLayer.cpp main.cpp -o pooling_test
// run with : ./pooling_test