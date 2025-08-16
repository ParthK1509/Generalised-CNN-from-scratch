#include <iostream>
#include <Eigen/Dense>
#include "DenseLayer.hpp"

int main() {
    Eigen::MatrixXf input(3, 1);
    input << 1.0f, 2.0f, 3.0f;

    DenseLayer layer(3, 2);

    Eigen::MatrixXf output = layer.forward(input);
    std::cout << "Forward output:\n" << output << std::endl;

    Eigen::MatrixXf grad_output = Eigen::MatrixXf::Ones(2, 1);
    Eigen::MatrixXf grad_input = layer.backward(grad_output, 0.01f);
    std::cout << "Backward output:\n" << grad_input << std::endl;

    return 0;
}
// Compile with
// g++ -I include -I ./eigen-master main.cpp src/DenseLayer.cpp -o test_dense
// ./test_dense