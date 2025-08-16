#include "DenseLayer.hpp"
#include <random>
#include <iostream>
//checked and debugged
DenseLayer::DenseLayer(int input_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(0.0f, 1.0f);

    weights = Eigen::MatrixXf::NullaryExpr(output_size, input_size, [&]() { return d(gen); });
    bias = Eigen::VectorXf::Zero(output_size);

}

Eigen::MatrixXf DenseLayer::forward(const Eigen::MatrixXf& input) {
    this->input = input;
    return (weights * input).colwise() + bias;
}

Eigen::MatrixXf DenseLayer::backward(const Eigen::MatrixXf& output_gradient, float learning_rate) {
    Eigen::MatrixXf grad_weights = output_gradient * input.transpose();
    Eigen::VectorXf grad_bias = output_gradient.rowwise().mean();

    weights -= learning_rate * grad_weights;
    bias -= learning_rate * grad_bias;

    return weights.transpose() * output_gradient;
}
