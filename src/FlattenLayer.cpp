#include "FlattenLayer.hpp"

FlattenLayer::FlattenLayer(int channels, int height, int width)
    : channels(channels), height(height), width(width) {}

Eigen::MatrixXf FlattenLayer::forward(const Eigen::MatrixXf& input) {
    // Store input (already flat) as 3D internally
    last_input = unflatten(input);
    return flatten(last_input);
}

Eigen::MatrixXf FlattenLayer::backward(const Eigen::MatrixXf& d_out, float learning_rate) {
    return flatten(unflatten(d_out));  // Keep consistent shape
}

Eigen::MatrixXf FlattenLayer::flatten(const std::vector<Eigen::MatrixXf>& input) {
    last_input = input;

    int size = input.size() * input[0].rows() * input[0].cols();
    Eigen::MatrixXf output(size, 1);

    int idx = 0;
    for (const auto& mat : input) {
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                output(idx++, 0) = mat(i, j);
            }
        }
    }

    return output;
}

std::vector<Eigen::MatrixXf> FlattenLayer::unflatten(const Eigen::MatrixXf& input) {
    std::vector<Eigen::MatrixXf> output(channels, Eigen::MatrixXf(height, width));
    int idx = 0;

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                output[c](i, j) = input(idx++, 0);
            }
        }
    }

    return output;
}
