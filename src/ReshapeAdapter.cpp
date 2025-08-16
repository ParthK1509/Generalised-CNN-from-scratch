#include "ReshapeAdapter.hpp"
#include <cassert>

ReshapeAdapter::ReshapeAdapter(int c, int h, int w)
    : channels(c), height(h), width(w) {}

std::vector<Eigen::MatrixXf> ReshapeAdapter::forward(const Eigen::MatrixXf& input) {
    int expected_size = channels * height * width;
    assert(input.rows() == expected_size && input.cols() == 1);

    std::vector<Eigen::MatrixXf> output(channels, Eigen::MatrixXf(height, width));
    int idx = 0;

    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                output[c](i, j) = input(idx++, 0);

    last_output = output;
    return output;
}

Eigen::MatrixXf ReshapeAdapter::backward(const std::vector<Eigen::MatrixXf>& grad_output) {
    Eigen::MatrixXf flattened(channels * height * width, 1);
    int idx = 0;

    for (const auto& m : grad_output)
        for (int i = 0; i < m.rows(); ++i)
            for (int j = 0; j < m.cols(); ++j)
                flattened(idx++, 0) = m(i, j);

    return flattened;
}
