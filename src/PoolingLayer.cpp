#include "PoolingLayer.hpp"
#include <limits>

PoolingLayer::PoolingLayer(int pool_h, int pool_w, int stride, Type type)
    : pool_h(pool_h), pool_w(pool_w), stride(stride), type(type) {}

std::vector<Eigen::MatrixXf> PoolingLayer::forward(const std::vector<Eigen::MatrixXf>& input) {
    this->input = input;
    std::vector<Eigen::MatrixXf> output;

    for (const auto& channel : input) {
        int out_h = (channel.rows() - pool_h) / stride + 1;
        int out_w = (channel.cols() - pool_w) / stride + 1;
        Eigen::MatrixXf pooled(out_h, out_w);

        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                float val;
                if (type == Type::MAX) {
                    val = std::numeric_limits<float>::lowest();
                } else {
                    val = 0.0f;
                }

                for (int m = 0; m < pool_h; ++m) {
                    for (int n = 0; n < pool_w; ++n) {
                        float current = channel(i * stride + m, j * stride + n);
                        if (type == Type::MAX) {
                            val = std::max(val, current);
                        } else {
                            val += current;
                        }
                    }
                }

                if (type == Type::AVERAGE) {
                    val /= (pool_h * pool_w);
                }

                pooled(i, j) = val;
            }
        }

        output.push_back(pooled);
    }

    return output;
}

std::vector<Eigen::MatrixXf> PoolingLayer::backward(const std::vector<Eigen::MatrixXf>& d_out, float /* learning_rate */) {
    std::vector<Eigen::MatrixXf> d_input(input.size());

    for (size_t c = 0; c < input.size(); ++c) {
        const auto& channel = input[c];
        int out_h = d_out[c].rows();
        int out_w = d_out[c].cols();
        Eigen::MatrixXf grad = Eigen::MatrixXf::Zero(channel.rows(), channel.cols());

        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                float grad_val = d_out[c](i, j);
                if (type == Type::MAX) {
                    float max_val = std::numeric_limits<float>::lowest();
                    int max_i = -1, max_j = -1;

                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            if (channel(row, col) > max_val) {
                                max_val = channel(row, col);
                                max_i = row;
                                max_j = col;
                            }
                        }
                    }

                    grad(max_i, max_j) += grad_val;

                } else {
                    float avg_grad = grad_val / (pool_h * pool_w);
                    for (int m = 0; m < pool_h; ++m) {
                        for (int n = 0; n < pool_w; ++n) {
                            int row = i * stride + m;
                            int col = j * stride + n;
                            grad(row, col) += avg_grad;
                        }
                    }
                }
            }
        }

        d_input[c] = grad;
    }

    return d_input;
}

// Dummy overrides to satisfy Layer interface

Eigen::MatrixXf PoolingLayer::forward(const Eigen::MatrixXf& input) {
    return input; // Not used in actual pipeline
}

Eigen::MatrixXf PoolingLayer::backward(const Eigen::MatrixXf& d_out, float learning_rate) {
    return d_out; // Not used in actual pipeline
}
