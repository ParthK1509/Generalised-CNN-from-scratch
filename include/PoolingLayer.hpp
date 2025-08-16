#ifndef POOLING_LAYER_HPP
#define POOLING_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>

class PoolingLayer : public Layer {
public:
    enum class Type { MAX, AVERAGE };

    PoolingLayer(int pool_h, int pool_w, int stride, Type type);

    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input);
    std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf>& d_out, float learning_rate);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;   // dummy
    Eigen::MatrixXf backward(const Eigen::MatrixXf& d_out, float learning_rate) override; // dummy

private:
    int pool_h, pool_w, stride;
    Type type;
    std::vector<Eigen::MatrixXf> input;
};
#endif
