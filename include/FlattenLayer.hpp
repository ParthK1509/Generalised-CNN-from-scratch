#ifndef FLATTEN_LAYER_HPP
#define FLATTEN_LAYER_HPP

#include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>

class FlattenLayer : public Layer {
public:
    FlattenLayer(int channels, int height, int width);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& d_out, float learning_rate) override;

    Eigen::MatrixXf flatten(const std::vector<Eigen::MatrixXf>& input);
    std::vector<Eigen::MatrixXf> unflatten(const Eigen::MatrixXf& input);

private:
    int channels, height, width;
    std::vector<Eigen::MatrixXf> last_input;
};
#endif
