#pragma once
#include "Layer.hpp"
#include <Eigen/Dense>

//checked and debugged
class DenseLayer : public Layer {
private:
    Eigen::MatrixXf weights;
    Eigen::VectorXf bias;
    Eigen::MatrixXf input;

public:
    DenseLayer(int input_size, int output_size);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& output_gradient, float learning_rate) override;
};
