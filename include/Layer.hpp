#pragma once
#include <Eigen/Dense>

using namespace Eigen;
//checked
class Layer {
public:
    virtual ~Layer() {}

    virtual MatrixXf forward(const MatrixXf &input) = 0;
    virtual MatrixXf backward(const MatrixXf &output_gradient, float learning_rate) = 0;
};
