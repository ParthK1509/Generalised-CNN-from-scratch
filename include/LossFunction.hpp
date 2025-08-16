#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include <Eigen/Dense>

class LossFunction {
public:
    enum Type {
        MSE,
        CROSS_ENTROPY,
        BINARY_CROSS_ENTROPY
    };

    // Constructor with a type parameter
    LossFunction(Type type = MSE);

    // Method to compute loss
    double loss(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred, Type type);

    // Method to compute derivative of loss
    Eigen::MatrixXf loss_derivative(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred, Type type);

private:
    Type lossType;  // Field to store the loss type
};

#endif // LOSSFUNCTION_HPP
