#include "LossFunction.hpp"
#include <iostream>
#include <cmath>

// Constructor that initializes the lossType field
LossFunction::LossFunction(Type type) : lossType(type) {}

double LossFunction::loss(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred, Type type) {
    Eigen::ArrayXf y_t = Eigen::Map<const Eigen::ArrayXf>(y_true.data(), y_true.size());
    Eigen::ArrayXf y_p = Eigen::Map<const Eigen::ArrayXf>(y_pred.data(), y_pred.size()).max(1e-15).min(1.0f - 1e-15);

    switch (type) {
        case MSE:
            return (y_t - y_p).square().mean();
        case CROSS_ENTROPY:
            return -(y_t * y_p.log()).mean();
        case BINARY_CROSS_ENTROPY:
            return (-y_t * y_p.log() - (1.0f - y_t) * (1.0f - y_p).log()).mean();
        default:
            std::cerr << "[LossFunction] Unsupported loss type!" << std::endl;
            return 0.0;
    }
}

Eigen::MatrixXf LossFunction::loss_derivative(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred, Type type) {
    Eigen::ArrayXf y_t = Eigen::Map<const Eigen::ArrayXf>(y_true.data(), y_true.size());
    Eigen::ArrayXf y_p = Eigen::Map<const Eigen::ArrayXf>(y_pred.data(), y_pred.size()).max(1e-15).min(1.0f - 1e-15);

    switch (type) {
        case MSE:
            return (2.0f * (y_p - y_t) / y_t.size()).matrix();
        case BINARY_CROSS_ENTROPY:
            return (-(y_t / y_p) + (1.0f - y_t) / (1.0f - y_p)).matrix() / y_t.size();
        case CROSS_ENTROPY:
            return (y_p - y_t).matrix() / y_t.size();
        default:
            std::cerr << "[LossFunction] Unsupported loss type!" << std::endl;
            return Eigen::MatrixXf::Zero(y_true.rows(), y_true.cols());
    }
}
