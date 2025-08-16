#include "LossFunction.hpp"
#include <iostream>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXf y_true(2, 1);
    Eigen::MatrixXf y_pred(2, 1);

    y_true << 1, 0;
    y_pred << 0.9, 0.1;

    std::cout << "=== MSE ===" << std::endl;
    double mse_loss = LossFunction::loss(y_true, y_pred, LossFunction::MSE);
    Eigen::MatrixXf mse_grad = LossFunction::loss_derivative(y_true, y_pred, LossFunction::MSE);
    std::cout << "Loss: " << mse_loss << "\nDerivative:\n" << mse_grad << std::endl;

    std::cout << "\n=== Binary Cross Entropy ===" << std::endl;
    double bce_loss = LossFunction::loss(y_true, y_pred, LossFunction::BINARY_CROSS_ENTROPY);
    Eigen::MatrixXf bce_grad = LossFunction::loss_derivative(y_true, y_pred, LossFunction::BINARY_CROSS_ENTROPY);
    std::cout << "Loss: " << bce_loss << "\nDerivative:\n" << bce_grad << std::endl;

    std::cout << "\n=== Cross Entropy ===" << std::endl;
    double ce_loss = LossFunction::loss(y_true, y_pred, LossFunction::CROSS_ENTROPY);
    Eigen::MatrixXf ce_grad = LossFunction::loss_derivative(y_true, y_pred, LossFunction::CROSS_ENTROPY);
    std::cout << "Loss: " << ce_loss << "\nDerivative:\n" << ce_grad << std::endl;

    return 0;
}

//compiling command : g++ -std=c++17 -Iinclude -I ./eigen-master src/LossFunction.cpp main.cpp -o loss_test