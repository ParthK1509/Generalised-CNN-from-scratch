#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include <Eigen/Dense>
#include <string>
#include <stdexcept>  // for std::invalid_argument
#include "Layer.hpp"

using namespace Eigen;
//perfect class man

class ActivationFunction : public Layer {
public:
    enum class Type {
        Tanh,
        Sigmoid,
        ReLU,
        Softmax
    };

    ActivationFunction(Type type);

    MatrixXf forward(const MatrixXf &input) override;
    MatrixXf backward(const MatrixXf &grad_output, float learning_rate) override;

private:
    Type type;
    MatrixXf last_input;

    MatrixXf apply(const MatrixXf &x);
    MatrixXf apply_derivative(const MatrixXf &x);
};

// 🔽 Add this outside the class but before #endif
inline ActivationFunction::Type activationTypeFromString(const std::string& str) {
    if (str == "tanh") return ActivationFunction::Type::Tanh;
    if (str == "sigmoid") return ActivationFunction::Type::Sigmoid;
    if (str == "relu") return ActivationFunction::Type::ReLU;
    if (str == "softmax") return ActivationFunction::Type::Softmax;
    throw std::invalid_argument("Unknown activation type: " + str);
}

#endif
