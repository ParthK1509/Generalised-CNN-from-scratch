#include "ActivationFunction.hpp"

using namespace Eigen;

//Checked

ActivationFunction::ActivationFunction(Type t) : type(t) {}

MatrixXf ActivationFunction::forward(const MatrixXf &input) {
    last_input = input;
    return apply(input);
}

MatrixXf ActivationFunction::backward(const MatrixXf &grad_output, float) {
    // if (type == Type::Softmax) {
    //     // Don't apply derivative; loss derivative handles it
    //     return grad_output;
    // }
    return grad_output.cwiseProduct(apply_derivative(last_input));
}


MatrixXf ActivationFunction::apply(const MatrixXf &x) {
    switch (type) {
        case Type::Tanh:
            return x.array().tanh().matrix();
        case Type::Sigmoid:
            return (1.0f / (1.0f + (-x.array()).exp())).matrix();
        case Type::ReLU:
            return x.cwiseMax(0.0f);
        case Type::Softmax: {
    if (x.cols() == 1) {
        // Apply softmax over the column (vector)
        VectorXf x_shifted = x.col(0).array() - x.col(0).maxCoeff();
        VectorXf exp = x_shifted.array().exp();
        float sum = exp.sum();
        return exp / sum;
    } else {
        // Usual softmax over rows
        // MatrixXf x_shifted = x.rowwise() - x.rowwise().maxCoeff().replicate(1, x.cols());
        VectorXf row_max = x.rowwise().maxCoeff();
        MatrixXf x_shifted = x.colwise() - row_max;

        MatrixXf exp = x_shifted.array().exp();
        VectorXf sum = exp.rowwise().sum();
        return exp.array().colwise() / sum.array();
    }
}
    }
    return x;
}

MatrixXf ActivationFunction::apply_derivative(const MatrixXf &x) {
    switch (type) {
        case Type::Tanh:
            return (1.0f - x.array().tanh().square()).matrix();
        case Type::Sigmoid: {
            MatrixXf sig = (1.0f / (1.0f + (-x.array()).exp())).matrix();
            return (sig.array() * (1.0f - sig.array())).matrix();
        }
        case Type::ReLU:
            return (x.array() > 0).cast<float>();
        case Type::Softmax: {
            MatrixXf s = apply(x);
            return (s.array() * (1.0f - s.array())).matrix(); // crude approx
        }
    }
    return x;
}
