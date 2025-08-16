#include <iostream>
#include <Eigen/Dense>
#include "Model.hpp"
#include "DenseLayer.hpp"
#include "ActivationFunction.hpp"
#include "LossFunction.hpp"
#include "GradientDescent.hpp"

int main()
{
    // Create a simple model
    Model model;

    // Add Dense layers with activation
    // model.addLayer(std::make_shared<DenseLayer>(2, 4, ActivationFunction::Type::ReLU));
    // model.addLayer(std::make_shared<DenseLayer>(4, 1, ActivationFunction::Type::Sigmoid));
    model.addLayer(std::make_shared<DenseLayer>(2, 4));
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::ReLU));

    model.addLayer(std::make_shared<DenseLayer>(4, 1));
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::Sigmoid));

    // Set loss and optimizer
    model.setLoss(std::make_shared<LossFunction>()); // Generic handler; type is passed in train()
    // model.setOptimizer(std::make_shared<StochasticGradientDescent>());  // Make sure class exists
    model.setOptimizer(std::make_shared<GradientDescent>("stochastic", 0.1f, 1));

    // XOR dataset
    std::vector<Eigen::MatrixXf> X = {
        (Eigen::MatrixXf(2, 1) << 0, 0).finished(),
        (Eigen::MatrixXf(2, 1) << 0, 1).finished(),
        (Eigen::MatrixXf(2, 1) << 1, 0).finished(),
        (Eigen::MatrixXf(2, 1) << 1, 1).finished()};

    std::vector<Eigen::MatrixXf> Y = {
        (Eigen::MatrixXf(1, 1) << 0).finished(),
        (Eigen::MatrixXf(1, 1) << 1).finished(),
        (Eigen::MatrixXf(1, 1) << 1).finished(),
        (Eigen::MatrixXf(1, 1) << 0).finished()};

    // Train the model
    model.train(X, Y, 1000, 0.1f, 1, LossFunction::Type::BINARY_CROSS_ENTROPY);

    // Test predictions
    for (const auto &input : X)
    {
        Eigen::MatrixXf output = model.predict(input);
        std::cout << "Input:\n"
                  << input.transpose()
                  << " -> Output:\n"
                  << output << "\n";
    }

    return 0;
}

// Compile with g++ -std=c++17 -Iinclude -I ./eigen-master main.cpp src/*.cpp -o test_model