#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <Eigen/Dense>

// Custom headers
#include "Model.hpp"
#include "DenseLayer.hpp"
#include "ActivationFunction.hpp"
#include "ConvolutionAdapter.hpp"
#include "PoolingLayer.hpp"
#include "FlattenLayer.hpp"
#include "LossFunction.hpp"
#include "GradientDescent.hpp"
#include "Utils.hpp" // for load_mnist_csv

using namespace std;


int main()
{
    constexpr int OUTPUT_CLASSES = 10;
    constexpr int EPOCHS = 10;
    constexpr float LEARNING_RATE = 0.01f;
    constexpr int BATCH_SIZE = 32;

    // Load training and testing datasets
    std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> train_data = load_mnist_csv("train.csv");
    std::vector<std::pair<Eigen::MatrixXf, Eigen::MatrixXf>> test_data = load_mnist_csv("test.csv");

    // Check if datasets are loaded correctly
    if (train_data.empty() || test_data.empty())
    {
        std::cerr << "Error: Failed to load datasets.\n";
        return -1;
    }

    // Split into x/y
    std::vector<Eigen::MatrixXf> x_train, y_train, x_test, y_test;
    for (const auto &[img, lbl] : train_data)
    {
        x_train.push_back(img);
        y_train.push_back(lbl);
    }
    for (const auto &[img, lbl] : test_data)
    {
        x_test.push_back(img);
        y_test.push_back(lbl);
    }

    // Check if training data is valid
    if (x_train.empty() || y_train.empty())
    {
        std::cerr << "Error: Training data is empty.\n";
        return -1;
    }

    std::cout << "Input shape: " << x_train[0].rows() << "x" << x_train[0].cols() << "\n";
    std::cout << "Label shape: " << y_train[0].rows() << "x" << y_train[0].cols() << "\n";

    Model model;

    // Convolutional Layer 1 using ConvolutionAdapter
    model.addLayer(std::make_shared<ConvolutionAdapter>(1, 28, 28, 3, 8, 1,1)); // in_c, in_h, in_w, k_size, num_filters, stride
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::ReLU));
    model.addLayer(std::make_shared<PoolingLayer>(2, 2, 2, PoolingLayer::Type::MAX));

    // Convolutional Layer 2 using ConvolutionAdapter
    model.addLayer(std::make_shared<ConvolutionAdapter>(8, 14, 14, 3, 16, 1,0));
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::ReLU));
    model.addLayer(std::make_shared<PoolingLayer>(2, 2, 2, PoolingLayer::Type::MAX));

    // Flatten Layer (use correct dimensions after pooling)
    model.addLayer(std::make_shared<FlattenLayer>(16, 5, 5)); // Adjust according to the dimensions after pooling

    // Dense Layers
    model.addLayer(std::make_shared<DenseLayer>(16 * 5 * 5, 64));
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::ReLU));
    model.addLayer(std::make_shared<DenseLayer>(64, OUTPUT_CLASSES));
    model.addLayer(std::make_shared<ActivationFunction>(ActivationFunction::Type::Softmax));

    // Loss and Optimizer
    model.setLoss(std::make_shared<LossFunction>(LossFunction::Type::CROSS_ENTROPY));
    model.setOptimizer(std::make_shared<GradientDescent>("mini-batch", LEARNING_RATE, BATCH_SIZE));

    // Train
    auto start = std::chrono::high_resolution_clock::now();
    // model.train(x_train, y_train, EPOCHS, BATCH_SIZE);
    model.train(x_train, y_train, EPOCHS, LEARNING_RATE, BATCH_SIZE, LossFunction::Type::CROSS_ENTROPY, true);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Training completed in "
              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
              << " seconds.\n";

    // Evaluate
    int correct = 0;
    for (size_t i = 0; i < x_test.size(); ++i)
    {
        Eigen::MatrixXf prediction = model.predict(x_test[i]);

        int predicted_label;
        prediction.col(0).maxCoeff(&predicted_label);

        int actual_label;
        y_test[i].col(0).maxCoeff(&actual_label);

        if (predicted_label == actual_label)
        {
            ++correct;
        }
    }

    float accuracy = 100.0f * correct / x_test.size();
    std::cout << "Test Accuracy: " << accuracy << "%\n";

    return 0;
}
