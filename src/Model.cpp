#include "../include/Model.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <iostream>
Model::Model() {}

void Model::addLayer(std::shared_ptr<Layer> layer)
{
    layers.push_back(layer);
}

void Model::setLoss(std::shared_ptr<LossFunction> lossFunction)
{
    this->lossFunction = lossFunction;
}

void Model::setOptimizer(std::shared_ptr<GradientDescent> optimizer)
{
    this->optimizer = optimizer;
}

Eigen::MatrixXf Model::predict(const Eigen::MatrixXf &input)
{
    Eigen::MatrixXf output = input;
    for (const auto &layer : layers)
    {
        output = layer->forward(output);
    }
    return output;
}

// Flatten conv output before Dense
Eigen::MatrixXf flatten(const std::vector<Eigen::MatrixXf> &input)
{
    int total_size = 0;
    for (const auto &mat : input)
        total_size += mat.size();

    Eigen::MatrixXf flat(total_size, 1);
    int offset = 0;
    for (const auto &mat : input)
    {
        Eigen::Map<const Eigen::VectorXf> vec(mat.data(), mat.size());
        flat.block(offset, 0, vec.size(), 1) = vec;
        offset += vec.size();
    }

    return flat;
}
void Model::train(const std::vector<Tensor3D> &X,
                  const std::vector<Eigen::MatrixXf> &Y,
                  int epochs, float learning_rate, int batch_size,
                  LossFunction::Type loss_type, bool verbose)
{
    assert(lossFunction != nullptr && optimizer != nullptr);

    int samples = X.size();
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float epoch_loss = 0.0f;

        for (int i = 0; i < samples; i += batch_size)
        {
            std::vector<Tensor3D> batch_X(X.begin() + i, X.begin() + std::min(i + batch_size, samples));
            std::vector<Eigen::MatrixXf> batch_Y(Y.begin() + i, Y.begin() + std::min(i + batch_size, samples));

            for (size_t j = 0; j < batch_X.size(); ++j)
            {
                Eigen::MatrixXf output;

                // Handle input as Tensor3D
                Tensor3D tensor = batch_X[j];

                if (auto conv = std::dynamic_pointer_cast<ConvolutionAdapter>(layers[0]))
                {
                    auto conv_output = conv->forward(tensor.channels);
                    output = tensor.fromMatrix(tensor.flatten(conv_output));
                }
                else
                {
                    output = layers[0]->forward(tensor.flatten());
                }

                for (size_t l = 1; l < layers.size(); ++l)
                {
                    output = layers[l]->forward(output);
                }

                float loss = lossFunction->loss(batch_Y[j], output, loss_type);
                epoch_loss += loss;

                Eigen::MatrixXf grad = lossFunction->loss_derivative(batch_Y[j], output, loss_type);
                for (int k = static_cast<int>(layers.size()) - 1; k >= 0; --k)
                {
                    grad = layers[k]->backward(grad, learning_rate);
                }
            }
        }

        epoch_loss /= samples;
        if (verbose)
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs << ", Loss: " << epoch_loss << std::endl;
    }
}