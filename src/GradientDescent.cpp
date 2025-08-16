#include "GradientDescent.hpp"
#include "Layer.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

GradientDescent::GradientDescent(const std::string& type, float lr, int batch_sz)
    : type(type), learning_rate(lr), batch_size(batch_sz) {}

void GradientDescent::train(
    std::vector<Layer*>& network,
    const std::vector<Eigen::MatrixXf>& x_train,
    const std::vector<Eigen::MatrixXf>& y_train,
    int epochs,
    std::function<float(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> loss,
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> loss_prime,
    bool verbose)
{
    int n = x_train.size();
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // 0 to n-1

    for (int e = 0; e < epochs; ++e)
    {
        float total_error = 0.0f;

        if (type != "batch") {
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(e));
        }

        int i = 0;
        while (i < n)
        {
            int current_batch_size = (type == "mini-batch") ? batch_size : (type == "stochastic" ? 1 : n);
            std::vector<Eigen::MatrixXf> batch_x, batch_y;

            for (int j = 0; j < current_batch_size && i + j < n; ++j)
            {
                int idx = (type == "batch") ? j : indices[i + j];
                batch_x.push_back(x_train[idx]);
                batch_y.push_back(y_train[idx]);
            }

            for (int k = 0; k < batch_x.size(); ++k)
            {
                Eigen::MatrixXf out = batch_x[k];

                std::cout << "Forward pass input (" << out.rows() << "x" << out.cols() << "):\n"
                          << out << "\n";

                for (size_t l = 0; l < network.size(); ++l)
                {
                    out = network[l]->forward(out);
                    std::cout << "After layer " << l << " output:\n"
                              << out << "\n";
                }

                float err = loss(batch_y[k], out);
                std::cout << "Loss: " << err << "\n";
                total_error += err;

                Eigen::MatrixXf grad = loss_prime(batch_y[k], out);
                std::cout << "Initial gradient (" << grad.rows() << "x" << grad.cols() << "):\n"
                          << grad << "\n";

                for (int l = network.size() - 1; l >= 0; --l)
                {
                    grad = network[l]->backward(grad, learning_rate);
                    std::cout << "Gradient after layer " << l << " (" << grad.rows() << "x" << grad.cols() << "):\n"
                              << grad << "\n";
                }
            }

            i += current_batch_size;
        }

        if (verbose && e % 10 == 0) // print only every 10th epoch
        {
            std::cout << "Epoch " << (e + 1) << "/" << epochs << " completed. Avg Loss = "
                      << (total_error / n) << "\n\n";
        }
    }
}
