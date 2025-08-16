#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <functional>

class GradientDescent {
public:
    GradientDescent(
        const std::string& type,
        float learning_rate,
        int batch_size = 1
    );

    void train(
        std::vector<class Layer*>& network,
        const std::vector<Eigen::MatrixXf>& x_train,
        const std::vector<Eigen::MatrixXf>& y_train,
        int epochs,
        std::function<float(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> loss,
        std::function<Eigen::MatrixXf(const Eigen::MatrixXf&, const Eigen::MatrixXf&)> loss_prime,
        bool verbose = true
    );

private:
    std::string type;
    float learning_rate;
    int batch_size;
};
