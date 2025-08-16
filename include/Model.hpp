#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "LossFunction.hpp"
#include "GradientDescent.hpp"
#include "ReshapeAdapter.hpp"
#include "Tensor3D.hpp"

class Model
{
private:
    std::vector<std::shared_ptr<Layer>> layers;
    std::shared_ptr<LossFunction> lossFunction;
    std::shared_ptr<GradientDescent> optimizer;

public:
    Model();

    void addLayer(std::shared_ptr<Layer> layer);
    void setLoss(std::shared_ptr<LossFunction> lossFunction);
    void setOptimizer(std::shared_ptr<GradientDescent> optimizer);

    void compile(std::shared_ptr<LossFunction> loss, std::shared_ptr<GradientDescent> optimizer)
    {
        setLoss(loss);
        setOptimizer(optimizer);
    }

        Eigen::MatrixXf Model::predict(const Tensor3D &input);
        Eigen::MatrixXf predict(const Eigen::MatrixXf &input);

        void Model::train(const std::vector<Tensor3D> &X,
                          const std::vector<Eigen::MatrixXf> &Y,
                          int epochs, float learning_rate, int batch_size,
                          LossFunction::Type loss_type, bool verbose);
};
