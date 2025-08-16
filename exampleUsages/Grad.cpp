#include "GradientDescent.hpp"
#include "Dense.hpp"        // will follow Layer interface
#include "ActivationFunction.hpp"
#include "LossFunctions.hpp"

int main() {
    std::vector<Layer*> network = {
        new Dense(2, 3),
        new ActivationFunction("tanh"),
        new Dense(3, 1),
        new ActivationFunction("sigmoid")
    };

    std::vector<Eigen::MatrixXd> X = { ... };
    std::vector<Eigen::MatrixXd> Y = { ... };

    GradientDescent trainer("mini-batch", 0.1, 2);
    trainer.train(network, X, Y, 10000, mse, mse_prime);

    // Cleanup
    for (auto* layer : network) delete layer;
}
