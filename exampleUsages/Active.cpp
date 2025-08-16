#include <iostream>
#include "ActivationFunction.hpp"

using namespace std;
using namespace Eigen;

void testActivation(const std::string& name, ActivationFunction::Type type, const MatrixXf& input) {
    ActivationFunction act(type);
    MatrixXf out = act.forward(input);
    MatrixXf grad_output = MatrixXf::Ones(input.rows(), input.cols());
    MatrixXf grad = act.backward(grad_output, 0.01f);

    cout << "\n=== " << name << " ===\n";
    cout << "Input:\n" << input << endl;
    cout << "Output:\n" << out << endl;
    cout << "Backward Gradient:\n" << grad << endl;
}

int main() {
    MatrixXf input(2, 2);
    input << -1, 0,
              1, 2;

    testActivation("ReLU", ActivationFunction::Type::ReLU, input);
    testActivation("Sigmoid", ActivationFunction::Type::Sigmoid, input);
    testActivation("Tanh", ActivationFunction::Type::Tanh, input);
    testActivation("Softmax", ActivationFunction::Type::Softmax, input);  // Softmax should ideally be tested with row-wise interpretation

    return 0;
}
//command : g++ -std=c++17 -I ./eigen-master -I include src/ActivationFunction.cpp main.cpp -o main
