#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>
#include <Eigen/Dense>
#include "Utils.hpp"

using namespace std;

vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> load_mnist_csv(const string& filename) {
    vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> dataset;
    ifstream file(filename);
    string line;

    // Skip header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;

        Eigen::MatrixXf input(784, 1);           // 784x1 input vector
        Eigen::MatrixXf label_onehot(10, 1);     // 10x1 one-hot label vector
        label_onehot.setZero();

        // Read label
        getline(ss, cell, ',');
        int label = stoi(cell);
        label_onehot(label, 0) = 1.0f;

        // Read pixels
        for (int i = 0; i < 784; ++i) {
            getline(ss, cell, ',');
            input(i, 0) = stof(cell) / 255.0f;
        }

        dataset.emplace_back(input, label_onehot);
    }

    return dataset;
}
