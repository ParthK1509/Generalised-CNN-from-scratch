#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <Eigen/Dense>

using namespace std;

vector<pair<Eigen::MatrixXf, Eigen::MatrixXf>> load_mnist_csv(const string& filename);

#endif // CSV_READER_H