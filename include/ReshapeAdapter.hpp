#pragma once
#include <vector>
#include <Eigen/Dense>

class ReshapeAdapter {
private:
    int channels;
    int height;
    int width;
    std::vector<Eigen::MatrixXf> last_output;

public:
    ReshapeAdapter(int c, int h, int w);

    std::vector<Eigen::MatrixXf> forward(const Eigen::MatrixXf& input);
    Eigen::MatrixXf backward(const std::vector<Eigen::MatrixXf>& grad_output);
};
