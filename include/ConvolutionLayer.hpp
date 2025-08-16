#ifndef CONVOLUTION_LAYER_HPP
#define CONVOLUTION_LAYER_HPP

// #include "Layer.hpp"
#include <Eigen/Dense>
#include <vector>
//checked
class ConvolutionLayer {
public:
    ConvolutionLayer(int input_channels, int input_height, int input_width,
                     int kernel_size, int num_filters, int stride = 1);

    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input);
    std::vector<Eigen::MatrixXf> backward(const std::vector<Eigen::MatrixXf>& d_out, float learning_rate);
    int getOutputHeight() const { return out_h; } // Getter for output height
    int getOutputWidth() const { return out_w; }   // Getter for output width
private:
    int in_c, in_h, in_w;
    int k_size;
    int stride;
    int out_h, out_w;
    int num_filters;

    // filters[f][c] = kernel matrix for filter f and channel c
    std::vector<std::vector<Eigen::MatrixXf>> filters;
    std::vector<std::vector<Eigen::MatrixXf>> d_filters;

    std::vector<Eigen::MatrixXf> input;
};

#endif // CONVOLUTION_LAYER_HPP
