#ifndef CONVOLUTION_ADAPTER_HPP
#define CONVOLUTION_ADAPTER_HPP

#include "Layer.hpp"
#include "ConvolutionLayer.hpp"
#include <vector>

class ConvolutionAdapter : public Layer {
public:
    ConvolutionAdapter(int in_c, int in_h, int in_w, int k_size, int num_filters, int stride = 1, int padding = 0);

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
    Eigen::MatrixXf backward(const Eigen::MatrixXf& d_out, float learning_rate) override;

private:
    ConvolutionLayer conv;
    std::vector<Eigen::MatrixXf> last_input;

    int in_channels, in_height, in_width, kernel_size, num_filters, stride, padding;  // Declare padding here
    Eigen::MatrixXf reshapeInput(const Eigen::MatrixXf& input);
    Eigen::MatrixXf applyConvolution(const Eigen::MatrixXf& input);
    
    Eigen::MatrixXf flatten(const std::vector<Eigen::MatrixXf>& maps);
    std::vector<Eigen::MatrixXf> unflatten(const Eigen::MatrixXf& input, int h, int w);
    Eigen::MatrixXf applyPadding(const Eigen::MatrixXf& input, int padding);
};

#endif // CONVOLUTION_ADAPTER_HPP
