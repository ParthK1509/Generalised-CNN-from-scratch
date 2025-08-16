#include "ConvolutionAdapter.hpp"
#include <cassert>
#include <iostream>

using namespace Eigen;
using namespace std;
ConvolutionAdapter::ConvolutionAdapter(int in_c, int in_h, int in_w, int k_size, int num_filters, int stride, int padding)
    : in_channels(in_c), in_height(in_h), in_width(in_w), kernel_size(k_size), num_filters(num_filters), stride(stride), padding(padding),
      conv(in_c, in_h, in_w, k_size, num_filters, stride) { // Initialize ConvolutionLayer
    // Other initialization code (if necessary)
    /*
    self.depth = num_filters
    input shape is (in_c, in_h, in_w)
    outputshape should be (num_filters, in_h-kernal_size + 1, in_w-kernel_size + 1)
    kernel shape should be (num_filters, in_c, kernel_size, kernel_size)

    kernel have random values of kernal shape calculated
    biases have random values of output shape calculated

    where out_h = (in_h - k_size) / stride + 1
    */
}

MatrixXf ConvolutionAdapter::applyPadding(const MatrixXf& input, int padding) {
    int new_rows = input.rows() + 2 * padding;
    int new_cols = input.cols() + 2 * padding;

    MatrixXf padded_input(new_rows, new_cols);
    padded_input.setZero();  // Fill with zeros

    // Copy the original input into the center of the padded matrix
    padded_input.block(padding, padding, input.rows(), input.cols()) = input;

    return padded_input;
}

MatrixXf ConvolutionAdapter::forward(const MatrixXf& input) {
    int padding = this->padding;
    // MatrixXf padded_input = applyPadding(input, padding);
    MatrixXf reshaped_input = reshapeInput(input);

    // Apply padding before convolution
    MatrixXf padded_input = applyPadding(reshaped_input, padding);
    
    // Now you can apply the convolution operation
    MatrixXf output = applyConvolution(padded_input);
    
    return output;
}

MatrixXf ConvolutionAdapter::reshapeInput(const MatrixXf& input) {
    // Check that the input size matches what is expected (784, 1)
    if (input.rows() != 784 || input.cols() != 1) {
        cerr << "Error: Input size is not 784x1.\n";
        exit(1); // or return some error value
    }

    // Create a 3D structure (1x28x28) using Tensor or other structures
    // Eigen does not natively support 3D matrices, so we simulate it as a flattened array
    MatrixXf reshaped_input(28, 28); // 28x28 for the image
    
    // Copy data from the input (784x1) into the 28x28 matrix
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            reshaped_input(i, j) = input(i * 28 + j, 0);
        }
    }

    // Return as 28x28, simulating a 3D structure (1 channel, 28x28)
    return reshaped_input;
}

MatrixXf ConvolutionAdapter::backward(const MatrixXf& d_out, float learning_rate) {
    auto d_out_maps = unflatten(d_out, conv.getOutputHeight(), conv.getOutputWidth());
    auto d_input_maps = conv.backward(d_out_maps, learning_rate);
    return flatten(d_input_maps);
}

MatrixXf ConvolutionAdapter::flatten(const vector<MatrixXf>& maps) {
    int total_size = 0;
    for (const auto& m : maps) total_size += m.size();

    MatrixXf flat(total_size, 1);
    int idx = 0;
    for (const auto& m : maps) {
        for (int i = 0; i < m.rows(); ++i)
            for (int j = 0; j < m.cols(); ++j)
                flat(idx++, 0) = m(i, j);
    }
    return flat;
}

vector<MatrixXf> ConvolutionAdapter::unflatten(const MatrixXf& input, int h, int w) {
    int total_elements = input.rows();
    assert(total_elements % (h * w) == 0);
    int channels = total_elements / (h * w);

    vector<MatrixXf> maps(channels, MatrixXf(h, w));
    int idx = 0;
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < h; ++i)
            for (int j = 0; j < w; ++j)
                maps[c](i, j) = input(idx++, 0);
    return maps;
}

MatrixXf ConvolutionAdapter::applyConvolution(const MatrixXf& input) {
    // Apply convolution logic here (not implemented in this example)
    int spatial_size = (conv.getOutputHeight()+2*padding) * (conv.getOutputWidth()+2*padding);
    int channels = input.rows() / (conv.getOutputHeight() * conv.getOutputWidth());
    cout << "[DEBUG] Input rows: " << input.rows() << endl;
    cout << "[DEBUG] Input cols: " << input.cols() << endl;
    cout << "[DEBUG] Output height: " << conv.getOutputHeight() << endl;
    cout << "[DEBUG] Output width: " << conv.getOutputWidth() << endl;
    cout << "[DEBUG] Expected spatial size: " << spatial_size << endl;
    assert((input.rows()-2*padding) % spatial_size == 0); // Must divide evenly

    this->last_input = unflatten(input, conv.getOutputHeight(), conv.getOutputWidth());
    auto output = conv.forward(this->last_input);
    return flatten(output);
    // After applying convolution, you would return the output (same dimensionality as needed)
    return input; // Dummy return, replace with actual convolution logic
}
