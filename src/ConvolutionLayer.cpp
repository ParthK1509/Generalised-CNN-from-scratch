#include "ConvolutionLayer.hpp"
#include <random>
#include <iostream>
// checked
ConvolutionLayer::ConvolutionLayer(int input_channels, int input_height, int input_width,
                                   int kernel_size, int num_filters, int stride)
    : in_c(input_channels), in_h(input_height), in_w(input_width),
      k_size(kernel_size), num_filters(num_filters), stride(stride)
{
    out_h = (in_h - k_size) / stride + 1;
    out_w = (in_w - k_size) / stride + 1;

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0f, 1.0f);

    filters.resize(num_filters);
    for (int f = 0; f < num_filters; ++f)
    {
        filters[f].resize(in_c);
        for (int c = 0; c < in_c; ++c)
        {
            filters[f][c] = Eigen::MatrixXf(k_size, k_size).unaryExpr([&](float)
                                                                      { return distribution(generator) * 0.1f; });
        }
    }
}

std::vector<Eigen::MatrixXf> ConvolutionLayer::forward(const std::vector<Eigen::MatrixXf> &in)
{
    input = in;
    // std::cout<<"bahar\n";
    std::vector<Eigen::MatrixXf> output(num_filters, Eigen::MatrixXf::Zero(out_h, out_w));
    // std::cout<<"andar\n";
    for (int f = 0; f < num_filters; ++f)
    {
        for (int i = 0; i < out_h; ++i)
        {
            for (int j = 0; j < out_w; ++j)
            {
                float val = 0.0f;
                for (int c = 0; c < in_c; ++c)
                {
                    int row = i * stride;
                    int col = j * stride;
                    // if (row + k_size > input[c].rows() || col + k_size > input[c].cols())
                    // {
                        // std::cerr << "[ConvolutionLayer] Patch size out of bounds! "
                        //           << "Channel: " << c
                        //           << ", Input size: " << input[c].rows() << "x" << input[c].cols()
                        //           << ", Requested block: (" << row << "," << col << ") size " << k_size << "x" << k_size
                        //           << std::endl;
                    //     exit(1);
                    // }
                    Eigen::MatrixXf patch = input[c].block(row, col, k_size, k_size);
                    val += (patch.array() * filters[f][c].array()).sum();
                }
                output[f](i, j) = val;
            }
        }
    }

    return output;
}

std::vector<Eigen::MatrixXf> ConvolutionLayer::backward(const std::vector<Eigen::MatrixXf> &d_out, float learning_rate)
{
    std::vector<Eigen::MatrixXf> d_input(in_c, Eigen::MatrixXf::Zero(in_h, in_w));
    d_filters = std::vector<std::vector<Eigen::MatrixXf>>(num_filters, std::vector<Eigen::MatrixXf>(in_c, Eigen::MatrixXf::Zero(k_size, k_size)));

    for (int f = 0; f < num_filters; ++f)
    {
        for (int i = 0; i < out_h; ++i)
        {
            for (int j = 0; j < out_w; ++j)
            {
                float grad = d_out[f](i, j);
                int row = i * stride;
                int col = j * stride;
                for (int c = 0; c < in_c; ++c)
                {
                    Eigen::MatrixXf patch = input[c].block(row, col, k_size, k_size);
                    d_filters[f][c] += patch * grad;
                    d_input[c].block(row, col, k_size, k_size).array() += filters[f][c].array() * grad;
                }
            }
        }
    }

    for (int f = 0; f < num_filters; ++f)
        for (int c = 0; c < in_c; ++c)
            filters[f][c] -= learning_rate * d_filters[f][c];

    return d_input;
}
