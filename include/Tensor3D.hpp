#ifndef TENSOR3D_HPP
#define TENSOR3D_HPP

#include <vector>
#include <Eigen/Dense>

class Tensor3D {
public:
    std::vector<Eigen::MatrixXf> channels;

    Tensor3D() {}
    Tensor3D(int depth, int height, int width) {
        channels.resize(depth, Eigen::MatrixXf(height, width));
    }

    int depth() const { return channels.size(); }
    int height() const { return channels.empty() ? 0 : channels[0].rows(); }
    int width() const { return channels.empty() ? 0 : channels[0].cols(); }

    Eigen::MatrixXf flatten() const {
        int total = 0;
        for (const auto& ch : channels)
            total += ch.size();

        Eigen::MatrixXf flat(total, 1);
        int offset = 0;
        for (const auto& ch : channels) {
            Eigen::Map<const Eigen::VectorXf> vec(ch.data(), ch.size());
            flat.block(offset, 0, vec.size(), 1) = vec;
            offset += vec.size();
        }
        return flat;
    }

    static Tensor3D fromMatrix(const Eigen::MatrixXf& mat) {
        Tensor3D t;
        t.channels.push_back(mat); // single-channel
        return t;
    }
};

#endif
