#include <iostream>
#include <Eigen/Dense>

int main() {
    // Define two 2x2 matrices
    Eigen::Matrix2d A;
    A << 1, 2,
         3, 4;

    Eigen::Matrix2d B;
    B << 5, 6,
         7, 8;

    // Addition
    Eigen::Matrix2d sum = A + B;
    std::cout << "A + B =\n" << sum << "\n\n";

    // Multiplication
    Eigen::Matrix2d product = A * B;
    std::cout << "A * B =\n" << product << "\n\n";

    // Inverse of A
    if (A.determinant() != 0) {
        Eigen::Matrix2d inverse = A.inverse();
        std::cout << "Inverse of A =\n" << inverse << "\n\n";
    } else {
        std::cout << "Matrix A is not invertible.\n\n";
    }

    return 0;
}
