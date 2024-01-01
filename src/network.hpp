#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include <vector>

struct Layer
{
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf activations;
    Eigen::VectorXf deltas;
};

[[nodiscard]] std::vector<Layer> network_init(const std::vector<int> &sizes);

void forward_pass(std::vector<Layer> &layers, const Eigen::VectorXf &input);

float training_pass(std::vector<Layer> &layers,
                    const Eigen::VectorXf &input,
                    const Eigen::VectorXf &output,
                    float learning_rate);

#endif // NETWORK_HPP
