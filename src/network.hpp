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

void network_init(std::vector<Layer> &layers, const std::vector<int> &sizes);

void forward_pass(std::vector<Layer> &layers, const Eigen::VectorXf &input);

void backward_pass(std::vector<Layer> &layers,
                   const Eigen::VectorXf &input,
                   const Eigen::VectorXf &output,
                   float learning_rate);

#endif // NETWORK_HPP
