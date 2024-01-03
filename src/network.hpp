#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Core>

#include <random>
#include <vector>

struct Layer
{
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::MatrixXf activations;
    Eigen::MatrixXf deltas;
};

[[nodiscard]] std::vector<Layer> network_init(const std::vector<int> &sizes,
                                              int batch_size,
                                              std::minstd_rand &rng);

void forward_pass(std::vector<Layer> &layers, const Eigen::MatrixXf &input);

void training_pass(std::vector<Layer> &layers,
                   const Eigen::MatrixXf &input,
                   const Eigen::MatrixXf &target_output,
                   int batch_size,
                   float learning_rate);

#endif // NETWORK_HPP
