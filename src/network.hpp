#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Dense>

#include <vector>

struct Layer
{
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
    Eigen::VectorXf activations;
    Eigen::VectorXf deltas;
};

void network_init(std::vector<Layer> &layers,
                  const std::vector<Eigen::Index> &sizes);

void network_predict(std::vector<Layer> &layers, const Eigen::VectorXf &input);

void network_update_weights(std::vector<Layer> &layers,
                            const Eigen::VectorXf &input,
                            const Eigen::VectorXf &output,
                            float learning_rate);

#endif // NETWORK_HPP
