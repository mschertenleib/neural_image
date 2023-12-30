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

struct Network
{
    std::vector<Layer> layers;
};

void network_init(Network &network, const std::vector<unsigned int> &sizes);

void network_predict(Network &network, const Eigen::VectorXf &input);

void network_update_weights(Network &network,
                            const Eigen::VectorXf &input,
                            const Eigen::VectorXf &output,
                            float learning_rate);

#endif // NETWORK_HPP
